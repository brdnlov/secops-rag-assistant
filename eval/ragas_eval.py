"""Phase 6 — RAGAS side-by-side eval harness (standalone script).

Companion to eval/run_eval.py: same golden dataset, RAGAS metrics computed
with Claude Haiku 4.5 as the LLM judge instead of the hand-rolled proxies.

Rather than re-running the pipeline (which would re-embed 50 queries), this
script reuses the stored outputs of the last run_eval.py pass
(eval/results/*.json, gitignored): question + generated answer +
retrieved chunk ids, with chunk texts resolved from corpus/corpus.json.
The RAGAS judge then only pays for LLM scoring, not retrieval.

Metrics (the AGENTS.md Layer 5 targets):
    faithfulness        (NLI judge)   target >= 0.8
    context_precision   (judge)       target >= 0.7

answer_relevancy/context_recall are informational in this harness and skipped
by default (--all runs them) — they double the judge calls for zero gate
value, and answer_relevancy additionally embeds generated questions through
the throttled Voyage API.

Negative items (empty expected_citations) are excluded from the aggregate,
mirroring run_eval.py: their "not covered" answers would unfairly score
answer_relevancy/context_recall against a nonexistent expected answer.

NOTE: the llm_factory(provider="anthropic") call in AGENTS.md is from an
older ragas release; 0.2.15 has no provider/client kwargs, so the judge is
wired via LangchainLLMWrapper(ChatAnthropic(...)) instead. Verified against
this venv's ragas 0.2.15.

SPEND SAFETY (added in the Phase 6 fix — direct response to the session-8
credit incident where unguarded judge runs exhausted the $5 Anthropic credit):
  * --dry-run                projects the judge cost from the resolved sample
                             contexts and exits with ZERO LLM calls — never
                             launch a judge run blind again.
  * exact cost tracking via ragas' CostCallbackHandler (this is the
                             `result.total_cost` the session summary wanted,
                             wired through evaluate(callbacks=...); RAGAS 0.2.15
                             has no cost_cb on RunConfig).
  * every judge run records its spend in eval/results/credit_budget.json
    (gitignored), and --budget-usd halts a run whose projected cost exceeds the
    remaining session budget BEFORE it spends.
  * --judge-max-chars truncates each context chunk fed to the judge, cutting
    per-call input tokens (the dominant cost driver) without reducing the
    number of contexts evaluated (classic precision measures presence, not
    full text).
  * free/local judge iteration: see --judge in the CLI and the README. Haiku
    is for the final reported run, not for every tuning loop.

Usage:
    python eval/ragas_eval.py                 # use latest eval/results/*.json
    python eval/ragas_eval.py --results eval/results/1788313993.json
    python eval/ragas_eval.py --dry-run       # project cost, zero spend
    python eval/ragas_eval.py --queries q1,q2 --budget-usd 1.0
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_JSON = ROOT / "eval" / "golden_dataset.json"
RESULTS_DIR = ROOT / "eval" / "results"
CORPUS_JSON = ROOT / "corpus" / "corpus.json"

from ragas.embeddings import BaseRagasEmbeddings  # noqa: E402

FAITHFULNESS_TARGET = 0.8
CONTEXT_PRECISION_TARGET = 0.7

# Judge model: same snapshot used by the generation layer (AGENTS.md: "Claude
# Haiku 4.5 as judge (existing $5 Anthropic credit)").
JUDGE_MODEL = "claude-haiku-4-5-20251001"

# Judge max output tokens. ChatAnthropic's default (1024) truncates the RAGAS
# faithfulness NLI/statement-generation JSON on multi-citation answers, which
# ragas 0.2.15 turns into NaN. Haiku 4.5 supports far more; 8192 is ample for
# the judge's structured output (validated live — see JUDGE_MODEL construction).
JUDGE_MAX_TOKENS = 8192

# Haiku 4.5 prices per AGENTS.md: $1/MTok input, $5/MTok output.
INPUT_COST_PER_TOKEN = 1e-6
OUTPUT_COST_PER_TOKEN = 5e-6

# Faithfulness runs a per-claim NLI pass, so one answer can cost several LLM
# calls. This is the conservative multiplier for cost projection; the real
# spend is measured exactly via the cost callback, this only gates the start.
JUDGE_CALL_MULTIPLIER = 8.0
COST_PER_OUTPUT_TOKEN_CALL = 300  # rough output tokens per judge call

MAX_SCORE_CHARS = 1500  # default per-chunk char cap fed to the judge


def _count_tokens(text):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def truncate_chunks(contexts, max_chars=None):
    """Optionally cap each chunk's length for the judge (Issue 1c).

    Capping per-chunk *text* (not the count of contexts) keeps every retrieved
    context in scope for context_precision while shrinking the input tokens
    re-shipped on every judge call — the dominant cost of faithfulness' NLI.
    """
    if not max_chars or max_chars <= 0:
        return contexts
    out = []
    for c in contexts:
        out.append(c if len(c) <= max_chars else c[:max_chars] + "…")
    return out


def build_samples(golden, stored, id_to_text, judge_topk=None, judge_max_chars=None):
    """Reconstruct RAGAS SingleTurnSamples from stored eval outputs.

    stored item -> query, generated_answer, retrieved_chunks (best-first ids)
    golden item (join on query) -> expected_answer, expected_citations

    reference = texts of the expected citation chunk ids resolved to their
    actual (post-split) chunk ids, e.g. 'AC-2(3)' -> nist_ac2_3.
    """
    sys.path.insert(0, str(ROOT))
    from eval.check_golden import citation_to_chunk_ids, resolve

    all_ids = set(id_to_text)
    by_query = {g["query"]: g for g in golden}

    samples = []
    skipped = []
    for item in stored:
        query = item["query"]
        gen = item["generation"]
        if gen.get("is_negative"):
            skipped.append((query, "negative"))
            continue
        golden_item = by_query.get(query)
        if golden_item is None:
            skipped.append((query, "no golden match"))
            continue

        contexts = [id_to_text[i] for i in item["retrieved_chunks"] if i in id_to_text]
        if judge_topk:
            contexts = contexts[:judge_topk]
        contexts = truncate_chunks(contexts, judge_max_chars)
        expected_ids = set()
        for cit in golden_item.get("expected_citations") or []:
            base = citation_to_chunk_ids(cit)
            if base:
                expected_ids.update(resolve(base, all_ids))
        reference = [id_to_text[i] for i in expected_ids if i in id_to_text]

        samples.append({
            "user_input": query,
            "response": item["generated_answer"],
            "retrieved_contexts": contexts,
            "reference": golden_item.get("expected_answer", ""),
            "reference_contexts": reference,
        })
    return samples, skipped


class VoyageRagasEmbeddings(BaseRagasEmbeddings):
    """Doorstop ragas' embedding needs onto the repo's own Voyage embedder.

    Only answer_relevancy embeds (its generated questions); Passing the
    project embedder keeps ragas off a second embedding provider.
    """

    def __init__(self):
        from embeddings.embedder import Embedder
        self._embedder = Embedder()

    def embed_documents(self, texts):
        return self._embedder.embed(texts, input_type="document")

    def embed_query(self, text):
        return self._embedder.embed([text], input_type="query")[0]

    async def aembed_documents(self, texts):
        import asyncio
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text):
        import asyncio
        return await asyncio.to_thread(self.embed_query, text)


def project_cost(n_samples, per_sample_input_tokens, metrics):
    """Project judge spend before launching (Issue 1b).

    faithfulness is claim-multiplied (per-claim NLI), so we apply a
    conservative CALL_MULTIPLIER to the token count that would be re-shipped
    on every judge call. This is a gate, not a guarantee — real spend is
    measured exactly by the cost callback afterwards.

    Both input and output terms are scaled by n_samples (the dataset size) —
    per_sample_input_tokens is the *average* over samples, so without the
    n_samples factor the projection would under-count input cost by the size
    of the run (a bug that would let the budget gate pass a run whose real
    spend was many times larger — the exact session-8 failure mode).
    """
    n_metrics = len(metrics)
    input_tok = n_samples * per_sample_input_tokens * JUDGE_CALL_MULTIPLIER * n_metrics
    output_tok = n_samples * COST_PER_OUTPUT_TOKEN_CALL * JUDGE_CALL_MULTIPLIER * n_metrics
    cost = input_tok * INPUT_COST_PER_TOKEN + output_tok * OUTPUT_COST_PER_TOKEN
    return input_tok, output_tok, cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None,
                        help="path to a run_eval.py results JSON (default: latest)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--all", action="store_true",
                        help="also run answer_relevancy and context_recall")
    parser.add_argument("--dry-run", action="store_true",
                        help="project judge cost from resolved contexts, exit with zero LLM calls")
    parser.add_argument("--queries", default=None,
                        help="comma-separated query substrings to score (default: all)")
    parser.add_argument("--budget-usd", type=float, default=None,
                        help="session credit ceiling (default: credit_budget.DEFAULT_BUDGET_USD)")
    parser.add_argument("--judge-topk", type=int, default=None,
                        help="limit count of contexts fed to the judge (default: all retrieved)")
    parser.add_argument("--judge-max-chars", type=int, default=MAX_SCORE_CHARS,
                        help="per-chunk char cap for the judge (default %(default)d)")
    parser.add_argument("--no-ledger", action="store_true",
                        help="skip recording spend in credit_budget.json")
    args = parser.parse_args()

    stored_path = args.results
    if stored_path is None:
        candidates = sorted(RESULTS_DIR.glob("[0-9]*.json"))
        if not candidates:
            print("no run_eval.py results found; run eval/run_eval.py first")
            return 1
        stored_path = candidates[-1]
    stored_path = Path(stored_path)

    golden = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    stored = json.loads(stored_path.read_text(encoding="utf-8"))["items"]
    corpus = json.loads(CORPUS_JSON.read_text(encoding="utf-8"))
    id_to_text = {c["id"]: c["text"] for c in corpus}

    if args.queries:
        substrings = [q.strip() for q in args.queries.split(",") if q.strip()]
        stored = [it for it in stored if any(s.lower() in it["query"].lower() for s in substrings)]
        if not stored:
            print(f"no stored items matched --queries {args.queries!r}")
            return 1
        print(f"[ragas_eval] subset mode: {len(stored)} items match "
              f"{[s for s in substrings]}")

    samples, skipped = build_samples(
        golden, stored, id_to_text,
        judge_topk=args.judge_topk, judge_max_chars=args.judge_max_chars,
    )
    print(f"building samples from {stored_path.name}: {len(samples)} non-negative, "
          f"skipped {len(skipped)} ({', '.join(s[1] for s in skipped[:4])}...)")

    # Projected input tokens from the ACTUAL resolved contexts (measured, not
    # guessed) so the dry-run and budget guard are grounded in real data.
    per_sample_input = [
        _count_tokens(s["user_input"])
        + sum(_count_tokens(c) for c in s["retrieved_contexts"])
        for s in samples
    ]
    metrics_planned = ["faithfulness", "context_precision"]
    if args.all:
        metrics_planned += ["answer_relevancy", "context_recall"]

    projected_in, projected_out, projected_cost = project_cost(
        len(samples), sum(per_sample_input) / max(len(per_sample_input), 1), metrics_planned
    )
    print("\n" + "=" * 60)
    print("RAGAS JUDGE COST PROJECTION")
    print("=" * 60)
    print(f"  samples (non-negative)      {len(samples)}")
    print(f"  metrics                     {', '.join(metrics_planned)}  "
          f"(call multiplier x{JUDGE_CALL_MULTIPLIER:.0f})")
    print(f"  avg input tokens/sample     {sum(per_sample_input)/max(len(per_sample_input),1):.0f}")
    print(f"  projected input tokens      {projected_in:,.0f}  (~${projected_in*INPUT_COST_PER_TOKEN:.3f})")
    print(f"  projected output tokens     {projected_out:,.0f}  (~${projected_out*OUTPUT_COST_PER_TOKEN:.3f})")
    print(f"  projected total cost        ~${projected_cost:.3f}")
    if args.judge_topk:
        print(f"  [judge contexts capped to top-{args.judge_topk}]")
    if args.judge_max_chars:
        print(f"  [judge chunks capped at {args.judge_max_chars} chars each]")
    print("=" * 60)

    # Budget guard BEFORE any LLM call (Issue 1a). --dry-run respects it too.
    from eval import credit_budget
    budget = args.budget_usd if args.budget_usd is not None else credit_budget.DEFAULT_BUDGET_USD
    credit_budget.print_status(budget)
    credit_budget.assert_budget(projected_cost, budget, "RAGAS judge run")
    if args.dry_run:
        print("[dry-run] no LLM calls made; exiting.")
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (load .env and retry)")
        return 1

    from langchain_anthropic import ChatAnthropic
    from ragas.cost import CostCallbackHandler, get_token_usage_for_anthropic
    from ragas.evaluation import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    run_config = RunConfig(timeout=120, max_retries=3)

    # Phase 6 diagnosis fix: ChatAnthropic defaults to max_tokens=1024, which is
    # too small for the RAGAS faithfulness NLI + statement-generation JSON
    # output on multi-citation answers. Claude truncates (stop_reason
    # "max_tokens"), ragas' is_finished() can't recognise that reason, so it
    # raises LLMDidNotFinishException and the sample is marked NaN. Raising the
    # cap lets the judge return complete, parseable JSON (validated live).
    judge = LangchainLLMWrapper(
        ChatAnthropic(model=JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS),
        run_config=run_config,
    )

    metrics = [faithfulness, context_precision]
    if args.all:
        metrics += [answer_relevancy, context_recall]
    for m in metrics:
        m.max_retries = 3

    dataset = EvaluationDataset(
        samples=[SingleTurnSample(**s) for s in samples]
    )

    # Exact spend tracking (Issue 1d): RAGAS 0.2.15 does not expose cost on the
    # judge; the CostCallbackHandler wired via evaluate(callbacks=...) is where
    # the actual per-call token usage lands (this is the `total_cost` the
    # session-8 summary wanted fronted).
    cost_cb = CostCallbackHandler(get_token_usage_for_anthropic)

    print(f"\njudge: {JUDGE_MODEL} on {len(samples)} samples (projected ~${projected_cost:.3f})")

    df = None
    result = None
    actual_cost = 0.0
    total_in = 0
    total_out = 0
    run_name = "ragas_eval"
    try:
        result = evaluate(
            dataset, metrics=metrics, llm=judge, embeddings=VoyageRagasEmbeddings(),
            show_progress=True, batch_size=args.batch_size,
            callbacks=[cost_cb],
        )
        df = result.to_pandas()
    finally:
        # Record exact spend even if evaluate()/scoring died mid-run (e.g. an
        # unhandled metric exception), so the ledger never under-counts prior
        # Anthropic spend. If cost capture failed (empty usage_data), record a
        # best-effort estimate rather than silently dropping the entry.
        usage = None
        if cost_cb.usage_data:
            try:
                usage = cost_cb.total_tokens()
                total_in = getattr(usage, "input_tokens", 0) or 0
                total_out = getattr(usage, "output_tokens", 0) or 0
                actual_cost = cost_cb.total_cost(
                    cost_per_input_token=INPUT_COST_PER_TOKEN,
                    cost_per_output_token=OUTPUT_COST_PER_TOKEN,
                )
            except Exception:
                usage = None
        if usage is None:
            # cost callback captured nothing (a ragas-version/quirk risk): fall
            # back to the projection so a spend is never lost from the ledger.
            print("[ragas_eval] WARNING: cost callback captured no token usage; "
                  f"recording projected ${projected_cost:.4f} instead")
            total_in = round(projected_cost / INPUT_COST_PER_TOKEN)
            total_out = 0
            actual_cost = projected_cost
        if not args.no_ledger:
            credit_budget.record(
                run_name,
                actual_cost,
                details={
                    "model": JUDGE_MODEL, "n_samples": len(samples),
                    "input_tokens": total_in, "output_tokens": total_out,
                    "metrics": metrics_planned,
                },
            )
            credit_budget.print_status(budget)

    if df is None:
        print("[ragas_eval] judge run aborted before scores could be computed")
        return 1

    means = {name: float(df[name].mean()) for name in [m.name for m in metrics]}
    nan_counts = {name: int(df[name].isna().sum()) for name in [m.name for m in metrics]}
    missing = df[df.isna().any(axis=1)]
    missing_rows = missing.to_dict(orient="records") if len(df) else []

    print("\n" + "=" * 60)
    print("RAGAS SIDE-BY-SIDE (Claude Haiku 4.5 judge)")
    print("=" * 60)
    for name in sorted(means):
        target = ""
        if name == "faithfulness":
            target = f"  target >= {FAITHFULNESS_TARGET}"
        if name == "context_precision":
            target = f"  target >= {CONTEXT_PRECISION_TARGET}"
        print(f"  {name:20s} {means[name]:.3f}{target}")
    for name, n in nan_counts.items():
        if n:
            print(f"  {name:20s} NaN on {n}/{len(df)} samples (judge JSON parse)")

    # per-category breakdown: where does faithfulness/context_precision suffer?
    cat_by_query = {g["query"]: g.get("category", "?") for g in golden}
    queries = [s["user_input"] for s in samples]
    for cat in ["exact", "synthesis", "cross_document"]:
        idx = [i for i, q in enumerate(queries) if cat_by_query.get(q) == cat]
        if not idx:
            continue
        print(f"  category {cat:14s} n={len(idx):2d}  "
              f"faithfulness={df['faithfulness'].iloc[idx].mean():.3f}  "
              f"context_precision={df['context_precision'].iloc[idx].mean():.3f}")

    ff_ok = means.get("faithfulness", 0) >= FAITHFULNESS_TARGET
    cp_ok = means.get("context_precision", 0) >= CONTEXT_PRECISION_TARGET
    print(f"\nLayer 5 gate: faithfulness >= {FAITHFULNESS_TARGET}: "
          f"{'PASS' if ff_ok else 'FAIL'} ({means.get('faithfulness'):.3f})")
    print(f"Layer 5 gate: context_precision >= {CONTEXT_PRECISION_TARGET}: "
          f"{'PASS' if cp_ok else 'FAIL'} ({means.get('context_precision'):.3f})")
    print(f"\njudge spend: {total_in:,} in / {total_out:,} out tokens "
          f"= ${actual_cost:.4f}")

    # Persist per-sample scores + category + spend (Issue 2 diagnosis).
    per_sample = []
    for i, s in enumerate(samples):
        row = {
            "query": s["user_input"],
            "category": cat_by_query.get(s["user_input"], "?"),
            "faithfulness": float(df["faithfulness"].iloc[i]) if not df["faithfulness"].isna().iloc[i] else None,
            "context_precision": float(df["context_precision"].iloc[i]) if not df["context_precision"].isna().iloc[i] else None,
        }
        per_sample.append(row)

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_results": stored_path.name,
        "judge_model": JUDGE_MODEL,
        "n_samples": len(samples),
        "n_skipped": len(skipped),
        "projected_cost_usd": round(projected_cost, 4),
        "actual_spend_usd": round(actual_cost, 6),
        "judge_input_tokens": total_in,
        "judge_output_tokens": total_out,
        "context_caps": {"judge_topk": args.judge_topk, "judge_max_chars": args.judge_max_chars},
        "metrics": means,
        "nan_counts": nan_counts,
        "nan_samples": missing_rows,
        "per_sample": per_sample,
    }
    fname = RESULTS_DIR / f"ragas_{int(time.time())}.json"
    fname.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"results written to {fname}")

    return 0 if (ff_ok and cp_ok) else 1


if __name__ == "__main__":
    sys.exit(main())