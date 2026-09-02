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

Usage:
    python eval/ragas_eval.py                 # use latest eval/results/*.json
    python eval/ragas_eval.py --results eval/results/1788313993.json
"""
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


def build_samples(golden, stored, id_to_text):
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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=None,
                        help="path to a run_eval.py results JSON (default: latest)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--all", action="store_true",
                        help="also run answer_relevancy and context_recall")
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

    samples, skipped = build_samples(golden, stored, id_to_text)
    print(f"building samples from {stored_path.name}: {len(samples)} non-negative, "
          f"skipped {len(skipped)} ({', '.join(s[1] for s in skipped[:4])}...)")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set (load .env and retry)")
        return 1

    from langchain_anthropic import ChatAnthropic
    from ragas.evaluation import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    run_config = RunConfig(timeout=120, max_retries=3)
    judge = LangchainLLMWrapper(
        ChatAnthropic(model=JUDGE_MODEL), run_config=run_config
    )

    metrics = [faithfulness, context_precision]
    if args.all:
        metrics += [answer_relevancy, context_recall]
    for m in metrics:
        m.max_retries = 3

    # The OpenAI-backed embedding factory ragas calls when embeddings is None
    # needs no credentials this way; only answer_relevancy (--all) ever
    # actually embeds, through the project's own Voyage embedder.

    dataset = EvaluationDataset(
        samples=[SingleTurnSample(**s) for s in samples]
    )

    print(f"judge: {JUDGE_MODEL} on {len(samples)} samples\n")
    result = evaluate(
        dataset, metrics=metrics, llm=judge, embeddings=VoyageRagasEmbeddings(),
        show_progress=True, batch_size=args.batch_size,
    )

    df = result.to_pandas()
    means = {name: float(df[name].mean()) for name in [m.name for m in metrics]}
    nan_counts = {name: int(df[name].isna().sum()) for name in [m.name for m in metrics]}

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

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_results": stored_path.name,
        "judge_model": JUDGE_MODEL,
        "n_samples": len(samples),
        "n_skipped": len(skipped),
        "metrics": means,
        "nan_counts": nan_counts,
    }
    fname = RESULTS_DIR / f"ragas_{int(time.time())}.json"
    fname.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"results written to {fname}")
    return 0 if (ff_ok and cp_ok) else 1


if __name__ == "__main__":
    sys.exit(main())