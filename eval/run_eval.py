"""Phase 5/6 — hand-rolled eval harness (standalone script, not pytest).

Loads eval/golden_dataset.json and runs each item through the full RAG
pipeline (retrieval -> generation), computing the gate metrics the AGENTS.md
testing strategy (Layer 5) cares about:

  - retrieval.precision_at_3        does a chunk whose citation family matches
                                    an expected citation appear in the top-3?
  - generation.citation_accuracy    fraction of expected citations the model
                                    actually produced in its answer
  - generation.faithfulness         (deterministic proxy) every citation the
                                    model made is backed by a retrieved chunk
  - generation.grounded             boolean: all generated citations are real
  - negative handling               for items with empty expected_citations,
                                    the answer must signal "not covered"

Results are written to eval/results/{timestamp}.json (gitignored) and a summary
pass/fail report is printed against the Layer 5 targets:
  citation_accuracy >= 0.9

NOTE: these are the deterministic, cost-free proxies from the AGENTS.md "zero
cost" strategy. Phase 6 wires RAGAS (with Claude Haiku as LLM judge) as the
side-by-side comparison on the same golden dataset.

Usage:
    python eval/run_eval.py
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_JSON = ROOT / "eval" / "golden_dataset.json"
RESULTS_DIR = ROOT / "eval" / "results"

CITATION_ACCURACY_TARGET = 0.9

# Negation phrases that indicate an honest "not covered" answer.
NEGATIVE_PHRASES = ["not covered", "does not", "isn't covered", "outside",
                    "not addressed", "no information", "do not"]

# Reuse the golden-dataset citation idiom -> chunk-id mapping.
sys.path.insert(0, str(ROOT))
from eval.check_golden import citation_to_chunk_ids  # noqa: E402


def _as_set(citations):
    return set(citations or [])


def _citation_family_match(retrieved_ids, chunk_ids):
    """True if any retrieved chunk id resolves to one of the expected chunk ids."""
    for rid in retrieved_ids:
        for cid in chunk_ids:
            if rid == cid or rid.startswith(cid + "_"):
                return True
    return False


def _negative_signal(answer):
    """Heuristic: does the answer signal the topic is out of scope?"""
    low = (answer or "").lower()
    return any(p in low for p in NEGATIVE_PHRASES)


def evaluate_item(item, generator, retriever, top_n=3):
    """Run one golden item through the pipeline and score it."""
    query = item["query"]
    expected_citations = item.get("expected_citations") or []
    # Map each expected citation to its base chunk-id family (nist_ac2_3 etc.).
    expected_ids = []
    for cit in expected_citations:
        base = citation_to_chunk_ids(cit)
        if base:
            expected_ids.append(base)
    expected_ids = [e for e in expected_ids if e]

    result = pipeline_generate(generator, retriever, query, top_n=top_n)
    retrieved = result["retrieved_chunks"]           # ids
    retrieved_context = result["retrieved_context"]  # full chunk dicts
    generated_citations = _as_set(result["citations"])

    # Map retrieved chunk ids to their citation labels (AC-2, Article 17, ...)
    # so the grounding check compares labels to labels, not label to chunk id.
    from generation.llm import _citation_label
    retrieved_labels = {_citation_label(c) for c in retrieved_context}

    # retrieval precision@3: any top-3 retrieved chunk matches an expected citation family
    precision_at_3 = _citation_family_match(retrieved[:3], expected_ids) if expected_ids else None

    # citation accuracy: recall of expected citations in the generated answer.
    # Negative items (no expected citations) are tracked separately via
    # not_covered_ok and EXCLUDED from this aggregate.
    expected_set = _as_set(expected_citations)
    is_negative = not expected_set
    if is_negative:
        citation_accuracy = None
        faithfulness = None
        grounded = None
    else:
        found = generated_citations & expected_set
        citation_accuracy = len(found) / len(expected_set)
        grounded = generated_citations.issubset(retrieved_labels) if generated_citations else True
        faithfulness = 1.0 if grounded else 0.0

    # negative handling
    not_covered_ok = None
    if is_negative:
        not_covered_ok = _negative_signal(result["answer"])

    return {
        "query": query,
        "expected_citations": expected_citations,
        "generated_citations": sorted(generated_citations),
        "retrieved_chunks": retrieved,
        "generated_answer": result["answer"],
        "retrieval": {"precision_at_3": precision_at_3},
        "generation": {
            "citation_accuracy": citation_accuracy,
            "faithfulness": faithfulness,
            "grounded": grounded,
            "is_negative": is_negative,
            "not_covered_ok": not_covered_ok,
        },
    }


def pipeline_generate(generator, retriever, query, top_n=3):
    """Minimal inline copy of generation.pipeline.generate_answer to avoid
    double-embedding cost in the harness loop."""
    from generation.pipeline import generate_answer
    return generate_answer(retriever, generator, query, top_n=top_n)


def summarize(items):
    acc_lst = [i["generation"]["citation_accuracy"] for i in items if i["generation"]["citation_accuracy"] is not None]
    gr_lst = [i["generation"]["grounded"] for i in items if i["generation"]["grounded"] is not None]
    precision = [i["retrieval"]["precision_at_3"] for i in items if i["retrieval"]["precision_at_3"] is not None]
    negatives = [i for i in items if i["generation"]["is_negative"]]

    summary = {
        "n_items": len(items),
        "citation_accuracy": round(sum(acc_lst) / len(acc_lst), 3) if acc_lst else None,
        "grounded": round(sum(gr_lst) / len(gr_lst), 3) if gr_lst else None,
        "precision_at_3": round(sum(precision) / len(precision), 3) if precision else None,
        "n_negative": len(negatives),
        "negatives_ok": sum(1 for n in negatives if n["generation"]["not_covered_ok"]),
    }
    return summary


def main():
    data = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    from generation.llm import Generator
    from generation.pipeline import _build_retriever

    generator = Generator()
    retriever = _build_retriever()

    evaluated = []
    for item in data:
        print(f"[eval] {item['query'][:60]}...")
        evaluated.append(evaluate_item(item, generator, retriever))

    summary = summarize(evaluated)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "generator": generator.cost_summary(),
        "items": evaluated,
    }

    fname = RESULTS_DIR / f"{int(time.time())}.json"
    fname.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print(f"  generator spend: {generator.cost_summary()['approx_spend_usd']}")
    acc = summary["citation_accuracy"]
    ok = acc is not None and acc >= CITATION_ACCURACY_TARGET
    print(f"\nLayer 5 gate: citation_accuracy >= {CITATION_ACCURACY_TARGET}: "
          f"{'PASS' if ok else 'FAIL'} ({acc})")
    print(f"results written to {fname}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
