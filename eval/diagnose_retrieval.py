"""Phase 6 — retrieval recall diagnostic (standalone, throwaway).

Figures out WHICH gate loses an expected chunk when the eval reports a
precision_at_3 miss. A chunk passes through four gates in the query path:

  1. sparse lane   (Qdrant qdrant/bm25, rare-token IDF matching)
  2. dense lane    (Voyage voyage-4 cosine, paraphrase matching)
  3. RRF fusion    (1/(60+rank_lane) summed, keeps the fused top-N here)
  4. cross-encoder rerank (reorders fused candidates -> top-5)

Where a chunk falls out tells you which fix applies:
  - absent from BOTH lanes (never a candidate)  -> no reranker/limit tuning
    helps; the fix is query-side (expansion) or chunk-side.
  - in one lane but beyond the fused top-N      -> fusion/limit drops it;
    raise PREFETCH_LIMIT (the retriever's candidate pool).
  - in the fused top-N but demoted out of top-5 -> reranker bottleneck;
    upgrade the reranker model.
  - in the final top-5 yet eval still fails     -> the eval's citation/
    precision matcher is the problem, not retrieval.

Usage:
  python eval/diagnose_retrieval.py [--prefetch 50] [--rrf-k 60]
                                    [--query "text|citation1,citation2" ...]
Defaults probe the queries the Phase 5 eval missed. Does NOT write to
logs/traces.jsonl (keeps the observability channel clean).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http import models

from embeddings.embedder import Embedder
from eval.check_golden import citation_to_chunk_ids, resolve
from retrieval.retriever import COLLECTION, RERANK_WINDOW, SPARSE_NAME

# Queries from eval/results/1788233321.json whose expected chunk missed top-3,
# restated as their citation idioms so the diagnostic re-resolves them today.
DEFAULT_PROBES = [
    "How does GDPR data minimization (Article 5(1)(c)) relate to NIST AC-6 "
    "Least Privilege?|Article 5,AC-6",
    "What GDPR article addresses the right to erasure, and how does it "
    "interact with NIST account management requirements?|Article 17,AC-2(3),AC-2",
    "What does ASVS require for how OAuth access tokens should be handled?"
    "|V10.1.1,V10.1.2",
    "How do GDPR security-of-processing obligations translate into engineering "
    "requirements for an application?|Article 32,V11.1.1",
    "What do NIST and GDPR require about keeping accounts and personal data "
    "that are no longer needed?|AC-2(3),Article 5",
]


def _ranked_ids(points):
    return [p.payload["nid"] for p in points if p.payload]


def sparse_ranks(client, query, limit, target_ids):
    res = client.query_points(
        collection_name=COLLECTION,
        query=models.Document(text=query, model="qdrant/bm25"),
        using=SPARSE_NAME,
        limit=limit,
        with_payload=True,
    )
    return _ranks(_ranked_ids(res.points), target_ids)


def dense_ranks(client, query_vector, limit, target_ids):
    res = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return _ranks(_ranked_ids(res.points), target_ids)


def fused_points(client, query_text, query_vector, limit, k):
    """Replicates retriever._hybrid_top with the SAME dense vector (no second
    embed) and an explicitly larger pool so we can see where fusion drops it."""
    res = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query_text, model="qdrant/bm25"),
                using=SPARSE_NAME,
                limit=limit,
            ),
            models.Prefetch(query=query_vector, using="dense", limit=limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return res.points


def _ranks(ordered_ids, target_ids):
    pos = {i + 1: cid for i, cid in enumerate(ordered_ids)}
    inverse = {cid: rank for rank, cid in pos.items()}
    return {tid: inverse.get(tid) for tid in target_ids}


def _fmt(pos):
    return str(pos) if pos else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefetch", type=int, default=50, help="candidate pool per lane")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF rank constant")
    ap.add_argument("--topn", type=int, default=5, help="rerank top-n to report (default 5)")
    ap.add_argument("--query", action="append", default=[], metavar="q|c1,c2")
    args = ap.parse_args()

    probes = list(args.query) or DEFAULT_PROBES

    corpus = json.loads((ROOT / "corpus" / "corpus.json").read_text(encoding="utf-8"))
    all_ids = {c["id"] for c in corpus}

    client = QdrantClient(url="http://localhost:6333")
    embedder = Embedder()
    reranker = None
    try:
        from retrieval.reranker import Reranker
        reranker = Reranker()
    except Exception as exc:
        print("(reranker unavailable:", exc, ")", file=sys.stderr)

    for probe in probes:
        query_part, cit_part = probe.split("|", 1)
        query = query_part.strip()
        citations = [c.strip() for c in cit_part.split(",")]
        targets = []
        for cit in citations:
            base = citation_to_chunk_ids(cit)
            if base:
                targets.extend(resolve(base, all_ids))
        targets = list(dict.fromkeys(targets))

        print("=" * 78)
        print("QUERY:", query)
        print("expected chunks:", ", ".join(targets) or "(none)")

        qvec = embedder.embed([query], input_type="query")[0]
        sparse = sparse_ranks(client, query, args.prefetch, targets)
        dense = dense_ranks(client, qvec, args.prefetch, targets)

        fused = fused_points(client, query, qvec, args.prefetch, args.rrf_k)
        fused_ids = _ranked_ids(fused)
        fused_ranks = _ranks(fused_ids, targets)

        reranked = None
        topn = args.topn
        if reranker is not None:
            chunks = [
                {
                    "id": p.payload["nid"],
                    "text": p.payload.get("text", ""),
                    "metadata": p.payload.get("metadata") or {},
                }
                for p in fused
                if p.payload
            ]
            topn = min(topn, len(chunks[:RERANK_WINDOW]))
            top5 = reranker.rerank(query, chunks[:RERANK_WINDOW], top_n=topn)
            reranked = _ranks([c["id"] for c in top5], targets)

        print(f"  {'chunk':<16}{'sparse':>8}{'dense':>8}{'fused':>8}"
              f"{('rerank' + str(topn)):>9}")
        for tid in targets:
            print(f"  {tid:<16}{_fmt(sparse.get(tid)):>8}{_fmt(dense.get(tid)):>8}"
                  f"{_fmt(fused_ranks.get(tid)):>8}"
                  f"{_fmt(reranked.get(tid)) if reranked else '—':>9}")

        labels = {
            "sparse": {t: s for t, s in sparse.items() if s},
            "dense": {t: d for t, d in dense.items() if d},
        }
        n_sparse = len(labels["sparse"])
        n_dense = len(labels["dense"])
        n_fused = _ranks(fused_ids, targets)
        n_fused = sum(1 for v in n_fused.values() if v)
        print(f"  found in: sparse {n_sparse}/{len(targets)}, "
              f"dense {n_dense}/{len(targets)}, fused {n_fused}/{len(targets)}")
        print("  fused window top-15:", ", ".join(fused_ids[:15]))
        if reranked:
            print(f"  reranked top-{topn}:", ", ".join(t5["id"] for t5 in top5))


if __name__ == "__main__":
    main()