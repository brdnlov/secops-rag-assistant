"""Phase 4 — hybrid retrieval + reranking.

The production query path (the Phase 4 "Query path" from AGENTS.md):

    1. Embed query as dense vector (Voyage voyage-4, input_type="query")
    2. Generate query as sparse vector (Qdrant server-side qdrant/bm25)
    3. Qdrant hybrid query: one dense prefetch + one sparse prefetch, fused
       with RRF (rank constant 60 — Elasticsearch default, "requires no
       tuning" per the Cormack paper)
    4. Cross-encoder rerank the fused top-k -> return top-5

Built on the primitives already prototyped in tests/test_layer2_retrieval.py
(hybrid_top3), promoted into a first-class module for Phase 5 (generation) and
the eval harness to consume. Every retrieve() emits a structured JSON trace
(logs/traces.jsonl) covering the numbers an interview asks about: what was
retrieved, scored, reranked, and cited for a given query.
"""

import time

from qdrant_client import QdrantClient
from qdrant_client.http import models

from embeddings.embedder import Embedder
from embeddings.indexer import (
    COLLECTION,
    HEADING_PATH,
    META,
    NID,
    SOURCE,
    SPARSE_NAME,
    TEXT,
)
from retrieval.reranker import Reranker
from retrieval.tracer import QueryTracer

QDRANT_URL = "http://localhost:6333"

# RRF rank constant: 60 = Elasticsearch default, "requires no tuning" per
# Cormack et al. The hybrid philosophy: a doc ranked #3 in both lanes can beat
# a doc ranked #1 in one lane and absent in the other (agreement wins).
RRF_K = 60

# How many candidates the hybrid stage returns before reranking. Rerank has
# room to reorder meaningfully without being wasteful (10-40 is typical).
PREFETCH_LIMIT = 20

# Number of chunks returned to the caller after reranking.
TOP_N = 5


def _payload_chunk(payload):
    """Project one Qdrant point payload back to a corpus-chunk-shaped dict."""
    return {
        "id": payload.get(NID),
        "source": payload.get(SOURCE),
        "heading_path": payload.get(HEADING_PATH),
        "text": payload.get(TEXT),
        "metadata": payload.get(META) or {},
    }


class Retriever:
    def __init__(self, client=None, embedder=None, reranker=None, tracer=None):
        self.client = client or QdrantClient(url=QDRANT_URL)
        self.embedder = embedder or Embedder()
        self.reranker = reranker
        self.tracer = tracer or QueryTracer()

    def _hybrid_top(self, query_text, limit=PREFETCH_LIMIT):
        """Dense + sparse prefetch fused with RRF. Returns Qdrant points."""
        query_vector = self.embedder.embed([query_text], input_type="query")[0]
        return self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text=query_text, model="qdrant/bm25"),
                    using=SPARSE_NAME,
                    limit=limit,
                ),
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        ).points

    def retrieve(self, query_text, top_n=TOP_N, rerank=True):
        """Full retrieval pass: hybrid search (+ optional rerank) -> chunks.

        Returns a dict:
            chunks:  list of chunk dicts (id, source, heading_path, text,
                     metadata), best-first
            scores:  alignment list of reranker (or fused) scores per chunk
        Also writes one JSON trace to logs/traces.jsonl.
        """
        start = time.time()

        points = self._hybrid_top(query_text)
        chunks = [_payload_chunk(p.payload) for p in points if p.payload]
        scores = [float(p.score) for p in points if p.payload]

        chunks_after_rerank = len(chunks)
        reranked_ids = [c["id"] for c in chunks]
        reranked_scores = scores

        if rerank and self.reranker is not None:
            chunks = self.reranker.rerank(query_text, chunks, top_n=top_n)
            reranked_ids = [c["id"] for c in chunks]
            chunks_after_rerank = len(chunks)
            reranked_scores = None  # cross-encoder scores not surfaced here

        elapsed = time.time() - start

        self.tracer.trace(
            query_text,
            retrieval={
                "chunks_returned": len(points),
                "top_scores": scores,
                "chunk_ids": [c["id"] for c in chunks],
                "chunk_sources": [c.get("source") for c in chunks],
            },
            reranking={
                "chunks_after_rerank": chunks_after_rerank,
                "reranked_ids": reranked_ids,
                "reranked_scores": reranked_scores,
            },
            elapsed_s=elapsed,
        )

        return {"chunks": chunks, "scores": reranked_scores}
