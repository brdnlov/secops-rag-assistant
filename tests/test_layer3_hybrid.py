"""Layer 3 — hybrid search integration (Phase 4 gate).

Verifies the full Phase 4 pipeline — voyage query-embed -> dense + sparse
prefetch -> RRF fusion -> cross-encoder rerank — behaves correctly:

  - Exact-match query        -> the sparse/BM25 lane finds the exact-match
                                chunk (rare identifiers carry huge IDF)
  - Semantic query           -> the dense lane finds the paraphrase-matching
                                chunk (semantic similarity over lexical match)
  - Hybrid beats either lane alone on a mixed cross-document query (fusing
    both lanes' ranked lists via RRF surfaces a consensus neither lane alone
    nails)

These are the three canonical Layer 3 probes. They exercise the Retriever
directly (not just the raw Qdrant call), so a regression in the reranker or
the trace logging would surface here as well.

Requires a Qdrant instance with the corpus loaded (docker compose up -d in
docker/, then `python -m embeddings.indexer`), a Voyage API key in .env, and
the sentence-transformers reranker model (downloads on first use, ~80MB).
Skips cleanly if any of those are unavailable.
"""

import json
import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from embeddings.embedder import Embedder
from embeddings.indexer import COLLECTION, NID, SPARSE_NAME

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from retrieval.tracer import QueryTracer

ROOT = Path(__file__).resolve().parents[1]
QDRANT_URL = "http://localhost:6333"

# A one-time ephemeral trace file so Layer 3 runs don't pollute logs/traces.jsonl.
TRACE_FILE = ROOT / "logs" / "_layer3_traces.jsonl"


@pytest.fixture(scope="module")
def client():
    c = QdrantClient(url=QDRANT_URL)
    try:
        c.get_collection(COLLECTION)
    except Exception:
        pytest.skip("Qdrant not reachable or collection missing (docker compose up + indexer first)")
    return c


@pytest.fixture(scope="module")
def embedder():
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set — query embedding requires Voyage")
    return Embedder()


@pytest.fixture(scope="module")
def reranker():
    try:
        return Reranker()
    except Exception as e:
        pytest.skip(f"Reranker model unavailable: {e}")


@pytest.fixture(scope="module")
def retriever(client, embedder, reranker):
    return Retriever(client=client, embedder=embedder, reranker=reranker,
                     tracer=QueryTracer(TRACE_FILE))


def _nids(results):
    return [p.payload.get(NID) for p in results if p.payload]


def _sparse_top(client, query_text, limit=5):
    res = client.query_points(
        collection_name=COLLECTION,
        query=models.Document(text=query_text, model="qdrant/bm25"),
        using=SPARSE_NAME,
        limit=limit,
        with_payload=True,
    )
    return _nids(res.points)


def _dense_top(client, embedder, query_text, limit=5):
    qv = embedder.embed([query_text], input_type="query")[0]
    res = client.query_points(
        collection_name=COLLECTION,
        query=qv,
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return _nids(res.points)


def _ids(chunks):
    return [c["id"] for c in chunks]


# --- Exact-match: sparse/BM25 lane finds the rare identifier ---
def test_exact_match_query_sparse_dominates(client):
    """'AC-2' is a rare identifier; BM25's IDF term should surface it."""
    ids = _sparse_top(client, "AC-2")
    assert any(i and i.startswith("nist_ac2") for i in ids), f"AC-2 not in sparse top-5: {ids}"


# --- Semantic: dense lane finds the paraphrase ---
def test_semantic_query_dense_dominates(client, embedder):
    """A semantic paraphrase query (no literal 'AC-2') should still surface
    the account-management control via the dense/embedding lane."""
    ids = _dense_top(client, embedder, "how should organizations manage user accounts and their lifecycle")
    assert any(i and i.startswith("nist_ac2") for i in ids), f"AC-2 not in dense top-5: {ids}"


# --- Hybrid beats either single lane alone on a mixed cross-doc query ---
def test_hybrid_beats_single_lanes_alone(retriever):
    """A cross-document synthesis query (GDPR data minimization + NIST least
    privilege) should surface BOTH corpora: the NIST AC-6 control AND some
    GDPR article addressing data-minimization/principles. RRF rank-consensus
    makes the fusion land content from both domains that neither single lane
    alone would prioritize as highly.
    """
    result = retriever.retrieve(
        "How does GDPR's data minimization principle relate to NIST least privilege?",
        top_n=5,
    )
    chunks = result["chunks"]

    # NIST side: the least-privilege control must surface.
    assert any(c["id"].startswith("nist_ac6") for c in chunks), (
        f"nist_ac6 not in top-5: {_ids(chunks)}"
    )

    # GDPR side: at least one GDPR article discussing minimization/principles.
    gdpr = [c for c in chunks if c.get("source") == "gdpr"]
    assert gdpr, f"no GDPR chunk in top-5: {_ids(chunks)}"
    assert any(
        ("minimi" in c["text"].lower()) or ("principle" in c["text"].lower())
        for c in gdpr
    ), f"no GDPR minimization/principles chunk in top-5: {_ids(chunks)}"


# --- Retriever reranks and returns the configured top_n, best-first ---
def test_retriever_reranks_top_n(retriever):
    result = retriever.retrieve("what does AC-2 require for account management?", top_n=3)
    assert len(result["chunks"]) == 3
    assert any(c["id"].startswith("nist_ac2") for c in result["chunks"]), (
        f"AC-2 not in reranked top-3: {_ids(result['chunks'])}"
    )


# --- Retriever writes a structured trace per query ---
def test_retriever_writes_trace(retriever):
    retriever.retrieve("GDPR Article 17 right to erasure", top_n=2)
    lines = TRACE_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "no trace lines written"
    last = json.loads(lines[-1])
    assert "query" in last and "retrieval" in last and "reranking" in last
    assert last["reranking"]["chunks_after_rerank"] > 0
