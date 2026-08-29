"""Layer 2 — retrieval sanity (Phase 3 gate).

Verifies the Qdrant collection serves the corpus correctly via the raw
hybrid query (dense prefetch + sparse prefetch, RRF fusion) BEFORE any
reranker or generation layer exists:

  - "AC-2"                   -> a NIST AC-2 chunk must be in the top-3
  - "GDPR Article 5"         -> a GDPR Art. 5 chunk must be in the top-3
  - "zzqxvbrionqiuu"         -> the sparse/BM25 side must have nothing for an
                                invented token (the "not covered" signal the
                                generator relies on isn't fabricated by dense
                                fallback alone)

The dense prefetch embeds the query with Voyage voyage-4 (the collection's
embedding model) — Qdrant has no text-inference for that vector, so raw text
can't be sent to the dense side. The sparse prefetch uses Qdrant's native
server-side BM25 ("qdrant/bm25").

The invented-token probe must be a single word: BM25 tokenizes at
underscores/punctuation, so "nonexistent_control_xyz" splits into real words
like "control" and would match legitimate chunks.

These are the three canonical Layer 2 probes from the 5-layer testing
strategy. Reranking (cross-encoder) lands in Phase 4 and is intentionally
absent here — this test measures the collection/index itself.

Requires a Qdrant instance with the corpus loaded (docker compose up -d in
docker/, then `python -m embeddings.indexer`) and a Voyage API key in .env.
Skips cleanly if either is missing, so Layer 1/2 checks pass before Phase 3
provisioning.
"""

import os
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from embeddings.embedder import Embedder
from embeddings.indexer import COLLECTION, NID, SPARSE_NAME

ROOT = Path(__file__).resolve().parents[1]
QDRANT_URL = "http://localhost:6333"

PREFETCH_LIMIT = 10

# Single-token gibberish with no dictionary collisions. Earlier versions used
# "nonexistent_control_xyz", but BM25 splits at underscores and the "control"
# sub-token legitimately matches corpus chunks.
INVENTED_TOKEN = "zzqxvbrionqiuu"


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


def _ids(results):
    return [hit.payload.get(NID) for hit in results if hit.payload]


def hybrid_top3(client, embedder, query_text):
    """Dense (voyage vector) + sparse (qdrant/bm25) prefetch fused with RRF."""
    query_vector = embedder.embed([query_text], input_type="query")[0]
    return client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=query_text, model="qdrant/bm25"),
                using=SPARSE_NAME,
                limit=PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=query_vector,
                using="dense",
                limit=PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=3,
        with_payload=True,
    ).points


def sparse_hits(client, query_text):
    """Raw BM25-only hits for a query (deterministic layer, no dense side)."""
    return client.query_points(
        collection_name=COLLECTION,
        query=models.Document(text=query_text, model="qdrant/bm25"),
        using=SPARSE_NAME,
        limit=5,
        with_payload=True,
    ).points


def test_ac2_in_top3(client, embedder):
    hits = hybrid_top3(client, embedder, "AC-2 account management")
    ids = _ids(hits)
    assert any(i.startswith("nist_ac2") for i in ids), f"AC-2 not in top-3: {ids}"


def test_gdpr_article5_in_top3(client, embedder):
    hits = hybrid_top3(client, embedder, "GDPR Article 5 principles relating to processing")
    ids = _ids(hits)
    assert any(i.startswith("gdpr_art_5") for i in ids), f"GDPR Art 5 not in top-3: {ids}"


def test_invented_token_has_no_sparse_signal(client):
    # "zzqxvbrionqiuu" is an invented single token: BM25 must return zero
    # hits. Dense fallback will always return nearest neighbours, so the
    # honest "not covered" answer depends on BM25 correctly returning nothing.
    hits = sparse_hits(client, INVENTED_TOKEN)
    assert len(hits) == 0, (
        f"BM25 returned {len(hits)} hits for an invented token: {_ids(hits)}"
    )