"""Layer 4 — generation quality (Phase 5 gate).

Verifies the generator produces grounded, cited answers and honest negative
responses:

  - a real query yields an answer that cites at least one control/article
  - every generated citation is backed by a chunk that was actually retrieved
    (grounding — no hallucinated citations)
  - a negative/out-of-coverage query yields a "not covered" response

These are the three canonical Layer 4 probes. They exercise the full
retrieve -> generate pipeline via generation.pipeline.

Requires Qdrant (docker compose up + indexer), VOYAGE_API_KEY (query
embedding), and ANTHROPIC_API_KEY (generation). Skips cleanly if any are
missing.
"""

import os
from pathlib import Path

import pytest

from generation.llm import Generator, _citation_label
from generation.pipeline import generate_answer, _build_retriever


@pytest.fixture(scope="module")
def retriever():
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not set — query embedding requires Voyage")
    from qdrant_client import QdrantClient
    from embeddings.indexer import COLLECTION
    from retrieval.retriever import QDRANT_URL
    try:
        QdrantClient(url=QDRANT_URL).get_collection(COLLECTION)
    except Exception:
        pytest.skip("Qdrant not reachable or collection missing (docker compose up + indexer first)")
    return _build_retriever()


@pytest.fixture(scope="module")
def generator():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — generation requires Claude")
    return Generator()


# --- Answer cites at least one control/article for a real query ---
def test_real_query_cites_source(retriever, generator):
    result = generate_answer(retriever, generator, "What does AC-2 require?", top_n=3)
    assert len(result["answer"]) > 0
    assert result["citations"], "answer should cite at least one source"


# --- Every generated citation is backed by a retrieved chunk (grounding) ---
def test_citations_grounded_in_context(retriever, generator):
    result = generate_answer(retriever, generator, "What does AC-2 require?", top_n=3)
    # Build the grounding set at both granularities: since Phase 6 the
    # generator cites exact enhancements ([AC-2.7]) when present, and that
    # specific label must ground against its enhancement chunk (nist_ac2_7).
    labels = set()
    for c in result["retrieved_context"]:
        labels.add(_citation_label(c, parent=True))
        labels.add(_citation_label(c, parent=False))
    assert result["citations"], "no citations to check"
    assert set(result["citations"]).issubset(labels), (
        f"citation(s) {result['citations']} not in retrieved context {sorted(labels)}"
    )


# --- Negative/out-of-coverage query yields a "not covered" response ---
def test_negative_query_not_covered(retriever, generator):
    result = generate_answer(retriever, generator, "What does NIST say about quantum computing?", top_n=3)
    low = result["answer"].lower()
    assert ("not covered" in low) or ("does not" in low) or ("do not" in low) or ("no information" in low), (
        f"negative query should say not covered, got: {result['answer'][:200]}"
    )
