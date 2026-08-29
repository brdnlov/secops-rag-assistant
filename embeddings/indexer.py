"""Phase 3 — self-hosted Qdrant indexer with content-hash checkpointing.

Creates the collection (dense 1024-dim cosine + native qdrant/bm25 sparse),
embeds the corpus with the Embedder, and upserts only chunks whose
chunk_content_hash + embedding_model_version are not already present
(idempotent re-embedding; see AGENTS.md "Corpus & chunk versioning").

Collection config (per AGENTS.md):
    vectors.dense:      size 1024, distance Cosine
    sparse_vectors.bm25: modifier idf     (Qdrant native server-side BM25)

Every point carries as payload:
    nid                 -> the stable corpus chunk id (nist_ac2, gdpr_art_4_01, ...)
    source              -> corpus source
    heading_path        -> for BM25 context / display
    chunk_content_hash  -> SHA-256 of chunk text (checkpoint key)
    embedding_model_version -> triggers full re-embed when changed (checkpoint key)
    metadata            -> the per-source unit metadata (control_id, title, ...)

Qdrant point ids are deterministic UUIDs derived from each chunk's stable id
(via uuid5) so re-runs overwrite the same point instead of duplicating it.
The human-readable `nid` (nist_ac2, gdpr_art_4_01, ...) is kept as a payload
field for retrieval/display and stable pagination.

Usage:
    python -m embeddings.indexer
Requires: Qdrant reachable at QDRANT_URL (default http://localhost:6333) and
a populated corpus/ (corpus.json).
"""

import json
import sys
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

from embeddings.embedder import EMBEDDING_DIM, MODEL, Embedder

ROOT = Path(__file__).resolve().parents[1]
CORPUS_JSON = ROOT / "corpus" / "corpus.json"
MANIFEST_JSON = ROOT / "corpus" / "corpus_manifest.json"

COLLECTION = "secops_corpus"
EMBEDDING_MODEL_VERSION = "voyage-4-2026-08"
QDRANT_URL = "http://localhost:6333"

# Qdrant point ids must be UUIDs or unsigned integers; map the stable chunk id
# to a deterministic uuid5 so re-runs overwrite the same point instead of duping.
# Kept stable across runs (idempotent upsert) — chunk ids never change after
# chunking, so the uuid5 mapping is a pure function of the chunk id.
POINT_NS = uuid.UUID("8a1e2f3c-4b5d-4e6f-8a9b-0c1d2e3f4a5b")

# BM25 slot name. Qdrant's server-side BM25 model ("qdrant/bm25") is the
# exception to cloud-only inference: it runs on self-hosted Qdrant, so the
# sparse side needs no separate tokenizer/index (AGENTS.md Phase 3).
SPARSE_NAME = "bm25"

# Payload field names (kept consistent with retrieval + tests).
NID = "nid"
SOURCE = "source"
HEADING_PATH = "heading_path"
TEXT = "text"  # the exact chunk text; kept on payload for traceability
CONTENT_HASH = "chunk_content_hash"
EMBED_VERSION = "embedding_model_version"
META = "metadata"


def _vector_params():
    return models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)


def _sparse_params():
    # Qdrant native BM25 modifier=idf, no tokenizer to manage.
    return models.SparseVectorParams(modifier=models.Modifier.IDF)


def create_collection(client):
    """Idempotent collection ensure. Returns True if created, False if existed."""
    existing = client.get_collections().collections
    if any(c.name == COLLECTION for c in existing):
        return False
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": _vector_params()},
        sparse_vectors_config={SPARSE_NAME: _sparse_params()},
    )
    return True


def _existing_hashes(client):
    """Set of content hashes already stored for this embedding model version."""
    hashes = set()
    offset = None
    while True:
        res = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=EMBED_VERSION, match=models.MatchValue(value=EMBEDDING_MODEL_VERSION)
                    )
                ]
            ),
            with_payload=[CONTENT_HASH],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        points, offset = res
        for p in points:
            if CONTENT_HASH in (p.payload or {}):
                hashes.add(p.payload[CONTENT_HASH])
        if offset is None:
            break
    return hashes


def _point(chunk, vector):
    return models.PointStruct(
        id=uuid.uuid5(POINT_NS, chunk["id"]),
        vector={
            "dense": vector,
            SPARSE_NAME: models.Document(text=chunk["text"], model="qdrant/bm25"),
        },
        payload={
            NID: chunk["id"],
            SOURCE: chunk["source"],
            HEADING_PATH: chunk["heading_path"],
            TEXT: chunk["text"],
            CONTENT_HASH: chunk["chunk_content_hash"],
            EMBED_VERSION: EMBEDDING_MODEL_VERSION,
            META: chunk.get("metadata") or {},
        },
    )


def upsert_chunks(client, embedder, chunks):
    """Embed then upsert only chunks not already present (checkpoint skip).

    Batch-granular: each batch is embedded, then upserted with wait=True
    before the next starts. Completed batches carry their content hashes into
    Qdrant immediately, so a run killed by the API mid-way resumes from the
    next batch rather than re-spending the finished ones (the "quota-killed
    run" lesson from the eval-harness project).
    """
    existing = _existing_hashes(client)
    to_index = [c for c in chunks if c["chunk_content_hash"] not in existing]
    print(f"Total chunks: {len(chunks)}")
    print(f"Already indexed for {EMBEDDING_MODEL_VERSION}: {len(chunks) - len(to_index)}")
    if not to_index:
        print("Nothing to embed — all chunks already present for this model version.")
        return None, []

    total = len(to_index)
    upserted = 0
    for i in range(0, total, embedder.batch_size):
        batch_chunks = to_index[i:i + embedder.batch_size]
        texts = [c["text"] for c in batch_chunks]
        tokens_before = embedder.total_tokens
        vectors = embedder.embed(texts)
        batch_tokens = embedder.total_tokens - tokens_before
        points = [_point(c, v) for c, v in zip(batch_chunks, vectors)]
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        upserted += len(points)
        # Pace the NEXT call inside the free-tier windows. embed() only sleeps
        # between ITS internal batches; with one batch per indexer iteration the
        # pacing must come from here, otherwise calls go back-to-back and trip
        # the 3 RPM burst check.
        if upserted < total:
            sleep_s = embedder.throttle_sleep(batch_tokens)
            print(f"[indexer] {upserted}/{total} upserted (batch {len(points)}, "
                  f"{batch_tokens} tok), cumulative ${embedder.total_spend:.4f} "
                  f"-> next call in {sleep_s:.0f}s")
            time.sleep(sleep_s)
        else:
            print(f"[indexer] {upserted}/{total} upserted (final batch "
                  f"{len(points)}, {batch_tokens} tok), cumulative "
                  f"${embedder.total_spend:.4f}")
    print(f"Upserted {upserted} points into '{COLLECTION}'.")
    return True, to_index


def main():
    # Lazy: require Qdrant only at runtime, so embed-only workflows can import.
    if not (CORPUS_JSON.exists() and MANIFEST_JSON.exists()):
        print("corpus/corpus.json + corpus_manifest.json required. Run chunking first.")
        raise SystemExit(1)

    client = QdrantClient(url=QDRANT_URL)
    created = create_collection(client)
    print(f"Collection '{COLLECTION}': {'created' if created else 'already exists'}")

    chunks = json.loads(CORPUS_JSON.read_text(encoding="utf-8"))
    embedder = Embedder()
    try:
        upsert_chunks(client, embedder, chunks)
    except Exception as e:
        print(f"Stopped: {e}")
        # surface cost accounting even on partial failure
        raise
    finally:
        summary = embedder.cost_summary()
        print("Embedding cost:", json.dumps(summary, indent=2))

    _update_manifest_embedding(embedder_cost=summary.get("total_tokens", 0))


def _update_manifest_embedding(embedder_cost=None):
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    manifest["embedding_model_version"] = EMBEDDING_MODEL_VERSION
    if embedder_cost is not None:
        manifest["embedding_progress"] = {
            "model": MODEL,
            "total_tokens": embedder_cost,
            "collection": COLLECTION,
        }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest updated: embedding_model_version = {EMBEDDING_MODEL_VERSION}")


if __name__ == "__main__":
    sys.exit(main())
