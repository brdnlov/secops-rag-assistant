"""Phase 3 — embedding + Qdrant ingestion.

embeddings/embedder.py   Voyage AI voyage-4 embedder with batching, token
                         accounting, and the $5 cost guard.
embeddings/indexer.py    Qdrant collection creation + idempotent upsert with
                         content-hash checkpointing.
"""

from embeddings.embedder import Embedder
from embeddings.indexer import (
    COLLECTION,
    CONTENT_HASH,
    EMBED_VERSION,
    EMBEDDING_MODEL_VERSION,
    HEADING_PATH,
    META,
    NID,
    POINT_NS,
    SOURCE,
    SPARSE_NAME,
)

__all__ = [
    "Embedder",
    "COLLECTION",
    "CONTENT_HASH",
    "EMBED_VERSION",
    "EMBEDDING_MODEL_VERSION",
    "HEADING_PATH",
    "META",
    "NID",
    "POINT_NS",
    "SOURCE",
    "SPARSE_NAME",
]