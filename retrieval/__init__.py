"""Phase 4 — hybrid retrieval + reranking.

retrieval/retriever.py  voyage query-embed -> dense + sparse prefetch -> RRF
                        fusion -> cross-encoder rerank -> top-5
retrieval/reranker.py   local cross-encoder wrapper
                        (cross-encoder/ms-marco-MiniLM-L-6-v2)
retrieval/tracer.py     structured JSON traces to logs/traces.jsonl
"""

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from retrieval.tracer import QueryTracer

__all__ = ["Retriever", "Reranker", "QueryTracer"]
