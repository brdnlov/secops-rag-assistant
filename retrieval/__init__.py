"""Phase 4 — hybrid retrieval + reranking.

retrieval/retriever.py  voyage query-embed -> dense + sparse prefetch -> RRF
                        fusion -> cross-encoder rerank -> top-TOP_N
retrieval/reranker.py   local cross-encoder wrapper
                        (BAAI/bge-reranker-v2-m3 since Phase 6)
retrieval/tracer.py     structured JSON traces to logs/traces.jsonl
"""

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from retrieval.tracer import QueryTracer

__all__ = ["Retriever", "Reranker", "QueryTracer"]
