"""Phase 4 — structured JSON query tracing (framework-free observability).

Appends one JSON object per query to logs/traces.jsonl, covering the full
retrieval + reranking pass: scores per lane, fused rank, reranked chunk ids,
and latency. This is the observability channel for interviewers and for
aggregating metrics (average latency, citation coverage) without any external
platform — per the AGENTS.md "Observability" spec.

Log format (one object per line, jsonl):
{
  "timestamp": 1720000000.0,
  "query": "...",
  "retrieval": {
    "chunks_returned": 20,
    "top_scores": [...],
    "chunk_ids": [...],
    "chunk_sources": [...]
  },
  "reranking": {
    "chunks_after_rerank": 5,
    "reranked_ids": [...]
  }
}
"""

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_FILE = ROOT / "logs" / "traces.jsonl"


class QueryTracer:
    """Appends a structured JSON trace per query. Thread-safe per call."""

    def __init__(self, trace_file=None):
        self.trace_file = Path(trace_file) if trace_file else DEFAULT_TRACE_FILE
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)

    def trace(self, query, retrieval, reranking, elapsed_s=None):
        """Write one JSON object for a query.

        retrieval:  dict with list fields (chunk_ids, chunk_sources, top_scores)
                    and counts.
        reranking:  dict with list fields (reranked_ids, reranked_scores).
        elapsed_s:  total pipeline seconds (optional; the caller times the pass).
        """
        record = {
            "timestamp": time.time(),
            "query": query,
            "retrieval": retrieval,
            "reranking": reranking,
        }
        if elapsed_s is not None:
            record["elapsed_s"] = round(elapsed_s, 4)
        with self.trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
