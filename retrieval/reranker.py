"""Phase 4 — local cross-encoder reranker.

Wraps `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, CPU-only) via
sentence-transformers. Reranks a candidate list by how well each chunk matches
the query when the two are encoded TOGETHER (cross-attention), which is more
accurate than the bi-encoder cosine used in the dense lane.

~50-200ms for a top-20 rerank. Deliberately CPU-only / local: no GPU, no
second API dependency.

Upgrade path (per AGENTS.md): if eval shows reranking quality is the
bottleneck, swap the model string to `BAAI/bge-reranker-v2-m3` (local, ~500MB,
8K context) or `Voyage rerank-2.5` (hosted).
"""

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name=MODEL_NAME, device=None):
        # Imported lazily so non-reranking code (e.g. the API serving a query
        # with reranking disabled) can import this module without the ~80MB
        # local model download.
        from sentence_transformers import CrossEncoder

        kwargs = {}
        if device:
            kwargs["device"] = device
        self._model = CrossEncoder(model_name, **kwargs)
        self.model_name = model_name

    def rerank(self, query, chunks, top_n=None):
        """Return the input chunks reordered by relevance to the query.

        chunks: list of dicts with at least a 'text' key.
        Returns a list of the same chunk dicts, sorted best-first. If top_n
        is given, only the top top_n are returned.
        """
        if not chunks:
            return []
        pairs = [[query, c["text"]] for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, chunks), key=lambda t: t[0], reverse=True)
        ordered = [c for _, c in ranked]
        if top_n is not None:
            return ordered[:top_n]
        return ordered
