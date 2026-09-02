"""Phase 4 — local cross-encoder reranker.

Wraps a local cross-encoder via sentence-transformers. Reranks a candidate
list by how well each chunk matches the query when the two are encoded
TOGETHER (cross-attention), which is more accurate than the bi-encoder cosine
used in the dense lane.

Deliberately CPU-only / local: no GPU, no second API dependency.

Model history (Phase 6, 2026-09-01):
  - v1 `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB) — retired after the
    Phase 6 diagnosis showed it demoting correct chunks out of top-5 on 4/5
    probe queries (e.g. GDPR Art. 32 at fused rank #1 dropped from the answer
    context). General 2019-era IR model, English-only, no domain signal.
  - v2 `BAAI/bge-reranker-v2-m3` (~500MB, multilingual, 8K context) — the
    AGENTS.md-planned upgrade path; stronger deep-relevance ranking. This is
    now the default.
"""

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# v1 model kept for reference / comparison runs.
MODEL_NAME_MSMARCO = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cross-encoders are trained on short passages (~100 words, MS-MARCO) and
# systematically demote long documents regardless of content. The Phase 6
# comparison run showed full-text scoring ranked GDPR Art. 32 (fused rank #1)
# 7th; truncating each candidate to 320 chars for SCORING only lifted it to
# 4th. Full chunk text is still delivered to generation unchanged.
MAX_SCORE_CHARS = 320


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

        Scoring uses a character-trimmed copy of each chunk's text
        (MAX_SCORE_CHARS) to neutralize the trained length bias; the chunk
        dicts themselves are returned untouched.
        """
        if not chunks:
            return []
        texts = [c["text"][:MAX_SCORE_CHARS] for c in chunks]
        pairs = [[query, t] for t in texts]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(scores, chunks), key=lambda t: t[0], reverse=True)
        ordered = [c for _, c in ranked]
        if top_n is not None:
            return ordered[:top_n]
        return ordered
