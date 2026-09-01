"""Phase 5 — end-to-end answer pipeline.

Threads the two halves together: retrieve context for a query (Phase 4), then
generate a grounded, cited answer (Phase 5). This is what the FastAPI endpoint
(api/) and the eval harness consume, and it's where the per-query observability
trace gets its generation block.

Pipeline:
    query -> Retriever.retrieve()  -> top-N chunks (with trace)
          -> Generator.generate()  -> (answer, citations)
          -> combined result dict
"""

from generation.llm import Generator
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


def _build_retriever(client=None, embedder=None, reranker=None, tracer=None):
    """Retriever with the local cross-encoder attached (best-effort)."""
    if reranker is None:
        try:
            reranker = Reranker()
        except Exception:
            reranker = None
    return Retriever(client=client, embedder=embedder, reranker=reranker, tracer=tracer)


def generate_answer(retriever, generator, query, top_n=5, rerank=True):
    """Run retrieval + generation for a single query. Returns a result dict."""
    retrieved = retriever.retrieve(query, top_n=top_n, rerank=rerank)
    chunks = retrieved["chunks"]
    if not chunks:
        answer_text = "The topic is not covered by the available sources."
        citations = []
    else:
        answer_text, citations = generator.generate(query, chunks)

    return {
        "query": query,
        "answer": answer_text,
        "citations": citations,
        "retrieved_chunks": [c["id"] for c in chunks],
        "retrieved_context": chunks,
        "generation": {
            "model": generator.model,
            "input_tokens": generator.total_input_tokens,
            "output_tokens": generator.total_output_tokens,
        },
    }


def answer(query, top_n=5, rerank=True):
    """One-shot convenience: build defaults and answer a query."""
    generator = Generator()
    retriever = _build_retriever()
    return generate_answer(retriever, generator, query, top_n=top_n, rerank=rerank)
