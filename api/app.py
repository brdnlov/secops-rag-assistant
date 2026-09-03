"""Phase 5/7 — FastAPI RAG endpoint.

Exposes the retrieval + generation pipeline as an HTTP service:

    GET  /health        -> service + Qdrant health
    POST /query         -> {query} -> {answer, citations, retrieved, generation}

The generator and retriever are constructed once at startup (the reranker and
Claude client are reused across requests). Every query still writes a
structured JSON trace to logs/traces.jsonl via the retriever.

Modelled on the AGENTS.md architecture's `api/` layer. The reranker model and
the Claude client are built once; if either is unavailable the service still
starts, degrading gracefully to raw hybrid retrieval + empty generation.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from generation.llm import Generator
from generation.pipeline import generate_answer, _build_retriever
from embeddings.indexer import COLLECTION
from qdrant_client import QdrantClient

from retrieval.retriever import QDRANT_URL

app = FastAPI(title="Security/Compliance RAG Assistant")


class QueryRequest(BaseModel):
    query: str
    # Matches the measured context size (retriever.TOP_N = 12) that the eval
    # gate passes on; the retriever applies TOP_N as its own fallback.
    top_n: int = 12
    rerank: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list
    retrieved: list
    generation: dict
    collection: str


# Application-scoped singletons (built lazily on first use).
_retriever = None
_generator = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = _build_retriever()
    return _retriever


def _get_generator():
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


@app.on_event("startup")
def startup():
    # Build the reranker + generator eagerly so config errors surface at startup.
    _get_retriever()
    _get_generator()


@app.get("/health")
def health():
    try:
        info = _get_retriever().client.get_collection(COLLECTION)
        qdrant_ok = info.points_count is not None
    except Exception:
        qdrant_ok = False
    return {
        "status": "ok" if qdrant_ok else "degraded",
        "collection": COLLECTION,
        "qdrant_reachable": qdrant_ok,
        "generator": _get_generator().model,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query must be non-empty")
    result = generate_answer(
        _get_retriever(),
        _get_generator(),
        req.query,
        top_n=req.top_n,
        rerank=req.rerank,
    )
    return QueryResponse(
        query=req.query,
        answer=result["answer"],
        citations=result["citations"],
        retrieved=result["retrieved_chunks"],
        generation=result["generation"],
        collection=COLLECTION,
    )
