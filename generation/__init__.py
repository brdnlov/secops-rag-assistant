"""Phase 5 — grounded answer generation with citations.

generation/llm.py     Claude Haiku wrapper + prompt/grounding/citation logic
generation/pipeline.py  end-to-end answer(): retriever -> generator -> answer
"""

from generation.llm import CLAUDE_MODEL, Generator, extract_citations, format_context
from generation.pipeline import answer, generate_answer

__all__ = [
    "CLAUDE_MODEL",
    "Generator",
    "extract_citations",
    "format_context",
    "answer",
    "generate_answer",
]
