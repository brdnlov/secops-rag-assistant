"""Phase 5 — grounded answer generation with citations.

The final stage of the RAG pipeline: given a user query and the retrieved
context (top-N chunks from retrieval/), produce a grounded answer that cites
the specific NIST control / GDPR article / ASVS requirement it is based on.

Design follows the AGENTS.md "Generation with citations" phase:
  - Every claim is grounded in the provided context (no outside knowledge).
  - Answers cite real identifiers (AC-2, Article 5, V13.3.2) that map to
    actual chunks in the context.
  - Negative/out-of-coverage queries produce an explicit "not covered"
    answer rather than a confident guess.

The LLM is Claude Haiku (per AGENTS.md), but the prompt/grounding/citation
logic here is framework-free — the LLM is a thin replaceable backend.
"""

import json
from pathlib import Path

# Load ANTHROPIC_API_KEY from the repo-local .env (gitignored, never committed).
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"


def _en_nist_subcontrol(label):
    """'AC-2.5' -> 'AC-2'; 'AC-2' -> 'AC-2' (parent of a NIST enhancement)."""
    return label.split(".")[0] if "." in label else label


def _normalize(label):
    """Canonicalize an identifier for citation matching.

    Tolerates real-world notation drift between the catalog's 'AC-2.5' and a
    model's 'AC-2(5)': strip spaces and parens, collapse '.5' and '(5)' to a
    common form, lowercase for matching.
    """
    import re
    s = str(label).strip()
    s = s.lower()
    s = s.replace("article", "art ")
    # 'AC-2(5)' -> 'ac-2(5)' ; 'AC-2.5' -> 'ac-2.5'
    s = re.sub(r"[()]", "", s)       # ac-25
    s = re.sub(r"\.", "", s)          # ac-25
    s = re.sub(r"\s+", "", s)
    return s


def _citation_label(chunk, parent=True):
    """Derive the human-facing citation label for a chunk.

    Maps a corpus chunk to the identifier users/evals expect:
      NIST  nist_ac6   -> "AC-6"     (metadata.control_id)
      NIST  nist_ac2_5 -> "AC-2"      (parent of the AC-2.5 enhancement, so a
                                       human "[AC-2]" maps to a sub-control chunk)
      GDPR  gdpr_art_5 -> "Article 5" (metadata.article_number)
      ASVS  asvs_v13_3_2 -> "V13.3.2" (metadata.req_id)
    Falls back to the chunk id when no friendlier label exists.
    """
    meta = chunk.get("metadata") or {}
    cid = chunk.get("id", "")
    if meta.get("control_id"):
        label = meta["control_id"]
        if parent:
            label = _en_nist_subcontrol(label)
        return label
    if meta.get("req_id"):
        return meta["req_id"]
    if "article_number" in meta:
        return f"Article {meta['article_number']}"
    return cid


def _doc_label(chunk):
    """Human-readable identity line for the prompt's context listing."""
    parts = [chunk.get("id"), chunk.get("heading_path")]
    return " | ".join(p for p in parts if p)


def format_context(chunks):
    """Render the retrieved chunks into the prompt's context block.

    Each chunk is prefixed with a SOURCE line carrying BOTH the stable chunk id
    and the canonical citation identifier ([AC-2], [Article 5], [V13.3.2]) so
    the generator cites exact labels we can map back to the context.
    """
    if not chunks:
        return "(no relevant source passages were retrieved)"
    lines = []
    for i, c in enumerate(chunks, 1):
        cite = _citation_label(c, parent=False)
        lines.append(
            f"[{i}] SOURCE {_doc_label(c)}  CITATION [{cite}]\n{c.get('text', '')}"
        )
    return "\n\n".join(lines)


def extract_citations(answer, chunks):
    """Pull the set of citation labels actually referenced in the answer.

    Walks the provided chunks' known labels and returns those whose normalized
    form also appears in the answer. Normalization tolerates notation drift
    ('AC-2(5)' vs 'AC-2.5', 'Article 5' vs 'art 5'), and NIST sub-control
    chunks report their parent ('AC-2.5' -> 'AC-2') so a human-style "[AC-2]"
    in the answer maps back to an enhancement chunk. Used to verify the answer
    is grounded in the context (faithfulness) and to surface citations.
    """
    labels = [_citation_label(c) for c in chunks]
    normalized_answer = _normalize(answer)
    found = []
    for label in labels:
        if label and label.strip() and _normalize(label) in normalized_answer:
            if label not in found:
                found.append(label)
    return found


class Generator:
    """Produce a grounded, cited answer from retrieved context via Claude."""

    def __init__(self, client=None, model=CLAUDE_MODEL, max_tokens=500):
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _system_prompt(self):
        return (
            "You are a security and privacy compliance assistant. Answer the user's "
            "question using ONLY the source passages provided in the context. "
            "Ground every claim in a specific source and cite it inline using the "
            "exact bracketed identifier shown in the CONTEXT's SOURCE line, e.g. "
            "[AC-2], [Article 5], [V13.3.2]. Use ONLY those exact identifiers, never "
            "a different format, never a bare [1] or a URL. "
            "Only cite a source that is actually present in the context. "
            "If the context does not cover the question, say the topic is not "
            "covered by the available sources. Do not use outside knowledge."
        )

    def _user_prompt(self, query, chunks):
        return (
            f"CONTEXT (retrieved source passages):\n{format_context(chunks)}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer the question based only on the CONTEXT above. Cite sources "
            f"inline with the exact bracketed identifiers from the CONTEXT SOURCE "
            f"lines, and explain the reasoning."
        )

    def generate(self, query, chunks):
        """Run the generator. Returns (answer, citations).

        citations is the list of labels found in the answer that map to real
        context chunks (for the caller and the eval/fidelity check).
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": self._user_prompt(query, chunks)}],
        )
        usage = getattr(message, "usage", None)
        if usage:
            self.total_input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.total_output_tokens += getattr(usage, "output_tokens", 0) or 0

        answer = "".join(block.text for block in message.content if block.type == "text")
        citations = extract_citations(answer, chunks)
        return answer, citations

    def cost_summary(self):
        # Approx prices: haiku-4.5 $1/MTok in, $5/MTok out (per AGENTS.md).
        spend = (
            self.total_input_tokens * 1e-6
            + self.total_output_tokens * 5e-6
        )
        return {
            "model": self.model,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "approx_spend_usd": round(spend, 6),
        }
