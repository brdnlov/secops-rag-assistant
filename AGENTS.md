# AGENTS.md — Security/Compliance RAG Assistant

Project rules and architecture reference for OpenCode (and any other agent
working in this repo). This file is stable/rarely-changing — it describes
what the project is and how it's built, not day-to-day session state.

## Do this every time

Whenever you type 'Start Work' in the chat, open the 'session-summaries'
folder, read all the daily summaries, and report: what we accomplished last
session, where we left off, and where overall project progress stands.

## What this project is

A RAG (retrieval-augmented generation) assistant over security and privacy
compliance frameworks — NIST SP 800-53 (subset), OWASP ASVS, and GDPR
statutory text. It answers grounded, cited questions about these frameworks
using hybrid search (BM25 + embeddings) with reranking, backed by Qdrant.

This is a portfolio project for an AI Engineer job transition (see broader
context below). It is a **separate repo** from the eval-harness project
that preceded it — the eval methodology from that project is being carried
forward and adapted here, not merged wholesale.

## Why this domain

- Extends the eval harness's existing domain flavor (row-level security,
  audit logging, CVSS severity, role-gating) so the two portfolio projects
  read as one coherent narrative arc rather than disconnected demos.
- Connects directly to real resume material: a CVSS 7.9 auth bypass and
  CVSS 6.4 IDOR finding from a prior security assessment, and a
  multi-tenant GDPR/CCPA cookie-consent platform built on Cloudflare
  Workers/D1. This gives an interview-ready bridge from "portfolio project"
  to "actual job history."
- Source corpora are genuinely public/open-licensed (NIST SP 800-53, OWASP
  ASVS, GDPR statutory text) — no copyright/ToS risk in a public repo.
- These documents have real cross-referencing and hierarchical structure
  (control IDs, article numbers), giving hybrid search + reranking
  something meaningful to do, unlike a domain with only a handful of flat
  facts.
- Named trade-off, accepted deliberately: this domain is drier to demo
  live and requires real data-wrangling (pulling/cleaning actual
  documents) rather than synthetic content.

## Architecture (target)

```
corpus/       raw + cleaned source documents (NIST subset, OWASP ASVS, GDPR)
              corpus_manifest.json — content hashes, model version, chunking
              config hash for idempotent re-embedding
chunking/     heading/section-aware chunker (RecursiveCharacterTextSplitter),
              preserves control IDs / article numbers / heading paths as
              chunk metadata; target 512 tokens, zero overlap (tunable)
embeddings/   Voyage AI voyage-4 ($0.06/1M tokens), batched, content-hash
              checkpointing to skip unchanged chunks on re-runs
retrieval/    Qdrant with native BM25 sparse vectors + dense vectors,
              RRF fusion (rank constant 60), local cross-encoder reranker
eval/         golden dataset + hand-rolled harness (standalone script),
              RAGAS for side-by-side comparison; minimal 10-item eval wired
              in during generation phase, scaled to 50-100 items in eval phase
              Claude Haiku 4.5 as judge (existing $5 Anthropic credit)
api/          FastAPI app serving the RAG endpoint
logs/         structured JSON traces per query (retrieval scores, reranked
              chunks, citations, latency) — framework-free observability
docker/       docker-compose for Qdrant (self-hosted, not Pinecone —
              avoids the free-tier quota friction hit during eval-harness
              work with Gemini's daily cap)
tests/        5-layer test suite: corpus schema, retrieval sanity, hybrid
              search integration, generation quality, eval regression
README.md     problem, architecture + diagram, chunking/retrieval
              rationale, published eval numbers, known limitations,
              next steps
```

## Build phases (in order)

1. **Corpus acquisition & cleaning** — pull NIST SP 800-53 (AC-2 Account
   Management and AC-6 Least Privilege only, ~15-20 controls — connects
   directly to the CVSS 7.9 auth bypass resume story), OWASP ASVS, and
   full GDPR statutory text. Normalize all three into a consistent
   structured format `{id, source, heading_path, text}` per section —
   the full canonical chunk schema is defined once in Layer 1 of the
   Testing strategy below. Each source
   has a different native format: NIST is XML/HTML, OWASP ASVS is
   structured markdown, GDPR is legal text with article numbers.
   Target ~150-250 pages total. **This is the first task — nothing else
   starts until corpus/ has clean, structured source files.**

   Corpus-specific guidance:
   - NIST: Parse at control level. Each control (AC-2, AC-6) is a
     natural unit. Preserve control metadata (ID, title, family).
   - OWASP ASVS: Each numbered requirement is a chunk. Preserve
     parent verification row as metadata.
   - GDPR: Each article is a natural chunk (most are 200-800 words,
     fit in a single chunk). Preserve article number and chapter.

   **Estimated time: 3-5 days.**

2. **Chunking** — heading/section-aware via RecursiveCharacterTextSplitter.
   Concrete spec (backed by 2025-2026 benchmarks):

   ```
   chunk_size:        512 tokens (pragmatic default — ranked #1 of 7
                      strategies in Feb 2026 vendor benchmark)
   overlap:           0 tokens (start with zero — Jan 2026 arXiv
                      systematic analysis found overlap added no
                      measurable benefit; re-test with 50 if eval
                      shows boundary failures)
   splitter hierarchy:
     1. "\n\n" (section breaks)
     2. "\n" (paragraph breaks)
     3. ". " (sentence breaks)
   max_chunk_size:    1024 tokens (hard ceiling — context cliff at
                      ~2,500 tokens, per arXiv Jan 2026)
   min_chunk_size:    128 tokens (merge into previous chunk if smaller)
   ```

   NIST controls that exceed 1024 tokens: split at sub-control
   boundaries. GDPR articles fit in single chunks without splitting.

   Per-chunk metadata matches the Layer 1 schema exactly: `id` (derived
   from control ID or article number), `source`, `heading_path`, `text`,
   and `chunk_content_hash` (SHA-256 of chunk text). This metadata is
   what makes the BM25 half of hybrid search useful.

   After chunking, compute `chunk_content_hash` for every chunk —
   this enables idempotent re-embedding (see Phase 3).

   **Estimated time: 2-3 days.**

3. **Embedding + Qdrant setup** — Docker-compose for local Qdrant.
   Embed with Voyage AI `voyage-4` ($0.06/1M tokens, 1024 dimensions).
   Free tier: 200M tokens per account — covers initial corpus (~1-2M
   tokens) with massive headroom. The real cost risk is re-embedding
   during eval iterations, not the initial corpus.

   Content-hash checkpointing (idempotent re-embedding):
   - Before embedding, query Qdrant for chunks with matching
     `chunk_content_hash` + `embedding_model_version`.
     Skip embedding for chunks already present.
   - When corpus files change → `corpus_version` increments → only
     changed chunks re-embed.
   - When embedding model changes → `embedding_model_version` changes
     → all chunks re-embed (vector space changed).
   - Store `embedding_model_version` and `chunk_content_hash` as
     Qdrant payload fields on every point.

   Cost guard: halt if cumulative token spend exceeds $5 (~83M tokens
   at voyage-4 — far beyond corpus needs, catches runaway loops).

   Load into Qdrant with chunk metadata as payload. Use Qdrant's
   native BM25 server-side inference (`model: "qdrant/bm25"`) for
   sparse vectors — no separate tokenizer or index to manage.

   **Estimated time: 2-3 days.**

4. **Hybrid retrieval + reranking** — Qdrant-native hybrid search with
   RRF fusion + local cross-encoder reranker.

   Collection config:
   ```
   vectors:
     dense:
       size: 1024          # voyage-4 dimensions
       distance: Cosine
   sparse_vectors:
     bm25:
       modifier: idf       # BM25-style document frequency normalization
   ```

   Query path:
   1. Embed query as dense vector (Voyage AI voyage-4)
   2. Generate query as sparse vector (Qdrant server-side BM25)
   3. Qdrant hybrid query: one dense prefetch + one sparse prefetch,
      fused with RRF (rank constant 60 — Elasticsearch default,
      "requires no tuning" per Cormack paper)
   4. Cross-encoder rerank top-k → return top-5

   Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` via
   sentence-transformers. ~80MB, CPU-only, ~50-200ms for top-20
   reranking. No GPU required, no second API dependency.

   ```python
   from sentence_transformers import CrossEncoder
   reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

   def rerank(query: str, chunks: list[str], top_n: int = 5):
       pairs = [[query, chunk] for chunk in chunks]
       scores = reranker.predict(pairs)
       ranked = sorted(zip(scores, chunks), reverse=True)
       return [chunk for _, chunk in ranked[:top_n]]
   ```

   Upgrade path: if eval shows reranking quality is the bottleneck,
   swap to `BAAI/bge-reranker-v2-m3` (local, ~500MB, 8K context,
   better multilingual) or `Voyage rerank-2.5` (hosted, 32K context,
   instruction-following, +7.9% vs Cohere per vendor claims).

   **Estimated time: 3-4 days.**

5. **Generation with citations** — answers must cite which control/article
   they're grounded in, not just prose. This is where the eval harness
   gets wired in for real, replacing the cheated `contexts` field used
   in the Weeks 2-4 project.

   Wire in a **minimal eval pass during this phase** (10 golden items)
   so generation prompt tuning has feedback. The 10 items cover:
   - 3 exact-retrieval queries (e.g., "What does AC-2 require?")
   - 3 synthesis queries (e.g., "How does GDPR Art. 5(1)(c) relate
     to NIST AC-6?")
   - 2 cross-document queries (e.g., "What NIST controls address data
     retention vs GDPR Art. 17?")
   - 2 negative queries (e.g., "What does NIST say about quantum
     computing?" — must say "not covered")

   **Eval judge strategy (zero-cost goal):**

   During iteration, use RAGAS with Anthropic Claude Haiku via the
   existing $5 API credit. RAGAS supports Anthropic natively via
   `llm_factory`:

   ```python
   from anthropic import Anthropic
   from ragas.llms import llm_factory

   client = Anthropic()  # reads ANTHROPIC_API_KEY from env
   llm = llm_factory("claude-haiku-4-5-20251001", provider="anthropic",
                      client=client)
   ```

   Caveat: the `llm_factory` API has changed across RAGAS releases —
   verify this adapter against the installed RAGAS version before
   relying on it in the eval script.

   Claude Haiku 4.5 costs $1.00/MTok input, $5.00/MTok output.
   A 10-item eval across 4 metrics = ~40 judge calls, costing
   well under $1.00 total. The $5 credit covers this phase
   comfortably.

   For rapid iteration where cost matters more than precision,
   swap to a free judge:
   - **Google Gemini 2.5 Flash Lite** — free tier, OpenAI-compatible
     endpoint, use via LiteLLM adapter
   - **Ollama + Llama 3.2** — fully local, zero API cost, but
     weaker judge (noisier scores — use for relative comparisons
     only, not reported numbers)

   RAGAS also supports **deterministic metrics** that need no
   LLM judge at all: exact match, string presence, ROUGE, BLEU.
   Use these for fast regression checks during development.

   Target: faithfulness >= 0.8, context_precision >= 0.7.

   **Estimated time: 3-4 days.**

6. **Eval layer** — scale the minimal eval from Phase 5 into a full
   evaluation suite. Two parallel systems:

   **Hand-rolled harness** (`eval/run_eval.py` — standalone script):
   - Loads golden dataset from `eval/golden_dataset.json`
   - For each item, calls the RAG endpoint (FastAPI `/query` or
     direct function call)
   - Computes: retrieval precision@3, recall, citation accuracy,
     faithfulness, answer relevancy
   - Writes results to `eval/results/{timestamp}.json`
   - Prints summary pass/fail report

   **RAGAS side-by-side:**
   - Same golden dataset, RAGAS metrics for comparison
   - context_precision, context_recall, faithfulness, answer_relevancy
   - Wrap in retry wrapper (RAGAS returns NaN on malformed LLM judge JSON)
   - Target: faithfulness >= 0.8, context_precision >= 0.7

   Golden dataset: 50-100 items. Start with RAGAS synthetic generation,
   then manually refine. Specifically include cross-document synthesis
   items that probe the failure mode from the prior project:
   - "How does GDPR's data minimization (Art. 5(1)(c)) relate to
     NIST AC-6 (Least Privilege)?" — must cite BOTH
   - "What NIST controls address data retention vs GDPR Art. 17?"
     — must cite NIST AU-11 AND GDPR Art. 17

   Eval interface contract: see the standalone
   [Eval harness interface contract](#eval-harness-interface-contract)
   section below — defined once, shared by both harnesses.

   **Estimated time: 3-5 days.**

7. **Deploy + README** — FastAPI + Docker, public URL. README structured
   like the eval-harness project's: problem, architecture with diagram,
   corpus/chunking rationale, retrieval design, published eval numbers,
   known limitations, next steps. Explicitly link back to the eval-harness
   repo as the source of the eval methodology.

   **Estimated time: 2-3 days.**

### Total estimated time: 18-27 days

## Conventions carried over from the eval-harness project

- Framework-free where it teaches something (chunking, hybrid scoring
  logic) — reach for RAGAS deliberately in eval, not everywhere, so the
  mechanics stay visible.
- Prefer self-hosted infra (Qdrant via Docker) over vendor free tiers with
  hard daily quotas — the Gemini 20-req/day cap cost real time in the
  prior project and shaped this choice directly.
- Every generated answer should be checkpointable / resumable against
  rate-limited or quota-limited APIs — don't repeat the "quota-killed run
  re-spends already-scored work" mistake from the eval harness.
- README gets written against real, published numbers — no placeholder
  text, no claims the eval layer hasn't actually produced yet.

## Corpus & chunk versioning

Every chunk carries a content hash (SHA-256 of chunk text) and embedding
model version. This enables idempotent re-embedding across re-runs:

| Field | Type | Purpose |
|---|---|---|
| `corpus_version` | string (semver) | Tied to git commit of corpus files |
| `chunk_content_hash` | SHA-256 of chunk text | Detect unchanged chunks across re-runs |
| `embedding_model_version` | string | e.g. `voyage-4-2026-07` — triggers full re-embed when changed |
| `chunking_config_hash` | SHA-256 of chunker params | Detect when chunk size/strategy changed |

Manifest file: `corpus/corpus_manifest.json`, committed to git.

Re-embedding logic:
- Corpus changes → `corpus_version` increments → only changed chunks re-embed
- Embedding model changes → `embedding_model_version` changes → all chunks re-embed
- Chunking config changes → `chunking_config_hash` changes → all chunks re-embed

## Testing strategy

Five test layers aligned to RAG pipeline stages:

**Layer 1 — Corpus schema validation** (after Phase 1-2):
- Every chunk has: `id`, `source`, `heading_path`, `text`, `chunk_content_hash`
- Reject empty text, missing heading_path, duplicate IDs

**Layer 2 — Retrieval sanity** (after Phase 3):
- Query "AC-2" → must return AC-2 chunk in top-3
- Query "GDPR Article 5" → must return GDPR Art. 5 in top-3
- Query "nonexistent_control_xyz" → must return empty/low-score results

**Layer 3 — Hybrid search integration** (after Phase 4):
- Exact-match query (BM25 dominates): "AC-2" → high sparse score
- Semantic query (dense dominates): "how should organizations manage
  user accounts" → high dense score
- Hybrid should outperform either alone on mixed queries

**Layer 4 — Generation quality** (after Phase 5):
- Answer cites at least one control/article
- Citation matches a real chunk in the context
- Negative query produces "not covered" response

**Layer 5 — Eval regression gate** (Phase 6):
- RAGAS on golden dataset: faithfulness >= 0.8, context_precision >= 0.7
- Hand-rolled harness: citation accuracy >= 0.9
- Any regression blocks deployment

## Observability

Structured JSON logging per query, no external platform dependency.
One JSON object per query, appended to `logs/traces.jsonl`:

```json
{
  "timestamp": 1700000000.0,
  "query": "What does AC-2 require?",
  "retrieval": {
    "chunks_returned": 10,
    "top_scores": [0.92, 0.87, ...],
    "chunk_ids": ["nist_ac2_001", ...],
    "chunk_sources": ["nist_sp800_53", ...]
  },
  "reranking": {
    "chunks_after_rerank": 5,
    "reranked_ids": ["nist_ac2_001", ...]
  },
  "generation": {
    "answer_length": 342,
    "citations": ["AC-2"],
    "answer": "According to NIST SP 800-53..."
  }
}
```

This is enough to show interviewers exactly what the system retrieved,
scored, reranked, and cited for every query. Aggregate metrics
(average latency, citation coverage) computed from the log.

## Eval harness interface contract

Golden dataset format (`eval/golden_dataset.json`):
```json
{
  "query": "What does AC-2 require?",
  "expected_answer": "NIST AC-2 requires...",
  "expected_citations": ["AC-2"],
  "retrieved_chunks": ["chunk_id_1", "chunk_id_2"],
  "generated_answer": "According to NIST SP 800-53...",
  "generated_citations": ["AC-2"]
}
```

Harness output format (`eval/results/{timestamp}.json`):
```json
{
  "query_id": "golden_001",
  "retrieval": {
    "precision_at_3": 0.67,
    "recall": 1.0,
    "expected_citation_found": true
  },
  "generation": {
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "citation_accuracy": 1.0,
    "grounded": true
  },
  "overall_pass": true
}
```

Harness is a standalone script (`eval/run_eval.py`), not a pytest suite.
This keeps it compatible with the hand-rolled approach from the prior
project and enables side-by-side comparison with RAGAS on the same
golden dataset.

## Broader context (for reference, not this repo's concern directly)

This project is Step 3 of a 6-step, 90-day AI Engineer transition roadmap
(fundamentals → evals → **RAG (this repo)** → agent/tool-use system →
MCP server re-publish → deploy everything + write-ups). The person building
this is a software engineer transitioning from prompt-engineering-heavy
work into demonstrable AI engineering skill, with existing production MCP
server experience (122 role-gated tools, OAuth 2.1/PKCE, Aurora Postgres)
that's being reframed on the resume as agent tool-calling architecture
with governance, not glossed over as "just prompting."
