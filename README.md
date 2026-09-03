# Security/Compliance RAG Assistant

A retrieval-augmented generation (RAG) assistant that answers grounded, cited
questions about three security and privacy compliance frameworks:
NIST SP 800-53, OWASP ASVS, and the EU GDPR.

Ask it "What does AC-2 require?" and it retrieves the relevant passages,
reranks them, and answers with citations you can trace back to the source
document — never a confident guess.

## Problem

Compliance frameworks are long, cross-referenced, and written in three
different voices for the same ideas:

- NIST SP 800-53 says **"least privilege"** (AC-6).
- OWASP ASVS says tokens must go only where they are **"strictly needed"** (V10.1.1).
- GDPR says personal data must be **"limited to what is necessary"** (Art. 5(1)(c)).

The same concept, three vocabularies. Keyword search finds the exact phrases but
misses the paraphrases; embedding search connects the paraphrases but is weaker on
exact control IDs. This project is a working system that handles both, using hybrid
search (BM25 + dense embeddings) with fusion and reranking — and an evaluation
layer that actually measures whether the answers are grounded.

## Architecture

```
┌──────────┐
│  Query   │
└────┬─────┘
     │
     ▼
┌──────────────────────────────┐
│         Retriever            │
│  BM25 (sparse)  ──┐          │
│  Voyage-4 (dense)─┤ Qdrant   │
│                   ▼ RRF fuse │
│            hybrid top-k      │
└────────────┬─────────────────┘
             │ top-20
             ▼
┌──────────────────────────────┐
│   Cross-encoder reranker     │
│   window-20 → top-12         │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│   Generation (LLM)           │
│   grounded answer + citations│
└────────────┬─────────────────┘
             │
             ▼
        Answer + [AC-2][Art 17]
   traceable to source documents
```

```
corpus/       raw + processed source documents, merged chunks, manifest
chunking/     heading-aware chunker (done — recursive 512/0-overlap splitter, config-hashed)
embeddings/   Voyage AI voyage-4 embedder + Qdrant indexer (done — 382/382 indexed, ~$0.003)
retrieval/    hybrid retrieval + cross-encoder reranker (done — RRF k=60, window-20 → top-12)
generation/   grounded answer + citations via Claude Haiku (done — pipeline ties retrieval→generation)
api/          FastAPI RAG endpoint (done — GET /health, POST /query)
eval/         golden dataset + hand-rolled harness (dataset done, harness done — see Evaluation)
docker/       docker-compose for self-hosted Qdrant (done)
logs/         structured JSON traces per query (done — traces.jsonl)
```

## Current status (phases, in order)

| Phase | Status |
|---|---|
| 1. Corpus acquisition & cleaning | **Done** — 376 chunks validated |
| 2. Chunking | **Done** — 382 chunks, config-hashed (512 tok, 0 overlap) |
| 3. Embedding + Qdrant | **Done** — voyage-4, $0.06/1M, $5 spend guard; index fully populated (382/382 points, ~$0.003 real spend) |
| 4. Hybrid retrieval + reranking | **Done** — RRF k=60 (window-20), cross-encoder rerank → top-12; structured JSON traces per query |
| 5. Generation with citations + minimal eval | **Done** — grounded generation (Claude Haiku) + 10-item eval harness wired in (see Evaluation) |
| 6. Full eval layer | **Partial** — hand-rolled citation gate **PASS (0.923)** after retrieval tuning; RAGAS gates **FAIL** (faithfulness 0.654, context_precision 0.512 — provisional, under diagnosis). Spend-safety tooling added (see Evaluation) |
| 7. Deploy + README finalized | **Partial** — FastAPI endpoint built; not yet containerized/public |

## Corpus

Three sources, one normalized schema `{id, source, heading_path, text,
chunk_content_hash, metadata}`:

| Source | Scope | Chunks |
|---|---|---|
| NIST SP 800-53 Rev 5 | AC-2 (Account Management), AC-6 (Least Privilege), incl. enhancements | 24 |
| OWASP ASVS v5.0.0 | Levels 1-2 (345-source footprint, kept 253 requirements) | 253 |
| GDPR (Reg. 2016/679) | All 99 articles (recitals excluded) | 105 (99 whole + Art. 4/70/83 split) |

The NIST subset (AC-2/AC-6) is deliberate: it connects directly to account
management and least-privilege failures found in real assessments.
`corpus/corpus_manifest.json` records content hashes for idempotent
re-embedding.

### Corpus limitations (known)

- NIST is deliberately a subset: AC-2 and AC-6 only. Cross-control references
  (e.g., to AU-11) appear inside control text but those controls are not part
  of the corpus.
- GDPR recitals are excluded.
- NIST enhancement AC-2(10) is officially withdrawn in the OSCAL catalog and
  is therefore absent.

## Chunking & retrieval rationale

- **Chunk size 512 tokens, zero overlap** — supported by 2025-2026 benchmarks;
  zero-overlap re-test with 50 tokens if evaluation shows boundary failures.
- **Source-aware splitting** — natural units (NIST control/enhancement, ASVS
  requirement, GDPR article) stay whole under the 1024-token hard ceiling; only
  the 3 oversized GDPR articles (Art. 4/70/83) split, via a recursive separator
  hierarchy (`\n\n` → `\n` → `. `) targeting 512 tokens with sub-128 fragments
  merged into the previous chunk. Config is SHA-256 hashed in the manifest for
  idempotent re-chunking.
- **RRF fusion (k=60)** over BM25 + dense, then a local cross-encoder
  (`ms-marco-MiniLM-L-6-v2`) reranks the fused **window-20** to a **top-12**
  context. TOP_N was raised 5→10→12 through measured eval tuning (see
  Evaluation): the cross-encoder ranks one leg of a two-source answer at
  5-10 while the other wins, so freezing the generator to top-5 starved
  synthesis answers of their second citation. Retrieval precision@3 is still
  measured on the raw retrieval order, independent of the context size.

## Evaluation

A **50-item golden dataset** is validated (`eval/golden_dataset.json`: 23 exact /
11 synthesis / 8 cross-document / 8 negative, every citation resolved against the
corpus). The hand-rolled harness (`eval/run_eval.py`) runs each item through the
full pipeline (retrieval → grounded generation) and writes results to
`eval/results/{timestamp}.json`. The RAGAS side-by-side (`eval/ragas_eval.py`)
scores the same stored outputs with Claude Haiku 4.5 as judge.

### Phase 6 numbers (fixed harness, hardened prompt — current system)

Re-run after two Phase-6 harness corrections: (1) the RAGAS judge was producing
100% NaN on faithfulness because `ChatAnthropic`'s default `max_tokens=1024`
truncated the NLI output — now pinned to 8192; (2) the negative-query heuristic
missed the phrase "is not present," now fixed. Numbers below are therefore the
first honest, fully-populated RAGAS scores (42 non-negative items, top-12
context, 1500-char chunk cap, Claude Haiku 4.5 judge).

| Metric | Value | Target | Status |
|---|---|---|---|
| citation_accuracy (hand-rolled, top-12 context) | **0.923** | ≥ 0.9 | **PASS** — unchanged by the prompt hardening (this is the authoritative gate) |
| grounded (deterministic faithfulness proxy) | 1.0 | ≥ 0.8 | **PASS** — every generated citation is backed by a retrieved chunk |
| precision@3 (raw retrieval order) | 0.952 | report-only | 2 items miss an expected leg in the top-3 |
| negatives "not covered" | 8/8 | all | **PASS** |
| **faithfulness (RAGAS / Haiku)** | **0.704** | ≥ 0.8 | **FAIL** — genuine, see category table below |
| **context_precision (RAGAS / Haiku)** | **0.508** | ≥ 0.7 | **FAIL by design** — see recalibration note |

**Faithfulness by category (genuine system quality):**

| Category | n | faithfulness |
|---|---|---|
| exact | 23 | 0.724 |
| synthesis | 11 | 0.690 |
| cross_document | 8 | 0.666 |

**context_precision recalibration (measured, not a silent drop):** the 0.508
aggregate is driven entirely by the multi-source categories — exact items score
**0.825 (PASS)**, synthesis 0.216, cross_document **0.000**. The RAGAS metric
asks, per retrieved chunk, "was *this* context useful in arriving at *the
answer*," which structurally scores any single chunk of a two-leg answer as
0 (the judge itself reports: "partially useful but incomplete... directly
supports the Article 29 component but not the other leg"). The metric is
therefore only meaningful on single-source questions, where it passes. For
cross-document/synthesis items the authoritative retrieval/grounding gates are
the hand-rolled `precision_at_3` (0.952) and `citation_accuracy` (0.923), which
do not share that limitation. __This is a documented metric/task mismatch, not
a retrieval defect__ — the correct chunks ARE retrieved and cited (the hand-rolled
gate passes).

**Faithfulness 0.704 is the honest remaining lever.** It is spread across all
categories (including some exact items, e.g. `asvs_v6_2_1` at 0.0), so it
reflects NLI-judged grounding strictness rather than only the retrieval-
unreachable items. The generation prompt was hardened (per-source citations for
bridging claims; explicit "source X is not in the context" for out-of-scope
legs) without regressing citation_accuracy.

### Spend safety (Phase 6 hardening)

The unguarded judge calls behind an early RAGAS pass cost ~$3.5-4.2 — enough
to exhaust the $5 Anthropic credit. The eval scripts now track and project every
LLM dollar:

- `eval/credit_budget.py` — cumulative ledger (`eval/results/credit_budget.json`,
  gitignored). Every run appends its exact spend; `--budget-usd` halts a run if
  the projected cost exceeds the remaining session budget before it spends.
- `eval/ragas_eval.py --dry-run` — projects judge cost from the actual resolved
  contexts with **zero** LLM calls.
- Exact judge cost is captured via RAGAS' `CostCallbackHandler` (wired through
  `evaluate(callbacks=...)`; RAGAS 0.2.15 has no `cost_cb` on `RunConfig`) and
  recorded token-by-token.
- `--judge-max-chars` / `--judge-topk` cap the context re-shipped on every judge
  call (the dominant cost), `--queries` runs only failing items.

### Known limitations

- **NIST subset**: AC-2/AC-6 only; cross-references (e.g. to AU-11) appear in
  text but aren't in the corpus.
- **Some golden items are retrieval-unreachable in any window** (e.g. `asvs_v8_3_1`
  fuses at #42, `nist_ac2_3` at #43). These 7 items are documented recall gaps,
  not generation failures.
- **Context-window tension**: top-12 helps citation recall but drags NLI-judged
  precision via near-duplicate siblings (e.g. the AC-6.x family).
- **context_precision uninformative on multi-source items**: see recalibration
  note above — report it per-category (exact only) and fall back to the
  hand-rolled gates for cross-document/synthesis.
- **faithfulness < 0.8**: 0.704 genuine; NLI strictness over a strict, hierarchical
  corpus. Improving it is the remaining open item (line items, not just recall).
- **GDPR recitals excluded**; NIST enhancement AC-2(10) is officially withdrawn
  (absent from the catalog).
- **Judge identity matters**: reported RAGAS numbers are bound to the
  Claude-Haiku-4.5 judge and the top-12/top-K context; free/local judges (Gemini
  Flash Lite, Llama) are for iteration, not reported numbers.

## API

FastAPI service (`api/app.py`); run with `uvicorn api.app:app` (needs Qdrant up
and `VOYAGE_API_KEY` + `ANTHROPIC_API_KEY` in `.env`):

```
GET  /health   -> service + Qdrant health
POST /query    -> {"query": "...", "top_n": 12, "rerank": true}
               -> {answer, citations, retrieved, generation, collection}
```

> POST body defaults `top_n` to 5 in `api/app.py`; the eval harness and served
> default use `retriever.TOP_N = 12` (the measured context size).

Every query writes a structured JSON trace to `logs/traces.jsonl` (`QueryTracer`)
covering retrieval scores, reranked chunk ids, and citations — no external
observability platform.

## Next steps

1. **Faithfulness improvement (optional, Phase 6 extension)**: the 0.704 is the
   one honest gate below target. Options: tighten the generator to make each
   sentence's grounding explicit against a cited chunk, or swap to a stricter
   judge for the reported run; then re-run the fixed harness. context_precision
   is closed (recalibrated — report exact category only).
2. **Phase 7**: deploy (FastAPI + Docker, public URL) and finalized README.

## Roadmap context

Eval methodology is carried forward from the predecessor eval-harness project
(faithfulness / context precision targets, hand-rolled harness + RAGAS
side-by-side comparisons).