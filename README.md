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
│   → top-5 (context)          │
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
retrieval/    hybrid retrieval + cross-encoder reranker (done — RRF k=60, top-20 → top-5)
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
| 4. Hybrid retrieval + reranking | **Done** — RRF k=60 (top-20), cross-encoder rerank → top-5; structured JSON traces per query |
| 5. Generation with citations + minimal eval | **Done** — grounded generation (Claude Haiku) + 10-item eval harness wired in (see Evaluation) |
| 6. Full eval layer | **Partial** — deterministic proxies pass; retrieval recall + citation specificity are the open gaps (see Evaluation) |
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
  (`ms-marco-MiniLM-L-6-v2`) reranks to top-5.

## Evaluation

A 10-item golden dataset is validated (`eval/golden_dataset.json`: 3 exact-retrieval /
3 synthesis / 2 cross-document / 2 negative, every citation checked against the
corpus). The hand-rolled harness (`eval/run_eval.py`) runs each item through the
full pipeline (retrieval → grounded generation) and writes results to
`eval/results/{timestamp}.json`.

Published numbers (Phase 5 minimal pass, 10 items, deterministic proxies —
no RAGAS yet):

| Metric | Value | Target | Status |
|---|---|---|---|
| Grounded generation (faithfulness proxy) | 1.0 | ≥ 0.8 | **PASS** — every generated citation is backed by a retrieved chunk |
| Negative "not covered" behavior | 2/2 | all | **PASS** |
| Retrieval precision@3 | 0.625 | high | **below** — expected chunk missing from top-3 on ~4 items |
| Citation accuracy | 0.229 | ≥ 0.9 | **below** — see gap analysis |

**Gap analysis (honest):** citation accuracy is low for two distinct reasons —
(1) **retrieval recall**: for cross-document / specific queries (e.g. GDPR
Art. 32, ASVS V10.1.x, NIST AC-2(3)) the expected chunk isn't retrieved at all,
so the generator can't cite it; (2) **citation specificity**: even when the
right sub-control chunk is retrieved, Claude tends to cite the parent control
(`AC-2`, `AC-6`) rather than the specific enhancement (`AC-2(3)`, `AC-6(10)`)
the golden dataset expects. Both are retrieval/grounding-tuning work for Phase 6 —
generation itself is grounded at 1.0.

Phase 6 will scale to 50-100 items, add RAGAS side-by-side comparison
(faithfulness ≥ 0.8, context_precision ≥ 0.7), and address the retrieval-recall
and citation-specificity gaps above.

## API

FastAPI service (`api/app.py`); run with `uvicorn api.app:app` (needs Qdrant up
and `VOYAGE_API_KEY` + `ANTHROPIC_API_KEY` in `.env`):

```
GET  /health   -> service + Qdrant health
POST /query    -> {"query": "...", "top_n": 5, "rerank": true}
               -> {answer, citations, retrieved, generation, collection}
```

Every query writes a structured JSON trace to `logs/traces.jsonl` (`QueryTracer`)
covering retrieval scores, reranked chunk ids, and citations — no external
observability platform.

## Next steps

1. Phase 6: full eval (50-100 golden items) + RAGAS side-by-side; fix the two
   gaps the Phase 5 eval surfaced — retrieval recall (missing V10.1.x / Art. 32 /
   AC-2(3)) and citation specificity (parent vs sub-control)
2. Phase 7: deploy (FastAPI + Docker, public URL) and finalized README

## Roadmap context

Eval methodology is carried forward from the predecessor eval-harness project
(faithfulness / context precision targets, hand-rolled harness + RAGAS
side-by-side comparisons).