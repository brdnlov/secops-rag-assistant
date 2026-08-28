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
chunking/     heading-aware chunker (planned)
embeddings/   Voyage AI voyage-4, content-hash checkpointing (planned)
retrieval/    Qdrant hybrid search + RRF + cross-encoder rerank (planned)
eval/         golden dataset + hand-rolled harness (dataset done, harness planned)
api/          FastAPI RAG endpoint (planned)
logs/         structured JSON traces per query (planned)
```

## Current status (phases, in order)

| Phase | Status |
|---|---|
| 1. Corpus acquisition & cleaning | **Done** — 376 chunks validated |
| 2. Chunking | Next |
| 3. Embedding + Qdrant | Planned |
| 4. Hybrid retrieval + reranking | Planned |
| 5. Generation with citations + minimal eval | Planned |
| 6. Full eval layer | Planned |
| 7. Deploy + README finalized | Planned |

## Corpus

Three sources, one normalized schema `{id, source, heading_path, text,
chunk_content_hash, metadata}`:

| Source | Scope | Chunks |
|---|---|---|
| NIST SP 800-53 Rev 5 | AC-2 (Account Management), AC-6 (Least Privilege), incl. enhancements | 24 |
| OWASP ASVS v5.0.0 | Levels 1-2 (345-source footprint, kept 253 requirements) | 253 |
| GDPR (Reg. 2016/679) | All 99 articles (recitals excluded) | 99 |

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
- **Source-aware splitting** — NIST controls split at numbered-requirement
  boundaries; GDPR articles stay whole where under the hard 1024-token ceiling.
- **RRF fusion (k=60)** over BM25 + dense, then a local cross-encoder
  (`ms-marco-MiniLM-L-6-v2`) reranks to top-5.

## Evaluation

A 10-item golden dataset is drafted and validated (`eval/golden_dataset.json`, 3
exact-retrieval / 3 synthesis / 2 cross-document / 2 negative, every citation
checked against the corpus). Harness and published numbers are **pending** — no
metric is reported until the eval layer actually produces it.

## Next steps

1. Phase 2: heading-aware chunking + manifest config hash
2. Phase 3: Voyage AI voyage-4 embedding + self-hosted Qdrant
3. Phase 4: hybrid retrieval + reranking
4. Phase 5: generation with citations + the 10-item eval wired in
5. Phase 6: full eval (50-100 golden items, faithful ≥ 0.8, context precision ≥ 0.7)
6. Phase 7: deploy (FastAPI + Docker) and finalized README

## Roadmap context

Eval methodology is carried forward from the predecessor eval-harness project
(faithfulness / context precision targets, hand-rolled harness + RAGAS
side-by-side comparisons).