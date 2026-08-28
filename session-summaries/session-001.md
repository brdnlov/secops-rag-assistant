# Session 1 — Project Planning & Phase 1 Design

**Date:** 2026-08-26
**Duration:** ~30 min
**Status:** Planning complete, ready to implement Phase 1

## What happened

Reviewed and expanded the original AGENTS.md (134 → 433 lines) with 10 data-backed recommendations covering chunking strategy, reranker choice, cost ceiling, versioning, eval approach, observability, timeline, testing, and harness interface contract. Researched and locked in all three corpus sources with exact URLs and parsing strategies. Made final scoping decisions for Phase 1.

## Key decisions made

| Decision | Choice | Rationale |
|---|---|---|
| NIST scope | AC-2 + AC-6 only (~15-20 controls) | Connects to CVSS 7.9 auth bypass resume story |
| OWASP ASVS scope | Levels 1-2 (~200-255 requirements) | Standard + high-security apps; manageable for portfolio |
| GDPR scope | Articles only (99 articles) | Recitals are preambles with low retrieval value |
| Chunking | 512 tokens, zero overlap | Ranked #1 in Feb 2026 vendor benchmark; overlap found unnecessary in Jan 2026 arXiv analysis |
| BM25 | Qdrant native server-side (`model: "qdrant/bm25"`) | No separate tokenizer/index to manage |
| Reranker | Local `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80MB, CPU-only, $0 cost, upgrade path to BGE v2-m3 or Voyage rerank-2.5 |
| Embeddings | Voyage AI `voyage-4` ($0.06/1M tokens) | Free tier: 200M tokens; corpus needs ~1-2M |
| Eval judge | Anthropic Claude Haiku 4.5 via existing $5 credit | RAGAS supports Anthropic natively; 10-item eval < $1 |
| Output format | JSON array of normalized chunk objects | Simple, standard, appropriate for ~325-375 chunk corpus |
| Self-hosted infra | Qdrant via Docker | Avoids free-tier quota friction (Gemini 20-req/day cap from prior project) |

## Corpus sources

| Source | Format | URL | Est. Chunks |
|---|---|---|---|
| NIST SP 800-53 Rev 5 | OSCAL JSON | `https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json` | ~25 |
| OWASP ASVS v5.0.0 | CSV/JSON | `https://github.com/OWASP/ASVS/raw/v5.0.0/5.0/docs_en/OWASP_Application_Security_Verification_Standard_5.0.0_en.csv` | ~200-255 |
| GDPR (EU 2016/679) | HTML | `https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679` | ~99 |

**Total estimated corpus:** ~325-375 chunks

## Normalized chunk schema

```json
{
  "id": "nist_ac2_001",
  "source": "nist_sp800_53",
  "heading_path": "AC: Access Control > AC-2: Account Management",
  "text": "The organization defines and documents...",
  "metadata": {
    "control_id": "AC-2",
    "family": "AC",
    "baseline": ["Low", "Moderate", "High"],
    "enhancement": null
  }
}
```

## Directory structure (planned)

```
corpus/
├── raw/                                    # Downloaded source files
│   ├── nist_sp800_53_rev5_catalog.json
│   ├── owasp_asvs_v5.0.0.csv
│   └── gdpr_regulation_2016_679.html
├── processed/                              # Normalized chunk files
│   ├── nist_sp800_53_chunks.json
│   ├── owasp_asvs_chunks.json
│   └── gdpr_chunks.json
├── corpus.json                             # All chunks merged
└── corpus_manifest.json                    # Content hashes, version info
```

## Phase 1 execution plan (next session)

1. Create `corpus/raw/` and `corpus/processed/` directories
2. Download all three source files
3. Write NIST parser: OSCAL JSON → filter AC-2/AC-6 + enhancements → flatten → normalize
4. Write ASVS parser: CSV → filter Level 1-2 → normalize
5. Write GDPR parser: HTML → BeautifulSoup → extract articles → normalize
6. Merge all chunks into `corpus/corpus.json`
7. Compute SHA-256 content hashes → write `corpus/corpus_manifest.json`
8. Validate against Layer 1 schema tests (no empty text, no duplicate IDs, all required fields present)

## What's in AGENTS.md

Full 433-line project plan covering:
- 7 build phases with time estimates (18-27 days total)
- Architecture diagram
- Chunking spec (512 tokens, zero overlap, RecursiveCharacterTextSplitter)
- Embedding + Qdrant setup with content-hash checkpointing
- Hybrid retrieval + reranking design (RRF, local cross-encoder)
- Generation with citations + minimal 10-item eval during Phase 5
- Full eval layer (hand-rolled + RAGAS side-by-side, 50-100 golden items)
- Corpus & chunk versioning scheme
- 5-layer testing strategy
- Structured JSON observability logging
- Eval harness interface contract (golden dataset + results format)

## Context for next session

- Mode changed from plan-only to build — ready to write files and execute code
- Prior project context: eval harness had a cross-document synthesis under-triggering failure mode — explicitly test for this in eval golden dataset
- User has existing $5 Anthropic credit; no other API costs allowed
- All parsing decisions are final — no open questions remain
