# Session 2 — Phase 1: Corpus Acquisition & Cleaning (Complete)

**Date:** 2026-08-27
**Duration:** ~1 session
**Status:** Phase 1 COMPLETE — 376 chunks validated, ready for Phase 2 (chunking)

## What happened

Executed the full Phase 1 plan. Downloaded all three source corpora, wrote three
source-specific parsers plus a merge/manifest/validation script, and shipped
`corpus/corpus.json` (376 chunks) that passes Layer 1 schema validation.

## Deliverables

```
corpus/
├── raw/    nist_sp800_53_rev5_catalog.json (10.4MB OSCAL JSON)
│           owasp_asvs_v5.0.0.csv            (105KB, 345 reqs)
│           gdpr_regulation_2016_679.html    (809KB EUR-Lex XHTML)
├── processed/  nist_sp800_53_chunks.json (24)   owasp_asvs_chunks.json (253)
│               gdpr_chunks.json (99)
├── scripts/    parse_nist.py  parse_asvs.py  parse_gdpr.py  build_corpus.py
├── corpus.json                # 376 chunks, merged, validated
└── corpus_manifest.json       # version 0.1.0, content + source hashes
```

## Parser details / structural findings

- **NIST (OSCAL JSON):** `catalog.groups[].controls[]`; AC-2 has 13 enhancements,
  AC-6 has 10. Statements can be flat prose (AC-6) or nested item lists (AC-2).
  Resolve `{{ insert: param, X }}` placeholders to `[param label]`. **AC-2(10)
  "Shared and Group Account Credential Change" is `status: withdrawn`** (content
  merged into AC-2 base) — skipped, so NIST = 24 chunks not 25.
- **ASVS (CSV):** 345 reqs total, `L` column (70× L1, 183× L2, 92× L3). Kept L1+L2
  = 253.
- **GDPR (EUR-Lex XHTML):** exactly 99 `<p class="oj-ti-art">` article headings
  inside `<div class="eli-subdivision">`; chapters (`oj-ti-section-1`) and section
  titles (`oj-ti-section-2`) tracked in document order for heading paths. Recitals
  and preamble excluded. All 99 articles present 1-99, chapter assignments verified.

## Validation (Layer 1 gate)

`corpus/scripts/build_corpus.py` checks: required fields present, non-empty text,
no duplicate IDs, source ∈ known set, `chunk_content_hash` recomputed. Reusable
`validate()` function is importable for the Layer 1 test suite later.

**Result: PASS — 376 chunks (GDPR 99, NIST 24, ASVS 253)**

## Key decisions this session

- Chunk unit per source matches the plan: NIST control/enhancement, ASVS
  requirement, GDPR article. Phase 2 will split only where > max_chunk_size.
- Chunk id scheme: `nist_ac2`, `nist_ac2_1`, `asvs_v1_1_1`, `gdpr_art_5`
  (stable across phases; the `nist_ac2_001` example in AGENTS.md is illustrative).
- NIST labels: shortest OSCAL label prop wins (AC-2 over AC-02, AC-2(1) over AC-02(01)).
- `corpus_version` = `0.1.0` (repo not yet a git repo; bump when corpus changes).

## Hands-on opportunities for the user (offered this session)

1. Read 5-8 chunks from `corpus/corpus.json` to know the corpus cold (interview prep).
2. Draft a handful of golden eval queries now while the docs are fresh.
3. `git init` + initial commit (steps provided; repo not yet under git).

## Next steps (Phase 2 — chunking)

- RecursiveCharacterTextSplitter, 512 tokens, zero overlap, `\n\n` → `\n` → `. ` hierarchy
- max 1024 / min 128 tokens, split NIST >1024 at enhancement boundaries
- compute `chunking_config_hash`, move metadata onto final chunk ids