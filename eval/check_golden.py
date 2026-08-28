"""Validate evaluator/self-checker for eval/golden_dataset.json.

Checks:
  - required fields present on every item
  - every expected_citation is a recognised idiom and resolves to an
    existing chunk in corpus/corpus.json
  - dataset size (10 items: 3 exact, 3 synthesis, 2 cross-document, 2 negative)

Usage: python eval/check_golden.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"query", "expected_answer", "expected_citations", "retrieved_chunks", "generated_answer", "generated_citations"}

NIST_ENH = re.compile(r"AC-(\d+)\((\d+)\)$")
NIST_BASE = re.compile(r"AC-(\d+)$")
ASVS = re.compile(r"V\d+(?:\.\d+)+$")


def citation_to_chunk_ids(cit):
    """Map a citation idiom (AC-2(3), Article 17, V10.1.2) to base chunk id."""
    cit = cit.strip()
    m = NIST_ENH.match(cit)
    if m:
        return f"nist_ac{m.group(1)}_{m.group(2)}"
    if NIST_BASE.match(cit):
        return f"nist_ac{NIST_BASE.match(cit).group(1)}"
    if cit.startswith("Article"):
        n = cit.split()[1]
        return f"gdpr_art_{n}"
    if ASVS.match(cit):
        return f"asvs_{cit.lower().replace('.', '_')}"
    return None


def resolve(base_id, ids):
    """Resolve a base chunk id to actual ids — handles post-chunking splits.

    A split unit like gdpr_art_4 becomes gdpr_art_4_01..: the citation is
    satisfied by any of its parts, so we accept a zero-padded suffix match.
    """
    if base_id in ids:
        return [base_id]
    return sorted(i for i in ids if i.startswith(base_id + "_"))


def main():
    data = json.loads((ROOT / "eval" / "golden_dataset.json").read_text(encoding="utf-8"))
    chunks = json.loads((ROOT / "corpus" / "corpus.json").read_text(encoding="utf-8"))
    ids = {c["id"] for c in chunks}
    errors = []

    if len(data) != 10:
        errors.append(f"expected 10 items, got {len(data)}")

    for i, item in enumerate(data):
        label = f"[item {i+1}]"
        missing = REQUIRED - set(item)
        if missing:
            errors.append(f"{label} missing fields {sorted(missing)}")
            continue
        for cit in item["expected_citations"]:
            mapped = citation_to_chunk_ids(cit)
            if not mapped:
                errors.append(f"{label} citation {cit!r} unrecognised idiom")
                continue
            resolved = resolve(mapped, ids)
            if not resolved:
                errors.append(f"{label} citation {cit!r} -> no chunk for {mapped}")

    if errors:
        print(f"FALSE ({len(errors)} errors):")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)

    print("PASS: 10 items, all citations resolve to corpus chunks")
    print(json.dumps(
        {str(i+1): item["expected_citations"] for i, item in enumerate(data)}, indent=2))


if __name__ == "__main__":
    main()