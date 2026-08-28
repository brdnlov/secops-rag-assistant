"""Merge per-source processed chunks -> corpus/corpus.json + corpus_manifest.json.

Also runs Layer 1 schema validation (the Phase 1 gate):
  - every chunk has id, source, heading_path, text, chunk_content_hash
  - no empty text, no missing heading_path, no duplicate IDs
  - chunk_content_hash recomputed and verified
  - source value is one of the known corpora

Validation is importable so the Layer 1 test layer can reuse it later.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
CORPUS_JSON = ROOT / "corpus.json"
MANIFEST_JSON = ROOT / "corpus_manifest.json"

CORPUS_VERSION = "0.1.0"
KNOWN_SOURCES = {"nist_sp800_53", "owasp_asvs", "gdpr"}

CHUNK_FIELDS = {"id", "source", "heading_path", "text", "chunk_content_hash"}


def load_chunks():
    chunks = []
    for f in sorted(PROCESSED.glob("*_chunks.json")):
        chunks.extend(json.loads(f.read_text(encoding="utf-8")))
    return chunks


def validate(chunks):
    """Layer 1 schema validation. Returns list of error strings (empty = pass)."""
    errors = []
    ids = set()
    for i, c in enumerate(chunks):
        label = f"[chunk #{i}: {c.get('id', '<no-id>')}]"
        missing = CHUNK_FIELDS - set(c.keys())
        if missing:
            errors.append(f"{label} missing fields: {sorted(missing)}")
            continue
        if c["source"] not in KNOWN_SOURCES:
            errors.append(f"{label} unknown source: {c['source']!r}")
        if not c["text"].strip():
            errors.append(f"{label} empty text")
        if not (c["heading_path"] or "").strip():
            errors.append(f"{label} missing heading_path")
        if c["id"] in ids:
            errors.append(f"{label} duplicate id")
        ids.add(c["id"])
        recomputed = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
        if recomputed != c["chunk_content_hash"]:
            errors.append(f"{label} chunk_content_hash mismatch")
    return errors


def main():
    chunks = load_chunks()

    errors = validate(chunks)
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print(f"L1 validation passed: {len(chunks)} chunks")

    counts = Counter(c["source"] for c in chunks)

    # Per-corpus file content hash for manifest
    source_file_hashes = {}
    for f in sorted(PROCESSED.glob("*_chunks.json")):
        source_file_hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()

    # corpus-level content hash: over sorted (id, chunk_content_hash) pairs
    digest_input = "".join(
        f"{c['id']}:{c['chunk_content_hash']}" for c in sorted(chunks, key=lambda c: c["id"])
    )
    corpus_content_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "corpus_content_hash": corpus_content_hash,
        "generated_by": "corpus/scripts/build_corpus.py",
        "num_chunks": len(chunks),
        "chunks_by_source": dict(counts),
        "source_file_hashes": source_file_hashes,
        # set in later phases; kept explicit so the shape is stable
        "chunking_config_hash": None,
        "embedding_model_version": None,
    }

    CORPUS_JSON.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks -> {CORPUS_JSON}")
    print(f"Wrote manifest -> {MANIFEST_JSON}")
    print("By source:", dict(counts))
    print(f"corpus_content_hash: {corpus_content_hash[:16]}...")


if __name__ == "__main__":
    main()