"""Phase 2 — heading/section-aware chunker.

Turns Phase 1 units (corpus/processed/*_chunks.json) into the final chunk
corpus, applying the AGENTS.md chunking spec:

  - units <= max_chunk_size (1024 tokens) stay whole ("natural unit" rule:
    GDPR articles fit in one chunk, NIST controls/enhancements are atomic,
    ASVS requirements are atomic)
  - units > max_chunk_size are split with a recursive character splitter
    targeting chunk_size (512 tokens) with zero overlap, walking the
    separator hierarchy "\n\n" -> "\n" -> ". " (space as last-resort)
  - pieces below min_chunk_size (128 tokens) merge into the previous chunk
  - split chunks get zero-padded ids ("gdpr_art_4" -> "gdpr_art_4_01")
  - every chunk keeps the Layer 1 schema (id, source, heading_path, text,
    chunk_content_hash) plus the base unit's metadata; split chunks record
    part_of / part_index linkage

Hand-rolled (no langchain) so the mechanics stay visible, per project
convention. Token counting uses tiktoken's cl100k_base as the proxy for the
512-token target.

Writes corpus/chunked/{source}_chunks.json, re-merges corpus/corpus.json,
and updates corpus/corpus_manifest.json with chunking_config and
chunking_config_hash.

Usage: python -m chunking.chunker
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "corpus" / "processed"
CHUNKED = ROOT / "corpus" / "chunked"
CORPUS_JSON = ROOT / "corpus" / "corpus.json"
MANIFEST_JSON = ROOT / "corpus" / "corpus_manifest.json"

# Reuse the Layer 1 validation gate from the Phase 1 merge script.
sys.path.insert(0, str(ROOT / "corpus" / "scripts"))
from build_corpus import validate  # noqa: E402

CORPUS_VERSION = "0.2.0"

CHUNKING_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 0,
    "max_chunk_size": 1024,
    "min_chunk_size": 128,
    "separators": ["\n\n", "\n", ". "],
    "tokenizer": "cl100k_base",
    "id_suffix_format": "{:02d}",
}


def config_hash(config=None):
    """SHA-256 of the canonicalized chunker config — detects config drift."""
    cfg = config or CHUNKING_CONFIG
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Chunker:
    def __init__(self, config=None, encoding=None):
        self.config = {**CHUNKING_CONFIG, **(config or {})}
        self.encoding = encoding or tiktoken.get_encoding(self.config["tokenizer"])

    def count(self, text):
        return len(self.encoding.encode(text))

    def _greedy(self, texts, sep, target):
        """Greedily merge split pieces up to `target` tokens, rejoining with sep."""
        pieces = []
        buf, buf_len = [], 0
        sep_len = self.count(sep) if sep else 0
        for t in texts:
            t = t.strip()
            if not t:
                continue
            t_len = self.count(t)
            sep_cost = sep_len if buf else 0
            if buf and buf_len + sep_cost + t_len > target:
                pieces.append(sep.join(buf).strip())
                buf, buf_len = [t], t_len
            else:
                if buf:
                    buf_len += sep_len
                buf.append(t)
                buf_len += t_len
        if buf:
            pieces.append(sep.join(buf).strip())
        return [p for p in pieces if p]

    def _space_split(self, text, target):
        """Last resort: cut on words when no finer separator exists."""
        words = text.split(" ")
        return self._greedy(words, " ", target)

    def _recursive(self, text, seps):
        """Split text into pieces each <= max_chunk_size via separator hierarchy."""
        cfg = self.config
        max_c = cfg["max_chunk_size"]
        target = cfg["chunk_size"]

        sep = next((s for s in seps if s and s in text), None)
        if sep is not None:
            pieces = self._greedy(text.split(sep), sep, target)
            rest = seps[seps.index(sep) + 1:]
        else:
            pieces, rest = [text], []

        out = []
        for p in pieces:
            if self.count(p) <= max_c:
                out.append(p)
            elif rest:
                out.extend(self._recursive(p, rest))
            else:
                out.extend(self._space_split(p, target))
        return out

    def _min_merge(self, texts):
        """Merge sub-min chunks into the previous chunk (respecting the ceiling)."""
        cfg = self.config
        mn, mx = cfg["min_chunk_size"], cfg["max_chunk_size"]
        out = []
        for t in texts:
            if out and self.count(t) < mn:
                merged = out[-1] + "\n\n" + t
                if self.count(merged) <= mx:
                    out[-1] = merged
                    continue
            out.append(t)
        return out

    def chunk_unit(self, unit):
        """Chunk a single Phase 1 unit. Returns one or more final chunks."""
        cfg = self.config
        if self.count(unit["text"]) <= cfg["max_chunk_size"]:
            return [unit]

        pieces = self._min_merge(self._recursive(unit["text"], list(cfg["separators"])))
        fmt = cfg["id_suffix_format"]
        out = []
        for i, text in enumerate(pieces):
            meta = dict(unit.get("metadata") or {})
            meta["part_of"] = unit["id"]
            meta["part_index"] = i
            out.append(
                {
                    "id": f"{unit['id']}_{fmt.format(i + 1)}",
                    "source": unit["source"],
                    "heading_path": unit["heading_path"],
                    "text": text,
                    "chunk_content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "metadata": meta,
                }
            )
        return out


def main():
    units = []
    for f in sorted(PROCESSED.glob("*_chunks.json")):
        units.extend(json.loads(f.read_text(encoding="utf-8")))
    print(f"Loaded {len(units)} Phase 1 units from {PROCESSED.name}/")

    chunker = Chunker()
    grouped = defaultdict(list)
    split_log = []
    for unit in sorted(units, key=lambda u: u["id"]):
        out = chunker.chunk_unit(unit)
        for c in out:
            grouped[c["source"]].append(c)
        if len(out) > 1:
            split_log.append((unit["id"], len(out), [c["id"] for c in out]))

    CHUNKED.mkdir(exist_ok=True)
    for src in sorted(grouped):
        (CHUNKED / f"{src}_chunks.json").write_text(
            json.dumps(grouped[src], indent=2, ensure_ascii=False), encoding="utf-8"
        )

    merged = [c for src in sorted(grouped) for c in grouped[src]]

    errors = validate(merged)
    if errors:
        print(f"L1 VALIDATION FAILED ({len(errors)} errors):")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)

    counts = Counter(c["source"] for c in merged)
    processed_hashes = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(PROCESSED.glob("*_chunks.json"))
    }
    chunked_hashes = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(CHUNKED.glob("*_chunks.json"))
    }
    digest_input = "".join(
        f"{c['id']}:{c['chunk_content_hash']}" for c in sorted(merged, key=lambda c: c["id"])
    )
    corpus_content_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "corpus_content_hash": corpus_content_hash,
        "generated_by": "chunking/chunker.py",
        "num_chunks": len(merged),
        "chunks_by_source": dict(counts),
        "source_file_hashes": processed_hashes,
        "chunked_file_hashes": chunked_hashes,
        "chunking_config": CHUNKING_CONFIG,
        "chunking_config_hash": config_hash(),
        "embedding_model_version": None,
    }

    CORPUS_JSON.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"L1 validation passed: {len(merged)} chunks")
    print(f"chunking_config_hash: {manifest['chunking_config_hash'][:16]}...")
    print("By source:", dict(counts))
    if split_log:
        print(f"Split units ({len(split_log)}):")
        for unit_id, n, ids in split_log:
            print(f"  {unit_id}: {n} chunks -> {ids}")
    else:
        print("Split units: none")
    print(f"corpus_content_hash: {corpus_content_hash[:16]}...")
    print(f"corpus_version -> {CORPUS_VERSION}")


if __name__ == "__main__":
    main()