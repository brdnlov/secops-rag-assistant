"""Parse OWASP ASVS v5.0.0 (CSV) -> Level 1-2 requirements -> normalized chunks.

Each numbered requirement (req_id) is one chunk. Preserves the parent
verification section row (chapter_id/section_id) as metadata.

Emits a JSON array matching the Layer 1 schema:
  {id, source, heading_path, text, chunk_content_hash, metadata}
"""

import csv
import hashlib
import json
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "raw" / "owasp_asvs_v5.0.0.csv"
OUT = Path(__file__).resolve().parents[1] / "processed" / "owasp_asvs_chunks.json"

SOURCE = "owasp_asvs"
LEVELS = {"1", "2"}


def main():
    chunks = []
    with RAW.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["L"].strip() not in LEVELS:
                continue
            text = row["req_description"].strip()
            heading_path = f"{row['chapter_name']} > {row['section_name']}"
            chunks.append(
                {
                    "id": f"asvs_{row['req_id'].lower().replace('.', '_')}",
                    "source": SOURCE,
                    "heading_path": heading_path,
                    "text": text,
                    "chunk_content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "metadata": {
                        "req_id": row["req_id"],
                        "chapter_id": row["chapter_id"],
                        "chapter_name": row["chapter_name"],
                        "section_id": row["section_id"],
                        "section_name": row["section_name"],
                        "level": int(row["L"]),
                    },
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks -> {OUT}")


if __name__ == "__main__":
    main()