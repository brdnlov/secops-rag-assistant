"""Parse GDPR (EU 2016/679) EUR-Lex XHTML -> per-article normalized chunks.

Each article (99 total) is one chunk. Recitals/preamble are skipped.
Heading path tracks Chapter and (where applicable) Section context.

Emits a JSON array matching the Layer 1 schema:
  {id, source, heading_path, text, chunk_content_hash, metadata}
"""

import hashlib
import json
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

RAW = Path(__file__).resolve().parents[1] / "raw" / "gdpr_regulation_2016_679.html"
OUT = Path(__file__).resolve().parents[1] / "processed" / "gdpr_chunks.json"

SOURCE = "gdpr"
WS = re.compile(r"\s+")


def clean(text: str) -> str:
    return WS.sub(" ", text).strip()


def classes(el):
    """Normalize an element's class attribute to a list of tokens."""
    c = el.get("class") or ""
    return c.split() if isinstance(c, str) else list(c)


def main():
    soup = BeautifulSoup(RAW.read_text(encoding="utf-8"), "xml")

    chapter = None      # label like "CHAPTER I" + title
    section = None      # label like "Section 1" + title
    chunks = []

    for p in soup.find_all("p"):
        cls = classes(p)

        if "oj-ti-section-1" in cls:
            label = clean(p.get_text())
            if label.upper().startswith("CHAPTER"):
                section = None
                chapter = (label, None)
            else:
                section = (label, None)
            continue
        if "oj-ti-section-2" in cls:
            title = clean(p.get_text())
            if chapter and chapter[1] is None:
                chapter = (chapter[0], title)
            elif section and section[1] is None:
                section = (section[0], title)
            continue
        if "oj-ti-art" not in cls:
            continue

        art_div = p.parent
        if "eli-subdivision" not in classes(art_div):
            continue

        art_no = clean(p.get_text()).replace("Article", "").strip()
        title_el = art_div.find("div", class_="eli-title")
        title = clean(title_el.get_text()) if title_el else ""

        body_parts = []
        for child in art_div.children:
            if getattr(child, "name", None) is None:
                continue
            if child is p or "eli-title" in classes(child):
                continue
            t = clean(child.get_text(" "))
            if t:
                body_parts.append(t)
        text = "\n\n".join(body_parts)

        crumbs = []
        if chapter:
            ch_label = chapter[0]
            if chapter[1]:
                ch_label = f"{chapter[0]}: {chapter[1]}"
            crumbs.append(ch_label)
        if chapter and section:
            sec_label = section[0]
            if section[1]:
                sec_label = f"{section[0]}: {section[1]}"
            crumbs.append(sec_label)
        crumbs.append(f"Article {art_no}: {title}" if title else f"Article {art_no}")
        heading_path = " > ".join(crumbs)

        metadata = {
            "article_number": int(art_no),
            "title": title,
            "chapter": f"{chapter[0]}: {chapter[1]}" if chapter and chapter[1] else chapter[0] if chapter else None,
        }
        if section and section[1]:
            metadata["section"] = f"{section[0]}: {section[1]}"

        chunks.append(
            {
                "id": f"gdpr_art_{art_no}",
                "source": SOURCE,
                "heading_path": heading_path,
                "text": text,
                "chunk_content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata": metadata,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks -> {OUT}")


if __name__ == "__main__":
    main()