"""Parse NIST SP 800-53 Rev 5 (OSCAL JSON) -> AC-2 / AC-6 subset -> normalized chunks.

Chunk units:
  - Each base control (AC-2, AC-6) = one chunk (statement + guidance)
  - Each enhancement (AC-2(1) .. AC-2(13), AC-6(1) .. AC-6(10)) = one chunk

Emits a JSON array matching the Layer 1 schema:
  {id, source, heading_path, text, chunk_content_hash, metadata}
"""

import hashlib
import json
import re
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "raw" / "nist_sp800_53_rev5_catalog.json"
OUT = Path(__file__).resolve().parents[1] / "processed" / "nist_sp800_53_chunks.json"

SOURCE = "nist_sp800_53"
SELECTED = {"ac-2", "ac-6"}

PARAM_INSERT = re.compile(r"\{\{\s*insert:\s*param,\s*([a-z0-9_.\-]+)\s*\}\}")


def resolve_params(text, params):
    """Replace {{ insert: param, X }} placeholders with the param label."""
    def repl(m):
        pid = m.group(1)
        label = params.get(pid, {}).get("label")
        return f"[{label}]" if label else f"[{pid}]"
    return PARAM_INSERT.sub(repl, text)


def collect_prose(part, params):
    """Recursively gather prose text from an OSCAL part (statement/guidance)."""
    out = []
    prose = (part.get("prose") or "").strip()
    if prose:
        out.append(resolve_params(prose, params))
    for sub in part.get("parts", []) or []:
        out.append(collect_prose(sub, params))
    return "\n".join(p for p in out if p).strip()


def is_withdrawn(ctl):
    """Some enhancements are marked withdrawn (content merged into the base control)."""
    return any(
        p.get("name") == "status" and p.get("value") == "withdrawn"
        for p in ctl.get("props", []) or []
    )


def build_chunk(oid, ctl, params, heading_path, family):
    """Build a normalized chunk for one control/enhancement."""
    parts = ctl.get("parts", []) or []
    sections = []
    for part in parts:
        if part.get("name") in ("statement", "guidance"):
            text = collect_prose(part, params)
            if text:
                sections.append(text)
    text = "\n\n".join(sections).strip()

    metadata = {
        "control_id": ctl["id"].upper(),
        "title": ctl.get("title", ""),
        "family": family,
    }
    if "." in oid:
        metadata["parent_control_id"] = oid.split(".", 1)[0].upper()

    block = {
        "id": f"nist_{oid.replace('-', '').replace('.', '_')}",
        "source": SOURCE,
        "heading_path": heading_path,
        "text": text,
        "chunk_content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "metadata": metadata,
    }
    return block


def main():
    catalog = json.loads(RAW.read_text(encoding="utf-8"))
    chunks = []
    for group in catalog["catalog"]["groups"]:
        family = group.get("title", "")
        for ctl in group.get("controls", []) or []:
            cid = ctl["id"]
            if cid not in SELECTED:
                continue

            # params live on the control; enhancements inherit only their own params
            params = {p.get("id"): p for p in ctl.get("params", []) or [] if p.get("id")}
            os_label = os_label_from(ctl)
            heading = f"{family} > {os_label}: {ctl.get('title', '')}"

            chunks.append(build_chunk(cid, ctl, params, heading, family))

            for sub in ctl.get("controls", []) or []:
                if is_withdrawn(sub):
                    print(f"  skipping withdrawn enhancement {sub['id']}")
                    continue
                sub_params = {p.get("id"): p for p in sub.get("params", []) or [] if p.get("id")}
                sub_oid = sub["id"]
                sub_label = os_label_from(sub)
                sub_heading = f"{family} > {os_label}: {ctl.get('title', '')} > {sub_label}: {sub.get('title', '')}"
                chunks.append(build_chunk(sub_oid, sub, sub_params, sub_heading, family))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks -> {OUT}")


def os_label_from(ctl):
    """OSCAL label prop preferring the shortest form (AC-2 over AC-02, AC-2(1) over AC-02(01))."""
    labels = [p.get("value", "") for p in ctl.get("props", []) or [] if p.get("name") == "label"]
    if labels:
        return min(labels, key=len)
    return ctl["id"].upper()


if __name__ == "__main__":
    main()