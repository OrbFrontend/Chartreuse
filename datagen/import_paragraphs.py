"""Import the graded Haiku 5.5 paragraph set into a paragraph axis's interim/.

Source: PURPLE_PARAGRAPHS_SRC (default ../OrbRewriter/data/external/purple_paragraphs), a
read-only copy of Orb's purple_paragraphs: 13,064 paragraphs graded 0-3 for paragraph-level
purple (label = score >= 2, majority of blind Haiku audits). Its train/val/test split is by
group_id and is KEPT as a `split` field -- the test split is OrbRewriter's held-out modern_slop
eval, so it must stay out of training here too.

Cleaning (counts printed):
  - non-English rows (~690, nearly all label 0): a free "not English -> clean" shortcut;
  - refusals / out-of-character stops labelled 0 ("I won't continue this scene..."): not prose;
  - blank lines inside a row (87 synthetic rows that set dialogue off with "\\n\\n") collapse to a
    single newline, so the row stays one paragraph under text_segmentation.split_paragraphs.

    DEPURPLE_AXIS=purple_para python -m datagen.import_paragraphs
    DEPURPLE_AXIS=purple_para python -m datagen.import_paragraphs --demo
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter

from core.paths import INTERIM, PARAGRAPHS_SRC, UNIT

OUT = INTERIM / "haiku_paragraphs.jsonl"
SPLITS = ("train", "val", "test")
REFUSAL = re.compile(r"\b(I won't (continue|write)|I'm going to (stop|step out|pause)|I (can't|cannot) "
                     r"(help|write|continue)|as an AI\b)", re.I)


def english(text: str, floor: float = 0.9) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > floor


def one_paragraph(text: str) -> str:
    return re.sub(r"\n\s*\n+", "\n", text.strip())


def verify_checksums() -> None:
    sums = PARAGRAPHS_SRC / "SHA256SUMS"
    if not sums.exists():
        print(f"  note: no SHA256SUMS in {PARAGRAPHS_SRC}; skipping the drift check")
        return
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name.split("/")[-1].removesuffix(".jsonl") in SPLITS:
            got = hashlib.sha256((PARAGRAPHS_SRC / name).read_bytes()).hexdigest()
            if got != digest:
                raise SystemExit(f"FATAL: {name} does not match SHA256SUMS (source drifted)")


def convert(r: dict, split: str) -> dict:
    return {"text": one_paragraph(r["text"]), "label": int(r["label"]),
            "source": r["source"],                       # haiku_reply / haiku_synthetic / haiku_native / appdb_*
            "group_id": "pp_" + r["group_id"], "is_real": r["generator"] == "appdb",
            "unit": "paragraph", "split": split, "variant": r["variant"], "score": r["score"],
            "audit_agree": bool(r["audit_agree"])}


def clean(rows: list[dict]) -> tuple[list[dict], Counter]:
    kept, dropped = [], Counter()
    for r in rows:
        if not english(r["text"]):
            dropped[f"non-English (label {r['label']})"] += 1
        elif r["label"] == 0 and REFUSAL.search(r["text"]):
            dropped["refusal"] += 1
        else:
            if "\n\n" in r["text"]:
                dropped["blank lines collapsed (kept)"] += 1
            kept.append(r)
    return kept, dropped


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if UNIT != "paragraph":
        raise SystemExit("FATAL: import_paragraphs is for a paragraph axis (DEPURPLE_AXIS=purple_para)")
    verify_checksums()
    rows = []
    for split in SPLITS:
        src = [json.loads(l) for l in (PARAGRAPHS_SRC / f"{split}.jsonl").open(encoding="utf-8") if l.strip()]
        kept, dropped = clean(src)
        rows += [convert(r, split) for r in kept]
        print(f"  {split:<5} {len(src):>6} -> {len(kept):>6}   {dict(dropped)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    pos = sum(r["label"] for r in rows)
    print(f"-> {OUT}: {len(rows)} paragraphs ({pos} purple / {len(rows) - pos} not)")


def demo():
    assert english("She sets the glass down.") and not english("Nếu bạn muốn, mình có thể viết một câu chuyện khác")
    assert one_paragraph('He waits.\n\n"Go," she says.') == 'He waits.\n"Go," she says.'
    rows = [{"text": "I won't continue this scene. Kane is a child.", "label": 0},
            {"text": "Mình có thể viết một câu chuyện khác với các nhân vật.", "label": 0},
            {"text": "A pause.\n\nNothing more.", "label": 1},
            {"text": "I won't continue, she said, and left.", "label": 1}]   # label 1: kept, it is prose
    kept, dropped = clean(rows)
    assert [r["text"][:8] for r in kept] == ["A pause.", "I won't "], kept
    assert dropped["refusal"] == 1 and dropped["blank lines collapsed (kept)"] == 1
    r = convert({"text": "x\n\ny", "label": 1, "source": "haiku_reply", "group_id": "c_ab", "generator": GEN_HAIKU,
                 "variant": "reply", "score": 2, "audit_agree": True}, "test")
    assert r["group_id"] == "pp_c_ab" and r["split"] == "test" and not r["is_real"] and r["text"] == "x\ny"
    print("demo ok")


GEN_HAIKU = "claude-haiku-5-5"

if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
