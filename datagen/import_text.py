"""Import trusted human-written prose (stories, blog posts, essays) as label-0 curated rows.

Drop one document per file in data/raw/docs/*.txt, then run this. It sentence-splits
each document with syntok, labels every sentence 0 (real human prose, assumed not
purple), and appends the new rows to data/curated.jsonl. Each document keeps its own
group_id so all its sentences land in the same train/val split (no leakage).

    python -m datagen.import_text            # import data/raw/docs/*.txt
    python -m datagen.import_text --demo      # no-file self-check

Re-running is safe: rows whose text is already in curated.jsonl are skipped.
Labels are blanket 0 by assumption; if a document has a genuinely purple sentence,
fix that one row in data/curated.jsonl by hand. Then run build_dataset.py.
"""
import argparse
import json
import re
from pathlib import Path

import syntok.segmenter as segmenter

from core.paths import DOCS_DIR, CURATED

_LETTERS = re.compile(r"[A-Za-z]")


def split_sentences(text: str):
    for para in segmenter.process(text):
        for sent in para:
            # syntok normalizes the "n't" clitic to a glued "not" token (spacing=""); a real word
            # " not" always carries a leading space, so restore the contraction (wouldn't, can't, ...).
            parts = [t.spacing + ("n't" if i and t.value == "not" and not t.spacing else t.value)
                     for i, t in enumerate(sent)]
            yield "".join(parts).strip()


def is_sentence(s: str) -> bool:
    """Keep a real single sentence; drop blanks, headers, and symbol/number junk."""
    if not s or "\n" in s or "\t" in s:
        return False
    if not (15 <= len(s) <= 400):  # ponytail: loose bounds — real prose varies; tighten if noisy
        return False
    if s.isupper():  # chapter headers / shouted titles
        return False
    return len(_LETTERS.findall(s)) / len(s) >= 0.5


def load_existing_text(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            seen.add(json.loads(line)["text"].strip().lower())
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DOCS_DIR), help="folder of *.txt documents")
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("*.txt"))
    if not files:
        print(f"no *.txt documents in {args.dir} — add files there first")
        return

    seen = load_existing_text(CURATED)
    rows = []
    for f in files:
        gid = f"doc_{f.stem}"
        kept = 0
        for sent in split_sentences(f.read_text(encoding="utf-8")):
            if not is_sentence(sent) or sent.strip().lower() in seen:
                continue
            seen.add(sent.strip().lower())
            rows.append({"text": sent, "label": 0, "source": "curated_text",
                         "group_id": gid, "is_real": True})
            kept += 1
        print(f"  {f.name}: +{kept} sentences")

    if rows:
        with CURATED.open("a", encoding="utf-8") as out:
            for r in rows:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"appended {len(rows)} label-0 rows -> {CURATED}")
    print("now run: python -m datagen.build_dataset")


def demo():
    """Self-check sentence splitting and filtering on a tiny story."""
    text = 'The rain fell. "Come inside," she said softly. CHAPTER TWO\nok'
    sents = list(split_sentences(text))
    assert "The rain fell." in sents
    assert is_sentence("She walked to the old wooden door and knocked twice.")
    assert not is_sentence("CHAPTER TWO")
    assert not is_sentence("hi.")
    print("demo ok")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
    else:
        main()
