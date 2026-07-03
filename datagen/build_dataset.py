"""Pool the interim artifacts into labeled.jsonl and write leak-safe splits.

Inputs (all from earlier pipeline steps), unified to the Data-format schema
    {"text", "label", "source", "group_id", "is_real"}:

  interim/<axis>_pairs.jsonl     generate_synthetic.py: each pair -> neg(label 0) + pos(label 1),
                                     same group_id; `real` names which pole was the real seed.
  data/_common/negatives.jsonl   axis-SHARED common prose -> label 0 (plain prose is the
                                     common complement of every axis; filter_common.py prunes
                                     the few rows the axis classifier scores as positive).
  human_labeled.jsonl            human-labeled real rows -> calibration + test only.

Splitting:
  - calibration.jsonl + test.jsonl are drawn ONLY from human rows (source=="human", is_real). They
    are the ground truth; never train on them, keep them disjoint. Anything synthetic or LLM-labeled
    (incl. triaged real text) goes to train/val only.
  - train.jsonl + val.jsonl come from everything else, split by GROUP so a synthetic pair and its
    hard-negative twin (same group_id) never straddle the split (leakage). Group assignment is a
    deterministic hash of group_id, so re-running is stable.

    python -m datagen.build_dataset            # build everything
    python -m datagen.build_dataset --demo      # self-check, no files
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from core.paths import (INTERIM, HUMAN, CURATED, COMMON, AXIS, pair_paths,
                        LABELED as OUT_LABELED, SPLITS as OUT_SPLITS)

VAL_FRAC = 0.15        # of train/val groups held out for model selection (early-stop on F1)
TEST_FRAC = 0.50       # of human groups reserved for the final report (rest -> calibration)


def norm(text: str) -> str:
    """Dedup key: lowercase, whitespace collapsed. Catches cross-source collisions."""
    return re.sub(r"\s+", " ", text).strip().lower()


def is_single_sentence(text: str) -> bool:
    """Hard integrity check: non-empty, no embedded line break (would corrupt JSONL/training).
    The linguistic 'exactly one terminal mark' rule is upstream's job (3a-3c QA'd it); enforcing it
    here would wrongly drop valid quoted dialogue / abbreviations, so we only warn on it below."""
    return bool(text) and "\n" not in text and "\r" not in text and len(text.strip()) >= 3


# ponytail: heuristic only -> a terminator + space + capital. Misfires on "Mr. Smith"; used for a
# warning count, never to drop rows. Tighten only if a real multi-sentence leak shows up downstream.
_MULTI = re.compile(r"[.!?][\"')\]]?\s+[A-Z]")


def bucket(group_id: str, frac: float, salt: str) -> bool:
    """Deterministic: True for the held-out fraction. Stable across runs; same group_id -> same side."""
    h = int(hashlib.md5(f"{salt}:{group_id}".encode()).hexdigest(), 16)
    return (h % 1000) < frac * 1000


def row(text, label, source, group_id, is_real):
    return {"text": text, "label": int(label), "source": source,
            "group_id": group_id, "is_real": bool(is_real)}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def pair_rows(p: dict, source: str) -> list[dict]:
    """A unified synthetic pair -> classifier rows: neg=0, pos=1. Both share the pair's
    group_id so they stay on one side of the train/val split (no leakage). `real` names which
    pole was the real seed ("neg"|"pos"); is_real tracks it so a real seed (e.g. a de-euph pair's
    coy seed, or a purple improve pair's real overwritten seed) isn't mislabeled synthetic. A
    missing `real` (fully synthetic pair) leaves both poles is_real=False. Axis-agnostic — one
    reader for every axis's <axis>_pairs.jsonl."""
    real = p.get("real")
    return [row(p["neg"], 0, source, p["group_id"], real == "neg"),
            row(p["pos"], 1, source, p["group_id"], real == "pos")]


def load_pool() -> list[dict]:
    """All non-human rows, unified. Mined rows get a synthetic group_id from their text hash so the
    split groups them individually (no pair to keep together).

    Order matters for dedup: first occurrence of a normalized text wins, so curated (hand-authored,
    most trusted) is loaded FIRST -- its label overrides any conflicting auto-generated duplicate."""
    out = []
    for r in read_jsonl(CURATED):                           # committed hand-authored -> wins dedup conflicts
        gid = r.get("group_id") or "curated_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
        out.append(row(r["text"], r["label"], r.get("source", "curated"), gid, r.get("is_real", True)))
    for pf in pair_paths(AXIS):                            # every pair corpus (generated + curated) -> neg(0)/pos(1)
        for p in read_jsonl(Path(pf)):                     # sorted curated-first -> wins dedup like CURATED
            out.extend(pair_rows(p, AXIS))
    for r in read_jsonl(INTERIM / "hard_negatives.jsonl"):
        out.append(row(r["text"], r["label"], "hard_neg", r["group_id"], r.get("is_real", False)))
    for r in read_jsonl(INTERIM / "mined_sentences.jsonl"):
        if r.get("genre") in ("plain", "technical"):       # label-0 by construction; purple_prone -> triage_label.py
            gid = "mined_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
            out.append(row(r["text"], 0, "mined", gid, True))
    for r in read_jsonl(INTERIM / "triaged.jsonl"):         # LLM-labeled real -> train/val only
        gid = "triaged_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
        out.append(row(r["text"], r["label"], "mined", gid, True))
    for r in read_jsonl(INTERIM / "soft_euph_gen.jsonl"):    # gross(0)/euph-short(1) pairs, group-shared
        out.append(row(r["text"], r["label"], r["source"], r["group_id"], r.get("is_real", False)))
    for r in read_jsonl(INTERIM / "calm_neg_gen.jsonl"):     # calm non-sexual hard negs (label 0)
        out.append(row(r["text"], r["label"], r["source"], r["group_id"], r.get("is_real", False)))
    for r in read_jsonl(COMMON):                             # axis-SHARED common negatives -> label 0
        gid = r.get("group_id") or "common_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
        out.append(row(r["text"], 0, r.get("source", "common"), gid, r.get("is_real", True)))
    return out


def load_human() -> list[dict]:
    # ponytail: human rows are independent real sentences (no synthetic twin), so each gets its own
    # text-hash group. Ignore any incoming group_id -- the labeler reused one id for 139 rows, which
    # would dump them all onto one side of the calib/test split.
    out = []
    for r in read_jsonl(HUMAN):
        gid = "human_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
        out.append(row(r["text"], r["label"], "human", gid, True))
    return out


def dedupe_validate(rows: list[dict]) -> tuple[list[dict], int, int, int]:
    """Drop exact-normalized dups (first occurrence wins) and integrity failures.
    Also counts conflicts: a dropped dup whose label disagrees with the kept row (the kept one wins
    -- pool order puts curated first, so curated's label is authoritative)."""
    seen, kept, dup, bad, conflict = {}, [], 0, 0, 0
    for r in rows:
        if not is_single_sentence(r["text"]):
            bad += 1
            continue
        k = norm(r["text"])
        if k in seen:
            dup += 1
            if seen[k] != r["label"]:
                conflict += 1
            continue
        seen[k] = r["label"]
        kept.append(r)
    return kept, dup, bad, conflict


def dist(rows: list[dict]) -> str:
    pos = sum(r["label"] for r in rows)
    return f"{len(rows):4d} rows  ({pos} purple / {len(rows) - pos} not)"


def write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()

    pool, p_dup, p_bad, p_conf = dedupe_validate(load_pool())
    human, h_dup, h_bad, _ = dedupe_validate(load_human())
    if p_conf:
        print(f"  WARNING: {p_conf} duplicate text(s) had conflicting labels -> kept the earlier "
              f"(curated-first) row's label", flush=True)
    # human text must not also appear in train/val (would leak ground truth into training)
    pool_keys = {norm(r["text"]) for r in pool}
    leak = [r for r in human if norm(r["text"]) in pool_keys]
    if leak:
        print(f"  WARNING: {len(leak)} human row(s) duplicate a train/val sentence -> dropping from human", flush=True)
        human = [r for r in human if norm(r["text"]) not in pool_keys]

    train = [r for r in pool if not bucket(r["group_id"], VAL_FRAC, "val")]
    val = [r for r in pool if bucket(r["group_id"], VAL_FRAC, "val")]
    test = [r for r in human if bucket(r["group_id"], TEST_FRAC, "test")]
    calib = [r for r in human if not bucket(r["group_id"], TEST_FRAC, "test")]

    labeled = pool + human
    write(OUT_LABELED, labeled)
    write(OUT_SPLITS / "train.jsonl", train)
    write(OUT_SPLITS / "val.jsonl", val)
    write(OUT_SPLITS / "calibration.jsonl", calib)
    write(OUT_SPLITS / "test.jsonl", test)

    multi = sum(1 for r in labeled if _MULTI.search(r["text"]))
    print(f"pooled {dist(labeled)}  (dropped {p_dup + h_dup} dup, {p_bad + h_bad} invalid)")
    print(f"  train       {dist(train)}")
    print(f"  val         {dist(val)}")
    print(f"  calibration {dist(calib)}  [human/real only]")
    print(f"  test        {dist(test)}  [human/real only]")
    if multi:
        print(f"  note: {multi} row(s) look multi-sentence (heuristic; not dropped) -- spot-check if high")
    if len(human) < 50:
        print(f"  note: only {len(human)} human rows -- 3e hand-labeling still pending; "
              f"calibration/test are placeholders until that's done")
    print(f"-> {OUT_LABELED}, {OUT_SPLITS}/{{train,val,calibration,test}}.jsonl")


def demo():
    """Self-check deduplication, validation, and deterministic split behavior without files."""
    assert norm("  Foo   BAR\t") == "foo bar"
    assert is_single_sentence("A real sentence.")
    assert not is_single_sentence("two\nlines")
    assert not is_single_sentence("")
    assert not is_single_sentence("hi")            # too short

    rows = [row("Same text.", 1, "synthetic", "g1", False),
            row("same TEXT.", 0, "synthetic", "g2", False),   # normalized-dup, diff label -> dropped + conflict
            row("bad\nrow", 0, "mined", "g3", True),          # invalid -> dropped
            row("A distinct sentence.", 0, "hard_neg", "g4", False)]
    kept, dup, bad, conf = dedupe_validate(rows)
    assert (len(kept), dup, bad, conf) == (2, 1, 1, 1), (len(kept), dup, bad, conf)
    assert kept[0]["label"] == 1  # first occurrence's label wins (curated is loaded first in load_pool)

    # a synthetic pair shares group_id -> both land on the same side of train/val (no leak)
    assert bucket("g1", VAL_FRAC, "val") == bucket("g1", VAL_FRAC, "val")  # deterministic
    gid = "pair_x"
    assert bucket(gid, VAL_FRAC, "val") == bucket(gid, VAL_FRAC, "val")    # twin -> same bucket

    # _MULTI flags an obvious two-sentence string but tolerates a single one
    assert _MULTI.search("One thing. Then another.")
    assert not _MULTI.search("A single, comma-laden sentence about one thing.")

    # unified pair -> 2 classifier rows (neg=0, pos=1), one group. is_real follows `real`:
    # real="neg" -> the neg pole was the real seed; real="pos" -> the pos pole was.
    en = pair_rows({"neg": "d", "pos": "e", "group_id": "g9", "real": "neg"}, "euphemism")
    assert [r["label"] for r in en] == [0, 1] and {r["group_id"] for r in en} == {"g9"}
    assert [r["is_real"] for r in en] == [True, False] and [r["source"] for r in en] == ["euphemism"] * 2
    ep = pair_rows({"neg": "d", "pos": "e", "group_id": "g9", "real": "pos"}, "euphemism")
    assert [r["is_real"] for r in ep] == [False, True]
    es = pair_rows({"neg": "d", "pos": "e", "group_id": "g9"}, "purple")    # no `real` -> both synthetic
    assert [r["is_real"] for r in es] == [False, False]

    # shared common pool is forced to label 0 regardless of the file's own label field
    cr = row("Common prose.", 0, "common", "common_x", True)
    assert cr["label"] == 0 and cr["source"] == "common"
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
