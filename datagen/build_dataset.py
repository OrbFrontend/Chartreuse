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

A PARAGRAPH axis (core.paths.UNITS, e.g. purple_para) takes build_paragraph() instead: its own
paragraph rows keep their source split, its sentence axis's rows join as one-sentence paragraphs,
and every row carries `unit`. See data/purple_para/README.md.

    python -m datagen.build_dataset            # build everything
    python -m datagen.build_dataset --demo      # self-check, no files
"""
import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from core.paths import (DATA, INTERIM, HUMAN, CURATED, COMMON, AXIS, UNIT, SENTENCE_MIX, pair_paths,
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


def is_paragraph(text: str) -> bool:
    """Paragraph-unit integrity: non-empty, no BLANK line (that would be two paragraphs under
    text_segmentation.split_paragraphs). A single newline inside an RP paragraph is legitimate."""
    return bool(text) and "\r" not in text and not re.search(r"\n\s*\n", text) and len(text.strip()) >= 3


def valid(r: dict) -> bool:
    return is_paragraph(r["text"]) if r.get("unit") == "paragraph" else is_single_sentence(r["text"])


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


def load_pool(data_dir: Path = DATA, axis: str = AXIS) -> list[dict]:
    """All non-human rows, unified. Mined rows get a synthetic group_id from their text hash so the
    split groups them individually (no pair to keep together).

    Order matters for dedup: first occurrence of a normalized text wins, so curated (hand-authored,
    most trusted) is loaded FIRST -- its label overrides any conflicting auto-generated duplicate."""
    interim = data_dir / "interim"
    out = []
    for r in read_jsonl(data_dir / "curated.jsonl"):        # committed hand-authored -> wins dedup conflicts
        gid = r.get("group_id") or "curated_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
        out.append(row(r["text"], r["label"], r.get("source", "curated"), gid, r.get("is_real", True)))
    for pf in pair_paths(axis):                            # every pair corpus (generated + curated) -> neg(0)/pos(1)
        for p in read_jsonl(Path(pf)):                     # sorted curated-first -> wins dedup like CURATED
            out.extend(pair_rows(p, axis))
    for r in read_jsonl(interim / "hard_negatives.jsonl"):
        out.append(row(r["text"], r["label"], "hard_neg", r["group_id"], r.get("is_real", False)))
    for r in read_jsonl(interim / "mined_sentences.jsonl"):
        if r.get("genre") in ("plain", "technical"):       # label-0 by construction; purple_prone -> triage_label.py
            gid = "mined_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
            out.append(row(r["text"], 0, "mined", gid, True))
    for r in read_jsonl(interim / "triaged.jsonl"):         # LLM-labeled real -> train/val only
        gid = "triaged_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
        out.append(row(r["text"], r["label"], "mined", gid, True))
    for r in read_jsonl(interim / "soft_euph_gen.jsonl"):    # gross(0)/euph-short(1) pairs, group-shared
        out.append(row(r["text"], r["label"], r["source"], r["group_id"], r.get("is_real", False)))
    for r in read_jsonl(interim / "calm_neg_gen.jsonl"):     # calm non-sexual hard negs (label 0)
        out.append(row(r["text"], r["label"], r["source"], r["group_id"], r.get("is_real", False)))
    for r in read_jsonl(COMMON):                             # axis-SHARED common negatives -> label 0
        gid = r.get("group_id") or "common_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
        out.append(row(r["text"], 0, r.get("source", "common"), gid, r.get("is_real", True)))
    return out


def load_human(path: Path = HUMAN) -> list[dict]:
    # ponytail: human rows are independent real sentences (no synthetic twin), so each gets its own
    # text-hash group. Ignore any incoming group_id -- the labeler reused one id for 139 rows, which
    # would dump them all onto one side of the calib/test split.
    out = []
    for r in read_jsonl(path):
        gid = "human_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]
        out.append(row(r["text"], r["label"], "human", gid, True))
    return out


def dedupe_validate(rows: list[dict]) -> tuple[list[dict], int, int, int]:
    """Drop exact-normalized dups (first occurrence wins) and integrity failures.
    Also counts conflicts: a dropped dup whose label disagrees with the kept row (the kept one wins
    -- pool order puts curated first, so curated's label is authoritative)."""
    seen, kept, dup, bad, conflict = {}, [], 0, 0, 0
    for r in rows:
        if not valid(r):
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


HELDOUT_HUMAN_FRAC = 0.30   # graded human paragraphs held out for calibration + test (by doc group)


def assign_paragraph_split(r: dict) -> str | None:
    """Split for a paragraph-axis row. None = drop.

    haiku_paragraphs (import_paragraphs.py) keep the SOURCE split: test stays the held-out test
    (it is OrbRewriter's modern_slop eval); half the source-val groups become calibration, keeping
    only rows whose blind audits all agreed, so the 0.90-precision gate is set on clean labels;
    calibration-group rows with split audits are dropped so a group never straddles val/calib.
    human_graded (grade_paragraphs.py) and sentence rows: held-out human share -> calib/test by
    group hash, the rest -> train/val by group hash."""
    if "split" in r:
        if r["split"] in ("train", "test"):
            return r["split"]
        if bucket(r["group_id"], 0.5, "calib"):
            return "calibration" if r.get("audit_agree") else None
        return "val"
    if r.get("heldout"):
        return "test" if bucket(r["group_id"], TEST_FRAC, "test") else "calibration"
    return "val" if bucket(r["group_id"], VAL_FRAC, "val") else "train"


NATURAL = {"reply", "native", "real"}   # labelled by the graders' vote; synthetic variants are labelled by construction


LLM_SOURCES = ("haiku", "aug_", "appdb")   # Haiku output, Haiku-augmented copies, Orb app.db LLM drafts


def synthetic(r: dict) -> bool:
    """Model-written text. appdb rows carry is_real (they are real app traffic), but an LLM wrote them."""
    return not r.get("is_real") or str(r.get("source", "")).startswith(LLM_SOURCES)


def tic_relabel(r: dict, c: Counter) -> dict:
    """A Haiku-generated label-0 row carrying a house tic (datagen/tics.py) becomes label 1, whatever the vote
    or construction said: the Haiku grader passes its own stock moves as "one flourish" = clean. The label is
    by rule, so `audit_agree` is set and the row may calibrate. Human-written rows never come through here."""
    from datagen.tics import tics
    if r["label"] != 0:
        return r
    found = tics(r["text"])
    if not found:
        return r
    c[f"{r.get('variant')} tic 0->1"] += 1
    return {**r, "label": 1, "tics": found, "audit_agree": True}


def apply_regrade(rows: list[dict]) -> list[dict]:
    """Join the current-rubric votes (grade_paragraphs --set haiku) onto the imported Haiku rows.

    Natural rows take the majority side of the new votes (tie -> dropped). Synthetic variants keep
    their by-construction label -- camouflaged=1 vs terse=0 is the contrast they exist for -- and the
    votes only re-score `audit_agree`, which gates calibration. A row without >= 2 current-rubric
    votes is dropped: one dataset never mixes two rubrics' labels. No grades file -> source labels."""
    from datagen.grade_paragraphs import RUBRIC, label as vote, rubric_tag
    path = INTERIM / "haiku_grades.jsonl"
    if not path.exists():
        print("  note: no haiku_grades.jsonl -- using the source (v1 rubric) labels")
        return rows
    tag = rubric_tag(RUBRIC.read_text(encoding="utf-8"))
    votes = {norm(g["text"]): g["audit_scores"] for g in read_jsonl(path) if g.get("rubric") == tag}
    out, c = [], Counter()
    for r in rows:
        a = votes.get(norm(r["text"]), [])
        if len(a) < 2:
            c["no current-rubric votes (dropped)"] += 1
            continue
        if r["variant"] in NATURAL:
            lab = vote(a)
            if lab is None:
                c["tied (dropped)"] += 1
                continue
            if lab != r["label"]:
                c[f"{r['variant']} relabelled {r['label']}->{lab}"] += 1
            r = {**r, "label": lab, "score": round(statistics.median(a))}
        r = {**r, "audit_scores": a, "audit_agree": all((s >= 2) == bool(r["label"]) for s in a), "rubric": tag}
        out.append(r if r.get("is_real") else tic_relabel(r, c))       # appdb "real" rows are not Haiku's
    print(f"  regrade {tag}: {len(rows)} -> {len(out)}  {dict(sorted(c.items()))}")
    return out


def human_negatives() -> list[dict]:
    """Graded human paragraphs, NEGATIVES ONLY: graded-clean rows are label 0; graded-purple ones are
    left out of every split. The axis targets LLM slop -- the rubric would otherwise teach that dense
    literary human prose (McCarthy grades [2,2,2]) is purple -- yet purple fanfic is never forced to 0."""
    from datagen.grade_paragraphs import RUBRIC, rubric_tag
    tag = rubric_tag(RUBRIC.read_text(encoding="utf-8"))
    rows = list(read_jsonl(INTERIM / "human_graded.jsonl"))
    stale = sum(r.get("rubric") != tag for r in rows)
    if stale:
        raise SystemExit(f"FATAL: {stale} human_graded rows were graded under another rubric than rubric.txt "
                         f"({tag}) -- rerun datagen.grade_paragraphs --set human")
    keep = [r for r in rows if r["label"] == 0]
    if rows:
        print(f"  human paragraphs: {len(rows)} graded -> {len(keep)} clean kept, "
              f"{len(rows) - len(keep)} graded purple left out")
    return keep


def augmented() -> list[dict]:
    """Graded filler / concrete-control copies of clean train paragraphs (datagen.augment_filler). Labels are
    the current rubric's majority vote; a file graded under another rubric is refused, as for human_graded."""
    from datagen.grade_paragraphs import RUBRIC, rubric_tag
    rows = [r for f in sorted(INTERIM.glob("aug_graded*.jsonl")) for r in read_jsonl(f)]
    if not rows:
        return []
    tag = rubric_tag(RUBRIC.read_text(encoding="utf-8"))
    stale = sum(r.get("rubric") != tag for r in rows)
    if stale:
        raise SystemExit(f"FATAL: {stale} aug_graded rows were graded under another rubric than rubric.txt ({tag}) "
                         f"-- rerun datagen.grade_paragraphs --set aug")
    t = Counter()
    rows = [tic_relabel(r, t) for r in rows]        # every copy carries Haiku sentences (bases or inserted beats)
    c = Counter((r.get("round", 1), r["variant"], r["label"]) for r in rows)
    print("  filler augmentation: " + "  ".join(f"r{rd} {v}={lab}: {n}" for (rd, v, lab), n in sorted(c.items()))
          + f"  {dict(sorted(t.items()))}")
    return rows


def build_paragraph():
    """Paragraph axis: own paragraph rows + the sentence axis's rows as one-sentence paragraphs.

    Priority for dedup (first normalized text wins): held-out rows first so a train copy of a
    test text is dropped rather than leaking, then hand-curated, then the rest."""
    mix = SENTENCE_MIX.get(AXIS)
    rows = []
    rows += apply_regrade(list(read_jsonl(INTERIM / "haiku_paragraphs.jsonl")))
    for r in human_negatives():                                    # grade_paragraphs.py
        rows.append({**r, "heldout": bucket(r["group_id"], HELDOUT_HUMAN_FRAC, "heldout")})
    curated = [{**r, "unit": "paragraph", "group_id": r.get("group_id") or
                "curated_" + hashlib.md5(norm(r["text"]).encode()).hexdigest()[:12]} for r in read_jsonl(CURATED)]
    human = [{**r, "unit": "paragraph", "heldout": True} for r in load_human(HUMAN)]
    if mix:
        sent_dir = Path("data") / mix
        rows += [{**r, "unit": "sentence"} for r in load_pool(sent_dir, mix)]
        human += [{**r, "unit": "sentence", "heldout": True} for r in load_human(sent_dir / "human_labeled.jsonl")]
    if not any(r.get("unit") == "paragraph" for r in rows):
        raise SystemExit(f"FATAL: no paragraph rows in {INTERIM} -- run datagen.import_paragraphs first")

    split_of = {}
    for r in human + curated + rows:
        split_of[id(r)] = assign_paragraph_split(r)
    # filler augmentation (augment_filler + grade_paragraphs --set aug) follows its base's group, so the base,
    # its filler copy and its concrete control are never split apart; bases are train-side only by construction
    group_split = {r["group_id"]: split_of[id(r)] for r in rows}
    aug = augmented()
    for r in aug:
        sp = group_split.get(r["group_id"])
        split_of[id(r)] = sp if sp in ("train", "val") else None
    rows += aug
    # Synthetic text is never a negative: a Haiku / generated row is label 1 (graded, constructed or tic) or it is
    # out. Negatives are human-written only. This removes the terse/plain/vivid/isolated controls, the concrete
    # arm of the augmentation and the sentence axis's generated negatives.
    syn0 = Counter(r.get("source") for r in human + curated + rows if r["label"] == 0 and synthetic(r))
    human, curated, rows = ([r for r in part if r["label"] == 1 or not synthetic(r)] for part in (human, curated, rows))
    print(f"  synthetic negatives dropped: {sum(syn0.values())}  {dict(syn0.most_common())}")
    order = {"test": 0, "calibration": 1}
    ordered = sorted(human + curated + rows, key=lambda r: order.get(split_of[id(r)], 2))  # stable
    seen, out, dup, bad, conflict, dropped = {}, {k: [] for k in ("train", "val", "calibration", "test")}, 0, 0, 0, 0
    for r in ordered:
        sp = split_of[id(r)]
        if sp is None:
            dropped += 1
            continue
        if not valid(r):
            bad += 1
            continue
        k = norm(r["text"])
        if k in seen:
            dup += 1
            conflict += seen[k] != r["label"]
            continue
        seen[k] = r["label"]
        clean = {k2: v for k2, v in r.items() if k2 not in ("split", "heldout")}
        out[sp].append(clean)

    for name, part in out.items():
        write(OUT_SPLITS / f"{name}.jsonl", part)
    labeled = [r for part in out.values() for r in part]
    write(OUT_LABELED, labeled)

    def units(part):
        c = {u: sum(r.get("unit") == u for r in part) for u in ("paragraph", "sentence")}
        return f"[{c['paragraph']} para / {c['sentence']} sent]"
    print(f"pooled {dist(labeled)}  (dropped {dup} dup, {bad} invalid, {dropped} split-audit calib rows)")
    if conflict:
        print(f"  WARNING: {conflict} duplicate text(s) had conflicting labels -> kept the held-out/curated-first row")
    for name, part in out.items():
        print(f"  {name:<11} {dist(part)}  {units(part)}")
    if not any(r.get("source") == "human_graded" for r in labeled):
        print("  note: no graded human paragraphs yet (datagen.grade_paragraphs) -- the false-positive rate on "
              "human prose is unmeasured and untrained against")
    print(f"-> {OUT_LABELED}, {OUT_SPLITS}/{{train,val,calibration,test}}.jsonl")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if UNIT == "paragraph":
        return build_paragraph()

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

    # paragraph unit: a single newline is fine, a blank line is two paragraphs
    assert is_paragraph('She waits.\n"Go," he says.') and not is_paragraph("One.\n\nTwo.")
    assert valid({"text": "a\nparagraph", "unit": "paragraph"}) and not valid({"text": "a\nsentence"})
    # source split is kept; source test never moves; calib rows need unanimous audits
    assert assign_paragraph_split({"split": "test", "group_id": "g"}) == "test"
    assert assign_paragraph_split({"split": "train", "group_id": "g"}) == "train"
    calib_gid = next(f"g{i}" for i in range(100) if bucket(f"g{i}", 0.5, "calib"))
    assert assign_paragraph_split({"split": "val", "group_id": calib_gid, "audit_agree": True}) == "calibration"
    assert assign_paragraph_split({"split": "val", "group_id": calib_gid, "audit_agree": False}) is None
    assert assign_paragraph_split({"heldout": True, "group_id": "h"}) in ("test", "calibration")
    assert assign_paragraph_split({"group_id": "h"}) in ("train", "val")

    # regrade join: natural rows take the vote, synthetic keep their label, stale rubric / no votes drop
    import tempfile
    import datagen.grade_paragraphs as gp
    global INTERIM
    saved = INTERIM, gp.RUBRIC
    tmp = Path(tempfile.mkdtemp())
    INTERIM, gp.RUBRIC = tmp, tmp / "rubric.txt"
    try:
        gp.RUBRIC.write_text("v2")
        tag = gp.rubric_tag("v2")
        write(tmp / "haiku_grades.jsonl", [
            {"text": "Reply one.", "audit_scores": [2, 2, 1], "rubric": tag},
            {"text": "Camo one.", "audit_scores": [1, 2, 1], "rubric": tag},
            {"text": "Old rubric.", "audit_scores": [3, 3, 3], "rubric": "rOLD"}])
        rr = apply_regrade([{"text": "reply  ONE.", "label": 0, "variant": "reply"},
                            {"text": "Camo one.", "label": 1, "variant": "camouflaged"},
                            {"text": "Old rubric.", "label": 1, "variant": "reply"},
                            {"text": "Never graded.", "label": 0, "variant": "plain"}])
        assert [(r["text"], r["label"], r["audit_agree"]) for r in rr] == [
            ("reply  ONE.", 1, False), ("Camo one.", 1, False)], rr
        write(tmp / "human_graded.jsonl", [{"text": "a", "label": 0, "rubric": tag}, {"text": "b", "label": 1, "rubric": tag}])
        assert [r["text"] for r in human_negatives()] == ["a"]
        write(tmp / "aug_graded.jsonl", [{"text": "x", "label": 1, "variant": "aug:filler", "rubric": tag,
                                           "inserted": ["A breath."]},
                                          {"text": "y", "label": 1, "variant": "aug:filler", "rubric": tag,
                                           "inserted": ["Something in the room shifts."]}])
        write(tmp / "aug_graded_r2.jsonl", [{"text": "z", "label": 0, "variant": "aug:concrete", "rubric": tag,
                                              "inserted": ["She sets the cup down."], "round": 2}])
        assert sorted(r["text"] for r in augmented()) == ["x", "y", "z"]  # every round, every graded row
        (tmp / "aug_graded_r2.jsonl").unlink()
        write(tmp / "aug_graded.jsonl", [{"text": "x", "label": 1, "variant": "aug:filler", "rubric": "rOLD"}])
        try:
            augmented()
            raise AssertionError("stale-rubric augmentation was accepted")
        except SystemExit:
            pass
    finally:
        INTERIM, gp.RUBRIC = saved
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
