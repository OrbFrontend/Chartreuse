"""Grade human paragraphs on the paragraph-purple scale -> interim/human_graded.jsonl.

WHY. The Haiku paragraph set is ~97% Haiku-written; its only human prose is 353 app.db rows.
A head trained on it flags 15-25% of unseen human paragraphs (books worst), and many of those are
literary prose, not slop. Human paragraphs are the missing negatives -- but NOT blanket label 0:
some human prose (fanfic especially) really is purple. So they are graded the way the source set
was: the same rubric (data/<axis>/rubric.txt, the source's RUBRIC verbatim), the same blind
shuffled batch prompt, the same grader (claude-haiku-5-5 via `claude -p`), several independent
passes, majority side (score >= 2 is purple). `--set haiku` regrades the imported Haiku paragraphs
the same way, so a rubric revision relabels the natural rows (build_dataset joins the votes back).
`--set aug` grades datagen.augment_filler's filler / concrete-control items, which have no label until then.
Raw grades are keyed by the rubric's hash: editing rubric.txt starts a fresh vote. build_dataset holds out 30% of the doc groups for
calibration + test, so the human false-positive rate becomes a measured number.

Sources (--src, repeatable): a .txt is split on blank lines (doc group = 20 adjacent blocks); a
.jsonl reads `target` or `text` (doc group = its `doc` field, else the row). Paragraphs of 20-200
English words, deduped. The sample is frozen in interim/human_sample.jsonl on first run so passes
resume against the same items; --resample redraws it.

    DEPURPLE_AXIS=purple_para python -m datagen.grade_paragraphs --per-source 600 \\
        --src human_writing.txt --src ../OrbRewriter/data/corpus/paragraphs_books.jsonl ...
    DEPURPLE_AXIS=purple_para python -m datagen.grade_paragraphs --set haiku    # regrade the Haiku set
    DEPURPLE_AXIS=purple_para python -m datagen.grade_paragraphs --set aug [--round 2]   # grade the augmentation
    DEPURPLE_AXIS=purple_para python -m datagen.grade_paragraphs --limit 2      # 2 batches/pass smoke test
    DEPURPLE_AXIS=purple_para python -m datagen.grade_paragraphs --assemble     # rebuild output only
    DEPURPLE_AXIS=purple_para python -m datagen.grade_paragraphs --demo
"""
import argparse
import asyncio
import hashlib
import json
import random
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from core.paths import DATA, INTERIM, UNIT
from datagen.import_paragraphs import english

MODEL = "claude-haiku-5-5"
RUBRIC = DATA / "rubric.txt"
SAMPLE = INTERIM / "human_sample.jsonl"
HAIKU = INTERIM / "haiku_paragraphs.jsonl"            # import_paragraphs output, regraded by --set haiku
SETS = {  # set -> (raw grades, assembled output)
    "human": (INTERIM / "human_grades_raw.jsonl", INTERIM / "human_graded.jsonl"),
    "haiku": (INTERIM / "haiku_grades_raw.jsonl", INTERIM / "haiku_grades.jsonl"),
    "aug": (INTERIM / "aug_grades_raw.jsonl", INTERIM / "aug_graded.jsonl"),   # augment_filler items
}
AUG = INTERIM / "aug_items.jsonl"
BATCH = 12                       # the source's AUDIT_BATCH
PASSES = (5, 6, 7)               # shuffle seeds; each is an independent blind pass
MIN_WORDS, MAX_WORDS, DOC_BLOCKS = 20, 200, 20
AUDIT_TASK = ("Grade each paragraph independently on the 0-3 scale. They are unrelated and shuffled; "
              "some are deliberately tricky.\n{items}")
SCHEMA = {"type": "object", "properties": {"grades": {"type": "array", "items": {
    "type": "object", "properties": {"id": {"type": "string"}, "score": {"type": "integer", "minimum": 0, "maximum": 3}},
    "required": ["id", "score"], "additionalProperties": False}}}, "required": ["grades"], "additionalProperties": False}


def rubric_tag(text: str) -> str:
    """Raw grades are keyed by the rubric's hash, so editing rubric.txt starts a fresh grading and
    grades from two rubrics never mix in one vote."""
    return "r" + hashlib.sha256(text.encode()).hexdigest()[:8]


def read_jsonl(path: Path):
    if not path.exists():
        return
    for line in path.open(encoding="utf-8"):
        try:
            yield json.loads(line)
        except ValueError:            # half-written last line of a resumed run
            continue


def candidates(src: Path) -> list[tuple[str, str]]:
    """(doc group, paragraph) pairs from one source file."""
    if src.suffix == ".txt":
        blocks = [b.strip() for b in src.read_text(encoding="utf-8").split("\n\n")]
        return [(f"{src.stem}:{i // DOC_BLOCKS}", b) for i, b in enumerate(blocks)]
    out = []
    for i, r in enumerate(read_jsonl(src)):
        text = (r.get("target") or r.get("text") or "").strip()
        out.append((f"{src.stem}:{r.get('doc', i)}", text))
    return out


def eligible(text: str) -> bool:
    n = len(text.split())
    return MIN_WORDS <= n <= MAX_WORDS and english(text) and "\n\n" not in text and "```" not in text


def draw(srcs: list[Path], per_source: int, seed: int = 0) -> list[dict]:
    items, seen = [], set()
    for src in srcs:
        pool = [(d, t) for d, t in candidates(src) if eligible(t)]
        random.Random(f"{seed}:{src.name}").shuffle(pool)
        n = 0
        for doc, text in pool:
            key = " ".join(text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            items.append({"id": "h" + hashlib.md5(key.encode()).hexdigest()[:12], "text": text,
                          "doc": doc, "src": src.stem})
            n += 1
            if n == per_source:
                break
        print(f"  {src}: {len(pool)} eligible -> {n} sampled")
    return items


def batches(items: list[dict], seed: int, tag: str) -> list[dict]:
    order = list(items)
    random.Random(seed).shuffle(order)
    return [{"key": f"{tag}:s{seed}:" + "|".join(x["id"] for x in order[k:k + BATCH]),
             "ids": [x["id"] for x in order[k:k + BATCH]], "texts": [x["text"] for x in order[k:k + BATCH]]}
            for k in range(0, len(order), BATCH)]


def prompt(batch: dict) -> str:
    # opaque per-batch ids, as in the source audit: the grader never sees where a text came from
    return AUDIT_TASK.format(items="\n".join(f"\n[p{n}]\n{t}" for n, t in enumerate(batch["texts"])))


async def call(text: str, system: str, timeout: float = 240) -> dict:
    """One isolated `claude -p` call, the source's flags: no tools, no MCP, no session, structured output."""
    with tempfile.TemporaryDirectory(prefix="grade-") as workdir:
        sysfile = Path(workdir) / "system.txt"
        sysfile.write_text(system)
        args = ["claude", "-p", "--model", MODEL, "--effort", "low", "--system-prompt-file", str(sysfile),
                "--no-session-persistence", "--safe-mode", "--tools", "", "--disallowedTools", "mcp__*",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--permission-mode", "dontAsk",
                "--no-chrome", "--output-format", "json", "--json-schema", json.dumps(SCHEMA, separators=(",", ":"))]
        proc = await asyncio.create_subprocess_exec(*args, cwd=workdir, stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(text.encode()), timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise
    data = json.loads(out)
    if data.get("is_error") or not isinstance(data.get("structured_output"), dict):
        raise RuntimeError(f"{data.get('subtype')}: {str(data.get('result'))[:200]} {err.decode()[:200]}")
    return {"out": data["structured_output"], "cost": data.get("total_cost_usd") or 0.0}


async def run(todo: list[dict], system: str, jobs: int, raw: Path) -> None:
    gate, stats = asyncio.Semaphore(jobs), Counter()

    async def one(b: dict) -> None:
        async with gate:
            for attempt in range(3):
                try:
                    res = await call(prompt(b), system)
                    break
                except Exception as exc:  # noqa: BLE001 -- retry, then record the failure
                    err = f"{type(exc).__name__}: {exc}"
                    await asyncio.sleep(2 + 4 * attempt)
            else:
                stats["fail"] += 1
                print(f"  FAIL {b['key'][:40]}: {err[:160]}", flush=True)
                return
        stats["ok"] += 1
        stats["cost"] += res["cost"]
        with raw.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": b["key"], "ids": b["ids"], "grades": res["out"]["grades"]}) + "\n")
        if stats["ok"] % 20 == 0:
            print(f"  {stats['ok']}/{len(todo)} batches  ${stats['cost']:.3f}", flush=True)

    await asyncio.gather(*(one(b) for b in todo))
    print(f"graded ok={stats['ok']} fail={stats['fail']} cost=${stats['cost']:.3f}")


def grades_by_item(raw: Path, tag: str) -> dict[str, list[int]]:
    """One grade per item per pass, under rubric `tag` only (key = <tag>:<seed>:<ids>)."""
    seen, grades = set(), defaultdict(list)
    for r in read_jsonl(raw):
        parts = r["key"].split(":", 2)
        if len(parts) < 3 or parts[0] != tag:
            continue                                  # another rubric's pass (or a pre-tag v1 row)
        seed = parts[1]
        by_pid = {g["id"].strip("[]"): g["score"] for g in r["grades"]}
        for n, item in enumerate(r["ids"]):
            if f"p{n}" in by_pid and (seed, item) not in seen:
                seen.add((seed, item))
                grades[item].append(by_pid[f"p{n}"])
    return grades


def label(scores: list[int]) -> int | None:
    """Majority side over the passes (score >= 2 = purple); None on a tie or fewer than 2 grades."""
    if len(scores) < 2:
        return None
    pos = sum(s >= 2 for s in scores)
    return None if pos * 2 == len(scores) else int(pos * 2 > len(scores))


def haiku_items() -> list[dict]:
    """Every imported Haiku paragraph, id'd by normalized text (build_dataset joins on the same key)."""
    return [{"id": "k" + hashlib.md5(" ".join(r["text"].lower().split()).encode()).hexdigest()[:12],
             "text": r["text"]} for r in read_jsonl(HAIKU)]


def assemble_haiku(items: list[dict], raw: Path, out: Path, tag: str) -> None:
    grades = grades_by_item(raw, tag)
    rows = [{"text": it["text"], "audit_scores": grades[it["id"]], "rubric": tag} for it in items if it["id"] in grades]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    full = sum(len(r["audit_scores"]) >= len(PASSES) for r in rows)
    print(f"-> {out}: {len(rows)}/{len(items)} paragraphs graded under {tag}, {full} with all {len(PASSES)} passes")


def assemble_aug(items: list[dict], raw: Path, out: Path, tag: str) -> None:
    """Augmentation items -> labelled rows (majority side, ties dropped), each arm reported by k."""
    grades, rows, skipped = grades_by_item(raw, tag), [], Counter()
    for it in items:
        a = grades.get(it["id"], [])
        lab = label(a)
        if lab is None:
            skipped["ungraded/tied"] += 1
            continue
        rows.append({"text": it["text"], "label": lab, "source": f"aug_{it['arm']}", "group_id": it["group_id"],
                     "round": it.get("round", 1),
                     "is_real": it["is_real"], "unit": "paragraph", "variant": f"aug:{it['arm']}", "k": it["k"],
                     "inserted": it["inserted"], "base_variant": it["base_variant"], "score": round(statistics.median(a)),
                     "audit_scores": a, "audit_agree": len({s >= 2 for s in a}) == 1, "rubric": tag})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"-> {out}: {len(rows)} graded under {tag} ({dict(skipped)})")
    for arm in ("filler", "concrete"):
        for k in sorted({r["k"] for r in rows}):
            part = [r for r in rows if r["variant"] == f"aug:{arm}" and r["k"] == k]
            if part:
                print(f"  {arm:<9} k={k}  n={len(part):<5} purple={sum(r['label'] for r in part) / len(part):.3f}  "
                      f"unanimous={sum(r['audit_agree'] for r in part) / len(part):.3f}")


def assemble(items: list[dict], raw: Path, out: Path, tag: str) -> None:
    grades, rows, skipped = grades_by_item(raw, tag), [], Counter()
    for it in items:
        a = grades.get(it["id"], [])
        lab = label(a)
        if lab is None:
            skipped["ungraded/tied"] += 1
            continue
        rows.append({"text": it["text"], "label": lab, "source": "human_graded",
                     "group_id": "hg_" + hashlib.md5(it["doc"].encode()).hexdigest()[:12], "is_real": True,
                     "unit": "paragraph", "variant": f"human:{it['src']}", "score": round(statistics.median(a)),
                     "audit_scores": a, "audit_agree": len({s >= 2 for s in a}) == 1, "rubric": tag})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"-> {out}: {len(rows)} graded under {tag} ({dict(skipped)})")
    for src in sorted({r["variant"] for r in rows}):
        part = [r for r in rows if r["variant"] == src]
        print(f"  {src:<32} n={len(part):<5} purple={sum(r['label'] for r in part) / len(part):.3f}  "
              f"unanimous={sum(r['audit_agree'] for r in part) / len(part):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), default="human",
                    help="human: the frozen human sample; haiku: regrade interim/haiku_paragraphs.jsonl; "
                         "aug: grade interim/aug_items.jsonl (datagen.augment_filler)")
    ap.add_argument("--src", action="append", type=Path, default=[], help="paragraph source (.txt or .jsonl)")
    ap.add_argument("--per-source", type=int, default=600)
    ap.add_argument("--resample", action="store_true", help="redraw the frozen sample")
    ap.add_argument("--limit", type=int, default=0, help="batches per pass (smoke test)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--round", type=int, default=1, help="--set aug: which augment_filler round (aug_items_rN.jsonl)")
    ap.add_argument("--assemble", action="store_true", help="only rebuild the set's output from the raw grades")
    a = ap.parse_args()
    if UNIT != "paragraph":
        raise SystemExit("FATAL: grade_paragraphs is for a paragraph axis (DEPURPLE_AXIS=purple_para)")
    INTERIM.mkdir(parents=True, exist_ok=True)
    system = RUBRIC.read_text(encoding="utf-8")
    tag, (raw, out) = rubric_tag(system), SETS[a.set]
    aug_items = AUG
    if a.set == "aug" and a.round > 1:          # each round is graded on its own, so earlier batches stay valid
        raw, out, aug_items = (p.with_name(f"{p.stem}_r{a.round}.jsonl") for p in (raw, out, AUG))
    if a.set == "haiku":
        items = haiku_items()
        if not items:
            raise SystemExit(f"FATAL: {HAIKU} is empty -- run datagen.import_paragraphs first")
    elif a.set == "aug":
        items = list(read_jsonl(aug_items))
        if not items:
            raise SystemExit(f"FATAL: {aug_items} is empty -- run datagen.augment_filler --round {a.round} first")
    elif a.resample or not SAMPLE.exists():  # human
        if not a.src:
            raise SystemExit("FATAL: no frozen sample yet -- pass at least one --src")
        items = draw(a.src, a.per_source)
        SAMPLE.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
        print(f"froze {len(items)} paragraphs -> {SAMPLE}")
    if a.set == "human":
        items = list(read_jsonl(SAMPLE))
    if not a.assemble:
        done = {r["key"] for r in read_jsonl(raw)}
        todo = []
        for seed in PASSES:
            bs = [b for b in batches(items, seed, tag) if b["key"] not in done]
            todo += bs[:a.limit] if a.limit else bs
        print(f"{a.set}: {len(items)} paragraphs x {len(PASSES)} passes under rubric {tag}: {len(todo)} batches to grade")
        asyncio.run(run(todo, system, a.jobs, raw))
    {"haiku": assemble_haiku, "aug": assemble_aug, "human": assemble}[a.set](items, raw, out, tag)


def demo():
    assert label([0, 1, 2]) == 0 and label([2, 3, 1]) == 1 and label([2, 1]) is None and label([3]) is None
    items = [{"id": f"h{i}", "text": f"t{i}"} for i in range(30)]
    bs = batches(items, 5, "rX")
    assert sum(len(b["ids"]) for b in bs) == 30 and max(len(b["ids"]) for b in bs) == BATCH
    assert batches(items, 5, "rX")[0]["key"] == bs[0]["key"] and batches(items, 6, "rX")[0]["key"] != bs[0]["key"]
    assert bs[0]["key"].startswith("rX:s5:") and rubric_tag("a") != rubric_tag("b")
    # grades from another rubric (or the untagged v1 keys) never enter the vote
    import tempfile as _t
    raw = Path(_t.mkdtemp()) / "raw.jsonl"
    g = [{"id": "p0", "score": 3}]
    raw.write_text("".join(json.dumps(x) + "\n" for x in [
        {"key": "rX:s5:h0", "ids": ["h0"], "grades": g}, {"key": "rX:s6:h0", "ids": ["h0"], "grades": g},
        {"key": "rX:s6:h0", "ids": ["h0"], "grades": [{"id": "[p0]", "score": 0}]},   # regraded batch: first wins
        {"key": "rY:s7:h0", "ids": ["h0"], "grades": g}, {"key": "s5:h0", "ids": ["h0"], "grades": g}]))
    assert grades_by_item(raw, "rX") == {"h0": [3, 3]}, grades_by_item(raw, "rX")
    assert "[p0]" in prompt(bs[0]) and "h" not in prompt(bs[0]).split("[p0]")[1].split()[0]
    assert eligible(" ".join(["word"] * 30)) and not eligible("too short") and not eligible("Mình " * 30)
    print("demo ok")


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else main()
