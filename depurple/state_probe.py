"""Plant-and-query world-state and spatial probe with deterministic grading.

Scripted RP scenarios plant a fact early (inventory count, room position, who-is-where) and
query it late; the reply is graded by word-boundary regex. Measures world-state
and spatial behavior that can degrade in generation even when logprob benchmarks
look stable.

Selection mirrors serve.py / eval.py / tool_probe.py: `gen <variant> [--strength S]` builds the
model in memory via serve.load (base | best | trialN), so the model you probe is the model you
serve. `gen` loads one model, rolls out every scenario, dumps the replies to disk, and exits.
`grade [base] [edited]` then runs as pure regex and is re-runnable without regenerating.

    python -m depurple.state_probe gen base
    python -m depurple.state_probe gen best                # or: gen trial7 --strength 0.4
    python -m depurple.state_probe grade                   # no GPU; reads base + best reply dumps
    python -m depurple.state_probe grade base trial7       # compare a specific variant
    STATE_SCEN=depurple/babi_scenarios.jsonl python -m depurple.state_probe gen base   # babi rows
    python -m depurple.state_probe --demo                  # grader self-check, no model/network

The delta is the useful signal; absolute pass rates depend on the base model's bookkeeping.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

# STATE_SCEN points the same probe at another scenario file (e.g. babi_scenarios.jsonl from
# babi_snapshot.py); reply dumps are namespaced off the file stem so they never collide.
SCEN = Path(os.environ.get("STATE_SCEN", "depurple/state_scenarios.jsonl"))
_stem = SCEN.stem.replace("_scenarios", "")     # state_scenarios -> state ; babi_scenarios -> babi
REGRESS = -0.05   # flag an aggregate accuracy drop past this (~1 query on this small set)
MAX_ROWS = 30     # per-scenario table cap -- babi runs up to ~1000 scenarios and would flood the console
MAX_FAILS = 15    # newly-failed dump cap, ditto -- the aggregate accuracy line already carries the signal


def replies_path(variant: str) -> Path:
    # Namespaced by scenario stem, MODEL slug, AND variant so {state,babi} x {e4b,31b} x
    # {base,trialN} never collide. SLUG matters because a 'base' dump is the BASE MODEL's
    # rollouts: without it, switching DEPURPLE_MODEL silently reuses the old model's base
    # (is_fresh keys on the scenario file, not the model) and grades a delta against the wrong base.
    from depurple._model import SLUG
    return Path(f"depurple/{_stem}-replies-{variant}-{SLUG}.jsonl")


def _fingerprint() -> str:
    """sha1 of the scenario file -- the key for reusing a base reply dump across runs. Stored as a
    meta line in the dump; eval_bench skips the base pass iff it still matches. babi_snapshot
    rewrites babi_scenarios.jsonl deterministically, so identical (n, fewshot) -> identical bytes ->
    same key -> base reused; any change (different --n/--fewshot, new babi data) busts it -> regen."""
    return hashlib.sha1(SCEN.read_bytes()).hexdigest()


def is_fresh(variant: str) -> bool:
    """True iff variant's reply dump exists AND was generated from the current scenarios."""
    p = replies_path(variant)
    if not p.exists():
        return False
    first = next((l for l in p.read_text().splitlines() if l.strip()), "")
    try:
        return json.loads(first).get("_scen") == _fingerprint()
    except (json.JSONDecodeError, AttributeError):
        return False    # old pre-fingerprint dump (no meta line) -> treat as stale, regenerate


def grade(reply: str, check: dict) -> bool:
    """True if an expected answer appears (word-boundary, case-insensitive) and no rejected
    distractor does. reject fires first so 'I think it's two or three' fails a 'two' check that
    rejects 'three' — the model hedging both answers is a miss, not a pass.
    ponytail: presence match; if a long chatty reply mentions the right token in passing and
    false-passes, switch the query to 'answer with one word' rather than parsing harder."""
    low = reply.lower()
    def hit(p: str) -> bool:
        return re.search(rf"\b{re.escape(p.lower())}\b", low) is not None
    if any(hit(p) for p in check.get("reject", [])):
        return False
    return any(hit(p) for p in check["expect"])


def load_scenarios() -> list[dict]:
    scens = [json.loads(l) for l in SCEN.read_text().splitlines() if l.strip()]
    for s in scens:                                  # authoring guard: checks must align to turns
        assert len(s["checks"]) == len(s["user_turns"]), \
            f"{s['id']}: {len(s['checks'])} checks for {len(s['user_turns'])} turns"
    return scens


# --- phase 1: generate (one model resident, then this process exits to free VRAM) ---

def gen(variant: str, strength: float = 1.0) -> None:
    from depurple import serve                       # reuse variant->in-memory-model selection
    from depurple.objective import rollout_many

    serve.load(variant, strength)                    # base | best | trialN, built in memory (no baked copy)
    scenarios = load_scenarios()
    # Batch the rollouts: rollout_many packs up to ROLLOUT_BATCH scenarios into one left-padded
    # generate() per turn (set ROLLOUT_BATCH to the GPU's parallel width). base and edit gen run as
    # separate processes but over identical scenarios/order, so padding lands the same way in both and
    # the base-vs-edit delta stays consistent. A leftover batch-1 base dump from before batching won't
    # share this padding; rm depurple/<stem>-replies-base-*.jsonl so the fresh check regenerates it.
    replies_by_scen = rollout_many(serve.model, serve.tok, scenarios)
    out = replies_path(variant)
    with out.open("w") as f:
        f.write(json.dumps({"_scen": _fingerprint()}) + "\n")     # freshness key (see is_fresh); _read skips it
        for s, replies in zip(scenarios, replies_by_scen):
            f.write(json.dumps({"id": s["id"], "replies": replies}) + "\n")
    print(f"wrote {out} ({len(scenarios)} scenarios for variant '{variant}')")


# --- phase 2: grade (no model — regex only) ---

def _read(variant: str) -> dict[str, list[str]]:
    p = replies_path(variant)
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python -m depurple.state_probe gen {variant}` first")
    return {d["id"]: d["replies"] for d in (json.loads(l) for l in p.read_text().splitlines() if l.strip()) if "id" in d}


def _graded(scen: dict, replies: list[str]) -> list[tuple]:
    """[(turn_idx, query, passed, reply)] for each GRADED turn (check not null)."""
    return [(i, scen["user_turns"][i], grade(replies[i], c), replies[i])
            for i, c in enumerate(scen["checks"]) if c]


def compare(base: dict, edited: dict) -> None:
    sids = list(base)
    print(f"{'scenario':<22}{'base':>8}{'edited':>9}")
    print("-" * 39)
    tb = te = n = 0
    regressions = []
    rows = []
    for sid in sids:
        bp = sum(ok for _, _, ok, _ in base[sid]); g = len(base[sid])
        ep = sum(ok for _, _, ok, _ in edited[sid])
        tb += bp; te += ep; n += g
        rows.append(f"{sid:<22}{bp:>5}/{g:<2}{ep:>6}/{g:<2}")
        for (i, q, bok, _), (_, _, eok, er) in zip(base[sid], edited[sid]):
            if bok and not eok:
                regressions.append((sid, q, er))
    if len(rows) <= MAX_ROWS:
        for r in rows:
            print(r)
    else:
        print(f"({len(rows)} scenarios -- per-row table suppressed, see aggregate below)")
    print("-" * 39)
    ab, ae = (tb / n if n else 0.0), (te / n if n else 0.0)
    d = ae - ab
    flag = "  <-- REVIEW: state/spatial regressed" if d < REGRESS else ""
    print(f"{'accuracy':<22}{ab:>8.3f}{ae:>9.3f}   delta {d:+.3f}{flag}")
    if regressions:
        shown = regressions[:MAX_FAILS]
        more = f", showing first {MAX_FAILS}" if len(regressions) > MAX_FAILS else ""
        print(f"\nnewly failed under the edit ({len(regressions)} total{more}) "
              "-- eyeball these, the scalar can't:")
        for sid, q, er in shown:
            print(f"  [{sid}] {q}\n    edited> {er[:200]}")
    print("\nOK: no state/spatial drop beyond noise" if not flag
          else "\nthe edit hurt world-state/spatial tracking — lower strength or re-pick the trial")


def do_grade(base_var: str = "base", edited_var: str = "best") -> None:
    scen_by_id = {s["id"]: s for s in load_scenarios()}
    base, edited = _read(base_var), _read(edited_var)
    missing = set(scen_by_id) - (set(base) & set(edited))
    if missing:
        raise SystemExit(f"reply dumps stale — missing {sorted(missing)}; rerun `gen {base_var}` and `gen {edited_var}`")
    b = {sid: _graded(scen_by_id[sid], base[sid]) for sid in scen_by_id}
    e = {sid: _graded(scen_by_id[sid], edited[sid]) for sid in scen_by_id}
    compare(b, e)


def demo() -> None:
    """Self-check pure grader logic without model or network calls."""
    assert grade("You now have 4 gold coins.", {"expect": ["4", "four"], "reject": ["3", "5"]})
    assert grade("Four coins jingle in my purse.", {"expect": ["4", "four"], "reject": ["3", "5"]})
    assert not grade("You still have 3 gold.", {"expect": ["4", "four"], "reject": ["3", "5"]})
    assert grade("I'm in the library now, dust everywhere.", {"expect": ["library"], "reject": ["foyer"]})
    assert not grade("Back in the foyer, I catch my breath.", {"expect": ["library"], "reject": ["foyer"]})
    assert not grade("I think it's two or three.", {"expect": ["2", "two"], "reject": ["three", "3"]})  # hedge = miss
    assert not grade("4th time's the charm", {"expect": ["4", "four"]})   # \b4\b doesn't match '4th'
    print("state_probe demo ok")


def main() -> None:
    args = sys.argv[1:]
    if "--demo" in args:
        return demo()
    strength = 1.0
    if "--strength" in args:
        i = args.index("--strength")
        strength = float(args[i + 1]); del args[i:i + 2]
    cmd = args[0] if args else ""
    if cmd == "gen" and len(args) > 1:
        return gen(args[1], strength)
    if cmd == "fresh" and len(args) > 1:                  # exit 0 = reply dump matches current scenarios
        raise SystemExit(0 if is_fresh(args[1]) else 1)
    if cmd == "grade":
        return do_grade(args[1] if len(args) > 1 else "base",
                        args[2] if len(args) > 2 else "best")
    raise SystemExit("usage: state_probe.py [gen <variant> [--strength S] | fresh <variant> | grade [base] [edited] | --demo]")


if __name__ == "__main__":
    main()
