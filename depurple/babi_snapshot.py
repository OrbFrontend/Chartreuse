"""Convert bAbI rows into state_probe scenarios.

state_probe grades chat outputs by word-boundary containment and runs through the
same chat template used for serving. This script reshapes single-word bAbI rows
into that schema so the state/spatial signal can be tested generatively.

Output is depurple/babi_scenarios.jsonl in the state_scenarios schema
({id, system, user_turns:[passage+question], checks:[{expect:[answer]}]}). Run the existing
probe over it via the STATE_SCEN env knob -- no new probe, no new grader:

Easiest: `./depurple/eval_bench.sh --variant trial12 babi 100` drives the whole thing. Manually:

    python -m depurple.babi_snapshot --n 100 --fewshot 3   # writes depurple/babi_scenarios.jsonl
    STATE_SCEN=depurple/babi_scenarios.jsonl python -m depurple.state_probe gen base
    STATE_SCEN=depurple/babi_scenarios.jsonl python -m depurple.state_probe gen trial12
    STATE_SCEN=depurple/babi_scenarios.jsonl python -m depurple.state_probe grade base trial12
    python -m depurple.babi_snapshot --demo               # conversion self-check, no network

This is a data snapshot, not a benchmark framework; grading stays in state_probe.
Comma and multi-word answers are dropped because the grader expects a single
word-boundary match.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("depurple/babi_scenarios.jsonl")
SYSTEM = "Read the statements, then answer the question with a single word."
N = 100


def to_scenario(r: dict, idx: int, primer: str = "") -> dict:
    """One babi row -> one single-turn scenario graded on the answer word. `primer` prepends any
    few-shot solved examples in front of the question."""
    user = primer + f"{r['passage'].strip()}\n{r['question'].strip()}"
    return {"id": f"babi_t{r['task']}_{idx}", "system": SYSTEM,
            "user_turns": [user], "checks": [{"expect": [r["answer"].lower()]}]}


def _single_word(ans: str) -> bool:
    # word-boundary containment can't honestly match "milk,football" or "n,e"; drop them.
    return " " not in ans and "," not in ans


def _primer(exemplars: list[dict]) -> str:
    if not exemplars:
        return ""
    blocks = [f"{r['passage'].strip()}\n{r['question'].strip()}\nAnswer: {r['answer']}" for r in exemplars]
    return "Solved examples:\n\n" + "\n\n".join(blocks) + "\n\nNow answer this question:\n"


def build(rows: list[dict], n: int = N, fewshot: int = 0) -> list[dict]:
    """Round-robin across task ids (1..20) for reasoning variety, deterministic order. The first
    `fewshot` single-word rows become baked-in exemplars (held out of the test set)."""
    singles = [r for r in rows if _single_word(r["answer"])]
    primer = _primer(singles[:fewshot])
    by_task: dict[int, list[dict]] = {}
    for r in singles[fewshot:]:
        by_task.setdefault(r["task"], []).append(r)
    tasks = sorted(by_task)
    out: list[dict] = []
    depth = 0
    while len(out) < n and any(depth < len(by_task[t]) for t in tasks):
        for t in tasks:
            if depth < len(by_task[t]):
                out.append(to_scenario(by_task[t][depth], len(out), primer))
                if len(out) >= n:
                    break
        depth += 1
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N, help="number of test scenarios")
    ap.add_argument("--fewshot", type=int, default=0, help="solved exemplars prepended to each prompt")
    args = ap.parse_args()
    from datasets import load_dataset
    ds = load_dataset("Muennighoff/babi", split="validation")
    scens = build([ds[i] for i in range(len(ds))], n=args.n, fewshot=args.fewshot)
    OUT.write_text("".join(json.dumps(s) + "\n" for s in scens))
    print(f"wrote {OUT} ({len(scens)} scenarios across "
          f"{len({s['id'].split('_')[1] for s in scens})} babi task types, fewshot={args.fewshot})")


def demo() -> None:
    """Self-check conversion and filtering without model or network calls."""
    fake = [
        {"passage": "Daniel went to the kitchen.\nDaniel went to the bedroom.\n",
         "question": "Where is Daniel?", "answer": "bedroom", "task": 1},
        {"passage": "p", "question": "q", "answer": "milk,football", "task": 8},  # dropped
        {"passage": "p", "question": "q", "answer": "yes", "task": 6},
    ]
    out = build(fake, n=10)
    assert len(out) == 2, out                                   # comma answer filtered
    s = out[0]
    assert s["user_turns"][0].endswith("Where is Daniel?")      # passage + question joined
    assert s["checks"] == [{"expect": ["bedroom"]}]             # answer lowercased into expect
    assert len(s["checks"]) == len(s["user_turns"])             # state_probe's load assert
    assert s["id"].startswith("babi_t1_")
    # few-shot: first single-word row becomes an exemplar (held out), prepended as a primer
    fs = build(fake, n=10, fewshot=1)
    assert len(fs) == 1, fs                                      # 2 singles - 1 exemplar = 1 test row
    assert fs[0]["user_turns"][0].startswith("Solved examples:")
    assert "Answer: bedroom" in fs[0]["user_turns"][0]          # the exemplar's answer is shown
    # grades through the real probe grader, space/case-insensitively (the whole point)
    from depurple.state_probe import grade
    assert grade("Bedroom.", s["checks"][0])                    # chat model's spaced/capitalized reply
    assert not grade("The kitchen, I think.", s["checks"][0])
    print("babi_snapshot demo ok")


if __name__ == "__main__":
    (demo if "--demo" in sys.argv else main)()
