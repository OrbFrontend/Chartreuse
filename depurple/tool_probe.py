"""Tool-calling probe with deterministic base-vs-edited grading.

Tool calling is the cheapest, sharpest lobotomy canary an abliteration edit has: the model
has to pick the right function, fill the right arguments, AND abstain when no tool fits.
That's instruction-following + extraction + judgment in one short, exactly-gradable reply --
when an edit eats brain cells it shows here before it shows in prose. Each scenario is
isolated (its own tool set, one user turn) and graded by regex on gemma's native tool-call
emission (`<|tool_call>call:NAME{arg:"val",...}<tool_call|>`), so a pass/fail is a fact, not
a vibe. Negatives (`tool: null`) catch the OTHER failure mode -- an edit that makes the model
fire a tool at plain chit-chat.

Selection mirrors serve.py / eval.py exactly: `gen <variant> [--strength S]` builds the model
in memory via serve.load (base | best | trialN), so the model you probe is the model you serve.
Each `gen` loads one model, rolls out every scenario greedily, dumps replies, and exits.
`grade` then runs as pure regex and is re-runnable.

    python -m depurple.tool_probe gen base
    python -m depurple.tool_probe gen best                 # or: gen trial7 --strength 0.4
    python -m depurple.tool_probe grade                    # no GPU; reads the two reply dumps
    python -m depurple.tool_probe --demo                   # grader self-check, no model/network

eval_bench.sh drives all three for you: `./depurple/eval_bench.sh tools`.

The delta is the useful signal; absolute call rates depend on the base model's tool habits.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCEN = Path("depurple/tool_scenarios.jsonl")
REGRESS = -0.05   # flag an aggregate accuracy drop past this (~1 scenario on this set)

SYSTEM = ("You are a helpful assistant with access to external tools. When a tool can fulfill "
          "the user's request, call it; otherwise just reply normally.")
MAX_NEW = 256     # tool calls are short; plenty of room, and negatives only need the no-call check


def replies_path(variant: str) -> Path:
    # Slug by MODEL: a 'base' dump is the base model's rollouts, so an un-slugged path lets a
    # 31b run grade against a leftover e4b base on a manual `grade base trialN`. Mirrors every
    # other depurple artifact (directions-<slug>, bench-<slug>, ...). See state_probe.replies_path.
    from depurple._model import SLUG
    return Path(f"depurple/tool-replies-{variant}-{SLUG}.jsonl")


# --- grading (pure regex; the markers are special tokens so a real call is unambiguous) ---
# We decode WITH special tokens, so a genuine tool call is the literal <|tool_call>...<tool_call|>
# block -- prose that merely says "call get_weather" can't false-trigger it. Inside, gemma wraps
# string values in the <|"|> token; we drop it so {city:<|"|>Tokyo<|"|>} grades as {city:Tokyo}.
_CALL = re.compile(r"<\|tool_call>\s*call:\s*([A-Za-z0-9_]+)\s*\{(.*?)\}\s*<tool_call\|>", re.S)


def parse_call(reply: str):
    """(name, body) of the first tool call, or None if the reply called no tool."""
    m = _CALL.search(reply)
    if not m:
        return None
    return m.group(1), m.group(2).replace('<|"|>', "")     # strip the string-quote marker token


def grade(reply: str, check: dict) -> bool:
    """True iff the reply did the right thing. tool=null means 'must NOT call a tool'.
    args = key-bound exact-ish values (value tied to its key, word-boundary, case-insensitive);
    contains = substrings that must appear ANYWHERE in the call body (free-text / commutative /
    list args, where the value isn't pinned to one key)."""
    call = parse_call(reply)
    if check.get("tool") is None:               # negative scenario: abstaining is the pass
        return call is None
    if call is None:
        return False
    name, body = call
    if name != check["tool"]:                   # wrong function (covers the disambiguation traps)
        return False
    low = body.lower()
    for k, v in check.get("args", {}).items():
        # value bound to its own key: `key: [opt] value` with a boundary so 15 != 150, celsius is whole
        if not re.search(rf"\b{re.escape(k.lower())}\s*:\s*\[?\s*{re.escape(str(v).lower())}\b", low):
            return False
    for c in check.get("contains", []):
        if str(c).lower() not in low:
            return False
    return True


def load_scenarios() -> list[dict]:
    scens = [json.loads(l) for l in SCEN.read_text().splitlines() if l.strip()]
    for s in scens:                              # authoring guard
        assert "tools" in s and "user" in s and "check" in s, f"{s.get('id')}: missing field"
        assert "tool" in s["check"], f"{s['id']}: check needs a 'tool' (null = must-not-call)"
    return scens


# --- phase 1: generate (one model resident via serve.load, then this process exits) ---

def gen(variant: str, strength: float) -> None:
    import torch
    from depurple import serve                   # reuse variant->in-memory-model selection wholesale

    serve.load(variant, strength)                # sets serve.model / serve.tok (base or ablated)
    scenarios = load_scenarios()
    out = replies_path(variant)
    with torch.inference_mode(), out.open("w") as f:
        for s in scenarios:
            msgs = [{"role": "system", "content": s.get("system", SYSTEM)},
                    {"role": "user", "content": s["user"]}]
            # tools rendered natively by gemma's template; wrap our bare function dicts in the
            # OpenAI {type:function,function:...} envelope the template expects.
            tools = [{"type": "function", "function": fn} for fn in s["tools"]]
            enc = serve.tok.apply_chat_template(
                msgs, tools=tools, add_generation_prompt=True,
                return_tensors="pt", return_dict=True).to(serve.model.device)
            gen_ids = serve.model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                           pad_token_id=serve.tok.eos_token_id)
            # keep special tokens: the <|tool_call> marker is how grade() tells a call from prose
            reply = serve.tok.decode(gen_ids[0, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=False)
            f.write(json.dumps({"id": s["id"], "reply": reply}) + "\n")
    print(f"wrote {out} ({len(scenarios)} scenarios for variant '{variant}')")


# --- phase 2: grade (no model) ---

def _read(variant: str) -> dict[str, str]:
    p = replies_path(variant)
    if not p.exists():
        raise SystemExit(f"missing {p} -- run `python -m depurple.tool_probe gen {variant}` first")
    return {d["id"]: d["reply"] for d in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def do_grade(base_var: str, edited_var: str) -> None:
    scens = load_scenarios()
    base, edited = _read(base_var), _read(edited_var)
    print(f"{'scenario':<22}{'base':>6}{'edited':>8}")
    print("-" * 36)
    tb = te = 0
    regressions = []
    for s in scens:
        sid = s["id"]
        bok = grade(base.get(sid, ""), s["check"])
        eok = grade(edited.get(sid, ""), s["check"])
        tb += bok; te += eok
        print(f"{sid:<22}{('ok' if bok else 'MISS'):>6}{('ok' if eok else 'MISS'):>8}")
        if bok and not eok:
            regressions.append((sid, edited.get(sid, "")))
    n = len(scens)
    ab, ae = tb / n, te / n
    d = ae - ab
    print("-" * 36)
    flag = "  <-- REVIEW: tool calling regressed" if d < REGRESS else ""
    print(f"{'accuracy':<22}{ab:>6.3f}{ae:>8.3f}   ({tb}/{n} -> {te}/{n}, delta {d:+.3f}){flag}")
    if regressions:
        print("\nnewly failed under the edit (eyeball these -- the scalar can't):")
        for sid, er in regressions:
            print(f"  [{sid}]\n    edited> {er[:200]}")
    print("\nOK: no tool-calling drop beyond noise" if not flag
          else "\nthe edit hurt tool calling -- lower --strength or re-pick the trial")


def demo() -> None:
    """Self-check pure grader logic without model or network calls."""
    call = lambda name, body: f'<|tool_call>call:{name}{{{body}}}<tool_call|>'
    # positive: name + key-bound value
    assert grade(call("get_weather", 'city:<|"|>Tokyo<|"|>'), {"tool": "get_weather", "args": {"city": "tokyo"}})
    # wrong value fails
    assert not grade(call("get_weather", 'city:<|"|>Paris<|"|>'), {"tool": "get_weather", "args": {"city": "tokyo"}})
    # int boundary: a 15-check must NOT pass on 150
    assert grade(call("set_timer", "minutes:15"), {"tool": "set_timer", "args": {"minutes": 15}})
    assert not grade(call("set_timer", "minutes:150"), {"tool": "set_timer", "args": {"minutes": 15}})
    # wrong function fails (disambiguation trap)
    assert not grade(call("get_current_weather", "city:london"), {"tool": "get_forecast", "args": {"city": "london"}})
    # contains: every item must be present (order-free), one missing fails
    assert grade(call("add_to_cart", "items:[milk,eggs,bread]"), {"tool": "add_to_cart", "contains": ["milk", "eggs", "bread"]})
    assert not grade(call("add_to_cart", "items:[milk,eggs]"), {"tool": "add_to_cart", "contains": ["milk", "eggs", "bread"]})
    # boolean value
    assert grade(call("set_alarm", "time:<|\"|>7am<|\"|>,recurring:true"), {"tool": "set_alarm", "args": {"recurring": "true"}})
    # negative: abstaining (no call block) is the pass; firing a tool is the miss
    assert grade("Sure, glad I could help -- let me know if you need anything else!", {"tool": None})
    assert not grade(call("set_timer", "minutes:5"), {"tool": None})
    print("tool_probe demo ok")


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
    if cmd == "grade":
        return do_grade(args[1] if len(args) > 1 else "base",
                        args[2] if len(args) > 2 else "best")
    raise SystemExit("usage: tool_probe.py [gen <variant> [--strength S] | grade [base] [edited] | --demo]")


if __name__ == "__main__":
    main()
