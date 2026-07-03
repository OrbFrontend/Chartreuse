"""Base-vs-edited rollout for the best trial, with per-turn score and words-per-sentence
(the staccato-collapse tell). Both single-axis and JOINT runs
(DEPURPLE_AXIS=purple,euphemism) reconstruct the lowest-value trial from the optimize log
and eyeball each axis's own scenarios with each axis's own scorer.

Runs on the GPU:
    python -m depurple.eyeball
    DEPURPLE_AXIS=purple,euphemism python -m depurple.eyeball
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from depurple._model import AXES, LOG, MULTI, NORM_PRESERVE, norm_preserve_for, classifier_dir, load_directions
from depurple.ablate import (AblationSpec, apply_ablation, build_kernel, decoder_layers,
                            restore, snapshot)
from depurple.objective import (GEN_MODEL, intra_repeat, load_scenarios, opener_repeat,
                                rollout, struct_frac, turn_purple)
from text_segmentation import split_sentences

_TRIAL_RE = re.compile(r"Trial \d+ finished with value:\s*(\S+)\s+and parameters:\s*(\{.*?\})")

# strength scales the whole kernel (mirrors serve.py): 0 == base, 1 == the trial as optimized.
STRENGTH = float(os.environ.get("DEPURPLE_STRENGTH", "1.0"))


def _best_params() -> tuple[str, dict]:
    """Lowest-value trial in the optimize log for the current axis(es) — or the specific trial
    in DEPURPLE_TRIAL (eyeball a non-best trial, e.g. to compare two near-tied trials)."""
    text = Path(LOG).read_text() if Path(LOG).exists() else ""
    pick = os.environ.get("DEPURPLE_TRIAL")
    if pick:
        m = re.search(rf"Trial {int(pick)} finished with value:\s*(\S+)\s+and parameters:\s*(\{{.*?\}})", text)
        if not m:
            raise SystemExit(f"trial {pick} not found in {LOG}")
        return m.group(1), ast.literal_eval(m.group(2))
    trials = _TRIAL_RE.findall(text)
    if not trials:
        raise SystemExit(f"no trials in {LOG}; run "
                         f"`DEPURPLE_AXIS={','.join(AXES)} python -m depurple.optimize` first")
    value, params_s = min(trials, key=lambda t: float(t[0]))
    return value, ast.literal_eval(params_s)


def wps(reply: str) -> float:
    sents = split_sentences(reply)
    words = sum(len(s.split()) for s in sents)
    return words / len(sents) if sents else 0.0


def show(model, tok, scen, cls_dir: str | None = None, label: str = "purple") -> dict:
    """Print the rollout and return its summary metrics (for the base-vs-edited flag block)."""
    replies = rollout(model, tok, scen)
    ps = []
    for i, r in enumerate(replies):
        p = turn_purple(r, cls_dir=cls_dir) if cls_dir else turn_purple(r)
        ps.append(p)
        print(f"\n--- turn {i+1}  P({label})={p:.3f}  words/sent={wps(r):.1f}")
        print(f"USER: {scen['user_turns'][i]}")
        print(r)
    return {"P": sum(ps) / len(ps) if ps else 0.0,
            "wps": sum(wps(r) for r in replies) / len(replies) if replies else 0.0,
            "struct": struct_frac(replies),
            "opener": opener_repeat(replies),
            "intra": intra_repeat(replies)}


def red_flags(base: dict, edit: dict, label: str = "purple") -> None:
    """Print advisory flags for mechanical failure modes before the human read.

    Flags cover weak movement, sentence shortening, report-mode structure, and
    repetition. They are advisory only: register drift can score well, so reading
    the prose remains mandatory."""
    print(f"\n--- red flags [{label}]  (base -> edited)")
    print(f"    P={base['P']:.3f}->{edit['P']:.3f}  w/s={base['wps']:.1f}->{edit['wps']:.1f}  "
          f"struct={base['struct']:.3f}->{edit['struct']:.3f}  "
          f"opener={base['opener']:.3f}->{edit['opener']:.3f}  intra={base['intra']:.3f}->{edit['intra']:.3f}")
    flags = []
    if edit["P"] > base["P"] - 0.15 * base["P"]:
        flags.append("WEAK MOVE: classifier barely moved vs base — no-op edit, or a register "
                     "shift the scorer can't see (read for it)")
    if edit["wps"] < base["wps"] - 1.0:
        flags.append("SENTENCES SHORTENED: staccato/choppy register shift "
                     "(the normpres-purple and raw-euphemism damage tell)")
    if edit["struct"] > base["struct"] + 0.05:
        flags.append("REPORT-MODE: markdown/list structure in narrative prose "
                     "(classifier rewards it — low P here is FAKE progress)")
    if edit["opener"] > base["opener"] + 0.10 or edit["intra"] > base["intra"] + 0.10:
        flags.append("REPETITION: opener or within-reply loop above base")
    for f in flags:
        print(f"    !! {f}")
    if not flags:
        print("    (none — still read the prose; register-match has no scalar)")


def _joint_best_specs(n: int) -> list[AblationSpec]:
    """Rebuild the joint edit from the lowest-value trial in the optimize log — one
    AblationSpec per axis (each axis's kernel params are namespaced '<axis>_...')."""
    value, p = _best_params()
    print(f"joint best value={value}  params={p}")
    specs = []
    for a in AXES:
        dir_layer = None if p[f"{a}_per_layer"] else p[f"{a}_dir_layer"]
        kernel = [w * STRENGTH for w in build_kernel(n, p[f"{a}_max_weight"], p[f"{a}_max_pos"],
                                                     p[f"{a}_min_weight"], p[f"{a}_min_dist"])]
        specs.append(AblationSpec(load_directions(a), kernel, dir_layer,
                                  p.get(f"{a}_mlp_scale", 1.0), norm_preserve_for(a)))
    return specs


def _run_joint(model, tok, n: int) -> None:
    specs = _joint_best_specs(n)                            # parse log first: cheap, exits early if empty
    scens = [(a, load_scenarios(axis=a)[2]) for a in AXES]   # eyeball each axis's own scenario
    print("=" * 70, "\nBASE\n", "=" * 70, sep="")
    base_m = {}
    for a, s in scens:
        print(f"\n### axis: {a}")
        base_m[a] = show(model, tok, s, cls_dir=classifier_dir(a), label=a)
    snap = snapshot(model)
    apply_ablation(model, specs)   # per-axis norm-preserve rides on each spec; one pass
    print("\n\n", "=" * 70, "\nEDITED (joint best)\n", "=" * 70, sep="")
    for a, s in scens:
        print(f"\n### axis: {a}")
        edit_m = show(model, tok, s, cls_dir=classifier_dir(a), label=a)
        red_flags(base_m[a], edit_m, label=a)
    restore(model, snap)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    n = len(decoder_layers(model))
    if MULTI:
        return _run_joint(model, tok, n)

    directions = load_directions()
    scen = load_scenarios()[2]

    print("=" * 70, "\nBASE\n", "=" * 70, sep="")
    base_m = show(model, tok, scen)

    value, p = _best_params()
    print(f"best value={value}  params={p}")
    dir_layer = None if p.get("per_layer") else p["dir_layer"]

    snap = snapshot(model)
    kernel = [w * STRENGTH for w in
              build_kernel(n, p["max_weight"], p["max_pos"], p["min_weight"], p["min_dist"])]
    apply_ablation(model, directions, kernel, dir_layer, p.get("mlp_scale", 1.0), norm_preserve=NORM_PRESERVE)

    print("\n\n", "=" * 70, "\nEDITED (best trial)\n", "=" * 70, sep="")
    edit_m = show(model, tok, scen)
    red_flags(base_m, edit_m)
    restore(model, snap)


if __name__ == "__main__":
    main()
