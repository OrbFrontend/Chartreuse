"""Optuna/TPE search for depurple ablation parameters.

Minimizes the active style objective over multi-turn rollouts, subject to
capability and repetition guards.

One model is loaded once; weights are snapshot/restored between trials. The best
params are re-applied and the edited model saved.

    python depurple/optimize.py --trials 30          # fresh search
    python -m depurple.optimize  --trials 40         # resume: replay the log, run the rest

Resume: every completed trial is mirrored to LOG (depurple/optimize-<slug>.log) in
Optuna's own format. A re-run replays those trials with study.add_trial() to warm-start
the TPE sampler, then runs only the remainder up to --trials -- so a crash at trial 19/40
costs the rest, not the whole search. --fresh ignores the log and starts at trial 0.
The resumed sampler re-seeds from replayed history, so trials after the resume
point may differ from an uninterrupted run while still using valid TPE sampling.
"""
from __future__ import annotations

import argparse
import ast
import logging
import re
import time
from pathlib import Path

import optuna
import torch
from optuna.distributions import (CategoricalDistribution, FloatDistribution,
                                  IntDistribution)
from optuna.trial import TrialState, create_trial
from transformers import AutoModelForCausalLM, AutoTokenizer

from depurple._model import (LOG, OUT_DIR, AXIS, AXES, MULTI, NORM_PRESERVE, norm_preserve_for,
                             axis_weights, classifier_dir, load_directions)
from depurple.ablate import (AblationSpec, apply_ablation, build_kernel, decoder_layers,
                            restore, snapshot)
from depurple.objective import (BETA, GEN_MODEL, RolloutAborted, evaluate, euph_scores,
                                gen_perplexity, intimate_replies, intra_repeat,
                                late_weighted, load_scenarios, opener_repeat, perplexity, struct_frac,
                                rollout_all, two_sided)
from depurple.repetition import audit_density, coverage

PPL_SLACK = 0.15        # plain-prose perplexity free zone: no penalty up to +15% vs base
PPL_LAMBDA = 1.0        # soft penalty per unit of (plain-prose ppl / base - 1) ABOVE the slack.
                        # Replaces an old hard reject (return base_obj + ppl-ceiling): that cliff
                        # threw away all style signal on any over-ceiling trial and fed TPE raw-ppl-
                        # scale values (10^3-10^9), so a deeper model whose every edit grazes the
                        # ceiling (gemma 31b) never produced a real score. A strong style edit
                        # legitimately costs some control-corpus drift, so trade it on a slope
                        # (heretic-style soft KL), capped like gen_ppl below.
PPL_CAP = 6.0           # ratio ceiling so ppl=inf can't produce an inf/NaN objective (mirrors GEN_PPL_CAP)
# Repetition penalties are base-anchored AND superlinear: penalty = LAMBDA*e + QUAD*e**2 on the
# excess e = (trial_rate - base_rate), and 0 at/below base (see _excess_penalty). The quadratic is
# the fix for the "lowest-scoring trial still has a repetitive opener" failure — a near-base wobble
# stays cheap, but a real collapse (one opener/word locked across a whole reply, which also tanks
# the babi benchmark) costs far more than any purple drop can buy back. Opener + intra (the staccato
# and word-fixation collapse modes) carry the steepest quadratics.
OPENER_LAMBDA = 2.0     # opener-repeat (linear) per unit ABOVE the base model's rate
OPENER_QUAD = 8.0       # opener-repeat (quadratic): severe opener collapse dominates the objective
STRUCT_LAMBDA = 2.0     # markdown/list lines in narrative rollouts (objective.struct_frac): the
STRUCT_QUAD = 8.0       # report-mode collapse scores LOW P(purple), so the classifier alone
                        # actively selects for it (the len+np A/B cell) — superlinear like opener
AUDIT_LAMBDA = 1.0      # Orb repetition-density (linear) per unit ABOVE base (tunable knob)
AUDIT_QUAD = 3.0        # Orb repetition-density (quadratic)
COLLAPSE_LAMBDA = 2.0   # penalty per unit of sentence shortfall vs base (prose-collapse guard)
DIRECT_FLOOR_LAMBDA = 2.0   # euphemism only: penalty per unit of direct-density shortfall below
                            # base — the hard fade-to-black ratchet on top of the two-sided drive
# Brain-damage guard: penalty per unit of (gen-ppl-under-base / base-gen-ppl - 1). The edit's OWN
# rollouts scored under the un-edited base weights — spikes on word-spam ("ass ass ass") AND
# incoherence, both invisible to the four guards above (see objective.gen_perplexity). CAP bounds
# the ratio so a total collapse (gen_ppl=inf) can't produce an inf/NaN objective.
COHERENCE_LAMBDA = 1.0  # gen-ppl-under-base ratio penalty (untuned default; catches incoherence)
GEN_PPL_SLACK = 0.15    # voice-shift free zone (mirrors PPL_SLACK): a healthy restyled edit reads
                        # 1.0-1.14x to base while damage reads 1.6-5.4x, so charging from exactly
                        # 1.0x only taxed the intended edit and biased TPE toward base-like output
GEN_PPL_CAP = 6.0       # ratio ceiling: output >=6x more perplexing than base gets the max penalty
INTRA_LAMBDA = 3.0      # within-reply repetition/fixation (linear) ABOVE base — the
                        # 'My ass is X. My ass is Y.' / 'ass' x9 collapse the other guards miss
INTRA_QUAD = 8.0        # within-reply repetition/fixation (quadratic): word-fixation dominates
EARLY_OUT_RATIO = PPL_CAP   # skip the (~20-min) rollout once plain-prose ppl is this many x over
                            # base. At PPL_CAP the ppl penalty is already pinned at its max, so the
                            # rollout can't change the ranking, and a model this brain-damaged
                            # generates garbage anyway -> it loses regardless. Lower this to skip
                            # MORE trials, but anything below PPL_CAP re-introduces the cliff the
                            # soft penalty removed (it discards the style signal of edits in the
                            # [N, cap] ppl zone). Caveat: a cheap PRE-rollout gate can only catch
                            # plain-prose-broken trials; an edit that READS fine but GENERATES
                            # garbage (high gen_ppl/intra) is only visible after the rollout.


BRAIN_DAMAGE_MARGIN = 0.05  # slack above the (already-conservative) dilution bound before the
                            # MID-rollout brain-damage abort fires (see _brain_damage_reject). The
                            # pre-rollout EARLY_OUT above only sees plain-prose ppl; an edit that
                            # READS fine but GENERATES garbage (looping/word-fixation/opener collapse
                            # — the dominant de-purple failure) is invisible until after the ~20-min
                            # rollout. With ROLLOUT_BATCH < bait count the bait rollout runs in >1
                            # pass, so we can score the model-free structural guards on the partial
                            # replies after pass 1 and skip pass 2 (+ collapse + classifier + gen_ppl)
                            # for a clearly-doomed trial. Granularity == number of batches: at
                            # ROLLOUT_BATCH >= bait count there's a single bait pass and this can only
                            # fire after the whole bait rollout (still ahead of the rest). gen_ppl is
                            # NOT usable here (it needs base weights restored, discarding the edited
                            # weights pass 2 still needs) -> the structural guards are the stand-in.


def _saturated_reject(base_obj: float) -> float:
    """Objective for a trial whose plain-prose ppl already saturates the ppl penalty: assume the
    worst-case ppl + coherence penalties (the rollout is skipped, so we don't measure the rest)
    on top of base. Reliably worse than any real non-saturated trial, so TPE buries the region
    without paying the rollout."""
    return base_obj + PPL_LAMBDA * (PPL_CAP - (1 + PPL_SLACK)) + COHERENCE_LAMBDA * (GEN_PPL_CAP - (1 + GEN_PPL_SLACK))


def _excess_penalty(lin: float, quad: float, value: float, base: float) -> float:
    """Base-anchored SUPERLINEAR penalty: 0 at/below base, else lin*e + quad*e**2 with e=value-base.
    The quadratic makes a large excess cost disproportionately more than a small one, so a real
    repetition collapse can't be traded away against a big primary (purple) drop the way a flat
    linear penalty could."""
    e = max(0.0, value - base)
    return lin * e + quad * e * e


def _brain_damage_reject(replies_by_scen, base: dict, base_obj: float, total_replies: int,
                         primary_floor: float = 0.0) -> float | None:
    """MID-rollout brain-damage early-out, called per batch on the PARTIAL replies. Returns a
    reject value to ABORT the rollout, or None to keep going. Provably no false abort: the
    not-yet-generated replies are assumed PERFECTLY clean (zero excess), so each partial
    structural rate is diluted by frac = n_done/total -> a LOWER bound on the final rate,
    hence on the final structural penalty; and the trial is credited the MOST primary it
    could ever recover (base_obj - primary_floor; primary_floor=0 for purple). The abort fires
    only when that lower-bound penalty STILL exceeds the max recoverable credit + the margin —
    so even a perfect pass 2 (primary driven to its floor) and perfectly clean remaining
    replies can't make the trial beat doing nothing. `base` carries the unedited repeat/audit/
    intra/struct anchors; the guards mirror _rep_components (opener/audit/intra/struct, superlinear)."""
    flat = [r for scen in replies_by_scen for r in scen]
    n_done = len(flat)
    if n_done == 0 or n_done >= total_replies:
        return None                          # nothing measured yet, or last batch (no pass to skip)
    frac = n_done / total_replies            # remaining assumed zero-excess -> scale the partial rate down
    done = [scen for scen in replies_by_scen if scen]
    comps = {
        "opener": _excess_penalty(OPENER_LAMBDA, OPENER_QUAD, opener_repeat(flat) * frac, base["repeat"]),
        "audit":  _excess_penalty(AUDIT_LAMBDA, AUDIT_QUAD, audit_density(done) * frac, base["audit"]),
        "intra":  _excess_penalty(INTRA_LAMBDA, INTRA_QUAD, intra_repeat(flat) * frac, base["intra"]),
        "struct": _excess_penalty(STRUCT_LAMBDA, STRUCT_QUAD, struct_frac(flat) * frac, base["struct"]),
    }
    penalty_floor = sum(comps.values())
    if penalty_floor > (base_obj - primary_floor) + BRAIN_DAMAGE_MARGIN:
        return base_obj + penalty_floor      # honest, > base_obj -> reliably buried by TPE
    return None


def _axis_primary_floor(axis: str) -> float:
    """Provably-safe LOWER bound on an axis's (minimized) primary, for the brain-damage
    credit. purple: late_weighted P(purple) >= 0. euphemism: two_sided = P(euph) - BETA*direct
    >= 0 - BETA*1 (direct_density is a per-word fraction <= 1). A loose bound only inflates the
    credit -> makes the abort MORE conservative, never less safe."""
    return -BETA if axis == "euphemism" else 0.0


def _rep_components(repeat: float, audit: float, intra: float, struct: float, cov: int,
                    base: dict, floor: float = 0.0) -> dict[str, float]:
    """The repetition/collapse penalty for one axis split by NAMED guard (0.0 = that guard
    didn't fire). Base-anchored, shared by the single-axis and joint objectives so their
    severity can't drift. Each repetition signal fires only on the excess ABOVE the base
    model's own rate; OPENER/AUDIT/INTRA are superlinear (a near-base wobble is cheap, a
    locked opener/word across a whole reply is prohibitive — the failure where the
    lowest-scoring trial still read 'She... She... She...' and tanked babi). COLLAPSE
    (writing LESS than base) and the euphemism fade-to-black `floor` stay linear. `base`
    carries keys repeat/audit/intra/cov. The objective sums these; the per-guard split is
    what the per-trial log prints so you can see WHICH guard fired."""
    return {
        "opener":   _excess_penalty(OPENER_LAMBDA, OPENER_QUAD, repeat, base["repeat"]),
        "audit":    _excess_penalty(AUDIT_LAMBDA, AUDIT_QUAD, audit, base["audit"]),
        "intra":    _excess_penalty(INTRA_LAMBDA, INTRA_QUAD, intra, base["intra"]),
        "struct":   _excess_penalty(STRUCT_LAMBDA, STRUCT_QUAD, struct, base["struct"]),
        "collapse": COLLAPSE_LAMBDA * max(0.0, 1.0 - cov / base["cov"] if base["cov"] else 0.0),
        "floor":    floor,
    }


def _coherence_penalty(gen_ppl: float, base_gen_ppl: float) -> float:
    """Brain-damage penalty: the edit's OWN output scored under base weights, ratio-capped."""
    return COHERENCE_LAMBDA * max(0.0, min(gen_ppl / base_gen_ppl, GEN_PPL_CAP) - (1 + GEN_PPL_SLACK))


def _ppl_penalty(ppl: float, base_ppl: float) -> float:
    """Plain-prose capability penalty: held-out-prose ppl drift above the free zone, capped."""
    return PPL_LAMBDA * max(0.0, min(ppl / base_ppl, PPL_CAP) - (1 + PPL_SLACK))


# Per-trial breakdown goes through Optuna's own logger, so it reaches the same console AND
# the .log file handler _attach_log_file installs — without disturbing resume (replay_log only
# matches the "Trial N finished" lines Optuna itself emits, never these).
_OPTUNA_LOG = logging.getLogger("optuna")


def _fmt_comps(comps: dict[str, float]) -> str:
    """Only the guards that actually fired, biggest first — '[]' when the edit is penalty-free."""
    fired = sorted(((k, v) for k, v in comps.items() if v > 1e-4), key=lambda kv: -kv[1])
    return "[" + " ".join(f"{k} {v:.3f}" for k, v in fired) + "]"


def _log_trial(trial: optuna.Trial, primary: float, comps: dict[str, float],
               ppl: float, base_ppl: float, gen_ppl: float, base_gen_ppl: float) -> None:
    """Decompose one finished trial's objective into primary (the style scalar being driven
    down) vs the total penalty and which guards paid it, plus the two capability ratios. This
    is the line that tells you a low score came from killing purple, not from gaming a guard."""
    pen = sum(comps.values())
    _OPTUNA_LOG.info(
        f"  trial {trial.number}: obj={primary + pen:+.3f} = primary {primary:+.3f} + pen {pen:.3f} "
        f"{_fmt_comps(comps)}  ppl={ppl / base_ppl:.2f}x gen_ppl={gen_ppl / base_gen_ppl:.2f}x")


def _layer_window(n_layers: int, axis: str = AXIS) -> tuple[int, int]:
    """dir_layer search bounds for one axis. Purple: mid-stack (15-85%) — style lives there
    and the early surface-lexical layers let the optimizer Goodhart the metric. Euphemism: do
    not assume mid-stack; widen to near-full depth and let the search find where the
    euphemistic register lives."""
    if axis == "euphemism":
        return int(0.05 * n_layers), int(0.95 * n_layers)
    return int(0.15 * n_layers), int(0.85 * n_layers)


def _flat(replies_by_scen):
    return [r for scen in replies_by_scen for r in scen]


# Same line Optuna emits per completed trial (mirrors serve.resolve_params' parser):
#   "Trial 12 finished with value: 0.358... and parameters: {'per_layer': True, ...}."
_TRIAL_RE = re.compile(r"Trial \d+ finished with value:\s*(\S+)\s+and parameters:\s*(\{.*?\})")


def _kernel_distributions(n_layers: int, axis: str = AXIS, prefix: str = "") -> dict:
    """The exact search space one axis's kernel samples -- needed to rebuild logged trials
    as FrozenTrials whose distributions match (Optuna rejects a mismatch). dir_layer's
    range tracks the per-axis window the objective restricts to (_layer_window). `prefix`
    namespaces the param keys for a joint run (e.g. 'euphemism_max_weight')."""
    lo, hi = _layer_window(n_layers, axis)
    return {
        f"{prefix}per_layer": CategoricalDistribution([True, False]),
        f"{prefix}dir_layer": IntDistribution(lo, hi),
        f"{prefix}max_weight": FloatDistribution(0.2, 1.0),
        f"{prefix}max_pos": FloatDistribution(0.2, 0.9),
        f"{prefix}min_weight": FloatDistribution(0.0, 0.4),
        f"{prefix}min_dist": FloatDistribution(0.1, 1.0),
        f"{prefix}mlp_scale": FloatDistribution(0.0, 1.0),
    }


def _joint_distributions(n_layers: int) -> dict:
    """Union of every axis's kernel search space, each namespaced by '<axis>_' — the joint
    objective samples one kernel per axis under these keys."""
    dists: dict = {}
    for a in AXES:
        dists.update(_kernel_distributions(n_layers, a, prefix=f"{a}_"))
    return dists


def replay_log(study: optuna.Study, path: str, dists: dict) -> int:
    """Warm-start `study` from a prior optimize log: parse every 'Trial N finished'
    line and re-add it via add_trial so the TPE sampler sees the completed trials as
    priors (and the study's trial count advances, so new trials number contiguously).
    Returns how many were replayed. `dists` is the full search space (single- or multi-axis);
    conditional params (no dir_layer when per_layer is True) carry only the distributions
    actually present, matching how Optuna stored them."""
    p = Path(path)
    if not p.exists():
        return 0
    text = p.read_text()
    n = 0
    for value_s, params_s in _TRIAL_RE.findall(text):
        params = ast.literal_eval(params_s)
        study.add_trial(create_trial(
            state=TrialState.COMPLETE,
            value=float(value_s),
            params=params,
            distributions={k: dists[k] for k in params},
        ))
        n += 1
    if n == 0 and text.strip():
        # Non-empty log, nothing parsed -> the regex no longer matches Optuna's
        # trial-finished line (version bump?). Without this, a botched resume is
        # silently indistinguishable from an intended --fresh run.
        print(f"WARNING: {path} is non-empty but 0 trials parsed -- resume found "
              f"nothing to warm-start from (Optuna log-format change?); running from trial 0")
    return n


def _attach_log_file(path: str, fresh: bool) -> None:
    """Mirror Optuna's trial log to `path` in its own default format so the next run can
    replay it. Append by default; `fresh` starts a clean log. Attached AFTER
    replay, so the add_trial()'d trials (already in the file) aren't written back twice.

    With --fresh, a non-empty existing log is renamed to a timestamped .bk first so
    the previous run can still be resumed by pointing --resume-from at the backup."""
    p = Path(path)
    if fresh and p.exists() and p.stat().st_size:
        bk = f"{path}.{time.strftime('%Y%m%d-%H%M%S')}.bk"   # timestamped -> never clobbers an older .bk
        p.rename(bk)
        print(f"--fresh: backed up existing log -> {bk} (resume it with --resume-from {bk})")
    fh = logging.FileHandler(path, mode="w" if fresh else "a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(levelname).1s %(asctime)s] %(message)s"))
    logging.getLogger("optuna").addHandler(fh)


def _axis_scens(axis: str, n_scen: int) -> dict:
    """Scenario sets for one axis, mirroring single-axis main(): euphemism uses every
    scenario (all escalate to intimacy) as the primary 'bait'; purple splits bait_/collapse_."""
    if axis == "euphemism":
        scen = load_scenarios(axis=axis)
        return {"bait": scen[:n_scen] if n_scen else scen, "collapse": []}
    bait = load_scenarios("bait", axis=axis)
    collapse = load_scenarios("collapse", axis=axis)
    if n_scen:
        bait, collapse = bait[:n_scen], collapse[:n_scen]
    return {"bait": bait, "collapse": collapse}


def _eval_axis(model, tok, axis: str, scens: dict, base_direct: float | None = None,
               on_batch=None):
    """One axis's (minimized) primary scalar + its replies (for the structural guards) +
    a fade-to-black floor (euphemism only) + per-trial attrs. Mirrors the single-axis
    branches in main(), but reads THIS axis's classifier and scenarios so a joint trial
    scores each axis with its own scorer. `on_batch` is forwarded to this axis's bait
    rollout for the mid-rollout brain-damage early-out."""
    cls = classifier_dir(axis)
    if axis == "euphemism":
        replies = rollout_all(model, tok, scens["bait"], on_batch=on_batch)
        euph, direct = euph_scores(intimate_replies(scens["bait"], replies), cls_dir=cls)
        floor = DIRECT_FLOOR_LAMBDA * max(0.0, 1.0 - direct / base_direct) if base_direct else 0.0
        return two_sided(euph, direct), replies, floor, {f"{axis}_p_euph": euph, f"{axis}_direct": direct}
    curve, bait_replies = evaluate(model, tok, scens["bait"], cls_dir=cls, on_batch=on_batch)
    _, collapse_replies = evaluate(model, tok, scens["collapse"], cls_dir=cls)
    primary = late_weighted(curve)
    return primary, bait_replies + collapse_replies, 0.0, {f"{axis}_primary": primary}


def run_joint(args) -> None:
    """Bake EVERY axis in AXES into one model in a SINGLE non-stacking ablation pass
    (DEPURPLE_AXIS=purple,euphemism). One Optuna study samples a kernel per axis; the
    objective is the axis-weighted sum of each axis's primary plus that axis's structural
    guards, under one shared perplexity ceiling. Each axis must already have its direction
    file + classifier + eval scenarios built (run depurple.direction / classifier.train per axis)."""
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    n_layers = len(decoder_layers(model))
    dirs = {a: load_directions(a) for a in AXES}      # one [L+1, d] direction per axis
    weights = dict(zip(AXES, axis_weights()))
    scens = {a: _axis_scens(a, args.scenarios) for a in AXES}
    snap = snapshot(model)

    base_ppl = perplexity(model, tok)
    ppl_ceiling = base_ppl * (1 + PPL_SLACK)
    print(f"JOINT axes={AXES} weights={[weights[a] for a in AXES]}  "
          f"ppl={base_ppl:.2f} freezone<={ppl_ceiling:.2f} (soft penalty above)")

    # Per-axis base on the UNEDITED model: primary (headline delta), the direct-density anchor
    # (euphemism fade-to-black floor), and the structural floors the penalties fire above.
    base: dict = {}
    base_obj = 0.0
    base_all_replies: list[str] = []
    for a in AXES:
        primary, replies, _, attrs = _eval_axis(model, tok, a, scens[a])
        base[a] = {"direct": attrs.get(f"{a}_direct"),     # euphemism only; None for purple
                   "repeat": opener_repeat(_flat(replies)),
                   "audit": audit_density(replies),
                   "intra": intra_repeat(_flat(replies)),
                   "struct": struct_frac(_flat(replies)),
                   "cov": coverage(replies)}
        base_obj += weights[a] * primary
        base_all_replies += _flat(replies)
        print(f"  BASE[{a}]: primary={primary:.3f}  repeat={base[a]['repeat']:.3f}  "
              f"audit={base[a]['audit']:.3f}  intra={base[a]['intra']:.3f}  "
              f"struct={base[a]['struct']:.3f}  cov={base[a]['cov']}")
    base_gen_ppl = gen_perplexity(model, tok, base_all_replies)   # base scoring its OWN output: coherence anchor
    print(f"  BASE gen_ppl (own output under base)={base_gen_ppl:.2f}")

    # Mid-rollout brain-damage early-out (joint): credit the trial the MOST weighted primary it
    # could ever recover (lower bound per axis, see _axis_primary_floor); a single axis's diluted
    # structural penalty exceeding that joint credit dooms the whole trial. Per-axis reply totals
    # are the dilution denominators (bait + collapse of that axis).
    joint_primary_floor = sum(weights[a] * _axis_primary_floor(a) for a in AXES)
    total_by_axis = {a: sum(len(s["user_turns"]) for s in scens[a]["bait"] + scens[a]["collapse"])
                     for a in AXES}

    def objective(trial: optuna.Trial) -> float:
        specs = []
        for a in AXES:
            lo, hi = _layer_window(n_layers, a)
            per_layer = trial.suggest_categorical(f"{a}_per_layer", [True, False])
            dir_layer = None if per_layer else trial.suggest_int(f"{a}_dir_layer", lo, hi)
            kernel = build_kernel(
                n_layers,
                trial.suggest_float(f"{a}_max_weight", 0.2, 1.0),
                trial.suggest_float(f"{a}_max_pos", 0.2, 0.9),
                trial.suggest_float(f"{a}_min_weight", 0.0, 0.4),
                trial.suggest_float(f"{a}_min_dist", 0.1, 1.0),
            )
            mlp_scale = trial.suggest_float(f"{a}_mlp_scale", 0.0, 1.0)
            specs.append(AblationSpec(dirs[a], kernel, dir_layer, mlp_scale, norm_preserve_for(a)))
        restore(model, snap)
        apply_ablation(model, specs)   # per-axis norm-preserve rides on each spec; one non-stacking pass
        ppl = perplexity(model, tok)              # plain-prose capability guard -> soft penalty below
        if ppl >= EARLY_OUT_RATIO * base_ppl:     # brain-dead on plain prose -> skip the ~20-min rollout
            trial.set_user_attr("ppl", ppl)
            trial.set_user_attr("early_out", True)
            _OPTUNA_LOG.info(f"  trial {trial.number}: EARLY OUT ppl={ppl / base_ppl:.2f}x "
                             f"(>= {EARLY_OUT_RATIO}x base) -> saturated reject, rollout skipped")
            return _saturated_reject(base_obj)
        weighted_primary = 0.0
        comps: dict[str, float] = {}              # per-axis-prefixed penalty guards + the two global ones
        all_replies: list[str] = []
        try:
            for a in AXES:
                def _early_out(replies_by_scen, a=a):   # this axis's bait batch -> abort if brain-damaged
                    rej = _brain_damage_reject(replies_by_scen, base[a], base_obj,
                                               total_by_axis[a], joint_primary_floor)
                    if rej is not None:
                        _OPTUNA_LOG.info(f"  trial {trial.number}: EARLY OUT (brain damage, {a}) "
                                         f"mid-rollout -> reject {rej:+.3f}, remaining passes skipped")
                        raise RolloutAborted(rej)
                primary, replies, floor, attrs = _eval_axis(model, tok, a, scens[a],
                                                            base[a]["direct"], on_batch=_early_out)
                repeat = opener_repeat(_flat(replies))
                audit = audit_density(replies)
                intra = intra_repeat(_flat(replies))
                struct = struct_frac(_flat(replies))
                cov = coverage(replies)
                for k, v in _rep_components(repeat, audit, intra, struct, cov, base[a], floor).items():
                    comps[f"{a}_{k}"] = v
                weighted_primary += weights[a] * primary
                all_replies += _flat(replies)
                trial.set_user_attr(f"{a}_intra", intra)
                for k, v in attrs.items():
                    trial.set_user_attr(k, v)
        except RolloutAborted as e:                # normal return -> Optuna still logs the trial value+params
            trial.set_user_attr("ppl", ppl)
            trial.set_user_attr("early_out", True)
            return e.value
        restore(model, snap)                       # base weights -> judge the edit's OWN output's coherence
        gen_ppl = gen_perplexity(model, tok, all_replies)
        comps["coher"] = _coherence_penalty(gen_ppl, base_gen_ppl)
        comps["ppl"] = _ppl_penalty(ppl, base_ppl)
        for k, v in comps.items():
            trial.set_user_attr(f"pen_{k}", v)
        trial.set_user_attr("primary", weighted_primary)
        trial.set_user_attr("gen_ppl", gen_ppl)
        trial.set_user_attr("ppl", ppl)
        _log_trial(trial, weighted_primary, comps, ppl, base_ppl, gen_ppl, base_gen_ppl)
        return weighted_primary + sum(comps.values())

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    n_replayed = 0 if args.fresh else replay_log(study, args.resume_from, _joint_distributions(n_layers))
    _attach_log_file(LOG, fresh=args.fresh)
    remaining = max(0, args.trials - len(study.trials))
    if n_replayed:
        print(f"RESUME: replayed {n_replayed} trial(s) from {args.resume_from} "
              f"(best so far {study.best_value:.3f}); running {remaining} more "
              f"to reach {args.trials}")
    if remaining:
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"log already has >= {args.trials} trials; re-applying best, no new search")

    if not study.trials:
        raise SystemExit("no trials to choose from (empty log and nothing to run)")
    bt = study.best_trial
    bp_primary = bt.user_attrs.get("primary")
    print(f"\nBEST trial {bt.number}: joint={study.best_value:+.3f} (base {base_obj:+.3f})")
    if bp_primary is not None:
        comps = {k[len("pen_"):]: v for k, v in bt.user_attrs.items() if k.startswith("pen_")}
        print(f"  primary {bp_primary:+.3f} (base {base_obj:+.3f}, delta {bp_primary - base_obj:+.3f})  "
              f"+ pen {sum(comps.values()):.3f} {_fmt_comps(comps)}")
    print(f"  params={study.best_params}")
    bp = study.best_params
    specs = []
    for a in AXES:
        dir_layer = None if bp[f"{a}_per_layer"] else bp[f"{a}_dir_layer"]
        kernel = build_kernel(n_layers, bp[f"{a}_max_weight"], bp[f"{a}_max_pos"],
                              bp[f"{a}_min_weight"], bp[f"{a}_min_dist"])
        specs.append(AblationSpec(dirs[a], kernel, dir_layer, bp[f"{a}_mlp_scale"], norm_preserve_for(a)))
    restore(model, snap)
    apply_ablation(model, specs)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"saved jointly-edited model -> {OUT_DIR}")


def demo() -> None:
    """No-network self-check for the mid-rollout brain-damage early-out: a clearly
    brain-damaged partial batch aborts, varied prose does not, and dilution (lots of
    not-yet-generated replies) spares a partial that would otherwise look bad."""
    spam = "My ass is exposed. My ass is big. My ass is cold. My ass, my ass, my ass."
    # three DISTINCT replies: repeating one string verbatim is cross-message repetition the
    # Orb auditor correctly flags, which is not what this "varied prose" case is probing
    varied = ["The harbor smelled of salt. Gulls argued over a crate. A ferry pushed off the dock.",
              "Rain ticked against the window while the kettle worked itself into a whistle.",
              "She counted the change twice, then slid the coins back across the counter."]
    # distinct contents so ONLY the struct guard can be what aborts (audit stays near base)
    listy = ["**Assessment:**\n1. **Location:** dead station.\n2. **Resources:** none left.",
             "**Plan:**\n- secure the airlock\n- inventory the med bay\n- restart the generator",
             "# Status\n1. fuel low\n2. radio dead\n3. storm rising\n## Next\n- wait it out"]
    base = {"repeat": 0.1, "audit": 0.1, "intra": 0.1, "struct": 0.02}
    base_obj = 0.4
    # Catastrophic partial (3/4 replies done, all spam) -> abort with a reject worse than base.
    rej = _brain_damage_reject([[spam, spam, spam]], base, base_obj, total_replies=4)
    assert rej is not None and rej > base_obj, "brain-damaged partial must abort with reject > base_obj"
    # Varied prose at the same fill fraction -> no abort.
    assert _brain_damage_reject([varied], base, base_obj, 4) is None, \
        "varied prose must not abort"
    # Same spam, but it's a tiny slice of a long run -> dilution drops it below base -> no abort.
    assert _brain_damage_reject([[spam, spam, spam]], base, base_obj, 1000) is None, \
        "heavily-diluted partial must not abort (provably-safe lower bound)"
    # Report-mode collapse (markdown lists in narrative prose) -> the struct guard aborts it
    # even though nothing repeats (the failure the classifier itself rewards).
    rej = _brain_damage_reject([listy], base, base_obj, total_replies=4)
    assert rej is not None and rej > base_obj, "report-mode partial must abort via the struct guard"
    # Last batch (nothing left to skip) and empty input -> never abort.
    assert _brain_damage_reject([[spam, spam, spam, spam]], base, base_obj, 4) is None, "last batch: no abort"
    assert _brain_damage_reject([[]], base, base_obj, 4) is None, "empty partial: no abort"
    print("optimize brain-damage early-out demo ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30,
                    help="TARGET total trials; on resume only the remainder is run")
    ap.add_argument("--scenarios", type=int, default=0, help="0 = all")
    ap.add_argument("--resume-from", default=LOG,
                    help=f"optimize log to replay completed trials from (default: {LOG})")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing log and start the search at trial 0")
    ap.add_argument("--demo", action="store_true",
                    help="no-network self-check of the mid-rollout brain-damage early-out, then exit")
    args = ap.parse_args()

    if args.demo:
        return demo()

    if MULTI:                               # DEPURPLE_AXIS lists several axes -> bake them jointly
        return run_joint(args)

    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    directions = load_directions()
    n_layers = len(decoder_layers(model))
    scen = load_scenarios()
    if args.scenarios:
        scen = scen[:args.scenarios]
    snap = snapshot(model)

    base_ppl = perplexity(model, tok)
    ppl_ceiling = base_ppl * (1 + PPL_SLACK)
    base_direct = None                      # set on the euphemism axis -> direct-density floor
    if AXIS == "euphemism":
        # Every scenario escalates to intimacy; one rollout feeds both the intimate
        # two-sided primary and the shared repetition/coverage guards. `bait` is just
        # the rollout set name reused by objective() below.
        bait, collapse = scen, []
        base_replies = rollout_all(model, tok, scen)
        base_euph, base_direct = euph_scores(intimate_replies(scen, base_replies))
        base_obj = two_sided(base_euph, base_direct)
        print(f"BASE[euph]: P(euph)={base_euph:.3f}  direct={base_direct:.4f}  "
              f"two_sided={base_obj:.3f}  ppl={base_ppl:.2f}  freezone<={ppl_ceiling:.2f}")
    else:
        # bait scenarios drive the purple scalar; collapse probes only feed the
        # repetition/coverage floors below (they don't escalate purple by design).
        bait = [s for s in scen if s["id"].startswith("bait_")]
        collapse = [s for s in scen if s["id"].startswith("collapse_")]
        base_curve, base_bait_replies = evaluate(model, tok, bait)
        _, base_collapse_replies = evaluate(model, tok, collapse)
        base_replies = base_bait_replies + base_collapse_replies
        base_obj = late_weighted(base_curve)
        print(f"BASE: late-weighted purple={base_obj:.3f}  ppl={base_ppl:.2f}  freezone<={ppl_ceiling:.2f}")
    base_repeat = opener_repeat(_flat(base_replies))     # floor: don't punish natural reuse
    base_audit = audit_density(base_replies)             # floor: only punish repetition the EDIT adds
    base_intra = intra_repeat(_flat(base_replies))       # floor: only punish within-reply spam the EDIT adds
    base_structf = struct_frac(_flat(base_replies))      # floor: only punish list/report-mode the EDIT adds
    base_cov = coverage(base_replies)                    # floor: only punish writing LESS than base
    base_gen_ppl = gen_perplexity(model, tok, _flat(base_replies))   # coherence anchor: base on its OWN output
    # Denominator for the mid-rollout brain-damage dilution: every reply the final structural
    # penalty is averaged over (bait + collapse), so a partial-batch rate is scaled by n_done/total.
    total_replies = sum(len(s["user_turns"]) for s in scen)
    base_struct = {"repeat": base_repeat, "audit": base_audit, "intra": base_intra,
                   "struct": base_structf}
    print(f"  opener_repeat={base_repeat:.3f}  audit_density={base_audit:.3f}  intra_repeat={base_intra:.3f}  "
          f"struct_frac={base_structf:.3f}  coverage={base_cov}  gen_ppl={base_gen_ppl:.2f}")

    def objective(trial: optuna.Trial) -> float:
        per_layer = trial.suggest_categorical("per_layer", [True, False])
        lo, hi = _layer_window(n_layers)            # mid-stack (purple) / near-full (euphemism)
        dir_layer = None if per_layer else trial.suggest_int("dir_layer", lo, hi)
        kernel = build_kernel(
            n_layers,
            max_weight=trial.suggest_float("max_weight", 0.2, 1.0),
            max_pos=trial.suggest_float("max_pos", 0.2, 0.9),
            min_weight=trial.suggest_float("min_weight", 0.0, 0.4),
            min_dist=trial.suggest_float("min_dist", 0.1, 1.0),
        )
        mlp_scale = trial.suggest_float("mlp_scale", 0.0, 1.0)
        restore(model, snap)
        apply_ablation(model, directions, kernel, dir_layer, mlp_scale, norm_preserve=NORM_PRESERVE)
        ppl = perplexity(model, tok)                # plain-prose capability guard -> soft penalty below
        if ppl >= EARLY_OUT_RATIO * base_ppl:       # brain-dead on plain prose -> skip the ~20-min rollout
            trial.set_user_attr("ppl", ppl)
            trial.set_user_attr("early_out", True)
            _OPTUNA_LOG.info(f"  trial {trial.number}: EARLY OUT ppl={ppl / base_ppl:.2f}x "
                             f"(>= {EARLY_OUT_RATIO}x base) -> saturated reject, rollout skipped")
            return _saturated_reject(base_obj)

        def _early_out(replies_by_scen):    # per bait-batch: abort if already brain-damaged (see helper)
            rej = _brain_damage_reject(replies_by_scen, base_struct, base_obj, total_replies)
            if rej is not None:
                _OPTUNA_LOG.info(f"  trial {trial.number}: EARLY OUT (brain damage) mid-rollout "
                                 f"-> reject {rej:+.3f}, remaining passes skipped")
                raise RolloutAborted(rej)

        floor = 0.0
        try:
            if AXIS == "euphemism":
                replies = rollout_all(model, tok, bait)   # single batch in practice -> mid-abort dormant
                euph, direct = euph_scores(intimate_replies(bait, replies))
                primary = two_sided(euph, direct)
                # fade-to-black ratchet: direct density must not fall below base.
                floor = DIRECT_FLOOR_LAMBDA * max(0.0, 1.0 - direct / base_direct) if base_direct else 0.0
                trial.set_user_attr("p_euph", euph)
                trial.set_user_attr("direct_density", direct)
            else:
                curve, bait_replies = evaluate(model, tok, bait, on_batch=_early_out)
                _, collapse_replies = evaluate(model, tok, collapse)
                replies = bait_replies + collapse_replies
                primary = late_weighted(curve)
        except RolloutAborted as e:          # normal return -> Optuna still logs 'Trial N finished with value:'
            trial.set_user_attr("ppl", ppl)
            trial.set_user_attr("early_out", True)
            return e.value
        repeat = opener_repeat(_flat(replies))
        audit = audit_density(replies)
        intra = intra_repeat(_flat(replies))
        struct = struct_frac(_flat(replies))
        cov = coverage(replies)
        # Penalties fire only on degradation BEYOND the base model (see _rep_components):
        #   opener  — hand-rolled opener collapse (owns this axis; Orb's opener detector is off, see repetition._TOGGLES)
        #   audit   — Orb's template/structural/phrase/contrastive repetition the edit adds
        #   intra   — within-reply word/phrase spam + lexical fixation the edit adds (the dominant collapse mode)
        #   struct  — markdown/list report-mode the edit adds (the classifier REWARDS it, so it needs its own guard)
        #   collapse — sentence shortfall vs base; closes the "write nothing -> metric=0" loophole
        #   floor   — euphemism only: direct-density shortfall vs base (fade-to-black)
        #   coher/ppl — the edit's own-output coherence + plain-prose capability ratios (added below)
        # opener/audit/intra/struct are superlinear so a real collapse can't be bought back by purple.
        comps = _rep_components(repeat, audit, intra, struct, cov,
                                {"repeat": base_repeat, "audit": base_audit,
                                 "intra": base_intra, "struct": base_structf,
                                 "cov": base_cov}, floor)
        restore(model, snap)                        # base weights -> judge the edit's OWN output's coherence
        gen_ppl = gen_perplexity(model, tok, _flat(replies))
        comps["coher"] = _coherence_penalty(gen_ppl, base_gen_ppl)
        comps["ppl"] = _ppl_penalty(ppl, base_ppl)
        trial.set_user_attr("ppl", ppl)
        trial.set_user_attr("gen_ppl", gen_ppl)
        trial.set_user_attr("opener_repeat", repeat)
        trial.set_user_attr("audit_density", audit)
        trial.set_user_attr("intra_repeat", intra)
        trial.set_user_attr("struct_frac", struct)
        trial.set_user_attr("coverage", cov)
        trial.set_user_attr("primary", primary)
        for k, v in comps.items():
            trial.set_user_attr(f"pen_{k}", v)
        _log_trial(trial, primary, comps, ppl, base_ppl, gen_ppl, base_gen_ppl)
        return primary + sum(comps.values())

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    n_replayed = 0 if args.fresh else replay_log(study, args.resume_from, _kernel_distributions(n_layers))
    _attach_log_file(LOG, fresh=args.fresh)   # after replay, so replayed trials aren't re-logged
    remaining = max(0, args.trials - len(study.trials))
    if n_replayed:
        print(f"RESUME: replayed {n_replayed} trial(s) from {args.resume_from} "
              f"(best so far {study.best_value:.3f}); running {remaining} more "
              f"to reach {args.trials}")
    if remaining:
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"log already has >= {args.trials} trials; re-applying best, no new search")

    if not study.trials:
        raise SystemExit("no trials to choose from (empty log and nothing to run)")
    bt = study.best_trial
    bp_primary = bt.user_attrs.get("primary")
    print(f"\nBEST trial {bt.number}: {AXIS}={study.best_value:+.3f} (base {base_obj:+.3f})")
    if bp_primary is not None:
        comps = {k[len("pen_"):]: v for k, v in bt.user_attrs.items() if k.startswith("pen_")}
        print(f"  primary {bp_primary:+.3f} (base {base_obj:+.3f}, delta {bp_primary - base_obj:+.3f})  "
              f"+ pen {sum(comps.values()):.3f} {_fmt_comps(comps)}")
    print(f"  params={study.best_params}")
    # re-apply best and save
    bp = study.best_params
    dir_layer = None if bp["per_layer"] else bp["dir_layer"]
    kernel = build_kernel(n_layers, bp["max_weight"], bp["max_pos"], bp["min_weight"], bp["min_dist"])
    restore(model, snap)
    apply_ablation(model, directions, kernel, dir_layer, bp["mlp_scale"], norm_preserve=NORM_PRESERVE)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"saved edited model -> {OUT_DIR}")


if __name__ == "__main__":
    main()
