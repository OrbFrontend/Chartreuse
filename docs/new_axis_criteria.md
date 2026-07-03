# When to add a new axis (vs. more data on an existing one)

`DEPURPLE_AXIS` is cheap to *name* and expensive to *carry*. Every axis added to a joint
run is a full classifier resident in GPU memory for the rest of that run's life, its own
rollout per trial, and its own 7-param block in the Optuna search space. This doc is the
gate before reaching for a new axis directory.

All numbers below are from the gemma-4-E4B-it runs on this box:
`optimize-gemma-4-e4b-it-purple-euphemism.log.bk` (joint, 97 trials, 2026-06-30 22:36 →
2026-07-01 13:24, ~14h48m) vs. the single-axis logs in `depurple/` (20 trials each,
~1h–2h).

## The cost, with numbers

- **Search space**: each axis contributes its own 7-param kernel block —
  `{per_layer, dir_layer, max_weight, max_pos, min_weight, min_dist, mlp_scale}`,
  namespaced `<axis>_*` (see `_kernel_distributions`, `depurple/optimize.py:231`). Two
  axes = 14 params. TPE's sample efficiency degrades with dimension, so a joint run
  needs *more* trials per axis than the axes would alone, not the same number split.
- **Wall clock per trial adds**: a joint trial runs one full rollout + classifier
  scoring pass *per axis*. Measured: purple single-axis ≈ 5.5 min/trial, euphemism ≈
  3.3 min/trial, joint ≈ 9.2 min/trial — almost exactly the sum. Combined with the
  extra trials the bigger space needs, the purple+euphemism run took 97 trials /
  ~14h48m against a single-axis run's 20 trials / 1–2h. Budget a third axis at
  (sum of per-trial costs) × (even more trials), i.e. multi-day, before starting.
- **Memory**: a joint run holds one classifier scorer per axis resident at once
  (`objective.py:57` — `_scorers` keyed by classifier dir). Two fp32 ettin-400m
  (~1.6GB each) already starved Gemma-4E's generate activations on a 24GB card and
  forced the scorers down to bf16. A third axis is the next squeeze: smaller
  `ROLLOUT_BATCH`, or an OOM mid-run.
- **Every refinement decision reruns per axis.** `_auto_project`, `norm_preserve_for`,
  the Stage-0 diagnostic in `docs/ablation_refinements_plan.md` — each existing axis is
  a fixed cost paid once; each new axis re-triggers all of it, plus its own
  `eval_scenarios-<axis>.jsonl`, kill-test, and eyeball read.

## Stability: joint runs EARLY OUT much more, by design

The `EARLY OUT ppl>=6.0x` saturated rejects (`optimize.py:405`) are the visible
instability of a joint run. The measured rates:

| run | EARLY OUT / trials |
|---|---|
| purple single-axis (lengthproj, normpres, lengthproj-normpres) | 0 / 20 each |
| euphemism single-axis (raw, normpres) | 3 / 20 each |
| purple+euphemism joint (.bk) | 9 / 97 — **8 of them in the first 48** (trials 0, 4, 5, 9, 24, 39, 44, 47, 58) |
| purple+euphemism joint (current run) | 3 of the first 6, same pattern |

Why it scales badly with axis count:

- **The safe region is an intersection.** Each axis's spec independently damages
  `o_proj`/`down_proj`; the ppl-safe region of the joint space is roughly the
  intersection of the per-axis safe regions, so a random point is safe with roughly the
  *product* of the per-axis probabilities. TPE's first ~10 trials are pure random
  samples (its startup phase), which is exactly where the early-outs cluster — half of
  the current run's first 6 trials, 8 of the .bk run's first 48, then only 1 in the
  back 49 once the sampler learned the boundary.
- **Saturated rejects are a flat plateau.** Every ppl-saturated trial returns the same
  constant (`_saturated_reject`, `optimize.py:103`), so TPE learns only "avoid this
  region", never a gradient within it. Correct (the region genuinely loses regardless)
  but each one is a burned slot in `--trials`: the *effective* trial count is
  trials − early-outs.
- **The mid-rollout brain-damage abort gets more conservative per axis.** Its credit is
  the most *joint* primary a trial could recover (`joint_primary_floor` +
  `base_obj`, both sums over axes — `optimize.py:383`). More axes → bigger credit →
  the expensive abort fires later, so a doomed-but-not-ppl-broken trial runs more of
  its full multi-rollout cost before being caught.

What is *not* a worry: the pre-rollout EARLY OUT itself is nearly free — it fires after
the plain-prose ppl check, before any rollout (trials 4 and 5 of the current run
finished 3 seconds apart). A high early-out rate in a joint run's first ~10–50 trials
is the sampler mapping the joint safe region, not a broken run. Worry if saturated
rejects are still frequent past trial ~60 (the .bk run was clean from 59 on) — that
suggests the safe region is too thin for the search space, and a third axis makes it
thinner still.

None of this is a reason to never add an axis — it's the reason to be sure the thing
you want isn't already representable as *data* on `purple` or `euphemism`.

## Gate: is it actually a new axis?

A new axis is justified only when the answer is **yes** to all three:

1. **Different direction, not different degree.** The residual direction that separates
   good/bad has to point somewhere genuinely different in activation space — not just
   further along an axis you already have. "More purple" or "more euphemistic" is data,
   not an axis. "Bad" for a reason a purple/euphemism classifier structurally can't
   see — a different semantic contrast entirely — is a candidate axis. This is
   mechanically checkable: build ~30+ neg/pos pairs and run
   `python -m depurple.axis_compat cand_pairs.jsonl` — it extracts the candidate
   direction the same way `direction.py` does and cosines it against every saved axis
   (with a split-half reliability gate so noisy pairs can't fake "distinct"). For
   reference, purple-vs-euphemism raw directions measure ~0.1 mid-stack (distinct);
   a candidate at ≥0.6 with an existing axis is data for that axis, not a new one.
2. **A single sentence-level P(label) score means something.** The classifier contract
   (`README.md`) requires the row schema `{"text", "label", ...}` to be judged one
   sentence at a time. If the property you're gating needs paragraph- or
   conversation-level context to label, it doesn't fit this pipeline's classifier *or*
   its depurple objective (which scores rollouts sentence-by-sentence) — that's a
   different tool, not a new axis here.
3. **You've tried it as negatives/positives on an existing axis first, and it didn't
   separate.** Concretely: take ~30–50 examples of the new failure mode, hand-label
   them 0/1 against the *existing* axis's contract, and check whether the existing
   classifier already scores them sensibly (or would, with more of that data in
   `curated.jsonl`). If existing axis + more data gets you 80% of the way, that's
   cheaper than a new axis forever paying the joint-run tax above.

If any answer is no, it's a data problem: add rows to `data/<axis>/curated.jsonl` or
extend `datagen/generate_synthetic.py`'s seeds for the axis that already exists, then
`python -m datagen.build_dataset` and retrain.

## If you do add one

- New axis = new `data/<new_axis>/` tree (mirror `purple/` or `euphemism/`'s layout),
  its own `curated.jsonl`, its own classifier (`models/<encoder-slug>-<new_axis>`), its
  own `depurple/eval_scenarios-<new_axis>.jsonl`.
- Run it **single-axis first** end to end (direction → optimize → eyeball) before ever
  putting it in a joint `DEPURPLE_AXIS=a,b,c` run. A single-axis run is the only way to
  learn the axis's own `_auto_project`/`norm_preserve` decision and get a clean
  baseline objective value — joint runs conflate axes in every collapse/coherence
  penalty (see any joint trial's penalty breakdown: `euphemism_collapse` and
  `euphemism_floor` dominate whether or not purple's edit caused them).
- The joint objective is the axis-weighted sum of per-axis primaries; the weights come
  from `DEPURPLE_AXIS_WEIGHTS` (default 1.0 each — `_model.axis_weights`). If one axis
  matters more, set the weights rather than starving the other axis of trials.
- Budget trials for the early-out burn: expect roughly half the TPE startup trials to
  saturate-reject (see the stability section), and size `--trials` so the *effective*
  count after early-outs still covers the bigger space. The purple+euphemism run
  needed ~97; three axes will need more, overnight-plus, and `--resume-from` the log
  rather than restarting after a crash.
