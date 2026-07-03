"""Depurple model, axis, and artifact configuration.

The shared configuration lives in core.paths so datagen, classifier, and depurple
agree on axes, classifier directories, pair paths, and encoder slugs. This module
adds the generation-model and ablation-artifact settings depurple needs:

    DEPURPLE_MODEL  -- base model to edit   (REQUIRED, no default; e.g. google/gemma-4-E4B-it)
    DEPURPLE_AXIS   -- style axis to remove  (REQUIRED; read via core.paths; comma-list = joint)

    DEPURPLE_MODEL=google/gemma-4-31B-it python -m depurple.optimize
    DEPURPLE_AXIS=euphemism              python -m depurple.optimize
    DEPURPLE_AXIS=purple,euphemism       python -m depurple.optimize   # JOINT: both, one model

DEPURPLE_AXIS may list several axes (comma-separated) for a joint run: every axis's
direction is baked into ONE model in a single non-stacking pass. The joint artifacts list
all axes (models/<slug>-purple-euphemism-depurpled) so they never collide with a single-axis
run; each axis still needs its own direction file + classifier built beforehand.

Refinement defaults are decided automatically from the axis's pairs data. Purple
uses length projection when its pairs are length-confounded; euphemism keeps the
raw direction and preserves norms when matched pairs do not show that confound.
Artifact names carry the resolved variant tag (for example -lengthproj or
-normpres) so variants never collide; non-purple axes also get an explicit
"-<axis>" suffix.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

# Shared truth — the same module the classifier (classifier/, datagen/) imports. AXES/AXIS/
# MULTI parse DEPURPLE_AXIS; classifier_dir/CLASSIFIER/pairs_path are the dual-classifier and
# matched-pair seams (re-exported so the rest of depurple still imports them from here).
from core.paths import (req, AXES, AXIS, MULTI,  # noqa: F401  (AXES/AXIS/MULTI/CLASSIFIER re-exported)
                        classifier_dir, CLASSIFIER, pairs_path, pair_paths)

MODEL = req("DEPURPLE_MODEL")
SLUG = MODEL.rstrip("/").split("/")[-1].lower()        # e.g. gemma-4-e4b-it (NOT the encoder slug)

# Ablation-refinement switches (docs/ablation_refinements_plan.md). Their DEFAULTS are decided
# PER AXIS from the axis's own pairs data (below), so a bare run (single-axis OR joint) lands on
# the right config for each axis. Env vars still override for experiments.
#   PROJECT (#1) picks the direction refinement and keys the DIRECTION file (direction.py bakes
#   it, load_directions reads it):
#     ""      raw r = mean(purple) - mean(plain)
#     plain   project r orthogonal to the per-layer plain mean (grimjim's #1; sanitizes
#             euphemism into vanilla -> never chosen automatically, env-only experiment)
#     length  project r orthogonal to the empirical per-layer word-count direction (de-confounds
#             a length signal the DATA injected into r -> grounded-but-vivid on purple)
#   NORM_PRESERVE (#2) keys only the ablation kernel (ablate.py column-renorm; PER SPEC, so a
#   joint edit can preserve one axis and not another). Does NOT touch the direction file.
#
# WHY the pairs word-delta decides PROJECT (audit of the 2026-07 A/B, docs/ablation_refinements_plan.md):
# the Stage-0 activation gates CANNOT pick it -- BOTH axes fire corr(words,proj)>0.4 (purple +0.90,
# euphemism -0.71). What separates them is the DATA: r = mu_pos - mu_neg inherits the length
# direction exactly in proportion to the pairs' label<->length MEAN gap. Purple's +10.4-word gap
# means a chunk of r IS the length axis (artifact -> strip it: length-proj won the A/B); euphemism's
# ~0 gap means its length overlap is intrinsic register geometry (crude = punchier prose -> keep it:
# raw won, plain-proj sanitized to vanilla). NORM_PRESERVE couples to that: a raw direction's removal
# costs real write magnitude (restore it: raw+np was the coherent-crude winner); a projected direction
# already removes less, and restoring on top overshoots (len+np was the report-mode Goodhart cell).
_LEN_CONFOUND_WORDS = 2.0   # |mean pos-neg word delta| above this = length-confounded pairs
# ponytail: decision rule validated on n=2 axes (it reproduces both A/B winners); a 3rd axis that
# disagrees with its eyeball gets the env override + this table revisited.
_NO_DATA_FALLBACK = {"purple": ("length", False), "euphemism": ("", True)}  # (PROJECT, NORM_PRESERVE)


@lru_cache(maxsize=None)
def _auto_project(axis: str) -> str | None:
    """Data-level confound test: mean pos-neg word delta over the axis's FULL pair corpus
    (generated + curated, same rows direction.py trains on). None when no pair file is on this
    box (e.g. a serve-only checkout) -> caller uses the fallback."""
    deltas = []
    for p in pair_paths(axis):
        with open(p) as f:
            deltas += [len(r["pos"].split()) - len(r["neg"].split()) for r in map(json.loads, f)]
    if not deltas:
        return None
    return "length" if abs(sum(deltas) / len(deltas)) > _LEN_CONFOUND_WORDS else ""


def project_for(axis: str) -> str:
    """Direction projection for an axis.

    DEPURPLE_PROJECT overrides all axes. Without an override, the axis's pairs data
    decides the default: length-confounded pairs use length projection, matched
    pairs use the raw direction. If the pairs file is absent, a per-axis fallback
    keeps artifact resolution deterministic. The result keys the direction file."""
    p = os.environ.get("DEPURPLE_PROJECT")
    if p is None:
        p = _auto_project(axis)
    if p is None:
        p = _NO_DATA_FALLBACK.get(axis, ("", False))[0]
    assert p in ("", "plain", "length"), f"DEPURPLE_PROJECT must be ''|plain|length, got {p!r}"
    return p


def norm_preserve_for(axis: str) -> bool:
    """Whether column norms are restored after ablation for an axis.

    DEPURPLE_NORM_PRESERVE overrides all axes. Without an override, raw directions
    preserve norms and projected directions do not. The choice is per AblationSpec,
    so a joint edit can differ by axis."""
    v = os.environ.get("DEPURPLE_NORM_PRESERVE")
    if v is not None:
        return v == "1"
    if os.environ.get("DEPURPLE_PROJECT") is None and _auto_project(axis) is None:
        return _NO_DATA_FALLBACK.get(axis, ("", False))[1]
    return project_for(axis) == ""


PROJECT = project_for(AXIS)              # primary axis (direction.py + single-axis naming)
NORM_PRESERVE = norm_preserve_for(AXIS)  # primary axis (single-axis apply_ablation kwarg + naming)


def _suffix(axis: str) -> str:
    # Purple keeps the legacy un-suffixed names (the whole pipeline + eval_bench.sh were
    # built on them); any other axis gets a "-<axis>" suffix so its artifacts sit beside
    # purple's instead of overwriting them.
    return "" if axis == "purple" else f"-{axis}"


def directions_path(axis: str) -> str:
    # A projected direction lives in a distinct file so #1 vs baseline is an env swap, not a
    # rebuild (the baseline .pt stays put). Projection is PER AXIS -> a joint run loads e.g.
    # purple's -lengthproj file and euphemism's raw file. norm-preserve does NOT touch the file.
    proj = project_for(axis)
    return f"depurple/directions-{SLUG}{_suffix(axis)}{f'-{proj}proj' if proj else ''}.pt"


_AX = _suffix(AXIS)
DIRECTIONS = directions_path(AXIS)      # primary-axis direction (direction.py / serve.py / eyeball.py)
PAIRS = pairs_path(AXIS)                # primary-axis GENERATED pairs (direction.py unions all via pair_paths)

# The edited model + optimize log key off the WHOLE selection so a joint run never collides
# with either single-axis run. Single axis keeps the legacy names exactly (purple un-suffixed);
# a joint run lists every axis: models/<slug>-purple-euphemism-depurpled.
_RUN_AX = _AX if not MULTI else "-" + "-".join(AXES)
# Variant tag keeps a 4-way A/B from colliding. Single-axis: tag the axis's resolved refinement
# (purple -> -lengthproj, euphemism -> -normpres). Joint: the per-axis defaults ARE the canonical
# joint model, so no tag; only an explicit uniform env override tags it (so an override experiment
# can't overwrite the default joint artifacts).
if MULTI:
    _ep, _en = os.environ.get("DEPURPLE_PROJECT"), os.environ.get("DEPURPLE_NORM_PRESERVE")
    _VAR = (f"-{_ep}proj" if _ep else "") + ("-normpres" if _en == "1" else "")
else:
    _VAR = (f"-{PROJECT}proj" if PROJECT else "") + ("-normpres" if NORM_PRESERVE else "")
OUT_DIR = f"models/{SLUG}{_RUN_AX}{_VAR}-depurpled"
LOG = f"depurple/optimize-{SLUG}{_RUN_AX}{_VAR}.log"


def axis_weights() -> list[float]:
    """Per-axis multiplier on each axis's primary objective in a JOINT run (aligned to AXES).
    Default 1.0 each; override with DEPURPLE_AXIS_WEIGHTS=1.0,0.5 to favour one axis."""
    raw = os.environ.get("DEPURPLE_AXIS_WEIGHTS")
    if not raw:
        return [1.0] * len(AXES)
    ws = [float(x) for x in raw.split(",")]
    assert len(ws) == len(AXES), f"DEPURPLE_AXIS_WEIGHTS has {len(ws)} entries, need {len(AXES)} for {AXES}"
    return ws


def load_directions(axis: str | None = None):
    """Load a direction and verify that it matches the requested model and axis.

    axis=None uses the primary axis. Joint runs call this once per axis in AXES."""
    import torch
    axis = axis or AXIS
    path = directions_path(axis)
    d = torch.load(path, weights_only=False)
    got = d.get("model")
    assert got == MODEL, f"{path} was built for model {got!r}, not {MODEL!r}"
    got_axis = d.get("axis", "purple")   # direction files predating the axis field are purple
    assert got_axis == axis, f"{path} was built for axis {got_axis!r}, not {axis!r}"
    return d["direction"]
