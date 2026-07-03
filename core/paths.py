"""Shared layout for BOTH halves of the repo — the single source of truth that the
classifier (classifier/, datagen/) and the style-edit (depurple/) agree on without one
importing the other. One env var, DEPURPLE_AXIS, roots every corpus path at data/<axis>/
and names the classifier models/<encoder-slug>-<axis>, so the purple and euphemism
pipelines stay fully separate.

DEPURPLE_AXIS and ETTIN_MODEL are REQUIRED (no defaults — fatal if unset); export them
before any datagen/classifier/depurple script. NOTE: this module never reads DEPURPLE_MODEL
— that belongs to depurple/_model.py alone, so classifier/datagen keep a smaller env contract.

DEPURPLE_AXIS may name ONE axis ("purple") or SEVERAL comma-separated ("purple,euphemism")
for a joint depurple run; AXES is the full list, AXIS is the primary (first) one the
single-axis module-level constants key off. classifier/datagen are always single-axis.

Everything runs from the repo root via `python -m`, so the import is uniform everywhere:

    DEPURPLE_AXIS=purple    python -m datagen.build_dataset
    DEPURPLE_AXIS=euphemism python -m classifier.train
"""
from __future__ import annotations

import os
from pathlib import Path


def req(name: str) -> str:
    """Required env var: fatal if unset/empty. No silent defaults — explicit config only."""
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"FATAL: required env var {name} is unset/empty (no default).")
    return v


# DEPURPLE_AXIS may name ONE axis ("euphemism") or SEVERAL, comma-separated
# ("purple,euphemism") for a JOINT depurple run. AXES is the full selection; AXIS is the
# primary (first) one that the single-axis constants below key off, so single-axis runs are
# byte-for-byte unchanged. (Mirrors depurple/_model.py, which now imports these from here.)
AXES = [a.strip() for a in req("DEPURPLE_AXIS").split(",") if a.strip()]
assert AXES, "DEPURPLE_AXIS has no usable axis names (e.g. empty/comma-only)"
AXIS = AXES[0]                                          # primary axis: purple | euphemism | ...
MULTI = len(AXES) > 1

DATA = Path("data") / AXIS
RAW = DATA / "raw"
# generate_synthetic "good" seeds (the clean/anchor side): purple's plain sentences and
# euphemism's real direct prose both live here (raw/good/*.txt), rewritten toward the bad pole.
SEED_DIR = RAW / "good"
WORSE_DIR = RAW / "bad"           # generate_synthetic "improve" seeds (auto, alongside good): bad -> better twins
DOCS_DIR = RAW / "docs"           # import_text source documents
INTERIM = DATA / "interim"        # generator outputs (<axis>_pairs, mined, ...)
CURATED = DATA / "curated.jsonl"
HUMAN = DATA / "human_labeled.jsonl"
LABELED = DATA / "labeled.jsonl"
SPLITS = DATA / "splits"          # build_dataset outputs train/val/calibration/test

# Shared, axis-INDEPENDENT common negatives (label 0). Plain prose is the common
# complement of every style axis (the purple/euphemism directions are near-orthogonal, yet
# both score plain text 0), so this pool is reused across axes. build_dataset pools it for
# whichever DEPURPLE_AXIS is active; per-axis self-filtering is filter_common.py.
COMMON = Path("data") / "_common" / "negatives.jsonl"

# Base encoder + its dir slug. ETTIN_MODEL is the single source of the encoder identity
# (classifier/train.py, classifier/sanity_check.py import it from here); the slug names dir
# artifacts after the actual encoder so models/ettin400m-* stops lying when the model changes.
ETTIN_MODEL = req("ETTIN_MODEL")


def _slug(model: str) -> str:
    # ponytail: ettin-encoder-400m -> ettin400m; basename for any non-ettin model. Tighten
    # if you point ETTIN_MODEL at a path whose basename collides across runs.
    return model.rstrip("/").split("/")[-1].lower().replace("-encoder-", "").replace("-encoder", "")


# ETTIN_SLUG, not SLUG: depurple/_model.py has its OWN SLUG (the gemma slug) and imports the
# encoder slug from here, so the two names must not collide.
ETTIN_SLUG = _slug(ETTIN_MODEL)
SLUG = ETTIN_SLUG                 # backward-compat alias (used by the self-check below)


def classifier_dir(axis: str) -> str:
    """Per-axis scorer dir. The classifier name encodes the axis (purple -> ettin400m-purple),
    so a joint depurple run just loads one scorer per axis. Shared by classifier (writes it)
    and depurple/objective.py (reads it) — the single classifier-dir convention."""
    return f"models/{ETTIN_SLUG}-{axis}"


def pairs_path(axis: str) -> str:
    """Canonical GENERATED matched-pair file — the one generate_synthetic writes and
    build_dataset reads: data/<axis>/interim/<axis>_pairs.jsonl. Every axis emits the same
    unified {neg, pos, group_id, source, real, ...} schema, so both readers are axis-agnostic
    and a new axis needs no path or reader change. For the FULL pair corpus (generated +
    curated) that direction.py trains on, use pair_paths()."""
    return f"data/{axis}/interim/{axis}_pairs.jsonl"


def pair_paths(axis: str) -> list[str]:
    """Every matched-pair corpus for an axis, sorted (deterministic train/test split): the
    generated <axis>_pairs.jsonl PLUS any hand-authored curated_<axis>_pairs.jsonl beside it.
    The glob matches only *_pairs.jsonl, so mining intermediates (neg_candidates, soft_*) and
    .attempted files are excluded. direction.py (and _model._auto_project's confound decision)
    build the style direction from ALL of them — more signal and the curated preferred-rewrite
    pole; generate_synthetic still WRITES only pairs_path(axis). Empty list on a serve-only
    checkout with no interim dir (callers fall back)."""
    return sorted(str(p) for p in (Path("data") / axis / "interim").glob("*_pairs.jsonl"))


# Classifier artifacts for the primary axis (single-axis convenience; the functions above
# serve joint depurple runs that need each axis's dir/pairs explicitly).
CLASSIFIER = Path(classifier_dir(AXIS))                   # train --out, calibrate/evaluate --model
ONNX = Path("models") / f"{ETTIN_SLUG}-{AXIS}-onnx-int8"  # quantize --out, predict default

if __name__ == "__main__":   # slug self-check
    assert _slug("jhu-clsp/ettin-encoder-400m") == "ettin400m"
    assert _slug("jhu-clsp/ettin-encoder-1b") == "ettin1b"
    assert _slug("Local/My-Encoder") == "my"
    print("slug ok:", ETTIN_SLUG)
