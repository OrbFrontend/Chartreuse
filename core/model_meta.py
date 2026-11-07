"""What a trained classifier dir says about itself: axis, scoring unit, token budget.

train.py writes `chartreuse.json` beside the weights (quantize.py copies it into the ONNX dir),
so every consumer -- evaluate, predict, app.py -- scores with the unit and max_length the model
was trained at, without needing DEPURPLE_AXIS. A dir with no file is a pre-paragraph sentence
model: unit "sentence", max_length 128.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

NAME = "chartreuse.json"
DEFAULT = {"unit": "sentence", "max_length": 128}


def write(model_dir, axis: str, unit: str, max_length: int) -> None:
    Path(model_dir, NAME).write_text(json.dumps(
        {"axis": axis, "unit": unit, "max_length": max_length}, indent=2) + "\n")


def read(model_dir) -> dict:
    p = Path(model_dir, NAME)
    return {**DEFAULT, **json.loads(p.read_text())} if p.exists() else dict(DEFAULT)


def copy(src_dir, dst_dir) -> None:
    p = Path(src_dir, NAME)
    if p.exists():
        shutil.copy(p, Path(dst_dir, NAME))
