"""Purple-prose classifier web app.

Serves a single-page frontend (index.html) and one JSON endpoint: it segments
the pasted text into sentences (text_segmentation.split_sentences), scores every
sentence with the fine-tuned ettin classifier on the GPU in one batched forward
pass, and returns each sentence with P(purple) and the purple/not decision at the
calibrated threshold. Scoring mirrors classifier/evaluate.py's predict_probs so
probabilities match the offline numbers the threshold was calibrated against.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from text_segmentation import split_sentences

MODEL_DIR = os.environ.get("MODEL_DIR", "models/ettin400m-purple")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "100000"))     # trust-boundary guard
MAX_SENTENCES = int(os.environ.get("MAX_SENTENCES", "2000"))  # bound latency/VRAM
HERE = Path(__file__).parent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_tok = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()


def _threshold() -> float:
    # calibrate.py/evaluate.py keep threshold.json INSIDE the model dir; prefer that
    # (per-axis). Fall back to the legacy parent-dir copy (purple's models/threshold.json),
    # then 0.5 if neither exists.
    for p in (Path(MODEL_DIR) / "threshold.json",
              Path(MODEL_DIR).resolve().parent / "threshold.json"):
        if p.exists():
            return float(json.loads(p.read_text())["threshold"])
    return 0.5


THRESHOLD = _threshold()


@torch.inference_mode()
def score(sentences: list[str], batch: int = 64) -> list[float]:
    """P(purple) for each sentence. fp32 to match the calibrated threshold."""
    probs: list[float] = []
    for i in range(0, len(sentences), batch):
        enc = _tok(sentences[i:i + batch], return_tensors="pt", padding=True,
                   truncation=True, max_length=128).to(DEVICE)
        probs.extend(_model(**enc).logits.softmax(-1)[:, 1].tolist())
    return probs


app = FastAPI(title="Purple Prose Detector")


class Req(BaseModel):
    text: str


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.post("/api/detect")
def detect(req: Req):
    sentences = split_sentences(req.text[:MAX_CHARS])[:MAX_SENTENCES]
    probs = score(sentences) if sentences else []
    results = [{"text": s, "prob": round(p, 4), "purple": p >= THRESHOLD}
               for s, p in zip(sentences, probs)]
    return {"threshold": THRESHOLD,
            "purple_count": sum(r["purple"] for r in results),
            "sentences": results}


if __name__ == "__main__":
    # smoke check: the florid sentence must outscore the plain one end-to-end.
    a, b = score(["The sun bled its dying gold across the bruised and weeping hills.",
                  "The committee published the revised schedule on Tuesday."])
    assert a > b, (a, b)
    print(f"selfcheck OK  purple={a:.3f} plain={b:.3f} threshold={THRESHOLD:.3f} device={DEVICE}")
