"""Report test-set classifier metrics at the calibrated threshold.

Reports precision/recall/F1, confusion matrix, and a PR-curve summary, then
lists every false positive and false negative. The threshold is read from
threshold.json inside the model directory; it is not re-picked here because
re-tuning on test would leak test rows into the operating point.

  python -m classifier.evaluate
  python -m classifier.evaluate --model models/ettin400m-purple --split data/purple/splits/test.jsonl
  python -m classifier.evaluate --demo
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve,
                             precision_recall_fscore_support)

from core.paths import SPLITS, CLASSIFIER

MODEL = str(CLASSIFIER)


def load_rows(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows, np.array([int(r["label"]) for r in rows])


def predict_probs(model_dir, texts, batch=64):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=128).to(dev)
            out.append(model(**enc).logits.softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def read_threshold(model_dir):
    p = os.path.join(model_dir, "threshold.json")   # per-classifier: lives inside the model dir
    if os.path.exists(p):
        return float(json.load(open(p))["threshold"]), p
    return 0.5, None


def report(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", pos_label=1, zero_division=0)
    print(f"threshold={threshold:.3f}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    print("confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm)

    # PR curve for diagnostics only — average precision summarizes it in one number.
    if 0 < y_true.sum() < len(y_true):
        print(f"average_precision={average_precision_score(y_true, probs):.3f}")
        prec, rec, thr = precision_recall_curve(y_true, probs)
        # highest-recall point that still holds precision >= 0.90, for reference
        ok = [(t, pr, rc) for t, pr, rc in zip(thr, prec, rec) if pr >= 0.90]
        if ok:
            t, pr, rc = max(ok, key=lambda x: x[2])
            print(f"  (diagnostic: P>=0.90 reachable at thr~{t:.3f} -> P={pr:.3f} R={rc:.3f})")
    else:
        print("average_precision=n/a (test set is single-class)")
    return p, r, f1, preds


def show_errors(rows, y_true, preds, probs):
    for kind, cond in (("FALSE POSITIVES (good prose flagged purple)", (preds == 1) & (y_true == 0)),
                       ("FALSE NEGATIVES (purple slipped through)", (preds == 0) & (y_true == 1))):
        idx = np.where(cond)[0]
        print(f"\n{kind}: {len(idx)}")
        for i in idx:
            print(f"  p={probs[i]:.2f}  {rows[i]['text'][:100]}")


def main(model_dir, split):
    rows, y_true = load_rows(split)
    if not rows:
        print(f"{split} is empty — add human-labeled rows first (README step 6).")
        return
    threshold, src = read_threshold(model_dir)
    print(f"threshold source: {src or 'default 0.5 (run calibrate.py first)'}\n")
    probs = predict_probs(model_dir, [r["text"] for r in rows])
    _, _, _, preds = report(y_true, probs, threshold)
    show_errors(rows, y_true, preds, probs)


def demo():
    # logic check: threshold application + confusion matrix orientation.
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.8, 0.4, 0.9])  # one FP (0.8), one FN (0.4) at thr=0.5
    p, r, f1, preds = report(y, probs, 0.5)
    assert list(preds) == [0, 1, 0, 1], preds
    assert p == 0.5 and r == 0.5, (p, r)  # 1 TP, 1 FP, 1 FN
    print("demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--split", default=str(SPLITS / "test.jsonl"))
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else main(a.model, a.split)
