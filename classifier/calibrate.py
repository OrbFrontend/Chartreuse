"""Pick the classifier operating threshold on the calibration split.

Sweeps every candidate threshold (the model's own probabilities) and picks the
LOWEST threshold whose precision >= TARGET (0.90). Lowest = highest recall while
still clearing the precision gate. Writes threshold.json inside the model
directory. Calibration and test are disjoint human/real splits, so the operating
point never sees the test rows.

  python -m classifier.calibrate
  python -m classifier.calibrate --target 0.90 --split data/purple/splits/calibration.jsonl
  python -m classifier.calibrate --demo
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from classifier.evaluate import load_rows, predict_probs  # reuse — same model/IO conventions
from core.paths import SPLITS, CLASSIFIER

TARGET = 0.90


def _sweep(y_true, probs):
    """(threshold, precision, recall, f1) at every candidate threshold — each
    distinct prob (preds only flip there) plus 1.0 (predict-none)."""
    rows = []
    for t in sorted(set(probs.tolist()) | {1.0}):
        preds = (probs >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", pos_label=1, zero_division=0)
        rows.append((t, p, r, f1))
    return rows


def pick_threshold(y_true, probs, target=TARGET):
    """Lowest threshold with precision >= target (max recall under the gate).
    Returns (threshold, precision, recall, met) — met=False -> gate unreachable,
    falls back to the best-F1 threshold so there's still a usable operating point."""
    rows = _sweep(y_true, probs)
    ok = [row for row in rows if row[1] >= target]
    if ok:
        t, p, r, _ = min(ok, key=lambda x: x[0])  # lowest threshold => highest recall
        return t, p, r, True
    t, p, r, _ = max(rows, key=lambda x: x[3])     # gate unreachable -> best F1
    return t, p, r, False


def band_too_wide(y_true, probs, target=TARGET, low=0.05, high=0.5):
    """True when precision >= target holds both at a near-zero threshold (<= low)
    AND well up the range (>= high): the calibration set is too cleanly separable
    to locate an operating point, so the lowest-qualifying pick is arbitrary and
    unlikely to generalize. The floor and ceiling are heuristics, not learned
    bounds."""
    ok = [t for t, p, r, f1 in _sweep(y_true, probs) if p >= target]
    return bool(ok) and min(ok) <= low and max(ok) >= high


def main(model_dir, split, target):
    rows, y_true = load_rows(split)
    if not rows:
        print(f"{split} is empty — run build_dataset.py after labeling (README step 6).")
        return
    if not (0 < y_true.sum() < len(y_true)):
        print(f"{split} is single-class ({int(y_true.sum())} purple / {len(y_true)} total) — "
              "can't calibrate precision on it.")
        return

    probs = predict_probs(model_dir, [r["text"] for r in rows])
    t, p, r, met = pick_threshold(y_true, probs, target)

    if met and band_too_wide(y_true, probs, target):
        print(f"calibration: {len(rows)} rows ({int(y_true.sum())} purple)")
        print(f"REFUSED — precision >= {target:.2f} is met across nearly the whole "
              f"threshold range, so the calibration set is too cleanly separable to "
              f"locate an operating point (it would pick threshold={t:.3f}, which "
              f"won't generalize — check the test split).")
        print("threshold.json left unchanged. Fix: add harder, test-like negatives "
              "to the calibration split, or set the threshold from a test sweep.")
        return

    out = os.path.join(model_dir, "threshold.json")   # per-classifier: lives inside the model dir
    json.dump({"threshold": round(float(t), 4), "target_precision": target,
               "calibration_precision": round(float(p), 4),
               "calibration_recall": round(float(r), 4),
               "split": split, "n": len(rows)}, open(out, "w"), indent=2)

    tag = "" if met else "  *** precision target UNREACHABLE on calibration — using best-F1 fallback ***"
    print(f"calibration: {len(rows)} rows ({int(y_true.sum())} purple)")
    print(f"chosen threshold={t:.3f}  precision={p:.3f}  recall={r:.3f}{tag}")
    print(f"-> {out}")


def demo():
    # logic check: with a clean separation, the gate is reachable and we take the
    # lowest qualifying threshold (best recall). Probs: negs low, pos high.
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.55, 0.6, 0.7, 0.9])
    t, p, r, met = pick_threshold(y, probs, 0.90)
    assert met and p >= 0.90, (p, met)
    assert r == 1.0, r                      # all 3 positives caught at t<=0.6
    assert t > 0.55, t                       # but above the 0.55 negative (else P<0.90)

    # unreachable gate -> fallback, met=False, still returns a real threshold
    y2 = np.array([0, 0, 1, 1])
    probs2 = np.array([0.6, 0.6, 0.6, 0.6])  # nothing separable -> P maxes at 0.5
    _, _, _, met2 = pick_threshold(y2, probs2, 0.90)
    assert not met2

    # refuse gate: the clean-but-localized case above (band starts well above 0)
    # must NOT trip; the degenerate case (negatives ~0, gate met from a near-zero
    # threshold up to ~1 — the 0.019 pattern) MUST trip.
    assert not band_too_wide(y, probs, 0.90)
    yw = np.array([0] * 10 + [1] * 9)
    pw = np.array([0.001] * 9 + [0.02] + [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    assert band_too_wide(yw, pw, 0.90)
    print("demo OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CLASSIFIER))
    ap.add_argument("--split", default=str(SPLITS / "calibration.jsonl"))
    ap.add_argument("--target", type=float, default=TARGET)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else main(a.model, a.split, a.target)
