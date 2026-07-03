"""CPU/RAM-only purple-prose gate. Loads the int8 ONNX model (classifier/quantize.py)
and scores one sentence per line. No torch, no GPU.

  echo "The sun bled its dying gold across the weeping hills." | python -m classifier.predict
  python -m classifier.predict --text "The committee met on Tuesday."
  python -m classifier.predict < sentences.txt           # one sentence per line

Prints: <prob>\t<0|1>\t<text>   (1 = purple, at the calibrated threshold)
"""
import argparse
import json
import os
import sys

import numpy as np

from core.paths import CLASSIFIER, ONNX


def load_session(model_dir, threads=None):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    # prefer fused+int8, then plain int8, then un-quantized export
    for name in ("model_optimized_quantized.onnx", "model_quantized.onnx", "model.onnx"):
        f = os.path.join(model_dir, name)
        if os.path.exists(f):
            break
    sess = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(model_dir)
    return sess, tok, {i.name for i in sess.get_inputs()}


def onnx_probs(session, texts, batch=64):
    sess, tok, want = session
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="np", padding=True,
                  truncation=True, max_length=128)
        feed = {k: v for k, v in enc.items() if k in want}
        logits = sess.run(None, feed)[0]
        e = np.exp(logits - logits.max(-1, keepdims=True))
        out.append((e / e.sum(-1, keepdims=True))[:, 1])
    return np.concatenate(out) if out else np.array([])


def read_threshold(model_dir):
    # prefer threshold.json inside the model dir; fall back to the torch classifier's.
    for p in (os.path.join(model_dir, "threshold.json"), os.path.join(str(CLASSIFIER), "threshold.json")):
        if os.path.exists(p):
            return float(json.load(open(p))["threshold"])
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ONNX))
    ap.add_argument("--text", help="score a single sentence instead of stdin")
    ap.add_argument("--threshold", type=float, help="override calibrated threshold")
    ap.add_argument("--threads", type=int, help="onnxruntime intra-op threads")
    a = ap.parse_args()

    texts = [a.text] if a.text else [l.strip() for l in sys.stdin if l.strip()]
    if not texts:
        return
    thr = a.threshold if a.threshold is not None else read_threshold(a.model)
    probs = onnx_probs(load_session(a.model, a.threads), texts)
    for t, p in zip(texts, probs):
        print(f"{p:.4f}\t{int(p >= thr)}\t{t}")


if __name__ == "__main__":
    main()
