"""Export the fine-tuned classifier to a q8_0 GGUF for llama.cpp consumers (Orb's slop
classifier), and check it scores like torch when served the way Orb serves it.

  LLAMA_CPP_DIR=~/llama.cpp python -m classifier.export_gguf            # -> models/<name>-gguf/<name>-q8_0.gguf
  LLAMA_CPP_DIR=~/llama.cpp python -m classifier.export_gguf --check --server-bin ../Orb/backend/data/llama-bin/cpu

LLAMA_CPP_DIR is a llama.cpp source checkout (convert_hf_to_gguf.py + gguf-py/); it must
support ModernBertForSequenceClassification with classifier_pooling "mean" (writes
`classifier.pooling_type`). The serving binary must read that key too, or it silently
falls back to CLS pooling -- --check is what catches that (AUC collapses to ~0.5).

Two traps this script owns so callers don't:
  * The converter maps `classifier.weight` to the output head only when config.json carries
    real id2label names; without them (models trained before train.py set LABELS) the head
    and the dense layer both become `cls.weight` and conversion dies. A staging dir with
    symlinked weights and a labelled config fixes old models without touching them.
  * Head logits come back through /embedding in a full-width buffer whose tail is garbage,
    so the request must send embd_normalize=-1 (as Orb does) or every logit reads ~0.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from core.paths import CLASSIFIER, SPLITS, req

DUMMY = {"LABEL_0", "LABEL_1"}
# What Orb's _head_profile passes for the slop classifier (backend/inference/local_ml.py).
N_CTX = 512


def default_out(model_dir, outtype):
    m = Path(model_dir)
    return m.parent / f"{m.name}-gguf" / f"{m.name}-{outtype}.gguf"


def stage(model_dir, tmp):
    """Symlinked copy of model_dir whose config names its labels (see module docstring)."""
    src = Path(model_dir).resolve()
    for f in src.iterdir():
        if f.is_file() and f.name not in ("config.json", "training_args.bin"):
            (tmp / f.name).symlink_to(f)
    cfg = json.loads((src / "config.json").read_text())
    if not (set((cfg.get("id2label") or {}).values()) - DUMMY):
        axis = json.loads((src / "chartreuse.json").read_text())["axis"] \
            if (src / "chartreuse.json").exists() else "positive"
        cfg["id2label"] = {"0": "clean", "1": axis}
        cfg["label2id"] = {"clean": 0, axis: 1}
        print(f"config has no real id2label; staging with {cfg['id2label']}")
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2))


def export(model_dir, out, outtype):
    llama = Path(os.path.expanduser(req("LLAMA_CPP_DIR")))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(model_dir, Path(tmp))
        env = dict(os.environ, PYTHONPATH=str(llama / "gguf-py"))
        subprocess.run([sys.executable, str(llama / "convert_hf_to_gguf.py"), tmp,
                        "--outtype", outtype, "--outfile", str(out),
                        # else general.name is the tempdir's name
                        "--model-name", Path(model_dir).resolve().name], env=env, check=True)
    for name in ("threshold.json", "chartreuse.json"):   # keep the calibration beside the weights
        p = Path(model_dir, name)
        if p.exists():
            (out.parent / name).write_bytes(p.read_bytes())
    print(f"-> {out} ({out.stat().st_size / 1e6:.0f} MB)")


def gguf_probs(gguf, server_bin, texts, port=18931):
    """P(class 1) per text from llama-server, launched and queried exactly as Orb does."""
    server_bin = Path(os.path.expanduser(server_bin))
    argv = [str(server_bin / "llama-server"), "-m", str(gguf), "--port", str(port), "-ngl", "0",
            "-c", str(N_CTX), "-np", "1", "-b", str(N_CTX), "-ub", str(N_CTX),
            "--flash-attn", "off", "--embeddings", "--pooling", "rank"]
    env = dict(os.environ, LD_LIBRARY_PATH=str(server_bin))
    proc = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def post(path, body):
        r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=300) as resp:
            return json.loads(resp.read())

    try:
        for _ in range(240):
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited ({proc.returncode}): {' '.join(argv)}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("llama-server never became healthy")
        out = []
        for t in texts:
            ids = post("/tokenize", {"content": t, "add_special": True})["tokens"][:N_CTX]
            a, b = post("/embedding", {"content": ids, "embd_normalize": -1})[0]["embedding"][0][:2]
            m = max(a, b)
            out.append(math.exp(b - m) / (math.exp(a - m) + math.exp(b - m)))
        return out
    finally:
        proc.terminate()
        proc.wait()


def check(model_dir, gguf, server_bin, n, split=str(SPLITS / "test.jsonl")):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from classifier.evaluate import predict_probs, read_threshold

    rows = [json.loads(l) for l in open(split) if l.strip()]
    rows = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:n]]
    texts, labels = [r["text"] for r in rows], [int(r["label"]) for r in rows]
    thr, _ = read_threshold(model_dir)
    got = np.asarray(gguf_probs(gguf, server_bin, texts))
    ref = np.asarray(predict_probs(model_dir, texts))
    agree = int(((ref >= thr) == (got >= thr)).sum())
    print(f"gguf vs torch on {len(texts)} test rows: decisions agree {agree}/{len(texts)} at {thr:.4f}, "
          f"mean|dp|={np.abs(ref - got).mean():.4f}, AUC torch {roc_auc_score(labels, ref):.4f} / "
          f"gguf {roc_auc_score(labels, got):.4f}")
    assert agree >= 0.98 * len(texts), "gguf flipped too many decisions (pooling key unread by this binary?)"
    print("check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CLASSIFIER))
    ap.add_argument("--out", help="default: models/<name>-gguf/<name>-<outtype>.gguf")
    ap.add_argument("--outtype", default="q8_0")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--server-bin", help="dir holding the llama-server that will serve it (needed by --check)")
    ap.add_argument("--n", type=int, default=400, help="test rows for --check")
    a = ap.parse_args()
    if a.check and not a.server_bin:
        ap.error("--check needs --server-bin")
    out = a.out or default_out(a.model, a.outtype)
    export(a.model, out, a.outtype)
    if a.check:
        check(a.model, out, a.server_bin, a.n)
