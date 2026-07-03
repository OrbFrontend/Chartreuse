"""Export the fine-tuned classifier to ONNX and dynamic-int8 quantize it for
CPU/RAM-only inference (no torch, no GPU needed at deploy time).

  python -m classifier.quantize                 # export + quantize -> models/<name>-onnx-int8
  python -m classifier.quantize --check         # also compare int8 vs torch probs on the test split

Output dir holds tokenizer + model_quantized.onnx; run it with classifier/predict.py.
"""
import argparse
import os

from core.paths import SPLITS, CLASSIFIER, ONNX

# avx2 preset: dynamic (weight-only) int8, reduce_range for broad x86/AMD CPUs.
# ponytail: avx2 over avx512_vnni — runs anywhere; bump preset only if the
# target CPU is known and you want the last few % of speed.
# per_channel=True matters here: without it int8 dropped a true-purple row on
# this transformer (recall 1.0 -> 0.96). Cost is ~nil, so keep it on.


def quantize(model_dir, out_dir):
    from optimum.onnxruntime import (ORTModelForSequenceClassification, ORTOptimizer,
                                     ORTQuantizer)
    from optimum.onnxruntime.configuration import AutoQuantizationConfig, OptimizationConfig
    from transformers import AutoTokenizer

    m = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    m.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_dir).save_pretrained(out_dir)

    # Fuse attention/GELU/LayerNorm BEFORE quantizing — orthogonal to int8 and the
    # bigger batch=1 CPU win. level 99 = all transformer + layout opts. -> model_optimized.onnx
    ORTOptimizer.from_pretrained(out_dir).optimize(
        save_dir=out_dir, optimization_config=OptimizationConfig(optimization_level=99))

    # Symbolic shape inference: level-99 fusion leaves ettin MatMul outputs without an
    # inferable type, which the int8 quantizer chokes on. quant_pre_process fills them in.
    from onnxruntime.quantization.shape_inference import quant_pre_process
    opt = os.path.join(out_dir, "model_optimized.onnx")
    quant_pre_process(opt, opt, skip_symbolic_shape=False)

    # Quantize the fused graph. file_name pins the input so re-runs don't trip on stale output.
    q = ORTQuantizer.from_pretrained(out_dir, file_name="model_optimized.onnx")
    q.quantize(save_dir=out_dir,
               quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=True))
    final = os.path.join(out_dir, "model_optimized_quantized.onnx")
    print(f"-> {final} ({os.path.getsize(final) / 1e6:.1f} MB)")


def check(model_dir, out_dir, split=str(SPLITS / "test.jsonl")):
    import json

    import numpy as np

    from classifier.predict import load_session, onnx_probs, read_threshold
    from classifier.evaluate import predict_probs  # torch reference

    rows = [json.loads(l) for l in open(split) if l.strip()]
    texts = [r["text"] for r in rows]
    thr = read_threshold(out_dir)
    ref = predict_probs(model_dir, texts)
    got = onnx_probs(load_session(out_dir), texts)
    # what matters for a gate is gate DECISIONS, not raw probs — int8 moves a few
    # probabilities a lot but should flip almost no decisions at the threshold.
    agree = int(((ref >= thr) == (got >= thr)).sum())
    print(f"int8 vs torch on {len(texts)} test rows: decisions agree {agree}/{len(texts)} "
          f"(mean|Δp|={np.abs(ref - got).mean():.4f})")
    assert agree >= 0.95 * len(texts), f"int8 flipped too many decisions ({len(texts) - agree})"
    print("check OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CLASSIFIER))
    ap.add_argument("--out", default=str(ONNX))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    quantize(a.model, a.out)
    if a.check:
        check(a.model, a.out)
