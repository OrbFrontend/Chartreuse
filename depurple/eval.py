"""Run lm-eval benchmarks over an in-memory ablated variant.

Selection mirrors serve.py exactly: --variant base|best|trialN (+ --strength), params read
straight from the optimize.log via serve.resolve_params/load. So the model you bench is the
model you serve. lm_eval 0.4's Python API takes the live model object (HFLM(pretrained=<obj>)),
so there's nothing to bake to disk. Model + axis come from DEPURPLE_MODEL/DEPURPLE_AXIS via
_model.py (same as serve/optimize); this script adds no env vars of its own.

    python -m depurple.eval --variant base --tasks gsm8k --limit 20
    python -m depurple.eval --variant best --tasks gsm8k,ifeval
    python -m depurple.eval --variant trial7 --strength 0.4 --tasks ifeval

One variant is loaded per process. eval_bench.sh calls this for each variant and
bench_compare.py reports the deltas.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

from depurple import serve            # reuse its variant->in-memory-model selection wholesale
from depurple._model import SLUG


def run(variant: str, strength: float, tasks: str, limit: int | None, out_dir: str,
        fewshot: int | None = None, chat: bool = True) -> None:
    serve.load(variant, strength)     # builds serve.model / serve.tok in place (base or ablated)
    lm = HFLM(pretrained=serve.model, tokenizer=serve.tok, batch_size="auto")
    # serve.py serves this -it model via the chat template, so bench it the same way -- raw
    # completion prompts floor generative exact-match tasks (babi/gsm8k) at 0 on an instruct
    # model. num_fewshot=None keeps each task's yaml default; pass --fewshot to anchor format.
    res = simple_evaluate(model=lm, tasks=tasks.split(","), limit=limit, log_samples=False,
                          num_fewshot=fewshot, apply_chat_template=chat,
                          fewshot_as_multiturn=chat)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # bench_compare.py globs **/results_*.json and reads the newest, so the name is the contract.
    f = out / f"results_{int(time.time())}.json"
    f.write_text(json.dumps(res, default=str))   # default=str: skip the unserialisable config bits
    print(f"wrote {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="best",
                    help="'base', 'best', or 'trialN' (params read from optimize.log)")
    ap.add_argument("--strength", type=float, default=1.0, help="ablation scale 0..1 (0=base)")
    ap.add_argument("--tasks", required=True, help="comma-list of lm-eval task names")
    ap.add_argument("--limit", type=int, default=None, help="q per task/subject; omit = full set")
    ap.add_argument("--fewshot", type=int, default=None,
                    help="num_fewshot override; omit = each task's yaml default")
    ap.add_argument("--no-chat", dest="chat", action="store_false",
                    help="skip the chat template (use for raw logprob tasks like lambada)")
    ap.add_argument("--output_path", help="results dir (default depurple/bench-<slug>/<variant>)")
    args = ap.parse_args()
    run(args.variant, args.strength, args.tasks, args.limit,
        args.output_path or f"depurple/bench-{SLUG}/{args.variant}",
        fewshot=args.fewshot, chat=args.chat)
