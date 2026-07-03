"""Minimal OpenAI-compatible endpoint for base or ablated generation models.

Loads the configured base model on GPU and applies directional ablation in memory
for trial variants. Exposes
POST /v1/chat/completions and GET /v1/models.

    python -m depurple.serve --variant base       # unmodified, the comparison baseline
    python -m depurple.serve --variant best       # lowest-value (most de-purpled) trial
    DEPURPLE_AXIS=purple,euphemism python -m depurple.serve --variant best   # joint: both axes

One model variant is loaded per process. The stdlib HTTP server and global GPU
lock make this a single-user test endpoint; use a batched serving stack for
throughput.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from depurple._model import LOG as LOG_PATH
from depurple._model import MODEL, SLUG, AXIS, AXES, MULTI, norm_preserve_for, load_directions
from depurple.ablate import AblationSpec, apply_ablation, build_kernel, decoder_layers

LOG = Path(LOG_PATH)

# populated by main() before the server starts; the handler reads these globals.
tok = model = None
MODEL_ID = SLUG
_gpu_lock = threading.Lock()


# Optuna's value can be negative once euphemism is in the objective (two_sided = euph -
# BETA*direct), so the value group is sign/exponent-aware; purple-only values are always >= 0.
# (Plain `[\d.]+` would silently drop the '-' and pick the wrong 'best' trial in a joint run.)
_VAL = r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)"


def _axis_params(raw: dict, axis: str) -> dict:
    """Cook one axis's kernel params from a logged trial dict. A joint log namespaces every
    key by axis ('purple_max_weight'); a single-axis log has no prefix. per_layer True => the
    per-layer direction (dir_layer=None), and the log omits dir_layer in that case."""
    p = f"{axis}_" if MULTI else ""
    return dict(dir_layer=None if raw[f"{p}per_layer"] else raw[f"{p}dir_layer"],
                max_weight=raw[f"{p}max_weight"], max_pos=raw[f"{p}max_pos"],
                min_weight=raw[f"{p}min_weight"], min_dist=raw[f"{p}min_dist"],
                mlp_scale=raw.get(f"{p}mlp_scale", 1.0))   # pre-split logs lack it -> full MLP


def resolve_params(variant: str):
    """'base' -> None (no ablation). 'best' -> the lowest-value (most de-styled) trial.
    'trialN' -> that specific trial. Params are read from the (axis-keyed) optimize.log so no
    trial number is hardcoded; any completed trial is servable without re-optimizing. Returns
    a list of (axis, kernel-params): one entry for a single-axis run, one PER axis for a joint
    run (DEPURPLE_AXIS=purple,euphemism) -- load() bakes them in one non-stacking pass."""
    if variant == "base":
        return None
    log_text = LOG.read_text() if LOG.exists() else ""
    if variant == "best":
        trials = re.findall(
            rf"Trial \d+ finished with value: {_VAL} and parameters: (\{{.*?\}})", log_text)
        if not trials:
            raise SystemExit(f"no trials in {LOG} (have you run/kept the optimize log?)")
        value, params_s = min(trials, key=lambda t: float(t[0]))
        print(f"best trial: objective={value}")
    else:
        m = re.fullmatch(r"trial(\d+)", variant)
        if not m:
            raise SystemExit(f"unknown variant {variant!r}; use 'base', 'best' or 'trialN'")
        n = int(m.group(1))
        hit = re.search(
            rf"Trial {n} finished with value: {_VAL} and parameters: (\{{.*?\}})", log_text)
        if not hit:
            raise SystemExit(f"trial {n} not found in {LOG} (have you run/kept the optimize log?)")
        value, params_s = hit.group(1), hit.group(2)
        print(f"trial {n}: objective={value}")
    raw = ast.literal_eval(params_s)
    return [(a, _axis_params(raw, a)) for a in (AXES if MULTI else [AXIS])]


def load(variant: str, strength: float = 1.0):
    global tok, model, MODEL_ID
    MODEL_ID = f"{SLUG}-{variant}" + ("" if strength == 1.0 else f"-s{strength}")
    specs_params = resolve_params(variant)           # None | [(axis, params), ...]
    print(f"loading base model on GPU for variant '{variant}'...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    # sdpa = torch's fused attention, no extra dep. flash-attn2 buys ~nothing here:
    # single-user short-context decode is bound by reading 16GB of weights per token,
    # not by attention math (that only dominates at long-context prefill).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    if specs_params is None or strength == 0:
        print(f"no ablation (base). serving {MODEL_ID}")
        return
    n = len(decoder_layers(model))
    # strength scales every axis's kernel: 0 == base, 1 == the trial as optimized. The
    # optimizer minimizes the style score so it lands at full flattening; dial down to trade
    # some editing back for the model's natural voice. A joint run bakes one spec per axis in
    # a single non-stacking pass (same math as optimize/eyeball), not axis-after-axis chaining.
    specs = [AblationSpec(load_directions(a),
                          [w * strength for w in build_kernel(
                              n, p["max_weight"], p["max_pos"], p["min_weight"], p["min_dist"])],
                          p["dir_layer"], p["mlp_scale"], norm_preserve_for(a))
             for a, p in specs_params]
    apply_ablation(model, specs)
    print(f"{variant} ablation applied at strength {strength} "
          f"(axes: {', '.join(a for a, _ in specs_params)}). serving {MODEL_ID}")


@torch.inference_mode()
def generate(messages: list[dict], max_tokens: int, temperature: float) -> str:
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    sample = bool(temperature and temperature > 0)
    kw = dict(max_new_tokens=max_tokens, do_sample=sample, pad_token_id=tok.eos_token_id)
    if sample:
        kw["temperature"] = temperature
    with _gpu_lock:                                  # ponytail: one model, serialize generates
        out = model.generate(**enc, **kw)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_stream(messages: list[dict], max_tokens: int, temperature: float):
    """Yield text deltas. generate() runs in a worker thread feeding the streamer;
    the GPU lock is held for the whole generation."""
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    sample = bool(temperature and temperature > 0)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kw = dict(enc, max_new_tokens=max_tokens, do_sample=sample,
              pad_token_id=tok.eos_token_id, streamer=streamer)
    if sample:
        kw["temperature"] = temperature
    with _gpu_lock:                                  # generate() wraps itself in no_grad
        worker = threading.Thread(target=model.generate, kwargs=kw)
        worker.start()
        for delta in streamer:
            yield delta
        worker.join()


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list",
                             "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if req.get("stream"):
            self._stream(req)
            return
        try:
            text = generate(req["messages"], int(req.get("max_tokens", 512)),
                            float(req.get("temperature", 0.7)))
        except Exception as e:                       # surface errors as JSON, don't drop the conn
            self._json(500, {"error": {"message": repr(e)}})
            return
        self._json(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", MODEL_ID),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {},
        })

    def _stream(self, req: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        model_id = req.get("model", MODEL_ID)

        def chunk(delta: dict, finish):
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        def sse(obj) -> None:
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        try:
            sse(chunk({"role": "assistant"}, None))
            for delta in generate_stream(req["messages"], int(req.get("max_tokens", 512)),
                                         float(req.get("temperature", 0.7))):
                sse(chunk({"content": delta}, None))
            sse(chunk({}, "stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):   # client hung up mid-stream
            pass

    def log_message(self, *a):                        # quiet; generation logs would interleave
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="best",
                    help="'base', 'best', or 'trialN' (params read from optimize.log)")
    ap.add_argument("--strength", type=float, default=1.0, help="ablation scale 0..1 (0=base)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    load(args.variant, args.strength)
    print(f"http://0.0.0.0:{args.port}/v1")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
