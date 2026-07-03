# Ablate gemma-4-31B-it (JOINT purple+euphemism axes) on a rented RTX 6000 PRO (96 GB)

Runbook for a clean rented box. Axes = `purple,euphemism` (joint single-pass edit), model
= `google/gemma-4-31B-it`. A joint run bakes BOTH directions into one model in one
non-stacking pass, so it needs BOTH scorers + BOTH pairs files + BOTH directions present.

Artifact names this run (slug `gemma-4-31b-it`, joint suffix `-purple-euphemism`):
- directions : `depurple/directions-gemma-4-31b-it.pt` (purple, un-suffixed)
              + `depurple/directions-gemma-4-31b-it-euphemism.pt`   (both built ON the remote)
- scorers    : `models/ettin400m-purple` + `models/ettin400m-euphemism`  (both TRAINED on the remote, gitignored)
- pairs      : `data/purple/interim/purple_pairs.jsonl`
              + `data/euphemism/interim/euphemism_pairs.jsonl`          (ride repo rsync)
- eval scens : `depurple/eval_scenarios.jsonl` + `depurple/eval_scenarios-euphemism.jsonl`  (committed)
- output     : `models/gemma-4-31b-it-purple-euphemism-depurpled`        (~62 GB, push to HF)
- log (most important!): `depurple/optimize-gemma-4-31b-it-purple-euphemism.log` (used for ablation resumption and baking)

## Box requirements
Ubuntu 24+

160 GiB disk space required for training + ablating + quantization.

31B bf16 peaks at 85GB VRAM at ROLLOUT_BATCH=10.

Currently the code supports single GPU only and does NOT shard!

---

## Phase 0 — disk + HF token on the remote
Verify you have enoguh disk space:
```
echo $HF_HOME                             # vast preset: /workspace/.hf_home — leave it, don't override
df -h /workspace                          # need ~160 GB free (62 base + 62 output + cache)
ls $HF_HOME/hub                           # base may already be cached (models--google--gemma-4-31B-it)
hf auth login                             # or export HF_TOKEN=hf_xxx — if the model is gated,
                                          # accept the license on the model page first
```

## Phase 1 — copy the repo up (from your local box)
Set the connection vars once (reused in Phase 8's pull-back). This node uses a
**non-default ssh port**, so every rsync below passes `-e "ssh -p $GPU_PORT"` — without it
rsync silently hits port 22 and hangs:
```
export GPU_USER=root                      # rental's ssh user
export GPU_HOST=121.158.120.134           # rental's host/IP
export GPU_PORT=24143                     # rental's ssh port (non-default)
# shell in:  ssh -p $GPU_PORT $GPU_USER@$GPU_HOST -L 8080:localhost:8080
```
```
rsync -az --info=progress2 -e "ssh -p $GPU_PORT" \
  --exclude='.venv' --exclude='models' --exclude='__pycache__' --exclude='*.out' \
  ./ $GPU_USER@$GPU_HOST:~/Chartreuse/
```
`models/` is excluded — the scorers aren't shipped; they're trained on the box in Phase 4.
The repo rsync carries the working tree (no `--exclude` for `data/`).

## Phase 2 — venv + torch (from remote console)
From this point on, we'll work from inside the remote GPU node. SSH into it with:
```
ssh -p $GPU_PORT $GPU_USER@$GPU_HOST -L 8080:localhost:8080
```

RTX 6000 PRO is **Blackwell (sm_120)**
```
cd ~/Chartreuse
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install --upgrade "transformers==5.12.1"   # requirements.txt may pin an older release; 5.12.1 required
python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"
                                          # expect 2.9+cu130 and (12, 0)
```

Export required env vars:
```
export DEPURPLE_MODEL=google/gemma-4-31B-it
export ETTIN_MODEL=jhu-clsp/ettin-encoder-400m
```

## Phase 3 — pull the base model from HF (background it)
HF downloader sometimes craps out. Use wget to download the weights:
```
scripts/download_model.sh &       # reads DEPURPLE_MODEL; wget -c, resumable, idempotent
```
Let it finish before Phase 5 (the search loads the base model) — `wait`, or just re-run it
(idempotent: complete blobs are skipped) to confirm it's all there.

## Phase 4 — train BOTH scorers on the remote
The joint objective needs one classifier per axis. Train each with `scripts/train_classifier.sh`
```
DEPURPLE_AXIS=purple    scripts/train_classifier.sh
DEPURPLE_AXIS=euphemism scripts/train_classifier.sh
```
Both write under `models/` (gitignored, not shipped — that's why they're built here). The
per-line `DEPURPLE_AXIS=` overrides the joint `purple,euphemism` export, so each run builds
exactly one scorer. Phase 5's search loads both — don't skip either.

## Phase 5 — build BOTH directions + run the joint search (`scripts/depurple.sh`)
One wrapper does it all: it loops over every axis in `DEPURPLE_AXIS` and builds each
per-(model,axis) direction (reusing an existing `.pt`), runs the ablation + objective
self-checks, then runs the Optuna search, passing through whatever args you give it:

```
export DEPURPLE_AXIS=purple,euphemism
```

Now the actual ablation, this will take some time:

```
ROLLOUT_BATCH=10 scripts/depurple.sh --trials 180 # resumable: re-run to replay the log + finish the remainder; --fresh restarts at 0
```
- Directions are keyed to `(model, axis)` and built ON the box — scripts/depurple.sh extracts fresh →
  `directions-gemma-4-31b-it.pt` (purple) + `directions-gemma-4-31b-it-euphemism.pt`.
  WATCH **both** self-checks: if EITHER axis's twins DON'T project higher, STOP — that axis
  isn't linearly separable at 31B, abort the run.
- Joint namespaces the kernel params per axis (`purple_*`, `euphemism_*`), so the search
  space ~doubles — budget more trials than a single-axis run.
- Optional: `export DEPURPLE_AXIS_WEIGHTS=1.0,0.5` to weight purple over euphemism (aligned
  to the `DEPURPLE_AXIS` order) if one axis dominates the objective.
- Output: `models/gemma-4-31b-it-purple-euphemism-depurpled`.

## Phase 5.5 — benchmarking (before you eyeball)
Catch a lobotomy with some benchmarks:
```
ROLLOUT_BATCH=10 scripts/eval_bench.sh --fewshot 5 babi 500              # best trial vs base, 500 questions
ROLLOUT_BATCH=10 scripts/eval_bench.sh ifeval 50
```
Bench the trial you'll actually serve: default `--variant best`; pass `--variant trialN` to floor
a specific chosen trial instead (same names/strength as serve.py, so bench == serve). A big
negative delta = the trial won by lobotomy.

## Phase 6 — eyeball check and benchmark flooring (MANDATORY before you trust/upload)
The scalar may lie. Destination is "bland", not "good". No `.sh` wrapper —
`eyeball.py` reads the best (lowest-value) trial straight from the joint optimize log and
eyeballs **each axis's own scenarios with each axis's own scorer**:
```
python -m depurple.eyeball
```
If it collapsed (staccato / empty / repetition) OR either axis didn't actually move
(purple still ornate, euphemisms still indirect), tune and re-search — don't upload.

**Benchmark flooring** — the lowest-value trial can win by lobotomy. Rank the
near-best trials by retained accuracy and pick a non-lobotomized one, not just the
cheapest scalar:
```
ROLLOUT_BATCH=10 scripts/bench_floor.sh   # babi 1000; scripts/eval_bench.sh for one base-vs-variant task
```

### Optional — remote OpenAI endpoint to hand-test on top of the eyeball
To poke the de-purpled model with your own prompts from your laptop, serve it on the box and expose it with 
a cloudflared quick tunnel:
```
python -m depurple.serve --variant best --strength 0.8 &        # OpenAI-compatible at :8000/v1 (one GPU)
cloudflared tunnel --url http://localhost:8000                  # prints https://<random>.trycloudflare.com
```
Then call it from anywhere as `OPENAI_BASE_URL=https://<random>.trycloudflare.com/v1`
(`--variant base` to A/B the unmodified model, or `--strength` to dial the edit). No
`cloudflared` on the box? `wget -qO cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x cloudflared && ./cloudflared tunnel --url http://localhost:8000`.
Simpler if you already have the ssh shell open: drop the tunnel and just `-L 8000:localhost:8000` the port forward.

---

## Phase 7 — push the final model to HF (from the remote console)
```
hf auth whoami
hf repo create <you>/gemma-4-31b-it-purple-euphemism-depurpled --type model   # one-time; --private if wanted
hf upload <you>/gemma-4-31b-it-purple-euphemism-depurpled \
  models/gemma-4-31b-it-purple-euphemism-depurpled . \
  --commit-message "joint depurple (purple+euphemism) ablation of gemma-4-31B-it"
```
`hf upload` is resumable and chunks the ~62 GB; rerun on flaky network and it skips finished shards.

## Phase 8 — pull cheap artifacts back, then kill the box
Exit the GPU node, go back to local box:
```
exit
```

The model is on HF now; just retrieve the records (log + BOTH direction files) before
destroying the rental:
```
rsync -az -e "ssh -p $GPU_PORT" $GPU_USER@$GPU_HOST:~/Chartreuse/depurple/optimize-gemma-4-31b-it-purple-euphemism.log ./depurple/
rsync -az -e "ssh -p $GPU_PORT" $GPU_USER@$GPU_HOST:~/Chartreuse/depurple/directions-gemma-4-31b-it.pt ./depurple/
rsync -az -e "ssh -p $GPU_PORT" $GPU_USER@$GPU_HOST:~/Chartreuse/depurple/directions-gemma-4-31b-it-euphemism.pt ./depurple/
```
Serving is in-memory (no disk copy): `python -m depurple.serve --variant best --strength 0.8`
— but `serve.py` also loads on one GPU, so a 31B serve needs a 96 GB card or a quantized variant.

## Gotchas recap
- Joint needs **both scorers + both pairs + both directions** — train both scorers ON the remote (Phase 4), both pairs ride the repo rsync, scripts/depurple.sh builds both directions on the remote.
- torch **cu130** (sm_120, matches the node's CUDA 13.0 / Blackwell), not cu124 — wrong wheel = no Blackwell kernels.
- train **both scorers** on the box (Phase 4, after the venv) — gitignored and no longer shipped; cu130 GPU torch handles training too, no CPU fallback.
- Two scorers resident at once in a joint run (~3.2 GB) — that's what the peak-memory budget already accounts for.
- **transformers 5.12.1** — upgrade after `requirements.txt` (Phase 2); the pinned version in requirements may be older and will silently misbehave on Blackwell.
- ~**160 GB** free disk on `/workspace`. **Don't set `HF_HOME`** — the vast image presets it to `/workspace/.hf_home` (big volume) in `/etc/environment`; overriding it re-downloads the 62 GB base and fills the disk.
