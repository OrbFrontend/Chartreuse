# Chartreuse

**Hand-written story here:** [docs/write_up.md](docs/write_up.md)

Two tools for taming prose style, built on the same dataset and scorer:

- **Classifier** — sentence-level binary classifier gating **purple prose** (and
  **euphemism**) at `1` / `0`. Input is always one sentence; callers segment upstream and
  score each sentence independently. Favors **precision** — a false positive rewrites good
  prose. The hard part is the dataset, not the model.
- **Depurple** — weight-baked style edit for open LLMs: find the residual direction
  separating purple/plain in matched pairs, orthogonalize `o_proj`/`down_proj` against it. 
  The trained classifier is its scorer; objective is classifier score over rollouts, not KL vs. base.

Both run on `DEPURPLE_AXIS` (`purple`, `euphemism`, or jointly). Switching the axis shifts
all paths and artifact names automatically; the pipeline is identical for each.

## Quick start

**Install dependencies**
```
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

**Train and use the classifier:**
```
export DEPURPLE_AXIS=purple # either purple or euphemism
export ETTIN_MODEL=jhu-clsp/ettin-encoder-400m # There are also 68m, 150m versions - already capable for daily use

# Run this if you have added new stuff in data/, this generates synth pairs and rebuilds train/eval sets
OPENAI_BASE_URL=http://localhost:8000/v1 ./scripts/generate_dataset.sh

# Train the specified classifier
./scripts/train_classifier.sh

# Serve the webapp for eyeball testing (requires Docker) => on http://localhost:9000
PORT=9000 ./scripts/host_classifier_app.sh
```

**Full end-to-end ablation guide on a fresh machine:** [docs/remote_runbook.md](docs/remote_runbook.md)

If you don't want to read the above guide, these commands work out of the box:
```
export DEPURPLE_MODEL=google/gemma-4-31B-it    # Model will be automatically downloaded to HF_HOME
export ETTIN_MODEL=jhu-clsp/ettin-encoder-400m # Model will be automatically downloaded to HF_HOME

# Train both classifiers for the joint run
DEPURPLE_AXIS=purple ./scripts/train_classifier.sh
DEPURPLE_AXIS=euphemism ./scripts/train_classifier.sh

# Joint-directional ablation
export DEPURPLE_AXIS=purple,euphemism
ROLLOUT_BATCH=10 scripts/depurple.sh --trials 180 # This takes a while
ROLLOUT_BATCH=10 scripts/bench_floor.sh # Benchmark final candidates
```

## Env vars

**Required** vars have no default — a script aborts (`FATAL: required env var …`) if one is
unset/empty. Export them before running anything (incl. `--demo` and self-checks).

| Var | Default | Effect |
|-----|---------|--------|
| `DEPURPLE_AXIS` | **required** | Which style axis to work on: `purple` or `euphemism`. Comma-separate both (`purple,euphemism`) for one joint depurple edit. Drives every data/model path in the repo. |
| `ETTIN_MODEL` | **required** | Which encoder to fine-tune, e.g. `jhu-clsp/ettin-encoder-400m` (68m/150m/400m sizes exist). |
| `DEPURPLE_MODEL` | **required** | Base LLM that depurple edits, e.g. `google/gemma-4-E4B-it`. |
| `OPENAI_BASE_URL` | api.openai.com | Endpoint for the LLM data-gen scripts — point at OpenRouter, vLLM, or a local llama.cpp server instead. |
| `GEN_MODEL` | unset | Model name the LLM data-gen scripts call. Omit it for a single-model local server that ignores the field — if the endpoint actually needs one, the request fails with that endpoint's error. |
| `OPENAI_API_KEY` | `sk-noauth` | API key for that endpoint; local servers ignore it. |
| `DEPURPLE_AXIS_WEIGHTS` | `1.0,…` | In a joint depurple run, how much each axis counts toward the objective (one weight per axis, in order). |
| `DEPURPLE_STRENGTH` | `1.0` | How hard to apply the edit in `serve.py`/`eyeball.py` — `0` is the base model, `1` is the full edit. Serve at `0.6–0.9` to soften. |
| `DEPURPLE_PROJECT` | auto per axis | Advanced: refines the style direction before it's baked in, to strip out a confound (e.g. sentence length) it would otherwise absorb. Chosen automatically per axis — override only for experiments. |
| `DEPURPLE_NORM_PRESERVE` | auto per axis | Advanced: whether to restore each edited weight's magnitude after ablation. Auto-coupled to `DEPURPLE_PROJECT` — override only for experiments. |
| `MODEL_DIR`, `MAX_CHARS`, `MAX_SENTENCES` | `models/ettin400m-purple`, `100000`, `2000` | `app.py` serving only: which trained classifier to load, plus input-size caps. **`MODEL_DIR`'s default is a fixed path, not derived from `DEPURPLE_AXIS`/`ETTIN_MODEL`** — set it explicitly if you're serving a different axis or encoder size. |

## Repo layout

Flat, symmetric: the three stages are sibling packages over a shared `core/`, each run
the same way (`python -m <pkg>.<module>`), in operation order datagen → classifier → depurple.

```text
core/
  paths.py                   DEPURPLE_AXIS+ETTIN_MODEL -> all data/<axis>/ + classifier paths;
                             single source BOTH halves import (AXES, classifier_dir, pairs_path)
datagen/                     build the dataset      (python -m datagen.<module>)
  generate_synthetic.py      Plain seed -> purple rewrite pairs (LLM)
  import_text.py             Whole docs -> sentence-split label-0 curated rows
  build_dataset.py           Pool per-axis + shared _common/ -> validate + dedupe + leak-safe split
classifier/                  train/eval the classifier  (python -m classifier.<module>)
  sanity_check.py            Load base encoder as a 2-class head (smoke test)
  train.py                   Fine-tune ettin -> models/ettin400m-<axis>, best-by-val-F1
  calibrate.py               Pick operating threshold on calibration split -> threshold.json
  evaluate.py                Report P/R/F1 + confusion + FPs/FNs on test split
  quantize.py                Export int8 ONNX for CPU/RAM-only inference
  predict.py                 Score stdin sentences via int8 ONNX (no torch/GPU)
depurple/                    bake the style edit      (python -m depurple.<module>)
  _model.py                  DEPURPLE_MODEL/AXIS -> depurple artifact paths; facade over core/paths.py
  direction.py               Per-layer style direction from matched pairs -> directions-*.pt
  ablate.py                  Orthogonalize o_proj/down_proj vs direction; depth kernel
  objective.py               Rollout -> classifier score + ppl/opener/repetition/coverage guards
  optimize.py                Optuna/TPE search -> models/<slug>-depurpled, optimize log
  eyeball.py                 Base-vs-edited rollouts (MANDATORY Goodhart check)
  serve.py                   In-memory edited model, OpenAI-compatible (--variant/--strength)
  repetition.py, *.py        Repetition auditor, euphemism lexicon, opener self-check
  eval_scenarios*.jsonl      Scripted roleplay scenarios that elicit the style
scripts/                     .sh wrappers: build_dataset, train_classifier, depurple, eval_bench, ...
app.py, index.html           Single-page web app: paste text -> per-sentence P(purple)
text_segmentation.py         Shared sentence/dialogue splitter (one source of truth)
data/_common/                Axis-SHARED label-0 negatives, reused by every axis (see Data format)
data/<axis>/                 Per-axis corpus (purple, euphemism); see Data format
models/                      Trained classifiers + depurpled LLMs (gitignored)
Dockerfile                   container for the web app
```

Commands run from the repo root (`python -m` puts it on the path); the `scripts/*.sh`
wrappers `cd` there themselves.

`models/`, and per-axis `labeled.jsonl` + `splits/` are gitignored; `curated.jsonl`,
`human_labeled.jsonl`, and raw/interim corpus files are committed when present.

## Data format

JSONL, one sentence per row:

```json
{"text": "A single sentence.", "label": 1, "source": "human", "group_id": "human_001", "is_real": true}
```

- `text` — exactly one sentence, no line breaks.
- `label` — `1` purple, `0` not.
- `source` — provenance (the axis name like `purple`/`euphemism` for synthetic pairs,
  `mined`, `hard_neg`, `common`, `curated_*`, `human`, or a generator tag like `calm_neg_gen`).
- `group_id` — shared by related rows so they land in the same split (no leakage).
- `is_real` — `false` for a generated sentence, `true` for real/human-authored text,
  including the real-seed pole of a synthetic pair (the `real` side).

### Dataset organization

Two tiers feed every axis. **Per-axis** corpora define what counts as positive *for that
axis* (plus axis-specific negatives); a single **shared** common-negative pool supplies the
label-0 floor that every axis agrees on — plain prose is the common complement of all of
them, so it is stored once and reused, not re-collected per axis.

```text
data/
  _common/
    negatives.jsonl           axis-SHARED label-0 pool (common acceptable sentences like wiki journals)
  <axis>/                     per-axis corpus — one of: purple, euphemism
    raw/good/*.txt            seed sentences, data here gets rewritten into bad twins, used for ablation
    raw/bad/*.txt             seed sentences, data here gets rewritten into good twins, used for ablation
    raw/docs/*.txt            whole documents for import_text (data here is converted to label-0)
    interim/*.jsonl           generator outputs (synthetic/euphemism pairs, mined, triaged, ...)
    curated.jsonl             committed hand-authored labeled train rows (wins dedup conflicts)
    human_labeled.jsonl       real human rows -> calibration + test ONLY (never trained on)
    labeled.jsonl             build_dataset pool, all sources unified (gitignored)
    splits/*.jsonl            train / val / calibration / test (gitignored)
```

`build_dataset.py` pools the active axis's `curated.jsonl` + `interim/*.jsonl` **and** the
shared `_common/negatives.jsonl` (forced to label 0) into `labeled.jsonl`, then splits. The
shared pool is only safe because plain prose is negative for every style axis — but a row
that is plainly negative for one axis can be a true positive for another (an ordinary
euphemism in otherwise plain prose).
Run it once per axis after each retrain, then re-run `build_dataset.py`.

Never hand-edit `splits/` — rerun `build_dataset.py`. `curated.jsonl`, `human_labeled.jsonl`,
`_common/`, and `raw/`+`interim/` are committed when present; `labeled.jsonl` and `splits/`
are gitignored.

## Pipeline

One full run from scratch. Export the required vars once up front (no defaults — bare
commands abort otherwise); switch axis by changing `DEPURPLE_AXIS` and paths/artifact names
shift automatically, nothing else changes:

```bash
export DEPURPLE_AXIS=purple                       # or euphemism / purple,euphemism
export ETTIN_MODEL=jhu-clsp/ettin-encoder-400m
export GEN_MODEL=gemma OPENAI_BASE_URL=http://127.0.0.1:8080/v1   # LLM data-gen steps only
```

LLM data-gen steps need the `OPENAI_*` endpoint (`GEN_MODEL` too, unless your endpoint ignores
the field); everything else is local. Most scripts take `--demo` for a no-network self-check
(still needs `DEPURPLE_AXIS`/`ETTIN_MODEL`).

**Build the dataset** — `data/<axis>/` → `splits/{train,val,calibration,test}.jsonl`

```bash
python -m datagen.generate_synthetic      # raw/good/*.txt + LLM -> interim/<axis>_pairs.jsonl  (neg 0 / pos 1, shared group_id)
python -m datagen.import_text             # raw/docs/*.txt -> appends label-0 rows to curated.jsonl
# add real human rows by hand              # you           -> human_labeled.jsonl (routed to calibration+test only)
python -m datagen.build_dataset           # all of the above + curated.jsonl + human_labeled.jsonl
#                                          #               -> labeled.jsonl + splits/{train,val,calibration,test}.jsonl
```

`build_dataset.py` pools every artifact — the active axis's `curated.jsonl` + `interim/`
*and* the shared `data/_common/negatives.jsonl` — dedupes normalized text, keeps `group_id`s
together, and rewrite new good/bad seeds. Re-run it after any data change.

**Train, calibrate, evaluate** — `splits/` → `models/ettin400m-<axis>/`

```bash
python -m classifier.train                        # splits/train.jsonl, best-by-val-F1 -> models/ettin400m-<axis>/
python -m classifier.calibrate                    # splits/calibration.jsonl           -> models/ettin400m-<axis>/threshold.json
# then overwrite threshold.json -> 0.66    # calibrate picks a degenerate threshold here; 0.66 is the deploy value
python -m classifier.evaluate                     # splits/test.jsonl at threshold      -> P/R/F1, confusion, FP/FN list
```

GPU training needs a CUDA torch wheel (`requirements.txt` documents the CPU one).
Calibration/test must hold real human rows or these steps no-op.

**Optional deploy** — classifier → CPU inference / web app

```bash
python -m classifier.quantize --check             # -> models/ettin400m-<axis>-onnx-int8 (vs torch on test)
echo "The sun bled its dying gold." | python -m classifier.predict   # int8, no GPU
python app.py                              # web app: paste text -> per-sentence P(purple)
```

**Depurple** — bake the style edit into an LLM, using the trained classifier as scorer.
Needs `models/ettin400m-<axis>`, gemma-4 access (`hf auth login`), and a GPU. `DEPURPLE_MODEL`,
`DEPURPLE_AXIS`, and `ETTIN_MODEL` are all required (no defaults) — export them before
`direction.py`; every depurple artifact keys off `(model, axis)`.

**Why the scorer is a classifier, not KL divergence.** Depurple's ablation machinery is
derived from [heretic](https://github.com/p-e-w/heretic), which scores each candidate edit
by the **KL divergence** between the edited and base model on benign prompts. That metric
fits heretic's job — abliterating *refusal*, a narrow behavior fired by a narrow class of
inputs: on ordinary prompts the ideal edit leaves the output distribution untouched, so
base-relative `KL ≈ 0` is simultaneously the target and a clean bound on collateral damage.
De-purple and de-euphemism are the opposite problem. Purple prose and euphemism are a
pervasive *style*, expressed on exactly the ordinary prompts heretic holds fixed, and the
edit's entire purpose is to move that distribution. So `KL(edited ‖ base)` measures the
magnitude of the change we *want*: minimizing it minimizes the edit — i.e. does nothing —
and there is no held-fixed reference distribution to bound damage against, because every
prompt is one we mean to change. The metric is anti-correlated with the goal. Instead the
**trained classifier is the objective** — drive late-weighted `P(purple)`/`P(euphemism)`
down over multi-turn rollouts — and capability is bounded not against the base model but
against an *external* anchor: perplexity on held-out human prose (`human_writing.txt`),
plus structural-collapse guards (opener repetition, repetition density, sentence coverage,
and markdown/list "report-mode" — the classifier itself rewards that register, so it needs
its own guard, see `objective.struct_frac`).

**Direction refinements (`DEPURPLE_PROJECT` / `DEPURPLE_NORM_PRESERVE`).** The raw direction
`d = mean(purple) - mean(plain)` can absorb a confound from the pairs data instead of pure
style — on the purple axis the purple twin is reliably ~10 words longer than the plain twin,
so a chunk of `d` is just "longer sentence" and ablating it collaterally shortens everything.
`direction.py` fixes this per axis, automatically, from the pairs' own word-count delta: if a
label is confounded with length beyond a small threshold, it strips the empirical length
direction out of `d` before saving (`DEPURPLE_PROJECT=length`); otherwise it saves `d` raw.
Euphemism's pairs aren't length-confounded (crude and vanilla phrasings run about the same
length), so it keeps the raw direction. Projecting a direction removes less, so the ablation
kernel's write-magnitude loss is smaller; `DEPURPLE_NORM_PRESERVE` couples to that — a raw
direction gets its post-ablation column norms restored (`1`), a projected one doesn't, because
restoring on top of an already-gentle removal overshoots into flat, report-style prose. Both
knobs are env-overridable per axis for experiments; a joint run can mix, e.g. purple's
length-projected direction with magnitude untouched alongside euphemism's raw direction with
magnitude restored, in the same non-stacking pass. See `docs/ablation_refinements_plan.md`
for the A/B this decided and `_model._auto_project` for the exact rule.

```bash
export DEPURPLE_MODEL=google/gemma-4-E4B-it DEPURPLE_AXIS=purple ETTIN_MODEL=jhu-clsp/ettin-encoder-400m
scripts/download_model.sh                        # pull gated base LLM into the HF cache
python -m depurple.direction               # matched pairs -> directions-<slug>{-axis}{-lengthproj}.pt (+ kill-test)
python -m depurple.ablate                  # ablation math self-check (proj -> ~0, restore clean, incl. norm-preserve)
python -m depurple.objective               # confirm the style rises over turns (kill-test 2)
python -m depurple.optimize --trials 40    # Optuna -> models/<slug>{-axis}{-lengthproj|-normpres}-depurpled + optimize log
python -m depurple.eyeball                 # base-vs-edited rollouts -- MANDATORY, the scalar lies (Goodhart)
python -m depurple.serve --variant best --strength 0.4   # in-memory edit, OpenAI-compatible
```

The direction/model/log filenames pick up a variant tag from the resolved refinement
(`-lengthproj`, `-normpres`, or both) so a refined run never collides with the bare baseline
produced by an env override — see `DEPURPLE_PROJECT`/`DEPURPLE_NORM_PRESERVE` above.

`scripts/depurple.sh --trials 40` chains direction→self-checks→optimize (reusing an existing
direction file). The destination is **bland, not good** — serve at `--strength 0.6–0.9` and
always read `eyeball.py` output; the objective minimum is the most aggressive edit, not the
best prose. `eyeball.py` now prints a mechanical "red flags" line after each edited block
(weak classifier move, sentence-shortening, markdown/list report-mode, opener/intra
repetition vs. base) as a pre-read — it's advisory, not a verdict; register drift like
report-mode scores *well* on the classifier, so the read is still mandatory.

The search is seeded TPE (`seed=0`), not random — later trials exploit earlier ones, so
more trials help. `--trials N` is a **target total, not an increment**: the optimize log is
the search memory, replayed to warm-start TPE on resume. So `--trials 30` then `--trials 60`
accumulates to 60 trials; running `--trials 30` twice no-ops the second time (replays 30,
runs 0). Deleting the log resets to trial 0 — and since the direction extraction and rollouts
(greedy, `do_sample=False`) are deterministic, a fresh `--trials 30` reproduces the previous
run exactly. Delete the log only to start a genuinely different search (changed space,
scenarios, scorer), not to "redo" one. `--fresh` discards the log without deleting it.

**Joint depurple** (one model, several axes) — build each axis's direction and classifier
first, then search jointly:

```bash
python -m depurple.direction                              # purple direction
DEPURPLE_AXIS=euphemism python -m depurple.direction      # euphemism direction
DEPURPLE_AXIS=purple,euphemism python -m depurple.optimize --trials 40   # one non-stacking pass -> models/<slug>-purple-euphemism-depurpled
DEPURPLE_AXIS=purple,euphemism python -m depurple.eyeball
```
