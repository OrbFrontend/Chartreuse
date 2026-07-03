#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."                       # run from repo root (scripts/ lives one level down)

[ -n "${VIRTUAL_ENV:-}" ] || source .venv/bin/activate

# Documented defaults so a bare `scripts/depurple.sh` reproduces the single-axis purple/e4b run;
# override any in the environment (DEPURPLE_AXIS=purple,euphemism for a joint edit).
export DEPURPLE_MODEL="${DEPURPLE_MODEL:-google/gemma-4-E4B-it}"
export DEPURPLE_AXIS="${DEPURPLE_AXIS:-purple}"
export ETTIN_MODEL="${ETTIN_MODEL:-jhu-clsp/ettin-encoder-400m}"

# Joint run (DEPURPLE_AXIS=purple,euphemism) holds one fp32 ettin classifier PER axis
# (~1.6GB each) resident at once, so the perplexity full-vocab logits spike can't find a
# contiguous block on a 24GB card. expandable_segments reclaims the reserved-but-unallocated
# fragmentation the OOM message flags. ponytail: env-var first; if it still OOMs, load the
# inline scorers in bf16 (objective._load_scorer) to free ~1.6GB outright.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# direction.py is deterministic and writes one artifact per axis; a joint run
# (DEPURPLE_AXIS=purple,euphemism) needs every axis's .pt since optimize loads them all.
# Build each missing one; reuse on a re-run. Delete a .pt to force its re-extraction.
while IFS=$'\t' read -r axis path; do
    if [ -f "$path" ]; then
        echo "direction: reusing existing $path (rm it to re-extract)"
    else
        DEPURPLE_AXIS="$axis" python -m depurple.direction
    fi
done < <(python -c 'from depurple._model import AXES, directions_path
for a in AXES: print(f"{a}\t{directions_path(a)}")')

python -m depurple.ablate        # ablation math self-check (fast)

# Baseline kill-test runs on the UNEDITED model — deterministic, no log state. On a resume
# it just reloads the model and re-runs rollouts that can't have changed (slow), so skip it
# when the optimize log already has trials and we're not starting --fresh.
# ponytail: keys off the default LOG only; a --resume-from <backup> with no default log will
# still run the kill-test. Add that case if you routinely resume from backups.
LOG=$(python -c 'from depurple._model import LOG; print(LOG)')
if [ -s "$LOG" ] && [[ " $* " != *" --fresh "* ]]; then
    echo "resume: skipping baseline kill-test (non-empty $LOG, no --fresh)"
else
    python -m depurple.objective # 'purple rises over turns' kill-test (baseline rollouts)
fi

# optimize auto-resumes from its log (depurple/optimize-<slug>.log): a re-run replays
# the completed trials and runs only the remainder up to --trials. It writes that log
# itself now, so don't also redirect this command's stderr to the same path. Pass
# --fresh to discard the log and start the search at trial 0.
python -m depurple.optimize "$@"
#python -m depurple.eyeball
