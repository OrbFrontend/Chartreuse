#!/usr/bin/env bash
# Stage 1 — build the per-axis dataset: pool curated + interim/*.jsonl + the shared common
# negatives into labeled.jsonl, then write the leak-safe splits/. Watch the printed warnings
# (dupes, invalid/multi-sentence rows, low human counts). Re-run after any data change.
#   DEPURPLE_AXIS=purple scripts/update_dataset.sh  # purple
#   DEPURPLE_AXIS=euphemism scripts/update_dataset.sh  # euphemism
set -euo pipefail
cd "$(dirname "$0")/.."                       # run from repo root (scripts/ lives one level down)
[ -n "${VIRTUAL_ENV:-}" ] || source .venv/bin/activate

export DEPURPLE_AXIS="${DEPURPLE_AXIS:-purple}"
export ETTIN_MODEL="${ETTIN_MODEL:-jhu-clsp/ettin-encoder-400m}"   # core.paths needs it at import
python -m datagen.generate_synthetic
python -m datagen.import_text
python -m datagen.build_dataset
