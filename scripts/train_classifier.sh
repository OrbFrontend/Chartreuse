#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."                       # run from repo root (scripts/ lives one level down)

[ -n "${VIRTUAL_ENV:-}" ] || source .venv/bin/activate

# Which classifier axis to (re)build. Default purple; set DEPURPLE_AXIS=euphemism for the
# euphemism classifier. Every step below roots its paths at data/<axis>/ via core/paths.py,
# so the two classifiers never clobber each other's splits/labels/model.
export DEPURPLE_AXIS="${DEPURPLE_AXIS:-purple}"
export ETTIN_MODEL="${ETTIN_MODEL:-jhu-clsp/ettin-encoder-400m}"   # encoder slug names the model dir
DEST="$(python -c "from core.paths import CLASSIFIER;print(CLASSIFIER)")"
echo "building '$DEPURPLE_AXIS' classifier (data/$DEPURPLE_AXIS/, $DEST)"

rm -f "data/$DEPURPLE_AXIS/splits/"*
rm -f "data/$DEPURPLE_AXIS/labeled.jsonl"
python -m datagen.build_dataset
python -m classifier.train
python -m classifier.calibrate
python -m classifier.evaluate
