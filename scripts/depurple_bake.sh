#!/usr/bin/env bash
# Bake a selected optimized VARIANT into a full standalone model on disk (config + weights +
# tokenizer), ready to load or upload. Same selection as serve.py / eval_bench.sh --
# --variant base|best|trialN (+ --strength), params read from the optimize.log -- so the model
# you bake is the model you serve. serve.py applies the edit IN MEMORY for A/B; this writes the
# ~16 GB standalone copy you actually ship.
#
#   scripts/depurple_bake.sh                                  # best trial, strength 1.0 -> default dir
#   scripts/depurple_bake.sh --variant trial12               # a specific trial
#   scripts/depurple_bake.sh --strength 0.4                  # softened edit baked in
#   scripts/depurple_bake.sh --dry-run                       # print where it'd land, write nothing
#   scripts/depurple_bake.sh models/my-edit                  # explicit output dir
#   DEPURPLE_AXIS=purple,euphemism scripts/depurple_bake.sh  # joint edit
#
# Model + axis come from DEPURPLE_MODEL / DEPURPLE_AXIS via depurple/_model.py (same as
# serve/optimize). Default out dir = optimize's OUT_DIR with the variant (+strength) folded in,
# so bake never clobbers optimize.py's models/<slug>-depurpled.
#
# ponytail: thin wrapper over serve.load + save_pretrained. One ~16 GB copy per bake -- that IS
# the point (serve/eval edit in memory; this is the persisted artifact).
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT=best
STRENGTH=1.0
DRY=0
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --variant)  VARIANT="$2";  shift 2;;
    --strength) STRENGTH="$2"; shift 2;;
    --dry-run)  DRY=1;         shift;;
    -h|--help)  sed -n '2,/^set -/{/^set -/d;s/^# \{0,1\}//p}' "$0"; exit 0;;
    *)          POS+=("$1");   shift;;
  esac
done
set -- "${POS[@]+"${POS[@]}"}"

source .venv/bin/activate

python - "$VARIANT" "$STRENGTH" "${1:-}" "$DRY" <<'PY'
import sys
from depurple._model import OUT_DIR

variant, strength, out, dry = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
if not out:                                  # default: OUT_DIR with variant (+strength) folded in
    sfx = "" if strength == 1.0 else f"-s{strength}"
    out = f"{OUT_DIR[: -len('-depurpled')]}-{variant}{sfx}-depurpled"
if dry:
    print(f"would bake {variant} (strength {strength}) -> {out}")
    sys.exit(0)

from depurple import serve                    # heavy (loads the base model); skip it for --dry-run
serve.load(variant, strength)                 # builds serve.model / serve.tok (base or ablated)
serve.model.save_pretrained(out)
serve.tok.save_pretrained(out)
print(f"baked {variant} (strength {strength}) -> {out}")
PY
