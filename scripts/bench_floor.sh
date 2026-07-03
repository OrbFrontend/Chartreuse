#!/usr/bin/env bash
# The optimizer minimizes a STYLE score, so the lowest-value trials are the most de-styled --
# but past a point they win by lobotomy (Goodhart: flatter style, dumber model). The scalar
# can't tell those apart. So: take the best (lowest-value) PCT% of trials as the "as good as it
# gets" floor, pad it by MARGIN% to admit slightly-less-flattened neighbours, then BENCH every
# trial under that floor on a real capability task (babi by default) and keep the ones that kept
# their IQ. Output: trials ranked by retained accuracy, best first -- pick from the top, not the
# bottom of the optimize log.
#
#   scripts/bench_floor.sh                 # babi 1000, strength 1 (mirrors eval_bench usage)
#   scripts/bench_floor.sh --strength 0.5 babi 500
#   scripts/bench_floor.sh tools 15        # any eval_bench task works
#
# ponytail: floor math in one python pass; bench loop reuses eval_bench.sh verbatim (no re-impl).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

PCT=20            # bottom % of trials (by value) that define the floor
MARGIN=20         # pad the floor up by this % to admit near-best trials
TOP=10            # how many survivors to print
STRENGTH=1.0

POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --strength) STRENGTH="$2"; shift 2;;
    --pct)      PCT="$2";      shift 2;;
    --margin)   MARGIN="$2";   shift 2;;
    --top)      TOP="$2";      shift 2;;
    *)          POS+=("$1");   shift;;
  esac
done
set -- "${POS[@]+"${POS[@]}"}"
TASK="${1:-babi}"
LIMIT="${2:-1000}"

LOG="$(python -c 'from depurple._model import LOG; print(LOG)')"
[ -f "$LOG" ] || { echo "no optimize log at $LOG" >&2; exit 1; }

# Parse (trial, value), compute floor = mean(best PCT%) + MARGIN% headroom, emit every trial at
# or below the floor (ascending value). abs() in the headroom so a negative objective still pads
# outward. Summary goes to stderr so stdout is just the candidate trial numbers.
CANDIDATES="$(python - "$LOG" "$PCT" "$MARGIN" <<'PY'
import re, sys, math
log, pct, margin = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
val = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
trials = [(int(n), float(v)) for n, v in
          re.findall(rf"Trial (\d+) finished with value: ({val})", open(log).read())]
trials.sort(key=lambda t: t[1])
k = max(1, math.ceil(len(trials) * pct / 100))
best = [v for _, v in trials[:k]]
mean = sum(best) / k
floor = mean + abs(mean) * margin / 100
cand = [n for n, v in trials if v <= floor]
print(f"{len(trials)} trials; best {k} mean={mean:.4f}; floor={floor:.4f}; "
      f"{len(cand)} under floor: {cand}", file=sys.stderr)
print(*cand)
PY
)"
read -ra TRIALS <<<"$CANDIDATES"
[ ${#TRIALS[@]} -eq 0 ] && { echo "no trials under floor" >&2; exit 1; }

echo "==> benching ${#TRIALS[@]} trials on $TASK $LIMIT (strength $STRENGTH)" >&2
declare -A SCORE BASE
for t in "${TRIALS[@]}"; do
  echo "==> trial$t" >&2
  # eval_bench.sh babi/tools prints an 'accuracy <base> <edited> delta <d>' line; scrape both
  # (base should be stable across trials -- it's the same scenarios each time -- but print it
  # per trial anyway as a sanity check that the cached base pass wasn't stale/mismatched).
  # Full output (per-scenario table, regression dump) stays suppressed here -- rerun
  # eval_bench.sh directly on a specific trial if you need to eyeball those.
  out="$(scripts/eval_bench.sh --variant "trial$t" --strength "$STRENGTH" "$TASK" "$LIMIT")"
  base="$(awk '/^accuracy/{print $2}' <<<"$out" | tail -1)"
  acc="$(awk '/^accuracy/{print $3}' <<<"$out" | tail -1)"
  echo "  accuracy base=${base:-NA} edited=${acc:-NA}" >&2
  BASE[$t]="${base:-NA}"
  SCORE[$t]="${acc:-NA}"
done

echo
echo "trials under floor, by retained $TASK accuracy (best first):"
for t in "${!SCORE[@]}"; do printf '%s\t%s\t%s\n' "${SCORE[$t]}" "trial$t" "${BASE[$t]}"; done \
  | sort -gr | head -n "$TOP" | awk '{printf "  %-10s edited=%-8s base=%s\n", $2, $1, $3}'
