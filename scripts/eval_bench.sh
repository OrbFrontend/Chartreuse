#!/usr/bin/env bash
# Lobotomy / capability + instruction-following benches: base vs an ablated VARIANT, both
# built IN MEMORY (no 16 GB -depurpled copy on disk). Selection mirrors serve.py exactly --
# --variant base|best|trialN (+ --strength), params read from the optimize.log -- so the
# model you bench is the model you serve. Runs base then the variant in SEPARATE processes
# (one 16 GB model fits the 3090 at a time, a fresh process guarantees the GPU is freed
# between them), then prints the per-task delta -- the only comparison that isolates the edit
# (absolute scores depend on shots/template, the delta doesn't).
# Generative RP-diversity metrics are a separate script; this is exact-match / rule-based only.
#
#   scripts/eval_bench.sh                                   # print this menu + time hints, exit
#   scripts/eval_bench.sh all                               # default suite, full sets, best trial
#   scripts/eval_bench.sh mmlu 20                           # one task, 20 q/subject (quick)
#   scripts/eval_bench.sh gsm8k,ifeval 40                   # pick tasks, 40 q each
#   scripts/eval_bench.sh --variant trial7 gsm8k,ifeval     # a specific trial (not just best)
#   scripts/eval_bench.sh --strength 0.4 all               # softened edit (same flag as serve)
#   DEPURPLE_MODEL=google/gemma-4-31B-it scripts/eval_bench.sh all
#   DEPURPLE_AXIS=purple,euphemism      scripts/eval_bench.sh all   # joint edit
#
# Model + axis come from DEPURPLE_MODEL / DEPURPLE_AXIS via depurple/_model.py (same as
# serve/optimize); this script no longer re-derives them.
#
# Other lm-eval tasks worth a look (add as args): truthfulqa_mc2, piqa, boolq,
# openbookqa, commonsense_qa, mmlu_pro, bbh, drop. List all: `lm_eval --tasks list`.
#
# ponytail: thin wrapper over depurple/eval.py + bench_compare.py. Add benches via args, not code.
set -euo pipefail
cd "$(dirname "$0")/.."

# Default = generative, chat-templated, edit-fragile tasks (the strongest lobotomy signal).
# These all want the chat template ON (default), so they share one run. babi was dropped:
# its gold answer carries a leading space the chat model never reproduces, so its raw
# exact_match scores 0 even when the model is right (verified) -- a metric artifact, no signal.
# lambada_openai is a LOGPROB task -- run it separately with --no-chat (mixing it here is wrong;
# the chat flag is global per invocation):  scripts/eval_bench.sh --no-chat lambada_openai
DEFAULT="gsm8k,ifeval"
usage() {
  # Relative cost = full set, run TWICE (base+variant). Add a LIMIT arg to shrink any of them.
  cat <<EOF
usage: eval_bench.sh [--variant base|best|trialN] [--strength 0..1] <tasks|all> [limit]

  --variant   which ablated trial to test (default best); same names as serve.py.
  --strength  ablation scale, 0=base .. 1=trial as optimized (default 1.0); same as serve.py.
  --fewshot   num_fewshot override (default = each task's yaml default). babi defaults to
              0-shot and floors an -it model -- pass e.g. --fewshot 5 to get it off zero.
  --no-chat   skip the chat template. Default is ON (the model is SERVED via chat, so bench
              it that way). Use --no-chat for raw logprob tasks like lambada_openai, which
              the chat wrapper degrades. Don't mix logprob + generative tasks in one run.
  <tasks>     comma-list of lm-eval task names, or "all" for the default RP suite.
  [limit]     int = q per task/subject (quick check). blank = full sets.

default suite ("all" = $DEFAULT), relative wall-time on the 3090 (each runs base+variant):
  gsm8k           ~~~~  slowest   generative CoT, 1319 q             (most edit-fragile)
  ifeval          ~~    medium    generative, ~540 prompts          (rule-based instr following)

generative PROBES (own engine, NOT lm-eval; honor --variant/--strength; [limit] = #questions):
  tools           ~     fast      15 generative tool-call scenarios  (sharpest lobotomy canary)
  babi            ~     fast      state/spatial QA, word-boundary    (--fewshot N bakes exemplars)

run SEPARATELY with --no-chat (logprob tasks, chat template degrades them):
  lambada_openai  ~~    medium    5k single-token logprob conts     (long-range coherence canary)
  scripts/eval_bench.sh --no-chat truthfulqa_mc2,piqa,boolq,openbookqa    (~ fast logprob MCQ)

other generative tasks (chat, pass as args): mmlu (~~~ 57 subj), mmlu_pro/bbh/drop (~~~~).
full list: lm_eval --tasks list

note: lm-eval's own `babi` floors at 0 for chat -it models (leading-space gold); this `babi`
  routes to the word-boundary probe instead.
tip: start with a small limit (e.g. \`gsm8k 20\`) to sanity-check before a full run.
example: scripts/eval_bench.sh --fewshot 3 --variant trial12 babi 100
EOF
}
[ $# -eq 0 ] && { usage; exit 0; }

VARIANT=best
STRENGTH=1.0
FEWSHOT=""        # babi: exemplars baked into each prompt; lm-eval tasks: num_fewshot
NOCHAT=0
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --variant)  VARIANT="$2";  shift 2;;
    --strength) STRENGTH="$2"; shift 2;;
    --fewshot)  FEWSHOT="$2";  shift 2;;
    --no-chat)  NOCHAT=1;      shift;;
    -h|--help)  usage; exit 0;;
    *)          POS+=("$1");   shift;;
  esac
done
set -- "${POS[@]+"${POS[@]}"}"
[ $# -eq 0 ] && { usage; exit 0; }

source .venv/bin/activate

TASKS="$1"; [ "$TASKS" = all ] && TASKS="$DEFAULT" # explicit task list, or "all" for the suite
LIMIT="${2:-}"                                     # blank = full sets; an int = q per task/subject

# `tools` and `babi` are generative PROBES, not lm-eval tasks (the model must EMIT an answer/call,
# not rank completions). Same two-process base-then-variant shape; LIMIT = #scenarios/#questions.
if [ "$TASKS" = tools ]; then
  python -m depurple.tool_probe gen base
  python -m depurple.tool_probe gen "$VARIANT" --strength "$STRENGTH"
  python -m depurple.tool_probe grade base "$VARIANT"
  exit 0
fi

# `babi` -> word-boundary state/spatial probe (NOT lm-eval babi, whose leading-space gold floors
# exact_match at 0 for a chat -it model). Snapshot real rows, then probe via state_probe honoring
# --variant/--strength/--fewshot. LIMIT = #questions; --fewshot N bakes N solved exemplars in.
if [ "$TASKS" = babi ]; then
  python -m depurple.babi_snapshot ${LIMIT:+--n "$LIMIT"} ${FEWSHOT:+--fewshot "$FEWSHOT"}
  export STATE_SCEN=depurple/babi_scenarios.jsonl
  # base doesn't change between variant sweeps -- reuse its reply dump if it still matches THESE
  # scenarios (fingerprint keyed; a different --n/--fewshot busts it). grade reads the cached base, so
  # it still reports. The variant always re-runs.
  if python -m depurple.state_probe fresh base; then
    echo "base already measured for these scenarios -- skipping base pass (rm the base reply dump depurple/babi-replies-base-*.jsonl to force)"
  else
    python -m depurple.state_probe gen base
  fi
  python -m depurple.state_probe gen "$VARIANT" --strength "$STRENGTH"
  python -m depurple.state_probe grade base "$VARIANT"
  exit 0
fi

SLUG="$(python -c 'from depurple._model import SLUG; print(SLUG)')"  # single source (no re-slug)
OUT="depurple/bench-${SLUG}"

EXTRA=()          # format levers for the lm-eval path (the probes above don't use them)
[ -n "$FEWSHOT" ] && EXTRA+=(--fewshot "$FEWSHOT")
[ "$NOCHAT" = 1 ] && EXTRA+=(--no-chat)

# sequential, one process each (eval.py builds the variant in memory via serve.load): base
# first, then the chosen variant. Separate processes so base and the edit never co-reside.
python -m depurple.eval --variant base --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"} \
  "${EXTRA[@]+"${EXTRA[@]}"}" --output_path "$OUT/base"
python -m depurple.eval --variant "$VARIANT" --strength "$STRENGTH" --tasks "$TASKS" \
  ${LIMIT:+--limit "$LIMIT"} "${EXTRA[@]+"${EXTRA[@]}"}" --output_path "$OUT/$VARIANT"

echo
python -m depurple.bench_compare "$OUT/base" "$OUT/$VARIANT"
