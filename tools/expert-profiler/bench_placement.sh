#!/usr/bin/env bash
# Phase-1 benchmark harness for expert-aware placement.
#
# Measures decode (tg) throughput under the placements that are achievable TODAY, so you can
# see the real ceiling before committing to the Phase-2 per-expert rewrite. In particular it
# isolates two effects:
#   - allreduce cost of tensor/row split over slow links  (baseline row vs layer split)
#   - moving cold experts to CPU / NUMA-local RAM          (--n-cpu-moe, --numa split)
#
# Edit the CONFIG block, then run. Each scenario is a separate llama-bench invocation so the
# results stay readable; comment out any that don't apply to your box.

set -u

# ---- CONFIG ----------------------------------------------------------------
BENCH=${BENCH:-./build/bin/llama-bench}
MODEL=${MODEL:?set MODEL=/path/to/model.gguf}
NGL=${NGL:-999}              # offload all layers to GPU; experts move back via --n-cpu-moe
TS=${TS:-}                   # llama-bench tensor-split (SLASH-separated), e.g. "1/1/1/1/1/0.5/0.5"
NCMOE=${NCMOE:-5}            # from expert_placement.py recommendation for your VRAM budget
PP=${PP:-512}               # prompt (prefill) length to bench
TG=${TG:-128}               # tokens to generate (decode) - the number that matters here
REPS=${REPS:-3}
OUT=${OUT:-placement-bench.txt}
# ----------------------------------------------------------------------------

ts_arg() { [ -n "$TS" ] && echo "-ts $TS"; }

run() {
    local name=$1; shift
    echo "=================================================================="
    echo "## $name"
    echo "   args: $*"
    echo "=================================================================="
    "$BENCH" -m "$MODEL" -ngl "$NGL" -p "$PP" -n "$TG" -r "$REPS" "$@" 2>&1
    echo
} | tee -a "$OUT"

: > "$OUT"
echo "model: $MODEL" | tee -a "$OUT"
echo "pp=$PP tg=$TG reps=$REPS ngl=$NGL ncmoe=$NCMOE ts=${TS:-none}" | tee -a "$OUT"
echo | tee -a "$OUT"

# 1. baseline: tensor split across GPUs (your best path; supersedes row split)
run "baseline (tensor split)"         -sm tensor $(ts_arg)

# 2. layer split, for comparison (you measured tensor > layer, but worth confirming on this rig)
run "layer split"                     -sm layer  $(ts_arg)

# 3. cold experts on CPU (whole-layer placement; approximates the idea at today's granularity)
run "n-cpu-moe=$NCMOE (tensor)"       -sm tensor $(ts_arg) -ncmoe "$NCMOE"

# 4. add NUMA-local binding for the CPU-resident experts (needs a libnuma build; disables mmap)
run "n-cpu-moe=$NCMOE + numa split"   -sm tensor $(ts_arg) -ncmoe "$NCMOE" --numa split

echo "done. results in $OUT"
echo
echo "Read the 'tg' (token generation) column. Compare:"
echo "  baseline vs n-cpu-moe            -> cost of spilling cold experts to CPU at your budget"
echo "  n-cpu-moe vs +numa split         -> value of NUMA-local cold experts"
echo "The n-cpu-moe tg is the whole-layer baseline. expert_placement.py shows the per-expert"
echo "ceiling at the same budget; the gap between them is the most Phase 2 could recover."
echo "If most of the model already fits in VRAM (small -ncmoe), that gap is small and Phase 2"
echo "buys little - re-check this once all your GPUs are connected."
