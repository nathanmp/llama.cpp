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

# 1. baseline: row split across GPUs (the allreduce-heavy path you have today)
run "baseline (row split)"            -sm row   $(ts_arg)

# 2. layer/pipeline split: no per-layer allreduce - isolates the OcuLink allreduce cost
run "layer split (no allreduce)"      -sm layer $(ts_arg)

# 3. cold experts on CPU (whole-layer placement; approximates the idea at today's granularity)
run "n-cpu-moe=$NCMOE (row)"          -sm row   $(ts_arg) -ncmoe "$NCMOE"
run "n-cpu-moe=$NCMOE (layer)"        -sm layer $(ts_arg) -ncmoe "$NCMOE"

# 4. add NUMA-local binding for the CPU-resident experts (needs a libnuma build; disables mmap)
run "n-cpu-moe=$NCMOE + numa split"   -sm layer $(ts_arg) -ncmoe "$NCMOE" --numa split

echo "done. results in $OUT"
echo
echo "Read the 'tg' (token generation) column. Compare:"
echo "  baseline row vs layer split      -> how much allreduce over OcuLink is costing you"
echo "  layer vs +n-cpu-moe              -> cost of spilling cold experts to CPU"
echo "  +n-cpu-moe vs +numa split        -> value of NUMA-local cold experts"
echo "If the best of these is already near the per-expert ceiling from expert_placement.py,"
echo "the Phase-2 rewrite has little headroom. If a big gap remains, Phase 2 is justified."
