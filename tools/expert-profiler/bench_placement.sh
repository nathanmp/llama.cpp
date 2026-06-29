#!/usr/bin/env bash
# Stage-0 decode placement sweep for hybrid GPU+CPU MoE inference.
#
# Goal: find the real single-stream decode (tg) ceiling using only mechanisms that exist
# today, before investing in a concurrent dual-backend executor. Each scenario isolates one
# lever. Read the 'tg' (token generation) column; everything else is context.
#
# Why these scenarios (see decode-placement.md for the full rationale):
#   - whole-layer offload keeps every layer on a single device -> no intra-layer GPU<->CPU
#     barrier. This is the baseline the per-expert split must beat.
#   - the per-expert split (LLAMA_MOE_SPLIT) runs hot experts on GPU and cold on CPU within a
#     layer; ggml runs those splits sequentially (no overlap), so at batch=1 it is expected to
#     LOSE here. We measure it to confirm the diagnosis, not because it should win.
#   - expert reduction (fewer active experts) cuts the bytes read per token directly.
#
# Edit the CONFIG block, then run. Comment out scenarios that don't apply.

set -u

# ---- CONFIG ----------------------------------------------------------------
BENCH=${BENCH:-./build/bin/llama-bench}
CLI=${CLI:-./build/bin/llama-cli}     # for the expert-reduction sweep (set to llama-completion if preferred)
MODEL=${MODEL:?set MODEL=/path/to/model.gguf}
ARCH=${ARCH:-}                        # gguf arch name for --override-kv, e.g. glm4moe / qwen3moe / olmoe (empty -> skip EUC sweep)

NGL=${NGL:-999}                       # offload all layers; experts move back via -ncmoe / -ot
TS=${TS:-}                            # tensor-split across GPUs, slash-separated, e.g. "1/1/1/1/1/0.5/0.5"
DEV=${DEV:-}                          # restrict devices, e.g. "ROCm0" (single GPU) - blank = all
NCMOE_SWEEP=${NCMOE_SWEEP:-0,8,16,24,32}   # llama-bench list: 0 = all experts on GPU, high = all on CPU
FIT_MARGIN=${FIT_MARGIN:-1024}        # -fitt margin (MiB/device) for the autofit ceiling run
EUC_SWEEP=${EUC_SWEEP:-}              # expert_used_count values for the reduction sweep, e.g. "8 6 4"
SPLIT_FRAC=${SPLIT_FRAC:-0.5}         # LLAMA_MOE_SPLIT fraction for the structural-confirmation run
PROFILE=${PROFILE:-}                  # optional expert-usage.csv for the split run (needs --no-mmap)

PP=${PP:-512}
TG=${TG:-128}
REPS=${REPS:-3}
OUT=${OUT:-placement-bench.txt}
# ----------------------------------------------------------------------------

ts_arg()  { [ -n "$TS"  ] && printf -- "-ts %s " "$TS"; }
dev_arg() { [ -n "$DEV" ] && printf -- "-dev %s " "$DEV"; }

bench() {
    local name=$1; shift
    {
        echo "=================================================================="
        echo "## $name"
        echo "   env : ${RUNENV:-}"
        echo "   args: $*"
        echo "=================================================================="
        env ${RUNENV:-} "$BENCH" -m "$MODEL" -ngl "$NGL" -p "$PP" -n "$TG" -r "$REPS" $(ts_arg) $(dev_arg) "$@" 2>&1
        echo
    } | tee -a "$OUT"
}

# decode-only timing via the CLI, used where llama-bench lacks a knob (--override-kv).
# prints the eval (token generation) tokens/sec line.
cli_tg() {
    local name=$1; shift
    {
        echo "=================================================================="
        echo "## $name"
        echo "   args: $*"
        echo "=================================================================="
        env ${RUNENV:-} "$CLI" -m "$MODEL" -ngl "$NGL" -n "$TG" -c "$((PP+TG))" \
            -p "Once upon a time" --temp 0 $(ts_arg) $(dev_arg) "$@" 2>&1 \
            | grep -E "eval time|tokens per second" || echo "  (no perf line - check the run)"
        echo
    } | tee -a "$OUT"
}

: > "$OUT"
{
    echo "model : $MODEL"
    echo "arch  : ${ARCH:-<unset>}   pp=$PP tg=$TG reps=$REPS ngl=$NGL ts=${TS:-none} dev=${DEV:-all}"
    echo "date  : $(date)"
    echo
} | tee -a "$OUT"

# 1. reference: all experts on CPU (today's --cpu-moe baseline). Big -ncmoe -> every layer on CPU.
bench "ref: all experts on CPU"                 -sm tensor -ncmoe 9999

# 2. autofit ceiling: let -fitt place as many whole layers on GPU as fit. The clean, no-barrier
#    target the per-expert split must beat.
bench "autofit (-fitt ${FIT_MARGIN})"           -sm tensor -fitt "$FIT_MARGIN"

# 3. whole-layer offload sweep: how tg scales as more layers' experts spill to CPU.
bench "n-cpu-moe sweep {$NCMOE_SWEEP}"           -sm tensor -ncmoe "$NCMOE_SWEEP"

# 4. NUMA-local cold experts (needs libnuma build; disables mmap).
bench "n-cpu-moe sweep + numa split"             -sm tensor -ncmoe "$NCMOE_SWEEP" --numa split

# 5. structural confirmation of the per-expert split. Expect tg << ref/autofit, and op-offload
#    (a prefill knob) to make no decode difference. Uses the LLAMA_MOE_SPLIT env path.
mmap_arg="--mmap 1"; [ -n "$PROFILE" ] && mmap_arg="--mmap 0"
RUNENV="LLAMA_MOE_SPLIT=$SPLIT_FRAC ${PROFILE:+LLAMA_MOE_PROFILE=$PROFILE}" \
    bench "per-expert split frac=$SPLIT_FRAC (op-offload on/off)" -sm tensor -ncmoe 9999 $mmap_arg -nopo 0,1
RUNENV=""

# 6. expert reduction (fewer active experts/token). llama-bench has no --override-kv, so use CLI.
if [ -n "$ARCH" ] && [ -n "$EUC_SWEEP" ]; then
    for euc in $EUC_SWEEP; do
        cli_tg "expert_used_count=$euc" -ncmoe 9999 --override-kv "${ARCH}.expert_used_count=int:${euc}"
    done
else
    echo "(skipping expert-reduction sweep: set ARCH and EUC_SWEEP)" | tee -a "$OUT"
fi

{
    echo
    echo "done. results in $OUT"
    echo
    echo "How to read (tg column / 'tokens per second'):"
    echo "  ref (all CPU)         - today's floor with everything on CPU experts"
    echo "  autofit / n-cpu-moe   - whole-layer offload ceiling (no intra-layer barrier)"
    echo "  per-expert split      - expected to be WORSE at batch=1 (sequential GPU+CPU, no overlap)"
    echo "  expert_used_count     - speed vs quality tradeoff from reading fewer experts"
    echo
    echo "Decision gate for the dual-backend executor (Stage 1):"
    echo "  compare best whole-layer tg against the per-expert ceiling from expert_placement.py"
    echo "  at the same VRAM budget. If the gap is large, an executor that overlaps GPU-hot and"
    echo "  CPU-cold work can recover it; if small, stay on whole-layer offload."
} | tee -a "$OUT"
