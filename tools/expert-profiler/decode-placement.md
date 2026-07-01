# Stage 0: single-stream decode placement

Where to put MoE experts (GPU vs CPU) for **batch=1 decode** when the model overflows VRAM.
`bench_placement.sh` runs the sweep; this explains what it measures and how to decide what to
build next.

## The problem with the per-expert split

The graph-level per-expert split (hot experts on GPU, cold on CPU within each layer, recombined
with masks - enabled by `LLAMA_MOE_SPLIT`) is ~4x slower than all-CPU experts at decode on the
rig (2.0 vs 8.5 tg). It is not a bug in the placement; it is structural:

- ggml runs graph splits **strictly sequentially** with a full backend sync between them
  (`ggml/src/ggml-backend.cpp:1549`), so the GPU-hot and CPU-cold work for a layer cannot
  overlap - each layer pays `cpu_cold + gpu_hot + sync + recombine copies`, which is more than
  all-CPU's `cpu_full`.
- At decode the GPU does not offload host-resident expert matmuls anyway (offload is gated at
  batch >= 32, `ggml/src/ggml-cuda/ggml-cuda.cu:5498`), so cold experts already run on CPU and
  `--no-op-offload` changes nothing for decode (it is a prefill knob).
- Multiple GPUs make it worse (more cross-device traffic): measured 4-GPU 2.0 vs 1-GPU 4.1 tg.

Implication: the per-expert split can only win at batch=1 if GPU and CPU work run concurrently
(layer time `max(gpu_hot, cpu_cold)` instead of the sum). That needs a dual-backend executor.
Before building it, measure whether the headroom justifies it.

## What the sweep measures

Read the `tg` (token generation) column; `pp` is prefill context only.

| scenario | what it isolates |
|---|---|
| ref: all experts on CPU | today's floor; everything on CPU experts |
| autofit (`-fitt`) | whole-layer offload ceiling - fits as many full layers on GPU as memory allows, no intra-layer barrier |
| n-cpu-moe sweep | how tg scales as more layers' experts spill from GPU to CPU |
| + numa split | value of NUMA-local cold experts (needs a libnuma build; disables mmap) |
| per-expert split | confirmation it is worse at batch=1; op-offload on/off should not change decode |
| expert_used_count | speed vs quality from reading fewer experts per token |

## Decision gate (do we build the Stage 1 executor?)

1. Take the best whole-layer `tg` (autofit or the best `n-cpu-moe`).
2. Run `expert_placement.py --csv expert-usage.csv --vram-budget <same budget>` for the
   per-expert activation-coverage ceiling at that budget.
3. The bandwidth headroom is roughly `(per-expert coverage - whole-layer coverage)` of the
   cold-expert bytes. If that gap is large (the profiler showed ~33pt at a ~49% budget), a
   concurrent executor that overlaps GPU-hot and CPU-cold work can recover much of it. If the
   gap is small (most of the model already fits in VRAM), stay on whole-layer offload.

## Notes

- Use `llama-perplexity`, not greedy token-identity, to check correctness on GPU/quantized
  runs: GPU and CPU expert math round differently, so greedy text legitimately diverges even
  when the computation is equivalent.
- `expert_used_count` overrides need the gguf arch prefix (set `ARCH`), e.g.
  `--override-kv glm4moe.expert_used_count=int:6`. llama-bench has no `--override-kv`, so the
  script runs these through the CLI and parses the eval tokens/sec line.
- The per-expert split run needs `--no-mmap` if you pass a `PROFILE` (reorder rewrites weights
  in place); without a profile it splits by file order.
