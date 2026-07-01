# Expert-parallel MoE for single-stream decode - implementation plan

Self-contained brief for a fresh session. Goal: make per-expert device placement (hottest
experts in VRAM, cold in CPU RAM) actually faster than whole-layer offload for single-stream
(batch=1) decode of MoE models that overflow VRAM (GLM-4.7/5, DeepSeek V3.2, Qwen 3.5 397B) on
a 2x EPYC + multi-R9700 (ROCm/HIP) rig.

## Why (the case, established)

- Decode is memory-bandwidth bound; MoE expert weights are large, so each expert read is
  bandwidth-bound even at batch=1 (a ~19 MB Q3 expert at 640 GB/s ~= 30 us >> launch ~10 us).
  So the GPU bandwidth advantage is real for single-stream decode.
- With a concurrent executor (per-layer time = max(gpu_hot, cpu_cold)), per-expert placement at
  the same VRAM budget beats whole-layer: ~25-30% when the model mostly fits, up to ~2x for
  heavy overflow (whole-layer forces 50-60% of layers fully onto CPU; per-expert keeps ~85% of
  activations on GPU via routing skew).
- Whole-layer + NUMA is the current validated baseline to beat: ~10 t/s on GLM-4.7 Q3_K_L
  (5x R9700). See decode-placement.md.

## What is already built (branch claude/moe-expert-placement-mxuoeo)

- Graph-level split in `src/llama-graph.cpp` `build_moe_ffn` (moe_split_simple path): runs hot
  experts on `*_exps_hot` (GPU) and cold on the full tensor (CPU), recombined with masks.
  Slimmed to global ids + full cold tensor (no arange/view). Bit-exact vs baseline on OLMoE.
- Load-time hot-copy + hotness reorder in `src/llama-model.cpp` (`LLAMA_MOE_SPLIT`,
  `LLAMA_MOE_PROFILE`), currently gated to `arch == LLM_ARCH_OLMOE` (other arches would waste
  VRAM: the copies are dead unless the graph consumes them).
- Only `src/models/olmoe.cpp` passes the hot tensors into `build_moe_ffn`. GLM etc. do not.
- Profiler + `expert_placement.py` + `bench_placement.sh` in `tools/expert-profiler/`.

## The missing piece and the plan

The blocker is that ggml runs graph splits strictly sequentially with a full
`ggml_backend_synchronize` between them (`ggml/src/ggml-backend.cpp` ~L1549), so hot(GPU) and
cold(CPU) never overlap. Everything else is wiring.

### M0 - concurrent-execution proof on OLMoE (GO/NO-GO on sync cost) [do this first]
Add concurrent execution of mutually-independent splits on distinct backends to
`ggml_backend_sched_compute_splits`: the GPU split is already async
(`ggml_backend_graph_compute_async`); run the independent CPU split on a worker thread (CPU
`graph_compute` is synchronous), and join/sync only before the split that consumes both
outputs (the recombine). Keep a sequential fallback.
- Test: OLMoE (`~/models/OLMoE.gguf`) + `--cpu-moe` + `LLAMA_MOE_SPLIT=0.5` + `--no-mmap`,
  concurrent vs sequential tg. GO if concurrent is meaningfully faster (per-layer sync is
  cheap); NO-GO if flat (sync dominates - stop, keep whole-layer).
- Correctness: `llama-perplexity` bit-exact vs baseline; poison the hot tensor -> PPL explodes
  (proves the path is live).

**Status: scheduler side implemented** (`GGML_SCHED_CONCURRENT=1`, off by default; see
`concurrent-scheduler-m0.md`). Independence-based grouping runs mutually-independent,
distinct-backend splits concurrently (CPU on worker threads, GPU async), with a one-time stderr
diagnostic reporting how many splits overlapped. Verified on CPU that the sequential path is
unchanged and the concurrent flag is inert with a single backend. **Needs a rig run** to (a)
confirm the OLMoE hot/cold splits are seen as independent (`G >= 1` in the diagnostic - if `G == 0`
the routing/`idsB` ops fused into the hot split and need a graph-side split boundary) and (b)
measure the GO/NO-GO tg delta. Cloud sessions cannot drive the GPU.

### M1 - robustify the concurrent scheduler
Generalize safely: event-based join, interaction with the `n_copies` pipeline path and multiple
GPU backends, and HIP-graph capture (disable graphs across the concurrent region or capture per
backend). Auto-fallback to sequential when a split pair is not independent.

### M2 - wire GLM-4.7 (and make hot-copy arch-general)
- `src/models/glm4-moe.cpp`: pass `ffn_*_exps_hot` into `build_moe_ffn`. Shared experts stay
  always-on-GPU (separate from the routed split). GLM's `exp_probs_b` / group top-k only affect
  which experts are selected, not the hot/cold clamp+mask mechanics - the split operates on
  `selected_experts` regardless.
- Replace the `arch == LLM_ARCH_OLMOE` gate in `src/llama-model.cpp` with a per-arch "consumes
  hot tensors" capability (set by the arches that wire it), so hot copies are only allocated
  where used.
- Reuse the existing `LLAMA_MOE_PROFILE` reorder for the hot set.

### M3 - multi-GPU expert placement
Distribute hot experts across GPUs to avoid per-layer inter-GPU thrash; integrate with `-sm`
(layer vs tensor). Single-GPU first (M0-M2), multi-GPU here.

### M4 - tune + benchmark
Derive frac from the VRAM budget (`-fit`) + profile; benchmark GLM-4.7, GLM-5, Qwen 397B vs
whole-layer + NUMA. Target: beat the whole-layer tg from decode-placement.md.

## Risks
- Per-layer cross-device sync cost (the real unknown; M0 gates it).
- Concurrent-scheduler correctness (races, events) - core shared infra, test hard.
- HIP graph capture vs concurrency.
- Duplication: hot experts live in both the CPU full tensor and the GPU hot copy. That is the
  intended VRAM use, but the "no-duplication loader" (load cold experts from a file offset into
  a CPU-only tensor) may be needed to reclaim host RAM for the largest models.

## Verification methodology (reused each milestone)
- Correctness: OLMoE perplexity bit-exact with vs without the split; poison test to prove the
  path is exercised. GPU/quant runs diverge in greedy text by rounding - use perplexity, not
  token-identity.
- Perf: `llama-bench` / `bench_placement.sh` tg on the rig. Cloud sessions cannot drive the GPU;
  validate correctness on CPU, user runs perf.
