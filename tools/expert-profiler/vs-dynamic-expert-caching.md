# Static hotness placement vs. FATE-style dynamic expert caching

Context: a separate implementation (FATE) adds a GPU-resident expert *cache* with predictive
prefetch — experts are streamed GPU<-CPU on demand and computed on the GPU, with cross-layer +
temporal prediction to prefetch them. This note explains why, for **this** rig and workload, the
plan's static hotness placement (hot experts permanently on GPU, cold permanently on CPU, computed
concurrently) is the right call, and where the FATE ideas do and don't transfer.

## The regimes are different, and that flips the answer

| | FATE test | This rig |
|---|---|---|
| VRAM fraction of the model | ~15% (784+2047 MB of an 18.6 GB model) | ~60–90% |
| Consequence | ~85% of experts must be *streamed every token* | only a ~10–40% cold tail is not GPU-resident |

FATE's own report is blunt about the outcome in its regime: **1.14 t/s, ~5.6× slower than vanilla
offload (6.4 t/s)**. The root cause it identifies is that the pool (15% of the model) churns
completely every token, so it transfers the *same* PCIe volume as vanilla but through a slower
(staging-`memcpy`, unpinnable-mmap) path. "For FATE to beat vanilla, it must transfer LESS data per
token than vanilla" — which requires the working set to fit in the pool with temporal reuse.

That prerequisite is exactly what a large VRAM fraction gives you — and it is exactly the case the
static split already handles without any caching machinery.

## Why static placement wins the cold tail here

The only experts in question are the cold tail (the ~10–40% not permanently on GPU). Two ways to
serve them:

- **Static + concurrent (the plan):** cold experts live in CPU RAM, are computed on the CPU
  (~270 GB/s combined DDR4-3200 across the two EPYC sockets), and that CPU work overlaps the GPU's
  hot-expert work via the M0 concurrent scheduler. **No PCIe weight transfer at all.**
- **FATE-style (stream + GPU-compute):** cold experts are copied H2D over PCIe (~25 GB/s per link;
  even 4 links ≈ 100 GB/s aggregate) and computed on the GPU. On a cache *hit* this is a fast D2D,
  but the cold tail is cold precisely because it is the low-frequency, low-reuse part of the
  distribution — so it mostly *misses* and pays full PCIe cost.

Bandwidth decides it: computing the cold tail on the CPU at ~270 GB/s beats streaming it over PCIe
at 25–100 GB/s, and it avoids the H2D entirely. FATE's genuine advantage — GPU cache *hits* at
640 GB/s — applies to the *hot* experts, which the static plan already keeps permanently on the GPU
with zero per-token transfer. So dynamic caching would mostly re-implement "hot experts on GPU" for
the hot set while doing something slower than CPU-compute for the cold set.

Static placement is also much simpler: no predictor, no prefetch stream, no pool eviction, no
GPU barriers, no pinned staging — none of the ~500 LoC of moving parts, and none of their failure
modes (mmap-can't-pin, prediction misses, pool thrash).

## What is worth borrowing from FATE — later

1. **Dynamic promotion of "warm" experts.** If M4 profiling shows a per-layer bottleneck where
   `cpu_cold > gpu_hot` (the CPU tail dominates a layer), promoting the *borderline* experts (some
   reuse, just outside the static hot set) to a small GPU cache could shave the tail. This is an
   optimization *on top of* static placement, not a replacement — and only if the data justifies it.
   This is the "dynamic movement" the FATE author did; the plan correctly treats it as a later issue.
2. **Pinned host memory + a dedicated H2D stream** (`fate_prefetch_*` in `ggml-cuda.cu`) are the
   right primitives *if* any expert ever needs streaming (dynamic promotion, or the plan's
   "no-duplication loader"). FATE confirms the key constraint: **mmap'd weights cannot be pinned**
   (`cudaHostRegister` fails on file-backed pages), so any streaming path needs `--no-mmap` — which
   this plan already uses.
3. **Per-kind sizing** (gate/up smaller than down) as a way to fit more experts per VRAM byte —
   relevant to how much of each layer we can pin as "hot".

## Bottom line

Stay on the static hotness-placement + concurrent-execution path. The larger VRAM fraction is
precisely the condition that makes the simple approach win: keep the hot majority resident on the
GPU (no transfer), compute the small cold tail on the CPU at memory-bandwidth speed, and overlap the
two. Revisit FATE-style dynamic promotion only if per-layer profiling later shows the CPU tail is
the bottleneck.
