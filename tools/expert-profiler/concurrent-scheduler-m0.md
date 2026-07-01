# M0: concurrent split execution (GO/NO-GO on cross-device sync cost)

This implements the M0 milestone from `expert-parallelism-plan.md`: make the ggml scheduler run
mutually-independent graph splits on distinct backends **concurrently** instead of strictly
sequentially, so the GPU-hot and CPU-cold expert work for a MoE layer can overlap (per-layer time
`max(gpu_hot, cpu_cold)` instead of `gpu_hot + cpu_cold + sync`).

The blocker identified in `decode-placement.md` was that
`ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp`) runs every split sequentially
with a full `ggml_backend_synchronize` between them. This milestone lifts that for independent
splits and gates the whole thing behind an opt-in env var so the default path is byte-for-byte
unchanged.

## What changed

`ggml/src/ggml-backend.cpp`:

- The per-split body of `ggml_backend_sched_compute_splits` is factored into two helpers:
  - `ggml_backend_sched_split_copy_inputs` — copy a split's inputs to its backend and synchronize.
  - `ggml_backend_sched_split_compute` — launch compute (async where supported) and record the event.
  The sequential path (including the `callback_eval` debug path) is unchanged behaviorally.

- `ggml_backend_sched_concurrent_group_end` detects a maximal run of consecutive splits that are
  safe to run at the same time: **pairwise-distinct backends** and **no intra-group data
  dependency** (no split consumes a tensor produced by an earlier member of the run). This makes
  concurrency correct by construction — a split is only overlapped with another when its inputs
  were all produced *before* the run.

- When such a group (size >= 2) is found and `GGML_SCHED_CONCURRENT=1`:
  1. **phase 1** copies the inputs of every split in the group (single-threaded — the expert-copy
     state is shared and not thread-safe). Because the group is dependency-free, doing all copies
     before launching any compute keeps a cross-device input copy from blocking on a backend that
     is already busy with the group's own compute.
  2. **phase 2** launches compute: asynchronous backends (GPU) return immediately on the scheduler
     thread; synchronous host backends (CPU) block, so they run on worker `std::thread`s.
  3. **phase 3** joins the worker threads. The async (GPU) splits stay in flight and are
     synchronized by the consuming split's input copy (or the final graph sync), exactly as before.

- `GGML_SCHED_CONCURRENT` (env, default off) enables the feature. Off => the sequential path, which
  is the exact prior behavior.

## Running the GO/NO-GO test (on the rig — needs a GPU)

Correctness first, then perf. Uses OLMoE, which already wires the hot/cold split
(`src/models/olmoe.cpp` + `LLAMA_MOE_SPLIT`). `llama-bench` has no `--cpu-moe`; use `-ncmoe 999`
(all expert layers' full tensors on CPU), which is the equivalent placement — `LLAMA_MOE_SPLIT`
then puts the hot prefix on the GPU.

```sh
# baseline (sequential) vs concurrent, same placement
LLAMA_MOE_SPLIT=0.5 ./build/bin/llama-bench -m ~/models/OLMoE.gguf -ncmoe 999 -mmp 0 -n 64 -p 0
GGML_SCHED_CONCURRENT=1 LLAMA_MOE_SPLIT=0.5 ./build/bin/llama-bench -m ~/models/OLMoE.gguf -ncmoe 999 -mmp 0 -n 64 -p 0
```

Note the large variance seen on the first `-ncmoe 999` runs (e.g. 20.5 ± 7.07 t/s) — average over
several `-r` repetitions so the GO/NO-GO delta is not swamped by noise.

- **GO**: concurrent tg is meaningfully higher than sequential (per-layer sync is cheap and the
  overlap wins). Proceed to M1 (robustify) / M2 (wire GLM-4.7).
- **NO-GO**: flat => per-layer cross-device sync dominates the overlap. Stop and keep whole-layer
  offload.

Correctness (must pass before trusting any perf number):

```sh
# bit-exact-ish: split ppl should match the no-split baseline (GPU rounding aside)
LLAMA_MOE_SPLIT=0.5 GGML_SCHED_CONCURRENT=1 ./build/bin/llama-perplexity -m ~/models/OLMoE.gguf --cpu-moe 1 -f wiki.test.raw
# poison test: corrupt the hot copy -> ppl must explode, proving the concurrent path is live
```

## FIRST, verify the split actually overlaps (the diagnostic)

When `GGML_SCHED_CONCURRENT=1`, the scheduler prints a one-time line to stderr:

```
ggml_backend_sched_compute_splits: [GGML_SCHED_CONCURRENT] N of M splits ran in G concurrent group(s)
```

- `G >= 1` with `N` covering the hot+cold splits => the MoE layers are overlapping; the perf
  numbers above measure the sync cost as intended.
- `G == 0` (`no independent distinct-backend splits found`) => the hot and cold splits are **not
  independent** as the scheduler sees them, so nothing overlaps and the perf test is inconclusive.

Add `GGML_SCHED_DEBUG=1` to get a one-time dump of the full split structure (printed to stderr via
`fprintf`, so it survives llama's log routing which suppresses ggml's own `GGML_LOG_DEBUG`). It lists
each split's backend and where each input comes from, and prints **before** compute, so it is safe to
`Ctrl-C` right after it to inspect the structure without running a full benchmark.

## Why G is 0: the graph is a strict pipeline (from the rig dump)

The first rig dump (OLMoE, `-ncmoe 999`, 4 GPUs) showed every split marked `[depends on prev split]`
— a strict pipeline, one layer after another:

```
split1 [ROCm: attn + routing + HOT]_L  ->  split2 [CPU: COLD + recombine]_L  ->  split3 [ROCm: ...]_L+1
```

Two fusions block overlap:

1. **Routing is fused into the hot(GPU) split.** The router `ffn_moe_topk` (a.k.a. `selected_experts`)
   is computed on the GPU alongside the hot matmul, so the cold split reads the routing ids from
   *inside* the hot split.
2. **The recombine is fused into the cold(CPU) split.** `split 2` (cold) has an input
   `ffn_moe_split_out_hot-0 (from ROCm0)` — the cold split literally consumes the hot output, because
   the final `add` of `experts = out_hot*maskA + out_cold*maskB` landed on the CPU next to the cold
   matmul. So cold cannot start until hot finishes.

Because of these, `ggml_backend_sched_concurrent_group_end` correctly refuses to overlap anything
(it would be a data race) and reports `G == 0`. The scheduler concurrency itself is fine — the graph
gives it nothing independent.

### gotcha: `selected_experts` is a view

`ggml_argsort_top_k` returns a `ggml_view_4d` of the argsort result, and the scheduler forces a view
onto its source's backend. So pinning `selected_experts` does nothing — you must pin
`selected_experts->view_src` (the underlying `argsort`).

## Experimental independence restructure: `LLAMA_MOE_INDEP=1`

**Status: opt-in, GPU-UNVALIDATED.** `build_moe_ffn` applies two placement-only pins (no math change)
to break both fusions:

1. routing -> CPU: pin the argsort (`selected_experts->view_src`) and the id/mask/weight tensors, so
   routing becomes a split that precedes both branches.
2. recombine -> GPU: pin the final `add` (`experts`) to backend 0 (a GPU), so it is a split *after*
   the cold matmuls instead of fusing into the cold split.

**This approach was tried and REMOVED — it is a dead end.** Two independent problems killed it:

- The first attempt hung a GPU because `selected_experts` is a `ggml_argsort_top_k` *view* and the
  scheduler forces views onto their source's backend, so the pin was a no-op that left a half-pinned
  graph.
- Pinning routing to the CPU **breaks the CUDA/ROCm topk-moe fusion** (`ggml_cuda_topk_moe_fusion`,
  `ggml/src/ggml-cuda/ggml-cuda.cu`), which fuses argsort+softmax+get_rows into one GPU kernel.
  Splitting that pattern across CPU/GPU leaves the fused kernel reading wrong-device pointers ->
  `ROCm error: an illegal memory access was encountered`.

So device pinning cannot expose the independence without fighting the routing fusion. `LLAMA_MOE_INDEP`
has been removed; `GGML_SCHED_CONCURRENT` is scheduler-only and safe.

## The correct next step: a split-boundary hint (no device change)

The right mechanism keeps every tensor on its natural backend (routing stays on the GPU, fusion
intact) and only tells the scheduler to **start a new split** at two points: after the routing/id/mask
prep, and after the cold matmul. That yields:

```
[routing+ids : GPU]  ->  ( [hot : GPU]  ||  [cold : CPU] )  ->  [recombine]
```

with hot and cold each consuming the routing outputs produced in the *preceding* split, so they are
mutually independent and the concurrent scheduler overlaps them — without moving anything between
devices and without touching the topk-moe fusion. The boundary must sit *after* the fusable
argsort/softmax/get_rows region so the fusion is preserved.

Implementing this cleanly needs a small scheduler addition (a per-node "force new split" mark checked
in `ggml_backend_sched_split_graph`'s split loop) plus two marks in `build_moe_ffn`. It changes only
where the graph is cut, not what runs where, so it is correctness-preserving by construction and
cannot cause the illegal-access seen above — but it is core scheduler code and needs rig validation.

Use a **single GPU** (`--device ROCm0`) for the M0 proof: the multi-GPU dump showed layers spread
across ROCm0..3 (layer-parallel), which adds cross-GPU dependencies that are M3's problem, not M0's.

## Not yet handled (M1)

- Event-based join and the `n_copies > 1` pipeline-parallel path (single-stream decode uses
  `n_copies == 1`, so this is inert for the target workload).
- HIP/CUDA graph capture across the concurrent region — if the rig uses graph capture, disable it
  for the concurrent run (`GGML_CUDA_DISABLE_GRAPHS=1` or the HIP equivalent) until M1 handles
  per-backend capture.
- Multiple GPU backends in one group (M3, multi-GPU placement).
