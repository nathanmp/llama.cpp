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
`fprintf`, so it survives llama's log routing which suppresses ggml's own `GGML_LOG_DEBUG`):

```
=== [GGML_SCHED_CONCURRENT] split structure: 34 splits ===
  split   4  backend=ROCm0  nodes=12  inputs=1  out=ffn_moe_split_out_hot
         in[0] <- ffn_moe_split_idsB       (from CPU)      <- cold ids come from a CPU split (good)
  split   5  backend=CPU    nodes=4   inputs=2  out=ffn_moe_split_out_cold
         in[0] <- ffn_norm-4               (from ROCm0)
         in[1] <- ffn_moe_split_idsB       (from CPU)
...
```

This shows, per split, its backend and where each input comes from. If the cold split (`out_cold`,
CPU) has an input `[depends on prev split]` pointing at the hot split (`out_hot`, GPU), that
dependency is what blocks overlap.

## hot/cold split independence (why G was 0, and the fix)

For the two expert matmuls to overlap, the CPU-cold split must not depend on anything the GPU-hot
split computes. By default the router (`selected_experts`) and the cold-side id cast (`idsB`) are
cheap ops derived from the GPU-resident router; the scheduler's "expand gpu down" pass sweeps them
onto the GPU and **fuses them into the hot split**. The cold split then consumes `idsB` from inside
the hot split, so `ggml_backend_sched_concurrent_group_end` correctly refuses to overlap them (it
would be a data race) and you get `G == 0`. This is exactly what the first rig run showed.

**Fix (implemented):** when `GGML_SCHED_CONCURRENT` is set, `build_moe_ffn`
(`src/llama-graph.cpp`) pins `selected_experts`, `idsB_f` and `idsB` to the CPU backend. That makes
the routing a split that **precedes both branches**, so hot and cold each consume routing outputs
produced *before* either matmul and become mutually independent. It is placement only (no change to
the math) and only active with the concurrency flag, so the default split path is untouched. Verify
with the `GGML_SCHED_DEBUG=1` dump that `out_hot` (GPU) and `out_cold` (CPU) no longer depend on each
other and that `G` jumps to roughly one group per MoE layer.

If `G` is still 0 with the pin on, paste the split-structure dump — the remaining dependency will be
visible in it (some other cold-branch input still resolves to the hot split) and the pin set can be
extended to cover it.

## Not yet handled (M1)

- Event-based join and the `n_copies > 1` pipeline-parallel path (single-stream decode uses
  `n_copies == 1`, so this is inert for the target workload).
- HIP/CUDA graph capture across the concurrent region — if the rig uses graph capture, disable it
  for the concurrent run (`GGML_CUDA_DISABLE_GRAPHS=1` or the HIP equivalent) until M1 handles
  per-backend capture.
- Multiple GPU backends in one group (M3, multi-GPU placement).
