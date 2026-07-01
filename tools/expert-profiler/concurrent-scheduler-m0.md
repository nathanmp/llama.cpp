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
(`src/models/olmoe.cpp` + `LLAMA_MOE_SPLIT`).

```sh
# baseline (sequential) vs concurrent, same placement
LLAMA_MOE_SPLIT=0.5 ./build/bin/llama-bench -m ~/models/OLMoE.gguf --cpu-moe 1 -mmp 0 -n 64 -p 0
GGML_SCHED_CONCURRENT=1 LLAMA_MOE_SPLIT=0.5 ./build/bin/llama-bench -m ~/models/OLMoE.gguf --cpu-moe 1 -mmp 0 -n 64 -p 0
```

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
  See below.

## Known caveat: hot/cold split independence

For the two expert matmuls to overlap, the CPU-cold split must not depend on anything the GPU-hot
split computes. In `build_moe_ffn` the routing (`selected_experts`) and the cold-side id cast
(`idsB`) are cheap ops derived from the GPU-resident router; the scheduler's "expand gpu down" pass
tends to assign them to the GPU and fuse them into the hot split. The cold split then consumes
`idsB` from inside the hot split, so `ggml_backend_sched_concurrent_group_end` correctly refuses to
overlap them (it would be a data race) and you get `G == 0`.

If the rig shows `G == 0`, the fix is graph-side, not scheduler-side: force the routing / cold-id
ops into a split that precedes the hot split (e.g. pin them so hot and cold both consume routing
outputs produced *before* either matmul). That wiring is best done against a real
`GGML_SCHED_DEBUG=2` split dump from the rig, which lists each split's backend and inputs:

```sh
GGML_SCHED_DEBUG=2 LLAMA_MOE_SPLIT=0.5 ./build/bin/llama-cli -m ~/models/OLMoE.gguf --cpu-moe 1 -n 1 -p hi 2>&1 | grep -A2 'SPLIT #'
```

## Not yet handled (M1)

- Event-based join and the `n_copies > 1` pipeline-parallel path (single-stream decode uses
  `n_copies == 1`, so this is inert for the target workload).
- HIP/CUDA graph capture across the concurrent region — if the rig uses graph capture, disable it
  for the concurrent run (`GGML_CUDA_DISABLE_GRAPHS=1` or the HIP equivalent) until M1 handles
  per-backend capture.
- Multiple GPU backends in one group (M3, multi-GPU placement).
