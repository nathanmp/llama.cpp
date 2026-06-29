# llama-expert-profiler

Profiles which routed experts a MoE model selects while processing a prompt, so
you can see how concentrated expert usage is before committing to an
expert-aware placement scheme (e.g. pinning the hottest experts in VRAM and
keeping cold ones in RAM).

It hooks the backend eval callback (the same mechanism `llama-imatrix` uses) and
counts the expert ids passed to each layer's `ffn_down_exps` `mul_mat_id`, giving
one tally per routing decision per layer. No graph changes are needed, so it
works for any MoE architecture that goes through `build_moe_ffn`.

Both phases are profiled: each prompt is prefilled and then tokens are generated
autoregressively, so decode routing (which dominates for reasoning models) is
counted, not just prefill. The summary reports the prefill / decode token and
decision split so you can confirm generation was actually exercised.

## Usage

```sh
llama-expert-profiler -m model.gguf -f prompts.txt [-c 4096] [-n 256] [-o expert-usage.csv]
```

- `-f` is a prompt file with **one prompt per line**. A literal `\n` inside a
  line is turned into a newline (so multi-line prompts go on a single line as
  `part one\npart two`). `-p` supplies a single prompt instead.
- `-n` is the number of tokens to generate per prompt (default 256); generation
  stops early on EOS.
- `-c` sets the context window; prompts longer than it are truncated.
- `-o` is the CSV dump path (default `expert-usage.csv`).
- Counts are aggregated across all prompts. Sampling/`-ngl`/`-t`/`--numa` and the
  other common args also apply.

## Output

Printed to stderr:

- overall counts and the fraction of completely unused `(layer, expert)` units
- a VRAM coverage curve: what share of activations you capture if the hottest
  N% of expert units are pinned
- a histogram of per-unit hotness (relative to a uniform baseline)
- the hottest units and a per-layer skew summary

The CSV holds the full `layer,expert,count,frac_of_layer` table for offline
plotting or driving a placement plan.

## Placement planning (Phase 1)

Two companions turn the profile into a go/no-go on expert-aware placement, using only
mechanisms that exist today (whole-tensor `-ot` / `--n-cpu-moe`, `--numa split`).

`expert_placement.py` reads the CSV and contrasts the GPU activation coverage you can get
today (whole-tensor offload, which can only place whole layers) against the per-expert
ceiling, at each VRAM budget:

```sh
python3 tools/expert-profiler/expert_placement.py --csv expert-usage.csv --vram-budget 0.70
```

It prints the coverage table, a recommended `--n-cpu-moe N` baseline, and the gap to the
per-expert ceiling. A small gap means a per-expert rewrite has little headroom for this
profile; a large gap means it could move real work off the CPU.

Note: every MoE layer routes the same number of tokens, so whole-tensor offload coverage is
essentially the GPU layer fraction - it cannot exploit per-expert skew. (The last layer reads
slightly lower because of output-id graph pruning during prefill.)

`bench_placement.sh` then measures decode throughput under the achievable-today placements
(row vs layer split to isolate allreduce cost; `--n-cpu-moe`; `--numa split`) so you can
compare the best real number against that ceiling:

```sh
MODEL=/path/model.gguf TS="1/1/1/1/1/0.5/0.5" NCMOE=5 bash tools/expert-profiler/bench_placement.sh
```

Profile with a large, representative run (10-50k generated tokens) before trusting the
numbers - a short sample understates how many experts are really used.

## Decode placement (Stage 0)

For choosing GPU vs CPU expert placement for single-stream decode, see
[decode-placement.md](decode-placement.md) and run `bench_placement.sh`. Short version:
whole-layer offload (`-fitt` / `--n-cpu-moe`) is the ceiling to beat, because each layer
stays on one device and avoids a cross-device barrier.

The experimental load-time per-expert device split (`LLAMA_MOE_SPLIT` /
`LLAMA_MOE_PROFILE`, in `src/llama-model.cpp` + `src/llama-graph.cpp`) is **not recommended
for decode**: it runs hot experts on GPU and cold on CPU within each layer, but ggml runs
those splits sequentially with no overlap, so at batch=1 it is ~8x slower than all-CPU
experts on the rig. It only becomes worthwhile with a concurrent dual-backend executor (see
the plan in `decode-placement.md`). The flags remain for experimentation:

- `LLAMA_MOE_SPLIT=<frac in (0,1)>` copies each simple-MoE layer's hottest `frac` of
  experts into a separate hot tensor on the layer's device.
- `LLAMA_MOE_PROFILE=<expert-usage.csv>` reorders each layer's experts by descending usage
  so the hot prefix holds the actual hottest experts (pure relabeling - router columns and
  expert weights are permuted together - so model output is unchanged). Needs `--no-mmap`.
