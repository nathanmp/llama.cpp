# llama-expert-profiler

Profiles which routed experts a MoE model selects while processing a prompt, so
you can see how concentrated expert usage is before committing to an
expert-aware placement scheme (e.g. pinning the hottest experts in VRAM and
keeping cold ones in RAM).

It hooks the backend eval callback (the same mechanism `llama-imatrix` uses) and
counts the expert ids passed to each layer's `ffn_down_exps` `mul_mat_id`, giving
one tally per routing decision per layer. No graph changes are needed, so it
works for any MoE architecture that goes through `build_moe_ffn`.

## Usage

```sh
llama-expert-profiler -m model.gguf -f prompt.txt [-c 4096] [-o expert-usage.csv]
```

- `-p` / `-f` supply the prompt (longer / more representative is better).
- `-c` sets the context window; prompts longer than it are processed in
  independent windows.
- `-o` is the CSV dump path (default `expert-usage.csv`).
- Standard `-ngl`, `-t`, `--numa`, etc. from the common args also apply.

## Output

Printed to stderr:

- overall counts and the fraction of completely unused `(layer, expert)` units
- a VRAM coverage curve: what share of activations you capture if the hottest
  N% of expert units are pinned
- a histogram of per-unit hotness (relative to a uniform baseline)
- the hottest units and a per-layer skew summary

The CSV holds the full `layer,expert,count,frac_of_layer` table for offline
plotting or driving a placement plan.
