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
