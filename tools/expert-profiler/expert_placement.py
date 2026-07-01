#!/usr/bin/env python3
"""Turn an expert-usage.csv (from llama-expert-profiler) into a placement plan.

It answers two questions for the Phase-1 go/no-go on expert-aware placement:

  1. What --n-cpu-moe baseline should I benchmark at my VRAM budget?
  2. How much would true per-expert placement add over that baseline (the ceiling)?

Key fact this encodes: every MoE layer routes the same total number of tokens, so
whole-tensor offload (-ot / --n-cpu-moe, which moves entire layers' experts) gives GPU
activation coverage equal to the GPU *layer* fraction - it cannot exploit the per-expert
skew at all. Per-expert placement can; the gap between the two is exactly what a Phase-2
rewrite would buy.
"""

import argparse
import csv
import math
from collections import defaultdict


def load_counts(path):
    # layer -> {expert -> count}
    layers = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            layers[int(row["layer"])][int(row["expert"])] = int(row["count"])
    return layers


def per_expert_coverage(unit_counts, total, fracs):
    # coverage if the hottest `frac` of expert units are pinned to GPU
    ordered = sorted(unit_counts, reverse=True)
    n = len(ordered)
    return [sum(ordered[:int(round(f * n))]) / total if total else 0.0 for f in fracs]


def whole_layer_coverage(layer_totals, total, fracs, n_layers):
    # best whole-tensor placement: keep the highest-load layers (equal size) on GPU
    ordered = sorted(layer_totals.values(), reverse=True)
    out = []
    for f in fracs:
        gpu_layers = int(round(f * n_layers))
        out.append(sum(ordered[:gpu_layers]) / total if total else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="expert-usage.csv from llama-expert-profiler")
    ap.add_argument("--vram-budget", type=float, default=0.70,
                    help="fraction of routed-expert weight to keep in VRAM (default 0.70)")
    args = ap.parse_args()

    layers = load_counts(args.csv)
    n_layers = len(layers)
    n_expert = max(len(v) for v in layers.values())

    unit_counts = [c for v in layers.values() for c in v.values()]
    total = sum(unit_counts)
    if total == 0:
        raise SystemExit("no routing counts in CSV")
    n_unused = sum(1 for c in unit_counts if c == 0)

    layer_totals = {il: sum(v.values()) for il, v in layers.items()}
    lo, hi = min(layer_totals.values()), max(layer_totals.values())

    print(f"layers              : {n_layers}")
    print(f"experts per layer   : {n_expert}")
    print(f"expert units        : {len(unit_counts)}")
    print(f"routing decisions   : {total}")
    print(f"unused units        : {n_unused} ({100.0*n_unused/len(unit_counts):.1f}%)")
    spread = ("uniform" if hi - lo <= 0.02 * hi else
              "varies - last layer is undersampled during prefill (output-id pruning)")
    print(f"per-layer load spread: {lo}..{hi} ({spread})")

    # coverage curve: whole-layer (achievable today) vs per-expert (ceiling)
    fracs = [i / 10 for i in range(1, 11)]
    pe = per_expert_coverage(unit_counts, total, fracs)
    wl = whole_layer_coverage(layer_totals, total, fracs, n_layers)
    print("\n-- GPU activation coverage vs VRAM budget --")
    print(f"{'% expert VRAM':<14}{'whole-layer (today)':<22}{'per-expert (ceiling)':<22}{'gap':<8}")
    for frac, whole, cov in zip(fracs, wl, pe):
        print(f"{int(frac*100):<14}{whole*100:<22.1f}{cov*100:<22.1f}{(cov-whole)*100:<8.1f}")

    # recommendation at the chosen budget
    b = args.vram_budget
    gpu_layers = int(round(b * n_layers))
    n_cpu_moe = n_layers - gpu_layers
    whole = whole_layer_coverage(layer_totals, total, [b], n_layers)[0]
    ceil = per_expert_coverage(unit_counts, total, [b])[0]
    print(f"\n-- recommendation at {int(b*100)}% expert VRAM budget --")
    print(f"baseline to benchmark : --n-cpu-moe {n_cpu_moe}  "
          f"(experts of {n_cpu_moe} layers on CPU, {gpu_layers} on GPU)")
    print(f"whole-layer coverage  : {whole*100:.1f}% of expert activations served from VRAM")
    print(f"per-expert ceiling    : {ceil*100:.1f}%  (Phase 2 target)")
    print(f"potential gain        : {(ceil-whole)*100:.1f} percentage points of activations "
          "moved off CPU")
    if ceil - whole < 0.05:
        print("=> small gap: per-expert split unlikely to be worth the rewrite for this profile.")
    else:
        print("=> meaningful gap: per-expert split could move a real share of CPU work to VRAM.")

    print("\nnote: -ot/--n-cpu-moe place whole expert tensors, so the 'whole-layer' column is the\n"
          "      best achievable today. The per-expert column needs the Phase-2 graph split.")


if __name__ == "__main__":
    main()
