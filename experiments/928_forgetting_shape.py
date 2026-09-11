"""Is the shape of forgetting -- not just how much -- the same per seed across rule, or does it
change with the rule, the scenario, or both?

900-series PLOT SCRIPT. Loads 803's saved per-seed traces, trains nothing.

Applies `classify_forgetting` (src/metrics.py) to each seed's post-switch task-1 trace, across
the full 2x2 (10 seeds x 2 scenarios x 2 methods = 40 traces). Two things this can show that the
mean-only headline numbers (332, 911) cannot: whether the SAME seed gets the same shape under
both rules (a data/split effect, not a rule effect, if so), and whether one scenario is more
"mixed" in shape than its mean number suggests.

PROVENANCE: 803_mechanism_logged_{scenario}.npz -- steps_i, t1_i, switch0_i per row i, methods
and seeds arrays give the (method, seed) for each row.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import classify_forgetting

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
METHODS = ["backprop", "pc"]
COLOR = {"backprop": "black", "pc": "tab:red"}
LABELS = ["collapse", "delayed", "partial", "rising", "noisy"]


def load(scenario):
    return np.load(EXP / f"803_mechanism_logged_{scenario}.npz", allow_pickle=True)


def classify_all(scenario):
    d = load(scenario)
    methods, seeds = d["methods"], d["seeds"]
    out = {}
    for i, (m, s) in enumerate(zip(methods, seeds)):
        steps = d[f"steps_{i}"]
        t1 = d[f"t1_{i}"]
        switch0 = int(d[f"switch0_{i}"])
        post = t1[steps >= switch0]
        out[(str(m), int(s))] = classify_forgetting(post)
    return out


if __name__ == "__main__":
    results = {s: classify_all(s) for s in SCENARIOS}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharey=True)
    width = 0.35
    x = np.arange(len(LABELS))
    for ax, scenario in zip(axes, SCENARIOS):
        r = results[scenario]
        for j, method in enumerate(METHODS):
            counts = [sum(1 for (m, s), lab in r.items() if m == method and lab == L)
                      for L in LABELS]
            ax.bar(x + (j - 0.5) * width, counts, width, color=COLOR[method], label=method,
                   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_title(NICE[scenario], fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("seeds (of 10)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Forgetting shape, per seed, both rules", fontsize=10)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    print("\n  shape counts:")
    for scenario in SCENARIOS:
        r = results[scenario]
        print(f"  {NICE[scenario]}")
        for method in METHODS:
            counts = {L: sum(1 for (m, s), lab in r.items() if m == method and lab == L)
                      for L in LABELS}
            print(f"    {method:9s} " + "  ".join(f"{L}={c}" for L, c in counts.items() if c))

    print("\n  per-seed agreement between rules (same seed, backprop vs pc):")
    for scenario in SCENARIOS:
        r = results[scenario]
        seeds = sorted({s for (m, s) in r if m == "backprop"} & {s for (m, s) in r if m == "pc"})
        agree = sum(1 for s in seeds if r[("backprop", s)] == r[("pc", s)])
        print(f"  {NICE[scenario]:10s} {agree}/{len(seeds)} seeds get the SAME shape label "
              f"under both rules")
        for s in seeds:
            bp_l, pc_l = r[("backprop", s)], r[("pc", s)]
            flag = "" if bp_l == pc_l else "  <- differs"
            print(f"    seed {s:2d}  backprop={bp_l:9s} pc={pc_l:9s}{flag}")
