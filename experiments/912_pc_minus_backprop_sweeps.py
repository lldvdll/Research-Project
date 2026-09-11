"""Is PC's advantage over backprop stable across every axis we can vary?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R2 -- the section's headline.

ONE PLOT, one grid cell. Form is 007's SK_CONSOL: paired PC - backprop against lr, width, depth
AND activation, four panels sharing one y-axis, a zero line drawn, one line per scenario. The
three continuous sweep figures it consolidates go to the appendix as 914 -- this replaces them
rather than joining them. Activation (802) was added after the fact: it is categorical, not
swept, so it gets its own x-axis handling, but it earns the same panel treatment as the other
three because it is the one axis where Domain-IL's sign FLIPS rather than just moving (sigmoid,
+0.74 +- 0.19) -- exactly the kind of thing this figure exists to show and would otherwise hide
in a methods footnote.

WHY THE PAIRED DIFFERENCE AND NOT THE ABSOLUTE VALUE. Crossover's raw value is not stopping-point
independent (907 shows it drifting +29.6 across the threshold range), so an absolute crossover
plotted against a swept axis mixes the sweep's effect with the protocol's. At a given seed both
rules see the same class split and the same initialisation, so the paired difference cancels the
shared variance: 341's Class-IL retention ranges 40 points across seeds while the paired
difference has a SEM of 0.39 at the same point. 906 measures that directly.

THE READING. Above the zero line PC retains more than backprop; below it, less.

    working point (H = 32, depth 1, tanh)   Class-IL +1.42 ± 0.39     Domain-IL −0.72 ± 0.19
    depth 4                                 Class-IL +3.44 ± 0.85     Domain-IL −3.49 ± 0.61
    width 4                                 Class-IL −4.14 ± 0.70     Domain-IL +2.21 ± 0.46
    activation = sigmoid                    Class-IL +1.51 ± 0.91     Domain-IL +0.74 ± 0.19

Class-IL sits above zero and Domain-IL below it everywhere except the narrowest width, where both
reverse, AND under sigmoid, where Domain-IL alone reverses (large, real: Cohen's d = 1.25 -- see
808) while Class-IL stays positive as everywhere else. The effect is not fragile -- it is
systematic, and depth widens the split in both directions at once, which contradicts the pre-100
"depth doesn't matter" line.

⚠ CENSORED POINTS ARE MARKED, NOT AVERAGED THROUGH. At lr 0.16 the Class-IL crossover is undefined
on 10/10 backprop seeds and 6/10 PC seeds, so no paired difference exists there and the line
stops. A line that simply stops reads as "not run"; it was run, and it is the metric that ran out.
907's panel (b) carries the full censoring story.

Styling follows the 340/341/342 scripts whose arrays these are: tab: colours, dpi 120,
bbox_inches="tight", 9pt labels, 8pt legends. No shared style module.

PROVENANCE -- all three sweeps, config_300.yaml, seeds 10-19, 90% matched competence:
    340_lr_sweep_{scenario}.npz     shared absolute lr grid for every rule
    341_width_sweep_{scenario}.npz  H in {4, 8, 16, 32, 64}, each rule at its established lr
    342_depth_sweep_{scenario}.npz  1-4 hidden layers
    802_activation_sweep.npz        tanh/sigmoid/relu, H=32/depth=1, config_800.yaml, seeds 10-19
341 and 342 were re-run at the corrected dt = 0.2 after 346/347 found dt = 0.4 oscillates at
depth >= 2 (see 905). 340 stays at dt = 0.4, correct at its fixed H = 32 / depth 1. Do NOT unify
the dt across the three -- the correct value differs by configuration, and that is itself a result.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff

EXP = ROOT / "experiments"
SWEEPS = [("340_lr_sweep", "learning rate", "log"),
          ("341_width_sweep", "hidden width H", "log"),
          ("342_depth_sweep", "hidden layers", "linear")]
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
CX = 5                                   # crossover column in the sweep row layout
WORKING = {"341_width_sweep": 32, "342_depth_sweep": 1}


def by_seed(rows, method, x):
    d = {int(r[2]): float(r[CX]) for r in rows if r[0] == method and r[1] == x}
    return np.array([d[s] for s in sorted(d)], dtype=float)


ACTIVATIONS = ["tanh", "sigmoid", "relu"]


def by_seed_act(rows, scenario, act, method):
    """802's schema differs: (scenario, act, method, seed, final_t1, final_t2, crossover, ...)."""
    d = {int(r[3]): float(r[6]) for r in rows if r[0] == scenario and r[1] == act and r[2] == method}
    return np.array([d[s] for s in sorted(d)], dtype=float)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9), sharey=True)

    for ax, (stem, xlabel, xscale) in zip(axes[:3], SWEEPS):
        ax.axhline(0, color="k", lw=1.2)
        for s in SCENARIOS:
            rows = np.load(EXP / f"{stem}_{s}.npz", allow_pickle=True)["data"]
            grid = sorted({r[1] for r in rows})
            means, sems, cens = [], [], []
            for g in grid:
                bp, pc = by_seed(rows, "backprop", g), by_seed(rows, "pc", g)
                m, se, _ = paired_diff(pc, bp)
                means.append(m)
                sems.append(se)
                cens.append(int((~np.isfinite(bp)).sum() + (~np.isfinite(pc)).sum()))
            gx = np.asarray(grid, float)
            means, sems = np.asarray(means), np.asarray(sems)
            ax.errorbar(gx, means, yerr=sems, marker="o", capsize=3, lw=2,
                        color=COLORS[s], label=NICE[s])
            gone = ~np.isfinite(means)
            if gone.any():
                ax.plot(gx[gone], np.zeros(gone.sum()), marker="x", ms=9, mew=2.0,
                        ls="none", color="crimson", zorder=6)
            print(f"  {stem:16s} {NICE[s]:10s} " + "  ".join(
                f"{g}: {m:+5.2f}±{e:4.2f}" + (f"[{c} cens]" if c else "")
                for g, m, e, c in zip(grid, means, sems, cens)))

        if stem in WORKING:
            ax.axvline(WORKING[stem], color="0.5", ls=":", lw=1.2)
        else:
            ax.axvspan(0.01, 0.02, color="0.5", alpha=0.12)
        ax.set_xscale(xscale)
        # Ticks are the swept values themselves in every panel -- on the linear depth axis
        # matplotlib otherwise invents 1.5, 2.5, which are not configurations that were run.
        ax.set_xticks(grid)
        ax.set_xticklabels([str(g) for g in grid], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(alpha=0.25)

    # Fourth panel: activation (802), categorical -- own x handling, same y-axis.
    ax = axes[3]
    ax.axhline(0, color="k", lw=1.2)
    act_rows = np.load(EXP / "802_activation_sweep.npz", allow_pickle=True)["data"]
    ax_pos = np.arange(len(ACTIVATIONS))
    for s in SCENARIOS:
        means, sems = [], []
        for act in ACTIVATIONS:
            bp = by_seed_act(act_rows, s, act, "backprop")
            pc = by_seed_act(act_rows, s, act, "pc")
            m, se, _ = paired_diff(pc, bp)
            means.append(m); sems.append(se)
        means, sems = np.asarray(means), np.asarray(sems)
        ax.errorbar(ax_pos, means, yerr=sems, marker="o", capsize=3, lw=2,
                    color=COLORS[s], label=NICE[s])
        print(f"  802_activation   {NICE[s]:10s} " + "  ".join(
            f"{a}: {m:+5.2f}±{e:4.2f}" for a, m, e in zip(ACTIVATIONS, means, sems)))
    ax.axvline(0, color="0.5", ls=":", lw=1.2)   # tanh is the working point
    ax.set_xticks(ax_pos)
    ax.set_xticklabels(ACTIVATIONS, fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("activation", fontsize=9)
    ax.grid(alpha=0.25)

    axes[0].set_ylabel("paired PC $-$ backprop crossover (pp)", fontsize=9)
    axes[0].plot([], [], marker="x", ms=8, mew=2.0, ls="none", color="crimson",
                 label="no paired measurement")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Class-IL above zero, Domain-IL below it — except at the narrowest width "
                "(both reverse) and under sigmoid (Domain-IL alone reverses)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
