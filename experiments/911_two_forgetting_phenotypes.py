"""Does the scenario change the shape of forgetting, and does the learning rule?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R1 -- the load-bearing result.

FOUR PLOTS, ONE PER GRID CELL, in 007's card order:
    911_two_forgetting_phenotypes_a.png   backprop, Class-IL
    911_two_forgetting_phenotypes_b.png   pc,       Class-IL
    911_two_forgetting_phenotypes_c.png   backprop, Domain-IL
    911_two_forgetting_phenotypes_d.png   pc,       Domain-IL

script_plan_800_900.md calls 911 an "assemble 310 + 332" 2x2 with scenario across and rule down.
The four files ARE that 2x2 -- every panel is drawn on IDENTICAL, HARD-CODED axes (0-100 both
ways, equal aspect), so the report composes them with subfigure and the comparison survives.
Limits are set explicitly rather than by sharex/sharey precisely because the panels live in
separate files: a shared scale that depends on being in one figure would silently break.

THE CLAIM IS ABOUT THE GRID, NOT ANY PANEL. Down a column (same scenario, different rule) the
trajectories are near-superimposable. Across a row (same rule, different scenario) they are
different families: Class-IL turns hard left and terminates against the axis, Domain-IL stops on
a shelf well short of it. Changing the scenario changes the shape; changing the rule does not.

    final task 1   Class-IL  5.4 (bp)  6.0 (pc)     Domain-IL  38.5 (bp)  37.0 (pc)
    crossover      Class-IL 65.0 (bp) 66.4 (pc)     Domain-IL  75.8 (bp)  75.1 (pc)

FORM AND STYLING COPIED FROM 310, the script this regenerates: x = task-1 accuracy, y = task-2
accuracy with time removed, ten seed lines coloured by WHICH TASK IS TRAINING (orange then blue),
a dashed green x=y diagonal with each seed's crossing marked, a dashed red line at task-2's stop
threshold with each seed's endpoint marked, and thin marginal histograms for the two
distributions. tab: colours, dpi 120, bbox_inches="tight". No shared style module.

UNITS TRAP: in these arrays `t1`/`t2` are FRACTIONS (0-1) while `crossovers`/`finals` are already
PERCENT. The curves are scaled here; the scalars are not.

PROVENANCE
    310_forgetting_by_scenario_{scenario}.npz     backprop, seeds 10-19, config_300.yaml, 90%
    332_pc_forgetting_by_scenario_{scenario}.npz  pc, same seeds and protocol, dt 0.4
Paired seed-for-seed: at a given seed both rules see the same class split and the same init.
⚠ The backprop arm ran at lr 0.01 and the PC arm at 0.02, each rule's established default. 314
re-ran backprop at 0.02 and found no difference (all p > 0.34), which is why the pairing stands.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
# 007's card order: column 1 is Class-IL (bp then pc), column 2 is Domain-IL (bp then pc).
PANELS = [
    ("a", "310_forgetting_by_scenario_class_il.npz",     "backprop", "Class-IL"),
    ("b", "332_pc_forgetting_by_scenario_class_il.npz",  "PC",       "Class-IL"),
    ("c", "310_forgetting_by_scenario_domain_il.npz",    "backprop", "Domain-IL"),
    ("d", "332_pc_forgetting_by_scenario_domain_il.npz", "PC",       "Domain-IL"),
]
TASK1_COLOR = "tab:orange"      # as 310 already uses
TASK2_COLOR = "tab:blue"
CROSSOVER_COLOR = "tab:green"
FINAL_COLOR = "tab:red"
THRESHOLD = 90.0


def draw(tag, fname, rule, scenario):
    d = np.load(EXP / fname, allow_pickle=True)
    n = len(d["steps"])

    fig = plt.figure(figsize=(5.0, 5.0))
    gs = fig.add_gridspec(2, 2, width_ratios=(6, 1), height_ratios=(1, 6),
                          wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)     # final task-1 distribution
    axr = fig.add_subplot(gs[1, 1], sharey=ax)     # crossing-height distribution

    for i in range(n):
        s = np.asarray(d["steps"][i])
        x = np.asarray(d["t1"][i], float) * 100.0   # fractions -> percent
        y = np.asarray(d["t2"][i], float) * 100.0
        k = int(np.searchsorted(s, int(d["switch0"][i]), side="right"))
        ax.plot(x[:k + 1], y[:k + 1], color=TASK1_COLOR, lw=0.8, alpha=0.6)
        ax.plot(x[k:], y[k:], color=TASK2_COLOR, lw=0.8, alpha=0.6)

    cx = np.asarray(d["crossovers"], float)
    fin = np.asarray(d["finals"], float)
    ok = np.isfinite(cx)
    ax.plot([0, 100], [0, 100], ls="--", lw=1.0, color=CROSSOVER_COLOR, alpha=0.8)
    ax.scatter(cx[ok], cx[ok], s=26, marker="s", color=CROSSOVER_COLOR, zorder=5)
    ax.axhline(THRESHOLD, ls="--", lw=1.0, color=FINAL_COLOR, alpha=0.7)
    ends_y = [np.asarray(d["t2"][i], float)[-1] * 100.0 for i in range(n)]
    ax.scatter(fin, ends_y, s=26, marker="o", facecolor="none",
               edgecolor=FINAL_COLOR, lw=1.4, zorder=5)

    # marginals, as 310 draws them
    axt.hist(fin, bins=np.arange(0, 102, 5), color=FINAL_COLOR, alpha=0.6)
    axt.axvline(np.nanmean(fin), color=FINAL_COLOR, lw=1.6)
    axr.hist(cx[ok], bins=np.arange(0, 102, 5), orientation="horizontal",
             color=CROSSOVER_COLOR, alpha=0.6)
    axr.axhline(np.nanmean(cx), color=CROSSOVER_COLOR, lw=1.6)
    for a in (axt, axr):
        a.axis("off")

    # Hard-coded and identical in every panel -- the four files are one 2x2.
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("task-1 accuracy (%)", fontsize=9)
    ax.set_ylabel("task-2 accuracy (%)", fontsize=9)
    ax.grid(alpha=0.2)
    axt.set_title(f"{scenario} · {rule} — crossover {np.nanmean(cx):.1f}, "
                  f"final task-1 {np.nanmean(fin):.1f}", fontsize=9)
    if not ok.all():
        ax.annotate(f"{int((~ok).sum())}/{ok.size} never crossed", (0.97, 0.03),
                    xycoords="axes fraction", ha="right", fontsize=7.5, color="crimson")

    out = figure_path(__file__, tag)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}   crossover {np.nanmean(cx):5.2f}  final t1 {np.nanmean(fin):5.2f}")


if __name__ == "__main__":
    for args in PANELS:
        draw(*args)
