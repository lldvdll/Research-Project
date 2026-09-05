"""Slide asset: does PC beat backprop under crossover height, Domain-IL vs Class-IL?

Not a new experiment -- replots 52 and 56 (already run, already verified in findings.md C1),
restricted to backprop and pc, with each panel's crossover height marked so the two panels in a
row can be compared by eye.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import array_path, figure_path
from src.metrics import metric_grid, paired_diff

CELLS = [("52_do_the_rules_differ_fixed_budget.py", "Domain-IL"),
         ("56_do_the_rules_differ_class_il.py", "Class-IL")]
METHODS = ["backprop", "pc"]
COLOR = {"backprop": "tab:gray", "pc": "tab:red"}

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="row")
for row, (script, scen_label) in enumerate(CELLS):
    z = np.load(array_path(str(ROOT / "experiments" / script)), allow_pickle=True)
    sw = float(np.asarray(z["switches"]).ravel()[0])
    steps = z["steps"]
    xh, grid = {}, {}
    for m in METHODS:
        grid[m] = metric_grid(steps, z[f"argmax_{m}"], sw)
        xh[m] = np.nanmean(grid[m]["crossover"])
    d, se, n = paired_diff(grid["pc"]["crossover"], grid["backprop"]["crossover"])
    for col, m in enumerate(METHODS):
        ax = axes[row][col]
        a = z[f"argmax_{m}"] * 100  # (seeds, evals, 2)
        mean_t1, mean_t2 = a[:, :, 0].mean(0), a[:, :, 1].mean(0)
        for s in range(a.shape[0]):
            ax.plot(steps, a[s, :, 0], color="orange", alpha=0.15, lw=0.8)
            ax.plot(steps, a[s, :, 1], color="tab:blue", alpha=0.15, lw=0.8)
        ax.plot(steps, mean_t1, color="orange", lw=2.2, label="task 1")
        ax.plot(steps, mean_t2, color="tab:blue", lw=2.2, label="task 2")
        ax.axvspan(steps.min(), sw, color="orange", alpha=0.06)
        ax.axvspan(sw, steps.max(), color="tab:blue", alpha=0.06)
        for other in METHODS:
            ls = "-" if other == m else ":"
            lw = 1.6 if other == m else 1.1
            alpha = 0.9 if other == m else 0.6
            ax.axhline(xh[other], color=COLOR[other], ls=ls, lw=lw, alpha=alpha,
                       label=f"{other} crossover = {xh[other]:.1f}")
        ax.set_ylim(0, 100)
        ax.set_title(f"{scen_label} — {m}", fontsize=11)
        if col == 0:
            ax.set_ylabel("accuracy (%)")
        if row == 1:
            ax.set_xlabel("training step")
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(alpha=0.2)
        if col == 1:
            ax.annotate(f"pc − backprop = {d:+.2f}  ({n:.1f}σ)",
                        xy=(0.98, 0.03), xycoords="axes fraction", ha="right", fontsize=10,
                        fontweight="bold", color="darkred" if n >= 2 else "dimgray")

fig.suptitle("Does PC beat backprop? Crossover height, backprop vs PC only.\n"
             "Each panel's own crossover is the solid line; the OTHER rule's crossover from the "
             "same scenario is the dotted line, so the two panels in a row are read against the "
             "same ruler.",
             fontsize=11)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"saved {figure_path(__file__)}")
