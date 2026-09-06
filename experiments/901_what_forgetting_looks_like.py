"""What does catastrophic forgetting look like on a single run?

900-series PLOT SCRIPT. Loads a saved array, trains nothing. The report's opening figure (M1).

FORM IS SET BY 007's SK_DEF SKETCH, one grid cell = one plot:
    one axes, ten thin per-class lines behind two bold per-task means, the switch as a vertical
    line, the crossover marked, and two labelled arrows naming FORGETTING and LEARNING.

STYLING FOLLOWS THE 300 SERIES (310, 340) -- stock matplotlib defaults, tab: colours,
tight_layout, dpi 130, bbox_inches="tight". No shared style module.

TEN CLASS LINES, NOT TWO TASK LINES. The five task-1 classes falling to zero while the five
task-2 classes rise IS the output-competition mechanism, so the opening figure shows the thing R5
later explains rather than introducing it cold. The task means are drawn on top, bold, because the
crossover is defined on them.

ONE SEED, FIXED BUDGET. A ten-seed mean smooths away the shape the figure exists to show, and
under matched competence the seeds stop at different steps so the mean's tail is computed over a
shrinking sample. Nothing here is measured, so nothing here needs an error bar.

⚠ THIS RUN SATURATES AND THE CAPTION MUST SAY SO. Fixed budget of 900 updates per task, so task 1
ends at exactly 0.0% on all five classes where R1 reports 5.4% at matched competence. Both are
correct; the difference is the stopping rule. Left unstated a reader meets 0% here and 5.4% later
and concludes the report contradicts itself.

⚠ 0% IS NOT CHANCE. Chance on ten classes is 10%. Zero on every task-1 class means argmax has been
captured by the task-2 output units -- output suppression, not representation loss.

PROVENANCE
    801  Class-IL, seed 10, backprop, FIXED budget 900 updates per task, per-class logging,
         config_800.yaml for everything else (H = 32, tanh, SGD, batch 32, lr 0.02).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import crossover

SOURCE = ROOT / "experiments" / "801_definitional_run.npz"

# Plot-only choices, matching the 300 series' palette (310 uses these same four).
TASK1_COLOR = "tab:orange"
TASK2_COLOR = "tab:blue"
CROSSOVER_COLOR = "tab:green"
CHANCE = 10.0          # ten classes. Drawn because 0% must not be read as chance.


if __name__ == "__main__":
    d = np.load(SOURCE, allow_pickle=True)
    steps, acc = np.asarray(d["steps"]), np.asarray(d["acc"], dtype=float)
    classes, switch = list(d["classes"]), int(d["switch"])
    i1 = [classes.index(c) for c in list(d["task1"])]
    i2 = [classes.index(c) for c in list(d["task2"])]
    t1, t2 = acc[:, i1].mean(axis=1), acc[:, i2].mean(axis=1)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    for col in i1:
        ax.plot(steps, acc[:, col], color=TASK1_COLOR, lw=0.7, alpha=0.35)
    for col in i2:
        ax.plot(steps, acc[:, col], color=TASK2_COLOR, lw=0.7, alpha=0.35)
    ax.plot(steps, t1, color=TASK1_COLOR, lw=2.2, label="task 1 (mean of its 5 classes)")
    ax.plot(steps, t2, color=TASK2_COLOR, lw=2.2, label="task 2 (mean of its 5 classes)")

    ax.axvline(switch, color="0.25", lw=1.2)
    ax.annotate("switch", (switch, 101), xytext=(-4, 0), textcoords="offset points",
                fontsize=8, ha="right", va="bottom", color="0.25")
    ax.axhline(CHANCE, color="0.6", lw=0.8, ls="--")
    ax.annotate("chance (10 classes)", (steps[len(steps) // 6], CHANCE), xytext=(0, 3),
                textcoords="offset points", fontsize=7, color="0.45")

    cx_step, cx_height = crossover(steps, t1, t2, after=switch)
    ax.plot([cx_step], [cx_height], marker="o", ms=7, mfc="none",
            mec=CROSSOVER_COLOR, mew=1.8, zorder=5)
    ax.annotate(f"crossover  {cx_height:.0f}%", (cx_step, cx_height), xytext=(52, -6),
                textcoords="offset points", fontsize=8, color=CROSSOVER_COLOR,
                va="center", arrowprops=dict(arrowstyle="-", lw=0.8, color=CROSSOVER_COLOR))

    # The two names SK_DEF asks the figure to plant. Both labels sit in the empty band between
    # the collapsed task-1 curve and the risen task-2 curve, with a short arrow to the curve each
    # one describes -- drawn across the data they read as extra series.
    ax.annotate("forgetting", xy=(switch + 350, 3), xytext=(switch + 500, 30),
                fontsize=9, color=TASK1_COLOR,
                arrowprops=dict(arrowstyle="->", lw=1.3, color=TASK1_COLOR))
    ax.annotate("learning", xy=(switch + 350, 91), xytext=(switch + 500, 62),
                fontsize=9, color=TASK2_COLOR,
                arrowprops=dict(arrowstyle="->", lw=1.3, color=TASK2_COLOR))

    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(-2, 104)
    ax.set_xlabel("training updates")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Class-IL, backprop, one seed, fixed budget -- all ten classes drawn separately",
                 fontsize=9)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="center left")

    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")
    print(f"  switch {switch}   crossover {cx_height:.1f}% at step {cx_step:.0f} "
          f"({cx_step - switch:.0f} into task 2)")
    print(f"  end of task 1: t1 {t1[len(t1) // 2]:.1f}   end of run: t1 {t1[-1]:.1f}  "
          f"t2 {t2[-1]:.1f}")
    print(f"  per-class task-1 at end: {dict(zip(list(d['task1']), acc[-1, i1].round(1)))}")
