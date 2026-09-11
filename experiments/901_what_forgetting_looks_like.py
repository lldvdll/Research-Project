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

    # Sized for ONE COLUMN -- this figure now sits inline beside the text that describes it,
    # so it is built at \columnwidth and nothing in it may rely on a full-page span.
    fig, ax = plt.subplots(figsize=(3.4, 2.7))

    # Phase shading replaces the switch rule. The band a curve is being TRAINED in carries that
    # curve's own colour, so which task is live is readable without a line or a label, and the
    # switch is the edge between the two blocks rather than an annotation competing with the data.
    ax.axvspan(steps[0], switch, color=TASK1_COLOR, alpha=0.09, lw=0, zorder=0)
    ax.axvspan(switch, steps[-1], color=TASK2_COLOR, alpha=0.09, lw=0, zorder=0)

    # Per-class lines over a tinted background need more contrast than they did over white:
    # darker, slightly thicker, and drawn above the shading.
    for col in i1:
        ax.plot(steps, acc[:, col], color=TASK1_COLOR, lw=0.6, alpha=0.55, zorder=2)
    for col in i2:
        ax.plot(steps, acc[:, col], color=TASK2_COLOR, lw=0.6, alpha=0.55, zorder=2)
    ax.plot(steps, t1, color=TASK1_COLOR, lw=2.0, zorder=4, label="task 1")
    ax.plot(steps, t2, color=TASK2_COLOR, lw=2.0, zorder=4, label="task 2")

    cx_step, cx_height = crossover(steps, t1, t2, after=switch)
    ax.plot([cx_step], [cx_height], marker="o", ms=5, mfc="none",
            mec=CROSSOVER_COLOR, mew=1.5, zorder=6)
    # Crossover label sits LEFT of the switch, in the task-1 block, where the curves have
    # already separated and there is empty space.
    ax.annotate(f"crossover {cx_height:.0f}%", (cx_step, cx_height), xytext=(-96, -2),
                textcoords="offset points", fontsize=7, color=CROSSOVER_COLOR,
                va="center", ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=CROSSOVER_COLOR))

    # Retention is the other metric this figure defines: what task 1 still scores at the end.
    # It is annotated where it lives -- bottom right, just above the collapsed task-1 curve.
    # Anchored a little inside the right edge: at the very last step the label runs past the
    # axes and the leading character is clipped.
    ax.annotate(f"retention {t1[-1]:.0f}%", (steps[-1], t1[-1]), xytext=(-6, 18),
                textcoords="offset points", fontsize=7, color=CROSSOVER_COLOR,
                ha="right", va="bottom", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=CROSSOVER_COLOR))

    # The two named processes. Positions follow the slower curve the reduced learning rate
    # produces: the fall and the rise now take roughly a third of the second block.
    ax.annotate("forgetting", xy=(switch + 260, 26), xytext=(switch + 210, 52),
                fontsize=8, color=TASK1_COLOR, ha="left",
                arrowprops=dict(arrowstyle="->", lw=1.1, color=TASK1_COLOR))
    ax.annotate("learning", xy=(switch + 300, 74), xytext=(switch + 120, 88),
                fontsize=8, color=TASK2_COLOR, ha="left",
                arrowprops=dict(arrowstyle="->", lw=1.1, color=TASK2_COLOR))

    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(-2, 104)
    ax.set_xlabel("training updates", fontsize=8)
    ax.set_ylabel("test accuracy (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=7, loc="center left", framealpha=0.85)

    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    print(f"  switch {switch}   crossover {cx_height:.1f}% at step {cx_step:.0f} "
          f"({cx_step - switch:.0f} into task 2)")
    print(f"  retention (final task 1) {t1[-1]:.1f}%   final task 2 {t2[-1]:.1f}%")
