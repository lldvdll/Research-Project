"""What does catastrophic forgetting look like on a single run?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. The report's opening figure.

MODE A -- one file, one result. Definitional, not comparative: it names the three things every
later section refers to, so that none of them has to be re-explained in a Results caption.

    the switch      the vertical line where training moves from task 1 to task 2
    the plateau     where each task's curve settles once it stops changing
    the crossover   the step at which the falling task-1 curve meets the rising task-2 curve,
                    and the accuracy at which they meet -- the number every comparison in this
                    report is made on

TEN CLASS LINES, NOT TWO TASK LINES. The five task-1 classes fall to exactly zero while the five
task-2 classes rise to ~93%, and that is the observation R5's mechanism section is about. Drawing
only the two task means would average the effect into a single curve and the reader would meet the
per-class picture cold, twenty pages later. The task means are drawn ON TOP, bold, because the
crossover is defined on them.

ONE SEED, AND NO ERROR BAR ANYWHERE. A ten-seed mean would smooth away the shape the figure exists
to show, and under matched-competence stopping the seeds end at different steps, so the mean's
tail would be computed over a shrinking sample. Nothing here is measured, so nothing here needs a
confidence interval; every quantitative claim in the report comes from a ten-seed figure.

⚠ THIS RUN SATURATES, AND THE CAPTION MUST SAY SO. It uses a FIXED budget of 900 updates per task
rather than the matched-competence stopping every measured result uses, so task 1 ends at exactly
0.0% on all five of its classes -- where R1 reports 5.4% at matched competence. Both are correct:
a saturating budget drives the unmasked Class-IL condition to the floor, which is precisely why
the measured results stop at a competence threshold instead (42/43 established this). The
definitional figure wants the complete track and can afford saturation because it quantifies
nothing. Left unstated, a reader meets 0% here and 5.4% later and concludes the report contradicts
itself. The annotation on the figure carries this; the caption must repeat it.

⚠ 0% IS NOT CHANCE. Chance on ten classes is 10%, and a model that predicted one class for
everything would score 10% averaged over the ten. Zero on every task-1 class means argmax is being
captured by the task-2 output units -- output-layer suppression, not representation loss. That
distinction is the whole of R5 and the figure marks it rather than leaving it to be inferred.

PROVENANCE
    801  Class-IL, seed 10, backprop, FIXED budget 900 updates per task, per-class logging,
         config_800.yaml for every other parameter (H = 32, tanh, SGD, batch 32, lr 0.02).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import crossover
from src import style

SOURCE = ROOT / "experiments" / "801_definitional_run.npz"
CHANCE = 10.0        # ten classes. Drawn because 0% must not be read as "chance".

style.apply()


if __name__ == "__main__":
    d = np.load(SOURCE, allow_pickle=True)
    steps, acc = np.asarray(d["steps"]), np.asarray(d["acc"], dtype=float)
    classes, switch = list(d["classes"]), int(d["switch"])
    i1 = [classes.index(c) for c in list(d["task1"])]
    i2 = [classes.index(c) for c in list(d["task2"])]
    t1, t2 = acc[:, i1].mean(axis=1), acc[:, i2].mean(axis=1)

    fig, ax = plt.subplots(figsize=style.size("col", 0.70))

    # Which task is training, as a background band rather than a legend entry -- it is a property
    # of the x-axis, not a series.
    ax.axvspan(steps[0], switch, color=style.TASK[0], alpha=0.05, lw=0)
    ax.axvspan(switch, steps[-1], color=style.TASK[1], alpha=0.05, lw=0)
    ax.axvline(switch, color=style.ZERO_LINE, lw=0.8)
    ax.axhline(CHANCE, color=style.NEUTRAL, lw=0.6, ls=(0, (4, 3)))

    for col in i1:
        ax.plot(steps, acc[:, col], color=style.TASK[0], lw=0.5, alpha=0.4)
    for col in i2:
        ax.plot(steps, acc[:, col], color=style.TASK[1], lw=0.5, alpha=0.4)
    ax.plot(steps, t1, color=style.TASK[0], lw=1.6, label="task 1 (mean of 5 classes)")
    ax.plot(steps, t2, color=style.TASK[1], lw=1.6, label="task 2 (mean of 5 classes)")

    cx_step, cx_height = crossover(steps, t1, t2, after=switch)
    if cx_step is not None:
        ax.plot([cx_step], [cx_height], marker="o", ms=4.0, mfc="none",
                mec=style.ZERO_LINE, mew=1.0, zorder=5)
        ax.annotate(f"crossover\n{cx_height:.0f}%", (cx_step, cx_height),
                    xytext=(16, 24), textcoords="offset points", fontsize=6.5,
                    ha="left", va="bottom", linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=style.ZERO_LINE))

    ax.annotate("task switch", (switch, 101), xytext=(-3, 0), textcoords="offset points",
                fontsize=6.5, ha="right", va="bottom")
    # "chance" goes on the LEFT and the saturation note on the RIGHT: both want the band just
    # above y = 0, and at the right-hand end they overlap.
    ax.annotate("chance", (steps[len(steps) // 6], CHANCE), xytext=(0, 2),
                textcoords="offset points", fontsize=6.0, ha="left", va="bottom",
                color=style.NEUTRAL)
    # The saturation warning, on the figure and not only in the caption.
    ax.annotate("all five task-1\nclasses at 0%", (steps[-1], CHANCE), xytext=(-4, 4),
                textcoords="offset points", fontsize=6.5, ha="right", va="bottom",
                color=style.CAPPED, linespacing=1.3)

    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(-2, 102)
    ax.set_xlabel("training updates")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(loc="center left", fontsize=6.5, handlelength=1.4, borderpad=0.2,
              labelspacing=0.3)

    out = figure_path(__file__)
    fig.savefig(out)
    print(f"saved {out}")
    print(f"  switch {switch}   crossover {cx_height:.1f}% at step {cx_step:.0f} "
          f"({cx_step - switch:.0f} into task 2)")
    print(f"  end of task 1: t1 {t1[len(t1) // 2]:.1f}   end of run: t1 {t1[-1]:.1f}  t2 {t2[-1]:.1f}")
    print(f"  per-class task-1 at end: {dict(zip(list(d['task1']), acc[-1, i1].round(1)))}")
