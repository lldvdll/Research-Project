"""Do the two scenarios forget in different SHAPES, and does the learning rule change that shape?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing.

MODE B -- one file, four panels, ONE LaTeX caption, panels lettered by style.panel_label. This
figure earns shared axes because its claim is a statement ABOUT the axes: columns differ and rows
do not. Four separately-scaled files could not make that claim, since the reader would have to
trust that four y-limits were matched by hand.

THE FIGURE IS A PHASE PLOT: x = task-1 accuracy, y = task-2 accuracy, time removed. Accuracy
against step answers "how much was lost"; this answers "along what path", which is the question a
shape claim needs. The trajectory runs right along the bottom while task 1 trains, then turns and
climbs while task 2 trains, and the CHARACTER of that turn is what separates the scenarios:

    Class-IL    the turn is square. Task-2 accuracy rises while task-1 accuracy falls to the left
                wall, so the path exits through the top-LEFT corner -- one task is traded for the
                other almost completely.
    Domain-IL   the turn is rounded and stops short. The path ends in the upper-middle, both
                tasks partly held.

WHAT IS COMPARED, AND WHAT IS NOT. Rows are the learning rule; columns are the scenario. If the
rule mattered as much as the scenario, rows would differ as much as columns do. They do not, and
that asymmetry is the whole point of the figure -- it is the visual form of the paired numbers
912 reports, and it is the evidence for splitting the mechanism investigation by scenario rather
than by rule.

NO MEAN LINE. Matched-competence stopping ends each seed at a different absolute step, so a
step-aligned mean is computed over a shrinking sample in its tail and reports a trajectory no
individual run took. The ten seeds are drawn individually and the two SCALARS that summarise them
(crossover height, final task-1) are printed in the panel with their SEM.

PROVENANCE
    310  backprop, seeds 10-19, config_300.yaml, 90% matched competence, lr 0.01
    332  pc,       seeds 10-19, config_300.yaml, 90% matched competence, lr 0.02, dt 0.4
Paired seed-for-seed: at a given seed both rules see the same class split and the same init.
CAVEAT, and it must stay in the caption: the backprop arm ran at lr 0.01 and the PC arm at 0.02,
the established per-rule defaults. 314 re-ran backprop at 0.02 and found no difference (all
p > 0.34), which is why the pairing is allowed to stand; config_800.yaml adopts 0.02 for both, so
repointing this script at 816 later is a two-line edit.

UNITS TRAP: in these arrays `t1`/`t2` are FRACTIONS (0-1) while `crossovers`/`finals` are already
PERCENT. The curves are scaled here; the scalars are not.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import sem
from src import style

# ---------------------------------------------------------------- sources
EXP = ROOT / "experiments"
SOURCE = {
    ("backprop", "class_il"):  EXP / "310_forgetting_by_scenario_class_il.npz",
    ("backprop", "domain_il"): EXP / "310_forgetting_by_scenario_domain_il.npz",
    ("pc", "class_il"):        EXP / "332_pc_forgetting_by_scenario_class_il.npz",
    ("pc", "domain_il"):       EXP / "332_pc_forgetting_by_scenario_domain_il.npz",
}
RULES = ["backprop", "pc"]              # rows
SCENARIOS = ["class_il", "domain_il"]   # columns
LABEL = {"backprop": "backprop", "pc": "predictive coding",
         "class_il": "Class-IL", "domain_il": "Domain-IL"}

style.apply()


def panel(ax, path, letter, title):
    d = np.load(path, allow_pickle=True)
    # Trajectories are fractions; the saved scalars are percent. Scale one, not the other.
    for i in range(len(d["steps"])):
        s = np.asarray(d["steps"][i])
        x = np.asarray(d["t1"][i], dtype=float) * 100.0
        y = np.asarray(d["t2"][i], dtype=float) * 100.0
        sw = int(d["switch0"][i])
        # Split on the switch. The +1 overlaps one point so the two coloured segments join
        # rather than leaving a gap at the corner, which is exactly where the shape is read.
        k = int(np.searchsorted(s, sw, side="right"))
        ax.plot(x[:k + 1], y[:k + 1], color=style.TASK[0], lw=0.7, alpha=0.55)
        ax.plot(x[k:], y[k:], color=style.TASK[1], lw=0.7, alpha=0.55)

    lo, hi = 0.0, 100.0
    ax.plot([lo, hi], [lo, hi], ls=":", lw=0.8, color=style.NEUTRAL, zorder=1)

    cx, fin = np.asarray(d["crossovers"], float), np.asarray(d["finals"], float)
    # Each seed's crossing sits ON the diagonal at its crossover height, by definition of the
    # metric -- it is drawn there rather than searched for in the curve, so the marker and the
    # number 912 reports cannot disagree.
    ok = np.isfinite(cx)
    ax.scatter(cx[ok], cx[ok], s=9, facecolor="none", edgecolor=style.ZERO_LINE, lw=0.7, zorder=4)
    ends = np.array([[np.asarray(d["t1"][i], float)[-1] * 100.0,
                      np.asarray(d["t2"][i], float)[-1] * 100.0]
                     for i in range(len(d["steps"]))])
    ax.scatter(ends[:, 0], ends[:, 1], s=11, marker="s", color=style.TASK[1], zorder=5)

    mc, sc_ = sem(cx)
    mf, sf = sem(fin)
    ax.text(0.97, 0.06,
            f"crossover {mc:.1f}$\\pm${sc_:.1f}\nfinal t1 {mf:.1f}$\\pm${sf:.1f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, linespacing=1.35)
    if not ok.all():
        ax.text(0.97, 0.94, f"censored {int((~ok).sum())}/{ok.size}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color=style.CAPPED)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title)
    style.panel_label(ax, letter, dx=-0.13)


if __name__ == "__main__":
    w = style.WIDTH["page"] / 2.0          # two panels across a figure* -- 3.53 in each
    # `aspect="equal"` fixes the AXES aspect, not the panel's, so the figure height cannot be
    # w * ratio the way a normal panel's is: constrained layout shrinks the axes to whatever
    # square fits and pads the rest. 0.90 is the measured height at which the padding closes
    # without clipping the titles -- raise it and dead space returns.
    fig, axes = plt.subplots(2, 2, figsize=(2 * w, 2 * w * 0.90), sharex=True, sharey=True)
    for r, rule in enumerate(RULES):
        for c, scenario in enumerate(SCENARIOS):
            panel(axes[r][c], SOURCE[(rule, scenario)], "abcd"[r * 2 + c],
                  f"{LABEL[scenario]} · {LABEL[rule]}")
    for c in range(2):
        axes[1][c].set_xlabel("task-1 accuracy (%)")
    for r in range(2):
        axes[r][0].set_ylabel("task-2 accuracy (%)")

    handles = [plt.Line2D([], [], color=style.TASK[0], lw=1.3),
               plt.Line2D([], [], color=style.TASK[1], lw=1.3),
               plt.Line2D([], [], ls="none", marker="o", mfc="none",
                          mec=style.ZERO_LINE, ms=3.2),
               plt.Line2D([], [], ls="none", marker="s", color=style.TASK[1], ms=3.2)]
    axes[0][0].legend(handles, ["training task 1", "training task 2", "crossover", "run end"],
                      loc="upper right", fontsize=6.5, handlelength=1.4, borderpad=0.2)

    out = figure_path(__file__)
    fig.savefig(out)
    print(f"saved {out}")
    for rule in RULES:
        for scenario in SCENARIOS:
            d = np.load(SOURCE[(rule, scenario)], allow_pickle=True)
            mc, sc_ = sem(np.asarray(d["crossovers"], float))
            mf, sf = sem(np.asarray(d["finals"], float))
            print(f"  {LABEL[scenario]:10s} {rule:9s}  crossover {mc:5.2f}+-{sc_:.2f}"
                  f"   final t1 {mf:5.2f}+-{sf:.2f}")
