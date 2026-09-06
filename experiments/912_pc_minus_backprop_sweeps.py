"""Does PC's advantage over backprop survive the learning rate, the width and the depth -- and
does it keep the same sign in both scenarios?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing.

MODE B -- one file, six panels, ONE LaTeX caption. The three sweeps have DIFFERENT x-axes and
cannot share one, but they must share the Y axis, because the claim is about the SIZE of an effect
measured the same way in all three. Three separately-scaled files would let a +1.4 point difference
and a +3.4 point difference occupy the same visual height, which is the specific way this figure
could mislead.

WHAT IS PLOTTED IS THE PAIRED DIFFERENCE, NOT THE ABSOLUTE VALUE. Crossover's raw value is not
stopping-point independent (110: Domain-IL 36.2 -> 75.3 across a budget sweep), so an absolute
crossover plotted against a swept axis mixes the sweep's effect with the protocol's. The paired
PC - backprop difference is the stable quantity: at a given seed both rules see the same class
split and the same initialisation, so the shared variance cancels. 341's Class-IL retention ranges
40 points across seeds; the paired difference has a SEM of 0.39 at the same point.

ROW 2 IS NOT OPTIONAL. Crossover is UNDEFINED when task 1 never falls below task 2, and that is
censoring, not missing data -- the run did better than the metric can express. Averaging over it
silently drops each rule's best seeds, and it drops more of them from whichever rule is winning.
At lr 0.16 the Class-IL crossover is undefined on 10/10 backprop seeds and 6/10 PC seeds, so the
row-1 point there is not a null result, it is an absent measurement. The censoring row says which
of the two the reader is looking at, per point.

THE READING THE FIGURE SUPPORTS, stated so the sign convention cannot be reinterpreted later:
above the zero line PC retains more than backprop; below it, less. The effect is small everywhere
(single points, against a 65-76 point crossover), it is POSITIVE in Class-IL and NEGATIVE in
Domain-IL at the working point, and depth widens the gap in both directions at once. Whether that
is worth calling an advantage is what 913 puts on a scale.

PROVENANCE -- all three sweeps, config_300.yaml, seeds 10-19, 90% matched competence:
    340  lr     {0.005 ... 0.16}, shared absolute grid for every rule
    341  width  H in {4, 8, 16, 32, 64}, each rule at its established lr
    342  depth  1-4 hidden layers, each rule at its established lr
341 and 342 were re-run at the corrected dt = 0.2 after 346/347 found dt = 0.4 oscillates at
depth >= 2 and at Class-IL H = 4. 340 stays at dt = 0.4, which is correct at its fixed H = 32,
depth = 1. Do not "unify" the dt across the three -- the correct value differs by configuration
and that is itself a result (see report_track.md tier 4).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff
from src import style

EXP = ROOT / "experiments"
# (stem, axis label, x-scale). Column order is the order the report argues them in: the lr
# objection first, because a reader assumes tuning before anything else.
SWEEPS = [
    ("340_lr_sweep",    "learning rate",   "log"),
    ("341_width_sweep", "hidden width $H$", "log"),
    ("342_depth_sweep", "hidden layers",   "linear"),
]
SCENARIOS = ["class_il", "domain_il"]
LABEL = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
# Row layout of the saved sweeps: (method, swept_value, seed, retention, t2_final, cx_height, ...)
COL_CX = 5
# The configuration every other figure in the report is measured at.
WORKING = {"341_width_sweep": 32, "342_depth_sweep": 1}

style.apply()


def load(stem, scenario):
    return np.load(EXP / f"{stem}_{scenario}.npz", allow_pickle=True)["data"]


def by_seed(rows, method, x):
    """Crossover per seed at one point of the sweep, ordered by seed so pairs line up."""
    d = {int(r[2]): float(r[COL_CX]) for r in rows if r[0] == method and r[1] == x}
    return np.array([d[s] for s in sorted(d)], dtype=float)


if __name__ == "__main__":
    w = style.WIDTH["page"] / 3.0          # three panels across a figure* -- 2.35 in each
    fig, axes = plt.subplots(2, 3, figsize=(3 * w, 2 * w * 0.86),
                             sharey="row", height_ratios=[2.0, 1.0])

    for c, (stem, xlabel, xscale) in enumerate(SWEEPS):
        top, bot = axes[0][c], axes[1][c]
        top.axhline(0, color=style.ZERO_LINE, lw=0.7, zorder=1)
        for scenario in SCENARIOS:
            rows = load(stem, scenario)
            grid = sorted({r[1] for r in rows})
            means, sems, cens = [], [], {"backprop": [], "pc": []}
            for x in grid:
                bp, pc = by_seed(rows, "backprop", x), by_seed(rows, "pc", x)
                m, se, _ = paired_diff(pc, bp)
                means.append(m)
                sems.append(se)
                for meth, v in (("backprop", bp), ("pc", pc)):
                    cens[meth].append(100.0 * float((~np.isfinite(v)).mean()))
            g = np.asarray(grid, dtype=float)
            means, sems = np.asarray(means), np.asarray(sems)
            top.errorbar(g, means, yerr=sems, marker="o", lw=1.3,
                         color=style.SCENARIO[scenario], label=LABEL[scenario])
            # A point with no paired difference at all: EVERY seed of at least one rule was
            # censored there, so the line simply stops. Marked, because a line that stops reads
            # as "not run" and this was run -- it is the metric that ran out, not the sweep.
            gone = ~np.isfinite(means)
            if gone.any():
                top.plot(g[gone], np.zeros(gone.sum()), ls="none", marker="x", ms=4.5,
                         mew=1.1, color=style.CAPPED, zorder=6, clip_on=False)
            for meth, ls in (("backprop", "-"), ("pc", "--")):
                bot.plot(g, cens[meth], ls=ls, lw=1.0, marker="s", ms=2.2,
                         color=style.SCENARIO[scenario])

        if stem in WORKING:
            # The point every other figure is measured at. Drawn so a reader can see that the
            # working point is not the peak of the effect -- it is not chosen to flatter PC.
            for ax in (top, bot):
                ax.axvline(WORKING[stem], color=style.NEUTRAL, lw=0.7, ls=":", zorder=0)
        else:
            # lr has no single working point: backprop's default is 0.01 and PC's is 0.02, so the
            # band is shown rather than a line. 314 shows moving backprop to 0.02 changes nothing.
            for ax in (top, bot):
                ax.axvspan(0.01, 0.02, color=style.NEUTRAL, alpha=0.12, zorder=0)

        for ax in (top, bot):
            ax.set_xscale(xscale)
            if xscale == "log":
                ax.set_xticks(sorted({float(r[1]) for r in load(stem, SCENARIOS[0])}))
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
                ax.minorticks_off()
        top.tick_params(labelbottom=False)
        bot.set_xlabel(xlabel)
        bot.set_ylim(-4, 104)
        bot.set_yticks([0, 50, 100])
        style.panel_label(top, "abc"[c], dx=-0.24)

    axes[0][0].set_ylabel("$\\Delta$ crossover, PC $-$ backprop (pp)")
    axes[1][0].set_ylabel("seeds\ncensored (%)")
    axes[0][0].legend(
        [plt.Line2D([], [], color=style.SCENARIO["class_il"], marker="o", ms=3.0, lw=1.3),
         plt.Line2D([], [], color=style.SCENARIO["domain_il"], marker="o", ms=3.0, lw=1.3),
         plt.Line2D([], [], color=style.CAPPED, marker="x", ms=4.0, mew=1.1, ls="none")],
        [LABEL["class_il"], LABEL["domain_il"], "no paired measurement"],
        loc="upper left", fontsize=6.5, handlelength=1.6, borderpad=0.15, labelspacing=0.25)
    axes[1][0].legend([plt.Line2D([], [], color=style.NEUTRAL, ls="-", lw=1.0),
                       plt.Line2D([], [], color=style.NEUTRAL, ls="--", lw=1.0)],
                      ["backprop", "PC"], loc="upper left", fontsize=6.5, handlelength=1.6,
                      borderpad=0.15, labelspacing=0.25)

    out = figure_path(__file__)
    fig.savefig(out)
    print(f"saved {out}")
    for stem, _, _ in SWEEPS:
        for scenario in SCENARIOS:
            rows = load(stem, scenario)
            parts = []
            for x in sorted({r[1] for r in rows}):
                bp, pc = by_seed(rows, "backprop", x), by_seed(rows, "pc", x)
                m, se, _ = paired_diff(pc, bp)
                nb = int((~np.isfinite(bp)).sum()) + int((~np.isfinite(pc)).sum())
                parts.append(f"{x}: {m:+5.2f}+-{se:4.2f}" + (f" [{nb} cens]" if nb else ""))
            print(f"  {stem:16s} {LABEL[scenario]:10s} " + "  ".join(parts))
