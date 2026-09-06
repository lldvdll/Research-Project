"""Why is forgetting read at matched competence, and on crossover rather than an endpoint?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Methods M3.

ONE PLOT, one grid cell, as script_plan_800_900.md specifies for 907 ("merge 111 + 343"). Three
panels, because the stopping rule is DERIVED from two different single-rule failures plus the
mechanism behind the first:

    (a)  the ENDPOINT fails as the stopping point moves    (111)
    (b)  why it fails -- task 1's own peak moves with it    (113's mechanism, from 111's curves)
    (c)  CROSSOVER fails as the learning rate rises         (343 / 340)

Two candidate metrics, two different failure regimes, and a competence-matched rule read on
crossover is what survives both.

⚠ EVERY PANEL USES BACKPROP ALONE, AND THAT IS THE POINT. None of this references the
PC-vs-backprop comparison. 112 (drawn as 908) asks which metric preserves the sign of
PC - backprop, which would select the instrument by the answer it gives on the very comparison it
is about to be used for. 908 is therefore cited from Results as a robustness check and must not
appear in Methods. An earlier draft had these the wrong way round; CLAUDE.md already forbids the
same error for learning rates.

WHAT (b) ADDS. The endpoint does not drift for a mysterious reason. Raising the threshold makes
task 1 train longer -- 131 updates at 75% against 2403 at 95% -- so its peak rises from 78.6% to
93.8%, and "final task-1 accuracy" is a fall from a ceiling that is itself moving. An endpoint
number is a statement about where you stopped.

WHAT (c) MEANS. A censored crossover is not missing data: it means task 1 never fell below task 2,
the BEST outcome the metric can encounter. Dropping those runs removes each rule's best seeds, and
more of them from whichever rule is winning, so they are ranked (paired_sign with
censored_is_best) and never discarded.

Styling follows the 300-series scripts these regenerate: tab: colours, dpi 120,
bbox_inches="tight", 9pt labels, 8pt legends. No shared style module.

PROVENANCE
    111_metric_sensitivity_to_threshold.npz   PRE-800. BACKPROP ONLY, 10 seeds, threshold swept
                                              0.75-0.95. Backprop-only is the design, not a limit.
                                              Panel (b)'s peak is recomputed from the saved curves.
    340_lr_sweep_{scenario}.npz               Current protocol, 10 seeds, shared lr grid. Supplies
                                              panel (c)'s censoring counts (the 343 analysis).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
CX = 5          # crossover column in the 340 row layout


if __name__ == "__main__":
    z = np.load(EXP / "111_metric_sensitivity_to_threshold.npz", allow_pickle=True)
    thr = z["thresholds"] * 100.0
    tags = [f"{t:g}" for t in z["thresholds"]]

    # TWO FILES, because 007 gives this merge TWO grid cells and a cell takes one plot:
    #   _a  the endpoint failure and its mechanism   (card "does an endpoint metric survive")
    #   _b  crossover's own failure regime          (card "does crossover survive a change of lr")
    figA, axesA = plt.subplots(1, 2, figsize=(9.0, 4.0))
    figB, axB = plt.subplots(figsize=(5.4, 4.0))
    axes = [axesA[0], axesA[1], axB]

    # (a) the endpoint metrics, each on its own axis -- they are on different scales
    ax = axes[0]
    for s in SCENARIOS:
        v = np.nanmean(z[f"final_t1_{s}"], axis=1)
        e = np.nanstd(z[f"final_t1_{s}"], axis=1, ddof=1) / np.sqrt(z[f"final_t1_{s}"].shape[1])
        ax.errorbar(thr, v, yerr=e, marker="o", capsize=3, lw=2, color=COLORS[s],
                    label=f"final task-1 · {NICE[s]}")
    ax2 = ax.twinx()
    for s in SCENARIOS:
        v = np.nanmean(z[f"sb_mean_err_{s}"], axis=1)
        ax2.plot(thr, v, ls="--", marker="^", ms=4, lw=1.6, color=COLORS[s], alpha=0.55,
                 label=f"S&B mean error · {NICE[s]}")
    ax2.set_ylabel("S&B mean test error (%)", fontsize=9)
    ax.set_xlabel("stopping threshold (%)", fontsize=9)
    ax.set_ylabel("final task-1 accuracy (%)", fontsize=9)
    ax.set_title("the endpoint moves with the stopping rule", fontsize=9)
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper right")

    # (b) the mechanism: task 1's peak, and its phase length, both rise with the threshold
    ax = axes[1]
    for s in SCENARIOS:
        peak, plen = [], []
        for tg in tags:
            cv, t1 = z[f"curve_{s}_{tg}"], z[f"t1len_{s}_{tg}"]
            peak.append(100.0 * np.mean([np.nanmax(cv[i, :int(t1[i]), 0])
                                         for i in range(cv.shape[0])]))
            plen.append(float(np.mean(t1)))
        ax.plot(thr, peak, marker="o", lw=2, color=COLORS[s], label=f"task-1 peak · {NICE[s]}")
        print(f"  {NICE[s]:10s} peak " + " ".join(f"{p:.1f}" for p in peak)
              + "   phase " + " ".join(f"{p:.0f}" for p in plen))
    ax.set_xlabel("stopping threshold (%)", fontsize=9)
    ax.set_ylabel("task-1 peak accuracy (%)", fontsize=9)
    ax.set_title("(b) because task 1's own ceiling moves too\n"
                 "(its phase grows 131 → 2403 updates)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")

    # (c) crossover's own failure regime: it stops existing
    ax = axes[2]
    for s in SCENARIOS:
        rows = np.load(EXP / f"340_lr_sweep_{s}.npz", allow_pickle=True)["data"]
        grid = sorted({r[1] for r in rows})
        for m, ls, mk in (("backprop", "-", "s"), ("pc", "--", "o")):
            frac = [100.0 * float((~np.isfinite(np.array(
                [float(r[CX]) for r in rows if r[0] == m and r[1] == g]))).mean()) for g in grid]
            ax.plot(grid, frac, ls=ls, marker=mk, ms=4, lw=1.8, color=COLORS[s],
                    label=f"{m} · {NICE[s]}")
            if m == "backprop":
                print(f"  {NICE[s]:10s} censored % by lr: "
                      + " ".join(f"{g}:{f:.0f}" for g, f in zip(grid, frac)))
    ax.set_xscale("log")
    ax.set_xticks(grid)
    ax.set_xticklabels([str(g) for g in grid], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("learning rate", fontsize=9)
    ax.set_ylabel("seeds with NO crossover (%)", fontsize=9)
    ax.set_ylim(-4, 104)
    ax.set_title("(c) crossover fails the other way — it stops\n"
                 "existing: 10/10 undefined at lr 0.16", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, loc="upper left")

    for f, sfx in ((figA, "a"), (figB, "b")):
        f.tight_layout()
        o = figure_path(__file__, sfx)
        f.savefig(o, dpi=120, bbox_inches="tight")
        print(f"saved {o}")
    for name in ("final_t1", "sb_mean_err"):
        for s in SCENARIOS:
            v = np.nanmean(z[f"{name}_{s}"], axis=1)
            print(f"  {name:12s} {NICE[s]:10s} " + " -> ".join(f"{x:.2f}" for x in v))
