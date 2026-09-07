"""What do the three sweeps look like before they are consolidated?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R2, appendix evidence.

TWO PLOTS, ONE PER GRID CELL, as script_plan_800_900.md specifies for 914 ("copy 340/341/342"):
    914_sweeps_behind_912_a.png   absolute crossover against lr, one line per rule
    914_sweeps_behind_912_b.png   absolute crossover against width and against depth

APPENDIX, BECAUSE 912 CARRIES THE CLAIM. These are the absolute values 912 takes differences of.
They earn their place by answering the objection 912 cannot: that the comparison might have been
read at a learning rate favouring one rule.

    Both rules peak near lr 0.01-0.02 and decline past it, so the established defaults sit close
    to jointly optimal. The growing PC - backprop gap at high lr is BACKPROP DEGRADING FASTER,
    not PC improving -- visible here and invisible in a difference plot, which is the reason to
    keep these at all.

⚠ CENSORING IS DRAWN, NOT AVERAGED THROUGH. Above lr 0.04 the Class-IL crossover starts to be
undefined; a mean over the seeds that still produced a number is a mean over that rule's WORST
seeds, because a censored run is one where task 1 never fell below task 2. The count is annotated
on every point that has one.

Styling follows 340/341/342 directly, including their METHOD_COLOR: backprop 0.35 grey, replay
tab:green, pc tab:orange. dpi 120, bbox_inches="tight", 9pt labels, 8pt legends. No shared style
module.

PROVENANCE -- config_300.yaml, seeds 10-19, 90% matched competence:
    340_lr_sweep_{scenario}.npz     341_width_sweep_{scenario}.npz     342_depth_sweep_{scenario}.npz
341/342 ran at the corrected dt = 0.2; 340 at dt = 0.4, correct for its fixed H = 32 / depth 1.
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
METHODS = ["backprop", "replay", "pc"]
METHOD_COLOR = {"backprop": "0.35", "replay": "tab:green", "pc": "tab:orange"}   # 340's own
CX = 5


def cell(rows, method, x):
    return np.array([float(r[CX]) for r in rows if r[0] == method and r[1] == x], dtype=float)


def sweep_panel(ax, stem, s, xlabel, xscale):
    rows = np.load(EXP / f"{stem}_{s}.npz", allow_pickle=True)["data"]
    grid = sorted({r[1] for r in rows})
    for m in METHODS:
        mu, se, cens = [], [], []
        for g in grid:
            v = cell(rows, m, g)
            ok = np.isfinite(v)
            mu.append(np.nanmean(v) if ok.any() else np.nan)
            se.append(np.nanstd(v[ok], ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else np.nan)
            cens.append(int((~ok).sum()))
        ax.errorbar(grid, mu, yerr=se, marker="o", ms=4, capsize=3, lw=1.8,
                    color=METHOD_COLOR[m], label=m)
        for g, v, c in zip(grid, mu, cens):
            if c and np.isfinite(v):
                ax.annotate(f"{c}/10\ncens", (g, v), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=6, color="crimson", linespacing=1.1)
    ax.set_xscale(xscale)
    ax.set_xticks(grid)
    ax.set_xticklabels([str(g) for g in grid], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title(NICE[s], fontsize=9)
    return grid


if __name__ == "__main__":
    # (a) the lr sweep
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, s in zip(axes, SCENARIOS):
        sweep_panel(ax, "340_lr_sweep", s, "learning rate", "log")
        ax.axvspan(0.01, 0.02, color="0.5", alpha=0.12)
    axes[0].set_ylabel("crossover (%)", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Both rules peak near lr 0.01–0.02 and decline past it — the shaded band is "
                 "where the defaults sit", fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__, "a")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    # (b) width and depth, one row of four
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8), sharey=True)
    for i, (stem, xlabel, xscale) in enumerate([("341_width_sweep", "hidden width H", "log"),
                                                ("342_depth_sweep", "hidden layers", "linear")]):
        for j, s in enumerate(SCENARIOS):
            sweep_panel(axes[i * 2 + j], stem, s, xlabel, xscale)
    axes[0].set_ylabel("crossover (%)", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Width and depth, absolute crossover — the values 912 takes paired differences of",
                 fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__, "b")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    for stem in ("340_lr_sweep", "341_width_sweep", "342_depth_sweep"):
        for s in SCENARIOS:
            rows = np.load(EXP / f"{stem}_{s}.npz", allow_pickle=True)["data"]
            grid = sorted({r[1] for r in rows})
            best = {m: max(((np.nanmean(cell(rows, m, g)), g) for g in grid),
                           key=lambda t: (-np.inf if np.isnan(t[0]) else t[0]))[1]
                    for m in METHODS}
            print(f"  {stem:16s} {NICE[s]:10s} best point per rule: {best}")
