"""Which settling step sizes reach the same fixed point, and where does that stop being true?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Methods M2.

ONE PLOT, BOTH SCENARIOS ON ONE AXES. The earlier version drew a trace-per-dt panel for each
scenario, which asked the reader to compare two pictures to reach one conclusion. The quantity
that carries the argument is not the route a relaxation takes, it is where it ENDS, so this
plots the settled displacement against dt directly and puts both scenarios on the same axes.

WHAT THE FIGURE HAS TO ESTABLISH, in one reading:
    1. a SHARED fixed point at low dt -- every step size below the boundary lands on exactly the
       same value, so the choice inside that band is free and costs nothing;
    2. that the band has an upper edge, and beyond it the relaxation does not merely land
       somewhere else, it stops settling at all;
    3. that dt = 0.4, the value the whole series runs at, is inside the band.

HOW OSCILLATION IS SHOWN WITHOUT A SECOND PANEL. Each point carries a bar spanning the
min-to-max of the LAST HUNDRED STEPS (400-500). A relaxation that has converged has nothing
left to vary, so the bar collapses to the marker and is invisible; one that is ringing shows a
bar. That makes "the settle point is higher" and "it is still moving" the same mark on the
page rather than two claims to reconcile. Measured: the bar is exactly zero for every dt up to
0.5 in both scenarios, and opens only at 0.85 and 1.5.

⚠ THE BAND IS NOT A ONE-TIME GATE. dt = 0.4 is correct at H = 32 / depth 1 and silently WRONG
at depth >= 2 and at Class-IL H = 4, where the tail sits 3-7x above the true fixed point. 905
draws that, and the width/depth sweeps use dt = 0.2 because of it. Re-verify whenever the
configuration moves.

COLOURS follow the project standard: this is a scenario-only figure with no rule split, so
Class-IL purple and Domain-IL green. src/style.py is deliberately NOT used (retired as the
source of truth on 2026-09-13); it would have made Domain-IL teal.

PROVENANCE
    334_pc_settle_trace_by_dt_{scenario}.npz   PRE-800, regenerated 2026-09-11 on a trimmed
    grid that ADDS dt = 0.4. The previous arrays ran 0.01-1.5 but stepped 0.2 -> 0.5 straight
    past 0.4, so the chosen value was never actually measured; the old ten-value arrays are
    kept beside them as *.npz.bak10dt. Single-seed traces run to a fixed 500-step cap with no
    early exit, which is what makes the tail comparison possible at all.
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
COLOR = {"class_il": "tab:purple", "domain_il": "tab:green"}

CHOSEN_DT = 0.4
DT_TICKS = [0.02, 0.1, 0.2, 0.4, 0.5, 0.85, 1.5]   # the swept grid, labelled as numbers
TAIL_FROM, TAIL_TO = 400, 500     # steps the min-max bar is taken over


def load_scenario(scenario):
    d = np.load(EXP / f"334_pc_settle_trace_by_dt_{scenario}.npz", allow_pickle=True)["data"]
    rows = []
    for r in sorted(d, key=lambda r: float(r[0])):
        tr = np.asarray(r[2], dtype=float)
        tail = tr[TAIL_FROM - 1:TAIL_TO]
        rows.append((float(r[0]), tr[-1], tail.min(), tail.max()))
    return np.array(rows)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6.6, 2.9))

    for scenario in SCENARIOS:
        a = load_scenario(scenario)
        dt, final, lo, hi = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        c = COLOR[scenario]
        ax.plot(dt, final, color=c, lw=1.2, marker="o", ms=4.5, label=NICE[scenario], zorder=3)
        # Bars collapse to nothing wherever the relaxation has converged -- that is the point.
        ax.vlines(dt, lo, hi, color=c, lw=5, alpha=0.45, zorder=2)
        for x, f, l, h in zip(dt, final, lo, hi):
            print(f"  {NICE[scenario]:10s} dt={x:<5} settled={f:.4f}  "
                  f"tail {TAIL_FROM}-{TAIL_TO} range={h - l:.4f}"
                  f"{'   <- oscillating' if h - l > 1e-6 else ''}")

    ax.axvline(CHOSEN_DT, color="0.35", lw=1.0, ls=":", zorder=1)
    ax.annotate(f"chosen dt = {CHOSEN_DT}", (CHOSEN_DT, 0.9), xycoords=("data", "axes fraction"),
                xytext=(-5, 0), textcoords="offset points", rotation=90,
                fontsize=7.5, color="0.35", ha="right", va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Ticks are the SWEPT dt values, written out. A log axis labelled 10^-1 / 10^0 makes the
    # reader interpolate to find which parameter each point is, which is the one thing this
    # figure exists to let them read off directly.
    ax.set_xticks(DT_TICKS)
    ax.set_xticklabels([f"{v:g}" for v in DT_TICKS], fontsize=8)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("settle step size  dt", fontsize=9)
    ax.set_ylabel("settled displacement", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
