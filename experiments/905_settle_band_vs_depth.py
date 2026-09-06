"""Does the stable range of settle step sizes shrink as the network gets deeper?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Methods M2, appendix evidence.

ONE PLOT, one grid cell, as script_plan_800_900.md specifies for 905 ("copy 347"): a dt x depth
grid, cells coloured by the classification `src.metrics.classify_settle` returns.

WHY THIS FIGURE EXISTS AND WHY IT IS NOT A DETAIL. 904 establishes that a stable band exists and
that the choice inside it is free. This one says the band MOVES. dt = 0.4 was settled at
H = 32 / depth 1 and is silently wrong at depth >= 2: settling there does not converge, and the
oscillating tail sits 3-7x above the true fixed point, so the state being measured is
qualitatively wrong rather than slightly overshot. That cost two full sweep re-runs before it was
found, which is why 341 and 342 use dt = 0.2.

The standing rule this figure justifies: a control is re-verified whenever the configuration
moves, not once at the start.

Styling follows 347, the script this regenerates: dpi 120, bbox_inches="tight", 9pt labels.
No shared style module.

PROVENANCE
    347_pc_settle_dt_by_depth_{scenario}.npz   PRE-800, regenerated. Single-seed traces run to a
    fixed cap of 500 steps with no early exit; classify_settle is applied post-hoc to the full
    trace, which is what distinguishes "slow" from "oscillating" from "diverging".
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.protocol import figure_path
from src.metrics import classify_settle

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
SETTLE_DELTA_TOL, SETTLE_PATIENCE = 1e-4, 3

# Ordered worst-to-best so the colour ramp reads as "how bad is this cell".
LABELS = ["converged", "slow", "oscillating", "diverging"]
CELL_COLOR = ["#2c7fb8", "#a6bddb", "#fdae61", "#d7191c"]


if __name__ == "__main__":
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(9.6, 3.6), sharey=True)
    cmap = ListedColormap(CELL_COLOR)
    norm = BoundaryNorm(range(len(LABELS) + 1), cmap.N)

    for ax, s in zip(axes, SCENARIOS):
        rows = np.load(EXP / f"347_pc_settle_dt_by_depth_{s}.npz", allow_pickle=True)["data"]
        depths = sorted({int(r[0]) for r in rows})
        dts = sorted({float(r[1]) for r in rows})
        grid = np.full((len(depths), len(dts)), np.nan)
        for r in rows:
            di, ti = depths.index(int(r[0])), dts.index(float(r[1]))
            lab, _ = classify_settle(np.asarray(r[3], float),
                                     tol=SETTLE_DELTA_TOL, patience=SETTLE_PATIENCE)
            grid[di, ti] = LABELS.index(lab) if lab in LABELS else np.nan

        ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(len(dts)))
        ax.set_xticklabels(dts, fontsize=8)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels(depths, fontsize=8)
        ax.set_xlabel("dt", fontsize=9)
        ax.set_title(NICE[s], fontsize=9)
        for di in range(len(depths)):
            for ti in range(len(dts)):
                if np.isfinite(grid[di, ti]) and grid[di, ti] > 0:
                    ax.text(ti, di, LABELS[int(grid[di, ti])][:4], ha="center", va="center",
                            fontsize=6.5, color="0.15")
        print(f"  {NICE[s]}:")
        for di, dep in enumerate(depths):
            print(f"    depth {dep}: " + "  ".join(
                f"dt={dt}:{LABELS[int(grid[di, ti])][:4] if np.isfinite(grid[di, ti]) else '?'}"
                for ti, dt in enumerate(dts)))

    axes[0].set_ylabel("hidden layers", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CELL_COLOR]
    axes[-1].legend(handles, LABELS, fontsize=7.5, loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.suptitle("The stable dt band shrinks with depth — dt = 0.4 fails at depth $\\geq$ 2",
                 fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
