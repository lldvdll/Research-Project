"""Does PC's settling reach a fixed point, and does the step size move it?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Methods M2.

ONE PLOT, one grid cell, as script_plan_800_900.md specifies for 904 ("copy 334"): mean absolute
displacement against settle step, one line per dt, log y, both scenarios as panels.

THIS IS A CONTROL FOR PC BEING A FAITHFUL IMPLEMENTATION OF ITSELF, not a control for forgetting.
If settling oscillated, the thing being measured would not be PC. The reading:

    dt <= 0.5 all reach the IDENTICAL plateau -- 0.0850 (Class-IL), 0.0787 (Domain-IL).
    dt changes the ROUTE, not the destination, so the choice inside the stable band is free.

Above 0.5 the relaxation stops converging, which is why the band has an upper edge at all.

⚠ THE BAND IS NOT A ONE-TIME GATE. dt = 0.4 is correct at H = 32 / depth 1 and silently WRONG at
depth >= 2 and at Class-IL H = 4, where the tail sits 3-7x above the true fixed point. 905 draws
that; the sweeps at 341/342 use dt = 0.2 because of it. Re-verify whenever the configuration moves.

Styling follows 334, the script this regenerates: viridis by dt, log y, dpi 120,
bbox_inches="tight", labels at 9pt. No shared style module.

PROVENANCE
    334_pc_settle_trace_by_dt_{scenario}.npz   PRE-800, regenerated. Single-seed traces run to a
    fixed cap with no early exit, which is what makes the plateau comparison possible at all.
    script_plan's legacy table marks this source sufficient for what it claims.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
STABLE_MAX = 0.5          # the top of the stable band, from 330/334


if __name__ == "__main__":
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(9.6, 4.0), sharey=True)
    dts_all = sorted({float(r[0]) for r in
                      np.load(EXP / "334_pc_settle_trace_by_dt_class_il.npz",
                              allow_pickle=True)["data"]})
    norm = mpl.colors.LogNorm(vmin=min(dts_all), vmax=max(dts_all))
    cmap = mpl.colormaps["viridis"]

    for ax, s in zip(axes, SCENARIOS):
        rows = np.load(EXP / f"334_pc_settle_trace_by_dt_{s}.npz", allow_pickle=True)["data"]
        plateau = []
        for dt in dts_all:
            traces = [np.asarray(r[2], float) for r in rows if float(r[0]) == dt]
            if not traces:
                continue
            m = np.mean(np.vstack(traces), axis=0)
            ax.plot(np.arange(1, len(m) + 1), m, lw=1.6, color=cmap(norm(dt)),
                    ls="-" if dt <= STABLE_MAX else "--")
            if dt <= STABLE_MAX:
                plateau.append(m[-1])
        ax.set_yscale("log")
        ax.set_xlabel("settle step", fontsize=9)
        ax.set_title(f"{NICE[s]} — stable band plateau {np.mean(plateau):.4f}", fontsize=9)
        ax.grid(alpha=0.25, which="both")
        print(f"  {NICE[s]:10s} plateau over dt <= {STABLE_MAX}: "
              + "  ".join(f"{p:.4f}" for p in plateau))
    axes[0].set_ylabel("mean abs displacement", fontsize=9)
    axes[0].plot([], [], color="0.4", ls="-", label=f"dt $\\leq$ {STABLE_MAX} (converges)")
    axes[0].plot([], [], color="0.4", ls="--", label=f"dt > {STABLE_MAX} (does not)")
    axes[0].legend(fontsize=8, loc="lower left")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, label="dt", pad=0.015)
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
