"""What do the added-mechanism sweeps look like underneath the ranking?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Discussion D1, appendix evidence.

ONE PLOT, one grid cell, as script_plan_800_900.md specifies for 932 ("copy 210 + 220"): the two
sweeps 931 summarises to a single bar each, drawn in full.

WHY THE FULL SWEEP EARNS APPENDIX SPACE. 931 reports one cell per intervention, which is the right
summary and also the one that can mislead. Two things are only visible here:

    k-WTA is harmful MONOTONICALLY. Class-IL backprop: −2.47, −3.39, −6.43, −9.78 as k tightens
    from 24 to 3; PC: −7.70, −8.15, −15.48, −32.41. It is not a badly-chosen k, it is the
    mechanism, and it hurts PC roughly three times as much as backprop.
    ⚠ k = H = 32 IS THE CONTROL, not an intervention -- gating all 32 units gates nothing, and it
    returns +0.00 ± 0.00 exactly. It is drawn, marked, and excluded from any "best k".

    SI has an interior optimum, so its 931 bar is a genuine tuned result. EWC's does not, which is
    why EWC's bar is a lower bound and is hatched there.

Styling follows the 300-series scripts: tab: colours by scenario, dpi 120, bbox_inches="tight",
9pt labels. No shared style module.

PROVENANCE
    210_si_lambda_sweep.npz   PRE-800. SI lambda swept, both scenarios, backprop and pc.
    220_kwta_k_sweep.npz      PRE-800. k in {32, 24, 16, 8, 3} at H = 32.
    130_freeze_factorial.npz  supplies the shared control each is differenced against.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff

EXP = ROOT / "experiments"
SCEN = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
METHOD_LS = {"backprop": "-", "pc": "--"}
H = 32


def series(fname, key, gridkey):
    z = np.load(EXP / fname, allow_pickle=True)
    ctl = np.load(EXP / "130_freeze_factorial.npz", allow_pickle=True)
    grid = np.asarray(z[gridkey], float)
    out = {}
    for s in SCEN:
        for m in ("backprop", "pc"):
            arr = np.asarray(z[f"crossover_{s}_{m}_{key}"], float)
            base = np.asarray(ctl[f"crossover_{s}_{m}_control"], float)
            out[(s, m)] = np.array([paired_diff(arr[i], base)[0] for i in range(arr.shape[0])])
    return grid, out


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    grid, out = series("210_si_lambda_sweep.npz", "si", "lambdas")
    for (s, m), v in out.items():
        ax1.plot(grid, v, ls=METHOD_LS[m], marker="o", ms=4, lw=1.8, color=COLORS[s],
                 label=f"{NICE[s]} · {m}")
        print(f"  SI    {NICE[s]:10s} {m:9s} " + "  ".join(f"{g:g}:{x:+.2f}"
                                                           for g, x in zip(grid, v)))
    ax1.axhline(0, color="k", lw=1.2)
    ax1.set_xscale("log")
    ax1.set_xlabel("SI $\\lambda$", fontsize=9)
    ax1.set_ylabel("paired $\\Delta$ crossover vs control (pp)", fontsize=9)
    ax1.set_title("SI has an interior optimum — its 931 bar is a tuned result", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7.5)

    grid, out = series("220_kwta_k_sweep.npz", "kwta", "k_values")
    for (s, m), v in out.items():
        ax2.plot(grid, v, ls=METHOD_LS[m], marker="o", ms=4, lw=1.8, color=COLORS[s],
                 label=f"{NICE[s]} · {m}")
        print(f"  kWTA  {NICE[s]:10s} {m:9s} " + "  ".join(f"k={g:g}:{x:+.2f}"
                                                           for g, x in zip(grid, v)))
    ax2.axhline(0, color="k", lw=1.2)
    ax2.axvline(H, color="crimson", ls=":", lw=1.4)
    ax2.annotate(f"k = H = {H} gates nothing:\nthis IS the control", (H, -20),
                 xytext=(-8, 0), textcoords="offset points", ha="right", fontsize=7.5,
                 color="crimson", linespacing=1.3)
    ax2.set_xlabel("k-WTA $k$ (units left active, of $H=32$)", fontsize=9)
    ax2.set_title("k-WTA is harmful monotonically, and ~3× worse for PC", fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    out_path = figure_path(__file__)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")
