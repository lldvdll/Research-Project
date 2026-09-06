"""Which metrics keep the sign of the PC - backprop difference as the stopping point moves?

900-series PLOT SCRIPT. Loads a saved array, trains nothing.

ONE PLOT, one grid cell, as script_plan_800_900.md specifies for 908 ("copy 112").

⚠ APPENDIX ONLY, AND CITED FROM RESULTS -- NOT FROM METHODS. This figure is inherently a paired
PC-versus-backprop measurement, so using it to CHOOSE the metric would select the instrument by
the answer it gives on the very comparison it is then used for. The metric is derived in 907 from
single-rule failures alone. This runs afterwards, as a robustness check on a result that has
already been established, and the report cites it from R2 in exactly that role.

WHAT IT SHOWS. Five metrics x five stopping thresholds, paired PC - backprop with 2 SEM bars:
    Class-IL    crossover and crossover-of-peak hold ONE sign with bars clear of zero at all five
                thresholds. Every endpoint metric flips sign somewhere across the range.
    Domain-IL   nothing survives, crossover included -- consistent with there being no effect
                there to be robust about.

The Domain-IL half matters as much as the Class-IL half: a robustness check that only ever
confirms is not a check. Draw both.

Styling follows 112, the script this regenerates: dpi 120, bbox_inches="tight", 9pt labels.
No shared style module.

PROVENANCE
    112_which_metric_survives.npz   PRE-800. Paired PC - backprop means and SEMs at five
    thresholds, both scenarios. Only summary statistics were saved, not per-seed values, so the
    bars here are the stored SEMs rather than recomputed ones.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

SOURCE = ROOT / "experiments" / "112_which_metric_survives.npz"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
# Ordered: the two that survive first, then the endpoint metrics that do not.
METRICS = [("crossover", "crossover"),
           ("crossover_of_peak", "crossover of peak"),
           ("final_t1", "final task-1"),
           ("area_retained", "area retained"),
           ("sb_mean_err", "S&B mean error")]
COLORS = ["tab:blue", "tab:cyan", "tab:orange", "tab:red", "tab:brown"]


if __name__ == "__main__":
    z = np.load(SOURCE, allow_pickle=True)
    thr = z["thresholds"] * 100.0

    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(11.0, 4.2), sharey=True)
    for ax, s in zip(axes, SCENARIOS):
        ax.axhline(0, color="k", lw=1.2)
        for (key, lab), c in zip(METRICS, COLORS):
            m = z[f"{key}_{s}_mean"]
            e = z[f"{key}_{s}_sem"]
            survives = np.all(m - 2 * e > 0) or np.all(m + 2 * e < 0)
            ax.errorbar(thr, m, yerr=2 * e, marker="o", ms=4, capsize=3,
                        lw=2.2 if survives else 1.2,
                        alpha=1.0 if survives else 0.45, color=c,
                        label=f"{lab}{'  (holds sign)' if survives else ''}")
            print(f"  {NICE[s]:10s} {lab:18s} " + " ".join(f"{v:+6.2f}" for v in m)
                  + ("   HOLDS SIGN at 2 SEM" if survives else "   flips"))
        ax.set_xlabel("stopping threshold (%)", fontsize=9)
        ax.set_title(NICE[s], fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("paired PC $-$ backprop (pp), bars = 2 SEM", fontsize=9)
    fig.suptitle("Robustness check, cited from Results — not the basis for choosing the metric",
                 fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
