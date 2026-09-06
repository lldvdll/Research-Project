"""What does pairing on seeds buy, and does it move the answer or only the uncertainty?

900-series PLOT SCRIPT. Loads a saved array, trains nothing. Methods M2, appendix evidence.

ONE PLOT, one grid cell, as script_plan_800_900.md specifies for 906 ("new", source: legacy 69).
Form is 007's SK_PAIR: THE SAME POINT ESTIMATE, TWO ERROR BARS -- unpaired beside paired.

WHY IT MATTERS ENOUGH TO DRAW. Every rule in this project sees the SAME class split and the SAME
initialisation at a given seed, so most of the between-seed variance is shared. Comparing group
means throws that away. The point estimate is identical either way; only the interval changes:

    PC     - backprop   +1.65   unpaired SEM 1.119   paired SEM 0.250   4.5x tighter
    replay - backprop   +8.66   unpaired SEM 1.106   paired SEM 0.533   2.1x tighter

The bars are the argument: unpaired, PC's +1.65 sits well inside its own interval and reads as
nothing. Paired, the same +1.65 is a 6.6-sem effect. Nothing about the runs changed.

⚠ ONE NUMBER IN THE TRACK IS NOT REPRODUCIBLE FROM THIS ARRAY. 007 quotes a shrink of "4.5x in
Class-IL and 11.5x in Domain-IL". The 4.5x reproduces exactly. 69's saved array contains ONE
scenario (24 seeds, switch 840) and no Domain-IL arm, so the 11.5x cannot be re-derived here and
is NOT drawn. Either its source is identified and this script repointed, or the report quotes only
the half that is backed. Do not copy the 11.5x forward on the strength of the 4.5x agreeing.

Styling follows the 300-series scripts: tab: colours, dpi 120, bbox_inches="tight", 9pt labels.
No shared style module.

PROVENANCE
    69_does_the_data_split_confound_the_comparison.npz   PRE-800. 24 seeds, backprop/replay/pc,
    switch at 840, fixed budget. Crossover is recomputed here from the saved accuracy curves with
    src.metrics.crossover rather than read from a stored scalar, so the metric matches every other
    figure in the report.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import crossover, paired_diff

SOURCE = ROOT / "experiments" / "69_does_the_data_split_confound_the_comparison.npz"
METHOD_COLOR = {"pc": "tab:orange", "replay": "tab:green"}       # as 340 already uses


if __name__ == "__main__":
    z = np.load(SOURCE, allow_pickle=True)
    steps, sw = z["steps"], int(z["switch"])
    cx = {}
    for m in z["methods"]:
        a = z[f"argmax_{m}"] * 100.0
        cx[str(m)] = np.array([crossover(steps, a[i, :, 0], a[i, :, 1], after=sw)[1]
                               for i in range(a.shape[0])])

    treatments = ["pc", "replay"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = 0.0
    ticks, labels = [], []
    for m in treatments:
        d, se_p, _ = paired_diff(cx[m], cx["backprop"])
        a, b = cx[m], cx["backprop"]
        ok = np.isfinite(a) & np.isfinite(b)
        se_u = np.sqrt(np.var(a[ok], ddof=1) / ok.sum() + np.var(b[ok], ddof=1) / ok.sum())

        ax.errorbar(d, y, xerr=se_u, fmt="o", ms=7, capsize=5, lw=2,
                    color=METHOD_COLOR[m], alpha=0.45)
        ax.errorbar(d, y - 0.42, xerr=se_p, fmt="o", ms=7, capsize=5, lw=2,
                    color=METHOD_COLOR[m])
        ax.annotate(f"{se_u / se_p:.1f}$\\times$ tighter", (d, y - 0.21), xytext=(0, 0),
                    textcoords="offset points", fontsize=8, ha="center", va="center",
                    color=METHOD_COLOR[m],
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
        ticks += [y, y - 0.42]
        labels += [f"{m} $-$ backprop, unpaired", f"{m} $-$ backprop, paired"]
        y -= 1.25
        print(f"  {m:7s} vs backprop  {d:+5.2f}   unpaired SEM {se_u:.3f}   "
              f"paired SEM {se_p:.3f}   {se_u / se_p:.1f}x tighter")

    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(min(ticks) - 0.4, max(ticks) + 0.4)
    ax.set_xlabel("$\\Delta$ crossover vs backprop (pp)", fontsize=9)
    ax.grid(alpha=0.25, axis="x")
    ax.set_title("Pairing does not move the estimate — it removes the shared seed variance "
                 "(Class-IL, 24 seeds)", fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
