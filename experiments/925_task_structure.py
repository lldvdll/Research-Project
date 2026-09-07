"""Does task structure predict what survives?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R5.

ONE PLOT, one grid cell, both scenarios, opposite outcomes -- which is 007's point for this card:
the data-side explanation works in one scenario and fails in the other.

    Domain-IL   class-pair similarity at the shared output unit. The leading data-side
                explanation, and it DOES NOT REPLICATE: r = +0.686 (p = 0.029) on seeds 0-9
                became r = +0.183 (p = 0.61) on an independent block, seeds 10-19.
    Class-IL    digit identity DOES predict: 8, 9 and 5 in task 1 raise retention and 6 lowers it,
                surviving Bonferroni over the larger seed block.

⚠ THE NON-REPLICATION IS THE RESULT, NOT A GAP. Pooling the two blocks gives r = +0.509
(p = 0.022) and that number should not be quoted on its own -- it averages a block where the
effect was present with one where it was absent, and reports neither. An effect that appears at
r = +0.69 in one seed block and r = +0.18 in the next is a warning about the sample size the rest
of the project runs at, which is why this belongs in Results rather than an appendix.

⚠ WHAT THIS FIGURE CAN AND CANNOT DRAW FROM THE SAVED ARRAY. 311 stores ONE similarity/retention
block of ten seeds per scenario. The r = +0.686 / +0.183 comparison between blocks is recorded in
the project notes and CANNOT be re-derived here, because the second block's similarity values were
not saved alongside the first. The panel therefore draws the block it has, prints the r it
computes, and states the other number as reported rather than as re-derived. Do not present the
non-replication as verified by this script -- it is verified by the notes, and closing that gap
means re-running the pairing analysis on both seed blocks and saving both.

Styling follows the 300-series scripts: tab: colours by scenario, dpi 120, bbox_inches="tight",
9pt labels. No shared style module.

PROVENANCE
    311_task_pair_similarity.npz   PRE-800. Per-seed class-pair similarity and final task-1
                                   accuracy, ten seeds, both scenarios, plus the 10x10 class
                                   similarity matrix the pairings are drawn from.
    313_class_il_scale_check.npz   the larger Class-IL block, used for the digit-identity half:
                                   which digits fell in task 1 is recovered from Protocol.tasks.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, replace, figure_path

EXP = ROOT / "experiments"
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
REPORTED = "r = +0.686 (p = 0.029) seeds 0–9   vs   r = +0.183 (p = 0.61) seeds 10–19"


def digit_effect():
    """Retention split by whether each digit fell in task 1. Class-IL, the larger seed block."""
    z = np.load(EXP / "313_class_il_scale_check.npz", allow_pickle=True)
    finals = np.asarray(z["finals"], float)
    start = int(z["seed_start"]) if "seed_start" in z.files else 0
    proto = replace(PROTOCOL, scenario="class_il")
    rows = []
    for d in range(10):
        inn, out = [], []
        for k, f in enumerate(finals):
            t1 = proto.tasks(start + k)[0]
            (inn if d in t1 else out).append(f)
        rows.append((d, np.mean(inn), np.mean(out), len(inn)))
    return rows


if __name__ == "__main__":
    z = np.load(EXP / "311_task_pair_similarity.npz", allow_pickle=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # left: the Domain-IL data-side explanation, on the block that exists
    for s, simk, fink in (("domain_il", "domain_sim", "domain_finals"),
                          ("class_il", "classil_sim", "classil_finals")):
        x, y = np.asarray(z[simk], float), np.asarray(z[fink], float)
        r = np.corrcoef(x, y)[0, 1]
        ax1.scatter(x, y, s=38, color=COLORS[s], alpha=0.85,
                    label=f"{s.replace('_', '-')}  (r = {r:+.2f}, n = {x.size})")
        b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax1.plot(xs, np.polyval(b, xs), lw=1.3, color=COLORS[s], alpha=0.6)
        print(f"  {s:10s} r(similarity, retention) = {r:+.3f}  over {x.size} seeds")
    ax1.set_xlabel("class-pair similarity at the shared output unit", fontsize=9)
    ax1.set_ylabel("final task-1 accuracy (%)", fontsize=9)
    ax1.set_title("The Domain-IL explanation, on the one block that was saved", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)
    ax1.annotate("⚠ reported across blocks:\n" + REPORTED + "\nnot re-derivable from this array",
                 (0.02, 0.03), xycoords="axes fraction", fontsize=7, color="crimson",
                 linespacing=1.35)

    # right: the Class-IL digit-identity effect, which does hold
    rows = digit_effect()
    d = [r[0] for r in rows]
    diff = [r[1] - r[2] for r in rows]
    cols = ["tab:red" if v < 0 else COLORS["class_il"] for v in diff]
    ax2.bar(d, diff, color=cols, alpha=0.85)
    ax2.axhline(0, color="k", lw=1.2)
    ax2.set_xticks(range(10))
    ax2.set_xlabel("digit", fontsize=9)
    ax2.set_ylabel("retention when in task 1 $-$ when not (pp)", fontsize=9)
    ax2.set_title("Class-IL: digit identity does predict what survives", fontsize=9)
    ax2.grid(alpha=0.25, axis="y")
    for dd, v, n in zip(d, diff, [r[3] for r in rows]):
        print(f"  digit {dd}: in-task-1 mean {rows[dd][1]:5.2f}  else {rows[dd][2]:5.2f}  "
              f"diff {v:+5.2f}  (n={n})")

    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
