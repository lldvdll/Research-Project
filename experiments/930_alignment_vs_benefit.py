"""Does the mechanism credited with PC's advantage move where the advantage moves?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R3.

THE FIGURE IS THE MISMATCH BETWEEN ITS ROWS, and that is the whole argument. Columns are the
three conditions the project can compare, ordered by what PC does in them:

    Class-IL / tanh        PC helps
    Domain-IL / tanh       PC hurts        <- Song & Bogacz's own scenario
    Domain-IL / sigmoid    PC helps most   <- the largest standardised effect in the project

Rows are the benefit and then the mechanism claimed to produce it. If target alignment were
carrying the effect, row 2 would track row 1 across the columns.

⚠ IT DOES -- ON SIGN -- AND AN EARLIER VERSION OF THIS FILE CLAIMED THE OPPOSITE. Row 2 is
+0.0236 (4.6 sem) where PC helps, null at 1.3 sem where PC hurts, and +0.0048 (7.9 sem) where
PC helps most. Three out of three, with the null landing exactly on the negative column. That is
evidence CONSISTENT WITH Song & Bogacz, and the figure must not be captioned as a refutation.
The earlier claim came from reading only row 3 for the sigmoid column and generalising.

WHAT EACH ROW IS
    1  paired PC - backprop CROSSOVER, per seed. The benefit, in the metric the report defends.
    2  paired difference in TARGET ALIGNMENT on the batch being trained -- the quantity S&B
       credit. Positive means PC's updates point more directly at the target.
    3  paired difference in INTERFERENCE ALIGNMENT, measured on task-1 data during task 2.
       Negative means the update moves task 1 FURTHER from its targets.

EVERYTHING IS PAIRED, and it has to be. Backprop and PC see the same split and the same
initialisation at a given seed, and the split explains far more of the outcome than the rule
does (933 draws that). Group means would put every effect here inside its own noise; 906 is the
licence for the paired form.

WHAT THE FIGURE ACTUALLY SUPPORTS, stated carefully because the first pass got it wrong:

    row 2 tracks the SIGN of row 1 in all three columns. On that evidence alone the credited
    mechanism survives, and the honest reading is support rather than refutation.

WHAT STILL DOES NOT ADD UP, and both belong in the caption:

    MAGNITUDE DOES NOT SCALE. Class-IL has five times the alignment difference of the sigmoid
    column (+0.0236 against +0.0048) for a comparable benefit (+1.27 against +0.74). A quantity
    that causes the effect would be expected to move with it, and this does not.

    ROW 3 INVERTS WHERE THE BENEFIT IS LARGEST IN THE FIRST COLUMN. In Class-IL PC's updates
    move task 1 FURTHER from its targets than backprop's (-0.017, 3.1 sem) while PC forgets
    LESS. Better alignment on the trained batch and worse alignment on the retained task, in the
    condition where PC wins most, is not a clean mechanism story either way.

COLOURS follow the project standard. The plotted quantity is a PC-minus-backprop difference
rather than either rule on its own, so it is drawn in PC's RED against a black zero line.

PROVENANCE
    802_activation_sweep.npz        crossover per (scenario, activation, method, seed); the
                                    tanh and sigmoid columns both come from this one sweep, so
                                    the three columns are on one protocol.
    803_mechanism_logged_{sc}.npz   alignment per update, tanh, both scenarios, seeds 10-19.
    808_sigmoid_domain_mechanism_domain_il.npz   the same instrumentation under sigmoid.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
PC_COLOR = "tab:red"

# (label, scenario, activation, mechanism-array file)
COLUMNS = [
    ("Class-IL · tanh\nPC helps", "class_il", "tanh", "803_mechanism_logged_class_il.npz"),
    ("Domain-IL · tanh\nPC hurts", "domain_il", "tanh", "803_mechanism_logged_domain_il.npz"),
    ("Domain-IL · sigmoid\nPC helps most", "domain_il", "sigmoid",
     "808_sigmoid_domain_mechanism_domain_il.npz"),
]
ROWS = [("paired $\\Delta$ crossover (pp)", "crossover"),
        ("paired $\\Delta$ target\nalignment", "align"),
        ("paired $\\Delta$ interference\nalignment", "align_ref")]


def crossover_pairs(scenario, act):
    """Per-seed (backprop, pc) crossover from the activation sweep."""
    rows = np.load(EXP / "802_activation_sweep.npz", allow_pickle=True)["data"]
    by = {}
    for r in rows:
        if str(r[0]) == scenario and str(r[1]) == act:
            by.setdefault(int(r[3]), {})[str(r[2])] = float(r[6])
    seeds = sorted(s for s, v in by.items() if "backprop" in v and "pc" in v)
    return np.array([by[s]["pc"] - by[s]["backprop"] for s in seeds])


def alignment_pairs(fname, key):
    """Per-seed paired difference in a post-switch mean alignment quantity."""
    d = np.load(EXP / fname, allow_pickle=True)
    per = {}
    for i, (m, s) in enumerate(zip(d["methods"], d["seeds"])):
        sw = int(d[f"switch0_{i}"])
        # align_/align_ref_ are (step, value) PAIRS, not a bare series -- column 0 is the
        # update index. Averaging the whole array silently averages step numbers into the
        # alignment and produces values in the hundreds.
        a = np.asarray(d[f"{key}_{i}"], float)
        post = a[:, 0] > sw
        if post.sum() < 3:
            continue
        per.setdefault(int(s), {})[str(m)] = float(a[post, 1].mean())
    seeds = sorted(s for s, v in per.items() if "backprop" in v and "pc" in v)
    return np.array([per[s]["pc"] - per[s]["backprop"] for s in seeds])


if __name__ == "__main__":
    fig, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(7.0, 5.4))
    rng = np.random.RandomState(0)

    for r, (ylab, key) in enumerate(ROWS):
        for c, (title, scenario, act, fname) in enumerate(COLUMNS):
            ax = axes[r][c]
            vals = crossover_pairs(scenario, act) if key == "crossover" \
                else alignment_pairs(fname, key)
            mean = vals.mean()
            sem = vals.std(ddof=1) / np.sqrt(len(vals))
            ax.axhline(0, color="black", lw=1.0, zorder=2)
            ax.scatter(rng.uniform(-0.16, 0.16, len(vals)), vals, s=16,
                       color=PC_COLOR, alpha=0.5, zorder=3)
            ax.errorbar([0], [mean], yerr=[sem], color=PC_COLOR, marker="o", ms=6,
                        lw=2, capsize=4, zorder=4)
            ax.annotate(f"{mean:+.3g}\n{abs(mean/sem):.1f} sem" if key != "crossover"
                        else f"{mean:+.2f}\n{abs(mean/sem):.1f} sem",
                        (0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                        fontsize=7, fontweight="bold")
            ax.set_xlim(-0.45, 0.45)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(alpha=0.18, axis="y")
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=8)
            print(f"  row {r+1} {key:10s} {scenario:10s} {act:8s} "
                  f"{mean:+.4f} +- {sem:.4f}  ({abs(mean/sem):.1f} sem, n={len(vals)})")

    fig.tight_layout(pad=0.5)
    out = figure_path(__file__)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
