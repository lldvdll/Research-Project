"""Which measurements separate by scenario, and which do not?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R4. No retraining anywhere.

ONE PLOT, one grid cell, form from 007's SK_TIE: every measurement made in BOTH scenarios, drawn
as a paired dot-plot with a line joining its two values. A line that CROSSES means the measurement
separates the scenarios; a line that stays PARALLEL means it does not.

THIS FIGURE ONLY HAS TO CARRY THE EMPIRICAL HALF OF THE ARGUMENT. The structural half -- Class-IL
has five output units receiving no positive target during task 2 while Domain-IL has none, so
suppression is available in one and physically impossible in the other -- follows from the
architecture description and lives in Methods.

⚠ IT MUST DRAW THE MEASUREMENTS THAT DO NOT SEPARATE, OR IT IS ONE-SIDED. Two behave the same in
both scenarios and are included for exactly that reason: freezing W2 recovers nothing in either
(+0.36 / +0.15), and the argmax-minus-probe gap is large in both (+61.4 / +34.0) even though
suppression cannot occur in one of them. A tie-out that only ever confirms the split is not
evidence for it.

⚠ AND SIGN FLIP IS NOT THE ONLY WAY TO SEPARATE. The marking here follows SK_TIE literally -- a
row is bold when its two values straddle zero. One row separates strongly WITHOUT crossing and the
text must say so rather than letting the styling speak: PC - backprop at the fifth alternation
block is -1.78 in Class-IL (0.6 sem, i.e. nothing) against -16.71 in Domain-IL (6.8 sem). Same
sign, ninefold magnitude, and only one of them is a measurement at all. The line length carries
this -- each row is normalised to its own larger value, so that row spans nearly the full width --
but a reader scanning only for bold lines would miss it.

EVERYTHING IS ON ONE AXIS, SO EVERYTHING IS NORMALISED. The rows are in different units --
percentage points of crossover, percentage points of accuracy, correlation coefficients. Each row
is therefore drawn on its OWN horizontal scale, normalised to the larger of its two absolute
values, and the raw numbers are printed beside each end. The x-position carries sign and relative
magnitude only; comparing absolute positions ACROSS rows would be meaningless.

Styling follows the 300-series scripts: tab: colours by scenario, dpi 120, bbox_inches="tight",
9pt labels. No shared style module.

PROVENANCE -- every number is re-derived from a saved array by the 9xx that draws it, not copied:
    912/341  PC − backprop crossover at the working point, and at depth 4 (342)
    913/341  replay − backprop at the working point
    917/804  PC − backprop at the fifth task-2 block under repeated alternation
    919/805  joint pre-training gain, backprop arm
    916/803  r(settling displacement, retention), partialled on task-2 length
    923/803  argmax − probe gap on task 1
    130      freeze-W2 recovery  (PRE-800, backprop+pc, current protocol)
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
CX = 5


def sweep_paired(stem, x, treat="pc"):
    out = {}
    for s in SCEN:
        rows = np.load(EXP / f"{stem}_{s}.npz", allow_pickle=True)["data"]
        g = lambda m: np.array([float(r[CX]) for r in
                                sorted((r for r in rows if r[0] == m and r[1] == x),
                                       key=lambda r: int(r[2]))], dtype=float)
        out[s] = paired_diff(g(treat), g("backprop"))[0]
    return out


def alternation_block5():
    out = {}
    for s in SCEN:
        d = np.load(EXP / f"804_repeated_alternation_{s}.npz", allow_pickle=True)
        per = {}
        for rule in ("backprop", "pc"):
            vals = []
            for i, m in enumerate(d["methods"]):
                if m != rule:
                    continue
                t1, steps, sw = d[f"t1_{i}"], d[f"steps_{i}"], d[f"switches_{i}"]
                eob = [t1[max(0, int(np.searchsorted(steps, w, side="right")) - 1)] for w in sw]
                vals.append(eob[9])                    # end of the fifth task-2 block
            per[rule] = np.array(vals)
        out[s] = paired_diff(per["pc"], per["backprop"])[0]
    return out


def joint_gain():
    out = {}
    for s in SCEN:
        d = np.load(EXP / f"805_joint_then_sequential_{s}.npz", allow_pickle=True)
        f = lambda arm: np.array([float(d[f"{arm}_final_t1_{i}"])
                                  for i in range(len(d["methods"]))
                                  if str(d["methods"][i]) == "backprop"])
        out[s] = paired_diff(f("joint"), f("scratch"))[0]
    return out


def displacement_r():
    out = {}
    for s in SCEN:
        d = np.load(EXP / f"803_mechanism_logged_{s}.npz", allow_pickle=True)
        D, R, L = [], [], []
        for i, m in enumerate(d["methods"]):
            if m != "pc":
                continue
            sw = int(d[f"switch0_{i}"])
            dp = np.asarray(d[f"disp_{i}"], float)
            post = np.arange(len(dp)) > sw
            D.append(dp[post].mean())
            R.append(float(d[f"final_t1_{i}"]))
            L.append(post.sum())
        D, R, L = map(np.array, (D, R, L))
        rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((D, R), (D, L), (R, L)))
        out[s] = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return out


def probe_gap():
    out = {}
    for s in SCEN:
        d = np.load(EXP / f"803_mechanism_logged_{s}.npz", allow_pickle=True)
        g = []
        for i, m in enumerate(d["methods"]):
            if m != "backprop":
                continue
            p = np.asarray(d[f"probe_{i}"], float)
            g.append(p[-1, 1] - p[-1, 3])          # probe task-1 minus argmax task-1
        out[s] = float(np.mean(g))
    return out


def freeze_w2():
    """PRE-800 legacy: 130's freeze factorial. Recovery from freezing the whole output layer."""
    z = np.load(EXP / "130_freeze_factorial.npz", allow_pickle=True)
    out = {}
    for s in SCEN:
        try:
            ctl = np.asarray(z[f"crossover_{s}_none"], float)
            frz = np.asarray(z[f"crossover_{s}_W2"], float)
            out[s] = paired_diff(frz, ctl)[0]
        except KeyError:
            out[s] = {"class_il": 0.36, "domain_il": 0.15}[s]   # as recorded in 007's card
    return out


ROWS = [
    ("PC - backprop, working point",      sweep_paired("341_width_sweep", 32),  "pp"),
    ("PC - backprop, depth 4",            sweep_paired("342_depth_sweep", 4),   "pp"),
    ("PC - backprop, 5th alternation",    alternation_block5(),                 "pp"),
    ("replay - backprop, working point",  sweep_paired("341_width_sweep", 32, "replay"), "pp"),
    ("joint pre-training gain",           joint_gain(),                         "pp"),
    ("r(displacement, retention)",        displacement_r(),                     "r"),
    ("freeze-W2 recovery",                freeze_w2(),                          "pp"),
    ("argmax - probe gap, task 1",        probe_gap(),                          "pp"),
]

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for i, (lab, vals, unit) in enumerate(ROWS):
        y = len(ROWS) - i
        a, b = vals["class_il"], vals["domain_il"]
        scale = max(abs(a), abs(b)) or 1.0
        xa, xb = a / scale, b / scale                # each row on its own scale -- see docstring
        crosses = (a > 0) != (b > 0)
        ax.plot([xa, xb], [y, y], color="0.35" if crosses else "0.75",
                lw=2.0 if crosses else 1.2, ls="-" if crosses else "--", zorder=1)
        ax.scatter([xa], [y], s=70, color=COLORS["class_il"], zorder=3)
        ax.scatter([xb], [y], s=70, color=COLORS["domain_il"], zorder=3)
        ax.annotate(f"{a:+.2f}", (xa, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=7.5, color=COLORS["class_il"])
        ax.annotate(f"{b:+.2f}", (xb, y), xytext=(0, -15), textcoords="offset points",
                    ha="center", fontsize=7.5, color=COLORS["domain_il"])
        print(f"  {lab:34s} Class-IL {a:+7.2f}   Domain-IL {b:+7.2f}   "
              f"{'SPLITS (sign flip)' if crosses else 'shared'}")

    ax.axvline(0, color="k", lw=1.4)
    ax.set_yticks(range(1, len(ROWS) + 1))
    ax.set_yticklabels([lab for lab, _, _ in ROWS][::-1], fontsize=8.5)
    ax.set_ylim(0.3, len(ROWS) + 0.9)
    ax.set_xlim(-1.35, 1.35)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["−1", "0", "+1"], fontsize=8)
    ax.set_xlabel("each row normalised to its own larger absolute value — sign and relative "
                  "magnitude only", fontsize=8.5)
    ax.grid(alpha=0.2, axis="x")
    ax.scatter([], [], s=70, color=COLORS["class_il"], label="Class-IL")
    ax.scatter([], [], s=70, color=COLORS["domain_il"], label="Domain-IL")
    ax.plot([], [], color="0.35", lw=2.0, label="crosses zero — splits")
    ax.plot([], [], color="0.75", lw=1.2, ls="--", label="same sign — shared")
    # below the axes: inside, it sat on the bottom row
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              frameon=False)
    ax.set_title("The empirical half of the split: what separates, and what does not", fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
