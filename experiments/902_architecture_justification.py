"""Is the trunk doing real work, and does H=32 sit clear of a capacity limit for BOTH rules?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Methods M2.

TWO PANELS, both answering setup questions rather than making a claim about forgetting:
    (a)  trunk power   -- 811: random trunk + probe, trained trunk + SAME probe, and the
         jointly-trained network read through its own head.
    (b)  capacity      -- 810: joint accuracy against hidden width, both rules.

WHY THESE REPLACE THE LEGACY ARRAYS. 101 and 100 were BACKPROP ONLY, and the report uses one
architecture decision for both rules. A backprop-only capacity curve cannot license H=32 for
PC, and 101's random arm was read through a trained head while its reference was a separately
trained network, so the readout differed between the two things being compared. 811 fits the
same ridge probe on both sets of features, which leaves the features as the only difference.

WHAT (a) SHOWS, and it is not only the control it was written to be. The trunk is worth
+13.0 / +16.4 points to backprop (Class-IL / Domain-IL) and +7.0 / +9.0 to PC -- real in every
cell, so the hidden layer is not a dead layer and "where does forgetting live" is a fair
question. But PC's trunk gain is roughly HALF backprop's in both scenarios, and PC's joint
ceiling sits 6-8 points lower throughout (b). That is a Methods control turning up something
the Results have to own rather than inherit quietly.

⚠ trained_probe lands within ~1.4 points of joint in all four cells, which is the check that
the probe is not the weak link: a linear readout refitted on the trained features recovers what
the network's own linear head gets. If those two had diverged, (a) would be measuring the probe.

⚠ A capacity sweep run at too short a budget produces a flat region indistinguishable from a
ceiling. Both runs use a flat 25k-iteration budget with NO early stopping and average the last
few evaluations, for exactly that reason.

COLOURS follow the project standard: backprop BLACK, pc RED; Domain-IL solid, Class-IL dashed.
src/style.py is deliberately NOT used -- it sets backprop grey, pc orange and Domain-IL teal, and
was retired as the source of truth on 2026-09-13. Do not import its RULE/SCENARIO dicts here.

PROVENANCE
    811_trunk_power.npz        H=32, 10 seeds, both rules, both scenarios, ridge probe 1e-3.
    810_capacity_vs_width.npz  widths 4-128, 5 seeds, both rules, both scenarios. PC runs at
                               dt = 0.2, not the 0.4 default: this sweeps WIDTH, and 346 found
                               Class-IL H=4 oscillates at 0.4.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
SCENARIOS = ["domain_il", "class_il"]          # solid first, per the standard
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
METHODS = ["backprop", "pc"]

RULE_COLOR = {"backprop": "black", "pc": "tab:red"}
SCEN_STYLE = {"domain_il": "-", "class_il": "--"}

ARMS = ["random_probe", "trained_probe", "joint"]
ARM_LABEL = ["random trunk\n+ probe", "trained trunk\n+ probe", "jointly\ntrained"]
CHOSEN_H = 32


def panel_a(ax):
    d = np.load(EXP / "811_trunk_power.npz", allow_pickle=True)
    x = np.arange(len(ARMS))
    for s in SCENARIOS:
        for m in METHODS:
            vals = np.stack([d[f"{s}_{m}_{a}"] for a in ARMS])      # (arms, seeds)
            # every seed drawn faintly, so the spread is visible rather than asserted
            for k in range(vals.shape[1]):
                ax.plot(x, vals[:, k], color=RULE_COLOR[m], ls=SCEN_STYLE[s],
                        lw=0.5, alpha=0.22, zorder=2)
            ax.plot(x, vals.mean(axis=1), color=RULE_COLOR[m], ls=SCEN_STYLE[s],
                    lw=1.8, marker="o", ms=4, zorder=4,
                    label=f"{m} · {NICE[s]}")
            gain = vals[1] - vals[0]
            print(f"  (a) {NICE[s]:10s} {m:9s} random {vals[0].mean():5.1f}  "
                  f"trained {vals[1].mean():5.1f}  joint {vals[2].mean():5.1f}  "
                  f"trunk gain {gain.mean():+5.1f} +- {gain.std(ddof=1)/np.sqrt(len(gain)):.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels(ARM_LABEL, fontsize=7.5)
    ax.set_ylabel("joint accuracy (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)


def panel_b(ax):
    d = np.load(EXP / "810_capacity_vs_width.npz", allow_pickle=True)
    widths = d["widths"]
    for s in SCENARIOS:
        for m in METHODS:
            a = d[f"acc_{s}_{m}"]                                    # (widths, seeds)
            mean = a.mean(axis=1)
            sem = a.std(axis=1, ddof=1) / np.sqrt(a.shape[1])
            ax.errorbar(widths, mean, yerr=sem, color=RULE_COLOR[m], ls=SCEN_STYLE[s],
                        lw=1.5, marker="o", ms=4, capsize=2, zorder=3,
                        label=f"{m} · {NICE[s]}")
            at32 = mean[list(widths).index(CHOSEN_H)]
            print(f"  (b) {NICE[s]:10s} {m:9s} H=32 {at32:5.1f}  "
                  f"H=128 {mean[-1]:5.1f}  headroom above H=32 {mean[-1]-at32:+4.1f}")
    ax.axvline(CHOSEN_H, color="0.4", lw=1.0, ls=":", zorder=1)
    ax.annotate(f"H = {CHOSEN_H}", (CHOSEN_H, 0.03), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", fontsize=7.5, color="0.4",
                ha="left", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(int(w)) for w in widths], fontsize=8)
    ax.set_xticks([], minor=True)
    ax.set_xlabel("hidden width  H", fontsize=9)
    ax.set_ylabel("joint accuracy (%)", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)


if __name__ == "__main__":
    for name, fn, size in [("a", panel_a, (3.9, 3.0)), ("b", panel_b, (3.9, 3.0))]:
        fig, ax = plt.subplots(figsize=size)
        fn(ax)
        fig.tight_layout()
        out = figure_path(__file__, suffix=name)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"saved {out}")
