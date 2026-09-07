"""What moves the outcome distribution, rather than the mean?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R5.

TWO PLOTS, ONE PER GRID CELL, matching 007's two cards:
    921_retention_distribution_a.png   the distribution, with interventions overlaid (SK_DIST)
    921_retention_distribution_b.png   the freeze factorial underneath it, 130 regenerated

⚠ THE POINT IS THAT THE MEAN IS THE WRONG SUMMARY. Class-IL retention at 70 seeds is TWO GROUPS --
a large one at or near zero and a second around 15-30% -- so the reported mean is a number no
individual run produces. Domain-IL is a broad unimodal spread instead. That difference in SHAPE is
itself evidence for the scenario split, and it is invisible in any figure that plots a mean with an
error bar. Every other retention number in this report should be read against this figure.

WHAT MOVES IT. Masking shifts the whole Class-IL distribution bodily (+46.11 in 806/922, +6.5 to
+17.5 across output specifications in 120/903b). Freezing does not move it at all. The overlay
makes "what moves the distribution" the question, rather than "what moves the mean".

⚠ AN ASYMMETRY IN THE SOURCES, AND THE FIGURE SAYS SO. The large block exists for CLASS-IL ONLY.
Domain-IL is drawn from the 10-seed block (310, seeds 10-19). So the Class-IL shape claim is well
sampled and the Domain-IL one is suggestive; the panel labels carry the seed count rather than
letting the two read as equally supported.

⚠ AND IT IS FIFTY SEEDS, NOT SEVENTY. 007's card says "over 70 seeds". The saved array holds 50,
and its own summary elsewhere says "18/50 above 5%" -- so 70 is a slip that has been repeated
forward. The figure labels the panel with the count it actually loaded (50) and the report should
quote 50. Verified: mean 6.58, median 1.13, range 0.0-38.1, 64% of seeds below 5%, which matches
the recorded 313 numbers exactly.

WHAT (b) ADDS. 130's factorial, regenerated: freezing the output layer recovers nothing in either
scenario, freezing the hidden layer is strongly negative, and FREEZE-BOTH IS ABSENT BECAUSE
CROSSOVER IS UNDEFINED ON 0/10 SEEDS THERE -- a result, not a gap. That null is what makes 922's
partial-column experiment necessary.

Styling follows 101 and the 300-series scripts: tab: colours by scenario, dpi 120,
bbox_inches="tight", 9pt labels. No shared style module.

PROVENANCE
    313_class_il_scale_check.npz   PRE-800. Class-IL, backprop, FIFTY seeds, current protocol.
                                   The largest seed block in the project.
    310_forgetting_by_scenario_domain_il.npz   backprop, seeds 10-19 -- ten only.
    806_partial_column_freeze.npz  the masking arm, Class-IL, seeds 10-19.
    130_freeze_factorial.npz       PRE-800. Both scenarios, backprop and pc, current protocol.
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
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
METHOD_COLOR = {"backprop": "0.35", "pc": "tab:orange"}
FREEZE = ["control", "freeze_w1", "freeze_w2", "freeze_both"]
FNICE = {"control": "control", "freeze_w1": "freeze $W_1$",
         "freeze_w2": "freeze $W_2$", "freeze_both": "freeze both"}


def masked_finals():
    d = np.load(EXP / "806_partial_column_freeze.npz", allow_pickle=True)
    return np.array([float(d[f"final_t1_{i}"]) for i in range(len(d["conditions"]))
                     if str(d["conditions"][i]) == "mask" and str(d["methods"][i]) == "backprop"])


def panel_a():
    finals = {"class_il": np.asarray(np.load(EXP / "313_class_il_scale_check.npz",
                                             allow_pickle=True)["finals"], float),
              "domain_il": np.asarray(np.load(EXP / "310_forgetting_by_scenario_domain_il.npz",
                                              allow_pickle=True)["finals"], float)}
    bins = np.arange(0, 102, 5)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    for ax, s in zip(axes, SCENARIOS):
        v = finals[s]
        ax.hist(v, bins=bins, color=COLORS[s], alpha=0.85,
                label=f"{NICE[s]}, backprop ({v.size} seeds)")
        ax.axvline(v.mean(), color="k", ls="--", lw=1.6)
        ax.annotate(f"mean {v.mean():.1f}%", (v.mean(), ax.get_ylim()[1] * 0.72),
                    xytext=(6, 0), textcoords="offset points", fontsize=8)
        ax.set_ylabel("seeds", fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=8, loc="upper right")
        print(f"  {NICE[s]:10s} n={v.size}  mean {v.mean():5.2f}  median {np.median(v):5.2f}  "
              f"range {v.min():.1f}-{v.max():.1f}  frac below 5%: {100 * (v < 5).mean():.0f}%")
    # the intervention that moves it, on the Class-IL panel where it is defined
    mk = masked_finals()
    axes[0].hist(mk, bins=bins, color="tab:red", alpha=0.45,
                 label=f"…with absent classes masked ({mk.size} seeds)")
    axes[0].axvline(mk.mean(), color="tab:red", ls="--", lw=1.6)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Class-IL is two groups — the mean is a number no run produces. "
                      "Masking moves the whole distribution.", fontsize=9)
    axes[1].set_xlabel("final task-1 accuracy (%)", fontsize=9)
    print(f"  masked (Class-IL)  mean {mk.mean():5.2f}  range {mk.min():.1f}-{mk.max():.1f}")
    fig.tight_layout()
    out = figure_path(__file__, "a")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


def panel_b():
    z = np.load(EXP / "130_freeze_factorial.npz", allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, s in zip(axes, SCENARIOS):
        x = np.arange(len(FREEZE))
        width = 0.36
        for k, rule in enumerate(("backprop", "pc")):
            mu, se, note = [], [], []
            for c in FREEZE:
                v = np.asarray(z[f"crossover_{s}_{rule}_{c}"], float)
                ok = np.isfinite(v)
                mu.append(np.nanmean(v) if ok.any() else 0.0)
                se.append(np.nanstd(v[ok], ddof=1) / np.sqrt(ok.sum()) if ok.sum() > 1 else 0.0)
                note.append(int((~ok).sum()))
            ax.bar(x + (k - 0.5) * width, mu, width, yerr=se, capsize=3,
                   color=METHOD_COLOR[rule], alpha=0.85, label=rule)
            for i, n in enumerate(note):
                if n:
                    ax.annotate(f"{n}/{len(np.asarray(z[f'crossover_{s}_{rule}_{FREEZE[i]}']))}\n"
                                f"censored", (x[i] + (k - 0.5) * width, 2),
                                ha="center", fontsize=6, color="crimson", linespacing=1.1)
            d, se_, nsem = paired_diff(
                np.asarray(z[f"crossover_{s}_{rule}_freeze_w2"], float),
                np.asarray(z[f"crossover_{s}_{rule}_control"], float))
            print(f"  {NICE[s]:10s} {rule:9s} freeze_w2 vs control {d:+6.2f} ± {se_:.2f} "
                  f"({nsem:.1f} sem)")
        ax.set_xticks(x)
        ax.set_xticklabels([FNICE[c] for c in FREEZE], fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        ax.set_title(NICE[s], fontsize=9)
    axes[0].set_ylabel("crossover (%)", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Freezing the output layer recovers nothing — and freeze-both has no crossover "
                 "on any seed, which is a result", fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__, "b")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    panel_a()
    panel_b()
