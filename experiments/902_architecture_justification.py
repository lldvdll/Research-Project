"""Is the trunk doing enough work, and is the width sufficient, to ask mechanistic questions?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Methods M2, appendix evidence.

TWO PLOTS, ONE PER GRID CELL, as script_plan_800_900.md specifies for 902 ("copy 101 + 100"):
    902_architecture_justification_a.png   trunk power   -- form copied from 101
    902_architecture_justification_b.png   width sweep   -- form copied from 100

Form and styling follow the 300-series/legacy scripts they regenerate: tab: colours keyed by
scenario, dpi 120, bbox_inches="tight", labels at 9pt and legends at 8pt. No shared style module.

WHAT EACH ONE IS FOR, as a table row rather than a result:
    (a)  a frozen random projection with a trained head against the fully-trained network. If the
         gap were near zero the hidden layer would be doing no work and "where does forgetting
         live" would be a question about a dead layer.
    (b)  joint accuracy against hidden width. H = 32 has to sit OFF the bottleneck, or a capacity
         limit would be indistinguishable from a forgetting effect.

⚠ A capacity sweep run at too short a budget produces a flat region that is indistinguishable from
a capacity ceiling. 100 measured convergence first; this only redraws it.

PROVENANCE -- both are PRE-800 arrays, regenerated here rather than re-run:
    101_problem_complexity.npz   frozen-vs-trained, 5 init seeds x 5 data seeds, H = 32
    100_capacity_vs_width.npz    widths 2..128, 10 seeds, both scenarios
The 800-series slots that would replace them (810, 811) are deliberately unwritten -- the legacy
protocol matches the current one for these two measurements.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}   # as 100/101 already use
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}


def panel_a():
    """Trunk power: frozen random projection vs fully trained. Form copied from 101."""
    z = np.load(EXP / "101_problem_complexity.npz", allow_pickle=True)
    z100 = np.load(EXP / "100_capacity_vs_width.npz", allow_pickle=True)
    widths = z100["widths"].tolist()
    wi = widths.index(int(z["H"]))
    frozen = {s: z[f"final_{s}"].ravel() for s in SCENARIOS}
    trained = {s: float(z100[f"acc_{s}"][wi].mean()) for s in SCENARIOS}

    allv = np.concatenate([frozen[s] for s in SCENARIOS] + [np.array(list(trained.values()))])
    lo = float(np.floor(np.nanmin(allv) / 2) * 2 - 2)
    hi = float(np.ceil(np.nanmax(allv) / 2) * 2 + 2)
    bins = np.arange(lo, hi + 2, 2)

    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(7.2, 4.2), sharex=True)
    for ax, s in zip(axes, SCENARIOS):
        ax.hist(frozen[s], bins=bins, color=COLORS[s], alpha=0.85,
                label=f"{NICE[s]}, frozen random trunk")
        ax.axvline(trained[s], color="k", ls="--", lw=1.8)
        ax.annotate(f"trained {trained[s]:.1f}%", (trained[s], ax.get_ylim()[1] * 0.82),
                    xytext=(6, 0), textcoords="offset points", fontsize=8)
        ax.set_ylabel("count", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_xlim(lo, hi)
    axes[-1].set_xlabel("joint accuracy (%)")
    axes[0].set_title(f"Training the hidden layer is worth "
                      f"{trained['class_il'] - frozen['class_il'].mean():.1f} points (Class-IL), "
                      f"{trained['domain_il'] - frozen['domain_il'].mean():.1f} (Domain-IL)",
                      fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__, "a")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    for s in SCENARIOS:
        print(f"  {NICE[s]:10s} frozen {frozen[s].mean():5.1f}  trained {trained[s]:5.1f}  "
              f"gap {trained[s] - frozen[s].mean():+5.1f}")


def panel_b():
    """Width sweep. Form copied from 100."""
    z = np.load(EXP / "100_capacity_vs_width.npz", allow_pickle=True)
    widths = z["widths"].tolist()
    chosen = int(z["chosen"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for s in SCENARIOS:
        a = z[f"acc_{s}"]
        mean = a.mean(axis=1)
        sem = a.std(axis=1, ddof=1) / np.sqrt(a.shape[1])
        ax.errorbar(widths, mean, yerr=sem, marker="o", capsize=3, lw=2,
                    color=COLORS[s], label=NICE[s])
    ax.axvline(chosen, color="k", ls="--", lw=1.4)
    ax.annotate(f"H = {chosen}, the working point", (chosen, 20), xytext=(8, 0),
                textcoords="offset points", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel("hidden units")
    ax.set_ylabel("joint accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("H = 32 sits off the bottleneck; H = 4 and H = 8 are capacity-limited",
                 fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__, "b")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    for s in SCENARIOS:
        a = z[f"acc_{s}"].mean(axis=1)
        print(f"  {NICE[s]:10s} " + "  ".join(f"H={w}: {v:.1f}" for w, v in zip(widths, a)))


if __name__ == "__main__":
    panel_a()
    panel_b()
