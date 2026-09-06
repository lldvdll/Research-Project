"""Does the forgetting result depend on the hidden nonlinearity or on the output maths?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Methods M2, appendix evidence.

TWO PLOTS, ONE PER GRID CELL, as script_plan_800_900.md specifies for 903 ("new + copy 120"):
    903_specification_a.png   activation sweep -- NEW, from run 802, form set by 007's SK_ACT
    903_specification_b.png   output maths and masking -- form copied from 120

SK_ACT asks for crossover as bars, one per activation, grouped by scenario, with the reading
"flat = the conclusion does not depend on it" written on the plot. That is what (a) draws, with
the paired PC - backprop difference underneath, because the paired difference is the quantity the
report actually claims on and the absolute bars alone cannot show a sign flip.

⚠ THE INSENSITIVITY ARGUMENT HALF FAILS, AND THE PLOT SAYS SO RATHER THAN AVERAGING IT AWAY.
Paired PC - backprop crossover:
    Class-IL    +1.27 (tanh)   +1.51 (sigmoid)   +2.53 (relu)     sign held
    Domain-IL   -0.53 (tanh)   +0.74 (sigmoid)   -0.88 (relu)     SIGN FLIPS under sigmoid
So tanh can be defended as "the choice does not change the finding" in Class-IL and cannot in
Domain-IL. Since the Domain-IL effect is small (about -0.7) this is a flip inside a narrow band
rather than the reversal of a large effect -- but it is a limitation and the caption states it.

Styling follows the legacy scripts these regenerate: tab: colours, dpi 120, bbox_inches="tight",
labels at 9pt, legends at 8pt. No shared style module.

PROVENANCE
    802_activation_sweep.npz          NEW. tanh/sigmoid/relu x both scenarios x bp+pc, seeds
                                      10-19, config_800.yaml, matched competence. lr is NOT
                                      re-searched per activation -- that would confound the
                                      activation with its calibration -- so absolute differences
                                      between activations are expected and are not the measurement.
    120_output_maths_and_masking.npz  PRE-800, regenerated. backprop only, no PC arm.
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
METHOD_COLOR = {"backprop": "0.35", "pc": "tab:orange"}          # as 340 already uses
ACTS = ["tanh", "sigmoid", "relu"]
# 802 row layout: scenario, act, method, seed, final_t1, final_t2, crossover, switch0, reached
CX = 6


def act_cells():
    rows = np.load(EXP / "802_activation_sweep.npz", allow_pickle=True)["data"]
    out = {}
    for r in rows:
        out.setdefault((str(r[0]), str(r[1]), str(r[2])), []).append((int(r[3]), float(r[CX])))
    return {k: np.array([v for _, v in sorted(vs)]) for k, vs in out.items()}


def panel_a():
    d = act_cells()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    x = np.arange(len(ACTS) * len(SCENARIOS))
    labels = [f"{a}\n{NICE[s]}" for s in SCENARIOS for a in ACTS]

    # top: absolute crossover, one bar per rule -- SK_ACT's "flat = does not depend on it"
    ax = axes[0]
    width = 0.36
    for k, m in enumerate(("backprop", "pc")):
        vals = [np.nanmean(d[(s, a, m)]) for s in SCENARIOS for a in ACTS]
        errs = [np.nanstd(d[(s, a, m)], ddof=1) / np.sqrt(np.isfinite(d[(s, a, m)]).sum())
                for s in SCENARIOS for a in ACTS]
        ax.bar(x + (k - 0.5) * width, vals, width, yerr=errs, capsize=3,
               color=METHOD_COLOR[m], alpha=0.85, label=m)
    ax.set_ylabel("crossover (%)", fontsize=9)
    ax.set_ylim(50, 85)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Crossover is flat across activations within each scenario", fontsize=9)

    # bottom: the paired difference, which is what the report claims on
    ax = axes[1]
    ax.axhline(0, color="k", lw=1.0)
    for s in SCENARIOS:
        xs = [i for i, lb in enumerate(labels) if NICE[s] in lb]
        m = [paired_diff(d[(s, a, "pc")], d[(s, a, "backprop")])[0] for a in ACTS]
        e = [paired_diff(d[(s, a, "pc")], d[(s, a, "backprop")])[1] for a in ACTS]
        ax.errorbar(xs, m, yerr=e, marker="o", capsize=3, lw=2, color=COLORS[s], label=NICE[s])
    ax.set_ylabel("PC - backprop (pp)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    ax.annotate("sign flips under sigmoid", (4, 0.74), xytext=(0, 24),
                textcoords="offset points", fontsize=8, color="crimson", ha="center",
                arrowprops=dict(arrowstyle="->", lw=1.1, color="crimson"))
    ax.set_title("Class-IL keeps its sign under all three; Domain-IL does not", fontsize=9)

    fig.tight_layout()
    out = figure_path(__file__, "a")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    for s in SCENARIOS:
        for a in ACTS:
            m, e, n = paired_diff(d[(s, a, "pc")], d[(s, a, "backprop")])
            print(f"  {NICE[s]:10s} {a:8s} pc-bp {m:+5.2f} +- {e:.2f} ({n:.1f} sem)")


def panel_b():
    """Output maths and masking. Form copied from 120."""
    z = np.load(EXP / "120_output_maths_and_masking.npz", allow_pickle=True)
    specs = [("mse_onehot", "MSE / one-hot"), ("ce_onehot", "cross-entropy / one-hot"),
             ("hinge_pm1", "hinge / $\\pm$1")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(specs))
    width = 0.36
    for k, (masked, lab, c) in enumerate([(False, "unmasked", "0.45"),
                                          (True, "absent classes masked", "tab:red")]):
        vals = [float(np.nanmean(z[f"crossover_{s}_{masked}"])) for s, _ in specs]
        errs = [float(np.nanstd(z[f"crossover_{s}_{masked}"], ddof=1)
                      / np.sqrt(np.isfinite(z[f"crossover_{s}_{masked}"]).sum())) for s, _ in specs]
        ax.bar(x + (k - 0.5) * width, vals, width, yerr=errs, capsize=3, color=c,
               alpha=0.85, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in specs], fontsize=8)
    ax.set_ylabel("crossover (%)", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("The output specification sets the magnitude; masking removes suppression "
                 "under all three", fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__, "b")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    for s, lab in specs:
        u = float(np.nanmean(z[f"crossover_{s}_False"]))
        m = float(np.nanmean(z[f"crossover_{s}_True"]))
        print(f"  {lab:24s} unmasked {u:5.2f}   masked {m:5.2f}   {m - u:+5.2f}")


if __name__ == "__main__":
    panel_a()
    panel_b()
