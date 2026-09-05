"""Slide asset (plot a): backprop only, fixed budget above / matched competence below, same
scenario -- demonstrates that the two protocols stop at different points, not that the rule
behaves differently. No retraining: reads 72 and 73's already-saved arrays.

The horizontal line is the ACCURACY THRESHOLD (90%), not crossover -- the point of this figure is
to show training running past that line in the fixed-budget panel (task 2 keeps going after
reaching 90%) while the matched-competence panel stops right at it. Crossover is a different
story, told elsewhere; this one is about the stopping criterion only.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import array_path, figure_path

SCENARIOS = ["domain_il", "class_il"]
THRESHOLD = 90.0
E = str(ROOT / "experiments") + "\\"

z72 = np.load(E + "72_grid_fixed_budget_symmetric.npz", allow_pickle=True)

for scn in SCENARIOS:
    z73 = np.load(E + f"73_grid_matched_competence_symmetric_{scn}.npz", allow_pickle=True)

    fixed_steps = z72[f"steps_{scn}"]
    fixed_switch = float(np.asarray(z72[f"switches_{scn}"]).ravel()[0])
    fixed = z72[f"argmax_{scn}_backprop"] * 100.0        # [seeds, evals, 2]

    matched_steps = z73["steps"]                          # already relative to switch, switch=0
    matched = z73["argmax_backprop"] * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=False)

    for ax, steps, switch, A, label in [
            (axes[0], fixed_steps - fixed_switch, 0.0, fixed, "fixed budget"),
            (axes[1], matched_steps, 0.0, matched, "matched competence")]:
        finite = np.isfinite(A).any(axis=(0, 2))
        fs = steps[finite]
        ax.axvspan(fs.min(), switch, color="tab:orange", alpha=0.10, lw=0)
        ax.axvspan(switch, fs.max(), color="tab:blue", alpha=0.10, lw=0)
        for t, c, lab in [(0, "tab:orange", "task 1"), (1, "tab:blue", "task 2")]:
            for r in range(A.shape[0]):
                ax.plot(steps, A[r, :, t], color=c, lw=0.7, alpha=0.2)
            mean = np.nanmean(A[:, :, t], axis=0)
            mean[np.isnan(A[:, :, t]).any(axis=0)] = np.nan
            ax.plot(steps, mean, color=c, lw=2.4, label=lab)
        ax.axhline(THRESHOLD, color="k", ls=":", lw=1.4)
        ax.annotate(f"threshold {THRESHOLD:.0f}%", xy=(0.99, THRESHOLD),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha="right", fontsize=8)
        ax.set_ylim(-2, 103)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"backprop — {label}")
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].set_xlabel("training step, relative to the task switch")
    label = "Domain IL" if scn == "domain_il" else "Class IL"
    fig.suptitle(f"{label} — same rule, two stopping criteria.\n"
                 "Fixed budget runs past the threshold; matched competence stops once it's met.\n"
                 "Stopping is checked on a held-out sample separate from the one plotted, so the "
                 "plotted curve can sit just under 90% even once the run has stopped.",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(figure_path(__file__, scn), dpi=130, bbox_inches="tight")
    print(f"saved {figure_path(__file__, scn)}")
