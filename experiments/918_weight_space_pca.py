"""Are the weights converging, orbiting, or drifting under repeated alternation?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R4.

ONE PLOT, one grid cell, form from 007's SK_PCA: the weight trajectory projected onto its own
first two principal components, W1 and W2 SEPARATELY, both scenarios.

WHY THIS IS THE STRONGER VERSION OF 917. Accuracy can look settled while the parameters wander --
two different weight vectors can score the same on both tasks. 917 shows the accuracy trajectory
spiralling inward; this asks whether the WEIGHTS do, and it separates the trunk from the readout,
which is the question R5 then takes up.

THE SCALAR THAT STOPS IT BEING A PRETTY PICTURE. Per-block CONTRACTION is measured directly: the
distance in PCA space between the state at the end of block k and at the end of block k-2 -- the
same phase of the previous cycle. A closed loop returns to where it was, so that distance stays
FLAT. A converging system returns nearer each time, so it FALLS. The figure prints the ratio of
the last such distance to the first.

READ THE TWO LAYERS SEPARATELY. The question 007 poses is whether the trunk contracts while the
readout keeps orbiting. Drawing W1 and W2 on one axis would make that unanswerable, since they
have different dimensions (196x32 against 32x10) and different natural scales.

⚠ PCA IS FITTED PER RUN, so the axes are NOT comparable between panels -- each is that run's own
basis. Only the SHAPE within a panel and the contraction number are meaningful. Saying so matters:
a reader who compares PC1 across panels is comparing two different directions.

Styling follows the 300-series scripts: viridis for time, dpi 120, bbox_inches="tight", 9pt
labels. No shared style module.

PROVENANCE
    804_repeated_alternation_{scenario}.npz   weight traces stored for seeds 10-12 of each rule
    (TRACE_SEEDS = 3), every 10th eval, W1 (6272 = 196x32) and W2 (320 = 32x10) both saved.
    Legacy 68 stored W1 only, Domain-IL only, at a fixed budget; this supersedes it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
LAYERS = [("trace_W1", "$W_1$ (hidden)"), ("trace_W2", "$W_2$ (output)")]
SHOW_RULE, SHOW_SEED = "backprop", 10


def trace(scenario, rule, seed, key):
    d = np.load(EXP / f"804_repeated_alternation_{scenario}.npz", allow_pickle=True)
    for i, m in enumerate(d["methods"]):
        if m == rule and int(d["seeds"][i]) == seed and f"{key}_{i}" in d.files:
            return (np.asarray(d[f"{key}_{i}"], float), np.asarray(d[f"steps_{i}"]),
                    np.asarray(d[f"switches_{i}"]))
    return None, None, None


def pca2(X):
    """Project onto the trajectory's own first two PCs. Fitted per run -- see the docstring."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD: the trajectory has far fewer points than weights, so this is cheap.
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    return U[:, :2] * S[:2], var[:2]


def contraction(P, idx):
    """Distance between the same phase of successive cycles: |p_k - p_{k-2}| over block ends."""
    pts = P[idx]
    return np.array([np.linalg.norm(pts[k] - pts[k - 2]) for k in range(2, len(pts))])


if __name__ == "__main__":
    fig, axes = plt.subplots(len(LAYERS), len(SCENARIOS), figsize=(9.6, 9.4))
    norm = mpl.colors.Normalize(1, 10)
    cmap = mpl.colormaps["viridis"]

    for r, (key, llab) in enumerate(LAYERS):
        for c, s in enumerate(SCENARIOS):
            ax = axes[r][c]
            X, steps, switches = trace(s, SHOW_RULE, SHOW_SEED, key)
            if X is None:
                ax.set_axis_off()
                continue
            P, var = pca2(X)
            bounds = [0] + [int(np.searchsorted(steps, w, side="right")) for w in switches]
            for b in range(len(bounds) - 1):
                lo, hi = bounds[b], min(bounds[b + 1] + 1, len(P))
                ax.plot(P[lo:hi, 0], P[lo:hi, 1], color=cmap(norm(b + 1)), lw=1.2)
            ends = [min(b, len(P) - 1) for b in bounds[1:]]
            ax.scatter(P[ends, 0], P[ends, 1], s=18, facecolor="none", edgecolor="k", lw=0.7,
                       zorder=5)
            d = contraction(P, ends)
            ratio = d[-1] / d[0] if len(d) and d[0] > 0 else np.nan
            ax.set_title(f"{NICE[s]} · {llab}\n"
                         f"cycle-to-cycle distance {d[0]:.2f} → {d[-1]:.2f}  "
                         f"(×{ratio:.2f})", fontsize=9)
            ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}% var)", fontsize=9)
            ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}% var)", fontsize=9)
            ax.grid(alpha=0.2)
            print(f"  {NICE[s]:10s} {llab:16s} var {100 * var[0]:4.1f}/{100 * var[1]:4.1f}%   "
                  f"cycle distance {d[0]:.3f} -> {d[-1]:.3f}  ratio {ratio:.2f}")

    # Two-line titles need room, or the top row's x-label lands on the bottom row's title.
    fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.90, right=0.88)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, label="alternation block", pad=0.02)
    fig.suptitle(f"Weight-space trajectory, {SHOW_RULE}, seed {SHOW_SEED} — a closed loop would "
                 f"keep the cycle-to-cycle distance flat", fontsize=9)
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
