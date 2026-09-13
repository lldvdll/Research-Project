"""Does the weight oscillation under alternation actually die away, or does it move into a lower principal component?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Diagnostic for R4, alongside 918.

WHY THIS EXISTS SEPARATELY FROM 918. 918 draws the trajectory in the PC1-PC2 plane, and a
phase plane can only show what the first two components do. If the ringing simply MIGRATES --
PC1 and PC2 settling while PC3, PC4 or PC5 keep swinging at a tenth the amplitude -- 918 shows
a contracting loop and the report concludes the system converged, when what actually happened
is that the same oscillation moved somewhere the figure could not see. This plots the
components against training updates instead, one line per PC, so that failure mode is visible.
918 is untouched; the two answer different questions and disagreeing with each other would
itself be the finding.

EACH PANEL HAS ITS OWN Y-SCALE, AND THAT IS THE WHOLE POINT. PC1 carries ~70% of the variance
and PC5 perhaps 1-2%, so on one shared axis PC5 is a flat line whatever it is doing. Per-panel
scaling makes low-amplitude ringing legible; the variance annotation on every panel is what
stops that being misread as "PC5 matters as much as PC1". Read the shape from the panel and the
magnitude from the label, never the reverse.

LAYOUT. Rows are the four scenario/rule conditions, columns are PC1 to PC5, so reading ACROSS a
row answers the question the figure was made for: as the early components settle, does anything
further along keep moving?

THE SCALAR THAT STOPS IT BEING A PRETTY PICTURE. For every PC the peak-to-peak range is
measured inside each alternation block, and the last cycle is compared with the first. A ratio
near 1 means that component is still swinging as hard as it was at the start; a small ratio
means it genuinely damped. Printed per panel, so "the oscillation moved to PC4" is a number and
not an impression.

BACKGROUND SHADING marks which task is training, alternating block by block, taken from 804's
saved switch points. Neutral greys on purpose -- the rule colours are spent on the data.

COLOURS follow the project standard: backprop BLACK, pc RED. src/style.py is deliberately NOT
used -- retired as the source of truth on 2026-09-13.

PROVENANCE
    804_repeated_alternation_{scenario}.npz   weight traces for all 10 seeds of each rule
    (TRACE_SEEDS = 10), every 10th eval, W1 (6272 = 196x32) and W2 (160 or 320) both saved.
    One figure per layer; seed fixed at SHOW_SEED, matching 918 so the two can be read together.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
RULE_COLOR = {"backprop": "black", "pc": "tab:red"}

CONDITIONS = [("class_il", "backprop"), ("domain_il", "backprop"),
              ("class_il", "pc"), ("domain_il", "pc")]
LAYERS = [("trace_W1", "w1", "$W_1$ (hidden)"), ("trace_W2", "w2", "$W_2$ (output)")]
N_PCS = 5
SHOW_SEED = 10


def trace(scenario, rule, seed, key):
    d = np.load(EXP / f"804_repeated_alternation_{scenario}.npz", allow_pickle=True)
    for i, m in enumerate(d["methods"]):
        if m == rule and int(d["seeds"][i]) == seed and f"{key}_{i}" in d.files:
            return (np.asarray(d[f"{key}_{i}"], float), np.asarray(d[f"steps_{i}"]),
                    np.asarray(d[f"switches_{i}"]))
    return None, None, None


def pca_scores(X, n):
    """Project the trajectory onto its own first n PCs. Same per-run fit 918 uses."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    return U[:, :n] * S[:n], var[:n]


def block_ranges(score, steps, switches):
    """Peak-to-peak of one component inside each alternation block."""
    edges = [0] + [int(np.searchsorted(steps, w, side="right")) for w in switches]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = score[a:max(b, a + 2)]
        if len(seg) > 1:
            out.append(float(seg.max() - seg.min()))
    return out


def draw_layer(key, tag, llab):
    fig, axes = plt.subplots(len(CONDITIONS), N_PCS,
                             figsize=(2.0 * N_PCS, 1.35 * len(CONDITIONS)), sharex="col")
    for r, (scenario, rule) in enumerate(CONDITIONS):
        X, steps, switches = trace(scenario, rule, SHOW_SEED, key)
        if X is None:
            for c in range(N_PCS):
                axes[r][c].set_axis_off()
            continue
        P, var = pca_scores(X, N_PCS)
        edges = [steps[0]] + list(switches)
        for c in range(N_PCS):
            ax = axes[r][c]
            # which task is training, block by block
            for b in range(len(edges) - 1):
                if b % 2 == 1:
                    ax.axvspan(edges[b], edges[b + 1], color="0.5", alpha=0.10, lw=0, zorder=0)
            ax.plot(steps, P[:, c], color=RULE_COLOR[rule], lw=0.9, zorder=3)
            ax.axhline(0, color="0.8", lw=0.6, zorder=1)

            rng = block_ranges(P[:, c], steps, switches)
            ratio = (rng[-1] / rng[0]) if len(rng) > 1 and rng[0] > 0 else float("nan")
            ax.annotate(f"PC{c+1} · {100*var[c]:.0f}%", (0.03, 0.97),
                        xycoords="axes fraction", ha="left", va="top",
                        fontsize=6.5, fontweight="bold")
            ax.annotate(f"x{ratio:.2f}", (0.97, 0.04), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=6, color="0.35")
            ax.tick_params(labelsize=6)
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{rule}\n{NICE[scenario]}", fontsize=7)
            if r == len(CONDITIONS) - 1:
                ax.set_xlabel("training updates", fontsize=7)
            print(f"  {llab:16s} {NICE[scenario]:10s} {rule:9s} PC{c+1}  "
                  f"var {100*var[c]:5.1f}%  block range {rng[0]:.3f} -> {rng[-1]:.3f}"
                  f"  ratio {ratio:.2f}")

    fig.tight_layout(pad=0.4)
    out = figure_path(__file__, suffix=tag)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    for key, tag, llab in LAYERS:
        draw_layer(key, tag, llab)
