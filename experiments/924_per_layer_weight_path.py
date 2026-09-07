"""What do the weights themselves do, layer by layer, under repeated alternation?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R5.

ONE PLOT for now, one grid cell, form from 007's SK_PATH: per-layer weight movement per
alternation block, both scenarios. The companion card (SK_DRIFT, hidden-code drift) needs data
that does NOT exist in any saved array -- see the note at the bottom.

THE CLAIM IT TESTS, STATED IN WEIGHTS RATHER THAN ACCURACY. 007's card predicts that W1 converges
while W2 keeps oscillating in Class-IL -- the output-competition claim expressed in parameters.

⚠ THE RESULT SUPPORTS THE SCENARIO SPLIT BUT NOT THE CARD'S EXACT WORDING, and the caption should
say the second thing. Per-update weight path, relative to each layer's own first block:

    Class-IL   W1  1.00 → 1.34        W2  1.00 → 1.99      both ACCELERATE, W2 far more
    Domain-IL  W1  1.00 → 0.93        W2  1.00 → 1.11      both FLAT

Neither Class-IL layer converges: both move MORE per update as blocks go by, and the output layer
roughly doubles its rate while the trunk rises by a third. Domain-IL does neither -- its rate is
steady in both layers. So the asymmetry the card predicts is real and it is between SCENARIOS
rather than between layers within Class-IL: the readout is where Class-IL's acceleration
concentrates, which is the output-competition claim, but the trunk accelerates too and is not
"converged".

This is the same run 917 and 918 draw, analysed a third way -- one experiment, three figures.

⚠ NORMALISED PER LAYER, because W1 has 6272 parameters and W2 has 320. Raw norms would make W1
larger by construction and say nothing. Each layer's per-block path is divided by its own FIRST
block, so every series starts at 1.0 and the question becomes "how much of its initial movement is
this layer still doing".

⚠ BLOCK LENGTH IS A DEPENDENT VARIABLE under matched competence -- blocks shorten from ~400 to ~50
updates in Class-IL. A layer could move less per block simply because the block is shorter. Both
the raw per-block path AND the path per update are therefore printed, and the figure draws the
PER-UPDATE version, which is the one that is not confounded by block length.

PROVENANCE
    804_repeated_alternation_{scenario}.npz   weight traces for seeds 10-12 of each rule
    (TRACE_SEEDS = 3), W1 and W2 both stored, config_800.yaml, matched competence per block.

⚠ THE COMPANION FIGURE CANNOT BE BUILT FROM EXISTING DATA. 007's SK_DRIFT card asks for hidden-code
displacement during task 2, resolved by whether a unit is task-1 or task-2 selective, against a
frozen-weight baseline. 803 stores the linear probe's ACCURACY at each checkpoint, not the hidden
codes themselves, and no other array holds them. `src/probes.py` already provides
`code_snapshot`/`code_drift`, so this is instrumentation of an existing protocol rather than a new
experiment -- it needs an 800-series run (807) that logs codes on a fixed task-1 batch through
task 2. That is the largest genuine gap in the project and it is cheap to close.
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
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
LAYERS = [("trace_W1", "$W_1$ (hidden)", "-", "o"), ("trace_W2", "$W_2$ (output)", "--", "s")]
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
RULE = "backprop"


def per_block_path(scenario, key):
    """Total distance travelled in each block, and per update, averaged over traced seeds."""
    d = np.load(EXP / f"804_repeated_alternation_{scenario}.npz", allow_pickle=True)
    raw, per_upd = [], []
    for i, m in enumerate(d["methods"]):
        if m != RULE or f"{key}_{i}" not in d.files:
            continue
        X = np.asarray(d[f"{key}_{i}"], float)
        steps, sw = np.asarray(d[f"steps_{i}"]), np.asarray(d[f"switches_{i}"])
        bounds = [0] + [int(np.searchsorted(steps, w, side="right")) for w in sw]
        step_norm = np.linalg.norm(np.diff(X, axis=0), axis=1)
        blocks, lens = [], []
        for b in range(len(bounds) - 1):
            lo, hi = bounds[b], min(bounds[b + 1], len(step_norm))
            blocks.append(step_norm[lo:hi].sum())
            lens.append(max(1, (sw[b] - (sw[b - 1] if b else 0))))
        raw.append(blocks)
        per_upd.append(np.array(blocks) / np.array(lens))
    return np.array(raw), np.array(per_upd)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for s in SCENARIOS:
        for key, lab, ls, mk in LAYERS:
            raw, per_upd = per_block_path(s, key)
            mu = per_upd.mean(axis=0)
            mu = mu / mu[0]                       # each layer against its own first block
            se = (per_upd / per_upd[:, :1]).std(axis=0, ddof=1) / np.sqrt(per_upd.shape[0])
            x = np.arange(1, len(mu) + 1)
            ax.errorbar(x, mu, yerr=se, ls=ls, marker=mk, ms=5, lw=1.8, capsize=3,
                        color=COLORS[s], label=f"{NICE[s]} · {lab}")
            print(f"  {NICE[s]:10s} {lab:16s} per-update path / block 1: "
                  + "  ".join(f"{v:.2f}" for v in mu))
            print(f"  {NICE[s]:10s} {lab:16s} raw per-block path:        "
                  + "  ".join(f"{v:.2f}" for v in raw.mean(axis=0)))
    ax.axhline(1.0, color="k", lw=1.0, ls=":")
    ax.set_xticks(np.arange(1, 11))
    ax.set_xlabel("alternation block", fontsize=9)
    ax.set_ylabel("weight path per update, relative to block 1", fontsize=9)
    ax.set_title(f"{RULE}, traced seeds — a converged layer keeps falling, "
                 f"an oscillating one stays flat", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
