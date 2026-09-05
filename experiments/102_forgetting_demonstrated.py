"""What does catastrophic forgetting actually look like, for backprop and PC, in both scenarios?

Report series 2, topic 100 (context/setup), third script -- contextualises the 110-topic metrics
work: once we later say "backprop crossover minus PC crossover", the reader has already seen the
shape this is describing. Purely qualitative -- no metric is computed or annotated here.

Both tasks trained to 90% (matched competence, the value the 110-topic work adopts), backprop and
PC, both scenarios, 10 seeds. One 2x2 grid: rows = scenario, columns = rule. Each panel: task 1
(orange) / task 2 (blue) accuracy against training step relative to the switch, thin per-seed
lines behind a bold mean, shaded by which task is training.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, run, replace, figure_path as _figure_path, \
    array_path as _array_path

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):
    return _array_path(f, _tag(f, suffix))


# ---------------------------------------------------------------- settings
THRESHOLD = 0.90
SEEDS = 10
EVAL_EVERY = 10
MAX_ITERS = 3000
STOP_PATIENCE = 3
METHODS = ["backprop", "pc"]
LR = {"backprop": 0.01, "pc": 0.02}
SCENARIOS = ["domain_il", "class_il"]      # Domain-IL first, primary scenario
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"

if SMOKE:
    THRESHOLD, SEEDS, EVAL_EVERY, MAX_ITERS = 0.5, 2, 5, 200
    print("--smoke: tiny budget, results are NOT meaningful\n")

if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32")

if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__), allow_pickle=True)
    panels = {(s, m): (z[f"steps_{s}_{m}"], z[f"curve_{s}_{m}"]) for s in SCENARIOS for m in METHODS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    base = replace(PROTOCOL, hidden=H, eval_every=EVAL_EVERY, seeds=SEEDS,
                   max_iters_per_task=MAX_ITERS, stop_patience=STOP_PATIENCE, lr=LR)
    data = load(replace(base, scenario="class_il"))
    panels = {}
    for scen in SCENARIOS:
        proto = replace(base, scenario=scen, stop_threshold=[THRESHOLD, THRESHOLD])
        for m in METHODS:
            runs = [run(proto, m, seed, data=data) for seed in range(SEEDS)]
            n_bad = sum(1 for o in runs if not all(o["reached"]))
            if n_bad:
                print(f"  WARNING: {scen} {m}: {n_bad}/{SEEDS} seed(s) hit the cap")

            lo = -int(1.2 * np.median([o["switches"][0] for o in runs]))
            hi = int(1.2 * np.median([o["switches"][1] - o["switches"][0] for o in runs]))
            rel_grid = np.arange(lo, hi + EVAL_EVERY, EVAL_EVERY)
            A = np.full((SEEDS, len(rel_grid), 2), np.nan)
            for i, o in enumerate(runs):
                rel = np.asarray(o["steps"]) - o["switches"][0]
                for j, r in enumerate(rel):
                    k_ = int(round((r - lo) / EVAL_EVERY))
                    if 0 <= k_ < len(rel_grid):
                        A[i, k_] = o["curves"]["argmax"][j]
            panels[(scen, m)] = (rel_grid, A)
            print(f"  {scen:10s} {m:8s} done")

def common_window(steps, A):
    """Crop to the range where EVERY seed has finite data. Averaging ragged per-seed curves
    over a window some seeds only partly cover makes the mean jump each time a seed enters or
    leaves -- not a training artefact, a plotting one. This removes it by construction."""
    first, last = [], []
    for r in range(A.shape[0]):
        idx = np.where(np.isfinite(A[r, :, 0]) & np.isfinite(A[r, :, 1]))[0]
        if idx.size:
            first.append(idx[0]); last.append(idx[-1])
    lo_idx, hi_idx = max(first), min(last)
    return steps[lo_idx:hi_idx + 1], A[:, lo_idx:hi_idx + 1, :]


# ---------------------------------------------------------------- 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=False, sharey=True)
for row, scen in enumerate(SCENARIOS):
    for col, m in enumerate(METHODS):
        ax = axes[row, col]
        steps, A = common_window(*panels[(scen, m)])
        fs = steps[np.isfinite(A).any(axis=(0, 2))]
        if fs.size:
            ax.axvspan(fs.min(), 0, color="tab:orange", alpha=0.08, lw=0)
            ax.axvspan(0, fs.max(), color="tab:blue", alpha=0.08, lw=0)
        for r in range(A.shape[0]):
            ax.plot(steps, A[r, :, 0] * 100, color="tab:orange", lw=0.6, alpha=0.25)
            ax.plot(steps, A[r, :, 1] * 100, color="tab:blue", lw=0.6, alpha=0.25)
        ax.plot(steps, np.nanmean(A[:, :, 0], axis=0) * 100, color="tab:orange", lw=2.2, label="task 1")
        ax.plot(steps, np.nanmean(A[:, :, 1], axis=0) * 100, color="tab:blue", lw=2.2, label="task 2")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        if row == 0:
            ax.set_xlabel(m, fontsize=10)
            ax.xaxis.set_label_position("top")
        if row == 1:
            ax.set_xlabel("step from switch")
        if col == 0:
            ax.set_ylabel(f"{scen.replace('_', '-')}\naccuracy (%)", fontsize=9)
axes[0, 0].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), H=H, threshold=THRESHOLD,
         **{f"steps_{s}_{m}": panels[(s, m)][0] for s in SCENARIOS for m in METHODS},
         **{f"curve_{s}_{m}": panels[(s, m)][1] for s in SCENARIOS for m in METHODS})
print(f"saved {array_path(__file__)}")
