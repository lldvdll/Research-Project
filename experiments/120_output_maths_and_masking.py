"""Does Class-IL's output-layer suppression depend on which output maths we use, or does masking
fix it regardless of the specification?

Report series 2, decade 120 (new topic: output architecture / masking). Backprop only -- PC's
loss IS squared error by construction, so varying the loss stops testing PC rather than testing
PC under a different loss. Class-IL only: Domain-IL's output layer structurally cannot suppress
(every unit is a positive target in both tasks), so masking has no question to answer there.

mse+onehot, unmasked, is the loss/target coding the rest of this report series uses everywhere
else -- chosen (knowledge_base.md Sec.7.1) because it is the ONLY loss all three learning rules
can take unaltered, i.e. for EqProp's sake, not because it is neutral to the suppression
question. This asks what the two specs actually built for classification would show instead:
ce+softmax (the textbook loss) and hinge+-1 (EqProp's own -- knowledge_base.md flags it as the
harshest suppression of the three).

Factorial: 3 output specs x {unmasked, masked}, backprop, Class-IL, matched-competence 90%,
10 seeds.
  mse   / onehot   -- the series' standard elsewhere
  ce    / onehot   -- softmax cross-entropy
  hinge / pm1      -- EqProp's original spec

One report figure: crossover (the metric locked in by 112), grouped bars, x = spec, paired bars
= unmasked / masked.

A second set of figures (task1/task2 vs time, one per cell) is saved purely to sanity-check the
runs. NOT for the report.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, run, replace, figure_path as _figure_path, \
    array_path as _array_path
from src.metrics import metric_grid

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
SPECS = [("mse", "onehot"), ("ce", "onehot"), ("hinge", "pm1")]
MASKS = [False, True]
SEEDS = 10
EVAL_EVERY = 10
MAX_ITERS = 5000        # generous cap; a run hitting it is reported (see cell_metrics), not hidden
STOP_PATIENCE = 3
THRESHOLD = 0.90
COLORS = {False: "tab:red", True: "tab:blue"}
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"

if SMOKE:
    SEEDS, EVAL_EVERY, MAX_ITERS = 2, 5, 200
    print("--smoke: tiny budget, results are NOT meaningful\n")

if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32")


def cell_metrics(proto, data):
    """Per-seed crossover array for one (spec, mask) cell, backprop, Class-IL, 90% matched."""
    p = replace(proto, stop_threshold=[THRESHOLD, THRESHOLD])
    runs = [run(p, "backprop", seed, data=data) for seed in range(SEEDS)]
    n_bad = sum(1 for o in runs if not all(o["reached"]))
    if n_bad:
        print(f"  WARNING: {proto.loss}/{proto.target} mask={proto.mask}: "
              f"{n_bad}/{SEEDS} seed(s) hit the {MAX_ITERS}-update cap")
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
    g = metric_grid(rel_grid, A, 0.0)
    return g["crossover"], rel_grid, A


if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__), allow_pickle=True)
    crossover = {(loss, tgt): {m: z[f"crossover_{loss}_{tgt}_{m}"] for m in MASKS}
                 for loss, tgt in SPECS}
    diag = {(loss, tgt): {m: (z[f"steps_{loss}_{tgt}_{m}"], z[f"curve_{loss}_{tgt}_{m}"])
                          for m in MASKS} for loss, tgt in SPECS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    base = replace(PROTOCOL, hidden=H, scenario="class_il", eval_every=EVAL_EVERY, seeds=SEEDS,
                   max_iters_per_task=MAX_ITERS, stop_patience=STOP_PATIENCE,
                   lr={"backprop": 0.01})
    data = load(base)
    crossover = {(loss, tgt): {} for loss, tgt in SPECS}
    diag = {(loss, tgt): {} for loss, tgt in SPECS}
    t0 = time.perf_counter()

    for loss, tgt in SPECS:
        for m in MASKS:
            proto = replace(base, loss=loss, target=tgt, mask=m)
            cx, steps, A = cell_metrics(proto, data)
            crossover[(loss, tgt)][m] = cx
            diag[(loss, tgt)][m] = (steps, A)
            print(f"  {loss:6s}/{tgt:6s} mask={str(m):5s} crossover {np.nanmean(cx):5.1f}"
                  f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- report figure: crossover, grouped bars
labels = [f"{loss}/{tgt}" for loss, tgt in SPECS]
xpos = np.arange(len(SPECS))
width = 0.35
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for m in MASKS:
    means = [np.nanmean(crossover[spec][m]) for spec in SPECS]
    n = [np.sum(np.isfinite(crossover[spec][m])) for spec in SPECS]
    sems = [np.nanstd(crossover[spec][m], ddof=1) / np.sqrt(max(ni, 1)) if ni > 1 else np.nan
            for spec, ni in zip(SPECS, n)]
    offset = (-0.5 if not m else 0.5) * width
    ax.bar(xpos + offset, means, width=width, yerr=sems, capsize=3,
           color=COLORS[m], label="masked" if m else "unmasked")
ax.set_xticks(xpos)
ax.set_xticklabels(labels)
ax.set_ylabel("crossover (%)")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ------------------------------------------------------------------ diagnostic (NOT for the report)
for loss, tgt in SPECS:
    fig2, axes2 = plt.subplots(len(MASKS), 1, figsize=(6, 4.4), sharex=False)
    for ax2, m in zip(axes2, MASKS):
        steps, A = diag[(loss, tgt)][m]
        for r in range(A.shape[0]):
            ax2.plot(steps, A[r, :, 0] * 100, color="tab:orange", lw=0.6, alpha=0.25)
            ax2.plot(steps, A[r, :, 1] * 100, color="tab:blue", lw=0.6, alpha=0.25)
        ax2.plot(steps, np.nanmean(A[:, :, 0], axis=0) * 100, color="tab:orange", lw=2.0, label="task 1")
        ax2.plot(steps, np.nanmean(A[:, :, 1], axis=0) * 100, color="tab:blue", lw=2.0, label="task 2")
        ax2.axvline(0, color="k", ls="--", lw=1)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("masked" if m else "unmasked", fontsize=8)
        ax2.grid(alpha=0.2)
    axes2[0].legend(fontsize=7, loc="lower left")
    axes2[-1].set_xlabel("training step, relative to the task switch")
    fig2.tight_layout()
    fig2.savefig(figure_path(__file__, f"diagnostic_{loss}_{tgt}"), dpi=110, bbox_inches="tight")
    print(f"saved {figure_path(__file__, f'diagnostic_{loss}_{tgt}')}  (analysis only, not for the report)")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), H=H,
         **{f"crossover_{loss}_{tgt}_{m}": crossover[(loss, tgt)][m]
            for loss, tgt in SPECS for m in MASKS},
         **{f"steps_{loss}_{tgt}_{m}": diag[(loss, tgt)][m][0]
            for loss, tgt in SPECS for m in MASKS},
         **{f"curve_{loss}_{tgt}_{m}": diag[(loss, tgt)][m][1]
            for loss, tgt in SPECS for m in MASKS})
print(f"saved {array_path(__file__)}")
