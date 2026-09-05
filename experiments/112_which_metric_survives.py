"""Near our chosen stopping point, does every candidate metric agree on whether PC beats
backprop -- or does the sign flip depending which one you read?

Report series 2, topic 110, the decisive script: this is what fixes the metric control and
confirms (or kills) 90% as the operating point. 110 (budget) and 111 (threshold) already showed
that a single rule's raw metric VALUE moves with the stopping point -- expected, and not the
actual question. The actual claim under test (`findings.md` Sec.2) is that the PAIRED PC-minus-
backprop DIFFERENCE flips sign with the stopping point under endpoint metrics. This checks that,
for five candidate metrics, across a small neighbourhood of stopping values.

Backprop + PC, both scenarios, all five of 111's thresholds {75, 80, 85, 90, 95%}. 95% was broken
in 111 at MAX_ITERS=3000 (every seed hit the cap); raised to 10000 here to give it a real chance.
10 seeds.

Five metrics, all rescaled to a 0-100 accuracy-style axis (see 113 for why crossover_of_peak and
area_retained are added to the standard three): crossover, endpoint (final_t1), S&B mean error,
area_retained, crossover_of_peak (100 * crossover / peak_t1).

One report figure: 5 stacked panels (one per metric), x = threshold, y = PC minus backprop
(paired, per seed), error bars = SEM, zero-reference line, both scenarios overlaid. A metric
PASSES if its error bars never cross zero across all three points.
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
from src.metrics import metric_grid, paired_diff

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
# All 5 of 111's thresholds, not just the 80/85/90 neighbourhood: MAX_ITERS raised 3000 -> 10000
# so 95% -- broken in 111 at the old cap, 10/10 seeds never reached it -- gets a real chance.
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
SEEDS = 10
EVAL_EVERY = 10
MAX_ITERS = 10000
STOP_PATIENCE = 3
METHODS = ["backprop", "pc"]
LR = {"backprop": 0.01, "pc": 0.02}
SCENARIOS = ["class_il", "domain_il"]
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
METRICS = ["crossover", "final_t1", "sb_mean_err", "area_retained", "crossover_of_peak"]
LABELS = {"crossover": "crossover", "final_t1": "endpoint", "sb_mean_err": "S&B error",
          "area_retained": "area retained", "crossover_of_peak": "crossover / peak"}
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"

if SMOKE:
    THRESHOLDS, SEEDS, EVAL_EVERY, MAX_ITERS = [0.5, 0.6], 2, 5, 200
    print("--smoke: tiny budget, results are NOT meaningful\n")

if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32")


def cell_metrics(proto, method, thr, data):
    """Per-seed metric arrays for one (scenario, method, threshold) cell."""
    p = replace(proto, stop_threshold=[thr, thr])
    runs = [run(p, method, seed, data=data) for seed in range(SEEDS)]
    n_bad = sum(1 for o in runs if not all(o["reached"]))
    if n_bad:
        print(f"  WARNING: {proto.scenario} {method} thr={thr:.0%}: "
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
    return {
        "crossover": g["crossover"],
        "final_t1": g["final_t1"],
        "sb_mean_err": (g["mean_err_t1"] + g["mean_err_t2"]) / 2,
        "area_retained": g["area_retained"],
        "crossover_of_peak": 100 * g["crossover"] / g["peak_t1"],
    }


if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__), allow_pickle=True)
    diff = {k: {s: z[f"{k}_{s}_mean"] for s in SCENARIOS} for k in METRICS}
    diff_sem = {k: {s: z[f"{k}_{s}_sem"] for s in SCENARIOS} for k in METRICS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    base = replace(PROTOCOL, hidden=H, eval_every=EVAL_EVERY, seeds=SEEDS,
                   max_iters_per_task=MAX_ITERS, stop_patience=STOP_PATIENCE, lr=LR)
    data = load(replace(base, scenario="class_il"))
    diff = {k: {s: np.full(len(THRESHOLDS), np.nan) for s in SCENARIOS} for k in METRICS}
    diff_sem = {k: {s: np.full(len(THRESHOLDS), np.nan) for s in SCENARIOS} for k in METRICS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        proto = replace(base, scenario=scen)
        for ti, thr in enumerate(THRESHOLDS):
            m_bp = cell_metrics(proto, "backprop", thr, data)
            m_pc = cell_metrics(proto, "pc", thr, data)
            for k in METRICS:
                mean, sem_, _ = paired_diff(m_pc[k], m_bp[k])
                diff[k][scen][ti] = mean
                diff_sem[k][scen][ti] = sem_
            print(f"  {scen:10s} thr={thr:.0%}  " +
                  "  ".join(f"{k} {diff[k][scen][ti]:+.1f}" for k in METRICS) +
                  f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- report figure: 5 panels, grouped bars, PC - backprop
XTICKS = [75, 80, 85, 90, 95]      # matches 113's axis range/ticks, even though only 3 have bars
xs = np.asarray([t * 100 for t in THRESHOLDS])
width = 1.6
fig, axes = plt.subplots(len(METRICS), 1, figsize=(6.5, 11), sharex=True)
for ax, k in zip(axes, METRICS):
    for i, s in enumerate(SCENARIOS):
        offset = (i - 0.5) * width
        ax.bar(xs + offset, diff[k][s], width=width, yerr=diff_sem[k][s], capsize=3,
               color=COLORS[s], label=s.replace("_", "-"))
    ax.axhline(0, color="k", lw=1)
    ax.set_xlim(XTICKS[0] - 3, XTICKS[-1] + 3)
    ax.set_xticks(XTICKS)
    ax.set_ylabel(f"{LABELS[k]}\n(pc - bp)", fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
axes[0].legend(fontsize=8, loc="best")
axes[-1].set_xlabel("stop threshold (%)")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), thresholds=np.asarray(THRESHOLDS), H=H,
         **{f"{k}_{s}_mean": diff[k][s] for k in METRICS for s in SCENARIOS},
         **{f"{k}_{s}_sem": diff_sem[k][s] for k in METRICS for s in SCENARIOS})
print(f"saved {array_path(__file__)}")
