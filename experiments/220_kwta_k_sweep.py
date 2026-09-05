"""How large should k be for k-WTA lateral competition, and does the answer differ between
backprop and PC?

Report series 2, decade 200 (third of three: 200 EWC, 210 SI, 220 k-WTA, then 230's head-to-head
comparison). backprop_kwta + pc_kwta, both scenarios, matched-competence 90%, 10 seeds. Same
fixed controls as 200/210 -- H=32, lr 0.01 backprop / 0.02 PC, 90% matched competence, crossover
as the metric.

Simpler than 200/210: k is a construction-time parameter of the method itself (which units
transmit is fixed for the whole run, not switched on at a task boundary), so this uses
protocol.run() directly with k passed through **overrides -- no on_task_end hook needed.

k/H = 100% is k-WTA switched off entirely (every unit transmits) and is included for the same
reason lambda=0 is in 200/210: real-data regression check against plain backprop/pc, alongside
being the natural top of the sweep.

One report figure: crossover vs k/H (categorical, grouped bars), both rules, both scenarios.
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
K_FRACS = [1.00, 0.75, 0.50, 0.25, 0.10]
METHODS = ["backprop_kwta", "pc_kwta"]
LR = {"backprop_kwta": 0.01, "pc_kwta": 0.02}   # same values as backprop/pc elsewhere in this series
SCENARIOS = ["class_il", "domain_il"]
SEEDS = 10
EVAL_EVERY = 10
MAX_ITERS = 5000
STOP_PATIENCE = 3
THRESHOLD = 0.90
METHOD_COLORS = {"backprop_kwta": "0.4", "pc_kwta": "tab:orange"}
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"

if SMOKE:
    K_FRACS, SEEDS, EVAL_EVERY, MAX_ITERS = [1.0, 0.5], 2, 5, 200
    print("--smoke: tiny budget, results are NOT meaningful\n")

if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32")

K_VALUES = [max(1, round(f * H)) for f in K_FRACS]


def cell_metrics(proto, method, k, data):
    p = replace(proto, stop_threshold=[THRESHOLD, THRESHOLD])
    runs = [run(p, method, seed, data=data, k=k) for seed in range(SEEDS)]
    n_bad = sum(1 for o in runs if not all(o["reached"]))
    if n_bad:
        print(f"  WARNING: {proto.scenario} {method} k={k}: {n_bad}/{SEEDS} seed(s) hit the cap")
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
    return g["crossover"]


if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__), allow_pickle=True)
    crossover = {s: {m: z[f"crossover_{s}_{m}"] for m in METHODS} for s in SCENARIOS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    base = replace(PROTOCOL, hidden=H, eval_every=EVAL_EVERY, seeds=SEEDS, lr=LR,
                   max_iters_per_task=MAX_ITERS, stop_patience=STOP_PATIENCE)
    data = load(replace(base, scenario="class_il"))
    crossover = {s: {m: np.full((len(K_VALUES), SEEDS), np.nan) for m in METHODS} for s in SCENARIOS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        proto = replace(base, scenario=scen)
        for method in METHODS:
            for ki, k in enumerate(K_VALUES):
                cx = cell_metrics(proto, method, k, data)
                crossover[scen][method][ki] = cx
                print(f"  {scen:10s} {method:13s} k={k:3d}/{H} crossover {np.nanmean(cx):5.1f}"
                      f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- report figure: crossover vs k/H, grouped bars
xpos = np.arange(len(K_VALUES))
width = 0.35
fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(6.5, 7.0), sharex=True)
for ax, scen in zip(axes, SCENARIOS):
    for i, method in enumerate(METHODS):
        v = crossover[scen][method]
        mean = np.nanmean(v, axis=1)
        n = np.sum(np.isfinite(v), axis=1)
        sem = np.where(n > 1, np.nanstd(v, axis=1, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
        offset = (i - 0.5) * width
        ax.bar(xpos + offset, mean, width=width, yerr=sem, capsize=3,
               color=METHOD_COLORS[method], label=method.replace("_kwta", ""))
    ax.set_xticks(xpos)
    ax.set_ylabel(f"{scen.replace('_', '-')}\ncrossover (%)", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
axes[0].legend(fontsize=8, loc="best")
axes[-1].set_xticklabels([f"{int(f*100)}%" for f in K_FRACS])
axes[-1].set_xlabel("k / H")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), k_values=np.asarray(K_VALUES), k_fracs=np.asarray(K_FRACS), H=H,
         **{f"crossover_{s}_{m}": crossover[s][m] for s in SCENARIOS for m in METHODS})
print(f"saved {array_path(__file__)}")
