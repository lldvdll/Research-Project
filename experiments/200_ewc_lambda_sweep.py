"""How large should EWC's penalty strength (lambda) be, and does the answer differ between
backprop and PC?

Report series 2, decade 200: the first of three "does an added consolidation/gating mechanism
change forgetting" sweeps (200 EWC, 210 SI, 220 k-WTA), followed by 230's head-to-head
comparison at each mechanism's swept-optimal setting. backprop_ewc + pc_ewc, both scenarios,
matched-competence 90%, 10 seeds. Every OTHER control is held at the value the rest of this
report series uses -- H=32 (script 100), lr 0.01 backprop / 0.02 PC, 90% matched competence,
crossover as the metric -- so lambda is this script's only free variable.

The penalty switches on at the task1->task2 boundary via run_classil's own on_task_end hook,
the same mechanism 130/131 already use for freezing and NCM prototypes: task 1 trains
completely normally (lambda has no effect yet), Fisher information is estimated from a fixed
sample of task-1 training images the moment task 1 ends, and the penalty (weighted by the
swept lambda) applies through task 2.

lambda=0 is included deliberately. It is not just "no penalty" conceptually -- the whole EWC
bookkeeping (Fisher estimation, anchor snapshot) still runs, weighted by zero -- so this cell
doubles as a live regression check against plain backprop/pc, on real data, not just the
synthetic check already done in scratch.

One report figure: crossover vs lambda (log-x), both rules, both scenarios, error bars = SEM
over 10 seeds.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, build, replace, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import run_classil
from src.ewc import fisher_information, set_anchor
from src.probes import prototype_images
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
LAMBDAS = [0, 0.01, 0.1, 1, 10, 100]
METHODS = ["backprop_ewc", "pc_ewc"]
LR = {"backprop_ewc": 0.01, "pc_ewc": 0.02}   # same values as backprop/pc elsewhere in this series
SCENARIOS = ["class_il", "domain_il"]
SEEDS = 10
EVAL_EVERY = 10
MAX_ITERS = 5000
STOP_PATIENCE = 3
THRESHOLD = 0.90
FISHER_PER_CLASS = 100
METHOD_COLORS = {"backprop_ewc": "0.4", "pc_ewc": "tab:orange"}
SCENARIO_LS = {"class_il": "-", "domain_il": "--"}
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"

if SMOKE:
    LAMBDAS, SEEDS, EVAL_EVERY, MAX_ITERS, FISHER_PER_CLASS = [0, 1], 2, 5, 200, 20
    print("--smoke: tiny budget, results are NOT meaningful\n")

if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32")


def run_ewc(proto, method, seed, lam, data):
    """Task 1 trains normally; Fisher is estimated from a fixed task-1 sample the moment task 1
    ends; the penalty (weighted by lam) applies through task 2."""
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    handle = {}
    train_step, predict = build(proto, method, seed, handle=handle)

    def on_task_end(ti, step):
        if ti == 0:
            xs, ys = prototype_images(data.train, data.class_idx, tasks[0],
                                      per_class=FISHER_PER_CLASS, device=proto.device, seed=seed)
            if lmap is not None:
                ys = torch.tensor([lmap[int(v)] for v in ys], device=proto.device)
            F = fisher_information(handle["params"], handle["arch"], handle["obj"], xs, ys,
                                   device=proto.device)
            set_anchor(handle["ewc"], handle["params"], F, lam)

    return run_classil(train_step, predict, tasks, data.train, data.class_idx,
                       report_eval=data.report_eval, stop_eval=data.stop_eval,
                       max_iters_per_task=MAX_ITERS, batch=proto.batch, eval_every=proto.eval_every,
                       device=proto.device, stop_threshold=[THRESHOLD, THRESHOLD],
                       stop_patience=proto.stop_patience, data_seed=seed, label_map=lmap,
                       on_task_end=on_task_end)


def cell_metrics(proto, method, lam, data):
    runs = [run_ewc(proto, method, seed, lam, data) for seed in range(SEEDS)]
    n_bad = sum(1 for o in runs if not all(o["reached"]))
    if n_bad:
        print(f"  WARNING: {proto.scenario} {method} lambda={lam}: {n_bad}/{SEEDS} seed(s) hit the cap")
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
    crossover = {s: {m: np.full((len(LAMBDAS), SEEDS), np.nan) for m in METHODS} for s in SCENARIOS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        proto = replace(base, scenario=scen)
        for method in METHODS:
            for li, lam in enumerate(LAMBDAS):
                cx = cell_metrics(proto, method, lam, data)
                crossover[scen][method][li] = cx
                print(f"  {scen:10s} {method:12s} lambda={lam:<6g} crossover {np.nanmean(cx):5.1f}"
                      f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- report figure: crossover vs lambda
fig, ax = plt.subplots(figsize=(6.5, 4.5))
xpos = np.arange(len(LAMBDAS))   # lambda=0 breaks a log axis; equally-spaced ticks, labelled
for scen in SCENARIOS:
    for method in METHODS:
        v = crossover[scen][method]
        mean = np.nanmean(v, axis=1)
        n = np.sum(np.isfinite(v), axis=1)
        sem = np.where(n > 1, np.nanstd(v, axis=1, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
        ax.errorbar(xpos, mean, yerr=sem, marker="o", capsize=3, lw=1.8,
                   color=METHOD_COLORS[method], ls=SCENARIO_LS[scen],
                   label=f"{method.replace('_ewc', '')}, {scen.replace('_', '-')}")
ax.set_xticks(xpos)
ax.set_xticklabels([str(l) for l in LAMBDAS])
ax.set_xlabel("lambda")
ax.set_ylabel("crossover (%)")
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), lambdas=np.asarray(LAMBDAS), H=H,
         **{f"crossover_{s}_{m}": crossover[s][m] for s in SCENARIOS for m in METHODS})
print(f"saved {array_path(__file__)}")
