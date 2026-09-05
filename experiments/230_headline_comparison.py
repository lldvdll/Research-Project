"""Does consolidation (EWC/SI) or gating (k-WTA) let either rule forget less -- and is any
uplift larger when combined with PC than with backprop?

Report series 2, decade 200, script 4 of 4 (200 EWC sweep, 210 SI sweep, 220 k-WTA sweep, this
the head-to-head). 8 methods -- backprop, PC, and each crossed with EWC / SI / k-WTA at its own
swept-optimal parameter per scenario -- both scenarios, matched-competence 90%, 10 seeds. Same
fixed controls as 200/210/220: H=32, lr 0.01 backprop / 0.02 PC, 90% matched competence.

Optimal lambda/k is read automatically per (method, scenario) from 200/210/220's saved arrays --
whichever grid point gave the highest mean crossover -- the same convention every script in this
series already uses for reading H from exp100. Falls back to a stated default (lambda=1,
k=round(0.5*H)) and prints a warning if a sweep hasn't been run yet, so this script is runnable
before all three sweeps finish, at the cost of not yet using a tuned setting.

One report figure per scenario: the ACC1-vs-ACC2 trajectory (task-2 accuracy on x, task-1 on y,
traced from the task switch onward, mean over 10 seeds), all 8 methods overlaid, with the dotted
y=x diagonal marking where the two are equal -- crossover is exactly where a curve meets that
line. Colour = rule family (backprop / PC), linestyle = mechanism (plain / EWC / SI / k-WTA), so
eight lines stay organised rather than an arbitrary rainbow.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, build, run, replace, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import run_classil
from src.ewc import fisher_information, set_anchor, finalize_si
from src.probes import prototype_images

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
METHODS = ["backprop", "pc", "backprop_ewc", "pc_ewc", "backprop_si", "pc_si",
          "backprop_kwta", "pc_kwta"]
FAMILY = {m: ("pc" if m.startswith("pc") else "backprop") for m in METHODS}
MECH = {m: (m.split("_", 1)[1] if "_" in m else "plain") for m in METHODS}
COLORS = {"backprop": "0.4", "pc": "tab:orange"}
LINESTYLES = {"plain": "-", "ewc": "--", "si": ":", "kwta": "-."}
LR = {"backprop": 0.01, "pc": 0.02, "backprop_ewc": 0.01, "pc_ewc": 0.02,
     "backprop_si": 0.01, "pc_si": 0.02, "backprop_kwta": 0.01, "pc_kwta": 0.02}
SCENARIOS = ["class_il", "domain_il"]
SEEDS = 10
EVAL_EVERY = 10
MAX_ITERS = 5000
STOP_PATIENCE = 3
THRESHOLD = 0.90
FISHER_PER_CLASS = 100
DEFAULT_LAMBDA = 1.0
DEFAULT_K_FRAC = 0.5
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"
SWEEP_200 = Path(__file__).parent / "200_ewc_lambda_sweep.py"
SWEEP_210 = Path(__file__).parent / "210_si_lambda_sweep.py"
SWEEP_220 = Path(__file__).parent / "220_kwta_k_sweep.py"

if SMOKE:
    SEEDS, EVAL_EVERY, MAX_ITERS, FISHER_PER_CLASS = 2, 5, 200, 20
    print("--smoke: tiny budget, results are NOT meaningful\n")

if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32")


def best_param(sweep_script, methods, values_key, default):
    """{(method, scenario): best grid value by mean crossover}, or `default` with a warning if
    the sweep hasn't been run yet."""
    path = Path(_array_path(sweep_script))
    if not path.exists():
        print(f"  WARNING: {sweep_script.name}'s array not found -- using default {default} "
              f"for {methods}, not yet a tuned setting")
        return {(m, s): default for m in methods for s in SCENARIOS}
    z = np.load(path, allow_pickle=True)
    values = z[values_key]
    best = {}
    for m in methods:
        for s in SCENARIOS:
            v = z[f"crossover_{s}_{m}"]
            mean = np.nanmean(v, axis=1)
            best[(m, s)] = values[int(np.nanargmax(mean))]
    return best


ewc_lambda = best_param(SWEEP_200, ["backprop_ewc", "pc_ewc"], "lambdas", DEFAULT_LAMBDA)
si_lambda = best_param(SWEEP_210, ["backprop_si", "pc_si"], "lambdas", DEFAULT_LAMBDA)
kwta_k = best_param(SWEEP_220, ["backprop_kwta", "pc_kwta"], "k_values",
                    max(1, round(DEFAULT_K_FRAC * H)))
print("optimal settings in use:")
for (m, s), v in {**ewc_lambda, **si_lambda, **kwta_k}.items():
    print(f"  {m:14s} {s:10s} -> {v}")


def run_headline(proto, method, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    handle = {}
    train_step, predict = build(proto, method, seed, handle=handle)

    def on_task_end(ti, step):
        if ti != 0:
            return
        if method.endswith("_ewc"):
            xs, ys = prototype_images(data.train, data.class_idx, tasks[0],
                                      per_class=FISHER_PER_CLASS, device=proto.device, seed=seed)
            if lmap is not None:
                ys = torch.tensor([lmap[int(v)] for v in ys], device=proto.device)
            F = fisher_information(handle["params"], handle["arch"], handle["obj"], xs, ys,
                                   device=proto.device)
            set_anchor(handle["ewc"], handle["params"], F, ewc_lambda[(method, proto.scenario)])
        elif method.endswith("_si"):
            finalize_si(handle["ewc"], handle["si_omega"], handle["params"],
                       si_lambda[(method, proto.scenario)])

    return run_classil(train_step, predict, tasks, data.train, data.class_idx,
                       report_eval=data.report_eval, stop_eval=data.stop_eval,
                       max_iters_per_task=MAX_ITERS, batch=proto.batch, eval_every=proto.eval_every,
                       device=proto.device, stop_threshold=[THRESHOLD, THRESHOLD],
                       stop_patience=proto.stop_patience, data_seed=seed, label_map=lmap,
                       on_task_end=on_task_end)


def cell_trajectory(proto, method, data):
    """Mean (acc2, acc1) trajectory from the switch onward, on a common per-seed window."""
    if method.endswith("_kwta"):
        p = replace(proto, stop_threshold=[THRESHOLD, THRESHOLD])
        runs = [run(p, method, seed, data=data, k=kwta_k[(method, proto.scenario)])
               for seed in range(SEEDS)]
    else:
        runs = [run_headline(proto, method, seed, data) for seed in range(SEEDS)]

    n_bad = sum(1 for o in runs if not all(o["reached"]))
    if n_bad:
        print(f"  WARNING: {proto.scenario} {method}: {n_bad}/{SEEDS} seed(s) hit the cap")

    hi = int(1.2 * np.median([o["switches"][1] - o["switches"][0] for o in runs]))
    rel_grid = np.arange(0, hi + EVAL_EVERY, EVAL_EVERY)     # post-switch only
    A = np.full((SEEDS, len(rel_grid), 2), np.nan)
    for i, o in enumerate(runs):
        rel = np.asarray(o["steps"]) - o["switches"][0]
        for j, r in enumerate(rel):
            k_ = int(round(r / EVAL_EVERY))
            if 0 <= k_ < len(rel_grid):
                A[i, k_] = o["curves"]["argmax"][j]
    # common window: crop to where every seed has finite data (see 102/113)
    first, last = [], []
    for r in range(A.shape[0]):
        idx = np.where(np.isfinite(A[r, :, 0]) & np.isfinite(A[r, :, 1]))[0]
        if idx.size:
            first.append(idx[0]); last.append(idx[-1])
    lo_idx, hi_idx = max(first), min(last)
    A = A[:, lo_idx:hi_idx + 1, :]
    acc1_mean = np.nanmean(A[:, :, 0], axis=0) * 100
    acc2_mean = np.nanmean(A[:, :, 1], axis=0) * 100
    return acc2_mean, acc1_mean


if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__), allow_pickle=True)
    traj = {s: {m: (z[f"acc2_{s}_{m}"], z[f"acc1_{s}_{m}"]) for m in METHODS} for s in SCENARIOS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    base = replace(PROTOCOL, hidden=H, eval_every=EVAL_EVERY, seeds=SEEDS, lr=LR,
                   max_iters_per_task=MAX_ITERS, stop_patience=STOP_PATIENCE)
    data = load(replace(base, scenario="class_il"))
    traj = {s: {} for s in SCENARIOS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        proto = replace(base, scenario=scen)
        for method in METHODS:
            acc2, acc1 = cell_trajectory(proto, method, data)
            traj[scen][method] = (acc2, acc1)
            print(f"  {scen:10s} {method:14s} final acc1={acc1[-1]:5.1f} acc2={acc2[-1]:5.1f}"
                  f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- report figure: ACC1 vs ACC2, one panel per scenario
fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6.0 * len(SCENARIOS), 5.0))
for ax, scen in zip(axes, SCENARIOS):
    ax.plot([0, 100], [0, 100], color="k", ls=":", lw=1.2)
    for method in METHODS:
        acc2, acc1 = traj[scen][method]
        ax.plot(acc2, acc1, color=COLORS[FAMILY[method]], ls=LINESTYLES[MECH[method]],
               lw=1.8, label=method)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("task-2 acc (%)")
    ax.set_ylabel("task-1 acc (%)")
    ax.text(0.5, 1.02, scen.replace("_", "-"), fontsize=9, ha="center", transform=ax.transAxes)
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), H=H,
         **{f"acc2_{s}_{m}": traj[s][m][0] for s in SCENARIOS for m in METHODS},
         **{f"acc1_{s}_{m}": traj[s][m][1] for s in SCENARIOS for m in METHODS})
print(f"saved {array_path(__file__)}")
