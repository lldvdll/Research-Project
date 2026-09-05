"""Freezing the trunk (W1) or the output layer (W2) during task 2 -- which one, if either,
recovers task-1 retention, and does that differ by rule or by scenario?

Report series 2, decade 130 (new topic: where does the damage live in weight space). Backprop
+ PC, both scenarios, matched-competence 90%, 10 seeds. Task 1 always trains completely normally;
`freeze` is only switched on at the task 1 -> task 2 boundary (via run_classil's own
`on_task_end` hook), so a frozen condition never prevents task 1 itself from being learned.

Four conditions:
  control      nothing frozen during task 2 -- the baseline this whole series otherwise reports
  freeze_w1    W1, b1 held still during task 2 -- does protecting the TRUNK recover task 1?
  freeze_w2    W2, b2 held still during task 2 -- does protecting the OUTPUT LAYER recover task 1?
  freeze_both  everything held still -- the trivial ceiling (task 1 CANNOT change once task 2
               starts), included so the other three bars have a reference point, not because it
               answers a question. Task 2 cannot reach a competence threshold under total freeze,
               so this condition alone uses a short FIXED budget instead of stop_threshold.

Two report-figure variants on the SAME data (per the project's line/bar A-B convention): one
with 3 bars (control/freeze_w1/freeze_w2 -- the actual question) and one with the 4th
(freeze_both) added as context. Bar chart, x = condition, grouped by rule, one panel per
scenario. Metric: crossover (locked in by 112).

Built-in sanity check, not a separate experiment: freeze_w2 should recover ~nothing in
Domain-IL, since there is no output-layer suppression mechanism there to protect (every unit is
a positive target in both tasks). If it does, that is a flag on the design, not a result.

A second set of figures (task1/task2 vs time, one panel per condition, per scenario/method) is
saved purely to sanity-check the runs. NOT for the report.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, build, replace, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import run_classil
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
CONDITIONS = ["control", "freeze_w1", "freeze_w2", "freeze_both"]
FREEZE_SETS = {"control": set(), "freeze_w1": {"W1", "b1"}, "freeze_w2": {"W2", "b2"},
               "freeze_both": {"W1", "b1", "W2", "b2"}}
METHODS = ["backprop", "pc"]
LR = {"backprop": 0.01, "pc": 0.02}
METHOD_COLORS = {"backprop": "0.4", "pc": "tab:orange"}
SCENARIOS = ["class_il", "domain_il"]
SEEDS = 10
EVAL_EVERY = 10
MAX_ITERS = 5000            # matched-competence cap for control/freeze_w1/freeze_w2
FREEZE_BOTH_T2_BUDGET = 500  # fixed: under total freeze accuracy cannot move, so a threshold
                             # chase would only burn compute for a flat line already implied
STOP_PATIENCE = 3
THRESHOLD = 0.90
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"

if SMOKE:
    SEEDS, EVAL_EVERY, MAX_ITERS, FREEZE_BOTH_T2_BUDGET = 2, 5, 200, 40
    print("--smoke: tiny budget, results are NOT meaningful\n")

if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32")


def run_condition(proto, method, seed, cond, data):
    """One run of `cond`, freeze applied only at the task1 -> task2 boundary."""
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    handle = {}
    train_step, predict = build(proto, method, seed, handle=handle)

    def on_task_end(ti, step):
        if ti == 0 and cond != "control":
            handle["freeze"].update(FREEZE_SETS[cond])

    if cond == "freeze_both":
        thr, iters = [THRESHOLD, None], [MAX_ITERS, FREEZE_BOTH_T2_BUDGET]
    else:
        thr, iters = [THRESHOLD, THRESHOLD], [MAX_ITERS, MAX_ITERS]

    return run_classil(train_step, predict, tasks, data.train, data.class_idx,
                       report_eval=data.report_eval, stop_eval=data.stop_eval,
                       max_iters_per_task=iters, batch=proto.batch, eval_every=proto.eval_every,
                       device=proto.device, stop_threshold=thr, stop_patience=proto.stop_patience,
                       data_seed=seed, label_map=lmap, on_task_end=on_task_end)


def cell_metrics(proto, method, cond, data):
    """Per-seed crossover array for one (scenario, method, condition) cell."""
    runs = [run_condition(proto, method, seed, cond, data) for seed in range(SEEDS)]
    n_bad = sum(1 for o in runs if not all(o["reached"]))
    if n_bad:
        print(f"  WARNING: {proto.scenario} {method} {cond}: {n_bad}/{SEEDS} seed(s) hit the cap")
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
    crossover = {s: {m: {c: z[f"crossover_{s}_{m}_{c}"] for c in CONDITIONS} for m in METHODS}
                 for s in SCENARIOS}
    diag = {s: {m: {c: (z[f"steps_{s}_{m}_{c}"], z[f"curve_{s}_{m}_{c}"]) for c in CONDITIONS}
                for m in METHODS} for s in SCENARIOS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    base = replace(PROTOCOL, hidden=H, eval_every=EVAL_EVERY, seeds=SEEDS, lr=LR,
                   max_iters_per_task=MAX_ITERS, stop_patience=STOP_PATIENCE)
    data = load(replace(base, scenario="class_il"))
    crossover = {s: {m: {} for m in METHODS} for s in SCENARIOS}
    diag = {s: {m: {} for m in METHODS} for s in SCENARIOS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        proto = replace(base, scenario=scen)
        for method in METHODS:
            for cond in CONDITIONS:
                cx, steps, A = cell_metrics(proto, method, cond, data)
                crossover[scen][method][cond] = cx
                diag[scen][method][cond] = (steps, A)
                print(f"  {scen:10s} {method:8s} {cond:11s} crossover {np.nanmean(cx):5.1f}"
                      f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- report figure(s): bar, x=condition
def bar_figure(conds, suffix):
    xpos = np.arange(len(conds))
    width = 0.35
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(6.5, 7.0), sharex=True)
    for ax, scen in zip(axes, SCENARIOS):
        for i, method in enumerate(METHODS):
            means, sems = [], []
            for c in conds:
                v = crossover[scen][method][c]
                n = np.sum(np.isfinite(v))
                means.append(np.nanmean(v))
                sems.append(np.nanstd(v, ddof=1) / np.sqrt(n) if n > 1 else np.nan)
            offset = (i - 0.5) * width
            ax.bar(xpos + offset, means, width=width, yerr=sems, capsize=3,
                   color=METHOD_COLORS[method], label=method)
        ax.set_xticks(xpos)
        ax.set_ylabel(f"{scen.replace('_', '-')}\ncrossover (%)", fontsize=9)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8, loc="best")
    axes[-1].set_xticklabels([c.replace("_", "-") for c in conds])
    fig.tight_layout()
    fig.savefig(figure_path(__file__, suffix), dpi=130, bbox_inches="tight")
    print(f"saved {figure_path(__file__, suffix)}")


bar_figure(["control", "freeze_w1", "freeze_w2"], "")
bar_figure(CONDITIONS, "with_ceiling")

# ------------------------------------------------------------------ diagnostic (NOT for the report)
for scen in SCENARIOS:
    fig2, axes2 = plt.subplots(len(CONDITIONS), len(METHODS),
                                figsize=(4.2 * len(METHODS), 2.2 * len(CONDITIONS)), sharex=False)
    for row, cond in enumerate(CONDITIONS):
        for col, method in enumerate(METHODS):
            ax = axes2[row, col]
            steps, A = diag[scen][method][cond]
            for r in range(A.shape[0]):
                ax.plot(steps, A[r, :, 0] * 100, color="tab:orange", lw=0.6, alpha=0.25)
                ax.plot(steps, A[r, :, 1] * 100, color="tab:blue", lw=0.6, alpha=0.25)
            ax.plot(steps, np.nanmean(A[:, :, 0], axis=0) * 100, color="tab:orange", lw=1.8)
            ax.plot(steps, np.nanmean(A[:, :, 1], axis=0) * 100, color="tab:blue", lw=1.8)
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel(cond, fontsize=8)
            if row == 0:
                ax.text(0.5, 1.05, method, fontsize=9, ha="center", transform=ax.transAxes)
    fig2.tight_layout()
    fig2.savefig(figure_path(__file__, f"diagnostic_{scen}"), dpi=110, bbox_inches="tight")
    print(f"saved {figure_path(__file__, f'diagnostic_{scen}')}  (analysis only, not for the report)")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), H=H,
         **{f"crossover_{s}_{m}_{c}": crossover[s][m][c]
            for s in SCENARIOS for m in METHODS for c in CONDITIONS},
         **{f"steps_{s}_{m}_{c}": diag[s][m][c][0]
            for s in SCENARIOS for m in METHODS for c in CONDITIONS},
         **{f"curve_{s}_{m}_{c}": diag[s][m][c][1]
            for s in SCENARIOS for m in METHODS for c in CONDITIONS})
print(f"saved {array_path(__file__)}")
