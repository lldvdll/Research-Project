"""Does PC forget task 1 differently from backprop, read once BOTH tasks reach the SAME accuracy?

Backprop and PC only, same reason as 72. Without replay in the comparison, the thing that forced
53's original 90%/80% split is gone -- PC and backprop both clear 90% on task 2 comfortably in
every run so far, so both tasks use ONE number: 90%.

WHY THIS REPLACES 53 AND 57
    53 stopped task 1 at 90% and task 2 at 80% -- 80% chosen because a pilot showed replay
    (the slowest to learn task 2) could not reliably clear 90% in budget. With replay out of the
    picture there is no reason left for two numbers, or for the pilot that measured the second
    one. Both scenarios in one script, as in 72.

Deviation from the protocol: none beyond 52/53's original deviation (accuracy stopping is the
protocol; 72 is the deviation, for the fixed-budget comparison).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.protocol import (PROTOCOL, load, run, replace,
                          figure_path as _figure_path, array_path as _array_path)
from src.metrics import metric_grid, report_grid, crossover, half_life, value_when, paired_diff
from src.plotting import plot_learning_curves

SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):      # noqa: F811
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):       # noqa: F811
    return _array_path(f, _tag(f, suffix))


METHODS = ["backprop", "pc"]
SCENARIOS = ["domain_il", "class_il"]
SEEDS, EVAL_EVERY, STOP_PATIENCE = 5, 10, 3
MAX_ITERS = 2000                    # safety cap only; reached/capped is reported honestly
THRESHOLD = 0.90                    # SAME threshold, both tasks -- safe without replay in the mix
TASK_COLORS = ["tab:orange", "tab:blue"]
COLORS = {"backprop": "tab:gray", "pc": "tab:red"}

HIDDEN = int(np.load(array_path(str(ROOT / "experiments" /
                                    "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(array_path(str(ROOT / "experiments" / "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
SETTLE_TOL, EQ_MAX_STEPS = float(z51["settle_tol"]), int(z51["eq_max_steps"])
PC_STEPS = int(z51["pc_steps"])

if SMOKE:
    SEEDS, MAX_ITERS, THRESHOLD = 1, 60, 0.5
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} | symmetric threshold {THRESHOLD:.0%}/{THRESHOLD:.0%} | cap {MAX_ITERS} | "
      f"{SEEDS} seeds")


def settle_kw(m):
    if m == "pc":
        return dict(steps=PC_STEPS)
    if m == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


runs, t0 = {}, time.perf_counter()
for scn in SCENARIOS:
    base = replace(PROTOCOL, hidden=HIDDEN, scenario=scn, stop_patience=STOP_PATIENCE,
                   stop_threshold=[THRESHOLD, THRESHOLD], max_iters_per_task=MAX_ITERS,
                   eval_every=EVAL_EVERY, seeds=SEEDS)
    data = load(base)
    for m in METHODS:
        proto = replace(base, lr={m: LR[m]})
        rows = [run(proto, m, seed, data=data, **settle_kw(m)) for seed in range(SEEDS)]
        runs[(scn, m)] = rows
        r1 = sum(o["reached"][0] for o in rows)
        r2 = sum(o["reached"][1] for o in rows)
        print(f"  {scn:10s} {m:10s} reached: task1 {r1}/{SEEDS}  task2 {r2}/{SEEDS}"
              f"  [{time.perf_counter() - t0:6.0f}s]")

# ---------------------------------------------------------------- readings + figures
for scn in SCENARIOS:
    print(f"\n{'=' * 70}\n  {scn}, symmetric matched competence {THRESHOLD:.0%}/{THRESHOLD:.0%}\n"
          f"{'=' * 70}")
    LO = -int(1.2 * max(np.mean([o["switches"][0] for o in runs[(scn, m)]]) for m in METHODS))
    HI = int(1.2 * max(np.mean([o["switches"][1] - o["switches"][0] for o in runs[(scn, m)]])
                       for m in METHODS))
    grid = np.arange(LO, HI + EVAL_EVERY, EVAL_EVERY)
    padded = {}
    for m in METHODS:
        A = np.full((SEEDS, len(grid), 2), np.nan)
        for i, o in enumerate(runs[(scn, m)]):
            rel = np.asarray(o["steps"]) - o["switches"][0]
            for j, r in enumerate(rel):
                k = int(round((r - LO) / EVAL_EVERY))
                if 0 <= k < len(grid):
                    A[i, k] = o["curves"]["argmax"][j]
        padded[m] = A

    report_grid({m: metric_grid(grid, padded[m], 0) for m in METHODS},
                METHODS, control="backprop", primary="crossover")

    plot_learning_curves(
        grid, padded, METHODS, figure_path(__file__, scn),
        blocks=[(LO, 0, 0), (0, HI, 1)], ncols=2, task_colors=TASK_COLORS,
        task_labels=["task 1", "task 2"], crossover_after=0,
        xlabel="training step, relative to the task switch",
        legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False))

    np.savez(array_path(__file__, scn), steps=grid, hidden=HIDDEN, threshold=THRESHOLD,
             methods=np.asarray(METHODS), switch=0,
             **{f"argmax_{m}": padded[m] for m in METHODS})

np.savez(array_path(__file__), hidden=HIDDEN, threshold=THRESHOLD, cap=MAX_ITERS,
         methods=np.asarray(METHODS), scenarios=np.asarray(SCENARIOS))
print(f"\nsaved per-scenario arrays and {array_path(__file__)}")
