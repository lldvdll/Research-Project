"""Does PC forget task 1 differently from backprop, under a fixed, SYMMETRIC budget -- both tasks
get the same number of updates, replacing 52/56's asymmetric 840/630 for presentation clarity.

Backprop and PC only. Replay and EqProp are left out here deliberately -- EqProp is too slow to
iterate on under tonight's time constraint, and replay is what forced 53's original 90%/80% split
(it couldn't reliably reach 90% on task 2); without it in the comparison, both tasks can use the
same clean 90% and this stays genuinely simple to explain live.

WHY THIS REPLACES 52 AND 56
    52 gave task 1 two calibrated periods (2T=840) and task 2 one and a half (1.5T=630). Defensible,
    but two different numbers for two tasks is an extra thing to explain live, for no result that
    changes. This version gives both tasks 2T=840: task 1 is still comfortably past its plateau,
    and 840 updates of task 2 is still well short of the repeated-alternation regime (68) where
    rules re-converge, so the signal this experiment reports on is not at risk.

    Both scenarios (Domain-IL, Class-IL) in one script, so the two "usual grid" figures come from
    one run instead of two separately-maintained files.

Deviation from the protocol: fixed budget instead of accuracy stopping (73 is the matched-
competence pair, same deviation as 52/53 before it).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.protocol import (PROTOCOL, load, run, replace,
                          figure_path as _figure_path, array_path as _array_path)
from src.metrics import metric_grid, report_grid, paired_diff
from src.plotting import plot_learning_curves

SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):      # noqa: F811
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):       # noqa: F811
    return _array_path(f, _tag(f, suffix))


METHODS = ["backprop", "pc"]        # replay and eqprop left out: eqprop is too slow to iterate
                                    # on tonight, replay was only what forced the 80% threshold
                                    # in 73 -- without it this experiment can use 90% cleanly
SCENARIOS = ["domain_il", "class_il"]
SEEDS, EVAL_EVERY = 5, 10
TASK_COLORS = ["tab:orange", "tab:blue"]
COLORS = {"backprop": "tab:gray", "pc": "tab:red"}

HIDDEN = int(np.load(array_path(str(ROOT / "experiments" /
                                    "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(array_path(str(ROOT / "experiments" / "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
SETTLE_TOL, EQ_MAX_STEPS = float(z51["settle_tol"]), int(z51["eq_max_steps"])
PC_STEPS = int(z51["pc_steps"])
T = float(z51["target_steps"])
BUDGET = int(2 * T)                 # SAME budget, both tasks
ITERS = [BUDGET, BUDGET]

if SMOKE:
    SEEDS, ITERS = 1, [40, 40]
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} | symmetric fixed budget {BUDGET}+{BUDGET} | {SEEDS} seeds")


def settle_kw(m):
    if m == "pc":
        return dict(steps=PC_STEPS)
    if m == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


curves, steps_by_scn, switch_by_scn = {}, {}, {}
t0 = time.perf_counter()
for scn in SCENARIOS:
    base = replace(PROTOCOL, hidden=HIDDEN, scenario=scn, stop_threshold=None,
                   max_iters_per_task=ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)
    data = load(base)
    for m in METHODS:
        proto = replace(base, lr={m: LR[m]})
        rows = []
        for seed in range(SEEDS):
            out = run(proto, m, seed, data=data, **settle_kw(m))
            rows.append(out["curves"]["argmax"])
        curves[(scn, m)] = np.stack(rows)
        steps_by_scn[scn], switch_by_scn[scn] = out["steps"], out["switches"]
        print(f"  {scn:10s} {m:10s} done  [{time.perf_counter() - t0:6.0f}s]")

# ---------------------------------------------------------------- readings + figures
for scn in SCENARIOS:
    print(f"\n{'=' * 70}\n  {scn}, symmetric fixed budget {BUDGET}/{BUDGET}\n{'=' * 70}")
    report_grid({m: metric_grid(steps_by_scn[scn], curves[(scn, m)], switch_by_scn[scn][0])
                 for m in METHODS}, METHODS, control="backprop", primary="crossover")

    blocks = [(0, switch_by_scn[scn][0], 0), (switch_by_scn[scn][0], steps_by_scn[scn][-1], 1)]
    plot_learning_curves(
        steps_by_scn[scn], {m: curves[(scn, m)] for m in METHODS}, METHODS,
        figure_path(__file__, scn), blocks=blocks, ncols=2, task_colors=TASK_COLORS,
        task_labels=["task 1", "task 2"],
        legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False))

np.savez(array_path(__file__), hidden=HIDDEN, budget=BUDGET, methods=np.asarray(METHODS),
         scenarios=np.asarray(SCENARIOS),
         **{f"steps_{s}": steps_by_scn[s] for s in SCENARIOS},
         **{f"switches_{s}": np.asarray(switch_by_scn[s]) for s in SCENARIOS},
         **{f"argmax_{s}_{m}": curves[(s, m)] for s in SCENARIOS for m in METHODS})
print(f"\nsaved {array_path(__file__)}")
