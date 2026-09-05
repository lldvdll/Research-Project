"""Does depth change the answer under crossover, under S&B's own metric, and under the endpoint?

59 answered this for crossover height only, and did not save full curves -- so S&B's metric and
the endpoint can't be recovered from its saved arrays after the fact. This reruns the same
depth x width factorial (same per-cell calibration, same per-cell measured task-2 threshold,
same gates) but backprop and PC only -- no need to include replay, per this slide -- and saves the
full accuracy curves so all three metrics come from the same runs.

Deviation from the protocol: none beyond 59's own (accuracy stopping is the protocol; the
threshold is measured per cell, as 59 established was necessary).
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


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):
    return _array_path(f, _tag(f, suffix))


DEPTHS = [1, 2, 3, 5]
WIDTHS = [32, 128]
METHODS = ["backprop", "pc"]
SEEDS = 5
CAL_SEEDS = 2
EVAL_EVERY = 20
T1_THRESHOLD = 0.90
STOP_PATIENCE = 3
MAX_ITERS = 2500
TARGET_STEPS = 420
PC_STEPS = 50
LR_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
WIDTH_STYLE = {32: dict(ls="-", marker="o", color="tab:red"),
               128: dict(ls="-", marker="s", color="tab:purple")}

if SMOKE:
    DEPTHS, WIDTHS, SEEDS, CAL_SEEDS = [1, 2], [32, 128], 2, 1
    MAX_ITERS, TARGET_STEPS, LR_GRID = 400, 150, [0.02, 0.05, 0.1, 0.2]
    print("--smoke: tiny budget, results are NOT meaningful\n")

CELLS = [(d, w) for w in WIDTHS for d in DEPTHS]
print(f"depth x width factorial: {[f'{d}x{w}' for d, w in CELLS]}, backprop + pc only")
print(f"  {SEEDS} seeds | task 1 to {T1_THRESHOLD:.0%} | task 2 threshold measured per cell\n")


def settle_kw(m):
    return dict(steps=PC_STEPS) if m == "pc" else {}


def base_for(d, w):
    return replace(PROTOCOL, hidden=w, n_layers=d, scenario="domain_il",
                   stop_patience=STOP_PATIENCE, max_iters_per_task=MAX_ITERS,
                   eval_every=EVAL_EVERY, seeds=SEEDS)


def calibrate(b, data):
    out = {}
    for m in METHODS:
        best, best_d = None, np.inf
        for lr in LR_GRID:
            p = replace(b, stop_threshold=T1_THRESHOLD, lr={m: lr})
            steps = []
            for s in range(CAL_SEEDS):
                o = run(p, m, s, data=data, tasks=[p.tasks(s)[0]], **settle_kw(m))
                steps.append(o["switches"][0] if o["reached"][0] else np.nan)
            if not np.isfinite(steps).all():
                continue
            dist = abs(np.log(np.mean(steps) / TARGET_STEPS))
            if dist < best_d:
                best, best_d = lr, dist
        out[m] = best
    return out


def ceiling(b, lrs, data):
    """Highest task-2 accuracy the slower of backprop/PC reaches under this cell's cap."""
    tops = []
    for m in METHODS:
        p = replace(b, stop_threshold=[T1_THRESHOLD, None], lr={m: lrs[m]})
        for s in range(CAL_SEEDS):
            o = run(p, m, s, data=data, **settle_kw(m))
            c = o["curves"]["argmax"] * 100
            i = int(np.argmin(np.abs(np.asarray(o["steps"]) - o["switches"][0])))
            tops.append(float(c[i:, 1].max()))
    return min(tops)


data = load(replace(PROTOCOL, hidden=32, scenario="domain_il"))
curves, switches_by_cell, lrs_all, thr = {}, {}, {}, {}
t0 = time.perf_counter()

for (d, w) in CELLS:
    b = base_for(d, w)
    lrs = calibrate(b, data)
    if any(v is None for v in lrs.values()):
        miss = [m for m, v in lrs.items() if v is None]
        print(f"  {d}x{w}: {', '.join(miss)} reached {T1_THRESHOLD:.0%} at NO rate -- skipped")
        continue
    lrs_all[(d, w)] = lrs
    T2 = float(np.floor((ceiling(b, lrs, data) - 3.0) / 5.0) * 5.0 / 100.0)
    thr[(d, w)] = T2
    print(f"  {d}x{w}  lr " + ", ".join(f"{m} {lrs[m]:g}" for m in METHODS)
          + f"  | task 2 stops at {T2:.0%}   [{time.perf_counter()-t0:5.0f}s]")

    runs_cell = {}
    for m in METHODS:
        p = replace(b, stop_threshold=[T1_THRESHOLD, T2], lr={m: lrs[m]})
        runs_cell[m] = [run(p, m, s, data=data, **settle_kw(m)) for s in range(SEEDS)]
        last = runs_cell[m][-1]["curves"]["argmax"]
        print(f"      {m:9s} task 1 kept {last[-1, 0]*100:5.1f}%   [{time.perf_counter()-t0:5.0f}s]")

    # Runs are threshold-stopped, so seeds have DIFFERENT lengths -- pad onto one grid relative
    # to the switch, exactly as 73 does, rather than np.stack (which needs equal shapes).
    lo = -int(1.2 * max(np.mean([o["switches"][0] for o in runs_cell[m]]) for m in METHODS))
    hi = int(1.2 * max(np.mean([o["switches"][1] - o["switches"][0] for o in runs_cell[m]])
                       for m in METHODS))
    rel_grid = np.arange(lo, hi + EVAL_EVERY, EVAL_EVERY)
    for m in METHODS:
        A = np.full((SEEDS, len(rel_grid), 2), np.nan)
        for i, o in enumerate(runs_cell[m]):
            rel = np.asarray(o["steps"]) - o["switches"][0]
            for j, r in enumerate(rel):
                k = int(round((r - lo) / EVAL_EVERY))
                if 0 <= k < len(rel_grid):
                    A[i, k] = o["curves"]["argmax"][j]
        curves[(d, w, m)] = A
    switches_by_cell[(d, w)] = (rel_grid, 0.0)   # (grid, switch=0 -- already relative)

done = sorted({(d, w) for (d, w, m) in curves})

# ---------------------------------------------------------------- three metrics, paired
METRICS = ["crossover", "final_t1", "sb_mean_err"]
res = {k: {} for k in METRICS}
for (d, w) in done:
    rel_grid, sw = switches_by_cell[(d, w)]
    g = {m: metric_grid(rel_grid, curves[(d, w, m)], sw) for m in METHODS}
    sb = {m: (g[m]["mean_err_t1"] + g[m]["mean_err_t2"]) / 2 for m in METHODS}
    res["crossover"][(d, w)] = paired_diff(g["pc"]["crossover"], g["backprop"]["crossover"])
    res["final_t1"][(d, w)] = paired_diff(g["pc"]["final_t1"], g["backprop"]["final_t1"])
    res["sb_mean_err"][(d, w)] = paired_diff(sb["pc"], sb["backprop"])

print("\n  PAIRED pc - backprop, per cell\n")
for k in METRICS:
    print(f"  {k}")
    for (d, w) in done:
        dd, sd, n = res[k][(d, w)]
        print(f"    {d}x{w:<4d} {dd:+6.2f} +-{sd:5.2f} {n:4.1f}sem")

# ---------------------------------------------------------------- the 3-panel figure
TITLES = {"crossover": "crossover height", "final_t1": "endpoint (final task-1 accuracy)",
          "sb_mean_err": "Song & Bogacz mean error (LOWER is better -- axis flipped to match)"}
fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.0), sharex=True)
for ax, k in zip(axes, METRICS):
    for w in WIDTHS:
        xs = [d for d in DEPTHS if (d, w) in done]
        ys = [res[k][(d, w)][0] for d in xs]
        es = [res[k][(d, w)][1] for d in xs]
        if k == "sb_mean_err":
            ys = [-y for y in ys]  # flip sign: lower error is better, so "up" means "PC better" everywhere
        if xs:
            ax.errorbar(xs, ys, yerr=es, capsize=3, lw=2,
                        label=f"{w} units/layer", **WIDTH_STYLE[w])
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel(f"pc − backprop\n({TITLES[k]})", fontsize=8.5)
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8, loc="best")
axes[0].set_title("Does depth change the answer? Same runs, three metrics.\n"
                  "Positive = PC better, on every panel (S&B's axis is flipped to match).",
                  fontsize=10)
axes[-1].set_xlabel("hidden layers")
axes[-1].set_xticks(DEPTHS)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

np.savez(array_path(__file__), cells=np.asarray(done), methods=np.asarray(METHODS),
         lrs=lrs_all, thresholds=thr, eval_every=EVAL_EVERY,
         **{f"steps_{d}x{w}": switches_by_cell[(d, w)][0] for (d, w) in done},
         **{f"argmax_{d}x{w}_{m}": curves[(d, w, m)] for (d, w) in done for m in METHODS})
print(f"saved {array_path(__file__)}")
