"""Does backprop's absolute retention/crossover degrade FASTER than PC's as lr is pushed past
their shared optimum (both rules peak near lr=0.01-0.02, per 340) -- and does crossover's own
measurement reliability (the fraction of seeds where it's undefined because the curves never
resolve a crossing) break down at the same high-lr end?

Report series 2, decade 343. Reuses 340's saved arrays directly -- no new training. Built to make
two things from 340's console output into an actual figure, per direct request: the earlier
pc-minus-backprop DIFFERENCE plot (340's own diff figure) cannot distinguish "PC improving" from
"backprop degrading faster", since a difference is blind to which side moved -- that distinction
needs the two rules' own absolute curves side by side, which is what the top two rows here are.
The bottom row is the crossover-censoring-rate check: 340 found crossover fully undefined for
every seed at lr=0.16 in class-il, which is a measurement-validity failure, not a null result --
this plots how that failure rate grows with lr, so it's visible directly rather than only showing
up as a shrinking n in a console table.

Requires 340_lr_sweep_{scenario}.npz to already exist (run 340 first).

One figure: 343_lr_degradation_and_reliability.png, 2 scenario columns x 3 rows:
    row 1  absolute retention vs lr, backprop and pc, mean +- SEM
    row 2  absolute crossover height vs lr, backprop and pc, mean +- SEM
    row 3  crossover censoring rate (%) vs lr, backprop and pc -- the fraction of seeds where
           crossover height came back NaN (curves never crossed, per src.metrics.crossover)

A fitted degradation slope (linear least squares of the metric against lr, restricted to lr values
at or past each metric's own peak) is printed per rule/metric/scenario -- a shallower slope means
slower degradation, the direct quantitative form of the "PC degrades more gently" claim.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path as _figure_path, array_path as _array_path

SCENARIOS = ["class_il", "domain_il"]
METHODS = ["backprop", "pc"]
METHOD_COLOR = {"backprop": "0.35", "pc": "tab:orange"}

SOURCE = Path(__file__).parent / "340_lr_sweep.py"


def _load(scenario):
    p = _array_path(SOURCE, scenario)
    if not Path(p).exists():
        raise RuntimeError(f"{p} not found -- run 340_lr_sweep.py first")
    return list(np.load(p, allow_pickle=True)["data"])


def _mean_sem(rows, method, lr, col):
    v = np.array([r[col] for r in rows if r[0] == method and r[1] == lr], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def _censor_rate(rows, method, lr, col):
    v = [r[col] for r in rows if r[0] == method and r[1] == lr]
    if not v:
        return float("nan")
    return 100.0 * np.mean([not np.isfinite(x) for x in v])


COL_RETENTION = 3
COL_CROSSOVER = 5

data = {s: _load(s) for s in SCENARIOS}
lrs = sorted(set(r[1] for s in SCENARIOS for r in data[s]))

fig, axes = plt.subplots(3, len(SCENARIOS), figsize=(5.5 * len(SCENARIOS), 11.5), squeeze=False)

print("fitted degradation slope (least squares, metric vs lr, from each rule's own peak lr "
     "onward -- shallower = degrades more slowly)")
for col_i, scenario in enumerate(SCENARIOS):
    rows = data[scenario]

    # row 1: absolute retention
    ax = axes[0][col_i]
    for method in METHODS:
        means, sems = zip(*[_mean_sem(rows, method, lr, COL_RETENTION) for lr in lrs])
        ax.errorbar(lrs, means, yerr=sems, marker="o", color=METHOD_COLOR[method],
                   capsize=3, label=method)
    ax.set_xscale("log"); ax.set_xticks(lrs); ax.set_xticklabels([str(v) for v in lrs], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("learning rate"); ax.set_ylabel("task-1 retention (%)")
    ax.set_title(f"{scenario.replace('_', '-')}: retention vs lr (absolute)", fontsize=9)
    ax.grid(alpha=0.2)
    if col_i == 0:
        ax.legend(fontsize=7)

    # row 2: absolute crossover
    ax = axes[1][col_i]
    for method in METHODS:
        means, sems = zip(*[_mean_sem(rows, method, lr, COL_CROSSOVER) for lr in lrs])
        ax.errorbar(lrs, means, yerr=sems, marker="o", color=METHOD_COLOR[method],
                   capsize=3, label=method)
    ax.set_xscale("log"); ax.set_xticks(lrs); ax.set_xticklabels([str(v) for v in lrs], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("learning rate"); ax.set_ylabel("crossover height (%)")
    ax.set_title(f"{scenario.replace('_', '-')}: crossover vs lr (absolute)", fontsize=9)
    ax.grid(alpha=0.2)

    # row 3: crossover censoring rate
    ax = axes[2][col_i]
    for method in METHODS:
        rates = [_censor_rate(rows, method, lr, COL_CROSSOVER) for lr in lrs]
        ax.plot(lrs, rates, marker="o", color=METHOD_COLOR[method], label=method)
    ax.set_xscale("log"); ax.set_xticks(lrs); ax.set_xticklabels([str(v) for v in lrs], fontsize=7)
    ax.minorticks_off()
    ax.set_ylim(-5, 105)
    ax.set_xlabel("learning rate"); ax.set_ylabel("crossover undefined (%)")
    ax.set_title(f"{scenario.replace('_', '-')}: crossover censoring rate vs lr", fontsize=9)
    ax.grid(alpha=0.2)

    # Fitted degradation slopes, both rules over the SAME window -- from the LATER of the two
    # rules' own peaks onward, so neither slope gets an unfair head start or a thin/short window
    # relative to the other. Fitting each rule from its own peak (the first version of this
    # script did that) looked like a fair comparison but wasn't: whichever rule peaks later gets
    # fewer post-peak points, and in class-il that left pc's crossover slope fit on just 2 points
    # -- not a real regression, just the gap between two numbers presented as if it were one.
    print(f"\n  {scenario}")
    for name, col in [("retention", COL_RETENTION), ("crossover", COL_CROSSOVER)]:
        peaks = []
        for method in METHODS:
            means = np.array([_mean_sem(rows, method, lr, col)[0] for lr in lrs])
            peaks.append(int(np.nanargmax(means)) if np.any(np.isfinite(means)) else 0)
        start_i = max(peaks)   # later of the two peaks -- common window for both rules
        for method in METHODS:
            means = np.array([_mean_sem(rows, method, lr, col)[0] for lr in lrs])
            xs = np.log(lrs[start_i:])
            ys = means[start_i:]
            ok = np.isfinite(ys)
            if ok.sum() < 2:
                print(f"    {name:9s} {method:9s} common window from lr={lrs[start_i]}, "
                     f"not enough finite points to fit a slope")
                continue
            slope, intercept = np.polyfit(xs[ok], ys[ok], 1)
            print(f"    {name:9s} {method:9s} common window from lr={lrs[start_i]:<6} "
                 f"({ok.sum()} points)   slope (per unit log-lr) = {slope:+6.2f}")

fig.tight_layout()
out = _figure_path(__file__)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nsaved {out}")
