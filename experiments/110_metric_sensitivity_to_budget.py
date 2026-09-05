"""How much do the standard metrics move when you change the training budget alone, and does
crossover height stay stable while they do?

Fresh run, report series 2, topic 110 (numbering convention: each new question claims the next
decade -- see `claude_code_management/now.md` -- so 111 can sit alongside this one and a later
112 would not force a renumber). Backprop only, both scenarios, 10 seeds. Sweeps the per-task
update budget with early stopping OFF: task 1 always trains BUDGET updates, then task 2 always
trains BUDGET updates -- the fixed protocol from old 72/100, but BUDGET itself is now the swept
variable instead of held at 840.

One report figure: 3 vertical panels (endpoint = final task-1 accuracy, Song & Bogacz mean error,
crossover height), both scenarios overlaid, error bars = SEM over 10 seeds. Crossover is expected
to be undefined at small budgets -- task 2 has not run long enough to cross task 1's curve yet --
and that is left as a gap in the line, not filled in.

A second set of figures (task 1 orange / task 2 blue, one panel per budget, per scenario) is
saved purely to sanity-check the runs. NOT for the report.
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
BUDGETS = [25, 50, 100, 200, 400, 800, 1600]     # updates per task
SEEDS = 10
EVAL_EVERY = 10
SCENARIOS = ["class_il", "domain_il"]
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"

if SMOKE:
    BUDGETS, SEEDS, EVAL_EVERY = [20, 40], 2, 5
    print("--smoke: tiny budget, results are NOT meaningful\n")

if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32")

METRICS = ["crossover", "final_t1", "sb_mean_err"]
LABELS = {"crossover": "crossover (%)", "final_t1": "endpoint (%)", "sb_mean_err": "S&B error (%)"}

if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__), allow_pickle=True)
    res = {k: {s: z[f"{k}_{s}"] for s in SCENARIOS} for k in METRICS}
    diag = {s: {b: (z[f"steps_{s}_{b}"], z[f"curve_{s}_{b}"]) for b in BUDGETS} for s in SCENARIOS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    base = replace(PROTOCOL, hidden=H, stop_threshold=None, eval_every=EVAL_EVERY, seeds=SEEDS,
                   lr={"backprop": 0.01})   # BUG FIX: was missing, silently used the default 0.05
    data = load(replace(base, scenario="class_il"))
    res = {k: {s: np.full((len(BUDGETS), SEEDS), np.nan) for s in SCENARIOS} for k in METRICS}
    diag = {s: {} for s in SCENARIOS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        proto = replace(base, scenario=scen)
        for bi, budget in enumerate(BUDGETS):
            p = replace(proto, max_iters_per_task=budget)
            curves, steps, switch = [], None, None
            for seed in range(SEEDS):
                out = run(p, "backprop", seed, data=data)
                curves.append(out["curves"]["argmax"])
                steps, switch = out["steps"], out["switches"][0]
            A = np.stack(curves, axis=0)   # [seeds, evals, 2] -- fixed budget, same length every seed
            g = metric_grid(steps, A, switch)
            sb = (g["mean_err_t1"] + g["mean_err_t2"]) / 2
            res["crossover"][scen][bi] = g["crossover"]
            res["final_t1"][scen][bi] = g["final_t1"]
            res["sb_mean_err"][scen][bi] = sb
            diag[scen][budget] = (steps, A)   # full per-seed array, for the spaghetti diagnostic
            print(f"  {scen:10s} budget={budget:5d}  "
                  f"crossover {np.nanmean(g['crossover']):5.1f}  "
                  f"final_t1 {np.nanmean(g['final_t1']):5.1f}  "
                  f"sb_err {np.nanmean(sb):5.1f}  "
                  f"peak_t1 {np.nanmean(g['peak_t1']):5.1f}   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- report figure: 3 panels, both scenarios overlaid
fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.0), sharex=True)
for ax, k in zip(axes, METRICS):
    for s in SCENARIOS:
        v = res[k][s]
        mean = np.nanmean(v, axis=1)
        n = np.sum(np.isfinite(v), axis=1)
        semv = np.where(n > 1, np.nanstd(v, axis=1, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
        ok = np.isfinite(mean)
        ax.errorbar(np.asarray(BUDGETS)[ok], mean[ok], yerr=semv[ok], marker="o", capsize=3,
                    lw=2, color=COLORS[s], label=s.replace("_", "-"))
    ax.set_ylabel(LABELS[k], fontsize=9)
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8, loc="best")
axes[-1].set_xlabel("updates per task")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ------------------------------------------- same data, bar-chart alternative (BUDGETS spans 64x,
# so bars sit at equally-spaced positions with budget as the tick label, not at literal x-values)
xpos = np.arange(len(BUDGETS))
width = 0.38
fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.0), sharex=True)
for ax, k in zip(axes, METRICS):
    for i, s in enumerate(SCENARIOS):
        v = res[k][s]
        mean = np.nanmean(v, axis=1)
        n = np.sum(np.isfinite(v), axis=1)
        semv = np.where(n > 1, np.nanstd(v, axis=1, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
        offset = (i - 0.5) * width
        ax.bar(xpos + offset, mean, width=width, yerr=semv, capsize=3,
               color=COLORS[s], label=s.replace("_", "-"))
    ax.set_xticks(xpos)
    ax.set_ylabel(LABELS[k], fontsize=9)
    ax.grid(alpha=0.25, axis="y")
axes[0].legend(fontsize=8, loc="best")
axes[-1].set_xticklabels(BUDGETS)
axes[-1].set_xlabel("updates per task")
fig.tight_layout()
fig.savefig(figure_path(__file__, "bar"), dpi=130, bbox_inches="tight")
print(f"saved {figure_path(__file__, 'bar')}")

# ------------------------------------------------- diagnostic (NOT for the report): task1/task2 vs time, per scenario
for s in SCENARIOS:
    fig2, axes2 = plt.subplots(len(BUDGETS), 1, figsize=(6, 2.2 * len(BUDGETS)), sharex=False)
    for ax, budget in zip(axes2, BUDGETS):
        steps, A = diag[s][budget]                 # A: [seeds, evals, 2]
        for r in range(A.shape[0]):
            ax.plot(steps, A[r, :, 0] * 100, color="tab:orange", lw=0.6, alpha=0.25)
            ax.plot(steps, A[r, :, 1] * 100, color="tab:blue", lw=0.6, alpha=0.25)
        ax.plot(steps, A[:, :, 0].mean(0) * 100, color="tab:orange", lw=2.0, label="task 1")
        ax.plot(steps, A[:, :, 1].mean(0) * 100, color="tab:blue", lw=2.0, label="task 2")
        ax.axvline(budget, color="k", ls="--", lw=1)
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"{budget}", fontsize=8)
        ax.grid(alpha=0.2)
    axes2[0].legend(fontsize=7, loc="lower left")
    axes2[-1].set_xlabel("training step")
    fig2.tight_layout()
    fig2.savefig(figure_path(__file__, f"diagnostic_{s}"), dpi=110, bbox_inches="tight")
    print(f"saved {figure_path(__file__, f'diagnostic_{s}')}  (analysis only, not for the report)")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), budgets=np.asarray(BUDGETS), H=H,
         **{f"{k}_{s}": res[k][s] for k in METRICS for s in SCENARIOS},
         **{f"steps_{s}_{b}": diag[s][b][0] for s in SCENARIOS for b in BUDGETS},
         **{f"curve_{s}_{b}": diag[s][b][1] for s in SCENARIOS for b in BUDGETS})
print(f"saved {array_path(__file__)}")
