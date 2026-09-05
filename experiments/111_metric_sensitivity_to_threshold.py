"""How much do the standard metrics move when you change the accuracy threshold you stop at,
and does crossover height stay stable while they do?

Fresh run, report series 2, topic 110 (second experiment in the "does the stopping criterion
change the metrics" question -- see 110's docstring for the numbering convention). Backprop
only, both scenarios, 10 seeds. Sweeps the accuracy threshold BOTH tasks are trained to
(matched-competence style, as in old 73) instead of the fixed budget 110 sweeps.

Same three metrics, same report figure shape (3 vertical panels, both scenarios overlaid), same
treatment of crossover's censoring at the low end: a low threshold means task 2 trains only
briefly, so the curves may not have crossed within the observed window yet.

Same diagnostic figures (task 1 orange / task 2 blue, one panel per threshold, per scenario) --
analysis only, not for the report.
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
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
SEEDS = 10
EVAL_EVERY = 10
MAX_ITERS = 3000        # generous cap; a run hitting it would be reported, not silently truncated
STOP_PATIENCE = 3
SCENARIOS = ["class_il", "domain_il"]
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
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

METRICS = ["crossover", "final_t1", "sb_mean_err"]
LABELS = {"crossover": "crossover (%)", "final_t1": "endpoint (%)", "sb_mean_err": "S&B error (%)"}

if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__), allow_pickle=True)
    res = {k: {s: z[f"{k}_{s}"] for s in SCENARIOS} for k in METRICS}
    diag = {s: {t: (z[f"steps_{s}_{t}"], z[f"curve_{s}_{t}"], z[f"t1len_{s}_{t}"], z[f"t2len_{s}_{t}"])
                for t in THRESHOLDS} for s in SCENARIOS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    base = replace(PROTOCOL, hidden=H, eval_every=EVAL_EVERY, seeds=SEEDS,
                   max_iters_per_task=MAX_ITERS, stop_patience=STOP_PATIENCE,
                   lr={"backprop": 0.01})   # BUG FIX: was missing, silently used the default 0.05
    data = load(replace(base, scenario="class_il"))
    res = {k: {s: np.full((len(THRESHOLDS), SEEDS), np.nan) for s in SCENARIOS} for k in METRICS}
    diag = {s: {} for s in SCENARIOS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        proto = replace(base, scenario=scen)
        for ti, thr in enumerate(THRESHOLDS):
            p = replace(proto, stop_threshold=[thr, thr])
            runs = [run(p, "backprop", seed, data=data) for seed in range(SEEDS)]

            n_bad = sum(1 for o in runs if not all(o["reached"]))
            if n_bad:
                print(f"  WARNING: {scen} thr={thr:.0%}: {n_bad}/{SEEDS} seed(s) hit the "
                      f"{MAX_ITERS}-update cap without reaching threshold")

            # Threshold-stopped runs finish at different lengths -- pad onto a common grid
            # relative to the switch, as 73/74 do, rather than np.stack. MEDIAN, not mean: a
            # single slow-to-converge seed inflated the mean-based range enough to starve most
            # of the grid of coverage (thr=0.95 blew out to +-3000 steps against 0.90's +-300).
            t1_lens = [o["switches"][0] for o in runs]
            t2_lens = [o["switches"][1] - o["switches"][0] for o in runs]
            lo = -int(1.2 * np.median(t1_lens))
            hi = int(1.2 * np.median(t2_lens))
            rel_grid = np.arange(lo, hi + EVAL_EVERY, EVAL_EVERY)
            A = np.full((SEEDS, len(rel_grid), 2), np.nan)
            for i, o in enumerate(runs):
                rel = np.asarray(o["steps"]) - o["switches"][0]
                for j, r in enumerate(rel):
                    k_ = int(round((r - lo) / EVAL_EVERY))
                    if 0 <= k_ < len(rel_grid):
                        A[i, k_] = o["curves"]["argmax"][j]

            g = metric_grid(rel_grid, A, 0.0)
            sb = (g["mean_err_t1"] + g["mean_err_t2"]) / 2
            res["crossover"][scen][ti] = g["crossover"]
            res["final_t1"][scen][ti] = g["final_t1"]
            res["sb_mean_err"][scen][ti] = sb
            diag[scen][thr] = (rel_grid, A, np.asarray(t1_lens), np.asarray(t2_lens))
            print(f"  {scen:10s} threshold={thr:.0%}  "
                  f"crossover {np.nanmean(g['crossover']):5.1f}  "
                  f"final_t1 {np.nanmean(g['final_t1']):5.1f}  "
                  f"sb_err {np.nanmean(sb):5.1f}  "
                  f"peak_t1 {np.nanmean(g['peak_t1']):5.1f}  "
                  f"t1_len median {np.median(t1_lens):.0f} [{min(t1_lens):.0f},{max(t1_lens):.0f}]  "
                  f"t2_len median {np.median(t2_lens):.0f} [{min(t2_lens):.0f},{max(t2_lens):.0f}]"
                  f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- report figure: 3 panels, grouped bars
xs = np.asarray([t * 100 for t in THRESHOLDS])
width = 1.6
fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.0), sharex=True)
for ax, k in zip(axes, METRICS):
    for i, s in enumerate(SCENARIOS):
        v = res[k][s]
        mean = np.nanmean(v, axis=1)
        n = np.sum(np.isfinite(v), axis=1)
        semv = np.where(n > 1, np.nanstd(v, axis=1, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
        offset = (i - 0.5) * width
        ax.bar(xs + offset, mean, width=width, yerr=semv, capsize=3,
               color=COLORS[s], label=s.replace("_", "-"))
    ax.set_xticks(xs)
    ax.set_ylabel(LABELS[k], fontsize=9)
    ax.grid(alpha=0.25, axis="y")
axes[0].legend(fontsize=8, loc="best")
axes[-1].set_xlabel("stop threshold (%)")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ------------------------------------------------- diagnostic (NOT for the report): task1/task2 vs time, per scenario
for s in SCENARIOS:
    fig2, axes2 = plt.subplots(len(THRESHOLDS), 1, figsize=(6, 2.2 * len(THRESHOLDS)), sharex=False)
    for ax, thr in zip(axes2, THRESHOLDS):
        steps, A, t1_lens, t2_lens = diag[s][thr]     # A: [seeds, evals, 2]
        for r in range(A.shape[0]):
            ax.plot(steps, A[r, :, 0] * 100, color="tab:orange", lw=0.6, alpha=0.25)
            ax.plot(steps, A[r, :, 1] * 100, color="tab:blue", lw=0.6, alpha=0.25)
        ax.plot(steps, np.nanmean(A[:, :, 0], axis=0) * 100, color="tab:orange", lw=2.0, label="task 1")
        ax.plot(steps, np.nanmean(A[:, :, 1], axis=0) * 100, color="tab:blue", lw=2.0, label="task 2")
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_ylim(0, 100)
        ax.set_ylabel(f"{thr:.0%}", fontsize=8)
        ax.grid(alpha=0.2)
        # per-seed spread of how many updates each task actually took, to relate at a glance to 110
        ax.text(0.99, 0.04,
                f"t1 {np.median(t1_lens):.0f} [{t1_lens.min():.0f}-{t1_lens.max():.0f}]  "
                f"t2 {np.median(t2_lens):.0f} [{t2_lens.min():.0f}-{t2_lens.max():.0f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="dimgray")
    axes2[0].legend(fontsize=7, loc="lower left")
    axes2[-1].set_xlabel("training step, relative to the task switch")
    fig2.tight_layout()
    fig2.savefig(figure_path(__file__, f"diagnostic_{s}"), dpi=110, bbox_inches="tight")
    print(f"saved {figure_path(__file__, f'diagnostic_{s}')}  (analysis only, not for the report)")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), thresholds=np.asarray(THRESHOLDS), H=H,
         **{f"{k}_{s}": res[k][s] for k in METRICS for s in SCENARIOS},
         **{f"steps_{s}_{t}": diag[s][t][0] for s in SCENARIOS for t in THRESHOLDS},
         **{f"curve_{s}_{t}": diag[s][t][1] for s in SCENARIOS for t in THRESHOLDS},
         **{f"t1len_{s}_{t}": diag[s][t][2] for s in SCENARIOS for t in THRESHOLDS},
         **{f"t2len_{s}_{t}": diag[s][t][3] for s in SCENARIOS for t in THRESHOLDS})
print(f"saved {array_path(__file__)}")
