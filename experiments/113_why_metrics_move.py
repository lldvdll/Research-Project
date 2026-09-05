"""Why does a metric's value shift with the stopping point -- concretely, on the real curves?

Report series 2, topic 110 (the stopping-criterion/metric-choice question), third script.
Explains the MECHANISM behind 110/111's instability: as the stopping point moves, task 1's own
peak moves with it, and every metric that isn't self-normalised against that peak inherits the
shift. This is a presentation figure, not a decision figure -- it tells the reader "here is the
moving target", not "here is which metric to trust" (that is 112, a separate script).

No new training: reuses 111's already-saved backprop curves (threshold sweep) and recomputes the
metrics -- including two not saved by 111 (area_retained, peak_t1) -- from the saved per-seed
accuracy arrays via the same `metric_grid` call 111 already used.

One metric group, all rescaled to a 0-100 "percent" axis so they can share one plot as asked:
  crossover        raw accuory at the crossing point
  final_t1         endpoint
  sb_mean_err      Song & Bogacz combined error
  area_retained    already a %-of-peak (mean task-1 accuracy over task 2, / its own peak)
  crossover_of_peak  ADDED: 100 * crossover / peak_t1 -- crossover rescaled the same way
                     area_retained already is, so the two self-normalised metrics are visually
                     comparable to the three raw ones on the same axis.

Two figures (one per scenario), each: top = the five metrics vs threshold; bottom = a row of
small trajectory thumbnails, one per threshold, showing the actual task1/task2 curve driving
that point.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import metric_grid

SCENARIOS = ["class_il", "domain_il"]
METRICS = ["crossover", "final_t1", "sb_mean_err", "area_retained", "crossover_of_peak"]
METRIC_COLORS = {"crossover": "tab:red", "final_t1": "tab:blue", "sb_mean_err": "tab:gray",
                  "area_retained": "tab:green", "crossover_of_peak": "tab:orange"}
SRC = Path(__file__).parent / "111_metric_sensitivity_to_threshold.npz"

def common_window(steps, A):
    """Crop to the range where EVERY seed has finite data. Averaging ragged per-seed curves
    over a window some seeds only partly cover makes the mean jump each time a seed enters or
    leaves -- not a training artefact, a plotting one. Used for the trajectory thumbnails only;
    the metrics above them are computed on the full padded array, matching 111/112's numbers."""
    first, last = [], []
    for r in range(A.shape[0]):
        idx = np.where(np.isfinite(A[r, :, 0]) & np.isfinite(A[r, :, 1]))[0]
        if idx.size:
            first.append(idx[0]); last.append(idx[-1])
    lo_idx, hi_idx = max(first), min(last)
    return steps[lo_idx:hi_idx + 1], A[:, lo_idx:hi_idx + 1, :]


z = np.load(SRC, allow_pickle=True)
THRESHOLDS = z["thresholds"].tolist()

for scen in SCENARIOS:
    res = {k: [] for k in METRICS}
    trajectories = []
    for thr in THRESHOLDS:
        steps = z[f"steps_{scen}_{thr}"]
        A = z[f"curve_{scen}_{thr}"]                 # [seeds, evals, 2], already relative to switch
        g = metric_grid(steps, A, 0.0)
        sb = np.nanmean(g["mean_err_t1"] + g["mean_err_t2"]) / 2
        crossover_of_peak = 100 * np.nanmean(g["crossover"]) / np.nanmean(g["peak_t1"])
        res["crossover"].append(np.nanmean(g["crossover"]))
        res["final_t1"].append(np.nanmean(g["final_t1"]))
        res["sb_mean_err"].append(sb)
        res["area_retained"].append(np.nanmean(g["area_retained"]))
        res["crossover_of_peak"].append(crossover_of_peak)
        trajectories.append(common_window(steps, A))    # (steps, full per-seed array), cropped

    fig = plt.figure(figsize=(2.6 * len(THRESHOLDS), 6.5))
    gs = fig.add_gridspec(2, len(THRESHOLDS), height_ratios=[2.4, 1], hspace=0.4)
    ax_top = fig.add_subplot(gs[0, :])
    for k in METRICS:
        ax_top.plot([t * 100 for t in THRESHOLDS], res[k], marker="o", lw=2,
                    color=METRIC_COLORS[k], label=k.replace("_", " "))
    ax_top.set_xlabel("stop threshold (%)")
    ax_top.set_ylabel("value (%)")
    ax_top.legend(fontsize=8, ncol=2, loc="best")
    ax_top.grid(alpha=0.25)

    for i, thr in enumerate(THRESHOLDS):
        ax = fig.add_subplot(gs[1, i])
        steps, A_win = trajectories[i]
        for r in range(A_win.shape[0]):
            ax.plot(steps, A_win[r, :, 0] * 100, color="tab:orange", lw=0.5, alpha=0.25)
            ax.plot(steps, A_win[r, :, 1] * 100, color="tab:blue", lw=0.5, alpha=0.25)
        ax.plot(steps, np.nanmean(A_win[:, :, 0], axis=0) * 100, color="tab:orange", lw=1.6)
        ax.plot(steps, np.nanmean(A_win[:, :, 1], axis=0) * 100, color="tab:blue", lw=1.6)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_xlabel(f"{thr:.0%}", fontsize=8)
        if i == 0:
            ax.set_ylabel("acc (%)", fontsize=7)
        else:
            ax.set_yticks([])
        ax.grid(alpha=0.15)

    fig.savefig(figure_path(__file__, scen), dpi=130, bbox_inches="tight")
    print(f"saved {figure_path(__file__, scen)}")

    # same top-panel data, bar-chart alternative, kept alongside the line version to compare
    xpos = np.arange(len(THRESHOLDS))
    width = 0.16
    fig_bar, ax_bar = plt.subplots(figsize=(8, 4.2))
    for j, k in enumerate(METRICS):
        offset = (j - (len(METRICS) - 1) / 2) * width
        ax_bar.bar(xpos + offset, res[k], width=width, color=METRIC_COLORS[k],
                   label=k.replace("_", " "))
    ax_bar.set_xticks(xpos)
    ax_bar.set_xticklabels([f"{t:.0%}" for t in THRESHOLDS])
    ax_bar.set_xlabel("stop threshold (%)")
    ax_bar.set_ylabel("value (%)")
    ax_bar.legend(fontsize=8, ncol=2, loc="best")
    ax_bar.grid(alpha=0.25, axis="y")
    fig_bar.tight_layout()
    fig_bar.savefig(figure_path(__file__, f"{scen}_bar"), dpi=130, bbox_inches="tight")
    print(f"saved {figure_path(__file__, f'{scen}_bar')}")
