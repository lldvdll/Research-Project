"""Reusable plotting for continual-learning experiments.

plot_learning_curves : accuracy of each task vs training step (thin = runs, thick = mean).
plot_trajectory      : path through (task1, task2) accuracy space (2-task only).
plot_heatmap         : mean +/- std grid over two swept variables.

`curves` is {method_name: array[runs, evals, n_tasks]}; NaN entries are ignored
(runs of unequal length, e.g. under early stopping, are NaN-padded by src.metrics.align_runs).
"""
import numpy as np
import matplotlib.pyplot as plt

TASK_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def _grid(n, ncols):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    return fig, axes.ravel()


def plot_learning_curves(steps, curves, methods, out_path, title="",
                         switches=None, ncols=2, task_labels=None, xlabel="training step"):
    """steps: 1D array of eval steps.  curves[m]: array [runs, evals, n_tasks]. NaN-safe."""
    fig, axes = _grid(len(methods), ncols)
    n_tasks = np.asarray(next(iter(curves.values()))).shape[-1]
    labels = task_labels or [f"task {i + 1}" for i in range(n_tasks)]
    for ax, m in zip(axes, methods):
        A = np.asarray(curves[m], dtype=float) * 100.0                 # [runs, evals, tasks]
        for t in range(n_tasks):
            c = TASK_COLORS[t % len(TASK_COLORS)]
            for r in range(A.shape[0]):
                ax.plot(steps, A[r, :, t], color=c, lw=0.7, alpha=0.22)
            ax.plot(steps, np.nanmean(A[:, :, t], axis=0), color=c, lw=2.6, label=labels[t])
        for s in (switches or []):
            ax.axvline(s, color="k", lw=0.8, ls="--")
        ax.set_title(m)
        ax.set_ylim(-2, 103)
    axes[0].legend(fontsize=8)
    for ax in axes[len(axes) - ncols:]:
        ax.set_xlabel(xlabel)
    for i in range(0, len(axes), ncols):
        axes[i].set_ylabel("accuracy on task's classes (%)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")
    return fig


def plot_trajectory(curves, methods, out_path, title="", ncols=2, threshold=None):
    """2-task only: path through (task1 acc, task2 acc). Up-right of the line = retains both.

    threshold : if given (0-1), the reference line runs (T,0)-(0,T) instead of (100,0)-(0,100),
                i.e. the equal-trade-off line at the accuracy standard both tasks were
                trained to. The (T,T) corner marks 'both tasks held at that standard'.
    """
    fig, axes = _grid(len(methods), ncols)
    for ax, m in zip(axes, methods):
        A = np.asarray(curves[m], dtype=float) * 100.0
        for r in range(A.shape[0]):
            ax.plot(A[r, :, 0], A[r, :, 1], color="tab:purple", lw=0.7, alpha=0.22)
        M = np.nanmean(A, axis=0)
        ax.plot(M[:, 0], M[:, 1], color="tab:purple", lw=2.6)
        ax.plot(M[-1, 0], M[-1, 1], "o", color="k", ms=7)                     # final point
        if threshold is None:
            ax.plot([100, 0], [0, 100], color="gray", ls=":", lw=1, label="equal trade-off")
        else:
            T = threshold * 100.0
            ax.plot([T, 0], [0, T], color="gray", ls=":", lw=1.2,
                    label=f"equal trade-off at threshold ({T:.0f}%)")
            ax.plot([T], [T], marker="*", color="tab:green", ms=13, ls="none",
                    label="both tasks at threshold")
        ax.set_title(m)
        ax.set_xlim(-2, 103)
        ax.set_ylim(-2, 103)
    axes[0].legend(fontsize=8, loc="lower left")
    for ax in axes[len(axes) - ncols:]:
        ax.set_xlabel("task 1 accuracy (%)")
    for i in range(0, len(axes), ncols):
        axes[i].set_ylabel("task 2 accuracy (%)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")
    return fig


def plot_heatmap(mean, std, row_labels, col_labels, out_path, title="",
                 row_name="", col_name="", cbar_label="", fmt="{:.1f}",
                 cmap="viridis", vmin=None, vmax=None, figsize=None):
    """Heatmap of `mean` with 'mean +/- std' annotated in each cell. NaN cells show as '-'."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    fig, ax = plt.subplots(figsize=figsize or (1.35 * len(col_labels) + 3, 0.85 * len(row_labels) + 2.5))
    im = ax.imshow(mean, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_xlabel(col_name); ax.set_ylabel(row_name)
    finite = mean[np.isfinite(mean)]
    mid = (finite.max() + finite.min()) / 2 if finite.size else 0.0
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if not np.isfinite(mean[i, j]):
                txt, colour = "-", "gray"
            else:
                txt = fmt.format(mean[i, j])
                if np.isfinite(std[i, j]):
                    txt += "\n±" + fmt.format(std[i, j])
                colour = "white" if mean[i, j] < mid else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=colour)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")
    return fig
