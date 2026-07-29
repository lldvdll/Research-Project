"""Reusable plotting for continual-learning experiments.

plot_learning_curves : accuracy of each task vs training step (thin = runs, thick = mean).
plot_trajectory      : path through (task1, task2) accuracy space (2-task only).

Both take `curves` as {method_name: array[runs, evals, n_tasks]} and a list of method names,
so any experiment that produces per-task accuracy over time can reuse them.
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
                         switches=None, ncols=2, task_labels=None):
    """steps: 1D array of eval steps.  curves[m]: array [runs, evals, n_tasks]."""
    fig, axes = _grid(len(methods), ncols)
    n_tasks = np.asarray(next(iter(curves.values()))).shape[-1]
    labels = task_labels or [f"task {i + 1}" for i in range(n_tasks)]
    for ax, m in zip(axes, methods):
        A = np.asarray(curves[m]) * 100.0                       # [runs, evals, tasks]
        for t in range(n_tasks):
            c = TASK_COLORS[t % len(TASK_COLORS)]
            for r in range(A.shape[0]):
                ax.plot(steps, A[r, :, t], color=c, lw=0.7, alpha=0.22)
            ax.plot(steps, A[:, :, t].mean(0), color=c, lw=2.6, label=labels[t])
        for s in (switches or []):
            ax.axvline(s, color="k", lw=0.8, ls="--")
        ax.set_title(m)
        ax.set_ylim(-2, 103)
    axes[0].legend(fontsize=8)
    for ax in axes[len(axes) - ncols:]:
        ax.set_xlabel("training step")
    for i in range(0, len(axes), ncols):
        axes[i].set_ylabel("accuracy on task's classes (%)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")
    return fig


def plot_trajectory(curves, methods, out_path, title="", ncols=2):
    """2-task only: path through (task1 acc, task2 acc). Up-right of the diagonal = retains both."""
    fig, axes = _grid(len(methods), ncols)
    for ax, m in zip(axes, methods):
        A = np.asarray(curves[m]) * 100.0
        for r in range(A.shape[0]):
            ax.plot(A[r, :, 0], A[r, :, 1], color="tab:purple", lw=0.7, alpha=0.22)
        M = A.mean(0)
        ax.plot(M[:, 0], M[:, 1], color="tab:purple", lw=2.6)
        ax.plot(M[-1, 0], M[-1, 1], "o", color="k", ms=7)               # final point
        ax.plot([100, 0], [0, 100], color="gray", ls=":", lw=1)          # equal-tradeoff diagonal
        ax.set_title(m)
        ax.set_xlim(-2, 103)
        ax.set_ylim(-2, 103)
    for ax in axes[len(axes) - ncols:]:
        ax.set_xlabel("task 1 accuracy (%)")
    for i in range(0, len(axes), ncols):
        axes[i].set_ylabel("task 2 accuracy (%)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")
    return fig
