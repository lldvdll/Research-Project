"""11_consolidate_pairs_4methods
Q: Across many random digit pairings, how do backprop / replay / eqprop / pc trade off
   learning task 2 against forgetting task 1?
   Same 2-task Class-IL problem, N_RUNS independent runs with random digit pairings.
   Outputs: 2x2 accuracy-vs-step grid, 2x2 trajectory grid (thin = runs, thick = mean),
            a pairing table and a per-method summary table.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.data import load_mnist, class_indices, make_eval_set
from src.methods import make_backprop, make_replay, make_eqprop, make_pc, legacy

# ============================ constants ============================
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR        = ROOT / "data"
FIG             = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE        = 14
BASE_SEED       = 0
N_RUNS          = 10                   # random digit pairings

ITERS           = 100                  # updates per task
BATCH           = 32
EVAL_EVERY      = 1                    # raise to 2-5 if runtime is painful (eqprop dominates)
EVAL_PER_CLASS  = 100

BP_LR                          = 0.05
RP_LR, RP_PER_CLASS            = 0.05, 20
EQP_LR, EQP_BETA, EQP_DT       = 0.005, 0.3, 0.3
EQP_MAX_STEPS, EQP_SETTLE_PAT  = 500, 30
PC_LR, PC_DT, PC_STEPS         = 0.05, 0.1, 50

METHODS  = ["backprop", "replay", "eqprop", "pc"]      # panel order: controls top, EBMs bottom
IN_DIM   = IMG_SIZE * IMG_SIZE
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)


def build(name, tasks, seed):
    """Fresh model for this run. Hyperparameters fixed; only the seed and pairing change."""
    # **legacy(...) pins the pre-unification specification this script was written against; the
    # library default is now the unified protocol. See src/methods.py legacy(). Not for new work.
    if name == "backprop":
        return make_backprop(in_dim=IN_DIM, lr=BP_LR, seed=seed, device=DEVICE,
                             **legacy("backprop"))
    if name == "replay":
        return make_replay(train, cidx, in_dim=IN_DIM, lr=RP_LR, per_class=RP_PER_CLASS,
                           seed=seed, device=DEVICE, **legacy("replay"))
    if name == "eqprop":
        return make_eqprop(in_dim=IN_DIM, lr=EQP_LR, beta=EQP_BETA, dt=EQP_DT,
                           max_steps=EQP_MAX_STEPS, settle_patience=EQP_SETTLE_PAT,
                           seed=seed, device=DEVICE, **legacy("eqprop"))
    if name == "pc":
        return make_pc(in_dim=IN_DIM, lr=PC_LR, dt=PC_DT, steps=PC_STEPS, seed=seed,
                       device=DEVICE, **legacy("pc"))
    raise ValueError(name)


def run_once(train_step, predict, tasks, eval_x, eval_y, cls_pos, seed):
    """One Class-IL run. Returns per-task accuracy [n_evals, n_tasks] and the eval steps."""
    torch.manual_seed(seed)
    steps_log, task_acc = [], []
    step = 0
    for task in tasks:
        idx = torch.cat([cidx[c] for c in task])
        loader = DataLoader(Subset(train, idx.tolist()), batch_size=BATCH, shuffle=True)
        it = iter(loader)
        for _ in range(ITERS):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader); x, y = next(it)
            train_step(x.to(DEVICE), y.to(DEVICE))
            step += 1
            if step % EVAL_EVERY == 0:
                pred = predict(eval_x)
                acc = [(pred[eval_y == c] == c).float().mean().item() for c in cls_pos]
                steps_log.append(step)
                task_acc.append([float(np.mean([acc[cls_pos[c]] for c in t])) for t in tasks])
    return np.array(steps_log), np.array(task_acc)


def crossover(steps, t1, t2, switch):
    """Accuracy value where task-1 and task-2 curves cross after the switch (linear interp).
       High value = the model held both tasks at once; low = it traded one for the other."""
    ii = [i for i, s in enumerate(steps) if s > switch]
    for a, b in zip(ii[:-1], ii[1:]):
        d1, d2 = t1[a] - t2[a], t1[b] - t2[b]
        if d1 > 0 >= d2:
            w = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.0
            return steps[a] + w * (steps[b] - steps[a]), t1[a] + w * (t1[b] - t1[a])
    return None, float("nan")


def first_cross(steps, series, thresh, switch, rising):
    for s, v in zip(steps, series):
        if s <= switch:
            continue
        if (v >= thresh) if rising else (v < thresh):
            return s - switch
    return None


# ------------------------------ run everything ------------------------------
pairings, curves = [], {m: [] for m in METHODS}
t_start = time.time()
for r in range(N_RUNS):
    rng = np.random.default_rng(BASE_SEED + r)
    d = rng.permutation(10).tolist()
    tasks = [[d[0], d[1], d[2], d[3], d[4]], [d[5], d[6], d[7], d[8], d[9]]]
    classes = sorted({c for t in tasks for c in t})
    cls_pos = {c: i for i, c in enumerate(classes)}
    eval_x, eval_y = make_eval_set(test, classes=classes, per_class=EVAL_PER_CLASS, device=DEVICE)
    pairings.append(tasks)
    print(f"\nrun {r+1}/{N_RUNS}: task1={tasks[0]}  task2={tasks[1]}")
    for m in METHODS:
        t0 = time.time()
        step_fn, pred_fn = build(m, tasks, seed=BASE_SEED + r)
        steps, T = run_once(step_fn, pred_fn, tasks, eval_x, eval_y, cls_pos, seed=BASE_SEED + r)
        curves[m].append(T)
        _, xv = crossover(steps, T[:, 0], T[:, 1], ITERS)
        print(f"   {m:>9}: final T1 {T[-1,0]*100:5.1f}%  T2 {T[-1,1]*100:5.1f}%  "
              f"crossover {xv*100 if xv == xv else float('nan'):5.1f}%  ({time.time()-t0:4.0f}s)")
STEPS = steps                                        # identical across runs
print(f"\ntotal {time.time()-t_start:.0f}s")

# ------------------------------- tables -------------------------------
print("\n" + "=" * 60)
print("PAIRINGS")
print(f"{'run':>4} {'task 1':>10} {'task 2':>10}")
for i, t in enumerate(pairings):
    print(f"{i+1:>4} {str(t[0]):>10} {str(t[1]):>10}")

print("\n" + "=" * 92)
print("SUMMARY over runs (mean +/- sd)")
hdr = (f"{'method':>9} {'crossover%':>16} {'final T1%':>15} {'final T2%':>15} "
       f"{'learn T2':>10} {'forget T1':>10}")
print(hdr)
summary = {}
for m in METHODS:
    A = np.stack(curves[m])                          # [runs, evals, tasks]
    xs = [crossover(STEPS, A[r, :, 0], A[r, :, 1], ITERS)[1] for r in range(N_RUNS)]
    xs = np.array(xs, dtype=float) * 100
    f1, f2 = A[:, -1, 0] * 100, A[:, -1, 1] * 100
    L = [first_cross(STEPS, A[r, :, 1], 0.5, ITERS, True) for r in range(N_RUNS)]
    F = [first_cross(STEPS, A[r, :, 0], 0.5, ITERS, False) for r in range(N_RUNS)]
    L = np.array([v for v in L if v is not None], dtype=float)
    F = np.array([v for v in F if v is not None], dtype=float)
    summary[m] = dict(cross=xs, f1=f1, f2=f2)
    print(f"{m:>9} {np.nanmean(xs):>8.1f} +/-{np.nanstd(xs):>5.1f} "
          f"{f1.mean():>8.1f} +/-{f1.std():>4.1f} {f2.mean():>8.1f} +/-{f2.std():>4.1f} "
          f"{(L.mean() if len(L) else float('nan')):>10.0f} {(F.mean() if len(F) else float('nan')):>10.0f}")
print("crossover% = accuracy where the two task curves cross (higher = held both at once)")
print("learn T2 / forget T1 = steps after the switch to cross 50% (rising / falling)")

# --------------------- figure 1: accuracy vs step, 2x2 ---------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, m in zip(axes.ravel(), METHODS):
    A = np.stack(curves[m]) * 100
    for r in range(N_RUNS):
        ax.plot(STEPS, A[r, :, 0], color="tab:blue", lw=0.7, alpha=0.25)
        ax.plot(STEPS, A[r, :, 1], color="tab:orange", lw=0.7, alpha=0.25)
    ax.plot(STEPS, A[:, :, 0].mean(0), color="tab:blue", lw=2.6, label="task 1 (old)")
    ax.plot(STEPS, A[:, :, 1].mean(0), color="tab:orange", lw=2.6, label="task 2 (new)")
    ax.axvline(ITERS, color="k", lw=0.8, ls="--")
    ax.set_title(m); ax.set_ylim(-2, 103)
axes[0, 0].legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("training step")
for ax in axes[:, 0]:
    ax.set_ylabel("accuracy on task's classes (%)")
fig.suptitle(f"Learning vs forgetting over {N_RUNS} random digit pairings (thin = runs, thick = mean)")
plt.tight_layout(); plt.savefig(FIG, dpi=120, bbox_inches="tight"); plt.show()

# --------------------- figure 2: trajectories, 2x2 ---------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
for ax, m in zip(axes.ravel(), METHODS):
    A = np.stack(curves[m]) * 100
    for r in range(N_RUNS):
        ax.plot(A[r, :, 0], A[r, :, 1], color="tab:purple", lw=0.7, alpha=0.25)
    M = A.mean(0)
    ax.plot(M[:, 0], M[:, 1], color="tab:purple", lw=2.6)
    ax.plot(M[-1, 0], M[-1, 1], "o", color="k", ms=7)
    ax.plot([100, 0], [0, 100], color="gray", ls=":", lw=1)
    ax.set_title(m); ax.set_xlim(-2, 103); ax.set_ylim(-2, 103)
for ax in axes[1]:
    ax.set_xlabel("task 1 accuracy (%)")
for ax in axes[:, 0]:
    ax.set_ylabel("task 2 accuracy (%)")
fig.suptitle("Trajectory through (task1, task2) accuracy space — up-right of the diagonal = retains both")
plt.tight_layout(); plt.savefig(FIG.with_name(FIG.stem + "_traj.png"), dpi=120); plt.show()
print(f"\nsaved {FIG.name} and {FIG.stem}_traj.png")
