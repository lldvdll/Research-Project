"""15_matched_accuracy_forgetting
Q: With every method trained on each task to the SAME accuracy standard, how much of task 1
   survives while task 2 is learned?

This is experiment 12 with two changes that remove the confounds:
  1. HIDDEN is applied to every method, so architecture is identical.
  2. Per-task EARLY STOPPING at ACC_THRESHOLD -- a task ends once its accuracy has been at or
     above the threshold for STOP_PATIENCE consecutive evals. Methods therefore learn each task
     to the same standard, so differences in forgetting can no longer be attributed to one
     method simply learning faster than another.

Because tasks now end at different steps, runs have different lengths: they are aligned on the
task switch (step 0 = switch) and NaN-padded before averaging.

HEADLINE METRIC: task-1 accuracy at the moment task 2 first reaches the threshold. One number
per run, the same standard for every method, no speed confound.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import load_mnist, class_indices, make_eval_set
from src.methods import build_method, legacy
from src.runner import run_classil
from src.metrics import align_runs, value_when, crossover
from src.plotting import plot_learning_curves, plot_trajectory

# ============================ constants ============================
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR        = ROOT / "data"
FIG             = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE        = 14
BASE_SEED       = 0
N_RUNS          = 10

HIDDEN          = 64        # applied to EVERY method
ACC_THRESHOLD   = 0.80      # the standard each task is trained to (set from experiment 13)
STOP_PATIENCE   = 3         # consecutive evals at/above threshold before moving on
TAIL_ITERS      = 0         # extra updates after the criterion is met (0 = stop immediately)

CLASSES_PER_TASK   = 2      # 2 tasks x this many classes
MAX_ITERS_PER_TASK = 400    # cap if the threshold is never reached
BATCH           = 32
EVAL_EVERY      = 1

BP_LR                          = 0.05
RP_LR, RP_PER_CLASS            = 0.05, 20
EQP_LR, EQP_BETA, EQP_DT       = 0.005, 0.3, 0.3
EQP_MAX_STEPS, EQP_SETTLE_PAT  = 500, 30
PC_LR, PC_DT, PC_STEPS         = 0.05, 0.1, 50

METHODS = ["backprop", "replay", "eqprop", "pc"]      # controls top row, EBMs bottom row
IN_DIM  = IMG_SIZE * IMG_SIZE
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)

OVERRIDES = {
    "backprop": dict(lr=BP_LR),
    "replay":   dict(lr=RP_LR, per_class=RP_PER_CLASS),
    "eqprop":   dict(lr=EQP_LR, beta=EQP_BETA, dt=EQP_DT,
                     max_steps=EQP_MAX_STEPS, settle_patience=EQP_SETTLE_PAT),
    "pc":       dict(lr=PC_LR, dt=PC_DT, steps=PC_STEPS),
}

steps_by, curves_by, switch_by = {m: [] for m in METHODS}, {m: [] for m in METHODS}, {m: [] for m in METHODS}
pairings = []
t_start = time.time()

for r in range(N_RUNS):
    d = np.random.default_rng(BASE_SEED + r).permutation(10).tolist()
    k = CLASSES_PER_TASK
    tasks = [sorted(d[:k]), sorted(d[k:2 * k])]
    classes = sorted({c for t in tasks for c in t})
    eval_x, eval_y = make_eval_set(test, classes=classes, per_class=100, device=DEVICE)
    pairings.append(tasks)
    print(f"\nrun {r + 1}/{N_RUNS}: task1={tasks[0]} task2={tasks[1]}")
    for m in METHODS:
        t0 = time.time()
        # **legacy(...) pins the pre-unification specification this script was written against;
        # the library default is now the unified protocol. Not for new work.
        step_fn, pred_fn = build_method(m, in_dim=IN_DIM, hidden=HIDDEN, seed=BASE_SEED + r,
                                        device=DEVICE, train_data=train, class_idx=cidx,
                                        **legacy(m), **OVERRIDES[m])
        s, T, switches = run_classil(step_fn, pred_fn, tasks, train, cidx, eval_x, eval_y,
                                     max_iters_per_task=MAX_ITERS_PER_TASK, batch=BATCH,
                                     eval_every=EVAL_EVERY, device=DEVICE,
                                     stop_threshold=ACC_THRESHOLD, stop_patience=STOP_PATIENCE,
                                     tail_iters=TAIL_ITERS)
        steps_by[m].append(s); curves_by[m].append(T); switch_by[m].append(switches[0])
        survived = value_when(s, T[:, 1], ACC_THRESHOLD, T[:, 0],
                              after=switches[0], patience=STOP_PATIENCE)
        print(f"   {m:>9}: task1 {switches[0]:>4} steps, task2 {s[-1] - switches[0]:>4} steps | "
              f"task1 when task2 hit {ACC_THRESHOLD:.0%}: "
              f"{'never' if np.isnan(survived) else f'{survived:5.1%}'} | ({time.time() - t0:4.0f}s)")
print(f"\ntotal {time.time() - t_start:.0f}s")

# --------------------------- align and summarise ---------------------------
rel_steps, aligned = None, {}
for m in METHODS:
    grid, stacked = align_runs(steps_by[m], curves_by[m], switch_by[m], eval_every=EVAL_EVERY)
    aligned[m] = (grid, stacked)
lo = min(g[0] for g, _ in aligned.values())
hi = max(g[-1] for g, _ in aligned.values())
rel_steps = np.arange(lo, hi + EVAL_EVERY, EVAL_EVERY)
curves = {}
for m in METHODS:                                    # place each method onto the common axis
    g, stacked = aligned[m]
    out = np.full((stacked.shape[0], len(rel_steps), stacked.shape[2]), np.nan)
    off = int(round((g[0] - lo) / EVAL_EVERY))
    out[:, off:off + stacked.shape[1], :] = stacked
    curves[m] = out

print("\n" + "=" * 60)
print("PAIRINGS")
for i, t in enumerate(pairings):
    print(f"{i + 1:>4} task1={str(t[0]):>12} task2={str(t[1]):>12}")

print("\n" + "=" * 96)
print(f"SUMMARY over {N_RUNS} runs (mean +/- sd), threshold = {ACC_THRESHOLD:.0%}, hidden = {HIDDEN}")
print(f"{'method':>9} {'T1 @ T2-threshold':>20} {'steps T1':>10} {'steps T2':>10} "
      f"{'crossover%':>14} {'final T1%':>12}")
for m in METHODS:
    surv, sT1, sT2, xs, fT1 = [], [], [], [], []
    for r in range(N_RUNS):
        s, T, sw = steps_by[m][r], curves_by[m][r], switch_by[m][r]
        surv.append(value_when(s, T[:, 1], ACC_THRESHOLD, T[:, 0], after=sw, patience=STOP_PATIENCE))
        sT1.append(sw); sT2.append(s[-1] - sw)
        xs.append(crossover(s, T[:, 0], T[:, 1], after=sw)[1])
        fT1.append(T[-1, 0])
    surv, xs, fT1 = np.array(surv) * 100, np.array(xs) * 100, np.array(fT1) * 100
    print(f"{m:>9} {np.nanmean(surv):>13.1f} +/-{np.nanstd(surv):>5.1f} "
          f"{np.mean(sT1):>10.0f} {np.mean(sT2):>10.0f} "
          f"{np.nanmean(xs):>9.1f} +/-{np.nanstd(xs):>3.1f} {fT1.mean():>8.1f} +/-{fT1.std():>3.1f}")
print("T1 @ T2-threshold = task-1 accuracy the moment task 2 reached the standard  <- HEADLINE")
print("steps T1 / T2     = updates needed to reach the standard (should now be the fair part)")

# ------------------------------- figures -------------------------------
plot_learning_curves(rel_steps, curves, METHODS, FIG,
                     title=(f"Matched-accuracy forgetting: each task trained to {ACC_THRESHOLD:.0%} "
                            f"(hidden={HIDDEN}, {N_RUNS} runs, aligned on the task switch)"),
                     switches=[0], ncols=2,
                     task_labels=["task 1 (old)", "task 2 (new)"],
                     xlabel="updates relative to task switch")

plot_trajectory(curves, METHODS, FIG.with_name(FIG.stem + "_traj.png"),
                title="Trajectory through (task1, task2) accuracy space",
                ncols=2, threshold=ACC_THRESHOLD)

np.savez(FIG.with_suffix(".npz"), rel_steps=rel_steps,
         **{f"curves_{m}": curves[m] for m in METHODS},
         pairings=np.array(pairings), threshold=ACC_THRESHOLD, hidden=HIDDEN)
print(f"saved raw results to {FIG.with_suffix('.npz').name}")
