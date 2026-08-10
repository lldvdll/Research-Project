"""20_where_does_forgetting_live

QUESTION
    When a class-incremental network forgets task 1, is the damage in the HIDDEN LAYER (the
    compressed code is destroyed) or in the OUTPUT LAYER (the code survives but old units
    have been trained into silence)?

TEST
    Backprop only -- one learning rule, so nothing here is confounded by the rule. A 2x2:

                            readout: argmax over all classes | readout: nearest class mean
      loss over ALL classes            (a) total forgetting   | (b) hidden code only
      loss MASKED to current task      (c) suppression removed | (d) neither pathway active

    Masking means absent classes receive EXACTLY zero gradient, so output-layer suppression
    is removed by construction. Nearest class mean discards the output layer at evaluation,
    so calibration is removed. 10 random digit pairings, 2 classes per task, everything else
    held fixed (same pairings, same data order, same init, same eval set).

INTERPRETATION
    Read final task-1 accuracy in each cell.
      (a) low, (b) high        -> the hidden code SURVIVED. Forgetting is an output-layer
                                  problem. No learning rule can fix it: predictive coding
                                  buys a better path, not a better endpoint. Advisor point 4
                                  (node selection) is aimed at the wrong layer.
      (a) low, (b) low         -> the hidden code was DESTROYED. Representation drift is real
                                  and the learning-rule comparison is aimed correctly.
      (c) >> (a)               -> quantifies exactly how much of the loss is suppression.
      (d) - (b) small          -> masking does not protect the hidden code, i.e. the two
                                  pathologies are independent, as the decomposition predicts.

    Whatever the answer, it decides the second half of the project. Do not skip it.

FIGURE
    20_where_does_forgetting_live.png -- 2x2 grid of task-accuracy curves, one panel per cell.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import load_mnist, class_indices, make_eval_split
from src.model import Arch, Objective, replace
from src.methods import build_method
from src.runner import run_classil
from src.probes import prototype_images, live_ncm_fn
from src.plotting import plot_learning_curves

# ============================ constants ============================
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR    = ROOT / "data"
FIG         = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE    = 14
BASE_SEED   = 0
N_RUNS      = 10                 # random digit pairings

CLASSES_PER_TASK   = 2
MAX_ITERS_PER_TASK = 400
BATCH              = 32
EVAL_EVERY         = 2
EVAL_PER_CLASS     = 100         # per eval split; two disjoint splits are drawn
PROTO_PER_CLASS    = 50          # training images used to build NCM prototypes

ACC_THRESHOLD = 0.80             # per-task early stop; below the 2-class budget ceiling
STOP_PATIENCE = 3                #   (exp 13: hidden=64, 2 classes, 100 updates -> 93.8%)
TAIL_ITERS    = 0

# ONE architecture for both conditions. Only the loss mask changes.
ARCH = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=64, out_dim=10,
            act="tanh", bias=False, init="scaled_normal")
OBJ_FULL   = Objective(loss="mse", target="onehot", mask=False)
OBJ_MASKED = replace(OBJ_FULL, mask=True)
BP_LR = 0.05

CONDITIONS = {"loss: all classes": OBJ_FULL, "loss: masked to task": OBJ_MASKED}
READOUTS   = ["argmax", "ncm"]
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)

results = {(c, r): [] for c in CONDITIONS for r in READOUTS}
steps_by, switch_by = {c: [] for c in CONDITIONS}, {c: [] for c in CONDITIONS}
pairings, never_reached = [], {c: 0 for c in CONDITIONS}
t0 = time.time()

for run in range(N_RUNS):
    seed = BASE_SEED + run
    d = np.random.default_rng(seed).permutation(10).tolist()
    k = CLASSES_PER_TASK
    tasks = [sorted(d[:k]), sorted(d[k:2 * k])]
    classes = sorted({c for t in tasks for c in t})
    pairings.append(tasks)

    stop_eval, report_eval = make_eval_split(test, classes, EVAL_PER_CLASS, DEVICE, seed=seed)
    proto_x, proto_y = prototype_images(train, cidx, classes, PROTO_PER_CLASS, DEVICE, seed=seed)
    print(f"\nrun {run + 1}/{N_RUNS}: task1={tasks[0]} task2={tasks[1]}")

    for cname, obj in CONDITIONS.items():
        handle = {}
        step_fn, predict = build_method("backprop", in_dim=ARCH.in_dim, hidden=ARCH.hidden,
                                        out_dim=ARCH.out_dim, arch=ARCH, obj=obj, lr=BP_LR,
                                        seed=seed, device=DEVICE, handle=handle)
        readouts = {"argmax": predict,
                    "ncm": live_ncm_fn(handle["features"], proto_x, proto_y)}
        out = run_classil(step_fn, predict, tasks, train, cidx,
                          report_eval=report_eval, stop_eval=stop_eval, readouts=readouts,
                          max_iters_per_task=MAX_ITERS_PER_TASK, batch=BATCH,
                          eval_every=EVAL_EVERY, device=DEVICE,
                          stop_threshold=ACC_THRESHOLD, stop_patience=STOP_PATIENCE,
                          tail_iters=TAIL_ITERS, data_seed=seed)
        steps_by[cname].append(out["steps"])
        switch_by[cname].append(out["switches"][0])
        never_reached[cname] += sum(not r for r in out["reached"])
        for r in READOUTS:
            results[(cname, r)].append(out["curves"][r])
        fin = {r: out["curves"][r][-1, 0] * 100 for r in READOUTS}
        print(f"   {cname:<22} final task1  argmax {fin['argmax']:5.1f}%   ncm {fin['ncm']:5.1f}%"
              f"   (task1 {out['switches'][0]} steps)")

print(f"\ntotal {time.time() - t0:.0f}s")

# --------------------------- align on the task switch ---------------------------
from src.metrics import align_runs

grids, aligned = {}, {}
for cname in CONDITIONS:
    for r in READOUTS:
        g, stacked = align_runs(steps_by[cname], results[(cname, r)], switch_by[cname],
                                eval_every=EVAL_EVERY)
        grids[(cname, r)], aligned[(cname, r)] = g, stacked
lo = min(g[0] for g in grids.values())
hi = max(g[-1] for g in grids.values())
rel_steps = np.arange(lo, hi + EVAL_EVERY, EVAL_EVERY)

curves, MIN_RUNS = {}, max(2, N_RUNS // 2)
for key, stacked in aligned.items():
    out = np.full((stacked.shape[0], len(rel_steps), stacked.shape[2]), np.nan)
    off = int(round((grids[key][0] - lo) / EVAL_EVERY))
    out[:, off:off + stacked.shape[1], :] = stacked
    # trim to where at least MIN_RUNS runs contribute: NaN-means over 1-2 runs are noise
    enough = (~np.isnan(out[:, :, 0])).sum(0) >= MIN_RUNS
    out[:, ~enough, :] = np.nan
    curves[f"{key[0]}  |  {key[1]}"] = out

# ------------------------------- summary -------------------------------
print("\n" + "=" * 78)
print(f"FINAL TASK-1 ACCURACY (%), mean +/- sd over {N_RUNS} pairings, threshold {ACC_THRESHOLD:.0%}")
print(f"{'training condition':<26}{'argmax':>16}{'nearest class mean':>22}")
cell = {}
for cname in CONDITIONS:
    row = ""
    for r in READOUTS:
        v = np.array([c[-1, 0] for c in results[(cname, r)]]) * 100
        cell[(cname, r)] = v
        row += f"{v.mean():>13.1f} +/-{v.std():>4.1f}"
    print(f"{cname:<26}{row}")
    if never_reached[cname]:
        print(f"{'':<26}WARNING: {never_reached[cname]} task(s) never reached the threshold")

a = cell[("loss: all classes", "argmax")].mean()
b = cell[("loss: all classes", "ncm")].mean()
c_ = cell[("loss: masked to task", "argmax")].mean()
print("\nDECOMPOSITION")
print(f"  total forgetting          : {100 - a:5.1f} points lost from ~100%")
print(f"  recovered by NCM readout  : {b - a:5.1f} points   <- was calibration, not lost code")
print(f"  recovered by masking loss : {c_ - a:5.1f} points   <- was output-layer suppression")
print(f"  residual (hidden drift)   : {100 - b:5.1f} points   <- irreducible by output-layer fixes")
print("\nPAIRED test (same pairings, same seeds -- compare per-run differences, not means):")
dif = cell[("loss: all classes", "ncm")] - cell[("loss: all classes", "argmax")]
print(f"  NCM - argmax per run: mean {dif.mean():+.1f}, sd {dif.std():.1f}, "
      f"wins {int((dif > 0).sum())}/{N_RUNS}")

# ------------------------------- figure -------------------------------
plot_learning_curves(rel_steps, curves, list(curves), FIG,
                     title=("Where does forgetting live? backprop only, 2x2 class-IL, "
                            f"{N_RUNS} pairings, each task trained to {ACC_THRESHOLD:.0%}"),
                     switches=[0], ncols=2,
                     task_labels=["task 1 (old)", "task 2 (new)"],
                     xlabel="updates relative to task switch")

np.savez(FIG.with_suffix(".npz"), rel_steps=rel_steps,
         **{f"curves_{k}": v for k, v in curves.items()},
         pairings=np.array(pairings), threshold=ACC_THRESHOLD)
print(f"\nsaved {FIG.name} and {FIG.with_suffix('.npz').name}")
