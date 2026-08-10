"""10_pc_learning_vs_forgetting
Q: When task 2 arrives, how fast does predictive coding learn it and how fast does it lose task 1?
   Controls: backprop (forgets), replay (retains). Under test: predictive coding (the model whose literature claims reduced interference).
   Batch raised back to 32 to remove the batch-1 gradient noise; slowdown comes from lr instead.
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
from src.methods import make_backprop, make_replay, make_pc, legacy

# ============================ constants ============================
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR        = ROOT / "data"
FIG             = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE        = 14
SEED            = 0

TASKS           = [[0, 1], [2, 3]]     # two tasks, two classes each: enough to see one clean crossing
ITERS           = 100                  # updates per task
BATCH           = 32                   # >1: averages out the per-example tug-of-war
EVAL_EVERY      = 1
EVAL_PER_CLASS  = 100

BP_LR                          = 0.05
RP_LR, RP_PER_CLASS            = 0.05, 20
PC_LR, PC_DT, PC_STEPS          = 0.05, 0.1, 50

CLASSES  = sorted({c for t in TASKS for c in t})
CLS_POS  = {c: i for i, c in enumerate(CLASSES)}
COLLAPSE = 100 / len(CLASSES)
TITLE    = f"Class-IL {len(TASKS)}x{len(TASKS[0])}"
# ==================================================================

IN_DIM = IMG_SIZE * IMG_SIZE
train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)
eval_x, eval_y = make_eval_set(test, classes=CLASSES, per_class=EVAL_PER_CLASS, device=DEVICE)

# **legacy(...) pins the pre-unification specification this script was written against; the
# library default is now the unified protocol. See src/methods.py legacy(). Not for new work.
methods = {
    "backprop": make_backprop(in_dim=IN_DIM, lr=BP_LR, seed=SEED, device=DEVICE,
                              **legacy("backprop")),
    "replay":   make_replay(train, cidx, in_dim=IN_DIM, lr=RP_LR, per_class=RP_PER_CLASS,
                            seed=SEED, device=DEVICE, **legacy("replay")),
    "pc":       make_pc(in_dim=IN_DIM, lr=PC_LR, dt=PC_DT, steps=PC_STEPS,
                        seed=SEED, device=DEVICE, **legacy("pc")),
}

HDR = f"{'task':>5} {'step':>6} " + " ".join(f"{'T'+str(i+1)+'%':>6}" for i in range(len(TASKS))) + f" {'time':>6}"


def run(name, train_step, predict):
    """Class-IL run. Logs per-CLASS accuracy, per-TASK accuracy (fixed denominators),
       and the mean raw output per class (which moves before accuracy does)."""
    print(f"\n──── {name} " + "─" * max(2, 44 - len(name)))
    print(HDR)
    torch.manual_seed(SEED)
    hist = {"step": [], "acc": [], "task_acc": [], "out": []}
    step, t0 = 0, time.time()
    for ti, task in enumerate(TASKS):
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
                raw = predict(eval_x, raw=True)
                acc = [(pred[eval_y == c] == c).float().mean().item() for c in CLASSES]
                task_acc = [float(np.mean([acc[CLS_POS[c]] for c in t])) for t in TASKS]
                out_mean = [raw[:, c].mean().item() for c in CLASSES]   # mean output unit value per class
                hist["step"].append(step); hist["acc"].append(acc)
                hist["task_acc"].append(task_acc); hist["out"].append(out_mean)
                cols = " ".join(f"{a*100:>6.1f}" for a in task_acc)
                print(f"{ti+1:>5} {step:>6} {cols} {time.time()-t0:>5.0f}s")
    return hist


results = {name: run(name, s, p) for name, (s, p) in methods.items()}

# ------------------- learning / forgetting crossing table -------------------
def crossing(steps, series, thresh, after, rising):
    for s, v in zip(steps, series):
        if s <= after:
            continue
        if (v >= thresh) if rising else (v < thresh):
            return s - after
    return None

print("\n" + "=" * 62)
print(f"{'method':>9} {'to_learn_T2':>12} {'to_forget_T1':>13} {'ratio':>7} {'final_T1':>9} {'final_T2':>9}")
for name, h in results.items():
    T, steps, t0 = np.array(h["task_acc"]), h["step"], ITERS
    L = crossing(steps, T[:, 1], 0.5, t0, True)     # steps for task 2 to reach 50%
    F = crossing(steps, T[:, 0], 0.5, t0, False)    # steps for task 1 to fall below 50%
    r = (F / L) if (L and F) else float("nan")
    print(f"{name:>9} {str(L):>12} {str(F):>13} {r:>7.2f} {T[-1,0]*100:>8.1f}% {T[-1,1]*100:>8.1f}%")

# ------------------------------ figure 1: crossings ------------------------------
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
axes = np.atleast_1d(axes)
for ax, (name, h) in zip(axes, results.items()):
    T = np.array(h["task_acc"]) * 100
    for i in range(len(TASKS)):
        ax.plot(h["step"], T[:, i], lw=1.8, label=f"task {i+1} = {TASKS[i]}")
    ax.axvline(ITERS, color="k", lw=0.8, ls="--")
    ax.axhline(COLLAPSE, color="gray", ls=":", lw=1)
    ax.set_title(name); ax.set_xlabel("training step"); ax.set_ylim(-2, 103)
axes[0].set_ylabel("accuracy on task's classes (%)"); axes[0].legend(fontsize=8)
fig.suptitle(f"Learning vs forgetting at the task switch — {TITLE}")
plt.tight_layout(); plt.savefig(FIG, dpi=120, bbox_inches="tight"); plt.show()

# ------------------ figure 2: trajectory + raw outputs ------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for name, h in results.items():
    T = np.array(h["task_acc"]) * 100
    ax[0].plot(T[:, 0], T[:, 1], "-", lw=1.5, alpha=0.8, label=name)
    ax[0].plot(T[-1, 0], T[-1, 1], "o", ms=7)
ax[0].plot([100, 0], [0, 100], color="gray", ls=":", lw=1, label="pure trade-off")
ax[0].set_xlabel("task 1 accuracy (%)"); ax[0].set_ylabel("task 2 accuracy (%)")
ax[0].set_xlim(-2, 103); ax[0].set_ylim(-2, 103)
ax[0].set_title("trajectory (dot = end); up-right = retains both"); ax[0].legend(fontsize=8)

h = results["pc"]
O = np.array(h["out"])
for i, c in enumerate(CLASSES):
    ax[1].plot(h["step"], O[:, i], lw=1.4, label=f"class {c}")
ax[1].axvline(ITERS, color="k", lw=0.8, ls="--")
ax[1].set_xlabel("training step"); ax[1].set_ylabel("mean output unit value")
ax[1].set_title("pc raw outputs (move before accuracy flips)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG.with_name(FIG.stem + "_traj_outputs.png"), dpi=120); plt.show()
print(f"\nsaved {FIG.name} and {FIG.stem}_traj_outputs.png")
