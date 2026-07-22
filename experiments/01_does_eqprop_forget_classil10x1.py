"""does_eqprop_forget_classil10x1
Q: Does EqProp forget under Class-IL (10x1)?
   Controls: backprop (should forget), replay (should fix it). Under test: EqProp.
   Runs this one comparison and saves one figure named after this script.
"""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.data import load_mnist, class_indices, make_eval_set
from src.methods import make_backprop, make_replay, make_eqprop

# ============================ constants ============================
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR        = ROOT / "data"                    # <project>/data
FIG             = Path(__file__).resolve().with_suffix(".png")   # figure named after this script
IMG_SIZE        = 14
SEED            = 0
TASKS           = [[c] for c in range(10)]         # 10 tasks, one class each (10x1)
ITERS           = 100                              # weight updates per task
BATCH           = 64
EVAL_EVERY      = 1                               # evaluate every N updates
EVAL_PER_CLASS  = 100                              # held-out TEST images per class for evaluation

BP_LR                          = 0.1               # backprop control
RP_LR, RP_PER_CLASS            = 0.1, 20           # replay control
EQP_LR, EQP_BETA, EQP_DT       = 0.03, 0.3, 0.3    # eqprop (under test)
EQP_MAX_STEPS, EQP_SETTLE_PAT  = 500, 30
# ==================================================================

IN_DIM = IMG_SIZE * IMG_SIZE
train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)
eval_x, eval_y = make_eval_set(test, per_class=EVAL_PER_CLASS, device=DEVICE)

methods = {
    "backprop": make_backprop(in_dim=IN_DIM, lr=BP_LR, seed=SEED, device=DEVICE),
    "replay":   make_replay(train, cidx, in_dim=IN_DIM, lr=RP_LR, per_class=RP_PER_CLASS, seed=SEED, device=DEVICE),
    "eqprop":   make_eqprop(in_dim=IN_DIM, lr=EQP_LR, beta=EQP_BETA, dt=EQP_DT,
                            max_steps=EQP_MAX_STEPS, settle_patience=EQP_SETTLE_PAT, seed=SEED, device=DEVICE),
}

# columns:  cur%  = accuracy on the class(es) being trained right now (is it learning?)
#           seen% = mean accuracy over all classes seen so far   (is it forgetting them?)
#           all%  = mean over all 10 classes  (matches the plotted line)
HDR = f"{'task':>5} {'step':>6} {'cur%':>6} {'seen%':>7} {'all%':>6} {'time':>6}"


def run(name, train_step, predict):
    """Class-IL run for one method; prints a live progress row each eval; returns per-class accuracy history."""
    print(f"\n──── {name} " + "─" * max(2, 44 - len(name)))
    print(HDR)
    torch.manual_seed(SEED)
    hist = {"step": [], "acc": []}
    step, t0 = 0, time.time()
    for ti, task in enumerate(TASKS):
        seen = sorted({c for t in TASKS[:ti + 1] for c in t})
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
                acc = [(pred[eval_y == c] == c).float().mean().item() for c in range(10)]
                hist["step"].append(step); hist["acc"].append(acc)
                cur = np.mean([acc[c] for c in task]) * 100
                seen_pct = np.mean([acc[c] for c in seen]) * 100
                all_pct = np.mean(acc) * 100
                tstr = f"{ti + 1}/{len(TASKS)}"
                print(f"{tstr:>5} {step:>6} {cur:>6.1f} {seen_pct:>7.1f} {all_pct:>6.1f} {time.time() - t0:>5.0f}s")
    return hist


results = {name: run(name, step, pred) for name, (step, pred) in methods.items()}

# --------------------------- summary table ---------------------------
print("\n" + "=" * 46)
print(f"{'method':>9} {'final all%':>12} {'peak all%':>11}")
for name, h in results.items():
    m = np.array(h["acc"]).mean(1) * 100
    print(f"{name:>9} {m[-1]:>12.1f} {m.max():>11.1f}")

# ------------------------------- plot -------------------------------
plt.figure(figsize=(11, 5))
for name, h in results.items():
    plt.plot(h["step"], np.array(h["acc"]).mean(1) * 100, lw=2, label=name)
plt.axhline(10, color="gray", ls=":", label="chance")
for k in range(len(TASKS) + 1):
    plt.axvline(k * ITERS, color="gray", lw=0.3, alpha=0.3)
plt.xlabel("training step")
plt.ylabel("mean accuracy over all classes (%)")
plt.ylim(0, 105)
plt.title("Does EqProp forget? Class-IL 10x1  (backprop / replay / eqprop)")
plt.legend()
plt.tight_layout()
plt.savefig(FIG, dpi=120, bbox_inches="tight")
plt.show()
print(f"\nsaved {FIG.name}")

fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
for ax, (name, h) in zip(np.atleast_1d(axes), results.items()):
    A = np.array(h["acc"]) * 100
    for c in range(10):
        ax.plot(h["step"], A[:, c], lw=0.8, alpha=0.6)
    ax.plot(h["step"], A.mean(1), "k-", lw=2)
    ax.set_title(name); ax.set_xlabel("step"); ax.set_ylim(0, 105)
axes[0].set_ylabel("per-class accuracy (%)")
plt.tight_layout(); plt.savefig(FIG.with_name(FIG.stem + "_perclass.png"), dpi=120); plt.show()