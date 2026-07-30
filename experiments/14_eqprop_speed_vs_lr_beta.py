"""14_eqprop_speed_vs_lr_beta
Q: Which EqProp learning rate matches backprop's LEARNING SPEED, and does beta affect speed?

Motivation: EqProp currently needs lr=0.005 where backprop uses 0.05. Those numbers are not
comparable units -- cross-entropy and contrastive-energy gradients have different natural
scales -- so the meaningful thing to match is steps-to-reach-a-fixed-accuracy, not the lr value.

Why beta is here as a second axis, not the primary knob: the EqProp update is
    (grad_nudged - grad_free) / (beta * batch)
and the numerator scales ~linearly with beta, so to FIRST ORDER beta cancels. beta is a
finite-difference step size controlling the BIAS of the gradient estimate, not a speed knob.
Prediction: the steps-to-target heatmap shows vertical bands (varies with lr, flat along beta).
If it does not, that prediction is wrong and worth knowing.

Joint training on all 10 classes, no continual learning. Set TARGET_ACC from experiment 13
(pick something comfortably under the 10-class ceiling).
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import load_mnist, class_indices, make_eval_set
from src.methods import build_method
from src.runner import run_joint
from src.plotting import plot_heatmap

# ============================ constants ============================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR     = ROOT / "data"
FIG          = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE     = 14
SEED         = 0

HIDDEN       = 64
CLASSES      = list(range(10))
TARGET_ACC   = 0.90          # <- set from experiment 13 (below the 10-class ceiling)

LRS          = [0.005, 0.01, 0.02, 0.05, 0.1]
BETAS        = [0.1, 0.3, 0.5]
EQP_DT       = 0.3
EQP_MAX_STEPS, EQP_SETTLE_PAT = 500, 30

MAX_ITERS    = 400           # cap; combos that never reach TARGET_ACC are recorded as "-"
BATCH        = 32
EVAL_EVERY   = 10            # eqprop eval settles the whole eval set, so keep this coarse
EVAL_PER_CLASS = 50          # smaller eval set than usual, for speed
BP_LR        = 0.05          # reference: backprop's speed to the same target

IN_DIM = IMG_SIZE * IMG_SIZE
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)
eval_x, eval_y = make_eval_set(test, classes=CLASSES, per_class=EVAL_PER_CLASS, device=DEVICE)


def steps_to_target(steps, accs):
    """First eval step at which accuracy reached TARGET_ACC, else NaN."""
    hit = np.where(np.asarray(accs) >= TARGET_ACC)[0]
    return float(steps[hit[0]]) if hit.size else np.nan


# ---------------------- reference: backprop speed ----------------------
step_fn, pred_fn = build_method("backprop", in_dim=IN_DIM, hidden=HIDDEN, lr=BP_LR,
                                seed=SEED, device=DEVICE)
s, a = run_joint(step_fn, pred_fn, CLASSES, train, cidx, eval_x, eval_y,
                 max_iters=MAX_ITERS, batch=BATCH, eval_every=EVAL_EVERY,
                 device=DEVICE, stop_at=TARGET_ACC)
bp_steps = steps_to_target(s, a)
print(f"reference: backprop (lr={BP_LR}) reached {TARGET_ACC:.0%} in {bp_steps} steps "
      f"(best {a.max():.1%})\n")

# ---------------------------- eqprop grid ----------------------------
speed = np.full((len(LRS), len(BETAS)), np.nan)
final = np.full((len(LRS), len(BETAS)), np.nan)
t_start = time.time()
for i, lr in enumerate(LRS):
    for j, beta in enumerate(BETAS):
        step_fn, pred_fn = build_method("eqprop", in_dim=IN_DIM, hidden=HIDDEN, lr=lr, beta=beta,
                                        dt=EQP_DT, max_steps=EQP_MAX_STEPS,
                                        settle_patience=EQP_SETTLE_PAT, seed=SEED, device=DEVICE)
        s, a = run_joint(step_fn, pred_fn, CLASSES, train, cidx, eval_x, eval_y,
                         max_iters=MAX_ITERS, batch=BATCH, eval_every=EVAL_EVERY,
                         device=DEVICE, stop_at=TARGET_ACC)
        speed[i, j] = steps_to_target(s, a)
        final[i, j] = a.max()
        got = "never" if np.isnan(speed[i, j]) else f"{speed[i, j]:.0f} steps"
        print(f"lr {lr:<6} beta {beta:<4} -> {TARGET_ACC:.0%} in {got:>11} | "
              f"best {a.max():5.1%} | {time.time() - t_start:5.0f}s")

# ------------------------------- tables -------------------------------
print("\n" + "=" * 70)
print(f"STEPS TO REACH {TARGET_ACC:.0%}   (backprop reference: {bp_steps})")
print("lr".rjust(8) + "".join(f"{b:>10}" for b in BETAS) + "   <- beta")
for i, lr in enumerate(LRS):
    row = "".join(("     -    " if np.isnan(speed[i, j]) else f"{speed[i, j]:>10.0f}")
                  for j in range(len(BETAS)))
    print(f"{lr:>8}" + row)
finite = speed[np.isfinite(speed)]
if finite.size:
    ii, jj = np.unravel_index(np.nanargmin(np.abs(speed - bp_steps)), speed.shape)
    print(f"\nclosest match to backprop speed: lr={LRS[ii]}, beta={BETAS[jj]} "
          f"({speed[ii, jj]:.0f} steps vs {bp_steps})")
print("\nIf the columns are near-identical, beta does not control speed (as predicted);"
      "\nlr is the knob for matching learning speed across methods.")

# ------------------------------- figures -------------------------------
plot_heatmap(speed, np.full_like(speed, np.nan), LRS, BETAS, FIG,
             title=f"EqProp: updates to reach {TARGET_ACC:.0%} (backprop = {bp_steps:.0f})",
             row_name="learning rate", col_name="beta (nudge)",
             cbar_label="steps to target", fmt="{:.0f}", cmap="viridis_r")

plot_heatmap(final * 100, np.full_like(final, np.nan), LRS, BETAS,
             FIG.with_name(FIG.stem + "_best_acc.png"),
             title=f"EqProp: best accuracy within {MAX_ITERS} updates",
             row_name="learning rate", col_name="beta (nudge)",
             cbar_label="best accuracy (%)")

np.savez(FIG.with_suffix(".npz"), speed=speed, final=final, lrs=LRS, betas=BETAS,
         bp_steps=bp_steps, target=TARGET_ACC)
print(f"saved raw results to {FIG.with_suffix('.npz').name}")
