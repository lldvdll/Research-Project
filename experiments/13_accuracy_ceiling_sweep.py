"""13_accuracy_ceiling_sweep
Q: What accuracy is achievable on MNIST 14x14 as a function of hidden-layer size and the
   number of classes in the problem? Sets the ceiling against which every continual-learning
   result must be read.

Backprop only, JOINT training (no task structure). For each (hidden, n_classes) cell we draw
N_REPEATS random class subsets and train on each. Two readings from the SAME run:
  - accuracy at BUDGET_ITERS  : what is reachable in the per-task budget the CL runs use
  - best accuracy             : the true ceiling for that architecture / problem size
Note: accuracy for an n-class problem is over those n classes, so chance is 100/n --
cells are NOT comparable across columns as difficulty; each is its own problem.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import load_mnist, class_indices, make_eval_set
from src.methods import build_method, legacy
from src.runner import run_joint
from src.plotting import plot_heatmap

# ============================ constants ============================
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR      = ROOT / "data"
FIG           = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE      = 14
BASE_SEED     = 0

HIDDEN_SIZES  = [16, 32, 64, 128, 256]      # spans below and above the expected knee
N_CLASSES     = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # 1 is degenerate (a constant predictor scores 100%)
N_REPEATS     = 5                            # random class subsets per cell

BUDGET_ITERS  = 100                          # matches iters-per-task in the CL experiments
MAX_ITERS     = 800                          # cap for the convergence reading
BATCH         = 32
EVAL_EVERY    = 10
CONV_PATIENCE = 15                           # stop early once accuracy plateaus
BP_LR         = 0.05

IN_DIM = IMG_SIZE * IMG_SIZE
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)

budget = np.full((len(HIDDEN_SIZES), len(N_CLASSES), N_REPEATS), np.nan)
best   = np.full_like(budget, np.nan)
t_start = time.time()

for i, hidden in enumerate(HIDDEN_SIZES):
    for j, n_cls in enumerate(N_CLASSES):
        for r in range(N_REPEATS):
            seed = BASE_SEED + 1000 * i + 100 * j + r
            classes = sorted(np.random.default_rng(seed).permutation(10)[:n_cls].tolist())
            eval_x, eval_y = make_eval_set(test, classes=classes, per_class=100, device=DEVICE)
            # **legacy(...) pins the pre-unification specification this script was written
            # against; the library default is now the unified protocol. Not for new work.
            step_fn, pred_fn = build_method("backprop", in_dim=IN_DIM, hidden=hidden,
                                            lr=BP_LR, seed=seed, device=DEVICE,
                                            **legacy("backprop"))
            steps, accs = run_joint(step_fn, pred_fn, classes, train, cidx, eval_x, eval_y,
                                    max_iters=MAX_ITERS, batch=BATCH, eval_every=EVAL_EVERY,
                                    device=DEVICE, stop_patience=CONV_PATIENCE)
            at_budget = accs[steps <= BUDGET_ITERS]
            budget[i, j, r] = at_budget.max() if at_budget.size else np.nan
            best[i, j, r] = accs.max()
        print(f"hidden {hidden:>4} | {n_cls:>2} classes | "
              f"@{BUDGET_ITERS} {np.nanmean(budget[i, j]) * 100:5.1f}% | "
              f"best {np.nanmean(best[i, j]) * 100:5.1f}% | {time.time() - t_start:5.0f}s")

# ------------------------------- tables -------------------------------
print("\n" + "=" * 70)
print(f"CEILING (best accuracy %), mean over {N_REPEATS} random class subsets")
print("hidden".rjust(7) + "".join(f"{n:>8}" for n in N_CLASSES) + "   <- n classes")
for i, hidden in enumerate(HIDDEN_SIZES):
    print(f"{hidden:>7}" + "".join(f"{np.nanmean(best[i, j]) * 100:>8.1f}" for j in range(len(N_CLASSES))))

# ------------------------------- figures -------------------------------
plot_heatmap(np.nanmean(best, axis=2) * 100, np.nanstd(best, axis=2) * 100,
             HIDDEN_SIZES, N_CLASSES, FIG,
             title=f"Accuracy ceiling, MNIST {IMG_SIZE}x{IMG_SIZE} (backprop, converged)",
             row_name="hidden units", col_name="number of classes",
             cbar_label="best accuracy (%)")

plot_heatmap(np.nanmean(budget, axis=2) * 100, np.nanstd(budget, axis=2) * 100,
             HIDDEN_SIZES, N_CLASSES, FIG.with_name(FIG.stem + "_at_budget.png"),
             title=f"Accuracy within {BUDGET_ITERS} updates (the per-task CL budget)",
             row_name="hidden units", col_name="number of classes",
             cbar_label="accuracy (%)")

np.savez(FIG.with_suffix(".npz"), budget=budget, best=best,
         hidden=HIDDEN_SIZES, n_classes=N_CLASSES)
print(f"saved raw results to {FIG.with_suffix('.npz').name}")
