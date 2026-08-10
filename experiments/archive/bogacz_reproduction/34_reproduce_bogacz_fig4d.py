"""34_reproduce_bogacz_fig4d

ONE QUESTION
    At the learning rate each rule prefers, does predictive coding show lower test error than
    backpropagation during Song & Bogacz's two-task continual-learning schedule?

    Target: Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz (2024), Nature Neuroscience
    27:348-358, Figure 4d.  https://doi.org/10.1038/s41593-023-01514-1
    Config:  experiments/nature_forgetting/base-shuffle-task-5-FashionMNIST.yaml

    Fig 4d ONLY. No learning-rate sweep -- LR below is fixed at the value that minimised
    their metric in exp 32, which sat interior to their published grid for both rules. That
    cuts this from eighty runs to twenty. If the result looks wrong, the learning rate is the
    first thing to re-open, and a sweep is one edit away (make LR a dict of lists and loop).

SUPERSEDES exps 30, 32 and 33. Do not cite those: 30 read partial_num as a total rather than
per class, and 30/32 both paired classes to output units by RANK rather than at random, which
correlated each unit's two classes at +0.94 and suppressed the interference being measured.

THEIR CONFIGURATION -- every value traced to the yaml
    784-32-32-32-5   num_layers 4 with structure [Linear, PCLayer, Acf] -> four Linear layers,
                     hidden_size 32, and FIVE outputs shared by both tasks. Sharing the output
                     layer is what makes this DOMAIN-incremental, not class-incremental.
    sigmoid, xavier_normal (gain 1.0), bias False
    partial_num 600 PER CLASS -> 3000 per task -> six 500-example batches per iteration
    one iteration = one epoch over that task's subset; 4 iterations per task; 160 total
    their analysis then takes [:84], which is what ANALYSE_ITERS reproduces
    loss  0.5 * sum((out - target)^2), summed over batch AND outputs, no division
    T 64 inference steps, optimizer_x SGD lr 0.1, x_lr_discount 0.9, x_lr_amplifier 1.0
    backprop = the same network with the PC layers removed, i.e. T = 1

    LOSS_ACTIVE: their learn_code slices (outputs - target)[:, 0:-1], which excludes the last
    output unit from the loss entirely -- one class per task never receives a gradient at its
    own unit, yet is still argmaxed over at test time. It reads like an off-by-one that
    reached the published figure. Reproducing it COLLAPSES the model (see the LOSS_ACTIVE
    note in the constants), so the default here is all five units, as the paper describes.
    Set LOSS_ACTIVE = [0,1,2,3] to run their literal slice deliberately.

REMAINING DEVIATIONS
    1. Their seed list has 21 entries; the paper reports n = 10. We use seeds 0-9.
    2. partial_dateset takes the FIRST partial_num per class; we take a seeded random subset.
    3. They log each task in a separate trial and merge; we log both in one pass, which is
       equivalent at the same seed.
    4. No learning-rate sweep, as above.

WHAT IS RECORDED
    per_iter[rule, seed, iteration, task] -- test error on the full test set of BOTH tasks
    after every one of the 160 iterations, at their logging resolution. Any later metric can
    be computed from the saved file without re-running.

INTERPRETATION
    reproduced       PC below backprop in the curve and in the paired per-seed difference,
                     with the 68% CI excluding zero.
    not reproduced   before calling it a failure, check in this order:
                     (a) has the error flattened by iteration 84, or is it still descending
                         (exp 32 was undertrained and the alternation was a ripple on a slope)
                     (b) does the trained task fall and the untrained task rise within a block
                     (c) is LR still the right value now that the pairing bug is fixed
"""
import sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data import load_mnist, class_indices, make_domain_il_eval
from src.model import Arch, Objective
from src.methods import build_method
from src.runner import run_alternating
from src.metrics import bootstrap_ci

# ============================ constants ============================
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data"
FIG      = Path(__file__).resolve().with_suffix(".png")

IMG_SIZE, FASHION = 28, True
HIDDEN, OUT_DIM   = (32, 32, 32), 5
ACT, INIT, BIAS   = "sigmoid", "xavier_normal", False

PARTIAL_NUM    = 600      # PER CLASS -> 3000/task -> six 500-example batches per iteration
BATCH          = 500
ITERS_PER_TASK = 4
TOTAL_ITERS    = 160
ANALYSE_ITERS  = 84
SMOKE          = False    # True -> 1 seed, 16 iterations: every code path in ~30s
EVAL_PER_CLASS = 1000     # the whole Fashion-MNIST test set for each class

T_INFER, X_LR           = 64, 0.1
X_LR_DISCOUNT, X_LR_AMP = 0.9, 1.0

# Fixed, from exp 32's minimum. Both sat interior to their published grid
# [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.005]. Caveat worth keeping in mind: that
# minimum was found while the class-pairing bug was still present, so the optimum could
# have shifted. It is the first thing to re-open if the result looks wrong.
LR = {"backprop": 0.001, "pc": 0.001}

N_SEEDS, BASE_SEED = 10, 0
if SMOKE:
    N_SEEDS, TOTAL_ITERS = 1, 16
ANALYSE = min(ANALYSE_ITERS, TOTAL_ITERS)   # never slice past what was actually run
RULES = ["backprop", "pc"]
DIV_CHECK, DIV_FLOOR = 40, 0.795   # chance error is 0.80; 40 iterations above 0.795 is dead

ARCH = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=HIDDEN, out_dim=OUT_DIM,
            act=ACT, bias=BIAS, init=INIT)
# Their learn_code slices (outputs - target)[:, 0:-1], excluding the LAST output unit from
# the loss. Reproducing that collapses the model: unit 4 keeps its random initial weights
# while units 0-3 are trained toward one-hot targets and settle near 0.2, so whenever unit 4's
# random output happens to exceed that it wins the argmax on EVERY input. The network then
# answers one class forever and the error pins at exactly 1 - 1/5 = 0.8000 -- which is what
# the first smoke run showed for backprop, and which roughly 40% of seeds would hit. Their
# published figure is plainly not degenerate, so either something else in their pipeline
# compensates or the slice is an off-by-one their learning rates happened to escape. A figure
# cannot be reproduced by reproducing a pathology, so the default is the version the PAPER
# describes.
#   None      -> all five units contribute to the loss. DEFAULT.
#   [0,1,2,3] -> their literal slice, kept so the comparison can be run on purpose.
LOSS_ACTIVE = None
OBJ = Objective(loss="mse", target="onehot", mask=LOSS_ACTIVE is not None, reduction="sum")

COL = {"backprop": "tab:red", "pc": "tab:blue"}
LABEL = {"backprop": "backpropagation", "pc": "prospective configuration"}
# ==================================================================


def make_tasks(seed):
    """Their shuffle_mapper: permute all ten labels, split 5/5, position j -> output unit j%5.

    NOT sorted. Sorting pairs the i-th smallest class of task 1 with the i-th smallest of
    task 2, which on Fashion-MNIST (0-4 garments, 5-9 footwear and bags) correlates the two
    classes on each unit at +0.94 and partly makes task 2 REINFORCE task 1."""
    perm = np.random.default_rng(seed).permutation(10).tolist()
    return [perm[:5], perm[5:]]


def build(rule, seed):
    extra = dict(dt=X_LR, steps=T_INFER, x_lr_discount=X_LR_DISCOUNT,
                 x_lr_amplifier=X_LR_AMP) if rule == "pc" else {}
    return build_method(rule, in_dim=ARCH.in_dim, hidden=HIDDEN, out_dim=OUT_DIM,
                        arch=ARCH, obj=OBJ, lr=LR[rule], seed=seed, device=DEVICE, **extra)


train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR), fashion=FASHION)
cidx = class_indices(train)

t0 = make_tasks(BASE_SEED)
print(f"device: {DEVICE}   torch threads: {torch.get_num_threads()}")
print(f"learning rate: " + ", ".join(f"{r}={LR[r]}" for r in RULES))
print(f"loss covers output units {LOSS_ACTIVE if LOSS_ACTIVE else 'all'} "
      f"of {OUT_DIM} (their [:, 0:-1] slice)")
print(f"seed {BASE_SEED} pairing check -- unit: task-1 class / task-2 class")
print("   " + "   ".join(f"u{i}: {t0[0][i]}/{t0[1][i]}" for i in range(OUT_DIM)))
print("   (should look random; rank-ordered pairs would mean the sort bug is back)\n")

per_iter = np.full((len(RULES), N_SEEDS, TOTAL_ITERS, 2), np.nan)
class_sets = np.zeros((N_SEEDS, 2, OUT_DIM), dtype=int)
t_start = time.time()

for si in range(N_SEEDS):
    seed = BASE_SEED + si
    tasks = make_tasks(seed)
    class_sets[si] = np.array(tasks)
    ev = make_domain_il_eval(test, tasks, per_class=EVAL_PER_CLASS, device=DEVICE)
    for ri, rule in enumerate(RULES):
        step_fn, predict = build(rule, seed)
        out = run_alternating(step_fn, predict, tasks, train, cidx, ev,
                              iters_per_task=ITERS_PER_TASK, total_iters=TOTAL_ITERS,
                              batch=BATCH, partial_num=PARTIAL_NUM, device=DEVICE,
                              data_seed=seed, eval_per_update=False,
                              loss_active=LOSS_ACTIVE,
                              divergence_check=DIV_CHECK, divergence_floor=DIV_FLOOR)
        per_iter[ri, si] = out["per_iter"]
    el = time.time() - t_start
    print(f"  seed {si + 1}/{N_SEEDS}  task1={tasks[0]} task2={tasks[1]}  "
          f"elapsed {el:5.0f}s  eta {el / (si + 1) * (N_SEEDS - si - 1):5.0f}s")

# ------------------------------- numbers -------------------------------
win = per_iter[:, :, :ANALYSE, :]
hm = np.nanmean(win, axis=(2, 3))                                  # [rule, seed]

print("\n" + "=" * 84)
print(f"MEAN TEST ERROR over the first {ANALYSE} iterations, both tasks "
      f"({N_SEEDS} seeds, 68% bootstrap CI)")
print(f"{'rule':>30}{'lr':>10}{'mean err':>24}{'final t1':>11}{'final t2':>11}")
for ri, r in enumerate(RULES):
    m, lo, hi = bootstrap_ci(hm[ri])
    print(f"{LABEL[r]:>30}{LR[r]:>10}{m:>14.4f} [{lo:.3f},{hi:.3f}]"
          f"{np.nanmean(win[ri, :, -1, 0]):>11.4f}{np.nanmean(win[ri, :, -1, 1]):>11.4f}")

FLOOR = 1.0 - 1.0 / OUT_DIM     # 0.8 -- the score of a model answering one class always
for _ri, _r in enumerate(RULES):
    pinned = float(np.mean(np.abs(win[_ri] - FLOOR) < 1e-6))
    if pinned > 0.5:
        print(f"\n  DEGENERATE: {LABEL[_r]} sits at exactly {FLOOR:.4f} for {pinned:.0%} of"
              f" iterations.\n  That is the COLLAPSE FLOOR, not chance -- the model answers"
              f" one class on every input.\n  Do not interpret any comparison against it."
              f" Check LOSS_ACTIVE and the learning rate.")

diff = hm[RULES.index("backprop")] - hm[RULES.index("pc")]
dm, dlo, dhi = bootstrap_ci(diff)
print(f"\nPAIRED backprop minus pc: {dm:+.4f}  68% CI [{dlo:+.4f}, {dhi:+.4f}]  "
      f"pc better in {int((diff > 0).sum())}/{N_SEEDS} seeds")
print("positive = prospective configuration has the LOWER error, i.e. the claim reproduces")

# convergence: is 84 iterations enough, or is this still descending like exp 32 was?
print(f"\nCONVERGENCE -- slope over the last 20 iterations, in points per 10 iterations")
for ri, r in enumerate(RULES):
    def slope(y):
        w = min(20, len(y))          # a smoke run has fewer points than the 20-wide window
        return float("nan") if w < 3 else np.polyfit(np.arange(w), y[-w:], 1)[0] * 1000
    s84 = slope(np.nanmean(per_iter[ri, :, :ANALYSE, :], axis=(0, 2)))
    s160 = slope(np.nanmean(per_iter[ri, :, :, :], axis=(0, 2)))
    print(f"{LABEL[r]:>30}   at iter {ANALYSE}: {s84:+6.2f}   "
          f"at iter {TOTAL_ITERS}: {s160:+6.2f}")
print("  near zero = converged. Strongly negative = still learning, so the task alternation")
print("  is a ripple on a descent and the sawtooth will not be visible directly.")

# the trade-off, with the common descent removed: every block aligned and averaged
print(f"\nBLOCK-ALIGNED TRADE-OFF -- change across one {ITERS_PER_TASK}-iteration block, "
      f"averaged over every block and seed")
print(f"{'rule':>30}{'trained task':>16}{'untrained task':>17}{'exchange':>11}")
for ri, r in enumerate(RULES):
    segs = []
    for s in range(N_SEEDS):
        for b in range(TOTAL_ITERS // ITERS_PER_TASK):
            ti = b % 2
            seg = per_iter[ri, s, b * ITERS_PER_TASK:(b + 1) * ITERS_PER_TASK, :]
            segs.append(np.stack([seg[:, ti], seg[:, 1 - ti]], axis=1))
    m = np.nanmean(np.array(segs), axis=0) * 100
    dtr, dun = m[-1, 0] - m[0, 0], m[-1, 1] - m[0, 1]
    print(f"{LABEL[r]:>30}{dtr:>+13.2f} pts{dun:>+14.2f} pts{dun - dtr:>+8.2f}")
print("  trained should FALL and untrained should RISE within a block. If untrained also")
print("  falls, both tasks are still improving together and the run is simply too early.")

# ------------------------------- figure -------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
for ri, r in enumerate(RULES):
    for task, ls in ((0, "-"), (1, "--")):
        A = per_iter[ri, :, :ANALYSE, task]
        m = np.nanmean(A, 0); se = np.nanstd(A, 0) / np.sqrt(N_SEEDS)
        xs = np.arange(ANALYSE)
        ax.plot(xs, m, ls, color=COL[r], lw=2.0, label=f"{LABEL[r]}, task {task + 1}")
        ax.fill_between(xs, m - se, m + se, color=COL[r], alpha=0.13)
for b in range(ITERS_PER_TASK, ANALYSE, ITERS_PER_TASK):
    ax.axvline(b, color="k", lw=0.4, alpha=0.2)
ax.set_xlabel("iteration"); ax.set_ylabel("test error")
ax.grid(alpha=0.2); ax.legend(fontsize=9)
ax.set_title("Song & Bogacz (2024) Fig 4d reproduced from their configuration\n"
             f"Fashion-MNIST, {ARCH.widths}, sigmoid, no bias, {PARTIAL_NUM}/class, "
             f"batch {BATCH}, summed loss, T={T_INFER}, lr {LR['pc']}, {N_SEEDS} seeds",
             fontsize=10)
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}   total {time.time() - t_start:.0f}s")

cfg = dict(widths=list(ARCH.widths), act=ACT, init=INIT, bias=BIAS, out_dim=OUT_DIM,
           partial_num_per_class=PARTIAL_NUM, batch=BATCH, iters_per_task=ITERS_PER_TASK,
           total_iters=TOTAL_ITERS, analyse_iters=ANALYSE_ITERS, eval_per_class=EVAL_PER_CLASS,
           T=T_INFER, x_lr=X_LR, x_lr_discount=X_LR_DISCOUNT, x_lr_amplifier=X_LR_AMP,
           reduction="sum", loss_active=LOSS_ACTIVE, lr=LR, rules=RULES, n_seeds=N_SEEDS,
           source="base-shuffle-task-5-FashionMNIST.yaml", supersedes=["30", "32", "33"])
np.savez_compressed(FIG.with_suffix(".npz"), per_iter=per_iter, classes=class_sets,
                    rules=np.array(RULES), config=json.dumps(cfg))
print(f"saved {FIG.with_suffix('.npz').name}")
print("  per_iter[rule, seed, iteration, task] -- the complete record, both tasks, "
      "every iteration")
