"""33_reproduce_bogacz_fig4de_fixed

ONE QUESTION
    Under Song & Bogacz's configuration, correctly implemented, does predictive coding show
    lower test error during continual learning than backpropagation?

    This is the version to keep. Exps 30 and 32 are superseded and their results should not
    be cited: 30 used 1/3 of the training data per iteration, and both used a class-to-output
    mapping that suppressed interference (see BUG below).

WHAT CHANGED SINCE 32, AND WHY
    BUG -- class-to-output pairing.  Their config shuffles all ten labels and then maps with
        `i % 5`, so which two classes share an output unit is random. We sorted each task and
        paired by rank, so unit i always held the i-th smallest class of task 1 and the i-th
        smallest of task 2. Fashion-MNIST's class order is semantically structured (0-4
        garments, 5-9 footwear and bags), so rank-pairing made each unit's two classes more
        alike than chance. When they are alike, training task 2 partly REINFORCES task 1
        rather than overwriting it, which suppresses the interference the experiment exists
        to measure. Both prior runs are contaminated by this.
        Fixed in src/data.py: label_remapper now honours the order it is given, and the task
        lists below are NOT sorted.

    SPEED -- evaluation was almost all of the wall-clock. 5000 test images x 2 tasks after
        every one of ~960 weight updates is ~10 million image-forwards per run, x 120 runs.
        Two changes, together roughly an order of magnitude:
          phase A (the learning-rate sweep, Fig 4e) evaluates once per iteration -- exactly
            their logging frequency -- on a 200-per-class test subset. The sweep only uses
            the mean over iterations, so nothing is lost.
          phase B (the headline curve, Fig 4d) runs the winning learning rate only, on the
            full test set, at all ten seeds.
        Exp 31 established that the per-update panel added nothing the block-aligned analysis
        of per-iteration data did not already show, so paying for it across the whole grid
        was never justified.

CONFIGURATION -- every value from base-shuffle-task-5-FashionMNIST.yaml
    784-32-32-32-5   num_layers 4, hidden_size 32, five SHARED outputs (Domain-IL)
    sigmoid, xavier_normal, bias False
    partial_num 600 PER CLASS -> 3000 per task -> six 500-example batches per iteration
    one iteration = one epoch; 4 iterations per task; 160 total; analysis takes [:84]
    loss 0.5 * sum((out - target)^2), SUMMED over batch and outputs
    T 64, optimizer_x SGD lr 0.1, x_lr_discount 0.9, x_lr_amplifier 1.0
    optimizer_p SGD, lr grid 0.0001 - 0.005
    backprop = the same network with the PC layers removed, i.e. T = 1

REMAINING DEVIATIONS
    1. Their seed list has 21 entries; the paper reports n = 10. We take the first 10.
    2. They run each task's logging as a separate trial and merge; we log both in one pass,
       which is equivalent at the same seed.
    3. partial_dateset takes the FIRST partial_num per class; we take a seeded random subset.

INTERPRETATION
    reproduced       PC below backprop in 4d and a lower 4e minimum, with the paired seed
                     difference excluding zero.
    not reproduced   check, in order: is the winning lr interior to the grid; has the error
                     flattened by iteration 84 (exp 31 panel A on the new npz); does the
                     block-aligned trade-off index rise. Only then call it a failure.
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

PARTIAL_NUM    = 600      # PER CLASS -> 3000/task -> 6 batches of 500 per iteration
BATCH          = 500
ITERS_PER_TASK = 4
TOTAL_ITERS    = 160
ANALYSE_ITERS  = 84

T_INFER, X_LR          = 64, 0.1
X_LR_DISCOUNT, X_LR_AMP = 0.9, 1.0

LRS = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.005]
N_SEEDS, BASE_SEED = 10, 0
N_SEEDS_SWEEP = 5     # the lr sweep needs fewer seeds than the headline curve
RULES = ["backprop", "pc"]

EVAL_SWEEP = 200          # phase A: test images per class (sweep only uses the mean)
EVAL_FULL_NOTE = None
# Early-stop for learning rates that have diverged and sit pinned at chance. Chance error on
# five classes is 0.80. The defaults (20 iterations above 0.78) are too loose here: the
# SLOWEST honest learning rate, 1e-4, is still near 0.78 around iteration 20 and would be
# killed, silently deleting a real point from the Fig 4e curve. 40 iterations above 0.795 is
# genuinely dead.
DIV_CHECK, DIV_FLOOR = 40, 0.795
EVAL_FULL  = 1000         # phase B: the whole test set

ARCH = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=HIDDEN, out_dim=OUT_DIM,
            act=ACT, bias=BIAS, init=INIT)
# Their learn_code is:
#     error_start_index = 0
#     error_end_index   = -1
#     (outputs - target)[:, error_start_index:error_end_index].pow(2).sum() * 0.5
# With share_output_across_tasks True the conditional below it never fires, so those indices
# stand. `[:, 0:-1]` on a five-unit output covers units 0-3 and EXCLUDES UNIT 4 -- one class
# per task never receives any gradient at its own output unit, though it is still argmaxed
# over at test time. It reads like an off-by-one that reached the published figure. Set
# LOSS_ACTIVE to None (and mask=False) for the version that was presumably intended; the
# comparison between the two is cheap and worth reporting.
LOSS_ACTIVE = list(range(OUT_DIM - 1))          # [0,1,2,3] -- reproduces their slice
OBJ = Objective(loss="mse", target="onehot", mask=LOSS_ACTIVE is not None, reduction="sum")
COL = {"backprop": "tab:red", "pc": "tab:blue"}
LABEL = {"backprop": "backpropagation", "pc": "prospective configuration"}
# ==================================================================


def make_tasks(seed):
    """Their shuffle_mapper: permute all ten labels, split 5/5, map position j -> unit j % 5.
       NOT sorted -- sorting is the bug that contaminated exps 30 and 32."""
    perm = np.random.default_rng(seed).permutation(10).tolist()
    return [perm[:5], perm[5:]]


def build(rule, lr, seed):
    extra = dict(dt=X_LR, steps=T_INFER, x_lr_discount=X_LR_DISCOUNT,
                 x_lr_amplifier=X_LR_AMP) if rule == "pc" else {}
    return build_method(rule, in_dim=ARCH.in_dim, hidden=HIDDEN, out_dim=OUT_DIM,
                        arch=ARCH, obj=OBJ, lr=lr, seed=seed, device=DEVICE, **extra)


train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR), fashion=FASHION)
cidx = class_indices(train)

t0 = make_tasks(BASE_SEED)
print(f"device: {DEVICE}")
print(f"seed {BASE_SEED} pairing check -- output unit: task-1 class / task-2 class")
print("   " + "   ".join(f"u{i}: {t0[0][i]}/{t0[1][i]}" for i in range(5)))
print("   (should look random; consistently close pairs would mean the sort bug is back)\n")

# ==================== phase A: learning-rate sweep (Fig 4e) ====================
sweep = np.full((len(RULES), len(LRS), N_SEEDS_SWEEP, TOTAL_ITERS, 2), np.nan)
class_sets = np.zeros((N_SEEDS, 2, 5), dtype=int)
t_start = time.time()

for si in range(N_SEEDS_SWEEP):
    seed = BASE_SEED + si
    tasks = make_tasks(seed)
    class_sets[si] = np.array(tasks)
    ev = make_domain_il_eval(test, tasks, per_class=EVAL_SWEEP, device=DEVICE)
    for ri, rule in enumerate(RULES):
        for li, lr in enumerate(LRS):
            step_fn, predict = build(rule, lr, seed)
            out = run_alternating(step_fn, predict, tasks, train, cidx, ev,
                                  iters_per_task=ITERS_PER_TASK, total_iters=TOTAL_ITERS,
                                  batch=BATCH, partial_num=PARTIAL_NUM, device=DEVICE,
                                  data_seed=seed, eval_per_update=False,
                                  loss_active=LOSS_ACTIVE,
                                  divergence_check=DIV_CHECK, divergence_floor=DIV_FLOOR)
            sweep[ri, li, si] = out["per_iter"]
    done, tot = si + 1, N_SEEDS_SWEEP
    el = time.time() - t_start
    print(f"  phase A seed {done}/{tot}  elapsed {el:5.0f}s  eta {el / done * (tot - done):5.0f}s")

mean_err = np.nanmean(sweep[:, :, :, :ANALYSE_ITERS, :], axis=(3, 4))     # [rule, lr, seed]
fig4e = np.nanmean(mean_err, axis=2)
best = {r: int(np.nanargmin(fig4e[i])) for i, r in enumerate(RULES)}

print("\n" + "=" * 80)
print(f"FIG 4e -- mean test error over the first {ANALYSE_ITERS} iterations")
print(f"{'lr':>10}" + "".join(f"{LABEL[r]:>32}" for r in RULES))
for li, lr in enumerate(LRS):
    row = f"{lr:>10}"
    for ri, r in enumerate(RULES):
        m, lo, hi = bootstrap_ci(mean_err[ri, li])
        row += f"{m:>19.4f} [{lo:.3f},{hi:.3f}]{'*' if li == best[r] else ' '}".rjust(32)
    print(row)
for r in RULES:
    if best[r] in (0, len(LRS) - 1):
        print(f"  WARNING: {LABEL[r]}'s best lr {LRS[best[r]]} is at the EDGE of the grid")

# ==================== phase B: headline curve at full resolution (Fig 4d) ====================
print(f"\nphase B: {', '.join(f'{r} lr={LRS[best[r]]}' for r in RULES)}, "
      f"full test set, {N_SEEDS} seeds")
head_iter = {}
tB = time.time()
for si in range(N_SEEDS):
    seed = BASE_SEED + si
    tasks = make_tasks(seed)
    class_sets[si] = np.array(tasks)          # phase B covers all N_SEEDS, phase A only some
    ev = make_domain_il_eval(test, tasks, per_class=EVAL_FULL, device=DEVICE)
    for ri, rule in enumerate(RULES):
        step_fn, predict = build(rule, LRS[best[rule]], seed)
        out = run_alternating(step_fn, predict, tasks, train, cidx, ev,
                              iters_per_task=ITERS_PER_TASK, total_iters=TOTAL_ITERS,
                              batch=BATCH, partial_num=PARTIAL_NUM, device=DEVICE,
                              data_seed=seed, eval_per_update=False,
                              loss_active=LOSS_ACTIVE,
                              divergence_check=DIV_CHECK, divergence_floor=DIV_FLOOR)
        head_iter[(rule, si)] = out["per_iter"]
    el = time.time() - tB
    print(f"  phase B seed {si + 1}/{N_SEEDS}  elapsed {el:5.0f}s  "
          f"eta {el / (si + 1) * (N_SEEDS - si - 1):5.0f}s")

# [rule, seed, iter, task] -> [rule, 1, seed, iter, task]. The length-1 learning-rate axis
# is there so 31_analyse_reproduction can read this file unchanged apart from its SRC path.
# Per-update logging is retired: exp 31 showed the block-aligned analysis of per-iteration
# data reveals the trade-off just as clearly, and their own figure is per-iteration.
per_iter = np.stack([[head_iter[(r, s)] for s in range(N_SEEDS)] for r in RULES])[:, None]

hm = np.nanmean(per_iter[:, 0, :, :ANALYSE_ITERS, :], axis=(2, 3))        # [rule, seed]
print("\n" + "=" * 80)
print(f"{'rule':>32}{'lr':>11}{'mean err':>22}{'final t1':>11}{'final t2':>11}")
for ri, r in enumerate(RULES):
    m, lo, hi = bootstrap_ci(hm[ri])
    print(f"{LABEL[r]:>32}{LRS[best[r]]:>11}{m:>13.4f} [{lo:.3f},{hi:.3f}]"
          f"{np.nanmean(per_iter[ri, 0, :, ANALYSE_ITERS - 1, 0]):>11.4f}"
          f"{np.nanmean(per_iter[ri, 0, :, ANALYSE_ITERS - 1, 1]):>11.4f}")
diff = hm[RULES.index("backprop")] - hm[RULES.index("pc")]
dm, dlo, dhi = bootstrap_ci(diff)
print(f"\nPAIRED backprop minus pc: {dm:+.4f}  68% CI [{dlo:+.4f}, {dhi:+.4f}]  "
      f"pc better in {int((diff > 0).sum())}/{N_SEEDS} seeds")
print("positive = prospective configuration has the LOWER error, i.e. the claim reproduces")

# ------------------------------- figure -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
for ri, r in enumerate(RULES):
    for task, ls in ((0, "-"), (1, "--")):
        A = per_iter[ri, 0, :, :ANALYSE_ITERS, task]
        m = np.nanmean(A, 0); se = np.nanstd(A, 0) / np.sqrt(N_SEEDS)
        xs = np.arange(ANALYSE_ITERS)
        ax.plot(xs, m, ls, color=COL[r], lw=2.0, label=f"{LABEL[r]}, task {task + 1}")
        ax.fill_between(xs, m - se, m + se, color=COL[r], alpha=0.13)
for b in range(ITERS_PER_TASK, ANALYSE_ITERS, ITERS_PER_TASK):
    ax.axvline(b, color="k", lw=0.4, alpha=0.2)
ax.set_xlabel("iteration"); ax.set_ylabel("test error")
ax.set_title("Fig 4d: test error during continual learning")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
for ri, r in enumerate(RULES):
    st = [bootstrap_ci(mean_err[ri, li]) for li in range(len(LRS))]
    ax.plot(LRS, [s[0] for s in st], "o-", color=COL[r], lw=2.2, label=LABEL[r])
    ax.fill_between(LRS, [s[1] for s in st], [s[2] for s in st], color=COL[r], alpha=0.15)
    ax.plot(LRS[best[r]], st[best[r]][0], "*", color=COL[r], ms=16)
ax.set_xscale("log"); ax.set_xlabel("learning rate")
ax.set_ylabel(f"mean test error over {ANALYSE_ITERS} iterations")
ax.set_title("Fig 4e: mean test error vs learning rate")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

fig.suptitle("Song & Bogacz (2024) Fig 4d-e reproduced from their yaml, corrected pairing\n"
             f"Fashion-MNIST, {ARCH.widths}, sigmoid, no bias, {PARTIAL_NUM}/class, "
             f"batch {BATCH}, summed loss, T={T_INFER}, {N_SEEDS} seeds")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}   total {time.time() - t_start:.0f}s")

cfg = dict(widths=list(ARCH.widths), act=ACT, init=INIT, bias=BIAS, out_dim=OUT_DIM,
           partial_num_per_class=PARTIAL_NUM, batch=BATCH, iters_per_task=ITERS_PER_TASK,
           total_iters=TOTAL_ITERS, analyse_iters=ANALYSE_ITERS, T=T_INFER, x_lr=X_LR,
           x_lr_discount=X_LR_DISCOUNT, reduction="sum", lrs=LRS, rules=RULES,
           n_seeds=N_SEEDS, best_lr={r: LRS[best[r]] for r in RULES},
           eval_sweep=EVAL_SWEEP, eval_full=EVAL_FULL, n_seeds_sweep=N_SEEDS_SWEEP,
           loss_active=LOSS_ACTIVE,
           source="base-shuffle-task-5-FashionMNIST.yaml", supersedes=["exp30", "exp32"])
np.savez_compressed(FIG.with_suffix(".npz"), sweep=sweep, per_iter=per_iter,
                    classes=class_sets, rules=np.array(RULES), lrs=np.array(LRS),
                    config=json.dumps(cfg))
print(f"saved {FIG.with_suffix('.npz').name}")
print("  sweep[rule,lr,seed,iter,task]   all learning rates, N_SEEDS_SWEEP seeds")
print("  per_iter[rule,1,seed,iter,task] headline at the winning lr, all seeds")
print("  exp 31 can be pointed at this file by changing its SRC constant.")
