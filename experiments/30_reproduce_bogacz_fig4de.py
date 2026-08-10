"""30_reproduce_bogacz_fig4de  (v2 -- built from their actual config, not the paper text)

ONE QUESTION
    Under Song & Bogacz's own configuration, does predictive coding show lower test error
    during continual learning than backpropagation?

SOURCE OF TRUTH
    experiments/nature_forgetting/base-shuffle-task-5-FashionMNIST.yaml in
    github.com/YuhangSong/Prospective-Configuration, plus their analysis command for fig4-d
    and fig4-e. Every constant below is read off that file. Where the paper's Methods text
    and the yaml disagree, THE YAML WINS -- it is what produced the figure.

WHAT v1 GOT WRONG (v1 found no difference between the rules; here is why)
    depth        yaml num_layers: 4 with structure ['Linear','PCLayer','Acf'] builds FOUR
                 Linear layers: 784-32-32-32-5, i.e. THREE hidden layers. v1 used two.
                 Their Fig 3e claims the advantage grows with depth, so this matters.
    bias         yaml bias: False. v1 had biases on.
    training set partial_num: 600 examples per task (~120 per class). v1 used all ~30,000.
                 Far less data means far less shared structure to fall back on, which is a
                 large part of why forgetting is visible in their curves at all.
    batch size   500, not 32.
    loss         (outputs - target).pow(2).sum() * 0.5 -- SUMMED over batch and outputs, not
                 averaged. At batch 500 the gradient is ~500x a mean-reduced one. This alone
                 explains why their learning rates are 0.0001-0.005 and ours were 0.005-0.5:
                 the two grids were in different units and never comparable.
    iteration    one iteration = one EPOCH over the 600-example subset = 2 weight updates
                 (500 + 100, drop_last False). v1 treated an iteration as one update.
    total        num_iterations: 160; the fig4-d/4-e analysis then takes [:84].
    inference    T: 64 steps, optimizer_x = SGD(lr=0.1), x_lr_discount 0.9, x_lr_amplifier 1.0
                 -- the inference step size is multiplied by 0.9 whenever the energy fails to
                 fall (amplifier 1.0 = never increased). v1 used 50 fixed steps.
    backprop     implemented as the SAME network with the PCLayers removed and T=1, so the
                 only difference between the two arms is the relaxation.

STILL NOT MATCHED, and recorded as deviations
    1. Their seed list has 21 entries; the paper reports n = 10. We use the first N_SEEDS.
    2. partial_dateset's selection rule (first-n vs random-n) is not visible to us; we take a
       seeded random subset.
    3. They run log_task_i as two separate trials, one per task, and merge afterwards. We
       evaluate both tasks in one run, which is equivalent given the same seed and cheaper.

RESOLUTION
    Their curves are logged once per iteration. We log after EVERY WEIGHT UPDATE, on both
    tasks. `per_iter` reproduces their resolution exactly; `errors` is twice as fine, so any
    forgetting that happens on the first update after a switch is visible rather than
    averaged away. Nothing is skipped.

INTERPRETATION
    reproduced      PC below backprop in 4d, lower 4e minimum. Their claim holds, and our
                    class-incremental results become a boundary condition on it.
    not reproduced  check first: is the best lr interior to the grid; did both arms reach a
                    sensible error at all; does the 4e curve have a clear minimum. Only then
                    treat it as a failure to replicate.
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

# ==================== constants: every one from their yaml ====================
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data"
FIG      = Path(__file__).resolve().with_suffix(".png")

IMG_SIZE   = 28              # input_size 784
FASHION    = True            # dataset: FashionMNIST
HIDDEN     = (32, 32, 32)    # num_layers 4 -> 784-32-32-32-5, hidden_size 32
OUT_DIM    = 5               # share_output_across_tasks: True, 10/num_tasks
ACT        = "sigmoid"       # acf: Sigmoid
INIT       = "xavier_normal" # init_fn: torch.nn.init.xavier_normal_, gain 1.0
BIAS       = False           # bias: False

PARTIAL_NUM   = 600          # partial_num: 600
BATCH         = 500          # batch_size: 500
ITERS_PER_TASK = 4           # num_iterations/num_repeatations/num_tasks = 160/20/2
TOTAL_ITERS   = 160          # num_iterations: 160
ANALYSE_ITERS = 84           # their analysis takes df[...][:84]

T_INFER       = 64           # T: 64
X_LR          = 0.1          # optimizer_x_kwargs.lr
X_LR_DISCOUNT = 0.9          # x_lr_discount
X_LR_AMPLIFIER = 1.0         # x_lr_amplifier

LRS = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.005]   # optimizer_p_kwargs.lr grid
N_SEEDS, BASE_SEED = 10, 0
RULES = ["backprop", "pc"]
EVAL_PER_CLASS = 1000

ARCH = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=HIDDEN, out_dim=OUT_DIM,
            act=ACT, bias=BIAS, init=INIT)
OBJ = Objective(loss="mse", target="onehot", mask=False, reduction="sum")
COL = {"backprop": "tab:red", "pc": "tab:blue"}
LABEL = {"backprop": "backpropagation", "pc": "prospective configuration"}
# ==============================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR), fashion=FASHION)
cidx = class_indices(train)

all_err, all_per_iter, class_sets = {}, {}, np.zeros((N_SEEDS, 2, 5), dtype=int)
t0 = time.time()

for si in range(N_SEEDS):
    seed = BASE_SEED + si
    d = np.random.default_rng(seed).permutation(10).tolist()
    tasks = [sorted(d[:5]), sorted(d[5:])]
    class_sets[si] = np.array(tasks)
    eval_sets = make_domain_il_eval(test, tasks, per_class=EVAL_PER_CLASS, device=DEVICE)

    for rule in RULES:
        extra = dict(dt=X_LR, steps=T_INFER, x_lr_discount=X_LR_DISCOUNT,
                     x_lr_amplifier=X_LR_AMPLIFIER) if rule == "pc" else {}
        for lr in LRS:
            step_fn, predict = build_method(rule, in_dim=ARCH.in_dim, hidden=HIDDEN,
                                            out_dim=OUT_DIM, arch=ARCH, obj=OBJ, lr=lr,
                                            seed=seed, device=DEVICE, **extra)
            out = run_alternating(step_fn, predict, tasks, train, cidx, eval_sets,
                                  iters_per_task=ITERS_PER_TASK, total_iters=TOTAL_ITERS,
                                  batch=BATCH, partial_num=PARTIAL_NUM, device=DEVICE,
                                  data_seed=seed)
            all_err[(rule, lr, si)] = out["errors"]
            all_per_iter[(rule, lr, si)] = out["per_iter"]
    print(f"seed {si + 1}/{N_SEEDS}  task1={tasks[0]} task2={tasks[1]}  "
          f"({time.time() - t0:5.0f}s)")

n_upd = min(v.shape[0] for v in all_err.values())
errors = np.stack([[[all_err[(r, lr, s)][:n_upd] for s in range(N_SEEDS)]
                    for lr in LRS] for r in RULES])                 # [rule,lr,seed,upd,task]
per_iter = np.stack([[[all_per_iter[(r, lr, s)] for s in range(N_SEEDS)]
                      for lr in LRS] for r in RULES])               # [rule,lr,seed,iter,task]

# their metric: mean of test error over the first 84 iterations, both tasks
mean_err = np.nanmean(per_iter[:, :, :, :ANALYSE_ITERS, :], axis=(3, 4))   # [rule,lr,seed]
fig4e = np.nanmean(mean_err, axis=2)
best = {r: int(np.nanargmin(fig4e[i])) for i, r in enumerate(RULES)}

print("\n" + "=" * 84)
print(f"FIG 4e -- mean test error over the first {ANALYSE_ITERS} iterations (lower is better)")
print(f"{'lr':>10}" + "".join(f"{LABEL[r]:>32}" for r in RULES))
for li, lr in enumerate(LRS):
    row = f"{lr:>10}"
    for ri, r in enumerate(RULES):
        m, lo, hi = bootstrap_ci(mean_err[ri, li])
        row += f"{m:>20.4f} [{lo:.3f},{hi:.3f}]{'*' if li == best[r] else ' '}".rjust(32)
    print(row)

print(f"\n{'rule':>32}{'best lr':>11}{'mean err':>22}{'final t1':>11}{'final t2':>11}")
for ri, r in enumerate(RULES):
    li = best[r]
    m, lo, hi = bootstrap_ci(mean_err[ri, li])
    print(f"{LABEL[r]:>32}{LRS[li]:>11}{m:>13.4f} [{lo:.3f},{hi:.3f}]"
          f"{np.nanmean(per_iter[ri, li, :, ANALYSE_ITERS - 1, 0]):>11.4f}"
          f"{np.nanmean(per_iter[ri, li, :, ANALYSE_ITERS - 1, 1]):>11.4f}")
    if li in (0, len(LRS) - 1):
        print(f"{'':>32}WARNING: best lr is at the EDGE of their grid -- widen and re-run")

diff = mean_err[0, best["backprop"]] - mean_err[1, best["pc"]]
dm, dlo, dhi = bootstrap_ci(diff)
print(f"\nPAIRED backprop minus pc: {dm:+.4f}  68% CI [{dlo:+.4f}, {dhi:+.4f}]  "
      f"pc better in {int((diff > 0).sum())}/{N_SEEDS} seeds")
print("positive = prospective configuration has the LOWER error, i.e. the claim reproduces")

# ------------------------------- figure -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
for ri, r in enumerate(RULES):
    li = best[r]
    for task, ls in ((0, "-"), (1, "--")):
        A = per_iter[ri, li, :, :ANALYSE_ITERS, task]
        m = np.nanmean(A, 0); se = np.nanstd(A, 0) / np.sqrt(N_SEEDS)
        xs = np.arange(ANALYSE_ITERS)
        ax.plot(xs, m, ls, color=COL[r], lw=2.0, label=f"{LABEL[r]}, task {task + 1}")
        ax.fill_between(xs, m - se, m + se, color=COL[r], alpha=0.13)
for b in range(ITERS_PER_TASK, ANALYSE_ITERS, ITERS_PER_TASK):
    ax.axvline(b, color="k", lw=0.4, alpha=0.2)
ax.set_xlabel("iteration"); ax.set_ylabel("test error")
ax.set_title("Fig 4d: test error during continual learning\n(each rule at its own best lr)")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

ax = axes[1]
for ri, r in enumerate(RULES):
    st = [bootstrap_ci(mean_err[ri, li]) for li in range(len(LRS))]
    m = [s[0] for s in st]
    ax.plot(LRS, m, "o-", color=COL[r], lw=2.2, label=LABEL[r])
    ax.fill_between(LRS, [s[1] for s in st], [s[2] for s in st], color=COL[r], alpha=0.15)
    ax.plot(LRS[best[r]], m[best[r]], "*", color=COL[r], ms=16)
ax.set_xscale("log"); ax.set_xlabel("learning rate")
ax.set_ylabel(f"mean test error over {ANALYSE_ITERS} iterations")
ax.set_title("Fig 4e: mean test error vs learning rate\n(stars = optimum, bands = 68% CI)")
ax.legend(fontsize=8); ax.grid(alpha=0.2)

fig.suptitle("Reproduction of Song & Bogacz (2024) Fig 4d-e, built from their yaml\n"
             f"Fashion-MNIST, {ARCH.widths}, sigmoid, no bias, {PARTIAL_NUM} examples/task, "
             f"batch {BATCH}, summed loss, T={T_INFER}, {N_SEEDS} seeds")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}")

config = dict(widths=list(ARCH.widths), act=ACT, init=INIT, bias=BIAS, out_dim=OUT_DIM,
              partial_num=PARTIAL_NUM, batch=BATCH, iters_per_task=ITERS_PER_TASK,
              total_iters=TOTAL_ITERS, analyse_iters=ANALYSE_ITERS, T=T_INFER, x_lr=X_LR,
              x_lr_discount=X_LR_DISCOUNT, x_lr_amplifier=X_LR_AMPLIFIER,
              loss="0.5*sum((out-target)^2)", reduction="sum", lrs=LRS, rules=RULES,
              n_seeds=N_SEEDS, source="base-shuffle-task-5-FashionMNIST.yaml")
np.savez_compressed(FIG.with_suffix(".npz"), errors=errors, per_iter=per_iter,
                    classes=class_sets, rules=np.array(RULES), lrs=np.array(LRS),
                    config=json.dumps(config))
print(f"saved {FIG.with_suffix('.npz').name}")
print("  errors[rule,lr,seed,update,task]   -- every weight update, both tasks")
print("  per_iter[rule,lr,seed,iter,task]   -- their per-iteration resolution")
