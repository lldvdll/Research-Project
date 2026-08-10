"""23_dataset_and_capacity

ONE QUESTION
    Is MNIST too easy for this project -- i.e. does a harder dataset widen the range of
    hidden-layer widths at which the internal representation actually matters?

WHY IT IS ASKED THIS WAY
    Exp 21 showed that on MNIST the regime where the hidden layer does task-specific work is
    narrow. Two diagnostics, both from that figure, at 2 tasks x 5 classes:

        NCM above floor   trained-network NCM minus UNTRAINED-network NCM.
                          0 means the hidden layer is a random projection: training added
                          nothing that a random matrix did not already provide.
                          MNIST: 24 pts at width 8, 3 pts at 64, 0 pts at 128.
        drift             live-prototype NCM minus frozen-prototype NCM. How far the code
                          moved across the task switch.
                          MNIST: 39 pts at width 8, 2 pts at 64, 0 pts at 128.

    So on MNIST the interesting regime is hidden = 8-16, which is an awkward place to run a
    thesis: accuracy is low, variance is high, and a reviewer will ask whether the effect is
    an artefact of a crippled network. The question is whether Fashion-MNIST moves that regime
    up to a defensible width.

TEST
    Backprop only, 2 tasks x 5 classes, identical pipeline. Sweep width {2,4,8,16,32,64,128}
    on MNIST and Fashion-MNIST. Same three quantities per cell as exp 21, plus the causal one
    from exp 22 (freeze W1 during task 2), so this inherits exp 22's metric instead of
    inventing a new one.

    The 4-class arm from exp 21 is dropped, as you asked -- it varied the load without adding
    anything the 10-class arm did not show more clearly.

PROJECT DECISION THIS SETTLES
    Which dataset and which hidden width the whole method comparison runs at. Everything from
    the learning-rule comparison onward inherits this choice, and the report has to justify it.

EXPECTED
    Fashion-MNIST curves shifted RIGHT and DOWN: the NCM-above-floor and drift diagnostics
    should stay large out to width 32-64 instead of collapsing by 16-32, and the freeze-W1 gap
    should be larger at every width. Roughly, a defensible operating point around hidden = 32
    on Fashion-MNIST rather than 8-16 on MNIST.

WHAT THAT WOULD DEMONSTRATE
    That MNIST's ease, not the class-incremental setting itself, was suppressing the
    representational component of forgetting -- and that the earlier "predictive coding buys a
    better path but the same endpoint" result was measured at an operating point where no
    learning rule could have done better. Switching dataset is then a correction, not a
    goalpost move, and it is defensible in the report on exactly these numbers.

IF IT COMES OUT DIFFERENTLY
    Curves nearly identical to MNIST
        -> difficulty is not the lever; the bottleneck is the 196-dimensional flattened input
           and the single hidden layer. Next lever is DEPTH (Song & Bogacz report the
           advantage growing with depth) rather than dataset. Stay on MNIST and add layers.
    Fashion-MNIST diagnostics large at EVERY width including 128
        -> excellent: run at 64 or 128, where accuracy is respectable and no reviewer can
           object that the network was crippled.
    freeze-W1 gap stays near zero on both datasets at all widths
        -> the hidden-layer pathway is negligible however hard the task, and the thesis should
           be reframed around the output layer. Report this as the finding it is.

FIGURE
    23_dataset_and_capacity.png
    Two panels (MNIST, Fashion-MNIST). Per panel: NCM-above-floor, drift, and freeze-W1 gap
    against hidden width -- all three on one axis in POINTS, so "where does the hidden layer
    matter" is a single readable picture, with a shaded band marking the usable regime.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data import load_mnist, class_indices, make_eval_split
from src.model import Arch, Objective, replace
from src.methods import build_method
from src.runner import run_classil
from src.probes import (prototype_images, live_ncm_fn, frozen_ncm_fn, restricted_argmax_fn)

# ============================ constants ============================
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR  = ROOT / "data"
FIG       = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE  = 14
BASE_SEED = 0
N_RUNS    = 8                      # 2 datasets x 7 widths x 3 arms x 8 runs

WIDTHS            = [2, 4, 8, 16, 32, 64, 128]
DATASETS          = {"MNIST": False, "Fashion-MNIST": True}
CLASSES_PER_TASK  = 5
TASK1_THRESHOLD   = 0.70
STOP_PATIENCE     = 3
TASK1_MAX_ITERS   = 800
TASK2_ITERS       = 300             # fixed budget, matched across arms (as in exp 22)
BATCH             = 32
EVAL_EVERY        = 10
EVAL_PER_CLASS    = 100
PROTO_PER_CLASS   = 50

ARCH_BASE = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=64, out_dim=10,
                 act="tanh", bias=True, init="scaled_normal")
OBJ = Objective(loss="mse", target="onehot", mask=False)
BP_LR = 0.05

ARMS = {"normal": set(), "freeze W1": {"W1", "b1"}}
# ==================================================================

res = {}       # (ds, width, arm) -> dict of lists
floor = {}     # (ds, width) -> list
t0 = time.time()

for ds_name, is_fashion in DATASETS.items():
    train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR), fashion=is_fashion)
    cidx = class_indices(train)
    for width in WIDTHS:
        arch = replace(ARCH_BASE, hidden=width)
        for arm in ARMS:
            res[(ds_name, width, arm)] = dict(t1=[], t2=[], peak=[], live=[], frozen=[])
        floor[(ds_name, width)] = []

        for run in range(N_RUNS):
            seed = BASE_SEED + run
            d = np.random.default_rng(seed).permutation(10).tolist()
            k = CLASSES_PER_TASK
            tasks = [sorted(d[:k]), sorted(d[k:2 * k])]
            classes = sorted({c for t in tasks for c in t})
            stop_ev, rep_ev = make_eval_split(test, classes, EVAL_PER_CLASS, DEVICE, seed=seed)
            proto_x, proto_y = prototype_images(train, cidx, classes, PROTO_PER_CLASS,
                                                DEVICE, seed=seed)
            t1_sel = torch.isin(rep_ev[1], torch.tensor(tasks[0], device=DEVICE))

            # untrained-network NCM floor, on task-1 classes
            h0 = {}
            build_method("backprop", in_dim=arch.in_dim, hidden=width, out_dim=arch.out_dim,
                         arch=arch, obj=OBJ, lr=BP_LR, seed=seed, device=DEVICE, handle=h0)
            fl = live_ncm_fn(h0["features"], proto_x, proto_y)
            floor[(ds_name, width)].append(
                (fl(rep_ev[0][t1_sel]) == rep_ev[1][t1_sel]).float().mean().item())

            for arm, frozen_names in ARMS.items():
                handle, frozen_p = {}, {}
                step_fn, predict = build_method("backprop", in_dim=arch.in_dim, hidden=width,
                                                out_dim=arch.out_dim, arch=arch, obj=OBJ,
                                                lr=BP_LR, seed=seed, device=DEVICE,
                                                handle=handle)
                feats, freeze = handle["features"], handle["freeze"]
                ra = restricted_argmax_fn(predict, classes)

                def on_task_end(ti, step, _f=feats, _fp=frozen_p, _fz=freeze,
                                _fn=frozen_names, _px=proto_x, _py=proto_y):
                    if ti != 0:
                        return
                    pf = _f(_px).detach()
                    for c in sorted(set(_py.tolist())):
                        _fp[c] = pf[_py == c].mean(0)
                    _fz.update(_fn)

                o1 = run_classil(step_fn, predict, [tasks[0]], train, cidx,
                                 report_eval=rep_ev, stop_eval=stop_ev,
                                 readouts={"a": ra}, max_iters_per_task=TASK1_MAX_ITERS,
                                 batch=BATCH, eval_every=EVAL_EVERY, device=DEVICE,
                                 stop_threshold=TASK1_THRESHOLD, stop_patience=STOP_PATIENCE,
                                 data_seed=seed, on_task_end=on_task_end)
                run_classil(step_fn, predict, [tasks[1]], train, cidx,
                            report_eval=rep_ev, stop_eval=stop_ev, readouts={"a": ra},
                            max_iters_per_task=TASK2_ITERS, batch=BATCH,
                            eval_every=EVAL_EVERY, device=DEVICE, stop_threshold=None,
                            data_seed=seed + 7777)

                pred = ra(rep_ev[0])
                acc = {c: (pred[rep_ev[1] == c] == c).float().mean().item() for c in classes}
                R = res[(ds_name, width, arm)]
                R["peak"].append(o1["curves"]["a"][-1, 0])
                R["t1"].append(float(np.mean([acc[c] for c in tasks[0]])))
                R["t2"].append(float(np.mean([acc[c] for c in tasks[1]])))
                lv = live_ncm_fn(feats, proto_x, proto_y)
                fz = frozen_ncm_fn(feats, frozen_p)
                R["live"].append((lv(rep_ev[0][t1_sel]) == rep_ev[1][t1_sel]).float().mean().item())
                R["frozen"].append((fz(rep_ev[0][t1_sel]) == rep_ev[1][t1_sel]).float().mean().item())

        nf = np.mean(floor[(ds_name, width)]) * 100
        N = res[(ds_name, width, "normal")]
        gap = (np.mean(res[(ds_name, width, "freeze W1")]["t1"]) - np.mean(N["t1"])) * 100
        print(f"{ds_name:<15} h={width:>4} | normal t1 {np.mean(N['t1'])*100:5.1f} | "
              f"freezeW1 gap {gap:+5.1f} | ncm-above-floor "
              f"{np.mean(N['live'])*100 - nf:+5.1f} | drift "
              f"{(np.mean(N['live']) - np.mean(N['frozen']))*100:5.1f} | {time.time()-t0:5.0f}s")

# ------------------------------- table -------------------------------
for ds_name in DATASETS:
    print("\n" + "=" * 100)
    print(f"{ds_name}: where does the hidden layer matter?  (2x5 class-IL, {N_RUNS} runs)")
    print(f"{'hidden':>7}{'t1 peak':>10}{'t1 normal':>11}{'t1 freezeW1':>13}"
          f"{'freeze gap':>12}{'NCM>floor':>11}{'drift':>8}")
    for w in WIDTHS:
        N, F = res[(ds_name, w, "normal")], res[(ds_name, w, "freeze W1")]
        nf = np.mean(floor[(ds_name, w)]) * 100
        print(f"{w:>7}{np.mean(N['peak'])*100:>10.1f}{np.mean(N['t1'])*100:>11.1f}"
              f"{np.mean(F['t1'])*100:>13.1f}"
              f"{(np.mean(F['t1'])-np.mean(N['t1']))*100:>+12.1f}"
              f"{np.mean(N['live'])*100-nf:>+11.1f}"
              f"{(np.mean(N['live'])-np.mean(N['frozen']))*100:>8.1f}")

print("\nDECISION RULE: run the method comparison at the LARGEST width where the freeze-W1 gap")
print("is still clearly above zero. That is the largest network in which a drift-reducing")
print("learning rule has anything to win. Prefer the dataset that pushes that width higher.")

# ------------------------------- figure -------------------------------
fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 5), sharey=True)
axes = np.atleast_1d(axes)
for ax, ds_name in zip(axes, DATASETS):
    nf = np.array([np.mean(floor[(ds_name, w)]) for w in WIDTHS]) * 100
    live = np.array([np.mean(res[(ds_name, w, "normal")]["live"]) for w in WIDTHS]) * 100
    froz = np.array([np.mean(res[(ds_name, w, "normal")]["frozen"]) for w in WIDTHS]) * 100
    gap = np.array([np.mean(res[(ds_name, w, "freeze W1")]["t1"])
                    - np.mean(res[(ds_name, w, "normal")]["t1"]) for w in WIDTHS]) * 100
    ax.plot(WIDTHS, live - nf, "o-", color="tab:green", lw=2.2,
            label="NCM above untrained floor (is the code doing work?)")
    ax.plot(WIDTHS, live - froz, "s-", color="tab:purple", lw=2.2,
            label="drift: live minus frozen prototypes (did the code move?)")
    ax.plot(WIDTHS, gap, "^-", color="tab:blue", lw=2.6,
            label="freeze-W1 gap (does the movement COST anything?)")
    ax.axhline(0, color="k", lw=1.0)
    usable = [w for w, g in zip(WIDTHS, gap) if g > 2]
    if usable:
        ax.axvspan(min(usable), max(usable), color="tab:blue", alpha=0.07)
        ax.text(max(usable), ax.get_ylim()[1] * 0.92, f" usable to h={max(usable)}",
                fontsize=8, ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks(WIDTHS); ax.set_xticklabels(WIDTHS)
    ax.set_xlabel("hidden units"); ax.set_title(ds_name); ax.grid(alpha=0.2)
axes[0].set_ylabel("effect size (percentage points)")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Does a harder dataset widen the regime where the internal representation matters?")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}")
np.savez(FIG.with_suffix(".npz"), widths=WIDTHS,
         **{f"{k[0]}|{k[1]}|{k[2]}|{m}": np.array(v)
            for k, R in res.items() for m, v in R.items()},
         **{f"floor|{k[0]}|{k[1]}": np.array(v) for k, v in floor.items()})
