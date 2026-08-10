"""21_capacity_and_representation_drift

QUESTION
    At what hidden-layer width does REPRESENTATION DRIFT start to contribute to forgetting,
    rather than forgetting being entirely an output-layer effect?

WHY THIS ONE, AND WHY NOW
    Experiment 20 showed that masking the loss recovers most of task 1 (14% -> 67%), so in
    that setting forgetting is dominated by output-layer suppression. But its NCM probe sat
    flat at ~90% throughout, including BEFORE task 2 was ever trained -- which does not mean
    "the representation was preserved". It means a 64-unit random projection of MNIST 14x14
    already separates four digit classes near ceiling. The probe was saturated and had no
    dynamic range, so it could not have detected drift even if drift had occurred.

    Two consequences, and this experiment fixes both:
      1. The probe needs a floor (random-init baseline) and a version that can actually see
         movement (prototypes FROZEN at the task switch, not rebuilt at current weights).
      2. The setting needs to be made hard enough that the hidden layer is the bottleneck.
         Exp 13 says a 4-class problem is ~99% at ANY width from 16 to 256, so at hidden=64
         with 4 classes there is enormous capacity slack. If the hidden layer is never
         limiting, a learning rule -- which only changes how hidden-layer credit is assigned
         -- cannot possibly show an advantage. Every later method comparison depends on
         choosing an operating point where it can.

TEST
    Backprop only. Sweep hidden width over {2, 4, 8, 16, 32, 64, 128}, at two loads:
        4 classes  (2 tasks x 2)   -- the exp 20 setting
        10 classes (2 tasks x 5)   -- the exp 12 setting
    Crossed with loss {all classes, masked to current task}. 10 pairings per cell, same
    pairings / init / data order across every cell.

    Five readouts, so the output layer and the hidden layer are measured separately:
        argmax          argmax over all output units          (what the network answers)
        argmax_r        argmax restricted to classes in play   (removes the dead-unit artefact:
                        with out_dim=10 and 4 classes, 6 units are never a target and, under a
                        masked loss, never suppressed either -- they keep random weights and
                        can win the argmax, penalising the masked arm for no good reason)
        ncm_live        prototypes rebuilt at current weights  ("still linearly decodable?")
        ncm_frozen      prototypes snapshot at the task switch ("has the code MOVED?")
        (plus)          ncm on an UNTRAINED network of the same width, measured separately
                        as the floor. Any NCM number must be read against this.

    Also reported: hidden-code drift for task-1 images between end-of-task-1 and end-of-task-2
    (mean cosine and relative L2), which has dynamic range regardless of task difficulty.

INTERPRETATION
    ncm_live ~= random-init baseline at every width
        -> the hidden layer is not doing task-specific work; the probe is measuring the input
           distribution, not learning. Forgetting is entirely an output-layer story here.
    ncm_frozen falls while ncm_live stays flat
        -> the code MOVED but stayed decodable. This is drift that live prototypes hide, and
           it is the regime where prospective configuration should help.
    masked argmax_r falls as width shrinks
        -> capacity is now binding: removing suppression is no longer sufficient, so there is
           genuine representation interference for a learning rule to act on.
    the width at which the masked and unmasked curves converge
        -> THE OPERATING POINT for experiments 22 onward. Above it, no learning rule can show
           an effect. Report it explicitly and justify the choice in the methods chapter.

FIGURE
    21_capacity_and_representation_drift.png
    Two panels (4-class, 10-class): final task-1 accuracy vs hidden width, one line per
    readout/condition, with the random-init NCM floor drawn as a dashed reference.
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
from src.probes import (prototype_images, live_ncm_fn, frozen_ncm_fn, restricted_argmax_fn,
                        code_snapshot, code_drift)

# ============================ constants ============================
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR   = ROOT / "data"
FIG        = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE   = 14
BASE_SEED  = 0
N_RUNS     = 10

WIDTHS        = [2, 4, 8, 16, 32, 64, 128]
LOADS         = {"4 classes (2x2)": 2, "10 classes (2x5)": 5}   # classes per task
MAX_ITERS_PER_TASK = 600
BATCH         = 32
EVAL_EVERY    = 2
EVAL_PER_CLASS  = 100
PROTO_PER_CLASS = 50

ACC_THRESHOLD = 0.75      # below the tightest budget ceiling in exp 13; see note below
STOP_PATIENCE = 3

ARCH_BASE = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=64, out_dim=10,
                 act="tanh", bias=False, init="scaled_normal")
OBJ_FULL   = Objective(loss="mse", target="onehot", mask=False)
OBJ_MASKED = replace(OBJ_FULL, mask=True)
CONDITIONS = {"unmasked": OBJ_FULL, "masked": OBJ_MASKED}
BP_LR = 0.05

READOUTS = ["argmax", "argmax_r", "ncm_live", "ncm_frozen"]
# NOTE on ACC_THRESHOLD: at width 2-4 the threshold may be unreachable. That is a RESULT, not
# a failure -- the `reached` counter reports it and those cells fall back to the step cap.
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)

final = {(ld, w, c, r): [] for ld in LOADS for w in WIDTHS for c in CONDITIONS for r in READOUTS}
floor = {(ld, w): [] for ld in LOADS for w in WIDTHS}
drift = {(ld, w, c): [] for ld in LOADS for w in WIDTHS for c in CONDITIONS}
missed = {(ld, w, c): 0 for ld in LOADS for w in WIDTHS for c in CONDITIONS}
t0 = time.time()

for ld_name, k in LOADS.items():
    for width in WIDTHS:
        arch = replace(ARCH_BASE, hidden=width)
        for run in range(N_RUNS):
            seed = BASE_SEED + run
            d = np.random.default_rng(seed).permutation(10).tolist()
            tasks = [sorted(d[:k]), sorted(d[k:2 * k])]
            classes = sorted({c for t in tasks for c in t})
            stop_ev, rep_ev = make_eval_split(test, classes, EVAL_PER_CLASS, DEVICE, seed=seed)
            proto_x, proto_y = prototype_images(train, cidx, classes, PROTO_PER_CLASS,
                                                DEVICE, seed=seed)

            # ---- the floor: NCM on an UNTRAINED net of this width -------------------
            h0 = {}
            build_method("backprop", in_dim=arch.in_dim, hidden=width, out_dim=arch.out_dim,
                         arch=arch, obj=OBJ_FULL, lr=BP_LR, seed=seed, device=DEVICE, handle=h0)
            base_fn = live_ncm_fn(h0["features"], proto_x, proto_y)
            t1_mask = torch.isin(rep_ev[1], torch.tensor(tasks[0], device=DEVICE))
            floor[(ld_name, width)].append(
                (base_fn(rep_ev[0][t1_mask]) == rep_ev[1][t1_mask]).float().mean().item())

            for cname, obj in CONDITIONS.items():
                handle, frozen_p, snap = {}, {}, {}
                step_fn, predict = build_method("backprop", in_dim=arch.in_dim, hidden=width,
                                                out_dim=arch.out_dim, arch=arch, obj=obj,
                                                lr=BP_LR, seed=seed, device=DEVICE,
                                                handle=handle)
                feats = handle["features"]

                def on_task_end(ti, step, _f=feats, _fp=frozen_p, _s=snap,
                                _px=proto_x, _py=proto_y, _t1=tasks[0]):
                    if ti != 0:
                        return
                    pf = _f(_px).detach()
                    for c in sorted(set(_py.tolist())):
                        _fp[c] = pf[_py == c].mean(0)
                    _s["code"] = code_snapshot(_f, rep_ev[0][t1_mask])

                readouts = {
                    "argmax":     predict,
                    "argmax_r":   restricted_argmax_fn(predict, classes),
                    "ncm_live":   live_ncm_fn(feats, proto_x, proto_y),
                    "ncm_frozen": frozen_ncm_fn(feats, frozen_p),
                }
                out = run_classil(step_fn, predict, tasks, train, cidx,
                                  report_eval=rep_ev, stop_eval=stop_ev, readouts=readouts,
                                  max_iters_per_task=MAX_ITERS_PER_TASK, batch=BATCH,
                                  eval_every=EVAL_EVERY, device=DEVICE,
                                  stop_threshold=ACC_THRESHOLD, stop_patience=STOP_PATIENCE,
                                  data_seed=seed, on_task_end=on_task_end)
                missed[(ld_name, width, cname)] += sum(not r for r in out["reached"])
                for r in READOUTS:
                    final[(ld_name, width, cname, r)].append(out["curves"][r][-1, 0])
                if "code" in snap:
                    drift[(ld_name, width, cname)].append(
                        code_drift(snap["code"], code_snapshot(feats, rep_ev[0][t1_mask])))

        f_ = np.mean(floor[(ld_name, width)]) * 100
        um = np.mean(final[(ld_name, width, "unmasked", "argmax_r")]) * 100
        mk = np.mean(final[(ld_name, width, "masked", "argmax_r")]) * 100
        nl = np.mean(final[(ld_name, width, "masked", "ncm_live")]) * 100
        nf = np.mean(final[(ld_name, width, "masked", "ncm_frozen")]) * 100
        print(f"{ld_name:<18} h={width:>4} | argmax_r unmask {um:5.1f} mask {mk:5.1f} | "
              f"ncm live {nl:5.1f} frozen {nf:5.1f} | random-init floor {f_:5.1f} | "
              f"{time.time() - t0:5.0f}s")

# ------------------------------- tables -------------------------------
for ld_name in LOADS:
    print("\n" + "=" * 104)
    print(f"FINAL TASK-1 ACCURACY (%)  --  {ld_name},  {N_RUNS} pairings, threshold {ACC_THRESHOLD:.0%}")
    print(f"{'hidden':>7}" + "".join(f"{c[:4] + '/' + r[:9]:>16}"
                                     for c in CONDITIONS for r in ("argmax_r", "ncm_frozen"))
          + f"{'ncm_live':>11}{'floor':>8}{'drift cos':>11}")
    for w in WIDTHS:
        row = f"{w:>7}"
        for c in CONDITIONS:
            for r in ("argmax_r", "ncm_frozen"):
                v = np.array(final[(ld_name, w, c, r)]) * 100
                row += f"{v.mean():>10.1f}+/-{v.std():>4.1f}"
        row += f"{np.mean(final[(ld_name, w, 'masked', 'ncm_live')]) * 100:>11.1f}"
        row += f"{np.mean(floor[(ld_name, w)]) * 100:>8.1f}"
        ds = drift[(ld_name, w, "unmasked")]
        row += f"{np.mean([d['cosine'] for d in ds]) if ds else float('nan'):>11.3f}"
        print(row)
    miss = {k[1:]: v for k, v in missed.items() if k[0] == ld_name and v}
    if miss:
        print(f"  threshold never reached in: {miss}")

print("\nHOW TO READ THIS")
print("  ncm_live ~= floor            -> hidden layer is doing no task-specific work")
print("  ncm_frozen << ncm_live       -> the code moved but stayed decodable (real drift)")
print("  masked argmax_r drops with w -> capacity now binding; a learning rule has room to act")
print("  pick the OPERATING POINT as the largest width where masked argmax_r is clearly")
print("  below ceiling; run experiments 22+ there, not at hidden=64.")

# ------------------------------- figure -------------------------------
fig, axes = plt.subplots(1, len(LOADS), figsize=(7 * len(LOADS), 5), sharey=True)
axes = np.atleast_1d(axes)
STYLE = {("unmasked", "argmax_r"): ("tab:red", "-", "unmasked, argmax"),
         ("masked", "argmax_r"):   ("tab:blue", "-", "masked, argmax"),
         ("masked", "ncm_live"):   ("tab:green", "--", "masked, NCM (live prototypes)"),
         ("masked", "ncm_frozen"): ("tab:purple", "--", "masked, NCM (frozen at switch)")}
for ax, ld_name in zip(axes, LOADS):
    for (c, r), (col, ls, lab) in STYLE.items():
        m = np.array([np.mean(final[(ld_name, w, c, r)]) for w in WIDTHS]) * 100
        s = np.array([np.std(final[(ld_name, w, c, r)]) for w in WIDTHS]) * 100
        ax.plot(WIDTHS, m, color=col, ls=ls, lw=2.2, marker="o", ms=4, label=lab)
        ax.fill_between(WIDTHS, m - s, m + s, color=col, alpha=0.12)
    fl = np.array([np.mean(floor[(ld_name, w)]) for w in WIDTHS]) * 100
    ax.plot(WIDTHS, fl, color="gray", ls=":", lw=1.8, label="NCM floor (untrained net)")
    ax.axhline(100 / (2 * LOADS[ld_name]), color="k", lw=0.8, ls="-.", alpha=0.5)
    ax.set_xscale("log", base=2); ax.set_xticks(WIDTHS)
    ax.set_xticklabels(WIDTHS); ax.set_xlabel("hidden units")
    ax.set_title(ld_name); ax.set_ylim(-2, 103); ax.grid(alpha=0.2)
axes[0].set_ylabel("final task-1 accuracy (%)")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("When does the hidden layer start to matter? (backprop, "
             f"{N_RUNS} pairings; dash-dot = collapse floor)")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}")

np.savez(FIG.with_suffix(".npz"),
         **{f"{ld}|{w}|{c}|{r}": np.array(v) for (ld, w, c, r), v in final.items()},
         **{f"floor|{ld}|{w}": np.array(v) for (ld, w), v in floor.items()},
         widths=WIDTHS, threshold=ACC_THRESHOLD)
