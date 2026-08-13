"""Does PC show the higher target alignment [R1] Fig 3b claims, and if so does that alignment
buy it less interference on task 1?

WHY THIS IS THE RIGHT NEXT EXPERIMENT
    The A series (52, 53, 56, 57, 59, 60) established that PC does not forget less than backprop
    anywhere we have looked. But script 54 established that PC's updates genuinely ARE different
    -- cos 0.814 against backprop's on W2, 0.197 for EqProp. So credit assignment changes without
    the forgetting trade-off changing, and the obvious question is why those two come apart.

    [R1]'s answer would be target alignment. Their claim is that prospective configuration moves
    the outputs more directly toward their targets, and that this is what reduces interference.
    That claim is made in a per-update quantity, so it can be tested directly rather than
    inferred from accuracy curves -- and it is BUDGET-INDEPENDENT BY CONSTRUCTION, which no
    accuracy metric in this project is.

TWO NUMBERS, AND THE SECOND IS THE ONE THAT MATTERS HERE
    alignment     cos(target - out_before, out_after - out_before) on the batch being trained.
                  [R1]'s own measure, exactly as `knowledge_base.md` §11.4 records it. How much
                  of the update goes toward the current target.
    interference  the same cosine on a FIXED TASK-1 BATCH that is not being trained, measured
                  during task 2. How much of each update goes toward -- or away from -- task 1's
                  targets. Negative is forgetting caught per update instead of at an endpoint.

    Alignment is [R1]'s mechanism; interference is what the mechanism is supposed to buy. Running
    both in one pass is the point: it can separate "PC is not more aligned here" from "PC is more
    aligned and alignment does not control forgetting".

PRE-COMMITTED READINGS -- what each outcome would mean
    PC more aligned AND less interfering  -> [R1]'s mechanism reproduces, and the A-series null
                                             needs explaining as a failure to CONVERT it.
    PC more aligned, interference equal   -> alignment is real but is not what sets forgetting.
                                             This is the strongest result available here: it
                                             would locate exactly where their argument breaks.
    PC not more aligned                   -> the mechanism does not reproduce at this scale, and
                                             the A-series null is unsurprising rather than
                                             puzzling. Then the depth axis (55/59) is the place
                                             their claim could still live.
    EqProp: 54 measured its update as nearly orthogonal to backprop's. If orthogonal updates
    still give ordinary alignment, alignment is not sensitive to what 54 measured, and the two
    instruments are answering different questions.

DEVIATIONS FROM THE PROTOCOL, STATED
    * Alignment is measured every 5th update, not every update (`probes.alignment_probe`'s
      `every`). It is averaged over a window, so subsampling costs precision in the mean and
      nothing else; it is what makes the probe affordable for EqProp, where each measurement is
      a full relaxation rather than a forward pass.
    * Fixed budgets per task, as script 52, so the accuracy curves are comparable with it.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.protocol import (PROTOCOL, load, build, replace,
                          figure_path as _figure_path, array_path as _array_path)
from src.runner import run_classil
from src.probes import alignment_probe
from src.metrics import metric_grid, report_grid, paired_diff
from src.plotting import plot_learning_curves

SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):      # noqa: F811
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):       # noqa: F811
    return _array_path(f, _tag(f, suffix))


# ---------------------------------------------------------------- settings
METHODS = ["backprop", "replay", "pc", "eqprop"]
SEEDS = 5
EVAL_EVERY = 10
EVERY = 5                      # measure alignment on every 5th update
REF_N = 200                    # fixed task-1 images the interference probe watches
COLORS = {"backprop": "tab:gray", "replay": "tab:brown",
          "pc": "tab:red", "eqprop": "tab:green"}
TASK_COLORS = ["tab:orange", "tab:blue"]

# ---- read, not chosen: identical to 52 and 60 -----------------------------
HIDDEN = int(np.load(_array_path(str(ROOT / "experiments" /
                                     "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(_array_path(str(ROOT / "experiments" /
                              "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
SETTLE_TOL, EQ_MAX_STEPS = float(z51["settle_tol"]), int(z51["eq_max_steps"])
PC_STEPS = int(z51["pc_steps"])
T = float(z51["target_steps"])
ITERS = [int(2 * T), int(1.5 * T)]

if SMOKE:
    SEEDS, ITERS, REF_N = 2, [40, 30], 40
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} | domain-IL | {ITERS[0]}+{ITERS[1]} updates | {SEEDS} seeds | "
      f"alignment every {EVERY} updates")
print("  question: is PC more target-aligned than backprop, and does that buy it less "
      "interference?\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il", stop_threshold=None,
               max_iters_per_task=ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)


def settle_kw(method):
    if method == "pc":
        return dict(steps=PC_STEPS)
    if method == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


def ref_batch(data, tasks, lmap, n):
    """A fixed batch of TASK-1 images with the units they were trained on.

    Drawn from the reporting eval split, never from training data: the probe must not be able
    to influence the run, and it must not be measured on images the buffer could hold."""
    x, y = data.report_eval
    keep = torch.zeros(len(y), dtype=torch.bool)
    for c in tasks[0]:
        keep |= (y == c)
    idx = keep.nonzero(as_tuple=True)[0][:n]
    yy = y[idx]
    if lmap is not None:
        yy = torch.tensor([lmap[int(v)] for v in yy], device=yy.device)
    return x[idx], yy


# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    steps, switches = z["steps"], list(z["switches"])
    curves = {m: z[f"argmax_{m}"] for m in METHODS}
    align = {m: z[f"align_{m}"] for m in METHODS}
    inter = {m: z[f"inter_{m}"] for m in METHODS}
    upd = {m: z[f"upd_{m}"] for m in METHODS}
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(base)
    curves, align, inter, upd = ({m: [] for m in METHODS} for _ in range(4))
    t0 = time.perf_counter()
    for m in METHODS:
        proto = replace(base, lr={m: LR[m]})
        for seed in range(SEEDS):
            tasks = proto.tasks(seed)
            lmap = proto.label_map(tasks)
            handle = {}
            train_step, predict = build(proto, m, seed, handle=handle, **settle_kw(m))
            wrapped, a_log, i_log = alignment_probe(
                train_step, predict, handle["arch"], handle["obj"], device=proto.device,
                every=EVERY, ref=ref_batch(data, tasks, lmap, REF_N))
            out = run_classil(
                wrapped, predict, tasks, data.train, data.class_idx,
                report_eval=data.report_eval, stop_eval=data.stop_eval,
                max_iters_per_task=ITERS, batch=proto.batch, eval_every=EVAL_EVERY,
                device=proto.device, stop_threshold=None, data_seed=seed, label_map=lmap)
            curves[m].append(out["curves"]["argmax"])
            align[m].append([v for _, v in a_log])
            inter[m].append([v for _, v in i_log])
            upd[m].append([i for i, _ in a_log])
        steps, switches = out["steps"], out["switches"]
        for d in (curves, align, inter, upd):
            d[m] = np.array(d[m], dtype=float)
        print(f"  {m:10s} done  [{time.perf_counter() - t0:6.0f}s]")

sw = switches[0]

# ---------------------------------------------------------------- readings
# Alignment is split at the switch because the two halves answer different questions: during
# task 1 it is [R1]'s claim in its original setting (one task, learning it), during task 2 it is
# the same rule learning something new while task 1 sits underneath.
print(f"\n  {'rule':10s} {'align t1':>10s} {'align t2':>10s} {'INTERFERENCE t2':>17s}")
stat = {}
for m in METHODS:
    u = upd[m][0]
    pre, post = u < sw, u >= sw
    stat[m] = dict(
        a1=align[m][:, pre].mean(1), a2=align[m][:, post].mean(1),
        i2=inter[m][:, post].mean(1), i1=inter[m][:, pre].mean(1))
    s = stat[m]
    print(f"  {m:10s} {s['a1'].mean():+10.3f} {s['a2'].mean():+10.3f} "
          f"{s['i2'].mean():+17.3f}")

print(f"\n  paired against backprop, per seed:")
for m in METHODS[1:]:
    for k, lab in (("a2", "alignment during task 2"), ("i2", "interference on task 1")):
        d, se, n = paired_diff(stat[m][k], stat["backprop"][k])
        flag = "  *" if n > 2 else ""
        print(f"    {m:10s} {lab:26s} {d:+7.3f} +-{se:5.3f}  {n:4.1f}sem{flag}")

a_pc, a_bp = stat["pc"]["a2"].mean(), stat["backprop"]["a2"].mean()
i_pc, i_bp = stat["pc"]["i2"].mean(), stat["backprop"]["i2"].mean()
d_a = paired_diff(stat["pc"]["a2"], stat["backprop"]["a2"])
d_i = paired_diff(stat["pc"]["i2"], stat["backprop"]["i2"])
print("\n  READING (pre-committed in the docstring):")
if d_a[2] > 2 and d_a[0] > 0:
    if d_i[2] > 2 and d_i[0] > 0:
        print("    PC is more aligned AND interferes less -- [R1]'s mechanism reproduces here,")
        print("    so the A-series null is a failure to CONVERT it into retention.")
    else:
        print("    PC IS more target-aligned than backprop, and interference is unchanged.")
        print("    Alignment is real and is NOT what sets forgetting in this task family --")
        print("    which locates exactly where [R1]'s argument stops carrying.")
else:
    print("    PC is not more target-aligned than backprop here, so [R1]'s mechanism does not")
    print("    reproduce at this scale and the A-series null is unsurprising rather than")
    print("    puzzling. Depth (55/59) is where their claim could still live.")

report_grid({m: metric_grid(steps, curves[m], sw) for m in METHODS},
            METHODS, control="backprop", primary="crossover")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
for ax, (D, lab) in zip(axes, [(align, "target alignment  (on the batch being trained)"),
                               (inter, "interference  (task-1 batch, not being trained)")]):
    for m in METHODS:
        u = upd[m][0]
        for r in range(D[m].shape[0]):
            ax.plot(u, D[m][r], color=COLORS[m], lw=0.6, alpha=0.20)
        ax.plot(u, D[m].mean(0), color=COLORS[m], lw=2.4, label=m)
    ax.axvline(sw, color="k", lw=0.9, ls="--")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("update")
    ax.set_title(lab, fontsize=10)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("cos(target - out_before, out_after - out_before)")
axes[0].legend(fontsize=8)
fig.suptitle("Target alignment and interference, per update. Dashed line = task switch. "
             f"Domain-IL, H={HIDDEN}, {SEEDS} seeds, measured every {EVERY} updates.",
             fontsize=10)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

plot_learning_curves(
    steps, curves, METHODS, figure_path(__file__, "accuracy"),
    blocks=[(0, sw, 0), (sw, steps[-1], 1)], ncols=2, task_colors=TASK_COLORS,
    task_labels=["task 1", "task 2"], crossover_after=sw,
    title=f"Accuracy for the same runs -- {base.describe()}",
    legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False))

# ---------------------------------------------------------------- save
if not SMOKE:
    np.savez(array_path(__file__), steps=steps, switches=switches,
             methods=np.array(METHODS), every=EVERY,
             **{f"argmax_{m}": curves[m] for m in METHODS},
             **{f"align_{m}": align[m] for m in METHODS},
             **{f"inter_{m}": inter[m] for m in METHODS},
             **{f"upd_{m}": upd[m] for m in METHODS})
    print(f"saved {array_path(__file__)}")
