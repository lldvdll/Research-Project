"""What happens when the two tasks alternate many times instead of switching once?

WHY THIS IS A DIFFERENT QUESTION FROM THE A SERIES
    Every comparison so far ran ONE switch: learn task 1, learn task 2, measure. That measures a
    single displacement. Alternating the tasks repeatedly asks whether the network is converging
    on a joint solution or merely being dragged back and forth, and it is the schedule [R1] Fig
    4d actually uses -- so it is also the closest this project comes to their protocol.

    Two outcomes are distinguishable and neither is obvious in advance:
      CONVERGING  each cycle ends closer to a state that serves both tasks. Accuracy on the
                  unattended task decays less each time, the (task1, task2) trajectory spirals
                  IN toward the joint corner, and successive same-task weight states get closer.
      PING-PONG   the network oscillates between two fixed solutions, one per task, learning
                  nothing cumulative. The trajectory traces a closed loop rather than a spiral,
                  and successive same-task weight states stop converging.
      CHAOTIC     successive same-task weight states neither converge nor repeat -- the network
                  keeps finding new solutions and the weight-space picture wanders.

    The weight trace separates these directly: the distance between the weight state at the end
    of one task-1 block and the end of the NEXT task-1 block goes to zero if it is ping-ponging
    or converging, and does not if it is chaotic.

WHAT IS MEASURED, PER BLOCK, SO EVERY QUANTITY IS A TIME SERIES
    crossover height  the accuracy at which the two curves meet WITHIN that block. In a task-2
                      block task 1 falls through task 2; in a task-1 block the reverse, so the
                      arguments are swapped and the same geometric quantity is recovered.
    half-life         updates for the UNATTENDED task to fall to half its value at block start.
                      MEASURED AND FOUND UNUSABLE HERE TOO: defined on ~1 of 100 block x seed
                      cells. Once the network approaches the joint solution the unattended task
                      barely decays inside a block, so it never halves. Alternating does not
                      rescue the metric; it is reported so that is on the record rather than
                      assumed either way.
    mean test error   [R1]'s headline number, per block and cumulative. Reported throughout
                      because their claim is made in it -- and it rewards fast learning and low
                      forgetting together, so it is never read alone.

DEVIATIONS FROM THE PROTOCOL, STATED
    * The schedule. tasks = [T1, T2] x N_CYCLES instead of one switch.
    * A shorter budget per block than the calibrated time-to-competence, so a block cannot fully
      erase the other task and the cycle structure is visible.
    * EqProp dropped: ~350x backprop per update makes 3000 updates per seed impractical, and the
      question is about the schedule rather than about that rule.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.protocol import (PROTOCOL, load, build, replace,
                          figure_path as _figure_path, array_path as _array_path)
from src.runner import run_classil
from src.probes import weight_trace_probe
from src.metrics import crossover, half_life, mean_test_error
from src.plotting import plot_learning_curves

SMOKE = "--smoke" in sys.argv


def _tag(f, s):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (s + "_SMOKE").lstrip("_") if (SMOKE and own) else s


def figure_path(f, s=""):      # noqa: F811
    return _figure_path(f, _tag(f, s))


def array_path(f, s=""):       # noqa: F811
    return _array_path(f, _tag(f, s))


# ---------------------------------------------------------------- settings
METHODS = ["backprop", "replay", "pc"]
COLORS = {"backprop": "tab:gray", "replay": "tab:brown", "pc": "tab:red"}
TASK_COLORS = ["tab:orange", "tab:blue"]
N_CYCLES = 10                  # 10 of each task -> 20 blocks
BLOCK = 150                    # updates per block; deliberately below time-to-competence
SEEDS, EVAL_EVERY, TRACE_EVERY = 5, 5, 5
ILLUS_SEED, ILLUS_LAYER = 0, "W1"

HIDDEN = int(np.load(_array_path(str(ROOT / "experiments" /
                                     "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(_array_path(str(ROOT / "experiments" /
                              "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
PC_STEPS = int(z51["pc_steps"])

if SMOKE:
    SEEDS, N_CYCLES, BLOCK = 2, 3, 30
    print("--smoke: tiny budget, results are NOT meaningful\n")

N_BLOCKS = 2 * N_CYCLES
print(f"H = {HIDDEN} | domain-IL | {N_BLOCKS} blocks x {BLOCK} updates "
      f"({N_BLOCKS * BLOCK} total) | {SEEDS} seeds | {', '.join(METHODS)}")
print("  question: does repeated switching converge on a joint solution, ping-pong, "
      "or wander?\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il", stop_threshold=None,
               max_iters_per_task=BLOCK, eval_every=EVAL_EVERY, seeds=SEEDS)


def settle_kw(m):
    return dict(steps=PC_STEPS) if m == "pc" else {}


# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    curves = {m: z[f"argmax_{m}"] for m in METHODS}
    steps, switches = z["steps"], list(z["switches"])
    xh = {m: z[f"xh_{m}"] for m in METHODS}
    hl = {m: z[f"hl_{m}"] for m in METHODS}
    me = {m: z[f"me_{m}"] for m in METHODS}
    same_task_gap = {m: z[f"gap_{m}"] for m in METHODS}
    traces = {m: z[f"trace_{m}"] for m in METHODS}
    print("--replot: from saved arrays, no training\n")
else:
    data = load(base)
    curves, traces, t0 = {}, {}, time.perf_counter()
    for m in METHODS:
        proto = replace(base, lr={m: LR[m]})
        rows, tr_seed0 = [], None
        for seed in range(SEEDS):
            t12 = proto.tasks(seed)
            sched = [t12[i % 2] for i in range(N_BLOCKS)]     # T1, T2, T1, T2, ...
            lmap = proto.label_map(t12)
            handle = {}
            ts, pr = build(proto, m, seed, handle=handle, **settle_kw(m))
            wrapped, tr = (weight_trace_probe(ts, handle["params"], keys=(ILLUS_LAYER,),
                                              every=TRACE_EVERY)
                           if seed == ILLUS_SEED else (ts, None))
            out = run_classil(
                wrapped, pr, sched, data.train, data.class_idx,
                report_eval=data.report_eval, stop_eval=data.stop_eval,
                max_iters_per_task=BLOCK, batch=proto.batch, eval_every=EVAL_EVERY,
                device=proto.device, stop_threshold=None, data_seed=seed, label_map=lmap)
            # every even column is the same class set, likewise every odd one, so two columns
            # carry everything: column 0 is task 1's accuracy, column 1 is task 2's
            rows.append(out["curves"]["argmax"][:, :2])
            if tr is not None:
                tr_seed0 = np.stack(tr[ILLUS_LAYER])
        steps, switches = out["steps"], out["switches"]
        curves[m] = np.stack(rows)
        traces[m] = tr_seed0
        print(f"  {m:10s} done  [{time.perf_counter() - t0:5.0f}s]")

    # ---- per-block metrics -------------------------------------------------
    xh, hl, me, same_task_gap = {}, {}, {}, {}
    bounds = [0] + list(switches)
    for m in METHODS:
        A = curves[m] * 100
        X = np.full((SEEDS, N_BLOCKS), np.nan)
        H = np.full((SEEDS, N_BLOCKS), np.nan)
        E = np.full((SEEDS, N_BLOCKS, 2), np.nan)
        for r in range(SEEDS):
            for b in range(N_BLOCKS):
                sel = (np.asarray(steps) > bounds[b]) & (np.asarray(steps) <= bounds[b + 1])
                if sel.sum() < 3:
                    continue
                s_b, c_b = np.asarray(steps)[sel], A[r][sel]
                trained, other = b % 2, 1 - b % 2
                # the task NOT being trained is the one that decays in this block
                X[r, b] = crossover(s_b, c_b[:, other], c_b[:, trained], after=bounds[b])[1]
                H[r, b] = half_life(s_b, c_b[:, other], after=bounds[b],
                                    peak=c_b[0, other]) or np.nan
                E[r, b] = mean_test_error(s_b, c_b / 100.0, after=bounds[b])
        xh[m], hl[m], me[m] = X, H, E
        # ping-pong vs chaotic: distance between the weight state at the end of one task-1
        # block and the end of the next task-1 block. Converging or ping-ponging -> shrinks.
        T = traces[m]
        idx = [min(int(round(bounds[b + 1] / TRACE_EVERY)) - 1, len(T) - 1)
               for b in range(N_BLOCKS)]
        ends = T[idx]
        same_task_gap[m] = np.array([np.abs(ends[b + 2] - ends[b]).sum()
                                     for b in range(N_BLOCKS - 2)])

# ---------------------------------------------------------------- readings
print(f"\n  PER-BLOCK METRICS, mean over {SEEDS} seeds "
      f"(blocks alternate: even = task 1 trained, odd = task 2 trained)")
print(f"  {'rule':10s} {'block':>6s} {'crossover':>10s} {'half-life':>10s} "
      f"{'mean err t1':>12s} {'mean err t2':>12s}")
for m in METHODS:
    for b in sorted({0, 1, N_BLOCKS // 2, N_BLOCKS // 2 + 1, N_BLOCKS - 2, N_BLOCKS - 1}):
        print(f"  {m:10s} {b:6d} {np.nanmean(xh[m][:, b]):10.1f} "
              f"{np.nanmean(hl[m][:, b]):10.1f} "
              f"{np.nanmean(me[m][:, b, 0]) * 100:12.1f} "
              f"{np.nanmean(me[m][:, b, 1]) * 100:12.1f}")

# HOW OFTEN IS EACH METRIC EVEN DEFINED? Both censor, and both censor in the direction that
# flatters the better rule: crossover is undefined when the two curves never meet inside a
# block, half-life when the unattended task never loses half its value there. A rule that is
# retaining well produces MORE undefined blocks, so a mean over the defined ones is taken over
# that rule's worst blocks. Print the counts before any trend is read off the series.
print(f"\n  DEFINED ON HOW MANY OF {N_BLOCKS * SEEDS} block x seed cells?")
print(f"  {'rule':10s} {'crossover':>12s} {'half-life':>12s}")
for m in METHODS:
    print(f"  {m:10s} {int(np.isfinite(xh[m]).sum()):8d}/{N_BLOCKS * SEEDS:<4d}"
          f" {int(np.isfinite(hl[m]).sum()):8d}/{N_BLOCKS * SEEDS:<4d}")
print("  An undefined block is a RESULT: the curves did not meet, or the unattended task did")
print("  not halve. Read the trend only where the count is high; a series computed from a few")
print("  surviving blocks is a series about those blocks.")

print(f"\n  DOES IT CONVERGE?  crossover height, first cycle -> last cycle")
for m in METHODS:
    a = np.nanmean(xh[m][:, :2]); b_ = np.nanmean(xh[m][:, -2:])
    print(f"  {m:10s} {a:6.1f}%  ->  {b_:6.1f}%   ({b_ - a:+5.1f})")

print(f"\n  DOES IT SETTLE IN WEIGHT SPACE?  L1 distance between successive same-task")
print(f"  {ILLUS_LAYER} states (seed {ILLUS_SEED}). Shrinking = converging or ping-ponging;")
print(f"  flat = still finding new solutions each cycle.")
for m in METHODS:
    g = same_task_gap[m]
    print(f"  {m:10s} first {g[0]:8.2f}  last {g[-1]:8.2f}  ratio {g[-1] / g[0]:5.2f}")

print(f"\n  [R1]'s MEAN TEST ERROR over the whole run (lower better), per task:")
for m in METHODS:
    print(f"  {m:10s} task 1 {np.nanmean(me[m][:, :, 0]) * 100:5.1f}%   "
          f"task 2 {np.nanmean(me[m][:, :, 1]) * 100:5.1f}%   "
          f"mean {np.nanmean(me[m]) * 100:5.1f}%")
print("  Reported because [R1]'s claim is made in this number. It rewards fast learning and")
print("  low forgetting together and cannot separate them, so it is never read alone.")

# ---------------------------------------------------------------- figures
blocks_shade = [(bounds[b], bounds[b + 1], b % 2) for b in range(N_BLOCKS)] \
    if not REPLOT else [(switches[b - 1] if b else 0, switches[b], b % 2)
                        for b in range(N_BLOCKS)]
plot_learning_curves(steps, curves, METHODS, figure_path(__file__, "accuracy"),
                     blocks=blocks_shade, ncols=1, task_colors=TASK_COLORS,
                     task_labels=["task 1", "task 2"],
                     title=f"Accuracy under {N_BLOCKS} alternating blocks of {BLOCK} updates",
                     legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False))

# ---- the spiral --------------------------------------------------------
fig, axes = plt.subplots(1, len(METHODS), figsize=(5.0 * len(METHODS), 5.2))
for ax, m in zip(np.atleast_1d(axes), METHODS):
    A = curves[m] * 100
    for r in range(A.shape[0]):
        ax.plot(A[r, :, 0], A[r, :, 1], color="gray", lw=0.6, alpha=0.25)
    M = A.mean(0)
    pts = M.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap="viridis", lw=2.4,
                        norm=plt.Normalize(0, len(M)))
    lc.set_array(np.arange(len(M)))
    ax.add_collection(lc)
    ax.plot(M[0, 0], M[0, 1], "o", color="k", ms=8, label="start")
    ax.plot(M[-1, 0], M[-1, 1], "s", color="k", ms=8, label="end")
    ax.plot([100, 0], [0, 100], ls=":", color="gray", lw=1)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.set_aspect("equal")
    ax.set_xlabel("task 1 accuracy (%)"); ax.set_ylabel("task 2 accuracy (%)")
    ax.set_title(m, fontsize=11); ax.grid(alpha=0.25)
np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
fig.colorbar(lc, ax=np.atleast_1d(axes).tolist(), label="evaluation index (time)",
             fraction=0.02)
fig.suptitle("THE TRAJECTORY: task 1 against task 2 accuracy under repeated switching. "
             "Colour = time.\nSpiralling IN toward the upper-right = converging on a joint "
             "solution. A closed loop = ping-ponging between two solutions.", fontsize=10)
fig.savefig(figure_path(__file__, "spiral"), dpi=120, bbox_inches="tight")
print(f"saved {figure_path(__file__, 'spiral')}")

# ---- metrics over blocks + the weight-space question --------------------
fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
bx = np.arange(N_BLOCKS)
for m in METHODS:
    axes[0][0].plot(bx, np.nanmean(xh[m], 0), "o-", color=COLORS[m], ms=4, label=m)
    axes[0][1].plot(bx, np.nanmean(hl[m], 0), "o-", color=COLORS[m], ms=4, label=m)
    axes[1][0].plot(bx, np.nanmean(me[m], 0).mean(1) * 100, "o-", color=COLORS[m], ms=4,
                    label=m)
    axes[1][1].plot(np.arange(len(same_task_gap[m])), same_task_gap[m], "o-",
                    color=COLORS[m], ms=4, label=m)
axes[0][0].set_ylabel("crossover height (%)"); axes[0][0].set_xlabel("block")
axes[0][0].set_title("crossover height per block", fontsize=10)
axes[0][1].set_ylabel("half-life of the unattended task (updates)")
axes[0][1].set_xlabel("block")
axes[0][1].set_title("half-life per block — STILL UNDEFINED (see counts)", fontsize=10)
axes[0][1].annotate(
    "Alternating does NOT rescue half-life.\nOnce the network approaches the joint\n"
    "solution the unattended task barely\ndecays inside a block, so it never halves.\n"
    "Defined on ~1 of 100 block x seed cells.",
    xy=(0.5, 0.5), xycoords="axes fraction", ha="center", va="center", fontsize=9,
    color="dimgray")
axes[1][0].set_ylabel("[R1] mean test error (%)"); axes[1][0].set_xlabel("block")
axes[1][0].set_title("[R1]'s headline metric per block", fontsize=10)
axes[1][1].set_ylabel(f"L1 distance, successive same-task {ILLUS_LAYER}")
axes[1][1].set_xlabel("block pair")
axes[1][1].set_title("ping-pong or chaotic?  shrinking = settling", fontsize=10)
for ax in axes.ravel():
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
fig.suptitle("Metrics as the tasks alternate. Every quantity is a time series, which one "
             "switch cannot provide.", fontsize=10)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"saved {figure_path(__file__)}")

# ---- crossover against half-life ---------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 5.6))
for m in METHODS:
    x, y = np.nanmean(xh[m], 0), np.nanmean(hl[m], 0)
    ok = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[ok], y[ok], c=bx[ok], cmap="viridis", s=52,
               edgecolor=COLORS[m], linewidth=1.8, label=m)
ax.set_xlabel("crossover height (%)"); ax.set_ylabel("half-life (updates)")
ax.set_title("Crossover against half-life, one point per block. Fill = block index (time),\n"
             "outline = rule. Movement up-right over time means both metrics improving.",
             fontsize=10)
ax.grid(alpha=0.25); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figure_path(__file__, "metric_scatter"), dpi=120, bbox_inches="tight")
print(f"saved {figure_path(__file__, 'metric_scatter')}")

if not SMOKE and not REPLOT:
    np.savez(array_path(__file__), steps=steps, switches=switches,
             methods=np.array(METHODS), n_blocks=N_BLOCKS, block=BLOCK,
             **{f"argmax_{m}": curves[m] for m in METHODS},
             **{f"xh_{m}": xh[m] for m in METHODS},
             **{f"hl_{m}": hl[m] for m in METHODS},
             **{f"me_{m}": me[m] for m in METHODS},
             **{f"gap_{m}": same_task_gap[m] for m in METHODS},
             **{f"trace_{m}": traces[m] for m in METHODS})
    print(f"saved {array_path(__file__)}")
