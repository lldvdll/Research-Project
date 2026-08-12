"""Does predictive coding reduce forgetting once the network is deeper, or wider, or both?

THE LAST EXPERIMENT OF THE A SERIES, and the one that gives Song & Bogacz's claim its best
chance. [R1] report less forgetting under prospective configuration and argue that depth is part
of why. Every comparison in this project so far has run ONE hidden layer of 32 units, which is
the least favourable place to look for either.

WHAT IS SWEPT, AND WHY BOTH AXES
    DEPTH, because [R1]'s mechanism needs somewhere to act. Prospective configuration settles
    hidden activity before computing weight changes; with a single hidden layer, script 55
    measured PC's input-layer update at cos 0.985 with backprop's, so there is very little room
    for it to differ where it matters.

    WIDTH, because H=32 was fixed by script 41 as the smallest width within 1 point of the best,
    and "adequate" is not the same as "roomy". At 32 units both tasks must SHARE the same 32
    hidden dimensions, so some interference is forced by capacity whatever the learning rule
    does. If PC's benefit is about allocating representation rather than protecting it, a tight
    network hides it. Script 41 measured 32 -> 64 as +0.5 points of joint accuracy, so the
    headroom is real but modest; 128 is where its curve flattens.

    Both together, because the interaction is where an effect is most likely to hide, and
    running them as two separate scripts would never measure it.

WHAT THIS FIXES ABOUT SCRIPT 55
    55 ran depths 1-3 with the learning rates script 51 calibrated AT DEPTH 1. Its endpoint
    table then reported PC keeping 14 points more than backprop at depth 3, which was PC being
    slower at an unsuitable rate and travelling less far along a shared curve -- the retention
    curve showed the two superimposed. A feasibility probe for this script confirms the cause
    directly: at depth 5 backprop reaches competence at lr=0.005 while PC does not reach it at
    all below 0.05. A single learning rate cannot serve both.

    So EVERY CELL IS CALIBRATED SEPARATELY. Within each (depth, width) cell each rule gets the
    learning rate that brings it closest to a common time-to-competence on task 1, measured on
    task 1 alone before task 2 exists, exactly as script 51 does.

THE THREE GATES THIS RUN MUST PASS BEFORE ITS NUMBERS MEAN ANYTHING
    Each one is a mistake this project has already made, so each is checked rather than assumed.

      1. matched task-1 competence   every rule enters task 2 at the same standard. Reported.
      2. matched task-2 competence   both tasks stop on ACCURACY, and the task-2 threshold is
                                     MEASURED per cell under this script's own cap -- script 53
                                     inherited a threshold from a shorter run twice and was
                                     void both times. A cell whose rules end at different
                                     task-2 accuracies is not a comparison and is flagged, as
                                     script 58's legacy cells should have been.
      3. paired against backprop     rules share the class split and the initialisation at a
                                     given seed, so the comparison is per seed. On script 53's
                                     runs, group means put the POSITIVE CONTROL at 0.7 sem and
                                     read as failure; paired, the same runs give 5.1 sem.

    Replay is the positive control and must separate. If it does not, the cell is unreadable.

    EqProp is not run. It costs ~350x backprop per update, script 57 found it consistently worse
    than backprop, and [R1]'s claim is about prospective configuration.

READINGS COMMITTED BEFORE RUNNING
    If PC separates from backprop at some depth or width and not at 1x32, the earlier null
    results were a property of the network we chose, not of the rule, and the whole comparison
    should move to that size.
    If PC stays within a couple of points of backprop everywhere while replay separates in every
    cell, then PC does not reduce forgetting in this task family at any size tried -- which,
    given scripts 52, 53, 56 and 57 all agree, is the result the project reports.
    If the effect appears only in the widest and deepest cell, treat it as a lead needing its
    own replication rather than a finding, because that is the cell with the fewest constraints
    on it.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np

from src.protocol import PROTOCOL, load, run, replace, figure_path as _figure_path, \
    array_path as _array_path
from src.metrics import paired_diff
from src.plotting import plot_retention_curve

SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):
    return _array_path(f, _tag(f, suffix))


# ---------------------------------------------------------------- settings
DEPTHS = [1, 2, 3, 5]
WIDTHS = [32, 128]
METHODS = ["backprop", "replay", "pc"]
SEEDS = 5
CAL_SEEDS = 2                   # the calibration only needs to rank learning rates
EVAL_EVERY = 20
T1_THRESHOLD = 0.90
STOP_PATIENCE = 3
MAX_ITERS = 2500
TARGET_STEPS = 420              # the common time-to-competence, as in script 51
PC_STEPS = 50                   # verified fully settled by script 50 (needs <= 18 at 1x32)
COLORS = {"backprop": "tab:gray", "replay": "tab:brown", "pc": "tab:red"}

# PC needs a HIGHER rate as depth grows -- the feasibility probe found it unable to reach
# competence at depth 5 below 0.05, where backprop gets there at 0.005. The grid spans both.
LR_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

if SMOKE:
    DEPTHS, WIDTHS, SEEDS, CAL_SEEDS = [1, 2], [32], 2, 1
    MAX_ITERS, TARGET_STEPS, LR_GRID = 120, 60, [0.02, 0.05]
    print("--smoke: tiny budget, results are NOT meaningful\n")

CELLS = [(d, w) for w in WIDTHS for d in DEPTHS]
print(f"depth x width factorial: {[f'{d}x{w}' for d, w in CELLS]}")
print(f"  {SEEDS} seeds | task 1 to {T1_THRESHOLD:.0%} | task 2 threshold measured per cell\n")


def settle_kw(m):
    return dict(steps=PC_STEPS) if m == "pc" else {}


def base_for(d, w):
    return replace(PROTOCOL, hidden=w, n_layers=d, scenario="domain_il",
                   stop_patience=STOP_PATIENCE, max_iters_per_task=MAX_ITERS,
                   eval_every=EVAL_EVERY, seeds=SEEDS)


def calibrate(b, data):
    """Per-rule learning rate bringing task-1 time-to-competence closest to TARGET_STEPS.

    Measured on task 1 ALONE, before task 2 exists, so it cannot be tuned on the forgetting
    result. Only rates reaching threshold on EVERY calibration seed are eligible -- one seed
    failing is an unstable rate, not a fast one."""
    out = {}
    for m in METHODS:
        best, best_d = None, np.inf
        for lr in LR_GRID:
            p = replace(b, stop_threshold=T1_THRESHOLD, lr={m: lr})
            steps = []
            for s in range(CAL_SEEDS):
                o = run(p, m, s, data=data, tasks=[p.tasks(s)[0]], **settle_kw(m))
                steps.append(o["switches"][0] if o["reached"][0] else np.nan)
            if not np.isfinite(steps).all():
                continue
            dist = abs(np.log(np.mean(steps) / TARGET_STEPS))
            if dist < best_d:
                best, best_d = lr, dist
        out[m] = best
    return out


def ceiling(b, lrs, data):
    """Highest task-2 accuracy the slowest rule reaches under THIS cell's cap.

    Measured here, not inherited: script 53 twice took a "ceiling" from a shorter run and set a
    threshold that was either unreachable or so low that no forgetting had happened yet. A
    ceiling measured under a budget is not a ceiling."""
    tops = []
    for m in ["backprop", "replay"]:
        p = replace(b, stop_threshold=[T1_THRESHOLD, None], lr={m: lrs[m]})
        for s in range(CAL_SEEDS):
            o = run(p, m, s, data=data, **settle_kw(m))
            c = o["curves"]["argmax"] * 100
            i = int(np.argmin(np.abs(np.asarray(o["steps"]) - o["switches"][0])))
            tops.append(float(c[i:, 1].max()))
    return min(tops)


# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    kept = {tuple(k): z[f"kept_{k[0]}x{k[1]}_{m}"] for k in z["cells"].tolist() for m in METHODS}
    t2end = {tuple(k): z[f"t2_{k[0]}x{k[1]}_{m}"] for k in z["cells"].tolist() for m in METHODS}
    lrs_all = z["lrs"].item()
    thr = z["thresholds"].item()
    segs = {}
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(replace(PROTOCOL, hidden=32, scenario="domain_il"))
    kept, t2end, segs, lrs_all, thr = {}, {}, {}, {}, {}
    t0 = time.perf_counter()

    for (d, w) in CELLS:
        b = base_for(d, w)
        lrs = calibrate(b, data)
        if any(v is None for v in lrs.values()):
            miss = [m for m, v in lrs.items() if v is None]
            print(f"  {d}x{w}: {', '.join(miss)} reached {T1_THRESHOLD:.0%} at NO rate on the "
                  f"grid -- cell skipped, widen LR_GRID")
            continue
        lrs_all[(d, w)] = lrs
        T2 = float(np.floor((ceiling(b, lrs, data) - 3.0) / 5.0) * 5.0 / 100.0)
        thr[(d, w)] = T2
        print(f"  {d}x{w}  lr " + ", ".join(f"{m} {lrs[m]:g}" for m in METHODS)
              + f"  | task 2 stops at {T2:.0%}   [{time.perf_counter()-t0:5.0f}s]")

        for m in METHODS:
            p = replace(b, stop_threshold=[T1_THRESHOLD, T2], lr={m: lrs[m]})
            ks, t2s, sg = [], [], []
            for s in range(SEEDS):
                o = run(p, m, s, data=data, **settle_kw(m))
                c = o["curves"]["argmax"]
                i = int(np.argmin(np.abs(np.asarray(o["steps"]) - o["switches"][0])))
                ks.append(c[-1, 0] * 100)
                t2s.append(c[-1, 1] * 100)
                sg.append(c[i:, :])
            kept[(d, w, m)], t2end[(d, w, m)], segs[(d, w, m)] = ks, t2s, sg
            print(f"      {m:9s} task 1 kept {np.mean(ks):5.1f}%   task 2 {np.mean(t2s):5.1f}%"
                  f"   [{time.perf_counter()-t0:5.0f}s]")

done = sorted({(d, w) for (d, w, m) in kept})

# ---------------------------------------------------------------- readings
print(f"\n  GATE: is each cell a comparison? task-2 accuracy must match across rules.\n")
valid = {}
for (d, w) in done:
    t2 = {m: np.mean(t2end[(d, w, m)]) for m in METHODS}
    gap = max(t2.values()) - min(t2.values())
    valid[(d, w)] = gap <= 10.0
    print(f"    {d}x{w:<4d} " + ", ".join(f"{m} {v:.0f}%" for m, v in t2.items())
          + f"   spread {gap:4.1f}  " + ("ok" if valid[(d, w)] else "INVALID, not a comparison"))

print(f"\n  PAIRED difference in task-1 retention against backprop, per seed\n")
print(f"  {'cell':>8s} " + " ".join(f"{m:>22s}" for m in METHODS if m != "backprop"))
res = {}
for (d, w) in done:
    row = []
    for m in METHODS:
        if m == "backprop":
            continue
        dd, s, n = paired_diff(kept[(d, w, m)], kept[(d, w, "backprop")])
        res[(d, w, m)] = (dd, s, n)
        flag = "*" if n > 2 else " "
        row.append(f"{dd:+6.1f} +-{s:4.1f} {n:4.1f}sem{flag}")
    print(f"  {d}x{w:<4d}  " + " ".join(f"{r:>22s}" for r in row)
          + ("" if valid[(d, w)] else "   <- cell invalid"))
print("   * = separated at 2 sem")

ok_cells = [c for c in done if valid[c]]
rep_ok = [c for c in ok_cells if res[(c[0], c[1], "replay")][2] > 2]
pc_win = [c for c in ok_cells if res[(c[0], c[1], "pc")][0] > 0
          and res[(c[0], c[1], "pc")][2] > 2]
print(f"\n  VERDICT")
print(f"    positive control separates in {len(rep_ok)}/{len(ok_cells)} valid cells")
if not rep_ok:
    print("    Replay does not separate anywhere. Nothing here can be interpreted.")
elif pc_win:
    print(f"    PC separates FROM backprop, in the right direction, at: "
          + ", ".join(f"{d}x{w}" for d, w in pc_win))
    print("    The earlier nulls were a property of the network size, not of the rule. Move the"
          "\n    comparison to that size and replicate before treating it as a finding.")
else:
    print("    PC does not beat backprop in ANY valid cell, while replay separates. Depth and"
          "\n    width do not rescue it: taken with scripts 52, 53, 56 and 57, PC does not reduce"
          "\n    forgetting in this task family at any size tried.")

# ---------------------------------------------------------------- figures
if segs:
    for (d, w) in done:
        plot_retention_curve(
            {m: segs[(d, w, m)] for m in METHODS}, METHODS,
            figure_path(__file__, f"retention_{d}x{w}"), colors=COLORS,
            chance=1.0 / PROTOCOL.classes_per_task, threshold=thr[(d, w)],
            title=f"{d} hidden layer(s) x {w} units — task 1 retained against task-2 progress",
        )

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(WIDTHS), figsize=(6.2 * len(WIDTHS), 4.6), sharey=True,
                       squeeze=False)
for a, w in zip(ax[0], WIDTHS):
    for m in METHODS:
        if m == "backprop":
            continue
        xs = [d for d in DEPTHS if (d, w) in done]
        ys = [res[(d, w, m)][0] for d in xs]
        es = [res[(d, w, m)][1] for d in xs]
        if xs:
            a.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=2, color=COLORS[m], label=m)
    a.axhline(0, color="k", lw=1)
    a.set_title(f"{w} units per layer")
    a.set_xlabel("hidden layers")
    a.set_xticks(DEPTHS)
    a.grid(alpha=0.25)
ax[0][0].set_ylabel("task 1 kept, minus backprop (points)")
ax[0][0].legend(fontsize=9)
fig.suptitle("Above zero = forgets less than backprop, at matched task-2 competence. "
             "Paired, per seed.", fontsize=10)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

np.savez(array_path(__file__), cells=np.asarray(done), methods=np.asarray(METHODS),
         lrs=lrs_all, thresholds=thr,
         **{f"kept_{d}x{w}_{m}": np.asarray(kept[(d, w, m)]) for (d, w) in done for m in METHODS},
         **{f"t2_{d}x{w}_{m}": np.asarray(t2end[(d, w, m)]) for (d, w) in done for m in METHODS})
print(f"saved {array_path(__file__)}")
