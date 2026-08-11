"""Do the four learning rules forget task 1 differently, read at the moment each has learned
task 2 to the same standard?

Script 52 with ONE thing changed: both tasks stop on accuracy instead of running a fixed
budget. The pair is the argument, exactly as 42 and 43 were.

WHY THE PAIR IS THE ARGUMENT, AGAIN
    42 ran task 2 for a fixed budget until it saturated, every condition read zero, and it
    reported that freezing the hidden layer recovers nothing. That was false -- read at matched
    competence the same runs showed freezing holding 78.9% against a control's 47.6%. The
    measurement destroyed the effect, not the intervention.

    The same trap is available here and is worse, because a learning rule can walk into it
    quietly. Under a fixed budget a rule that happens to learn task 2 more slowly gets fewer
    effective updates of interference and keeps more of task 1 -- and that reads on the figure
    as "forgets less" when it is really "learned less". Stopping BOTH tasks on accuracy removes
    it by construction: every rule is read at the point where it has learned task 1 to the same
    standard AND task 2 to the same standard, so retention is the only thing left varying.

    That is why the thresholds are a single number for both tasks and not a mix of a threshold
    and a budget. A mix reintroduces exactly what this is removing.

WHAT CHANGES IN THE READING, AND WHY THE X AXIS IS DIFFERENT
    Runs now stop at different times, so they cannot share an absolute x axis -- the fast ones
    would be compressed into the left edge. Every run is plotted against UPDATES RELATIVE TO
    ITS OWN TASK SWITCH, so the switch is at x = 0 everywhere and negative x is task-1 training.

    The curves are NOT stretched to fill the panel. Rescaling each run to a common length would
    destroy the rate, and the rate is half of what is being measured -- a fast-forgetting run
    and a slow one would draw identically. The differing lengths are a result in themselves: a
    rule needing 700 updates to reach competence on task 2 while another needs 200 has said
    something. The companion trajectory plot is what handles the ragged lengths, by removing
    time entirely rather than by distorting it.

WHAT IS REPORTED, AND WHAT IS NOT ENOUGH ON ITS OWN
      task 1 retained at the matched moment   the headline
      task 2 reached                          the check that the headline is not "learned less"
      crossover HEIGHT                        the accuracy at which the two curves intersect.
                                              Not WHEN they intersect -- that is a different
                                              quantity and it is confounded by learning speed.
                                              Undefined when they never cross, which is itself
                                              the strongest statement it can make.
      half-life                               updates to lose half the pre-switch peak. Covers
                                              the case crossover cannot.
      reached / capped                        a rule that never met the threshold is reported,
                                              never silently folded in at whatever it managed.

IF 52 AND 53 DISAGREE, 53 IS THE ONE THAT COUNTS
    and the disagreement is itself the finding -- it is the same lesson 42 and 43 taught,
    arriving where it now costs something.

READINGS COMMITTED BEFORE RUNNING
    Backprop loses most of task 1 by the time task 2 is competent. Replay does not.
    If PC and EqProp sit on backprop, the energy-based rules do not help at this scale.
    If they sit between the controls, there is an effect, and it should be LARGER here than in
    52 if part of what 52 showed was a speed artefact -- or smaller, if 52's apparent effect
    WAS the speed artefact. Either way this is the number that gets reported.

Deviation from the protocol: none. This is the protocol. Script 52 is the deviation.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, run, replace, figure_path, array_path
from src.metrics import crossover, half_life, value_when, area_retained
from src.plotting import plot_learning_curves

# ---------------------------------------------------------------- settings
METHODS = ["backprop", "replay", "pc", "eqprop"]
SEEDS = 5
EVAL_EVERY = 10
THRESHOLD = 0.90        # the SAME standard for both tasks. See the docstring.
STOP_PATIENCE = 3
MAX_ITERS = 1500        # cap only, per task; how often it binds is reported
TASK_COLORS = ["tab:orange", "tab:blue"]
COLORS = {"backprop": "tab:gray", "replay": "tab:brown",
          "pc": "tab:red", "eqprop": "tab:green"}

# ---- everything below is READ, not chosen here ----------------------------
HIDDEN = int(np.load(array_path(str(ROOT / "experiments" /
                                    "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(array_path(str(ROOT / "experiments" /
                             "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
SETTLE_TOL, EQ_MAX_STEPS = float(z51["settle_tol"]), int(z51["eq_max_steps"])
PC_STEPS = int(z51["pc_steps"])
assert abs(float(z51["threshold"]) - THRESHOLD) < 1e-9, (
    "task 1's threshold here must match the one the learning rates were calibrated at, "
    f"which was {float(z51['threshold']):.0%}. Otherwise the rules are matched on one "
    "standard and read at another.")

if "--smoke" in sys.argv:
    SEEDS, MAX_ITERS, THRESHOLD = 1, 60, 0.5
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} (41) | both tasks stop at {THRESHOLD:.0%} | cap {MAX_ITERS} | {SEEDS} seeds")
print("  learning rates from 51: " + ", ".join(f"{m} {LR[m]:g}" for m in METHODS) + "\n")


def settle_kw(method):
    """Per-rule settling, as verified by script 50. Empty for the rules that do not settle."""
    if method == "pc":
        return dict(steps=PC_STEPS)
    if method == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il",
               stop_threshold=THRESHOLD, stop_patience=STOP_PATIENCE,
               max_iters_per_task=MAX_ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)

# ---------------------------------------------------------------- run
# Runs have different lengths, so they are kept as per-seed dicts rather than stacked.
data = load(base)
runs = {m: [] for m in METHODS}
t0 = time.perf_counter()

for m in METHODS:
    proto = replace(base, lr={m: LR[m]})
    for seed in range(SEEDS):
        runs[m].append(run(proto, m, seed, data=data, **settle_kw(m)))
    r = runs[m]
    print(f"  {m:10s} reached threshold: task 1 {sum(o['reached'][0] for o in r)}/{SEEDS}"
          f"  task 2 {sum(o['reached'][1] for o in r)}/{SEEDS}"
          f"   blocks {np.mean([o['switches'][0] for o in r]):5.0f}"
          f" + {np.mean([o['switches'][1]-o['switches'][0] for o in r]):5.0f} updates"
          f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- readings
print(f"\n  read at the moment task 2 first holds >= {THRESHOLD:.0%} for {STOP_PATIENCE} evals\n")
print(f"  {'rule':10s} {'t1 at switch':>13s} {'t1 kept':>9s} {'t2 final':>9s} "
      f"{'crossover':>10s} {'half-life':>10s} {'t2 block':>9s}")
summary = {}
for m in METHODS:
    rows = []
    for o in runs[m]:
        s, c = o["steps"], o["curves"]["argmax"] * 100
        sw = o["switches"][0]
        i_sw = int(np.argmin(np.abs(np.asarray(s) - sw)))
        peak = c[:i_sw + 1, 0].max()
        rows.append(dict(
            at_switch=c[i_sw, 0],
            kept=value_when(s, c[:, 1], THRESHOLD * 100, c[:, 0], after=sw,
                            patience=STOP_PATIENCE),
            t2=c[-1, 1],
            xh=crossover(s, c[:, 0], c[:, 1], after=sw)[1],
            hl=half_life(s, c[:, 0], after=sw, peak=peak),
            block=float(o["switches"][1] - o["switches"][0]),
        ))

    def agg(k):
        v = [r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
        return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) if len(v) > 1 \
            else (float(v[0]), 0.0) if v else (float("nan"), float("nan"))

    s_ = {k: agg(k) for k in rows[0]}
    s_["n_crossed"] = sum(1 for r in rows if r["xh"] is not None and np.isfinite(r["xh"]))
    s_["reached2"] = sum(o["reached"][1] for o in runs[m])
    summary[m] = s_
    xh = f"{s_['xh'][0]:8.1f}%" if s_["n_crossed"] else "   never"
    hl = f"{s_['hl'][0]:10.0f}" if np.isfinite(s_["hl"][0]) else "     never"
    print(f"  {m:10s} {s_['at_switch'][0]:12.1f}% {s_['kept'][0]:8.1f}% {s_['t2'][0]:8.1f}% "
          f"{xh} {hl} {s_['block'][0]:9.0f}")

bp, rp = summary["backprop"]["kept"][0], summary["replay"]["kept"][0]
print(f"\n  controls: backprop keeps {bp:.1f}%, replay keeps {rp:.1f}%.")
print("  " + ("Replay recovers task 1, so retention IS achievable here."
              if rp > bp + 10 else
              "REPLAY DID NOT RECOVER TASK 1. The positive control failed; nothing else on this "
              "figure can be interpreted."))
for m in METHODS:
    if summary[m]["reached2"] < SEEDS:
        print(f"  {m}: task 2 hit the {MAX_ITERS} cap on "
              f"{SEEDS - summary[m]['reached2']}/{SEEDS} seeds -- those runs are NOT at matched "
              f"competence and the comparison does not hold for them.")

# ---------------------------------------------------------------- figures
# Switch-relative axis. Every run's switch is at x = 0; negative x is task-1 training. This is
# the only axis on which runs of different lengths are comparable, and the curves are not
# rescaled to a common length -- see the docstring.
LO = -int(1.2 * max(np.mean([o["switches"][0] for o in runs[m]]) for m in METHODS))
HI = int(1.2 * max(np.mean([o["switches"][1] - o["switches"][0] for o in runs[m]])
                   for m in METHODS))
grid = np.arange(LO, HI + EVAL_EVERY, EVAL_EVERY)
padded = {}
for m in METHODS:
    A = np.full((SEEDS, len(grid), 2), np.nan)
    for i, o in enumerate(runs[m]):
        rel = np.asarray(o["steps"]) - o["switches"][0]
        for j, r in enumerate(rel):
            k = int(round((r - LO) / EVAL_EVERY))
            if 0 <= k < len(grid):
                A[i, k] = o["curves"]["argmax"][j]
    padded[m] = A

plot_learning_curves(
    grid, padded, METHODS, figure_path(__file__),
    blocks=[(LO, 0, 0), (0, HI, 1)], ncols=2,
    task_colors=TASK_COLORS, task_labels=["task 1", "task 2"],
    xlabel="training step, relative to the task switch",
    legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False),
)

# Each seed drawn on its own. A MEAN trajectory is meaningless here: runs stop at different
# times, so averaging position at eval index i averages runs at different points of their own
# path. The whole path is drawn, from initialisation -- not from the switch.
figt, axt = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
for ax, m in zip(axt, METHODS):
    for o in runs[m]:
        c = o["curves"]["argmax"] * 100
        i_sw = int(np.argmin(np.abs(np.asarray(o["steps"]) - o["switches"][0])))
        ax.plot(c[:i_sw + 1, 0], c[:i_sw + 1, 1], lw=1.0, alpha=0.35, color=COLORS[m])
        ax.plot(c[i_sw:, 0], c[i_sw:, 1], lw=1.6, alpha=0.85, color=COLORS[m])
        ax.plot(c[i_sw, 0], c[i_sw, 1], "s", ms=4, color=COLORS[m])
        ax.plot(c[-1, 0], c[-1, 1], "o", ms=5, color="k")
    T, CH = THRESHOLD * 100, 100.0 / base.classes_per_task
    ax.plot([100, 0], [0, 100], color="gray", ls=":", lw=1)
    ax.axhline(T, color="tab:blue", ls="--", lw=1)
    ax.plot([CH], [CH], "+", color="k", ms=10, mew=1.5)
    ax.plot([100], [T], marker="*", color="tab:green", ms=14, ls="none")
    ax.set_title(m)
    ax.set_xlabel("task 1 accuracy (%)")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.grid(alpha=0.2)
axt[0].set_ylabel("task 2 accuracy (%)")
figt.suptitle(f"One line per seed, whole path from initialisation (+). Square = task switch, "
              f"black dot = where the run stopped.\nRightward along the {THRESHOLD:.0%} line "
              f"= task 2 learned while task 1 kept; the green star is the ideal.")
figt.tight_layout()
figt.savefig(figure_path(__file__, "trajectory"), dpi=120, bbox_inches="tight")

np.savez(array_path(__file__), steps=np.asarray(grid), switch=0, hidden=HIDDEN,
         threshold=THRESHOLD, methods=np.asarray(METHODS),
         lr=np.asarray([LR[m] for m in METHODS]),
         **{f"argmax_{m}": padded[m] for m in METHODS},
         **{f"{k}_{m}": summary[m][k] for m in METHODS for k in summary[m]})
print(f"\nsaved {figure_path(__file__)}\nsaved {figure_path(__file__, 'trajectory')}"
      f"\nsaved {array_path(__file__)}")
