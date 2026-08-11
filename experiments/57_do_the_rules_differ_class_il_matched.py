"""Script 56 in CLASS-IL, read at matched competence instead of at a fixed budget.

This is to 56 what 53 is to 52, and it exists for the same reason: under a fixed budget a rule
that learns task 2 more slowly receives fewer effective updates of interference and keeps more
of task 1, which reads as "forgets less" when it is "learned less".

IT MATTERS MORE HERE THAN IN DOMAIN-IL. In 2x5 Class-IL every condition eventually reaches zero
on task 1 -- scripts 42 and 43 measured exactly that -- so an endpoint comparison has no dynamic
range left and cannot rank anything. Reading at the moment each rule has learned task 2 to one
standard is the only measurement with anything left to distinguish.

Its task-2 threshold is derived from script 56, so 56 must run first.

ORIGINAL QUESTION, unchanged: do the rules forget task 1 differently, read at the moment each
has learned task 2 to the same standard?

Script 56 with ONE thing changed: both tasks stop on accuracy instead of running a fixed
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

    That is why BOTH tasks stop on accuracy and neither stops on a budget. A mix reintroduces
    exactly what this removes.

    The two thresholds are not the same NUMBER, and they should not be. Task 1 stops at the
    standard the learning rates were calibrated to, so the rules enter task 2 equally competent.
    Task 2 stops at a standard read off script 52's measured ceilings, because a threshold has
    to be reachable BY EVERY RULE or the comparison quietly becomes "the rules that finished,
    against the rules that were capped". The first version of this script set both to 90% by
    assumption; replay missed it on 4 of 5 seeds and the whole run was void.

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

from src.protocol import PROTOCOL, load, run, replace, figure_path as _figure_path, array_path as _array_path
from src.metrics import crossover, half_life, value_when, area_retained
from src.plotting import plot_learning_curves, plot_retention_curve

# --smoke writes NOTHING. Its outputs would otherwise overwrite the real .npz that later
# scripts read their settings from -- script 53 takes its task-2 threshold from 52's measured
# ceilings, and a smoke-sized 52 silently poisons it.
SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    """Suffix only THIS script's own outputs, never the .npz files it reads from others."""
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):      # noqa: F811  - shadows the import, on purpose
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):       # noqa: F811
    return _array_path(f, _tag(f, suffix))

# ---------------------------------------------------------------- settings
METHODS = ["backprop", "replay", "pc", "eqprop"]
SEEDS = 5
EVAL_EVERY = 10
STOP_PATIENCE = 3
MAX_ITERS = 2000        # cap only, per task; how often it binds is reported
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
T1_THRESHOLD = float(z51["threshold"])          # the standard the learning rates were matched at

# THE TASK-2 T2_THRESHOLD IS DERIVED, NOT ASSUMED, AND THE FIRST ATTEMPT AT THIS SCRIPT FAILED
# BECAUSE IT WAS ASSUMED. It was set to 90% -- the same as task 1 -- without checking that task
# 2 can reach 90% from a network already committed to task 1. Replay missed it on 4 of 5 seeds
# and EqProp on 3, so most runs were never read at matched competence and the whole comparison
# was void. Task 2 is a harder problem than task 1 was: it starts from a committed network, and
# replay spends half its gradient preserving task 1 rather than learning the new task.
#
# So it is read off script 52, which ran the same rules to a long fixed budget and measured what
# each actually reaches. The threshold is the WORST rule's ceiling with a margin, because a
# threshold only one rule can reach silently turns the comparison into "the rules that finished,
# against the rules that were capped".
z52 = np.load(array_path(str(ROOT / "experiments" /
                             "56_do_the_rules_differ_class_il.py")))
reach = {m: float(z52[f"t2_{m}"][0]) for m in [str(x) for x in z52["methods"]]}
T2_THRESHOLD = np.floor(min(reach.values()) * 0.95 / 5) * 5 / 100    # round down to a 5% step

if "--smoke" in sys.argv:
    SEEDS, MAX_ITERS, T1_THRESHOLD, T2_THRESHOLD = 1, 60, 0.5, 0.4
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} (41) | task 1 stops at {T1_THRESHOLD:.0%}, task 2 at {T2_THRESHOLD:.0%}"
      f" | cap {MAX_ITERS} | {SEEDS} seeds")
print("  task-2 threshold derived from script 52's measured ceilings: "
      + ", ".join(f"{m} {v:.0f}%" for m, v in reach.items()))
print("  learning rates from 51: " + ", ".join(f"{m} {LR[m]:g}" for m in METHODS) + "\n")


def settle_kw(method):
    """Per-rule settling, as verified by script 50. Empty for the rules that do not settle."""
    if method == "pc":
        return dict(steps=PC_STEPS)
    if method == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


base = replace(PROTOCOL, hidden=HIDDEN, scenario="class_il",
               stop_threshold=[T1_THRESHOLD, T2_THRESHOLD], stop_patience=STOP_PATIENCE,
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
print(f"\n  read at the moment task 2 first holds >= {T2_THRESHOLD:.0%} for {STOP_PATIENCE} evals\n")
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
            kept=value_when(s, c[:, 1], T2_THRESHOLD * 100, c[:, 0], after=sw,
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

# All four rules on ONE axes, so they can be read against each other. Splitting them into four
# panels -- the first version of this script -- makes the comparison impossible to do by eye,
# which is the whole point of the figure.
#
# The runs have different lengths, so they are averaged against TASK-2 ACCURACY rather than
# against step count: "by the time task 2 had reached y%, what was task 1?" is a question each
# run answers on its own whatever its length. That is the matched-competence reading drawn as a
# whole curve rather than sampled at the single point the table reports.
plot_retention_curve(
    {m: [o["curves"]["argmax"][
            int(np.argmin(np.abs(np.asarray(o["steps"]) - o["switches"][0]))):, :]
         for o in runs[m]] for m in METHODS},
    METHODS, figure_path(__file__, "trajectory"), colors=COLORS,
    chance=1.0 / base.classes_per_task, threshold=T2_THRESHOLD,
    title="Task 1 retained, against how much of task 2 has been learned.\n"
          "Dashed line is where the runs stop. Further right at that line = better retention.",
)

np.savez(array_path(__file__), steps=np.asarray(grid), switch=0, hidden=HIDDEN,
         t1_threshold=T1_THRESHOLD, t2_threshold=T2_THRESHOLD, methods=np.asarray(METHODS),
         lr=np.asarray([LR[m] for m in METHODS]),
         **{f"argmax_{m}": padded[m] for m in METHODS},
         **{f"{k}_{m}": summary[m][k] for m in METHODS for k in summary[m]})
print(f"\nsaved {figure_path(__file__)}\nsaved {figure_path(__file__, 'trajectory')}"
      f"\nsaved {array_path(__file__)}")
