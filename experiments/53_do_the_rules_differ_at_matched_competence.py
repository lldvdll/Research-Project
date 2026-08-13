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
from src.metrics import metric_grid, report_grid, crossover, half_life, value_when, area_retained, paired_diff
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

# THE TASK-2 THRESHOLD IS MEASURED, UNDER THE CAP THIS SCRIPT ACTUALLY USES. Two attempts got
# it wrong in different ways, and the second is a repeat of a mistake script 41 already made.
#
#   1. Set to 90% by assumption, the same as task 1. Replay missed it on 4 of 5 seeds, so most
#      runs were never read at matched competence and the comparison was void.
#   2. Derived as 70% from script 52's task-2 "ceilings". But 52 ran a FIXED 630-update budget,
#      so those numbers say where each rule had GOT TO, not where it could get -- replay's 78%
#      was its progress, not its limit. A CEILING MEASURED UNDER A BUDGET IS NOT A CEILING.
#      Script 41 hit the same thing: a short-budget capacity sweep produced a flat region that
#      looked like a capacity limit and was really the budget binding. Measured under this
#      script's own 2000-update cap, replay reaches 88.7% and backprop 90.8%.
#      At 70% the reading came far too early to show any forgetting -- backprop's task-2 block
#      was 186 updates of 2000 and it had fallen only from 90% to 72%, leaving replay with
#      almost nothing to prevent and the four rules within 7 points of each other.
#
# So a pilot measures it here, under MAX_ITERS. Only the two cheap rules are piloted: replay is
# the slowest to learn task 2 -- it spends half of every batch on task 1 -- so it is the binding
# constraint, and piloting the energy-based rules would cost more than the rest of the script.
def _measure_ceiling(data_):
    """Highest task-2 accuracy the slowest rule actually reaches, under this script's cap."""
    tops = []
    for m in ["backprop", "replay"]:
        p = replace(base, stop_threshold=[T1_THRESHOLD, None], lr={m: LR[m]})
        for seed in range(SEEDS):
            o = run(p, m, seed, data=data_)
            c = o["curves"]["argmax"] * 100
            i = int(np.argmin(np.abs(np.asarray(o["steps"]) - o["switches"][0])))
            tops.append(float(c[i:, 1].max()))
    return min(tops)


if "--smoke" in sys.argv:
    SEEDS, MAX_ITERS, T1_THRESHOLD = 1, 60, 0.5
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} (41) | task 1 stops at {T1_THRESHOLD:.0%} | cap {MAX_ITERS}"
      f" | {SEEDS} seeds")
print("  learning rates from 51: " + ", ".join(f"{m} {LR[m]:g}" for m in METHODS) + "\n")


def settle_kw(method):
    """Per-rule settling, as verified by script 50. Empty for the rules that do not settle."""
    if method == "pc":
        return dict(steps=PC_STEPS)
    if method == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il",
               stop_patience=STOP_PATIENCE,
               max_iters_per_task=MAX_ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)

# ---------------------------------------------------------------- run
# Runs have different lengths, so they are kept as per-seed dicts rather than stacked.
data = load(base)
_ceiling = 60.0 if SMOKE else _measure_ceiling(data)
T2_THRESHOLD = float(np.floor((_ceiling - 3.0) / 5.0) * 5.0 / 100.0)
print(f"  pilot: the slowest rule reaches {_ceiling:.1f}% on task 2 under a {MAX_ITERS} cap"
      f"  ->  task 2 stops at {T2_THRESHOLD:.0%}\n")
base = replace(base, stop_threshold=[T1_THRESHOLD, T2_THRESHOLD])
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
summary, per_seed_kept = {}, {}
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
    per_seed_kept[m] = [r["kept"] if r["kept"] is not None else np.nan
                        for r in rows]
    summary[m] = s_
    xh = f"{s_['xh'][0]:8.1f}%" if s_["n_crossed"] else "   never"
    hl = f"{s_['hl'][0]:10.0f}" if np.isfinite(s_["hl"][0]) else "     never"
    print(f"  {m:10s} {s_['at_switch'][0]:12.1f}% {s_['kept'][0]:8.1f}% {s_['t2'][0]:8.1f}% "
          f"{xh} {hl} {s_['block'][0]:9.0f}")

# ---- THE COMPARISON: paired against backprop, seed by seed ------------------
# Group means are the wrong statistic here and script 53 proved it the hard way. Every rule sees
# the same class split and the same initialisation at a given seed, so most of the between-seed
# variance is shared; comparing group means discards that and the noise swamps everything. On
# 53's own runs, backprop retained 38.2% on one seed and 78.0% on another -- a 40-point range set
# by which digits the split happened to pair -- and REPLAY, the positive control, came out at
# 0.7 sem and read as a failure. Paired, the same runs give +11.6 +- 2.3, i.e. 5.1 sem.
print("\n  PAIRED against backprop, per seed. This is the comparison; the table above is "
      "descriptive.")
print(f"  {'rule':10s} {'diff in task 1 kept':>21s} {'sem':>7s} {'':>10s}")
paired = {}
for m in METHODS:
    if m == "backprop":
        continue
    d, s, n = paired_diff(per_seed_kept[m], per_seed_kept["backprop"])
    paired[m] = (d, s, n)
    print(f"  {m:10s} {d:+18.1f} pts {s:7.1f} {n:6.1f} sem  "
          + ("SEPARATED" if n > 2 else "not separated"))

rp_n = paired.get("replay", (0, 0, 0))[2]
print("  " + ("positive control holds: replay separates from backprop, so retention IS "
              "achievable here"
              if rp_n > 2 else
              "POSITIVE CONTROL FAILED even paired -- replay does not separate. Nothing else on "
              "this figure can be interpreted."))

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

# ---- the FULL metric grid ------------------------------------------------
# Reported in full because the choice of metric changed this project's conclusion once already:
# on endpoint retention PC was indistinguishable from backprop, on CROSSOVER HEIGHT -- the metric
# experiment 12 was read on -- it separates in Class-IL. Quoting one number invites quoting the
# flattering one, so every metric is printed, paired against backprop, every time.
report_grid({m: metric_grid(grid, padded[m], 0) for m in METHODS},
            METHODS, control="backprop", primary="crossover")

plot_learning_curves(
    grid, padded, METHODS, figure_path(__file__),
    blocks=[(LO, 0, 0), (0, HI, 1)], ncols=2,
    task_colors=TASK_COLORS, task_labels=["task 1", "task 2"],
    crossover_after=0,
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

np.savez(array_path(__file__), steps=np.asarray(grid),
         **{f"perseed_{m}": np.asarray(per_seed_kept[m], dtype=float) for m in METHODS}, switch=0, hidden=HIDDEN,
         t1_threshold=T1_THRESHOLD, t2_threshold=T2_THRESHOLD, methods=np.asarray(METHODS),
         lr=np.asarray([LR[m] for m in METHODS]),
         **{f"argmax_{m}": padded[m] for m in METHODS},
         **{f"{k}_{m}": summary[m][k] for m in METHODS for k in summary[m]})
print(f"\nsaved {figure_path(__file__)}\nsaved {figure_path(__file__, 'trajectory')}"
      f"\nsaved {array_path(__file__)}")
