"""Do the four learning rules forget task 1 differently, when each has learned task 1 equally
well and each gets the same training budget?

THE FIRST COMPARISON OF THE NEW SERIES, and the point of the whole project stated as one run.
Everything before it was groundwork: 41 fixed the width, 42/43 established where the forgetting
lives and that the measurement point decides the answer, 50 established that the two
energy-based rules genuinely settle.

WHAT IS HELD FIXED, WHICH IS WHAT MAKES IT A COMPARISON
    Held identical across all four rules, by src.protocol and src.methods rather than by each
    script restating them: the data and its resolution, the 2x5 class split (drawn per seed,
    the same split for every rule at that seed), the data ORDER, the initialisation seed, the
    architecture, the output structure (linear readout, squared error, one-hot targets), the
    optimiser, the batch size, the task budgets, and both evaluation sets.

    Two things differ per rule, deliberately:
      learning rate    from script 51, chosen so all four reach 90% on task 1 in about the
                       same number of updates. Without this the comparison is a learning-speed
                       comparison wearing a forgetting comparison's clothes.
      settling         PC relaxes 50 fixed steps, EqProp to a relative-residual tolerance.
                       Script 50 verified both reach equilibrium, so this is a difference in
                       stopping MECHANISM, not in whether the rule is the rule.

WHY A FIXED BUDGET HERE AND ACCURACY STOPPING IN 53
    A fixed budget puts every run on one x-axis, so the four panels are directly comparable and
    the shaded task blocks mean the same thing everywhere. That is what makes this readable as
    a first result. It is NOT the principled measurement point -- 42 and 43 are the pair that
    established why, and the same trap applies here: run task 2 long enough and every rule
    reaches the same place, so the endpoint stops discriminating. Script 53 repeats this with
    both tasks stopped on accuracy, which is the measurement that can be defended. If the two
    disagree, 53 is the one that counts and the disagreement is itself the finding.

CONTROLS, per .claude/rules/experiments.md
    backprop  negative control -- should forget.
    replay    positive control -- should not. If replay recovers task 1, the problem is
              provably solvable at this width and budget, so a rule that fails is failing at
              something achievable rather than at something impossible.

WHAT THIS CAN AND CANNOT CLAIM
    It CAN say whether the rules separate at all, and by how much, under matched task-1
    competence and a matched budget.
    It CANNOT attribute a difference to prospective configuration, or to any mechanism. That
    is E, and it needs the alignment and weight-path probes.
    It CANNOT be read at the endpoint alone. Retention is reported alongside FINAL TASK-2
    ACCURACY throughout, because "forgot less" is not a result if it is confounded with
    "learned less" -- freezing the hidden layer scored well on retention in 42 by refusing to
    learn task 2, and a learning rule can do the same thing softly.

READINGS COMMITTED BEFORE RUNNING
    Backprop collapses to near zero, as in script 40. Replay holds task 1. If PC and EqProp
    sit on top of backprop, the energy-based rules do not help at this scale and the project's
    question is answered negatively -- a real result, and the honest one to report.
    If they sit between backprop and replay, there is an effect to measure, and B1-B3 exist to
    measure it properly.
    If either matches replay, check for a bug before believing it: replay sees stored task-1
    data and the others do not.

Deviation from the protocol: fixed budgets per task instead of accuracy stopping. Stated so
because script 53 is the same experiment without that deviation.
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
SEEDS = 5                       # the protocol's minimum; 68% intervals reported
EVAL_EVERY = 10
TASK_COLORS = ["tab:orange", "tab:blue"]           # task 1 orange, task 2 blue, as in 40-43
COLORS = {"backprop": "tab:gray", "replay": "tab:brown",
          "pc": "tab:red", "eqprop": "tab:green"}  # per-rule, as in script 51

# ---- everything below is READ, not chosen here ----------------------------
HIDDEN = int(np.load(array_path(str(ROOT / "experiments" /
                                    "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(array_path(str(ROOT / "experiments" /
                             "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
ACHIEVED = {m: float(v) for m, v in zip(z51["methods"], z51["achieved"])}
SETTLE_TOL, EQ_MAX_STEPS = float(z51["settle_tol"]), int(z51["eq_max_steps"])
PC_STEPS = int(z51["pc_steps"])

# Budgets are set from the calibrated time-to-competence, not chosen. Task 1 gets 2x it, so
# every rule is comfortably at its plateau before the switch and no rule is still climbing when
# task 2 starts. Task 2 gets 1.5x, which is several forgetting half-lives at these learning
# rates -- enough to show the whole transition and its tail, without spending EqProp time on a
# saturated flat line. Both are fixed, so all four panels share one x-axis.
T = float(z51["target_steps"])
ITERS = [int(2 * T), int(1.5 * T)]

missing = [m for m in METHODS if not np.isfinite(LR.get(m, np.nan))]
assert not missing, (f"{', '.join(missing)} has no calibrated learning rate. Script 51 could "
                     f"not match it; fix that before running a comparison that assumes it.")

if "--smoke" in sys.argv:
    SEEDS, ITERS = 1, [40, 30]
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} (script 41) | {ITERS[0]}+{ITERS[1]} updates | {SEEDS} seeds")
print("  learning rates from script 51, matched on updates to 90% on task 1:")
for m in METHODS:
    print(f"    {m:10s} lr={LR[m]:<8g} reached 90% in {ACHIEVED[m]:.0f} updates")
print()


def settle_kw(method):
    """Per-rule settling, as verified by script 50. Empty for the rules that do not settle."""
    if method == "pc":
        return dict(steps=PC_STEPS)
    if method == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il", stop_threshold=None,
               max_iters_per_task=ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)

# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__))
    steps, switches = z["steps"], list(z["switches"])
    curves = {m: z[f"argmax_{m}"] for m in METHODS}
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(base)
    curves = {m: [] for m in METHODS}
    t0 = time.perf_counter()
    for m in METHODS:
        proto = replace(base, lr={m: LR[m]})
        for seed in range(SEEDS):
            out = run(proto, m, seed, data=data, **settle_kw(m))
            curves[m].append(out["curves"]["argmax"])
        steps, switches = out["steps"], out["switches"]
        curves[m] = np.stack(curves[m])
        A = curves[m] * 100
        i_sw = int(np.argmin(np.abs(np.asarray(steps) - switches[0])))
        print(f"  {m:10s} task 1 at switch {A[:, i_sw, 0].mean():5.1f}%"
              f"  ->  after task 2 {A[:, -1, 0].mean():5.1f}%"
              f"   (task 2 reached {A[:, -1, 1].mean():5.1f}%)"
              f"   [{time.perf_counter()-t0:5.0f}s]")

blocks = [(0, switches[0], 0), (switches[0], steps[-1], 1)]
i_sw = int(np.argmin(np.abs(np.asarray(steps) - switches[0])))

# ---------------------------------------------------------------- readings
# Retention is NEVER reported on its own. A rule that refuses to learn task 2 keeps task 1 for
# a reason that has nothing to do with credit assignment, and the task-2 column is what makes
# that visible instead of letting it read as a win.
print(f"\n  {'rule':10s} {'t1 at switch':>13s} {'t1 kept':>9s} {'t2 final':>9s} "
      f"{'crossover':>10s} {'half-life':>10s} {'area':>7s}")
summary = {}
for m in METHODS:
    rows = []
    for i in range(SEEDS):
        c = curves[m][i] * 100
        peak = c[:i_sw + 1, 0].max()
        rows.append(dict(
            at_switch=c[i_sw, 0], kept=c[-1, 0], t2=c[-1, 1],
            xh=crossover(steps, c[:, 0], c[:, 1], after=switches[0])[1],
            hl=half_life(steps, c[:, 0], after=switches[0], peak=peak),
            area=area_retained(steps, c[:, 0], after=switches[0], peak=peak) * 100,
        ))

    def agg(k):
        v = [r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
        return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) if len(v) > 1 \
            else (float(v[0]), 0.0) if v else (float("nan"), float("nan"))

    s = {k: agg(k) for k in rows[0]}
    s["n_crossed"] = sum(1 for r in rows if r["xh"] is not None and np.isfinite(r["xh"]))
    summary[m] = s
    # A missing crossover is not a failed calculation: it means task 1 never fell below task 2.
    xh = f"{s['xh'][0]:8.1f}%" if s["n_crossed"] else "   never"
    hl = f"{s['hl'][0]:10.0f}" if np.isfinite(s["hl"][0]) else "     never"
    print(f"  {m:10s} {s['at_switch'][0]:12.1f}% {s['kept'][0]:8.1f}% {s['t2'][0]:8.1f}% "
          f"{xh} {hl} {s['area'][0]:6.1f}%")

bp, rp = summary["backprop"]["kept"][0], summary["replay"]["kept"][0]
print(f"\n  controls: backprop keeps {bp:.1f}%, replay keeps {rp:.1f}%.")
print("  " + ("Replay recovers task 1, so retention IS achievable here and a rule that fails "
              "is failing at something possible."
              if rp > bp + 10 else
              "REPLAY DID NOT RECOVER TASK 1. The positive control failed, so nothing else on "
              "this figure can be interpreted. Check the buffer before reading further."))
for m in ["pc", "eqprop"]:
    k = summary[m]["kept"][0]
    band = ("matches replay -- check for a bug, replay sees stored task-1 data and this does not"
            if k >= rp - 5 else "sits with backprop -- no effect at this scale"
            if k <= bp + 5 else "sits between the controls -- there is an effect to measure")
    print(f"  {m:10s} keeps {k:5.1f}%  {band}")

# ---------------------------------------------------------------- figures
plot_learning_curves(
    steps, curves, METHODS, figure_path(__file__),
    blocks=blocks, ncols=2, task_colors=TASK_COLORS, task_labels=["task 1", "task 2"],
    legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False),
)

# Companion: the trade-off with time removed. Every run has the same budget here, so unlike
# script 43 a mean trajectory is well defined -- eval index i is the same update for every run.
#
# The WHOLE path is drawn, from initialisation, not from the task switch. Starting at the
# switch hides the task-1 learning phase and opens the plot at an arbitrary-looking point.
# Read it as: right along the bottom while task 1 is learned, then up and to the left as task 2
# displaces it. The corner it turns is the trade.
figt, axt = plt.subplots(figsize=(5.8, 5.6))
for m in METHODS:
    A = curves[m].mean(0) * 100
    axt.plot(A[:i_sw + 1, 0], A[:i_sw + 1, 1], lw=1.2, alpha=0.45,
             color=COLORS[m])                                     # task 1 block
    axt.plot(A[i_sw:, 0], A[i_sw:, 1], lw=2.2, color=COLORS[m], label=m)
    axt.plot(A[i_sw, 0], A[i_sw, 1], "s", ms=6, color=COLORS[m])  # the switch
    axt.plot(A[-1, 0], A[-1, 1], "o", ms=7, color=COLORS[m])      # the end
CH = 100.0 / base.classes_per_task
axt.plot([CH], [CH], "+", color="k", ms=12, mew=1.5)
axt.annotate("chance, at initialisation", xy=(CH, CH), xytext=(6, 6),
             textcoords="offset points", fontsize=8)
axt.plot([100, 0], [0, 100], color="gray", ls=":", lw=1)
axt.set_xlabel("task 1 accuracy (%)")
axt.set_ylabel("task 2 accuracy (%)")
axt.set_xlim(-2, 102); axt.set_ylim(-2, 102)
axt.grid(alpha=0.2)
axt.legend(fontsize=9, title="faint = task 1 block, square = switch", title_fontsize=7)
figt.tight_layout()
figt.savefig(figure_path(__file__, "trajectory"), dpi=120, bbox_inches="tight")

np.savez(array_path(__file__), steps=np.asarray(steps), switches=np.asarray(switches),
         hidden=HIDDEN, iters=np.asarray(ITERS), methods=np.asarray(METHODS),
         lr=np.asarray([LR[m] for m in METHODS]),
         **{f"argmax_{m}": curves[m] for m in METHODS},
         **{f"{k}_{m}": summary[m][k] for m in METHODS for k in summary[m]})
print(f"\nsaved {figure_path(__file__)}\nsaved {figure_path(__file__, 'trajectory')}"
      f"\nsaved {array_path(__file__)}")
