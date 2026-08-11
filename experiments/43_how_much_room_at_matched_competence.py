"""How much of task 1 survives at the moment task 2 has been learned to a fixed standard?

Slide 7. This is script 42 with ONE thing changed: task 2 stops when it reaches an accuracy
threshold, instead of running a fixed budget until it saturates. Both scripts are kept, because
the pair is the argument.

WHY THE PAIR IS THE ARGUMENT
    42 trained task 2 for a fixed 2000 updates -- long enough for it to saturate. By then every
    condition without masking has fallen to zero, so the comparison had no dynamic range left
    and reported that freezing the hidden layer recovers NOTHING. That is false. Read the same
    runs 50 updates after the switch and freezing holds 78.9% against the control's 47.6%, and
    it doubles the half-life. The effect was always there; the measurement destroyed it.

    This is not a small methodological point. In 2x5 Class-IL every condition eventually reaches
    zero, so FINAL RETENTION CANNOT RANK ANYTHING. Whatever is compared has to be measured at a
    matched, defined moment, and "when task 2 reaches accuracy T" is that moment: it is the point
    at which each condition has learned the new task equally well, so retention is not being
    confounded with simply having learned less.

WHY RETENTION ALONE IS STILL NOT ENOUGH, AND WHAT ELSE IS REPORTED
    Freezing the hidden layer keeps task 1 partly by refusing to learn task 2 properly. That is
    a TRADE, not a win, and a retention number cannot tell the two apart. So this also reports:

      trajectory (task-1 accuracy against task-2 accuracy)   the shape of the trade, time removed
      crossover height                                       were both held at once
      half-life                                              how fast task 1 falls
      final retention                                        reported, and expected to be ~0 for
                                                             every condition -- shown so it is
                                                             visibly useless rather than quietly
                                                             load-bearing

    These are the base metrics the rest of the project uses. They are introduced here, on
    backprop with simple interventions, so that when four learning rules arrive the instruments
    are already familiar and the argument is about the rules rather than about measurement.

WHAT THIS CAN AND CANNOT SETTLE
    It CAN show that stabilising the hidden representation changes retention at matched
    competence, i.e. that there is something for a learning rule to act on.
    It CANNOT show that any actual rule achieves that -- masking and freezing are hard
    interventions, not learning rules. That is script 44.

Deviation from the protocol (§2): backprop only, plus the two interventions. H from script 41.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, build, replace, figure_path, array_path
from src.runner import run_classil
from src.probes import live_ncm_fn, prototype_images
from src.metrics import crossover, half_life, value_when, area_retained
from src.plotting import plot_learning_curves, plot_trajectory

# ---------------------------------------------------------------- settings
T1_THRESHOLD = 0.90     # task 1 is trained to competence before task 2 starts
T2_THRESHOLD = 0.50     # the matched standard task 2 must reach. See note below.
STOP_PATIENCE = 3
CAP1, CAP2 = 3000, 3000
EVAL_EVERY = 10
SEEDS = 5
TASK_COLORS = ["tab:orange", "tab:blue"]

# T2_THRESHOLD is 0.50 and not higher for a reason worth stating. Accuracy is measured by argmax
# over all ten units, and under a MASKED loss the task-1 units are never pushed down, so they go
# on competing for the argmax and task 2 tops out near 55%. A threshold above that would never be
# reached by the masked conditions and the comparison would silently become "the two conditions
# that finished, measured against two that were capped". 0.50 is reachable by all four.

cap = np.load(array_path(str(ROOT / "experiments" / "41_capacity_vs_hidden_width.py")))
HIDDEN, CEILING = int(cap["chosen"]), float(cap["acc_class_il"].mean(1).max())
print(f"H = {HIDDEN} from script 41 (joint ceiling {CEILING:.1f}%)\n")

CONDITIONS = {"control": (False, False), "masked": (True, False),
              "frozen hidden": (False, True), "masked + frozen": (True, True)}

base = replace(PROTOCOL, hidden=HIDDEN, eval_every=EVAL_EVERY, seeds=SEEDS,
               stop_threshold=[T1_THRESHOLD, T2_THRESHOLD], stop_patience=STOP_PATIENCE,
               max_iters_per_task=[CAP1, CAP2])

# ---------------------------------------------------------------- run
data = load(base)
runs = {c: [] for c in CONDITIONS}          # per-seed dicts, since lengths differ per run

for name, (mask, freeze) in CONDITIONS.items():
    for seed in range(SEEDS):
        proto = replace(base, mask=mask)
        tasks = proto.tasks(seed)
        handle = {}
        train_step, predict = build(proto, "backprop", seed, handle=handle)
        px, py = prototype_images(data.train, data.class_idx, sorted(tasks[0] + tasks[1]),
                                  per_class=50, device=proto.device, seed=seed)

        def on_task_end(ti, step, _f=freeze, _h=handle):
            if ti == 0 and _f:
                _h["freeze"].update({"W1", "b1"})   # freeze at the SWITCH, after task 1 is learned

        out = run_classil(
            train_step, predict, tasks, data.train, data.class_idx,
            report_eval=data.report_eval, stop_eval=data.stop_eval,
            readouts={"argmax": predict, "ncm": live_ncm_fn(handle["features"], px, py)},
            max_iters_per_task=[CAP1, CAP2], batch=proto.batch, eval_every=EVAL_EVERY,
            device=proto.device, data_seed=seed,
            stop_threshold=[T1_THRESHOLD, T2_THRESHOLD], stop_patience=STOP_PATIENCE,
            on_task_end=on_task_end,
        )
        runs[name].append(out)
    r = runs[name]
    print(f"  {name:16s} reached task2 threshold: {sum(o['reached'][1] for o in r)}/{SEEDS}"
          f"   task-2 block length {np.mean([o['switches'][1]-o['switches'][0] for o in r]):6.0f}"
          f" updates")

# ---------------------------------------------------------------- metrics at the matched moment
print(f"\n  measured when task 2 first holds >= {T2_THRESHOLD:.0%} for {STOP_PATIENCE} evals\n")
print(f"  {'condition':16s} {'task1 kept':>11s} {'task2':>7s} {'crossover':>10s} "
      f"{'half-life':>10s} {'area':>7s} {'final t1':>9s} {'NCM t1':>7s}")

summary = {}
for name in CONDITIONS:
    rows = []
    for o in runs[name]:
        s, c = o["steps"], o["curves"]["argmax"] * 100
        sw = o["switches"][0]
        peak = c[:int(np.argmin(np.abs(s - sw))) + 1, 0].max()
        rows.append(dict(
            kept=value_when(s, c[:, 1], T2_THRESHOLD * 100, c[:, 0], after=sw,
                            patience=STOP_PATIENCE),
            t2=c[-1, 1],
            xh=crossover(s, c[:, 0], c[:, 1], after=sw)[1],
            hl=half_life(s, c[:, 0], after=sw, peak=peak),
            area=area_retained(s, c[:, 0], after=sw, peak=peak) * 100,
            final=c[-1, 0],
            ncm=o["curves"]["ncm"][-1, 0] * 100,
        ))
    def col(k):
        v = [r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")

    agg = {k: col(k) for k in rows[0]}
    agg["n_crossed"] = sum(1 for r in rows if r["xh"] is not None and np.isfinite(r["xh"]))
    summary[name] = agg
    # A NaN crossover is not a failed calculation: it means the two curves NEVER CROSSED, i.e.
    # task 1 stayed above task 2 throughout. Under masking that is exactly what happens, and it
    # is the strongest single statement the metric can make.
    xh = f"{agg['xh']:8.1f}%" if agg["n_crossed"] else "   never"
    hl = f"{agg['hl']:10.0f}" if np.isfinite(agg["hl"]) else "     never"
    print(f"  {name:16s} {agg['kept']:10.1f}% {agg['t2']:6.1f}% {xh} "
          f"{hl} {agg['area']:6.1f}% {agg['final']:8.1f}% {agg['ncm']:6.1f}%")

# ---------------------------------------------------------------- figures
# Runs have different lengths (each stops when its task 2 is competent), so pad to plot.
names = list(CONDITIONS)
n_ev = max(len(o["steps"]) for n in names for o in runs[n])
steps = max((o["steps"] for n in names for o in runs[n]), key=len)
padded = {}
for n in names:
    A = np.full((SEEDS, n_ev, 2), np.nan)
    for i, o in enumerate(runs[n]):
        A[i, :len(o["steps"])] = o["curves"]["argmax"]
    padded[n] = A

sw_mean = np.mean([o["switches"][0] for o in runs["control"]])
plot_learning_curves(
    steps, padded, names, figure_path(__file__),
    title=f"Task 2 stopped at {T2_THRESHOLD:.0%}, not run to saturation\n{base.describe()}"
          f"  |  backprop, {SEEDS} seeds  |  compare with script 42, where it saturated",
    blocks=[(0, sw_mean, 0), (sw_mean, steps[-1], 1)], ncols=2,
    task_colors=TASK_COLORS, task_labels=["task 1", "task 2"],
    legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False),
)

# Each seed is drawn on its own. A MEAN trajectory would be meaningless here: the runs stop at
# different times, so averaging position at eval index i averages runs that are at different
# points of their own path. src.plotting.plot_trajectory assumes equal-length runs, which is
# true under a fixed budget and false under accuracy stopping.
figt, axt = plt.subplots(1, 4, figsize=(18, 4.4), sharex=True, sharey=True)
for ax, n in zip(axt, names):
    for i, o in enumerate(runs[n]):
        c = o["curves"]["argmax"] * 100
        sw = int(np.argmin(np.abs(o["steps"] - o["switches"][0])))
        ax.plot(c[sw:, 0], c[sw:, 1], lw=1.2, alpha=0.75, color="tab:purple")
        ax.plot(c[-1, 0], c[-1, 1], "o", color="k", ms=5)
    T = T2_THRESHOLD * 100
    ax.plot([100, 0], [0, 100], color="gray", ls=":", lw=1)
    ax.axhline(T, color="tab:blue", ls="--", lw=1)
    ax.plot([100], [T], marker="*", color="tab:green", ms=14, ls="none")
    ax.set_title(n)
    ax.set_xlabel("task 1 accuracy (%)")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); ax.grid(alpha=0.2)
axt[0].set_ylabel("task 2 accuracy (%)")
figt.suptitle("The trade-off with time removed — each line is one seed, from the task switch onward.\n"
              f"Rightward along the {T2_THRESHOLD:.0%} line = task 2 learned while task 1 kept. "
              "Black dot = where the run stopped.")
figt.tight_layout()
figt.savefig(figure_path(__file__, "trajectory"), dpi=120, bbox_inches="tight")

np.savez(array_path(__file__), steps=np.asarray(steps), switch=sw_mean, hidden=HIDDEN,
         t1_threshold=T1_THRESHOLD, t2_threshold=T2_THRESHOLD,
         **{f"argmax_{n}": padded[n] for n in names},
         **{f"{k}_{n}": summary[n][k] for n in names for k in summary[n]})
print(f"\nsaved {figure_path(__file__)}\nsaved {figure_path(__file__, 'trajectory')}"
      f"\nsaved {array_path(__file__)}")
