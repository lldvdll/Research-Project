"""How much of the forgetting survives when the output push is removed, and when the hidden
layer is frozen?

Slide 7, second half, and the go/no-go for the whole second half of the talk.

THE ARGUMENT THIS HAS TO SETTLE
    Slides 5 and 6 claim, from the maths rather than from data, that a learning rule changes
    credit assignment and NOT the output error signal -- every rule gets the same e = -dL/dout.
    So if the forgetting is entirely the output layer being pushed down, no learning rule can
    help and there is nothing for slides 8-19 to compare. If a substantial part is the hidden
    representation drifting, a rule that protects the representation has room to act.

    Two interventions, factorially, because the point is to SPLIT the forgetting:
        masked   absent classes get zero error, so nothing pushes task-1 output units down.
                 This is the part slides 5-6 say a rule cannot change.
        frozen   W1 and b1 held from the task switch, so the hidden code cannot drift.
                 This is the part a rule could in principle protect.
    Neither alone identifies a component -- their interaction can be large -- which is why all
    four cells are run.

    Read alongside script 41: 41 says the architecture is not the limit, so whatever this finds
    is about the mechanism rather than about underfitting.

    NCM is measured on the same runs, at no extra cost, because it answers the same question
    from the other side. Nearest-class-mean discards the output layer and classifies from the
    hidden code alone:
        argmax low, NCM high  -> the code survived; the damage is in the output layer
        both low              -> the code was destroyed
    Script 40 already hinted at the answer -- relearning got five times faster across cycles and
    task 1 fell BELOW the collapse floor -- and both point at the output layer.

Deviation from the protocol (§2): backprop only, and the two interventions above. Fixed budgets
per task rather than accuracy thresholds so all four conditions share one x-axis and the panels
are directly comparable; task-1 accuracy at the switch is reported so any difference in how well
the conditions learned it in the first place is visible rather than assumed away.

H is read from script 41's saved output. Run 41 first.

ONE ARTEFACT TO READ PAST, IN THE MASKED PANELS
    Task-1 accuracy DECLINES during task 1 under masking, from ~89% to ~78%. That is not
    forgetting. Under a masked loss the task-2 output units are never a target, so they keep
    their random initial weights and are never pushed down, while the hidden code they read from
    keeps growing -- so they capture the argmax on task-1 images more and more often. It affects
    the at-the-switch number only. By the end of task 2 every unit has been a target, so the
    final retention figures, which are what this script reports, are unaffected.

READINGS COMMITTED BEFORE RUNNING
    If masking alone restores most of task 1, the forgetting is output-layer suppression, no
    rule can touch it, and slide 14 should be expected to find nothing -- a real result, and one
    worth knowing before running it.
    If freezing alone restores most of it, the forgetting is representation drift and there is
    room for a rule to help.
    If neither alone restores much but both together do, the two mechanisms interact and neither
    can be discussed on its own.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np

from src.protocol import PROTOCOL, load, build, replace, figure_path, array_path
from src.runner import run_classil
from src.probes import live_ncm_fn, prototype_images
from src.plotting import plot_learning_curves

# ---------------------------------------------------------------- settings
ITERS_TASK1 = 2000
ITERS_TASK2 = 2000
EVAL_EVERY = 25
SEEDS = 5
TASK_COLORS = ["tab:orange", "tab:blue"]           # task 1 orange, task 2 blue, as on slide 2

# H comes from script 41, not from a guess or an earlier experiment
cap = np.load(array_path(str(ROOT / "experiments" / "41_capacity_vs_hidden_width.py")))
HIDDEN = int(cap["chosen"])
CEILING = float(cap["acc_class_il"].mean(1).max())
print(f"H = {HIDDEN} from script 41 (joint ceiling {CEILING:.1f}%)\n")

CONDITIONS = {                                     # (mask the loss, freeze the hidden layer)
    "control": (False, False),
    "masked": (True, False),
    "frozen hidden": (False, True),
    "masked + frozen": (True, True),
}

base = replace(PROTOCOL, hidden=HIDDEN, stop_threshold=None,
               max_iters_per_task=[ITERS_TASK1, ITERS_TASK2],
               eval_every=EVAL_EVERY, seeds=SEEDS)

# ---------------------------------------------------------------- run
data = load(base)
curves = {c: [] for c in CONDITIONS}
ncm_curves = {c: [] for c in CONDITIONS}

for name, (mask, freeze) in CONDITIONS.items():
    proto = replace(base, mask=mask)
    for seed in range(SEEDS):
        tasks = proto.tasks(seed)                  # same split for every condition at this seed
        handle = {}
        train_step, predict = build(proto, "backprop", seed, handle=handle)

        px, py = prototype_images(data.train, data.class_idx,
                                  sorted(tasks[0] + tasks[1]), per_class=50,
                                  device=proto.device, seed=seed)
        readouts = {"argmax": predict,
                    "ncm": live_ncm_fn(handle["features"], px, py)}

        def on_task_end(ti, step, _f=freeze, _h=handle):
            # Freeze at the SWITCH, not from the start -- task 1 has to be learned first.
            # W1 and b1 together: both shape the hidden pre-activation, and freezing only W1
            # would leave the code free to drift through the bias.
            if ti == 0 and _f:
                _h["freeze"].update({"W1", "b1"})

        out = run_classil(
            train_step, predict, tasks, data.train, data.class_idx,
            report_eval=data.report_eval, stop_eval=data.stop_eval, readouts=readouts,
            max_iters_per_task=[ITERS_TASK1, ITERS_TASK2], batch=proto.batch,
            eval_every=EVAL_EVERY, device=proto.device, data_seed=seed,
            on_task_end=on_task_end,
        )
        curves[name].append(out["curves"]["argmax"])
        ncm_curves[name].append(out["curves"]["ncm"])

    steps, switches = out["steps"], out["switches"]
    A = np.stack(curves[name]) * 100
    i_sw = int(np.argmin(np.abs(np.asarray(steps) - switches[0])))
    print(f"  {name:16s} task 1 at switch {A[:, i_sw, 0].mean():5.1f}%"
          f"   -> after task 2 {A[:, -1, 0].mean():5.1f}%"
          f"   (task 2 {A[:, -1, 1].mean():5.1f}%)")

curves = {k: np.stack(v) for k, v in curves.items()}
ncm_curves = {k: np.stack(v) for k, v in ncm_curves.items()}
names = list(CONDITIONS)
blocks = [(0, switches[0], 0), (switches[0], steps[-1], 1)]

# ---------------------------------------------------------------- main figure
plot_learning_curves(
    steps, curves, names, figure_path(__file__),
    title=f"How much of the forgetting can an intervention remove?\n{base.describe()}"
          f"  |  backprop, {SEEDS} seeds  |  joint ceiling {CEILING:.1f}%",
    blocks=blocks, ncols=2, task_colors=TASK_COLORS, task_labels=["task 1", "task 2"],
    legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False),
)

# ------------------------------------------------- supporting figure: is the code still there?
plot_learning_curves(
    steps, ncm_curves, names, figure_path(__file__, "ncm"),
    title="Diagnostic: nearest-class-mean, output layer discarded\n"
          "high here while argmax is low means the hidden code survived and the head is at fault",
    blocks=blocks, ncols=2, task_colors=TASK_COLORS, task_labels=["task 1", "task 2"],
    legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False),
)

# ---------------------------------------------------------------- readings
print()
ctrl_peak = curves["control"][:, i_sw, 0].mean() * 100
print(f"{'condition':16s} {'task1 kept':>11s} {'recovered':>10s} {'NCM task1':>10s}")
recovered = {}
for n in names:
    kept = curves[n][:, -1, 0].mean() * 100
    base_kept = curves["control"][:, -1, 0].mean() * 100
    recovered[n] = (kept - base_kept) / max(1e-9, ctrl_peak - base_kept) * 100
    print(f"{n:16s} {kept:10.1f}% {recovered[n]:9.0f}% {ncm_curves[n][:, -1, 0].mean()*100:9.1f}%")

print(f"\n  'recovered' = fraction of the control's lost task-1 accuracy that this condition "
      f"restores\n  (control lost {ctrl_peak - curves['control'][:, -1, 0].mean()*100:.1f} points)")

m, f = recovered["masked"], recovered["frozen hidden"]
both = recovered["masked + frozen"]
print(f"\n  masking alone {m:.0f}%, freezing alone {f:.0f}%, both {both:.0f}% "
      f"-> interaction {both - m - f:+.0f} points")
print("  -> " + ("output-layer suppression dominates; a learning rule has little room here"
                 if m > f + 15 else
                 "representation drift dominates; a learning rule has room to act"
                 if f > m + 15 else
                 "neither dominates — the two mechanisms are comparable in size"))

np.savez(array_path(__file__), steps=np.asarray(steps), switches=np.asarray(switches),
         hidden=HIDDEN, ceiling=CEILING,
         **{f"argmax_{n}": curves[n] for n in names},
         **{f"ncm_{n}": ncm_curves[n] for n in names})
print(f"\nsaved {array_path(__file__)}")
