"""Does backprop forget task 1 when task 2 arrives?

Slide 2. This is a MOTIVATING EXAMPLE, not a measurement, and its deviation from the protocol
(presentation_plan.md §2) is exactly that. Three parts, all serving the same purpose:

  * backprop only, no controls -- nothing here is compared to anything
  * the two tasks ALTERNATE for three cycles (1 2 1 2 1 2), so the pattern is unmistakable
    rather than a single event the audience has to take on trust
  * each block gets a FIXED BUDGET rather than an accuracy threshold, so every seed shares one
    x-axis and the shaded background means the same thing for all of them

Everything else -- data, architecture, output structure, optimiser, batch size, eval sets -- is
the protocol.

PROVISIONAL WIDTH. Script 41 fixes H on capacity grounds. Until it has run, HIDDEN below is a
placeholder and this script is re-run afterwards. It is one backprop run; that costs about a
minute. The claim it makes has held at every width tried in this project.

READINGS COMMITTED BEFORE RUNNING
  Task 1 rises, then collapses once task 2 starts, and recovers when task 1 returns. Recovery
  should be FASTER than the original learning -- the weights are not starting from scratch, and
  that gap between "forgotten" and "gone" is what makes the phenomenon interesting rather than
  trivial. If task 1 does not collapse at all, backprop is not forgetting at this width and the
  premise of the whole talk needs re-examining before anything else runs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np

from src.protocol import PROTOCOL, load, run, replace, figure_path, array_path
from src.plotting import plot_learning_curves

# ---------------------------------------------------------------- settings
HIDDEN = 64             # PROVISIONAL -- script 41 decides this
CYCLES = 3              # 1 2 1 2 1 2
ITERS_PER_BLOCK = 400   # fixed budget, so all seeds share an x-axis
EVAL_EVERY = 10
SEEDS = 5

# task 1 orange, task 2 blue, as on the slide
TASK_COLORS = ["tab:orange", "tab:blue"]

proto = replace(PROTOCOL,
                hidden=HIDDEN,
                stop_threshold=None,               # fixed budget, not a threshold
                max_iters_per_task=ITERS_PER_BLOCK,
                eval_every=EVAL_EVERY,
                seeds=SEEDS)

# ---------------------------------------------------------------- run
data = load(proto)
runs, splits = [], []

for seed in range(proto.seeds):
    pair = proto.tasks(seed)                        # the 2x5 split drawn for this seed
    schedule = [pair[i % 2] for i in range(2 * CYCLES)]
    out = run(proto, "backprop", seed, data=data, tasks=schedule)

    # One column per BLOCK, so columns 0, 2, 4 are the same class set and 1, 3, 5 are the other.
    # Keep the first of each; they are duplicates by construction.
    runs.append(out["curves"]["argmax"][:, :2])
    splits.append(pair)

steps = out["steps"]
curves = {"backprop": np.stack(runs)}               # [seeds, evals, 2]

# every seed has the same block boundaries because the budget is fixed
bounds = [b * ITERS_PER_BLOCK for b in range(2 * CYCLES + 1)]
blocks = [(bounds[b], bounds[b + 1], b % 2) for b in range(2 * CYCLES)]

# ---------------------------------------------------------------- figure
plot_learning_curves(
    steps, curves, ["backprop"], figure_path(__file__),
    blocks=blocks, ncols=1, task_colors=TASK_COLORS,
    task_labels=["task 1", "task 2"],
    legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False),
)

# ---------------------------------------------------------------- numbers on the slide
A = curves["backprop"] * 100.0                      # [seeds, evals, 2]


def at(step, task):
    """Mean and SEM over seeds of `task` accuracy at the eval nearest `step`."""
    i = int(np.argmin(np.abs(np.asarray(steps) - step)))
    v = A[:, i, task]
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


print(f"\n{'':22s} {'task 1':>16s} {'task 2':>16s}")
for b in range(2 * CYCLES):
    end = bounds[b + 1]
    m1, e1 = at(end, 0)
    m2, e2 = at(end, 1)
    trained = f"after block {b + 1} (task {b % 2 + 1})"
    print(f"{trained:22s} {m1:8.1f} ± {e1:4.1f}%  {m2:8.1f} ± {e2:4.1f}%")

# how much of task 1 survives the first time task 2 is trained
peak1, _ = at(bounds[1], 0)
drop1, _ = at(bounds[2], 0)
print(f"\ntask 1: {peak1:.1f}% -> {drop1:.1f}% across the first task-2 block "
      f"({drop1 - peak1:+.1f} points)")

# The collapse floor is the score of a model that predicts one class for everything. Landing
# BELOW it is not "no knowledge" -- it means task-2 output units are winning the argmax on
# task-1 images, which is output-layer suppression rather than a destroyed representation.
# That distinction is what slides 5-7 are about, and script 42 measures it.
floor = 100.0 / proto.n_classes
print(f"collapse floor is {floor:.0f}%; task 1 lands at {drop1:.1f}%, "
      f"{'BELOW it — output-layer suppression' if drop1 < floor else 'at or above it'}")

# is relearning faster than learning? steps within a block to pass the first block's peak
def steps_to(level, block):
    lo, hi = bounds[block], bounds[block + 1]
    task = block % 2
    for i, s in enumerate(steps):
        if lo < s <= hi and A[:, i, task].mean() >= level:
            return s - lo
    return None


print(f"updates to reach {peak1:.0f}% on task 1 -- first time: {steps_to(peak1, 0)}, "
      f"second: {steps_to(peak1, 2)}, third: {steps_to(peak1, 4)}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__),
         steps=np.asarray(steps), curves=curves["backprop"],
         bounds=np.asarray(bounds), block_task=np.asarray([b % 2 for b in range(2 * CYCLES)]),
         splits=np.asarray(splits), hidden=HIDDEN, iters_per_block=ITERS_PER_BLOCK)
print(f"saved {array_path(__file__)}")
