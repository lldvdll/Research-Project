"""What does catastrophic forgetting look like, on one run, with every class drawn separately?

800-series TRAINING RUN. Arrays only; the figure is 901 -- the report's opening figure.

DEFINITIONAL, NOT COMPARATIVE. This figure exists to name three things every later section
refers to: the switch, the plateau each curve reaches, and the point where the two curves cross.
Nothing is compared, so nothing is averaged.

ONE SEED, FIXED BUDGET, CLASS-IL, BACKPROP -- and each of those is a deliberate departure from
the protocol used everywhere else:
    one seed         a ten-seed mean smooths away the shape the figure exists to show, and under
                     matched competence the seeds stop at different steps, so the mean's tail is
                     computed over a shrinking sample. 102 shows that averaged version; this is
                     the un-averaged one.
    fixed budget     matched competence stops each task the moment it clears threshold, which
                     truncates the picture exactly where the interesting part begins. A fixed
                     budget runs both phases to the same length and shows the whole track.
    Class-IL         the more dramatic scenario, and the one whose mechanism the rest of the
                     report spends longest on.
    backprop         rule identity is not introduced until R2.

PER-CLASS, NOT PER-TASK. Every other script logs accuracy averaged within a task. This logs all
ten classes separately, because the five task-1 classes falling to zero while the five task-2
classes rise IS the output-competition mechanism -- so the opening figure shows the thing R5
later explains, rather than introducing it cold.

DEVIATION FROM config_800.yaml: seeds=1, fixed budget instead of matched competence, per-class
logging. Everything else -- width, activation, output specification, learning rate, batch -- is
as written there, so this is the same network the rest of the report measures.

⚠ THE CAPTION MUST SAY THIS RUNS TO SATURATION. At BUDGET=900 task 1 ends at exactly 0.0% on all
five of its classes, where R1's matched-competence figure reports 5.4%. Both are correct: a
saturating budget drives the unmasked condition to the floor, which is precisely why every
MEASURED result in the report reads at matched competence instead (42/43 established this, and
progress.md records it as a decision). The definitional figure wants the complete picture and can
afford saturation because it quantifies nothing. Left unstated, a reader meets 0% here and 5.4%
twenty pages later and concludes the report contradicts itself.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import torch

from src.protocol import PROTOCOL, replace, load, build, array_path
from src.runner import _loader, _next

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

SEED = CFG["training"]["seed_start"]     # 10, the canonical block's first seed
BUDGET = 900                             # updates per task. Long enough that task 1 plateaus and
                                         # task 2's rise completes; short enough to stay legible.
LOG_EVERY = 5                            # this is one run, so it can afford a fine curve

# Learning rate is overridden BELOW the protocol default (0.02) on purpose. At the shared rate
# the whole transition happens in roughly a hundred updates, so the crossing, the forgetting
# slope and the surviving task-1 accuracy all pile into one narrow band and the opening figure
# cannot be read. A quarter of the rate stretches the same dynamics out far enough to annotate,
# and it leaves task 1 measurably above zero at the end so there is a retention to point at.
# NOTHING is measured from this run, so the rate does not have to match the calibrated series --
# but the caption must say it is lower, or a reader will compare its numbers with R1's.
LR = 0.005

SMOKE = "--smoke" in sys.argv
if SMOKE:
    BUDGET = 120

PROTO = replace(
    PROTOCOL, scenario="class_il",
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    hidden=CFG["architecture"]["hidden"], n_layers=CFG["architecture"]["n_layers"],
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    lr={"backprop": LR}, seeds=1,
    eval_per_class=CFG["evaluation"]["eval_per_class"], device=CFG["evaluation"]["device"],
)


if __name__ == "__main__":
    t0 = time.perf_counter()
    data = load(PROTO)
    tasks = PROTO.tasks(SEED)
    classes = sorted({c for t in tasks for c in t})
    train_step, predict = build(PROTO, "backprop", SEED, handle={})
    rep_x, rep_y = data.report_eval

    def per_class():
        """Accuracy on each of the ten classes separately. Class-IL, so class index IS unit."""
        with torch.no_grad():
            p = predict(rep_x)
        return [100.0 * float((p[rep_y == c] == c).float().mean()) for c in classes]

    steps, acc = [0], [per_class()]
    for ti, task in enumerate(tasks):
        loader = _loader(data.train, data.class_idx, task, PROTO.batch, seed=SEED)
        it = iter(loader)
        for k in range(1, BUDGET + 1):
            (x, y), it = _next(it, loader)
            train_step(x.to(PROTO.device), y.to(PROTO.device), task)
            step = ti * BUDGET + k
            if step % LOG_EVERY == 0:
                steps.append(step)
                acc.append(per_class())

    acc = np.asarray(acc, dtype=np.float32)        # [n_eval, 10]
    steps = np.asarray(steps)
    t1_cols = [classes.index(c) for c in tasks[0]]
    t2_cols = [classes.index(c) for c in tasks[1]]
    suffix = "SMOKE" if SMOKE else ""
    np.savez_compressed(array_path(__file__, suffix), steps=steps, acc=acc,
                        classes=np.array(classes), task1=np.array(tasks[0]),
                        task2=np.array(tasks[1]), switch=BUDGET, seed=SEED)
    print(f"  task 1 = {tasks[0]}   task 2 = {tasks[1]}   switch at {BUDGET}")
    print(f"  end of task 1: t1 {acc[len(acc)//2, t1_cols].mean():5.1f}  "
          f"t2 {acc[len(acc)//2, t2_cols].mean():5.1f}")
    print(f"  end of run   : t1 {acc[-1, t1_cols].mean():5.1f}  t2 {acc[-1, t2_cols].mean():5.1f}")
    print(f"  per-class task-1 at end: "
          f"{dict(zip(tasks[0], np.round(acc[-1, t1_cols], 1)))}")
    print(f"saved {array_path(__file__, suffix)}   [{time.perf_counter() - t0:.0f}s]")
