"""Is Class-IL forgetting the task-1 output weights being driven down, or the competition at
argmax -- and is that why freezing W2 fails where masking works?

800-series TRAINING RUN. Arrays only; the figure is 922.

THE CONTRADICTION THIS RESOLVES. Masking the absent classes recovers +17.5 points (120).
Freezing the whole output layer recovers nothing, +0.36 +- 0.43 (130). Both spare the task-1
output weights during task 2, so on the naive reading they should agree. They do not, and the
difference between them is precisely one thing:

    masking          is TRAIN-TIME ONLY. active_vector zeroes the error inside output_error, so
                     the task-1 units receive no gradient -- while `predict` still argmaxes over
                     all ten units at evaluation, and task 2's units keep learning normally.
    freeze all W2    spares the task-1 columns too, but ALSO blocks task 2 from learning through
                     the readout at all, forcing it into W1 and damaging the shared trunk.

FOUR CONDITIONS, paired on seed, so the two effects can be separated for the first time:

    control          nothing held
    freeze_w2        the whole output layer held (reproduces 130)
    freeze_w2_t1     ONLY the task-1 output columns held -- masking's intervention expressed in
                     weight space, with task 2's readout left free
    mask             masking itself (obj.mask=True), for the direct comparison

PRE-COMMITTED READING, so this cannot be rationalised afterwards:
    freeze_w2_t1 ~ mask       the mechanism is the task-1 readout weights being driven down, and
                              130's null was an artefact of crippling task-2 learning.
    freeze_w2_t1 ~ control    masking does NOT work by sparing those weights; it works through
                              the evaluation-time competition, and the readout account needs
                              rewriting.
    in between                both effects are real and the figure reports their relative size.

Class-IL only: Domain-IL has no absent classes, so neither masking nor a task-1 column freeze is
defined there. That asymmetry is the same structural fact the scenario split rests on.

DEVIATION FROM config_800.yaml: `mask` is varied per condition; every other field is as written.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np

from src.protocol import PROTOCOL, replace, load, build, array_path
from src.runner import run_classil
from src.metrics import crossover

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SEEDS = CFG["training"]["seeds"]
SEED_START = CFG["training"]["seed_start"]
CONDITIONS = ["control", "freeze_w2", "freeze_w2_t1", "mask"]

SMOKE = "--smoke" in sys.argv
if SMOKE:
    SEEDS = 2

BASE = replace(
    PROTOCOL, scenario="class_il",
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    hidden=CFG["architecture"]["hidden"], n_layers=CFG["architecture"]["n_layers"],
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=False, reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    lr=dict(CFG["training"]["learning_rate"]),
    stop_threshold=CFG["training"]["stop_threshold"],
    stop_patience=CFG["training"]["stop_patience"],
    max_iters_per_task=(200 if SMOKE else CFG["training"]["max_iters_per_task"]),
    seeds=SEEDS, eval_per_class=CFG["evaluation"]["eval_per_class"],
    eval_every=CFG["evaluation"]["eval_every"], device=CFG["evaluation"]["device"],
)
PC_KW = dict(dt=CFG["predictive_coding"]["dt"],
             steps=CFG["predictive_coding"]["settle_step_cap"],
             stop_delta=CFG["predictive_coding"]["stop_delta"],
             stop_patience=CFG["predictive_coding"]["stop_patience"])


def run_one(condition, method, seed, data_by_mask):
    proto = replace(BASE, mask=(condition == "mask"))
    data = data_by_mask[proto.mask]
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)          # None in Class-IL: class index IS the unit index
    handle = {}
    kw = dict(PC_KW) if method == "pc" else {}
    train_step, predict = build(proto, method, seed, handle=handle, **kw)

    def on_task_end(ti, step):
        """Applied at the END of task 1, so task 1 itself trains unhindered in every condition."""
        if ti != 0:
            return
        if condition == "freeze_w2":
            handle["freeze"].add("W2")
        elif condition == "freeze_w2_t1":
            # Class-IL: the task-1 CLASSES are the task-1 output UNITS. cols must be hashable,
            # because `freeze` is a set (see model.freeze_columns).
            #
            # BOTH W2's columns AND b2's entries. Masking zeroes the ERROR on those units, which
            # stops their weights and their biases moving. Holding only W2 would leave the
            # task-1 biases free to be driven down, so the condition would differ from masking
            # in two ways instead of one and the comparison would answer nothing. b2 is 1-D and
            # [..., cols] indexes its last axis, so the same freeze entry form works.
            t1_units = tuple(sorted(tasks[0]))
            handle["freeze"].add(("W2", t1_units))
            handle["freeze"].add(("b2", t1_units))

    out = run_classil(
        train_step, predict, tasks, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, log_every=CFG["evaluation"]["log_every"],
        device=proto.device, stop_threshold=proto.stop_threshold,
        stop_patience=proto.stop_patience, data_seed=seed, label_map=lmap,
        on_task_end=on_task_end)

    steps = np.asarray(out["steps"])
    curve = np.asarray(out["curves"]["argmax"], dtype=float) * 100.0
    switch0 = int(out["switches"][0])
    _, cx = crossover(steps, curve[:, 0], curve[:, 1], after=switch0)
    return dict(steps=steps, t1=curve[:, 0], t2=curve[:, 1], switch0=switch0,
                final_t1=float(curve[-1, 0]), final_t2=float(curve[-1, 1]),
                crossover=float(cx) if cx is not None else float("nan"),
                reached=np.asarray(out["reached"], dtype=bool))


if __name__ == "__main__":
    t0 = time.perf_counter()
    # mask changes the OBJECTIVE, not the data, but load() keys off the protocol, so both are
    # built once rather than per run.
    data_by_mask = {m: load(replace(BASE, mask=m)) for m in (False, True)}
    rows = []
    for condition in CONDITIONS:
        for method in METHODS:
            for seed in range(SEED_START, SEED_START + SEEDS):
                r = run_one(condition, method, seed, data_by_mask)
                rows.append((condition, method, seed, r))
            fin = [x[3]["final_t1"] for x in rows[-SEEDS:]]
            cx = [x[3]["crossover"] for x in rows[-SEEDS:]]
            nd = int(np.isfinite(cx).sum())
            print(f"  {condition:13s} {method:9s}  final t1 {np.mean(fin):5.1f}  "
                  f"crossover {np.nanmean(cx):5.1f} (defined {nd}/{SEEDS})"
                  f"  [{time.perf_counter() - t0:6.0f}s]", flush=True)
    suffix = "SMOKE" if SMOKE else ""
    np.savez_compressed(
        array_path(__file__, suffix),
        conditions=np.array([c for c, _, _, _ in rows]),
        methods=np.array([m for _, m, _, _ in rows]),
        seeds=np.array([s for _, _, s, _ in rows]),
        **{f"{k}_{i}": v for i, (_, _, _, r) in enumerate(rows) for k, v in r.items()})
    print(f"saved {array_path(__file__, suffix)}")
    print(f"done in {time.perf_counter() - t0:.0f}s")
