"""Does masking raise the probe's ceiling (a genuine representational change) or leave it flat
(unlocking existing information without changing it) -- and same question for the two freezes?

800-series TRAINING RUN. Arrays only; figure not yet written.

WHY THIS EXISTS. 806 found masking recovers task-1 retention by changing what W1 learns, not by
sparing the readout weights -- a trunk account, not a readout account. 923 measures the trunk's
information content (a refit linear probe) but only for the unmodified control. This is the
direct link between the two toolkits: refit the SAME probe on the FINAL trunk of each of 806's
four conditions, plus its own untrained floor, so "does masking change the representation" gets
a real answer instead of an inference from the retention number alone.

    probe rises with masking, above the floor      the trunk itself is more separable --
                                                    masking is not just calibration protection
    probe stays flat, near the unmasked control     masking recovers ACCESS to the same
                                                    information the control already had, via a
                                                    different route through the readout

FOUR CONDITIONS, same as 806: control, freeze_w2, freeze_w2_t1, mask. Same on_task_end freeze
logic, reused directly rather than reimplemented.

⚠ MASK CONDITION IS EXPENSIVE AND OFTEN DOES NOT REACH 90% ON TASK 2 (806, corrected numbers:
backprop 9/10 seeds capped, pc 5/10, at a 20000-step budget). This script starts at 20000
directly rather than 5000-then-extend, since 806 already established that 5000 undercounts badly
here -- no point re-deriving that.

PROVENANCE: fresh, config_800.yaml except max_iters_per_task (see above) and mask (per condition).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np

from src.protocol import PROTOCOL, replace, load, build, array_path
from src.runner import run_classil, _acc
from src.metrics import crossover
from src.probes import linear_probe_fn

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SEEDS = CFG["training"]["seeds"]
SEED_START = CFG["training"]["seed_start"]
CONDITIONS = ["control", "freeze_w2", "freeze_w2_t1", "mask"]
RIDGE = CFG["mechanism"]["probe_ridge"]
CAP = 20000   # see docstring -- 806 already showed 5000 undercounts the mask condition badly

SMOKE = "--smoke" in sys.argv
if SMOKE:
    SEEDS, CAP = 2, 200

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
    max_iters_per_task=CAP,
    seeds=SEEDS, eval_per_class=CFG["evaluation"]["eval_per_class"],
    eval_every=CFG["evaluation"]["eval_every"], device=CFG["evaluation"]["device"],
)
PC_KW = dict(dt=CFG["predictive_coding"]["dt"],
             steps=CFG["predictive_coding"]["settle_step_cap"],
             stop_delta=CFG["predictive_coding"]["stop_delta"],
             stop_patience=CFG["predictive_coding"]["stop_patience"])


def _task_acc(fn, x, y, task, classes, lmap):
    a = _acc(fn(x), y, classes, lmap)
    pos = {c: i for i, c in enumerate(classes)}
    return 100.0 * float(np.mean([a[pos[c]] for c in task]))


def run_one(condition, method, seed, data_by_mask):
    proto = replace(BASE, mask=(condition == "mask"))
    data = data_by_mask[proto.mask]
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)          # None in Class-IL
    classes = sorted({c for t in tasks for c in t})
    fit_x, fit_y = data.stop_eval
    rep_x, rep_y = data.report_eval

    handle = {}
    kw = dict(PC_KW) if method == "pc" else {}
    train_step, predict = build(proto, method, seed, handle=handle, **kw)

    # probe FLOOR, before a single update -- same convention as 803/808
    p0 = linear_probe_fn(handle["features"], fit_x, fit_y, proto.out_dim, ridge=RIDGE)
    floor = [_task_acc(p0, rep_x, rep_y, t, classes, lmap) for t in tasks]

    def on_task_end(ti, step):
        if ti != 0:
            return
        if condition == "freeze_w2":
            handle["freeze"].add("W2")
        elif condition == "freeze_w2_t1":
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

    # probe on the FINAL trunk -- fit on stop_eval, score on report_eval, disjoint (as 803/923)
    p_final = linear_probe_fn(handle["features"], fit_x, fit_y, proto.out_dim, ridge=RIDGE)
    probe_final = [_task_acc(p_final, rep_x, rep_y, t, classes, lmap) for t in tasks]

    return dict(final_t1=float(curve[-1, 0]), final_t2=float(curve[-1, 1]),
                crossover=float(cx) if cx is not None else float("nan"),
                reached=np.asarray(out["reached"], dtype=bool),
                probe_floor=np.asarray(floor, dtype=np.float32),
                probe_final=np.asarray(probe_final, dtype=np.float32))


if __name__ == "__main__":
    t0 = time.perf_counter()
    data_by_mask = {m: load(replace(BASE, mask=m)) for m in (False, True)}
    rows = []
    for condition in CONDITIONS:
        for method in METHODS:
            for seed in range(SEED_START, SEED_START + SEEDS):
                r = run_one(condition, method, seed, data_by_mask)
                rows.append((condition, method, seed, r))
            fin = [x[3]["final_t1"] for x in rows[-SEEDS:]]
            pf = [x[3]["probe_final"][0] for x in rows[-SEEDS:]]
            fl = [x[3]["probe_floor"][0] for x in rows[-SEEDS:]]
            print(f"  {condition:13s} {method:9s}  final_t1 {np.mean(fin):5.1f}  "
                  f"probe {np.mean(pf):5.1f}  floor {np.mean(fl):5.1f}  "
                  f"above_floor {np.mean(pf) - np.mean(fl):+5.1f}"
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
