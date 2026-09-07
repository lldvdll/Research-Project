"""Where does Domain-IL's damage sit -- does the hidden code drift, and for which units?

800-series TRAINING RUN. Arrays only; the figure is 924b.

WHY THIS RUN HAS TO EXIST. 007 identifies Domain-IL's localisation as the largest genuine gap in
the project: output suppression is structurally impossible there, so the explanation must be
representation drift -- and freezing only says where it is NOT. 803 logs the linear probe's
ACCURACY at each checkpoint but never the hidden codes themselves, and no other saved array holds
them, so no existing data can answer it. `src/probes.py` already provides `code_snapshot` and
`code_drift`, so this is instrumentation of the established protocol rather than a new experiment.

WHAT IS RECORDED, on a FIXED task-1 batch that is never trained on, at every eval through task 2:

    cosine   mean per-image cosine between the current hidden code and the pre-switch one.
             1.0 means the direction is unchanged.
    rel_l2   mean ||after - before|| / ||before||. 0.0 means identical.
    Both, because a rigid rotation gives low cosine while leaving distances intact, and the two
    readings call for different explanations. `code_drift` returns them together for that reason.

RESOLVED BY UNIT SELECTIVITY, which is the half 007's SK_DRIFT card asks for and the half no
existing measurement has. Each hidden unit is labelled at the switch by which task it responds to
more strongly (mean absolute activation on task-1 images against task-2 images), and drift is
reported separately for the two groups. If Domain-IL's damage is representational, the units that
carried task 1 should move more than the ones that did not.

THE FROZEN-TRUNK BASELINE IS THE CONTROL. A second arm freezes W1 AND b1 during task 2, so its
hidden code cannot move and its drift is zero by construction. It is a floor proving the
measurement reads the trunk rather than eval noise. ⚠ Freezing W1 alone is NOT sufficient -- the
hidden bias keeps learning and the code still drifts (rel_l2 0.14 in the smoke run, against 0.56
unfrozen). Both are frozen for that reason.

BOTH SCENARIOS, because the claim is comparative: Class-IL can forget by suppression, so its code
may drift less for the same accuracy loss. That contrast is the point.

DEVIATION FROM config_800.yaml: none. Same width, activation, output specification, optimiser,
batch and matched-competence stopping as every other 800 run.
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
from src.runner import run_classil
from src.probes import code_snapshot, code_drift

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SEEDS = CFG["training"]["seeds"]
SEED_START = CFG["training"]["seed_start"]
N_PER_CLASS = 40                       # images per task-1 class in the fixed probe batch
DRIFT_EVERY = 10                       # sample the hidden code every N updates

SMOKE = "--smoke" in sys.argv
if SMOKE:
    SEEDS = 2

BASE = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    hidden=CFG["architecture"]["hidden"], n_layers=CFG["architecture"]["n_layers"],
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    lr=dict(CFG["training"]["learning_rate"]),
    stop_threshold=CFG["training"]["stop_threshold"],
    stop_patience=CFG["training"]["stop_patience"],
    max_iters_per_task=(200 if SMOKE else CFG["training"]["max_iters_per_task"]),
    seeds=SEEDS, eval_per_class=CFG["evaluation"]["eval_per_class"],
    eval_every=CFG["evaluation"]["eval_every"], device=CFG["evaluation"]["device"],
)
PC_KW = dict(dt=CFG["predictive_coding"]["dt"], steps=CFG["predictive_coding"]["settle_step_cap"],
             stop_delta=CFG["predictive_coding"]["stop_delta"],
             stop_patience=CFG["predictive_coding"]["stop_patience"])


def selectivity(features_fn, x1, x2):
    """Per unit: does it respond more to task-1 or task-2 images? Labelled AT THE SWITCH."""
    with torch.no_grad():
        a1 = features_fn(x1).abs().mean(dim=0)
        a2 = features_fn(x2).abs().mean(dim=0)
    return (a1 > a2).cpu().numpy()          # True = task-1 selective


def run_one(proto, method, seed, data, freeze_w1):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    handle = {}
    kw = dict(PC_KW) if method == "pc" else {}
    train_step, predict = build(proto, method, seed, handle=handle, **kw)
    feats = handle["features"]

    def fixed(task):
        idx = [i for c in task for i in data.class_idx[c][:N_PER_CLASS]]
        return torch.stack([data.train[i][0] for i in idx]).to(proto.device)

    x1, x2 = fixed(tasks[0]), fixed(tasks[1])
    ref = [None]
    sel = [None]
    log = []            # (step, cosine, rel_l2, cosine_t1_units, cosine_t2_units)

    def on_task_end(ti, step):
        """At the END of task 1: fix the reference code and label unit selectivity."""
        if ti != 0:
            return
        ref[0] = code_snapshot(feats, x1)
        sel[0] = selectivity(feats, x1, x2)
        if freeze_w1:
            # BOTH W1 and b1. Freezing only W1 leaves the hidden bias free, and the code then
            # still moves -- the smoke run read rel_l2 0.14 rather than 0. A floor that is not
            # actually a floor is worse than no floor.
            handle["freeze"].add("W1")
            handle["freeze"].add("b1")

    # run_classil has on_task_end but NO on_eval hook, and adding one would be a src/ change.
    # 803 already solves this by wrapping train_step, so the same pattern is used here: count
    # updates and sample the code every DRIFT_EVERY of them, once the reference exists.
    step_n = [0]

    def wrapped(x, y, active=None):
        train_step(x, y, active=active)
        step_n[0] += 1
        if ref[0] is None or step_n[0] % DRIFT_EVERY:
            return
        cur = code_snapshot(feats, x1)
        d = code_drift(ref[0], cur)
        sub = lambda idx: (code_drift(ref[0][:, idx], cur[:, idx])["rel_l2"]
                           if len(idx) else float("nan"))
        log.append((step_n[0], d["cosine"], d["rel_l2"],
                    sub(np.where(sel[0])[0]), sub(np.where(~sel[0])[0])))

    out = run_classil(
        wrapped, predict, tasks, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, device=proto.device,
        stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
        data_seed=seed, label_map=lmap, on_task_end=on_task_end)

    curve = np.asarray(out["curves"]["argmax"], dtype=float) * 100.0
    return dict(drift=np.asarray(log, dtype=np.float32),
                n_t1_units=int(sel[0].sum()), n_t2_units=int((~sel[0]).sum()),
                final_t1=float(curve[-1, 0]), final_t2=float(curve[-1, 1]),
                switch0=int(out["switches"][0]))


if __name__ == "__main__":
    t0 = time.perf_counter()
    for scenario in CFG["data"]["scenarios"]:
        proto = replace(BASE, scenario=scenario)
        data = load(proto)
        rows = []
        for freeze_w1 in (False, True):
            for method in METHODS:
                if freeze_w1 and method != "backprop":
                    continue                 # the baseline needs one arm only
                for seed in range(SEED_START, SEED_START + SEEDS):
                    r = run_one(proto, method, seed, data, freeze_w1)
                    rows.append((method, freeze_w1, seed, r))
                tag = f"{method}{' +frozen W1' if freeze_w1 else ''}"
                fin = np.mean([x[3]["final_t1"] for x in rows[-SEEDS:]])
                dr = np.mean([x[3]["drift"][-1, 2] for x in rows[-SEEDS:]])
                print(f"  {scenario:10s} {tag:20s} final t1 {fin:5.1f}  end rel_l2 drift "
                      f"{dr:.4f}  [{time.perf_counter() - t0:5.0f}s]", flush=True)
        suffix = scenario + ("_SMOKE" if SMOKE else "")
        np.savez_compressed(
            array_path(__file__, suffix),
            methods=np.array([m for m, _, _, _ in rows]),
            frozen=np.array([f for _, f, _, _ in rows]),
            seeds=np.array([s for _, _, s, _ in rows]),
            **{f"{k}_{i}": v for i, (_, _, _, r) in enumerate(rows) for k, v in r.items()})
        print(f"saved {array_path(__file__, suffix)}")
    print(f"done in {time.perf_counter() - t0:.0f}s")
