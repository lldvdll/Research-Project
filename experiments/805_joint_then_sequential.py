"""If a network already solves both tasks, does training them sequentially still destroy task 1?

800-series TRAINING RUN. Arrays only; the figure is 919.

WHY THIS IS THE SHARPEST VERSION OF "WHERE DOES FORGETTING LIVE". Every localisation result so
far asks which LAYER the damage sits in. This asks something the layer question cannot reach:
whether forgetting is a failure to FIND a joint solution or a failure to STAY in one. A network
pre-trained on the joint distribution starts the sequential phase already at a point in weight
space that serves both tasks. If task 1 still collapses from there, then the joint solution
being reachable is not the issue and the account has to be about what the sequential gradient
does to parameter ORGANISATION -- not about which layer holds the damage.

TWO ARMS, PAIRED ON SEED, so the comparison is within-seed and the class split is shared:
    scratch   the standard protocol -- sequential from random init. The control.
    joint     train on all ten classes to convergence, THEN run the identical sequential
              protocol from those weights.

Both arms share one build per (rule, seed), so the joint arm's sequential phase begins from
exactly the weights the joint phase ended at. The scratch arm is rebuilt from the same seed, so
the two start from the same random initialisation.

THE JOINT ARM'S TASK-1 PHASE IS DEGENERATE, ON PURPOSE. After joint pre-training the network is
already above threshold on task 1, so that phase stops after roughly stop_patience * eval_every
steps -- about 30, against ~700 for scratch. It is kept rather than skipped so the protocol is
byte-identical between arms and nothing about the comparison depends on one of them having been
special-cased. What matters is that BOTH arms enter task-2 training sitting at ~90% on task 1,
one having reached that by training task 1 alone and the other by training both. Both phase
lengths are printed and saved (`switch0`) so this can be checked rather than trusted.

READ IT WITH 917/918. If Class-IL's alternation trace closes a loop from random init, the
question this run answers is whether joint pre-training turns that loop into a spiral. A joint
start that still collapses says the loop is a property of the sequential gradient, not of where
the run happened to begin.

DEVIATION FROM config_800.yaml: the joint phase uses its own iteration cap (JOINT_ITERS), which
is not a protocol parameter -- it is a pre-training budget, reported alongside the accuracy it
reached so a reader can see the arm actually converged before the comparison started.
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
from src.runner import run_classil, run_joint, _acc
from src.metrics import crossover

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SCENARIOS = CFG["data"]["scenarios"]
SEEDS = CFG["training"]["seeds"]
SEED_START = CFG["training"]["seed_start"]

JOINT_ITERS = 3000          # pre-training budget. 100 reaches the joint ceiling (89.7% Class-IL)
                            # well inside this; the reached accuracy is saved so it can be checked
                            # rather than assumed.
JOINT_PATIENCE = 5          # early stop once joint accuracy stops improving

SMOKE = "--smoke" in sys.argv
if SMOKE:
    SEEDS, JOINT_ITERS = 2, 200

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
PC_KW = dict(dt=CFG["predictive_coding"]["dt"],
             steps=CFG["predictive_coding"]["settle_step_cap"],
             stop_delta=CFG["predictive_coding"]["stop_delta"],
             stop_patience=CFG["predictive_coding"]["stop_patience"])


def _to_units(y, lmap):
    """Class labels -> output UNIT indices. Identity for Class-IL, where lmap is None."""
    if lmap is None:
        return y
    return torch.as_tensor([lmap[int(v)] for v in y])


def _joint_step(train_step, lmap):
    """run_joint knows nothing about label maps -- it hands raw class labels to train_step and
    scores predict() against raw labels. Both are wrong for Domain-IL, where ten classes share
    five units. run_classil does the mapping itself; this shim does the same for the joint phase.

    `active` is forced to None rather than passed through. run_joint sends the full class list,
    which would index past a 5-wide mask; and in the joint phase every unit is a legitimate
    target anyway, so no masking is the correct setting (config_800.yaml has mask: false)."""
    def step(x, y, active=None):
        return train_step(x, _to_units(y, lmap), active=None)
    return step


def sequential(proto, train_step, predict, tasks, lmap, data, seed):
    out = run_classil(
        train_step, predict, tasks, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, log_every=CFG["evaluation"]["log_every"],
        device=proto.device, stop_threshold=proto.stop_threshold,
        stop_patience=proto.stop_patience, data_seed=seed, label_map=lmap)
    steps = np.asarray(out["steps"])
    curve = np.asarray(out["curves"]["argmax"], dtype=float) * 100.0
    switch0 = int(out["switches"][0])
    _, cx = crossover(steps, curve[:, 0], curve[:, 1], after=switch0)
    return dict(steps=steps, t1=curve[:, 0], t2=curve[:, 1], switch0=switch0,
                final_t1=float(curve[-1, 0]), final_t2=float(curve[-1, 1]),
                crossover=float(cx) if cx is not None else float("nan"),
                reached=np.asarray(out["reached"], dtype=bool))


def run_one(proto, method, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    classes = sorted({c for t in tasks for c in t})
    kw = dict(PC_KW) if method == "pc" else {}

    # ---- arm 1: sequential from random init (control)
    ts, pr = build(proto, method, seed, handle={}, **kw)
    scratch = sequential(proto, ts, pr, tasks, lmap, data, seed)

    # ---- arm 2: joint pre-training, then the identical sequential protocol
    ts, pr = build(proto, method, seed, handle={}, **kw)          # same seed -> same init
    rep_x, rep_y = data.report_eval
    jsteps, jacc = run_joint(_joint_step(ts, lmap), pr, classes, data.train, data.class_idx,
                             rep_x, _to_units(rep_y, lmap),
                             max_iters=JOINT_ITERS, batch=proto.batch,
                             eval_every=proto.eval_every, device=proto.device,
                             stop_patience=JOINT_PATIENCE, data_seed=seed)
    joint_acc = 100.0 * float(np.max(jacc)) if len(jacc) else float("nan")
    a = _acc(pr(rep_x), rep_y, classes, lmap)
    pos = {c: i for i, c in enumerate(classes)}
    joint_t1 = 100.0 * float(np.mean([a[pos[c]] for c in tasks[0]]))
    joint_t2 = 100.0 * float(np.mean([a[pos[c]] for c in tasks[1]]))
    pre = sequential(proto, ts, pr, tasks, lmap, data, seed)

    return dict(
        scratch=scratch, joint=pre,
        joint_acc=joint_acc, joint_t1=joint_t1, joint_t2=joint_t2,
        joint_iters=int(jsteps[-1]) if len(jsteps) else 0)


if __name__ == "__main__":
    t0 = time.perf_counter()
    for scenario in SCENARIOS:
        proto = replace(BASE, scenario=scenario)
        data = load(proto)
        rows = []
        for method in METHODS:
            for seed in range(SEED_START, SEED_START + SEEDS):
                r = run_one(proto, method, seed, data)
                rows.append((method, seed, r))
                print(f"  {scenario:10s} {method:9s} seed {seed}  "
                      f"joint {r['joint_acc']:5.1f}% in {r['joint_iters']:4d} it "
                      f"(t1 {r['joint_t1']:5.1f}) | final t1  "
                      f"scratch {r['scratch']['final_t1']:5.1f} "
                      f"joint {r['joint']['final_t1']:5.1f}  | task-1 phase "
                      f"{r['scratch']['switch0']:4d} vs {r['joint']['switch0']:4d} steps"
                      f"  [{time.perf_counter() - t0:6.0f}s]", flush=True)
        suffix = scenario + ("_SMOKE" if SMOKE else "")
        flat = {}
        for i, (_, _, r) in enumerate(rows):
            for arm in ("scratch", "joint"):
                for k, v in r[arm].items():
                    flat[f"{arm}_{k}_{i}"] = v
            for k in ("joint_acc", "joint_t1", "joint_t2", "joint_iters"):
                flat[f"{k}_{i}"] = r[k]
        np.savez_compressed(array_path(__file__, suffix),
                            methods=np.array([m for m, _, _ in rows]),
                            seeds=np.array([s for _, s, _ in rows]), **flat)
        print(f"saved {array_path(__file__, suffix)}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s")
