"""Under sigmoid -- Song & Bogacz's own activation, not tanh -- does PC's mechanism (target
alignment, settling displacement) actually track its Domain-IL advantage where 803 found it
did not under tanh? And does their own metric (mean test error) tell a different story from ours?

800-series TRAINING RUN. Arrays only; figure not yet written.

WHY THIS EXISTS, NOT A BIGGER ONE. 802 already found PC - backprop crossover flips from
-0.53 +- 0.17 (tanh) to +0.74 +- 0.19 (sigmoid) in Domain-IL -- S&B's own scenario -- and
nowhere else. That is a correlation between activation and outcome; it says nothing about WHY.
803 ran the full mechanism toolkit (target alignment, displacement, a refit linear probe) but
only ever under tanh, where the outcome is null-to-negative and the toolkit found nothing to
explain. This re-runs exactly that toolkit on the ONE condition where PC now wins, and nothing
else -- one scenario, one activation, both rules, same 10 seeds. It does not move any further
toward Song & Bogacz's actual architecture (three hidden layers, batch 500, their own stopping
rule): that is the closed, failed reproduction (knowledge_base.md S6.5, experiments/archive/
bogacz_reproduction/), and this question does not need it -- only the activation is exotic here,
everything else stays this project's own protocol.

WHAT IS RECORDED, identical to 803 plus one addition:
    per update      settling displacement D (PC only), ||dW|| per layer, both rules
    every k updates target alignment, plus the task-1-interference variant (ref=)
    checkpoints     refit linear probe accuracy per task, and its untrained floor
    NEW             mean_test_error (Song & Bogacz's own headline metric) over the same curve,
                    computed from the same run as everything else -- so if their metric tells a
                    different story from crossover/retention, it is the same seeds saying so,
                    not a separate comparison.

DEVIATION FROM config_800.yaml: act="sigmoid". Scenario fixed to domain_il -- 802 found no
sign flip in Class-IL, so there is nothing there for this toolkit to explain.
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
from src.runner import run_classil, _acc
from src.probes import alignment_probe, linear_probe_fn
from src.metrics import mean_test_error

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SCENARIO = "domain_il"
SEEDS = CFG["training"]["seeds"]
SEED_START = CFG["training"]["seed_start"]
CHECKPOINTS = CFG["mechanism"]["checkpoints"]
ALIGN_EVERY = CFG["mechanism"]["alignment_every"]
RIDGE = CFG["mechanism"]["probe_ridge"]

SMOKE = "--smoke" in sys.argv
if SMOKE:
    SEEDS, CHECKPOINTS = 2, 5

BASE = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    hidden=CFG["architecture"]["hidden"], n_layers=CFG["architecture"]["n_layers"],
    act="sigmoid",                              # the one deviation -- see docstring
    bias=CFG["architecture"]["bias"], init=CFG["architecture"]["init"],
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


def _task_acc(fn, x, y, task, classes, lmap):
    a = _acc(fn(x), y, classes, lmap)
    pos = {c: i for i, c in enumerate(classes)}
    return 100.0 * float(np.mean([a[pos[c]] for c in task]))


def _to_units(y, lmap):
    if lmap is None:
        return y
    return torch.as_tensor([lmap[int(v)] for v in y], device=y.device if torch.is_tensor(y) else None)


def run_one(proto, method, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    classes = sorted({c for t in tasks for c in t})
    fit_x, fit_y = data.stop_eval
    rep_x, rep_y = data.report_eval
    fit_units = _to_units(fit_y, lmap)

    handle = {}
    kw = dict(PC_KW) if method == "pc" else {}
    train_step, predict = build(proto, method, seed, handle=handle, **kw)
    params = handle["params"]

    p0 = linear_probe_fn(handle["features"], fit_x, fit_units, proto.out_dim, ridge=RIDGE)
    floor = [_task_acc(p0, rep_x, rep_y, t, classes, lmap) for t in tasks]

    ref_idx = [i for c in tasks[0] for i in data.class_idx[c][:40]]
    ref_x = torch.stack([data.train[i][0] for i in ref_idx])
    ref_y = _to_units(torch.tensor([data.train[i][1] for i in ref_idx]), lmap)
    step_a, align_log, ref_log = alignment_probe(
        train_step, predict, proto.arch, proto.objective, device=proto.device,
        every=ALIGN_EVERY, ref=(ref_x, ref_y))

    disp, dW, probe_log, step = [], [], [], [0]
    every_ck = [None]

    def wrapped(x, y, active=None):
        before = {k: v.detach().clone() for k, v in params.named().items() if v is not None}
        step_a(x, y, active=active)
        i = step[0]; step[0] = i + 1
        disp.append(float(handle.get("diag", {}).get("displacement", 0.0)))
        dW.append([float((params.named()[k].detach() - before[k]).norm()) for k in ("W1", "W2")])
        if every_ck[0] and i % every_ck[0] == 0:
            pk = linear_probe_fn(handle["features"], fit_x, fit_units, proto.out_dim, ridge=RIDGE)
            probe_log.append((i, *[_task_acc(pk, rep_x, rep_y, t, classes, lmap) for t in tasks],
                              *[_task_acc(predict, rep_x, rep_y, t, classes, lmap) for t in tasks]))

    every_ck[0] = max(1, (2 * proto.max_iters_per_task) // CHECKPOINTS)

    out = run_classil(
        wrapped, predict, tasks, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, device=proto.device,
        stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
        data_seed=seed, label_map=lmap)

    steps = np.asarray(out["steps"])
    curve = np.asarray(out["curves"]["argmax"], dtype=float) * 100.0
    switch0 = int(out["switches"][0])
    mte = mean_test_error(steps, curve / 100.0)   # S&B's own metric wants 0-1, everything else here is percent

    return dict(
        steps=steps, t1=curve[:, 0], t2=curve[:, 1], switch0=switch0,
        reached=np.asarray(out["reached"], dtype=bool),
        disp=np.asarray(disp, dtype=np.float32),
        dW=np.asarray(dW, dtype=np.float32),
        align=np.asarray(align_log, dtype=np.float32),
        align_ref=np.asarray(ref_log, dtype=np.float32),
        probe=np.asarray(probe_log, dtype=np.float32),
        probe_floor=np.asarray(floor, dtype=np.float32),
        final_t1=float(curve[-1, 0]), final_t2=float(curve[-1, 1]),
        mean_test_error=np.asarray(mte, dtype=np.float32),
    )


if __name__ == "__main__":
    t0 = time.perf_counter()
    proto = replace(BASE, scenario=SCENARIO)
    data = load(proto)
    rows = []
    for method in METHODS:
        for seed in range(SEED_START, SEED_START + SEEDS):
            r = run_one(proto, method, seed, data)
            rows.append((method, seed, r))
            print(f"  {SCENARIO:10s} sigmoid {method:9s} seed {seed}  "
                  f"t1 {r['final_t1']:5.1f}  t2 {r['final_t2']:5.1f}  "
                  f"switch {r['switch0']:5d}  disp {r['disp'].mean():.4f}  "
                  f"mte {r['mean_test_error'][0]:.3f}/{r['mean_test_error'][1]:.3f}  "
                  f"floor {r['probe_floor'][0]:5.1f}  "
                  f"[{time.perf_counter() - t0:6.0f}s]", flush=True)
    suffix = SCENARIO + ("_SMOKE" if SMOKE else "")
    np.savez_compressed(
        array_path(__file__, suffix),
        methods=np.array([m for m, _, _ in rows]),
        seeds=np.array([s for _, s, _ in rows]),
        **{f"{k}_{i}": v for i, (_, _, r) in enumerate(rows) for k, v in r.items()})
    print(f"saved {array_path(__file__, suffix)}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s")
