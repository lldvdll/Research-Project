"""Does prospective configuration -- the internal state change that defines PC -- predict how
much is forgotten, and does Song & Bogacz's own credited mechanism (target alignment) track it?

800-series TRAINING RUN. Generates arrays only; every figure is drawn by a 900 script.
Feeds 915 (target alignment), 916 (displacement -> dW -> retention), 923 (probe vs argmax),
and half of 924 (per-layer weight path).

WHY ONE RUN AND NOT FOUR. All four figures need the same thing: the standard two-task
sequential run, instrumented. Splitting them would train the same networks four times and, worse,
would measure alignment on one set of networks and displacement on another -- so no scatter of
one against the other would be possible. The whole point of 916 is that D, ||dW|| and retention
are measured on the SAME seed.

WHAT IS RECORDED
    per update      settling displacement D (PC only -- it is zero by construction for backprop,
                    which is exactly the contrast), and ||dW|| per layer for both rules.
    every k updates target alignment cos(target - out_before, out_after - out_before), plus the
                    SAME cosine measured on a fixed task-1 batch that is not being trained on.
                    During task 2 that second number is interference caught per update: negative
                    means each update actively pushes task 1's outputs away.
    checkpoints     a refit linear probe's accuracy per task, and a weight snapshot.

    The probe is fitted ONLINE and only its accuracy is kept. Storing raw hidden codes at 20
    checkpoints x 10 seeds x 2 rules x 2 scenarios would run to hundreds of MB for numbers that
    are recoverable from the accuracy alone.

    PROBE FLOOR. The same probe is fitted on the UNTRAINED network of the same seed, before any
    training. Measured at H=32 the probe reads 81.8% at init against 86.2% after task 1, so the
    dynamic range attributable to training the trunk is about four points. Without the floor
    recorded beside it, a post-task-2 reading near 80% would look like a surviving trained code
    when it is really just a random projection of 14x14 MNIST being close to linearly separable.
    That is the error NCM made, one level up. The floor costs nothing: no training.

DEVIATION FROM config_800.yaml: none. Backprop's displacement is logged as 0.0 rather than
skipped, so the two rules produce identically shaped arrays.
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

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SCENARIOS = CFG["data"]["scenarios"]
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


def _task_acc(fn, x, y, task, classes, lmap):
    """Accuracy PERCENT on one task's own classes, under the run's label map.

    _acc returns fractions; everything saved by this script is in percent so the 900 scripts
    never have to ask which."""
    a = _acc(fn(x), y, classes, lmap)
    pos = {c: i for i, c in enumerate(classes)}
    return 100.0 * float(np.mean([a[pos[c]] for c in task]))


def _to_units(y, lmap):
    """Class labels -> OUTPUT UNIT indices.

    The probe has to predict in the same space `predict` does, because _acc scores both the
    same way. Class-IL has lmap=None and the two coincide; Domain-IL has five shared units, so
    fitting on raw labels 0-9 would index past the end of a 5-wide target."""
    if lmap is None:
        return y
    return torch.as_tensor([lmap[int(v)] for v in y], device=y.device if torch.is_tensor(y) else None)


def run_one(proto, method, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    classes = sorted({c for t in tasks for c in t})
    fit_x, fit_y = data.stop_eval          # disjoint from report_eval: probe fit vs probe score
    rep_x, rep_y = data.report_eval
    fit_units = _to_units(fit_y, lmap)     # probe predicts UNITS, as `predict` does

    handle = {}
    kw = dict(PC_KW) if method == "pc" else {}
    train_step, predict = build(proto, method, seed, handle=handle, **kw)
    params = handle["params"]

    # probe FLOOR, before a single update
    p0 = linear_probe_fn(handle["features"], fit_x, fit_units, proto.out_dim, ridge=RIDGE)
    floor = [_task_acc(p0, rep_x, rep_y, t, classes, lmap) for t in tasks]

    # target alignment, with a fixed task-1 reference batch that is never trained on
    ref_idx = [i for c in tasks[0] for i in data.class_idx[c][:40]]
    ref_x = torch.stack([data.train[i][0] for i in ref_idx])
    # make_target indexes output UNITS, so the reference labels are mapped exactly as the
    # probe's are -- Domain-IL has five units and raw labels run 0-9.
    ref_y = _to_units(torch.tensor([data.train[i][1] for i in ref_idx]), lmap)
    step_a, align_log, ref_log = alignment_probe(
        train_step, predict, proto.arch, proto.objective, device=proto.device,
        every=ALIGN_EVERY, ref=(ref_x, ref_y))

    disp, dW, probe_log, step = [], [], [], [0]
    every_ck = [None]                      # filled once the budget is known

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
    )


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
                      f"t1 {r['final_t1']:5.1f}  t2 {r['final_t2']:5.1f}  "
                      f"switch {r['switch0']:5d}  disp {r['disp'].mean():.4f}  "
                      f"floor {r['probe_floor'][0]:5.1f}  "
                      f"[{time.perf_counter() - t0:6.0f}s]", flush=True)
        np.savez_compressed(
            array_path(__file__, scenario + ("_SMOKE" if SMOKE else "")),
            methods=np.array([m for m, _, _ in rows]),
            seeds=np.array([s for _, s, _ in rows]),
            **{f"{k}_{i}": v for i, (_, _, r) in enumerate(rows) for k, v in r.items()})
        print(f"saved {array_path(__file__, scenario + ('_SMOKE' if SMOKE else ''))}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s")
