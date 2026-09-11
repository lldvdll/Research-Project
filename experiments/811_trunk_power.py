"""How much of what the network can classify is built by training the hidden layer rather than the readout?

800-series TRAINING RUN. Arrays only; the figure is 902 (panel a), Methods M2.

THE POINT OF THE CONTROL. Every mechanistic claim later in the report -- forgetting living in
the trunk versus the readout, probes recovering task-1 structure the argmax cannot reach --
presumes the hidden layer holds something worth measuring. If a random projection plus a
trained linear readout scored the same as the trained network, the trunk would be doing no
work, and "where does forgetting live" would be a question about a dead layer.

THREE ARMS, ALL READ THROUGH THE SAME PROBE SO THE COMPARISON IS FAIR:
    random_probe    frozen random trunk, linear probe fitted on its features
    trained_probe   jointly-trained trunk, FROZEN, same probe refitted on its features
    joint           the jointly-trained network read through its own output layer

    trained_probe - random_probe  is what training the trunk buys, in accuracy points.
    joint         is the reference the other two are read against; trained_probe should sit
                  close to it, because the network's own head is itself linear.

WHY A PROBE ON BOTH RATHER THAN 101's TRAINED HEAD. 101 measured the random arm with a
trained head and compared against a separately-trained network, so the readout differed
between the arms being compared. Fitting the same ridge probe on both sets of features
removes that difference, leaving the features as the only thing that varies.

DISTRIBUTIONS, NOT MEANS. Ten seeds, and every seed is saved, because the figure plots the
spread rather than three bars -- a gap that holds on every seed is a different claim from one
that holds on average.

BOTH RULES. PC and backprop build the trunk differently, and the report uses one architecture
decision for both, so the control has to cover both.

PROVENANCE
    Trains from scratch. config_800.yaml supplies every control parameter. Joint training over
    all ten classes with no task switch, the same form 810 uses; H is fixed at the chosen 32.
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
from src.probes import linear_probe_fn

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SCENARIOS = CFG["data"]["scenarios"]
SEED_START = CFG["training"]["seed_start"]
ARMS = ["random_probe", "trained_probe", "joint"]

HIDDEN = CFG["architecture"]["hidden"]
SEEDS = 10
MAX_ITERS = 25000
EVAL_EVERY = 250
PLATEAU_EVALS = 6
RIDGE = 1e-3

SMOKE = "--smoke" in sys.argv
if SMOKE:
    SEEDS, MAX_ITERS, EVAL_EVERY = 2, 400, 100
    print("--smoke: tiny budget, results are NOT meaningful\n")

PC_KW = dict(dt=CFG["predictive_coding"]["dt"],
             steps=CFG["predictive_coding"]["settle_step_cap"],
             stop_delta=CFG["predictive_coding"]["stop_delta"],
             stop_patience=CFG["predictive_coding"]["stop_patience"])


def base(scenario):
    return replace(
        PROTOCOL,
        img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
        classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
        hidden=HIDDEN, n_layers=CFG["architecture"]["n_layers"],
        act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
        init=CFG["architecture"]["init"],
        loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
        mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
        optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
        lr=dict(CFG["training"]["learning_rate"]),
        stop_threshold=None, max_iters_per_task=MAX_ITERS, eval_every=EVAL_EVERY,
        seeds=SEEDS, eval_per_class=CFG["evaluation"]["eval_per_class"],
        scenario=scenario, device=CFG["evaluation"]["device"])


def probe_accuracy(predict_fn, rep_x, rep_y, lmap):
    """Overall accuracy over all ten classes on the held-out report split."""
    with torch.no_grad():
        pred = predict_fn(rep_x)
    y = rep_y
    if lmap is not None:
        y = torch.as_tensor([lmap[int(v)] for v in rep_y], device=pred.device)
    return float((pred.cpu() == y.cpu()).float().mean()) * 100.0


def run_one(proto, method, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)              # None in Class-IL
    fit_x, fit_y = data.stop_eval
    rep_x, rep_y = data.report_eval
    if lmap is not None:
        fit_y = torch.as_tensor([lmap[int(v)] for v in fit_y])

    handle = {}
    kw = dict(PC_KW) if method == "pc" else {}
    train_step, predict = build(proto, method, seed, handle=handle, **kw)

    # Arm 1 -- untouched random trunk, probe fitted on its features.
    p_rand = linear_probe_fn(handle["features"], fit_x, fit_y, proto.out_dim, ridge=RIDGE)
    random_probe = probe_accuracy(p_rand, rep_x, rep_y, lmap)

    out = run_classil(
        train_step, predict, [tasks[0] + tasks[1]], data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=MAX_ITERS, batch=proto.batch, eval_every=EVAL_EVERY,
        device=proto.device, stop_threshold=None, data_seed=seed, label_map=lmap)
    curve = np.asarray(out["curves"]["argmax"], dtype=float)[:, 0] * 100.0
    joint = float(np.mean(curve[-PLATEAU_EVALS:]))

    # Arm 2 -- the SAME probe refitted on the now-trained features. The trunk is frozen by
    # construction here: linear_probe_fn only reads handle["features"], it never updates it.
    p_tr = linear_probe_fn(handle["features"], fit_x, fit_y, proto.out_dim, ridge=RIDGE)
    trained_probe = probe_accuracy(p_tr, rep_x, rep_y, lmap)

    return dict(random_probe=random_probe, trained_probe=trained_probe, joint=joint)


if __name__ == "__main__":
    t0 = time.perf_counter()
    res = {s: {m: {a: np.zeros(SEEDS) for a in ARMS} for m in METHODS} for s in SCENARIOS}
    for scenario in SCENARIOS:
        proto = base(scenario)
        data = load(proto)
        for method in METHODS:
            for si in range(SEEDS):
                r = run_one(proto, method, SEED_START + si, data)
                for a in ARMS:
                    res[scenario][method][a][si] = r[a]
            msg = "  ".join(f"{a} {res[scenario][method][a].mean():5.1f}" for a in ARMS)
            gain = (res[scenario][method]["trained_probe"]
                    - res[scenario][method]["random_probe"]).mean()
            print(f"  {scenario:10s} {method:9s} {msg}   trunk gain {gain:+5.1f}"
                  f"   [{time.perf_counter()-t0:6.0f}s]", flush=True)

    out = array_path(__file__) if not SMOKE else array_path(__file__, "SMOKE")
    np.savez_compressed(
        out, hidden=HIDDEN, seeds=np.arange(SEED_START, SEED_START + SEEDS),
        methods=np.array(METHODS), arms=np.array(ARMS),
        **{f"{s}_{m}_{a}": res[s][m][a] for s in SCENARIOS for m in METHODS for a in ARMS})
    print(f"saved {out}")
    print(f"done in {time.perf_counter()-t0:.0f}s")
