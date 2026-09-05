"""Does the {5,8}-in-task-1 retention pattern (312) hold at 50 more seeds, or was it a
small-sample coincidence -- and if it holds, is there a data-pipeline bug behind it?

Report series 2, decade 313. Class-IL only, backprop only, seeds 20-69 (50 new seeds,
continuing past 300's 0-9 and 310's 10-19 -- none reused). Same protocol as 300/310
(config_300.yaml). Saves the same array shape (steps, t1, t2, switch0, crossovers, finals) as
300/310 for direct reuse by follow-up analysis -- no figure here, this script only collects
data; the pattern check happens after, against the combined 0-69 pool.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np

from src.protocol import PROTOCOL, replace, load, build, array_path as _array_path
from src.runner import run_classil, _acc
from src.metrics import crossover

SMOKE = "--smoke" in sys.argv

SEED_START = 20
N_SEEDS = 50

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    N_SEEDS, EVAL_EVERY, LOG_EVERY, MAX_ITERS = 3, 5, 10, 200
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    EVAL_EVERY = CFG["evaluation"]["eval_every"]
    LOG_EVERY = CFG["evaluation"]["log_every"]
    MAX_ITERS = CFG["training"]["max_iters_per_task"]

METHOD = "backprop"
BASE_PROTOCOL = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    hidden=CFG["architecture"]["hidden"], n_layers=CFG["architecture"]["n_layers"],
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    lr={METHOD: CFG["training"]["learning_rate"][METHOD]},
    stop_threshold=CFG["training"]["stop_threshold"],
    stop_patience=CFG["training"]["stop_patience"],
    max_iters_per_task=MAX_ITERS, seeds=N_SEEDS,
    eval_per_class=CFG["evaluation"]["eval_per_class"], eval_every=EVAL_EVERY,
    device=CFG["evaluation"]["device"],
    scenario="class_il",
)


def _tag():
    return "SMOKE" if SMOKE else ""


def array_path():
    return _array_path(__file__, _tag())


def run_one(proto, seed, data):
    tasks = proto.tasks(seed)
    classes = sorted({c for t in tasks for c in t})
    pos = {c: i for i, c in enumerate(classes)}

    train_step, predict = build(proto, METHOD, seed)

    report_x, report_y = data.report_eval
    a0 = _acc(predict(report_x), report_y, classes, None)
    init_row = [float(np.mean([a0[pos[c]] for c in t])) for t in tasks]

    out = run_classil(
        train_step, predict, tasks, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, log_every=LOG_EVERY, device=proto.device,
        stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
        data_seed=seed,
    )
    steps = np.concatenate([[0], out["steps"]])
    curve = np.concatenate([[init_row], out["curves"]["argmax"]], axis=0)
    return steps, curve[:, 0], curve[:, 1], out["switches"][0], out["reached"]


data = load(BASE_PROTOCOL)
seed_runs = []
t0 = time.perf_counter()
for seed in range(SEED_START, SEED_START + N_SEEDS):
    steps, t1, t2, switch0, reached = run_one(BASE_PROTOCOL, seed, data)
    if not all(reached):
        print(f"  WARNING: seed {seed}: hit the cap before reaching threshold")
    seed_runs.append((steps, t1, t2, switch0))
    if (seed - SEED_START + 1) % 10 == 0:
        print(f"  {seed - SEED_START + 1}/{N_SEEDS} done   [{time.perf_counter() - t0:5.0f}s]")

crossovers, finals = [], []
for steps, t1, t2, switch0 in seed_runs:
    _, cx_height = crossover(steps, t1 * 100, t2 * 100, after=switch0)
    crossovers.append(cx_height)
    finals.append(t1[-1] * 100)

np.savez(array_path(),
        steps=np.array([s[0] for s in seed_runs], dtype=object),
        t1=np.array([s[1] for s in seed_runs], dtype=object),
        t2=np.array([s[2] for s in seed_runs], dtype=object),
        switch0=np.array([s[3] for s in seed_runs]),
        crossovers=np.asarray(crossovers), finals=np.asarray(finals),
        seed_start=SEED_START)
print(f"saved {array_path()}")
