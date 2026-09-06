"""Does the forgetting result depend on the hidden nonlinearity, or only on its magnitude?

800-series TRAINING RUN. Arrays only; the figure is 903.

WHY THIS EXISTS. tanh is the one setup choice in this project with NO justification behind it.
It was inherited, and Song & Bogacz use sigmoid, so an examiner can reasonably ask whether the
whole comparison is an artefact of a nonlinearity nobody argued for. Every other setup decision
in the Methods table is defended by a measurement; this closes the last hole.

THE ARGUMENT IS INSENSITIVITY, NOT OPTIMALITY. The claim is not that tanh is best. It is that
PC - backprop keeps its sign and rough size under all three activations, so the choice changes
the magnitude and not the finding -- which is exactly the argument 120 makes for the output
specification. If the sign flips under sigmoid, that is a much bigger result than a
justification and it belongs in the Results, not the Methods.

THREE ACTIVATIONS x TWO SCENARIOS x TWO RULES, paired on seed. relu is included even though the
project has never used it, because it is the qualitatively different case: unbounded and
one-sided, where tanh and sigmoid are both saturating. If the result survives relu it is not
about saturation.

DEVIATION FROM config_800.yaml: `act` is swept; every other field is as written. Note the
learning rates were grid-searched under tanh, so a per-activation lr is NOT re-searched here --
that would confound the activation with its calibration. Any accuracy differences between
activations are therefore expected and are not the measurement; the PAIRED PC - backprop
difference within each activation is.
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

ACTIVATIONS = ["tanh", "sigmoid", "relu"]
METHODS = ["backprop", "pc"]
SCENARIOS = CFG["data"]["scenarios"]
SEEDS = CFG["training"]["seeds"]
SEED_START = CFG["training"]["seed_start"]

SMOKE = "--smoke" in sys.argv
if SMOKE:
    SEEDS = 2

BASE = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    hidden=CFG["architecture"]["hidden"], n_layers=CFG["architecture"]["n_layers"],
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


def run_one(proto, method, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    kw = dict(PC_KW) if method == "pc" else {}
    train_step, predict = build(proto, method, seed, handle={}, **kw)
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
    return (float(curve[-1, 0]), float(curve[-1, 1]),
            float(cx) if cx is not None else float("nan"),
            switch0, bool(out["reached"][0] and out["reached"][1]))


if __name__ == "__main__":
    t0 = time.perf_counter()
    rows = []
    for scenario in SCENARIOS:
        for act in ACTIVATIONS:
            proto = replace(BASE, scenario=scenario, act=act)
            data = load(proto)
            for method in METHODS:
                cell = []
                for seed in range(SEED_START, SEED_START + SEEDS):
                    cell.append((scenario, act, method, seed) + run_one(proto, method, seed, data))
                rows.extend(cell)
                cx = [c[6] for c in cell]
                print(f"  {scenario:10s} {act:8s} {method:9s}  final t1 "
                      f"{np.mean([c[4] for c in cell]):5.1f}  t2 {np.mean([c[5] for c in cell]):5.1f}"
                      f"  crossover {np.nanmean(cx):5.1f} (defined {int(np.isfinite(cx).sum())}"
                      f"/{SEEDS})  [{time.perf_counter() - t0:6.0f}s]", flush=True)
    suffix = "SMOKE" if SMOKE else ""
    # columns: scenario, act, method, seed, final_t1, final_t2, crossover, switch0, reached
    np.savez_compressed(array_path(__file__, suffix), data=np.array(rows, dtype=object))
    print(f"saved {array_path(__file__, suffix)}")

    # paired PC - backprop per (scenario, activation) -- the measurement this run exists for
    print("\n  paired pc - backprop crossover:")
    for scenario in SCENARIOS:
        for act in ACTIVATIONS:
            bp = {r[3]: r[6] for r in rows if r[0] == scenario and r[1] == act and r[2] == "backprop"}
            pc = {r[3]: r[6] for r in rows if r[0] == scenario and r[1] == act and r[2] == "pc"}
            d = np.array([pc[s] - bp[s] for s in sorted(set(bp) & set(pc))], dtype=float)
            d = d[np.isfinite(d)]
            sem = d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else float("nan")
            print(f"    {scenario:10s} {act:8s}  {d.mean():+6.2f} +- {sem:.2f}  (n={d.size})")
    print(f"\ndone in {time.perf_counter() - t0:.0f}s")
