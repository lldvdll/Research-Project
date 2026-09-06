"""Under repeated alternation, does the network converge on a joint solution or ping-pong
between two incompatible ones -- and does the answer differ by scenario?

800-series TRAINING RUN. Arrays only; figures are drawn by 917 (accuracy-space geometry),
918 (weight-space PCA) and half of 924 (per-layer path).

THE QUESTION, GEOMETRICALLY. Plot task-1 accuracy against task-2 accuracy and remove time. A
network converging on a joint solution traces a SPIRAL contracting toward the upper-right
corner; one ping-ponging between two solutions traces a CLOSED LOOP that never contracts. Those
are qualitatively different behaviours, not different amounts of one, which is why this is the
strongest available justification for treating Class-IL and Domain-IL as separate
investigations.

WHAT 68 ALREADY SHOWED, AND WHY THIS IS NOT THAT RUN
    68 produced exactly this figure and Domain-IL does spiral inward. But it ran DOMAIN-IL ONLY,
    at FIXED BUDGET, five seeds, including replay, under the pre-300 protocol. The Class-IL arm
    -- the half that carries the comparison -- has never existed. This run adds it, moves to
    matched competence, and stores W2 as well as W1 (68 stored only W1, because it passed one
    key to weight_trace_probe, not because the probe could not do both).

MATCHED COMPETENCE PER BLOCK IS A DELIBERATE CHANGE, AND IT HAS A CONSEQUENCE. Each block now
trains until its own task reaches the threshold, so BLOCK LENGTH BECOMES A DEPENDENT VARIABLE
rather than a constant. That is protocol-consistent -- every other run in the series reads at
matched competence -- and the block lengths are themselves a result: whether relearning
accelerates across repeats is the "gradual relearn" shape from the four-way shape question.
Saved as `switches` so 917 can report it.

REPEATS. Five passes of (task 1, task 2), i.e. ten blocks. Extend without re-running the
existing seeds with --add-repeats=N: the schedule is regenerated and appended, and only the
new seeds are trained.

DEVIATION FROM config_800.yaml: `tasks` is overridden to repeat the two blocks, which
protocol.run documents as a stated deviation. Nothing else changes.
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
from src.probes import weight_trace_probe

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SCENARIOS = CFG["data"]["scenarios"]
SEEDS = CFG["training"]["seeds"]
SEED_START = CFG["training"]["seed_start"]

REPEATS = 5                 # passes of (task 1, task 2) -> 10 blocks
TRACE_EVERY = 10            # weight snapshots. 68's note: memory, not time -- W1 at 196x32 is
                            # 25 kB per snapshot, so every 10th update over ~7000 updates is
                            # ~18 MB per layer per traced seed.
TRACE_SEEDS = 3             # seeds whose full weight trajectory is kept, for 918's PCA. The
                            # statistics come from every seed; the trajectory is an illustration.

for a in sys.argv:
    if a.startswith("--add-repeats="):
        REPEATS = int(a.split("=", 1)[1])

SMOKE = "--smoke" in sys.argv
if SMOKE:
    SEEDS, REPEATS, TRACE_SEEDS = 2, 2, 1

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
    max_iters_per_task=(150 if SMOKE else CFG["training"]["max_iters_per_task"]),
    seeds=SEEDS, eval_per_class=CFG["evaluation"]["eval_per_class"],
    eval_every=CFG["evaluation"]["eval_every"], device=CFG["evaluation"]["device"],
)
PC_KW = dict(dt=CFG["predictive_coding"]["dt"],
             steps=CFG["predictive_coding"]["settle_step_cap"],
             stop_delta=CFG["predictive_coding"]["stop_delta"],
             stop_patience=CFG["predictive_coding"]["stop_patience"])


def run_one(proto, method, seed, data, keep_trace):
    t1, t2 = proto.tasks(seed)
    schedule = [t1, t2] * REPEATS
    # label_map is built from the ORIGINAL two tasks, not the repeated schedule: repeating a
    # block must not renumber its units, or task 1 would land on different outputs each pass.
    lmap = proto.label_map([t1, t2])
    classes = sorted(set(t1) | set(t2))
    pos = {c: i for i, c in enumerate(classes)}

    handle = {}
    kw = dict(PC_KW) if method == "pc" else {}
    train_step, predict = build(proto, method, seed, handle=handle, **kw)

    trace = None
    if keep_trace:
        train_step, trace = weight_trace_probe(train_step, handle["params"],
                                               keys=("W1", "W2"), every=TRACE_EVERY)

    out = run_classil(
        train_step, predict, schedule, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, log_every=CFG["evaluation"]["log_every"],
        device=proto.device, stop_threshold=proto.stop_threshold,
        stop_patience=proto.stop_patience, data_seed=seed, label_map=lmap)

    # curves has one column per BLOCK; columns for the same class set are duplicates, so collapse
    # back to the two real tasks.
    curve = np.asarray(out["curves"]["argmax"], dtype=float) * 100.0
    r = dict(steps=np.asarray(out["steps"]),
             t1=curve[:, 0], t2=curve[:, 1],
             switches=np.asarray(out["switches"], dtype=int),
             reached=np.asarray(out["reached"], dtype=bool))
    if trace is not None:
        for k, v in trace.items():
            r[f"trace_{k}"] = np.asarray(v, dtype=np.float32)
    return r


if __name__ == "__main__":
    t0 = time.perf_counter()
    for scenario in SCENARIOS:
        proto = replace(BASE, scenario=scenario)
        data = load(proto)
        rows = []
        for method in METHODS:
            for seed in range(SEED_START, SEED_START + SEEDS):
                keep = (seed - SEED_START) < TRACE_SEEDS
                r = run_one(proto, method, seed, data, keep)
                rows.append((method, seed, r))
                blocks = np.diff(np.concatenate([[0], r["switches"]]))
                print(f"  {scenario:10s} {method:9s} seed {seed}  "
                      f"blocks {blocks.tolist()}  reached {int(r['reached'].sum())}/"
                      f"{len(r['reached'])}  end t1 {r['t1'][-1]:5.1f} t2 {r['t2'][-1]:5.1f}"
                      f"  [{time.perf_counter() - t0:6.0f}s]", flush=True)
        suffix = scenario + ("_SMOKE" if SMOKE else "")
        np.savez_compressed(
            array_path(__file__, suffix),
            methods=np.array([m for m, _, _ in rows]),
            seeds=np.array([s for _, s, _ in rows]),
            repeats=REPEATS,
            **{f"{k}_{i}": v for i, (_, _, r) in enumerate(rows) for k, v in r.items()})
        print(f"saved {array_path(__file__, suffix)}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s")
