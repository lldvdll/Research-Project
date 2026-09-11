"""Does retention improve if task 1 keeps training past the matched-competence threshold, and
are the weights still moving or already settled by the time it is reached?

800-series TRAINING RUN. Arrays only; figures are drawn by 900 scripts not yet written (a
901/911-style trajectory plot, a weight-velocity plot, and a PCA + receptive-field plot).

MOTIVATION. 918 shows Domain-IL's weight trajectory settling steadily under repeated
alternation while Class-IL makes a couple of big moves then narrows fast. This asks the
single-switch version of the same question directly: if task 1 is still moving in weight-space
when it crosses 90%, does letting it keep moving change what survives task 2? It is the outer-
loop analogue of the question already asked of PC's OWN settling process (330-334, 345-347) --
does the stopping point need more time to reach a genuinely stable state, or is 90% already it.

DESIGN. For each (scenario, method, seed): train task 1 to threshold as normal (N steps -- this
IS the multiplier=1.0 condition). Then, WITHOUT restarting, keep training task 1 on its own data
for further fixed-length increments up to 1.5N, 2N and 4N total steps, taking a full parameter
snapshot at each of the four points. Task 1 is trained ONCE, continuously; the four conditions
are then independent BRANCHES from that one trajectory -- restore a snapshot, run task 2 exactly
as normal (matched competence) from there, and read retention/crossover for that branch. This
costs one task-1 pass per seed, not four.

WHY [0, K] / [K, 0]. Every run_classil call below passes BOTH tasks in `tasks=[t1, t2]` but
gives whichever one should not train that phase `max_iters_per_task` of 0 -- its inner loop
never executes, so it is never trained, but `curves` still reports ITS accuracy at every logged
step of the OTHER task's loop (curves is computed over the full `tasks` list regardless of
which index is currently training). This is what makes task 1's forgetting curve visible
throughout task 2's phase without a second, custom evaluation loop, and what makes task 2's
accuracy visible (flat, near its already-trained value from nothing, since it has not started)
during the task-1 overtraining phase.

WEIGHT CAPTURE, for a SEED-SUBSET only (TRACE_SEEDS, both methods). One weight_trace_probe wrap
around train_step, kept for the WHOLE per-seed run: phase A, every overtraining increment, and
all four task-2 branches (visited in sequence, each preceded by a snapshot restore -- which
shows up in the trace as a jump back to that branch's start, which is intended, not a bug).
Segment boundaries are saved explicitly (`seg_*`) so a 900 script can colour or cut the trace
without re-deriving where each phase starts. Full parameter SNAPSHOTS (not just the trace) are
saved separately at each of the four checkpoints, for the receptive-field plot -- W1 reshaped
per-column to 14x14 is literally that hidden unit's receptive field.

PROVENANCE: fresh, config_800.yaml, dt=0.4 (H=32/depth=1, unaffected by this script).
Needs the `tail_iters`-as-list change in src/runner.py (approved and verified bit-identical
against 801 for the scalar path, which every existing caller still uses).
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
from src.probes import weight_trace_probe

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SCENARIOS = CFG["data"]["scenarios"]
SEEDS = CFG["training"]["seeds"]
SEED_START = CFG["training"]["seed_start"]
MULTIPLIERS = [1.0, 1.5, 2.0, 3.0, 4.0]   # 3.0 added: PC's Domain-IL retention rose then dipped
                                          # at 4x in the first pass, finer resolution needed there
TRACE_SEEDS = 2          # first N seeds get the full weight trace + snapshots, both methods
TRACE_EVERY = 5          # thin the trace to every 5th update -- memory, not accuracy

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
PC_KW = dict(dt=CFG["predictive_coding"]["dt"],
             steps=CFG["predictive_coding"]["settle_step_cap"],
             stop_delta=CFG["predictive_coding"]["stop_delta"],
             stop_patience=CFG["predictive_coding"]["stop_patience"])
CAP = BASE.max_iters_per_task
LOG_EVERY = CFG["evaluation"]["log_every"]


def _snap(params):
    return {k: v.detach().clone() for k, v in params.named().items() if v is not None}


def _restore(params, snap):
    for k, v in params.named().items():
        if v is not None:
            v.data.copy_(snap[k])


def _run(train_step, predict, tasks, data, lmap, seed, active_idx, iters, thr):
    """One run_classil call. `active_idx` in {0, 1} says which of the two tasks trains this
    phase; the other gets max_iters_per_task=0 (never trains) and stop_threshold=None, but its
    accuracy is still logged throughout -- see the module docstring."""
    iters_per = [0, 0]
    thr_per = [None, None]
    iters_per[active_idx] = iters
    thr_per[active_idx] = thr
    return run_classil(
        train_step, predict, tasks, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=iters_per, stop_threshold=thr_per,
        stop_patience=BASE.stop_patience, log_every=LOG_EVERY,
        eval_every=BASE.eval_every, device=BASE.device,
        data_seed=seed, label_map=lmap)


def run_one(proto, method, seed, data, trace_wanted):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    kw = dict(PC_KW) if method == "pc" else {}
    handle = {}
    train_step, predict = build(proto, method, seed, handle=handle, **kw)
    params = handle["params"]

    wtrace, seg = None, []
    if trace_wanted:
        train_step, wtrace = weight_trace_probe(train_step, params, keys=("W1", "W2"),
                                                 every=TRACE_EVERY)

    def mark(name, n_before):
        n_after = len(wtrace["W1"]) if wtrace else 0
        seg.append((name, n_before, n_after))
        return n_after

    n = mark("start", 0) if trace_wanted else 0

    # Phase A: task 1 to threshold. This IS the multiplier=1.0 condition.
    outA = _run(train_step, predict, tasks, data, lmap, seed, 0, CAP, BASE.stop_threshold)
    N = outA["switches"][0]
    n = mark("task1_to_threshold", n) if trace_wanted else 0
    snapshots = {1.0: _snap(params)}
    curves_over = {}

    prev_total = N
    for m in MULTIPLIERS[1:]:
        target = int(round(m * N))
        extra = max(target - prev_total, 1)
        outB = _run(train_step, predict, tasks, data, lmap, seed, 0, extra, None)
        curves_over[m] = (outB["steps"] + prev_total, outB["curves"]["argmax"] * 100.0)
        snapshots[m] = _snap(params)
        n = mark(f"overtrain_to_{m}x", n) if trace_wanted else 0
        prev_total = target

    rows = []
    curves_t2 = {}
    for m in MULTIPLIERS:
        _restore(params, snapshots[m])
        outC = _run(train_step, predict, tasks, data, lmap, seed, 1, CAP, BASE.stop_threshold)
        steps2, curve2 = outC["steps"], outC["curves"]["argmax"] * 100.0
        curves_t2[m] = (steps2, curve2)
        n = mark(f"task2_from_{m}x", n) if trace_wanted else 0
        _, cx = crossover(steps2, curve2[:, 0], curve2[:, 1], after=0)
        rows.append((proto.scenario, method, seed, m, N,
                      float(curve2[-1, 0]), float(curve2[-1, 1]),
                      float(cx) if cx is not None else float("nan"),
                      bool(outC["reached"][1])))

    trace_out = None
    if trace_wanted:
        trace_out = dict(seg=seg, steps_A=outA["steps"],
                          curve_A=outA["curves"]["argmax"] * 100.0,
                          W1=np.array(wtrace["W1"]), W2=np.array(wtrace["W2"]))
        for m in MULTIPLIERS[1:]:
            s, c = curves_over[m]
            trace_out[f"steps_over_{m}"] = s
            trace_out[f"curve_over_{m}"] = c
        for m in MULTIPLIERS:
            s, c = curves_t2[m]
            trace_out[f"steps_t2_{m}"] = s
            trace_out[f"curve_t2_{m}"] = c
    return rows, trace_out


if __name__ == "__main__":
    t0 = time.perf_counter()
    for scenario in SCENARIOS:
        proto = replace(BASE, scenario=scenario)
        data = load(proto)
        all_rows = []
        traces = {}
        for method in METHODS:
            for i, seed in enumerate(range(SEED_START, SEED_START + SEEDS)):
                trace_wanted = i < TRACE_SEEDS
                rows, trace_out = run_one(proto, method, seed, data, trace_wanted)
                all_rows.extend(rows)
                if trace_out is not None:
                    traces[f"{method}_{seed}"] = trace_out
            print(f"  {scenario:10s} {method:9s}  done  [{time.perf_counter() - t0:6.0f}s]",
                  flush=True)

        save = dict(data=np.array(all_rows, dtype=object))
        for key, tr in traces.items():
            for k, v in tr.items():
                save[f"trace_{key}_{k}"] = v
        suffix = "SMOKE" if SMOKE else ""
        np.savez_compressed(array_path(__file__, f"{scenario}{('_' + suffix) if suffix else ''}"),
                             **save)
        print(f"  saved {scenario}")

        # paired pc - backprop, per multiplier, on the outcome only
        print(f"\n  {scenario} -- retention / crossover by multiplier:")
        for m in MULTIPLIERS:
            bp = {r[2]: (r[5], r[7]) for r in all_rows if r[1] == "backprop" and r[3] == m}
            pc = {r[2]: (r[5], r[7]) for r in all_rows if r[1] == "pc" and r[3] == m}
            for method, d in (("backprop", bp), ("pc", pc)):
                ret = np.array([v[0] for v in d.values()])
                cx = np.array([v[1] for v in d.values()], dtype=float)
                print(f"    {m:>4.1f}x  {method:9s}  retention {ret.mean():5.1f}"
                      f"  crossover {np.nanmean(cx):5.1f} (defined {int(np.isfinite(cx).sum())}"
                      f"/{len(cx)})")
    print(f"\ndone in {time.perf_counter() - t0:.0f}s")
