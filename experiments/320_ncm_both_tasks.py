"""When argmax says a task's accuracy has changed, does NCM (a readout-independent probe of the
hidden code) agree -- for BOTH tasks, across the whole run, not just task-1 post-switch?

Report series 2, decade 320 (§2 of the mechanism-first narrative). Backprop only -- PC is held
for §3, where the readout-vs-representation question this section establishes gets asked OF the
rule comparison, rather than introduced alongside it here. Seeds from config_300.yaml's
seed_start (10-19), same protocol as 310.

TWO frozen prototype sets, BOTH captured at INITIALISATION -- before any training at all, task-1
included -- so NCM is defined for the entire run, not just from the switch onward:
    ncm_t1 : task-1's classes, prototypes frozen at t=0 -- "has the code drifted from its random-
             init baseline?", trackable through both task-1's own training and task-2's.
    ncm_t2 : task-2's classes, prototypes ALSO frozen at t=0 -- "how decodable are task-2's
             classes using the untrained representation, and does that change once task-2 is
             actually trained on them?" If ncm_t2 tracks argmax_t2 closely once task-2 training
             starts, its accuracy gain is mostly readout calibration on an already-adequate
             representation. If ncm_t2 stays low while argmax_t2 climbs, real representational
             reorganisation happened.

x-axis is step RELATIVE TO SWITCH (needed for a meaningful aligned mean across seeds whose switch
happens at different absolute steps), NaN-padded rather than cropped to a common window, so no
seed's data is discarded to make the mean work -- the same principle 310 applied to its own
figure.

One figure, one panel per scenario: 4 lines (task x readout), thin per-seed + bold mean.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, replace, load, build, array_path as _array_path
from src.runner import run_classil
from src.probes import class_prototypes, frozen_ncm_fn

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

METHOD = "backprop"
TASK1_COLOR = "tab:orange"
TASK2_COLOR = "tab:blue"

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, SEED_START, EVAL_EVERY, LOG_EVERY, MAX_ITERS, PROTO_PER_CLASS = 3, 10, 5, 10, 200, 20
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = CFG["training"]["seeds"]
    SEED_START = CFG["training"]["seed_start"]
    EVAL_EVERY = CFG["evaluation"]["eval_every"]
    LOG_EVERY = CFG["evaluation"]["log_every"]
    MAX_ITERS = CFG["training"]["max_iters_per_task"]
    PROTO_PER_CLASS = 100

SCENARIOS = CFG["data"]["scenarios"]

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
    max_iters_per_task=MAX_ITERS, seeds=SEEDS,
    eval_per_class=CFG["evaluation"]["eval_per_class"], eval_every=EVAL_EVERY,
    device=CFG["evaluation"]["device"],
)


def _tag(scenario):
    return scenario + ("_SMOKE" if SMOKE else "")


def array_path(scenario):
    return _array_path(__file__, _tag(scenario))


def run_one(proto, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    handle = {}
    train_step, predict = build(proto, METHOD, seed, handle=handle)

    # Prototypes for BOTH tasks captured at INITIALISATION -- before any training at all -- so
    # NCM is defined for the whole run, task-1 training included, not just from the switch
    # onward. This asks "has the code drifted from its random-init baseline", a question with
    # no gap in its domain, unlike "drifted from where task-1 training left it" (131's framing),
    # which is undefined before task-1 ends.
    raw_t1 = class_prototypes(handle["features"], data.train, data.class_idx, tasks[0],
                              per_class=PROTO_PER_CLASS, device=proto.device, seed=seed)
    raw_t2 = class_prototypes(handle["features"], data.train, data.class_idx, tasks[1],
                              per_class=PROTO_PER_CLASS, device=proto.device, seed=seed)
    if lmap is None:
        protos_t1, protos_t2 = raw_t1, raw_t2
    else:
        protos_t1 = {lmap[c]: v for c, v in raw_t1.items()}
        protos_t2 = {lmap[c]: v for c, v in raw_t2.items()}

    readouts = {
        "argmax": predict,
        "ncm_t1": frozen_ncm_fn(handle["features"], protos_t1),
        "ncm_t2": frozen_ncm_fn(handle["features"], protos_t2),
    }
    out = run_classil(train_step, predict, tasks, data.train, data.class_idx,
                      report_eval=data.report_eval, stop_eval=data.stop_eval, readouts=readouts,
                      max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
                      eval_every=proto.eval_every, log_every=LOG_EVERY, device=proto.device,
                      stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
                      data_seed=seed, label_map=lmap)
    return out


def run_scenario(scenario):
    proto = replace(BASE_PROTOCOL, scenario=scenario)
    data = load(proto)
    seed_runs = []
    for seed in range(SEED_START, SEED_START + SEEDS):
        out = run_one(proto, seed, data)
        if not all(out["reached"]):
            print(f"  WARNING: {scenario} seed {seed}: hit the cap before reaching threshold")
        seed_runs.append(out)
    return seed_runs


def align_relative(seed_runs, key, col):
    """NaN-padded stack of `curves[key][:, col]`, x-axis = step relative to the switch. No
    common-window cropping -- every seed keeps its full extent on both sides of the switch."""
    # Grid spacing must match LOG_EVERY (how often curves are actually logged), not EVAL_EVERY
    # (the stop-check cadence) -- the two are equal in the real config, so this only bit in
    # smoke mode, but it was a latent bug either way.
    rels = [np.asarray(o["steps"]) - o["switches"][0] for o in seed_runs]
    lo = int(min(r.min() for r in rels))
    hi = int(max(r.max() for r in rels))
    grid = np.arange(lo, hi + LOG_EVERY, LOG_EVERY)
    out = np.full((len(seed_runs), len(grid)), np.nan)
    for i, (o, rel) in enumerate(zip(seed_runs, rels)):
        curve = o["curves"][key][:, col] * 100
        for j, r in enumerate(rel):
            k = int(round((r - lo) / LOG_EVERY))
            if 0 <= k < len(grid):
                out[i, k] = curve[j]
    return grid, out


if REPLOT and all(Path(array_path(s)).exists() for s in SCENARIOS):
    results = {}
    for s in SCENARIOS:
        z = np.load(array_path(s), allow_pickle=True)
        results[s] = dict(grid=z["grid"], t1_argmax=z["t1_argmax"], t2_argmax=z["t2_argmax"],
                          t1_ncm=z["t1_ncm"], t2_ncm=z["t2_ncm"])
    print("--replot: redrawing from saved arrays, no training\n")
else:
    results = {}
    t0 = time.perf_counter()
    for scenario in SCENARIOS:
        seed_runs = run_scenario(scenario)
        grid, t1_argmax = align_relative(seed_runs, "argmax", 0)
        _, t2_argmax = align_relative(seed_runs, "argmax", 1)
        _, t1_ncm = align_relative(seed_runs, "ncm_t1", 0)
        _, t2_ncm = align_relative(seed_runs, "ncm_t2", 1)
        results[scenario] = dict(grid=grid, t1_argmax=t1_argmax, t2_argmax=t2_argmax,
                                 t1_ncm=t1_ncm, t2_ncm=t2_ncm)
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6.5 * len(SCENARIOS), 5))
for ax, scenario in zip(axes, SCENARIOS):
    r = results[scenario]
    grid = r["grid"]
    for key, color, ls, label in [
        ("t1_argmax", TASK1_COLOR, "-", "task-1 argmax"),
        ("t1_ncm", TASK1_COLOR, "--", "task-1 NCM"),
        ("t2_argmax", TASK2_COLOR, "-", "task-2 argmax"),
        ("t2_ncm", TASK2_COLOR, "--", "task-2 NCM"),
    ]:
        A = r[key]
        for row in range(A.shape[0]):
            ax.plot(grid, A[row], color=color, ls=ls, lw=0.6, alpha=0.25)
        ax.plot(grid, np.nanmean(A, axis=0), color=color, ls=ls, lw=2.2, label=label)
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("step, relative to task switch")
    ax.set_ylabel("accuracy (%)")
    ax.text(0.5, 1.03, scenario.replace("_", "-"), fontsize=9, ha="center", transform=ax.transAxes)
    ax.grid(alpha=0.2)
axes[0].legend(fontsize=8, loc="center left")
fig.tight_layout()

out = str(Path(__file__).parent / ("320_ncm_both_tasks" + ("_SMOKE" if SMOKE else "") + ".png"))
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")

for scenario in SCENARIOS:
    r = results[scenario]
    np.savez(array_path(scenario), grid=r["grid"], t1_argmax=r["t1_argmax"],
            t2_argmax=r["t2_argmax"], t1_ncm=r["t1_ncm"], t2_ncm=r["t2_ncm"])
    print(f"saved {array_path(scenario)}")
