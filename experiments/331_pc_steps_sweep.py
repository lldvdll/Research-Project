"""Holding dt at 330's chosen control (0.4 -- minimum settle-steps, flat on every accuracy/
forgetting metric, comfortable margin before the 0.5->0.6 instability edge), does truncating
settling to a FIXED, small number of steps cost accuracy or forgetting relative to fuller
settling -- and does "few settle steps approximates backprop" hold under this project's own BP
control?

Report series 2, decade 331 (narrative_plan.md's PC sweep, §2 -- validates the adaptive-stopping
design used everywhere else in this series). Structure mirrors 330: same two conditions (joint,
sequential), same crossover tracking, same label-remap handling for Domain-IL. The one
fundamental difference from 330: STEPS IS FIXED PER RUN, not adaptive. 330 asked "how many steps
does settling need"; this script asks "what happens if you DELIBERATELY give it fewer than that,
by a known, controlled amount" -- so every call to pc_settle here uses steps=N with no
stop_delta, running exactly N iterations every time, intentionally, for every value of N tested.

STEPS_GRID = [1, 2, 3, 5, 10, 20, 50] -- fixed values, not a cap. Starts at 1, not 0: steps=0 is
NOT backprop (predictive_coding.py:38-42 -- every e_l is zero by construction, W1 does not move
at all), so steps=1 is the closest PC gets to a "backprop-like" single relaxation step, not a
claim that it equals it. Tops out at 50: scripts 50/51/59 established this is already a safety
margin over a measured worst case of 18 (at this H=32) for FULL settling, so nothing is gained by
testing higher.

BP INCLUDED HERE, PC ALONE IN 330/334 -- explicit instruction: those characterise PC's own
settle dynamics in isolation; 331 is where the literature's "low settle steps approximates
backprop" claim actually gets a control to compare against. BP has no dt/steps, so it runs once
per (scenario, seed) and is plotted as a flat reference across the steps axis, not swept.

Companion diagnostic: for each seed, ONE long relaxation (steps=300, trace=True) at dt=0.4 from
fresh init on a representative batch -- read off at each value in STEPS_GRID, this gives the
residual displacement a truncated run would have been left with, cheaply (no retraining), same
method as 330/333/334's settle diagnostics.

Figure: one PNG, 2 rows (class_il, domain_il) x 2 columns (steps vs residual displacement at
dt=0.4 | steps vs accuracy, four PC series solid + four BP reference lines dashed, same colours).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, replace, load, build, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import run_joint, run_classil, _loader
from src.predictive_coding import pc_settle
from src.model import make_target
from src.metrics import crossover as crossover_fn

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

PC_METHOD = "pc"
BP_METHOD = "backprop"
DT = 0.4   # fixed -- 330's chosen control. Settle-steps minimum shifted twice as resolution
          # around the trough improved: 0.2 -> 0.3 -> 0.4 (class-il's true minimum; domain-il
          # has 0.3 and 0.4 essentially tied). Accuracy/retention/crossover flat throughout, so
          # the choice is purely about settle-step efficiency, not learning/forgetting outcomes.

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, SEED_START = 2, 10
    STEPS_GRID = [1, 10]
    TRAIN_MAX_ITERS = 100
    EVAL_EVERY = 5
    DIAG_STEP_CAP = 60
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = CFG["training"]["seeds"] // 2   # 5 -- matches 330's convention for this pass
    SEED_START = CFG["training"]["seed_start"]
    STEPS_GRID = [1, 2, 3, 5, 10, 20, 50]
    TRAIN_MAX_ITERS = 1500   # matches 330's trimmed budget, same rationale
    EVAL_EVERY = CFG["evaluation"]["eval_every"]
    DIAG_STEP_CAP = 300      # 333's validated cap for the residual-displacement diagnostic

SCENARIOS = CFG["data"]["scenarios"]
STOP_PATIENCE_JOINT = 5
MIN_DELTA_JOINT = 1e-3

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
    lr={PC_METHOD: CFG["training"]["learning_rate"]["pc"],
        BP_METHOD: CFG["training"]["learning_rate"]["backprop"]},
    stop_threshold=CFG["training"]["stop_threshold"],
    stop_patience=CFG["training"]["stop_patience"],
    max_iters_per_task=TRAIN_MAX_ITERS, seeds=SEEDS,
    eval_per_class=CFG["evaluation"]["eval_per_class"], eval_every=EVAL_EVERY,
    device=CFG["evaluation"]["device"],
)


def _tag(scenario):
    return scenario + ("_SMOKE" if SMOKE else "")


def figure_path():
    return _figure_path(__file__, "SMOKE" if SMOKE else "")


def array_path(scenario, kind):
    return _array_path(__file__, f"{_tag(scenario)}_{kind}")


def _make_joint_step(train_step, lmap):
    """run_joint has no label_map parameter -- same remap 330 uses. `active` names output
    units, not classes, so it must be remapped too (see 330's docstring for the double-remap
    bug this pattern guards against on the run_classil side, which does NOT need this wrapper)."""
    def step(x, y, active=None):
        if lmap is not None:
            y = torch.tensor([lmap[int(v)] for v in y.tolist()], device=y.device)
            active = None if active is None else sorted({lmap[int(c)] for c in active})
        train_step(x, y, active=active)
    return step


def _filter_eval(eval_xy, classes, lmap):
    x, y = eval_xy
    mask = torch.isin(y, torch.as_tensor(classes, device=y.device))
    x, y = x[mask], y[mask]
    if lmap is not None:
        y = torch.tensor([lmap[int(v)] for v in y.tolist()], device=y.device)
    return x, y


def run_joint_condition(proto, method, seed, data, classes, lmap, **overrides):
    train_step, predict = build(proto, method, seed, **overrides)
    ex, ey = _filter_eval(data.report_eval, classes, lmap)
    ts = train_step if lmap is None else _make_joint_step(train_step, lmap)
    _, accs = run_joint(ts, predict, classes, data.train, data.class_idx, ex, ey,
                        max_iters=TRAIN_MAX_ITERS, batch=proto.batch, eval_every=proto.eval_every,
                        device=proto.device, stop_patience=STOP_PATIENCE_JOINT,
                        min_delta=MIN_DELTA_JOINT, data_seed=seed)
    return float(accs[-1]) * 100 if len(accs) else float("nan")


def run_sequential_condition(proto, method, seed, data, **overrides):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    train_step, predict = build(proto, method, seed, **overrides)   # run_classil remaps itself
    out = run_classil(train_step, predict, tasks, data.train, data.class_idx,
                      report_eval=data.report_eval, stop_eval=data.stop_eval,
                      max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
                      eval_every=proto.eval_every, device=proto.device,
                      stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
                      data_seed=seed, label_map=lmap)
    curve = out["curves"]["argmax"]
    _, cx_height = crossover_fn(out["steps"], curve[:, 0] * 100, curve[:, 1] * 100,
                                after=out["switches"][0])
    return float(curve[-1, 0]) * 100, float(curve[-1, 1]) * 100, cx_height, all(out["reached"])


def settle_trace(proto, seed, data):
    """One long relaxation at dt=0.4, fresh init, one batch spanning all classes, full trace --
    read off at each STEPS_GRID value to get residual displacement at that truncation, without
    retraining. Same method as 330/333/334."""
    handle = {}
    build(proto, PC_METHOD, seed, handle=handle, dt=DT, steps=1)
    p, arch, obj = handle["params"], handle["arch"], handle["obj"]
    classes = list(range(proto.n_classes))
    loader = _loader(data.train, data.class_idx, classes, proto.batch, seed=seed)
    x, y = next(iter(loader))
    x, y = x.to(proto.device), y.to(proto.device)
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    if lmap is not None:
        y = torch.tensor([lmap[int(v)] for v in y.tolist()], device=proto.device)
    target = make_target(y, arch, obj, device=proto.device)
    _, _, disp = pc_settle(x, p, arch, obj, target, dt=DT, steps=DIAG_STEP_CAP, trace=True)
    return np.asarray(disp)


# ---------------------------------------------------------------- run or reload
if REPLOT and all(Path(array_path(s, k)).exists() for s in SCENARIOS for k in ("pc", "bp", "trace")):
    pc_rows, bp_rows, trace_rows = {}, {}, {}
    for s in SCENARIOS:
        pc_rows[s] = list(np.load(array_path(s, "pc"), allow_pickle=True)["data"])
        bp_rows[s] = list(np.load(array_path(s, "bp"), allow_pickle=True)["data"])
        trace_rows[s] = list(np.load(array_path(s, "trace"), allow_pickle=True)["data"])
    print("--replot: redrawing from saved arrays, no training\n")
else:
    t0 = time.perf_counter()
    pc_rows, bp_rows, trace_rows = {s: [] for s in SCENARIOS}, {s: [] for s in SCENARIOS}, {s: [] for s in SCENARIOS}
    for scenario in SCENARIOS:
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        data = load(proto)
        classes_all = list(range(proto.n_classes))

        for seed in range(SEED_START, SEED_START + SEEDS):
            trace_rows[scenario].append((seed, settle_trace(proto, seed, data)))

            tasks = proto.tasks(seed)
            lmap = proto.label_map(tasks)
            joint_bp = run_joint_condition(proto, BP_METHOD, seed, data, classes_all, lmap)
            seq_t1_bp, seq_t2_bp, cx_bp, reached_bp = run_sequential_condition(
                proto, BP_METHOD, seed, data)
            if not reached_bp:
                print(f"  WARNING: {scenario} seed={seed}: BP sequential run hit the cap")
            bp_rows[scenario].append((seed, joint_bp, seq_t1_bp, seq_t2_bp, cx_bp))

            for steps in STEPS_GRID:
                joint_acc = run_joint_condition(proto, PC_METHOD, seed, data, classes_all, lmap,
                                                dt=DT, steps=steps)
                seq_t1, seq_t2, cx_height, reached_ok = run_sequential_condition(
                    proto, PC_METHOD, seed, data, dt=DT, steps=steps)
                if not reached_ok:
                    print(f"  WARNING: {scenario} steps={steps} seed={seed}: "
                         f"PC sequential run hit the cap")
                pc_rows[scenario].append((steps, seed, joint_acc, seq_t1, seq_t2, cx_height))
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]")

    for scenario in SCENARIOS:
        np.savez(array_path(scenario, "pc"), data=np.array(pc_rows[scenario], dtype=object))
        np.savez(array_path(scenario, "bp"), data=np.array(bp_rows[scenario], dtype=object))
        np.savez(array_path(scenario, "trace"), data=np.array(trace_rows[scenario], dtype=object))
    print("saved arrays")


# ---------------------------------------------------------------- aggregate
def _agg_pc(rows, col, grid):
    means, sems = [], []
    for v in grid:
        vals = np.asarray([r[col] for r in rows if r[0] == v], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            means.append(np.nan); sems.append(np.nan)
        elif vals.size == 1:
            means.append(vals[0]); sems.append(0.0)
        else:
            means.append(vals.mean()); sems.append(vals.std(ddof=1) / np.sqrt(vals.size))
    return np.array(means), np.array(sems)


def _agg_bp(rows, col):
    vals = np.asarray([r[col] for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), 0.0
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(vals.size))


def _set_step_ticks(ax, grid):
    ax.set_xscale("log")
    ax.set_xticks(grid)
    ax.set_xticklabels([str(s) for s in grid], fontsize=7)
    ax.minorticks_off()


PC_SERIES = [
    ("joint", 2, "tab:purple", "o", "PC joint (all 10)"),
    ("seq_t2", 4, "tab:green", "s", "PC task-2 after task-1 (final)"),
    ("seq_t1", 3, "tab:red", "s", "PC task-1 after task-2 (retention)"),
    ("crossover", 5, "tab:blue", "d", "PC crossover height"),
]
BP_COL = {"joint": 1, "seq_t1": 2, "seq_t2": 3, "crossover": 4}

fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(11, 4.5 * len(SCENARIOS)), squeeze=False)

for row, scenario in enumerate(SCENARIOS):
    ax_disp, ax_acc = axes[row]

    traces = trace_rows[scenario]
    residual = np.full((len(traces), len(STEPS_GRID)), np.nan)
    for i, (seed, disp) in enumerate(traces):
        for j, s in enumerate(STEPS_GRID):
            if s - 1 < len(disp):
                residual[i, j] = disp[s - 1]
    m = np.nanmean(residual, axis=0)
    se = np.nanstd(residual, axis=0, ddof=1) / np.sqrt(residual.shape[0])
    ax_disp.errorbar(STEPS_GRID, m, yerr=se, marker="o", color="tab:gray", capsize=3)
    _set_step_ticks(ax_disp, STEPS_GRID)
    ax_disp.set_xlabel("settle steps (fixed)")
    ax_disp.set_ylabel("displacement at truncation")
    ax_disp.set_title(f"{scenario.replace('_', '-')}: residual displacement (dt={DT})", fontsize=9)
    ax_disp.grid(alpha=0.2, which="both")

    pc = pc_rows[scenario]
    bp = bp_rows[scenario]
    for key, col, color, marker, label in PC_SERIES:
        m, se = _agg_pc(pc, col, STEPS_GRID)
        ax_acc.errorbar(STEPS_GRID, m, yerr=se, marker=marker, color=color, label=label, capsize=3)
        bp_mean, bp_sem = _agg_bp(bp, BP_COL[key])
        ax_acc.axhline(bp_mean, color=color, ls="--", lw=1.0, alpha=0.7)
    _set_step_ticks(ax_acc, STEPS_GRID)
    ax_acc.set_ylim(0, 100)
    ax_acc.set_xlabel("settle steps (fixed)")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_title(f"{scenario.replace('_', '-')}: accuracy vs fixed settle steps "
                     f"(dt={DT}, solid=PC dashed=BP)", fontsize=9)
    ax_acc.grid(alpha=0.2)
    if row == 0:
        ax_acc.legend(fontsize=7, loc="lower right")

fig.tight_layout()
fig.savefig(figure_path(), dpi=130, bbox_inches="tight")
print(f"saved {figure_path()}")
