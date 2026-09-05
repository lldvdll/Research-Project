"""In the unfrozen (control) condition, does the hidden code still carry task-1 class
information after argmax has already lost it?

Report series 2, decade 130 (companion to 130's freeze factorial), control condition only.
Backprop + PC, both scenarios, matched-competence 90%, 10 seeds. Uses the NCM probe already
built in src/probes.py -- unused so far this series.

FROZEN prototypes, not live: prototypes are snapshotted from the hidden code the moment task 1
ends (via run_classil's `on_task_end` hook) and never rebuilt afterwards. Live prototypes rotate
with the code and are blind to any consistent transformation of it -- see probes.py's own
docstring, and knowledge_base.md's record that an earlier probe using live prototypes had no
dynamic range for exactly this reason. Frozen prototypes measure whether the code has actually
moved away from where it was.

Domain-IL note: prototypes are built from task-1's classes then re-keyed from class id to OUTPUT
UNIT (label_map), because under Domain-IL several classes share a unit and the accuracy check
compares predictions against label_map'd targets. Not needed for Class-IL, where the two coincide.

This is a presentation figure, not a decision figure (see probes.py / knowledge_base.md Sec.4.5):
one panel per (scenario, rule), task-1 argmax accuracy vs task-1 NCM accuracy, over time relative
to the switch, spaghetti + mean. Argmax << NCM says the head is the problem and the code
survived; both low says the code itself was damaged.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, build, replace, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import run_classil
from src.probes import class_prototypes, frozen_ncm_fn

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):
    return _array_path(f, _tag(f, suffix))


# ---------------------------------------------------------------- settings
METHODS = ["backprop", "pc"]
LR = {"backprop": 0.01, "pc": 0.02}
SCENARIOS = ["class_il", "domain_il"]
SEEDS = 10
EVAL_EVERY = 10
MAX_ITERS = 5000
STOP_PATIENCE = 3
THRESHOLD = 0.90
PROTO_PER_CLASS = 100
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"

if SMOKE:
    SEEDS, EVAL_EVERY, MAX_ITERS, PROTO_PER_CLASS = 2, 5, 200, 20
    print("--smoke: tiny budget, results are NOT meaningful\n")

if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32")


def unit_prototypes(features_fn, train_data, class_idx, classes, label_map, per_class, device, seed):
    """class_prototypes, re-keyed by OUTPUT UNIT rather than raw class id -- so predictions
    compare correctly against label_map'd targets under Domain-IL, where several classes share a
    unit. A no-op under Class-IL, where label_map is None and unit == class id."""
    raw = class_prototypes(features_fn, train_data, class_idx, classes, per_class=per_class,
                           device=device, seed=seed)
    if label_map is None:
        return raw
    return {label_map[c]: v for c, v in raw.items()}


def run_with_ncm(proto, method, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    handle = {}
    train_step, predict = build(proto, method, seed, handle=handle)
    protos_holder = {}

    def on_task_end(ti, step):
        if ti == 0:
            protos_holder.update(unit_prototypes(
                handle["features"], data.train, data.class_idx, tasks[0], lmap,
                PROTO_PER_CLASS, proto.device, seed))

    readouts = {"argmax": predict, "ncm": frozen_ncm_fn(handle["features"], protos_holder)}
    return run_classil(train_step, predict, tasks, data.train, data.class_idx,
                       report_eval=data.report_eval, stop_eval=data.stop_eval, readouts=readouts,
                       max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
                       eval_every=proto.eval_every, device=proto.device,
                       stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
                       data_seed=seed, label_map=lmap, on_task_end=on_task_end)


def common_window(steps, A):
    """Crop to the range where EVERY seed has finite data -- see 102/113 for why."""
    first, last = [], []
    for r in range(A.shape[0]):
        idx = np.where(np.isfinite(A[r, :, 0]) & np.isfinite(A[r, :, 1]))[0]
        if idx.size:
            first.append(idx[0]); last.append(idx[-1])
    lo_idx, hi_idx = max(first), min(last)
    return steps[lo_idx:hi_idx + 1], A[:, lo_idx:hi_idx + 1, :]


if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__), allow_pickle=True)
    diag = {s: {m: {rd: (z[f"steps_{s}_{m}_{rd}"], z[f"curve_{s}_{m}_{rd}"])
                    for rd in ("argmax", "ncm")} for m in METHODS} for s in SCENARIOS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    base = replace(PROTOCOL, hidden=H, eval_every=EVAL_EVERY, seeds=SEEDS, lr=LR,
                   max_iters_per_task=MAX_ITERS, stop_patience=STOP_PATIENCE,
                   stop_threshold=THRESHOLD)
    data = load(replace(base, scenario="class_il"))
    diag = {s: {m: {} for m in METHODS} for s in SCENARIOS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        proto = replace(base, scenario=scen)
        for method in METHODS:
            runs = [run_with_ncm(proto, method, seed, data) for seed in range(SEEDS)]
            n_bad = sum(1 for o in runs if not all(o["reached"]))
            if n_bad:
                print(f"  WARNING: {scen} {method}: {n_bad}/{SEEDS} seed(s) hit the cap")
            lo = -int(1.2 * np.median([o["switches"][0] for o in runs]))
            hi = int(1.2 * np.median([o["switches"][1] - o["switches"][0] for o in runs]))
            rel_grid = np.arange(lo, hi + EVAL_EVERY, EVAL_EVERY)
            for rd in ("argmax", "ncm"):
                A = np.full((SEEDS, len(rel_grid), 2), np.nan)
                for i, o in enumerate(runs):
                    rel = np.asarray(o["steps"]) - o["switches"][0]
                    for j, r in enumerate(rel):
                        k_ = int(round((r - lo) / EVAL_EVERY))
                        if 0 <= k_ < len(rel_grid):
                            A[i, k_] = o["curves"][rd][j]
                diag[scen][method][rd] = (rel_grid, A)
            t1_final = np.nanmean(diag[scen][method]["argmax"][1][:, -1, 0]) * 100
            t1_ncm = np.nanmean(diag[scen][method]["ncm"][1][:, -1, 0]) * 100
            print(f"  {scen:10s} {method:8s} task1 final: argmax {t1_final:5.1f}  ncm {t1_ncm:5.1f}"
                  f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- report figure: 2x2, argmax vs ncm on task 1
fig, axes = plt.subplots(len(SCENARIOS), len(METHODS),
                          figsize=(4.4 * len(METHODS), 3.4 * len(SCENARIOS)), sharey=True)
for row, scen in enumerate(SCENARIOS):
    for col, method in enumerate(METHODS):
        ax = axes[row, col]
        steps_a, A = common_window(*diag[scen][method]["argmax"])
        steps_n, N = common_window(*diag[scen][method]["ncm"])
        for r in range(A.shape[0]):
            ax.plot(steps_a, A[r, :, 0] * 100, color="tab:red", lw=0.5, alpha=0.2)
        for r in range(N.shape[0]):
            ax.plot(steps_n, N[r, :, 0] * 100, color="tab:green", lw=0.5, alpha=0.2)
        ax.plot(steps_a, np.nanmean(A[:, :, 0], axis=0) * 100, color="tab:red", lw=2.0, label="argmax")
        ax.plot(steps_n, np.nanmean(N[:, :, 0], axis=0) * 100, color="tab:green", lw=2.0, label="ncm")
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        if col == 0:
            ax.set_ylabel(f"{scen.replace('_', '-')}\ntask-1 acc (%)", fontsize=9)
        if row == 0:
            ax.text(0.5, 1.05, method, fontsize=9, ha="center", transform=ax.transAxes)
axes[0, 0].legend(fontsize=8, loc="lower left")
for ax in axes[-1]:
    ax.set_xlabel("step, relative to switch")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), H=H,
         **{f"steps_{s}_{m}_{rd}": diag[s][m][rd][0]
            for s in SCENARIOS for m in METHODS for rd in ("argmax", "ncm")},
         **{f"curve_{s}_{m}_{rd}": diag[s][m][rd][1]
            for s in SCENARIOS for m in METHODS for rd in ("argmax", "ncm")})
print(f"saved {array_path(__file__)}")
