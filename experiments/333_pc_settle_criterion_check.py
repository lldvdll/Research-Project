"""Is 330's delta-based settle-convergence criterion well-calibrated, or is it what made the dt
sweep slow -- checked at PC's own KNOWN-WORKING configuration (dt=0.1, steps=50; scripts 50/51/59
measured a worst case of 18 settle steps at this exact H=32, 50 kept as margin), at several points
through an actual matched-competence training run, logging every settle step.

330's TRAIN_STEPS=237 was 1.5x the SLOWEST dt's (0.01) convergence step, applied UNIFORMLY to
EVERY dt in the sweep -- so dt=0.1, which this project has run PC at successfully since 102, was
being forced through 237 settle iterations per weight update instead of the ~18-50 history says it
needs. Two separate questions follow, and this script is built to tell them apart:
    (a) is the CRITERION (delta < 1e-4 for 10 consecutive steps) miscalibrated -- firing much later
        than where the trace actually, practically levels off, even at dt=0.1?
    (b) or is the criterion fine at dt=0.1, and 330's slowness purely an ALLOCATION problem --
        one global step budget set by the slowest dt, applied to every dt regardless of need?
If this script's dt=0.1 traces converge (by the same criterion) close to the historical ~18-50
window, the answer is (b): 330 needs a PER-DT step budget, not a smaller cap or a looser criterion.
If they converge much later, or never, at dt=0.1 despite the run visibly reaching high accuracy,
the criterion itself needs loosening.

Method: one matched-competence sequential run per (scenario, seed) at dt=0.1, steps=50, lr=0.02
(config's established pc lr) -- run_classil, UNCHANGED, so this is the exact "version we know
works", not a fixed-budget stand-in. train_step is wrapped (not modified) to intercept calls and,
at chosen checkpoints -- init, 50 calls into each task, and each task's end (via on_task_end) --
pause and run ONE long relaxation (pc_settle, trace=True, cap 300 steps -- 6x the historical
budget, to expose any noisy tail) on a representative batch, logging every single step. Accuracy
is checked at every checkpoint too, so a "settled fine" trace is not accidentally describing a run
that never learned anything.

Figure: one PNG, rows = scenario, columns = checkpoint stage (init, task1@50, task1-end,
task2@50, task2-end). Each panel: thin per-seed displacement traces, dashed vertical line at the
historical steps=50 budget, a marker at each trace's own detected convergence step (same
criterion as 330), panel title reports mean task-1/task-2 accuracy at that checkpoint.
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
from src.runner import run_classil, _loader, _acc
from src.predictive_coding import pc_settle
from src.model import make_target

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

METHOD = "pc"
KNOWN_DT = 0.1        # METHOD_DEFAULTS["pc"]["dt"] -- the config every PC script 100-230 actually ran
KNOWN_STEPS = 50      # scripts 50/51/59: worst case measured 18 at this H=32, 50 kept as margin

SETTLE_DELTA_TOL = 1e-4   # identical criterion to 330 -- this script tests THAT criterion
SETTLE_PATIENCE = 10

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, SEED_START = 2, 10
    DIAG_STEP_CAP = 60
    MID_CHECKPOINT = 20
    EVAL_EVERY = 5
    MAX_ITERS = 100
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = 3            # diagnostic, not a headline sweep -- 3 seeds is enough to see if the
                         # noise pattern is seed-consistent, not a statistical claim
    SEED_START = CFG["training"]["seed_start"]
    DIAG_STEP_CAP = 300  # 6x KNOWN_STEPS, generous enough to expose a noisy/long tail
    MID_CHECKPOINT = 50  # calls into each task at which to snapshot mid-task
    EVAL_EVERY = CFG["evaluation"]["eval_every"]
    MAX_ITERS = CFG["training"]["max_iters_per_task"]

SCENARIOS = CFG["data"]["scenarios"]
STAGES = ["init", "task1@mid", "task1_end", "task2@mid", "task2_end"]

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
    lr={METHOD: CFG["training"]["learning_rate"]["pc"]},
    stop_threshold=CFG["training"]["stop_threshold"],
    stop_patience=CFG["training"]["stop_patience"],
    max_iters_per_task=MAX_ITERS, seeds=SEEDS,
    eval_per_class=CFG["evaluation"]["eval_per_class"], eval_every=EVAL_EVERY,
    device=CFG["evaluation"]["device"],
)


def _tag():
    return "SMOKE" if SMOKE else ""


def figure_path():
    return _figure_path(__file__, _tag())


def array_path():
    return _array_path(__file__, _tag())


def _settle_converged_at(disp):
    """Same criterion as 330: first index where |disp[i]-disp[i-1]| stays below
    SETTLE_DELTA_TOL for SETTLE_PATIENCE consecutive steps. None if it never does."""
    delta = np.abs(np.diff(disp))
    below = delta < SETTLE_DELTA_TOL
    for i in range(len(below) - SETTLE_PATIENCE + 1):
        if below[i:i + SETTLE_PATIENCE].all():
            return i + 1
    return None


def run_scenario_seed(proto, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    classes = sorted({c for t in tasks for c in t})
    pos = {c: i for i, c in enumerate(classes)}
    handle = {}
    train_step, predict = build(proto, METHOD, seed, handle=handle, dt=KNOWN_DT, steps=KNOWN_STEPS)
    p, arch, obj = handle["params"], handle["arch"], handle["obj"]

    classes_all = list(range(proto.n_classes))
    # Fixed-seed loader re-iterated per snapshot: same representative batch every time, so
    # checkpoints differ only in the WEIGHTS, not in which examples happened to be sampled.
    diag_loader = _loader(data.train, data.class_idx, classes_all, proto.batch, seed=seed + 555)
    report_x, report_y = data.report_eval

    snapshots = {}

    def snapshot(label):
        x, y = next(iter(diag_loader))
        x, y = x.to(proto.device), y.to(proto.device)
        y2 = y if lmap is None else torch.tensor([lmap[int(v)] for v in y.tolist()], device=proto.device)
        target = make_target(y2, arch, obj, device=proto.device)
        _, _, disp = pc_settle(x, p, arch, obj, target, dt=KNOWN_DT, steps=DIAG_STEP_CAP, trace=True)
        a = _acc(predict(report_x), report_y, classes, lmap)
        acc_row = [float(np.mean([a[pos[c]] for c in t])) for t in tasks]
        snapshots[label] = (np.asarray(disp), acc_row)

    snapshot("init")

    state = {"calls": 0, "task_idx": 0, "mid_done": set()}

    def wrapped(x, y, active=None):
        train_step(x, y, active=active)
        state["calls"] += 1
        if state["calls"] == MID_CHECKPOINT and state["task_idx"] not in state["mid_done"]:
            snapshot(f"task{state['task_idx'] + 1}@mid")
            state["mid_done"].add(state["task_idx"])

    def on_task_end(ti, step):
        snapshot(f"task{ti + 1}_end")
        state["calls"] = 0
        state["task_idx"] = ti + 1

    run_classil(wrapped, predict, tasks, data.train, data.class_idx,
               report_eval=data.report_eval, stop_eval=data.stop_eval,
               max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
               eval_every=proto.eval_every, device=proto.device,
               stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
               data_seed=seed, label_map=lmap, on_task_end=on_task_end)
    return snapshots


# ---------------------------------------------------------------- run or reload
if REPLOT and Path(array_path()).exists():
    z = np.load(array_path(), allow_pickle=True)
    results = z["results"].item()
    print("--replot: redrawing from saved arrays, no training\n")
else:
    t0 = time.perf_counter()
    results = {}   # scenario -> seed -> stage -> (disp, acc_row)
    for scenario in SCENARIOS:
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        data = load(proto)
        results[scenario] = {}
        for seed in range(SEED_START, SEED_START + SEEDS):
            results[scenario][seed] = run_scenario_seed(proto, seed, data)
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]")
    np.savez(array_path(), results=np.array(results, dtype=object))
    print(f"saved {array_path()}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(len(SCENARIOS), len(STAGES),
                         figsize=(3.2 * len(STAGES), 3.4 * len(SCENARIOS)), squeeze=False)

for row, scenario in enumerate(SCENARIOS):
    seeds_data = results[scenario]
    for col, stage in enumerate(STAGES):
        ax = axes[row][col]
        convs, t1_accs, t2_accs = [], [], []
        for seed, stages in seeds_data.items():
            if stage not in stages:
                continue
            disp, acc_row = stages[stage]
            ax.plot(np.arange(1, len(disp) + 1), disp, lw=0.9, alpha=0.7)
            conv = _settle_converged_at(disp)
            if conv is not None and conv <= len(disp):
                ax.scatter([conv], [disp[conv - 1]], marker="x", color="black", s=35, zorder=5)
                convs.append(conv)
            t1_accs.append(acc_row[0] * 100)
            t2_accs.append(acc_row[1] * 100)
        ax.axvline(KNOWN_STEPS, color="tab:red", ls="--", lw=1.0)
        ax.set_yscale("log")
        conv_str = f"conv~{np.mean(convs):.0f}" if convs else "never conv."
        t1_str = f"{np.mean(t1_accs):.0f}" if t1_accs else "-"
        t2_str = f"{np.mean(t2_accs):.0f}" if t2_accs else "-"
        ax.set_title(f"{stage}\n{conv_str} | t1={t1_str}% t2={t2_str}%", fontsize=7.5)
        ax.grid(alpha=0.2, which="both")
        if col == 0:
            ax.set_ylabel(f"{scenario.replace('_', '-')}\ndisplacement", fontsize=8)
        if row == len(SCENARIOS) - 1:
            ax.set_xlabel("settle step", fontsize=8)

fig.suptitle(f"PC settle trace at KNOWN-GOOD config (dt={KNOWN_DT}, steps={KNOWN_STEPS} shown "
            f"dashed) -- {SEEDS} seeds, every step logged, cap={DIAG_STEP_CAP}", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(figure_path(), dpi=130, bbox_inches="tight")
print(f"saved {figure_path()}")
