"""Does PC's settle step size (dt) change WHERE it settles, or only how many steps it takes to
get there -- checked via ACTUAL settle-step counts recorded live during training, and via final
accuracy under joint and sequential training, both IL scenarios.

THIRD version of this script, trimmed for speed after the second version's real run showed its
actual cost. Changes from that version, all requested to cut wall-clock without dropping the
learning/forgetting question itself:
    - TWO conditions, not four. Dropped task-1-alone/task-2-alone: those exist to distinguish
      "PC can't learn well at this dt in general" from "it's a CL-specific interaction", which is
      a mechanism question for later (narrative_plan.md §3), not needed to just PICK a dt control.
      joint (learnability, no CL) + sequential (retention AND new-task accuracy, i.e. forgetting)
      is exactly the learning-and-forgetting check this script is actually for.
    - max_iters_per_task cut from config's 5000 to 1500 FOR THIS SWEEP ONLY (not a global change).
      Runs that converge do so in far fewer iterations than 5000; the larger cap only ever mattered
      for runs that fail anyway (e.g. dt=1.0), where it just burns more time to the same "did not
      reach threshold" outcome. 1500 bounds that waste without meaningfully affecting working runs.
    - SETTLE_STEP_CAP cut from 300 to 200. dt=1.0 never converges, so it pays this cap on EVERY
      weight update -- it directly sets that dt's cost. 200 stays comfortably above what the
      slowest kept dt (0.02) needs (334: single-seed settle ~150 steps at patience=3).
    - Seeds halved (10 -> 5) and dt grid trimmed using 334's actual settle-trace results (see
      below), both already in place before this pass.

Report series 2, decade 330 (narrative_plan.md's PC sweep, §2). PC in isolation -- no BP, no
replay; those return in the later lr/batch-size comparison phase. lr fixed at config_300.yaml's
existing pc value (0.02, grid-searched in 102 for learning speed, not retention). Every other
control parameter is read from config_300.yaml.

SETTLING NOW STOPS ITSELF -- this is the second version of this script. The first version picked
a FIXED number of settle steps per dt in advance (derived from a separate diagnostic probe) and
forced every weight update through that count regardless of need. That was wrong on its own terms
(steps is not a thing to choose, it's a thing to measure) and it made the sweep take upwards of
18 hours, because ONE global budget set by the slowest dt (0.01) was applied to every dt,
including 0.1 -- which this project has run PC at since 102 and which settles in ~20 steps.

src.predictive_coding.pc_settle/pc_update and src.methods.make_pc were extended (approved change)
to accept stop_delta/stop_patience: when set, `steps` becomes a CAP, and settling stops itself
once the displacement's step-to-step change stays below stop_delta for stop_patience consecutive
steps -- the exact criterion 333 validated (fires ~14-26 steps in at dt=0.1, matching scripts
50/51/59's historical "worst case 18", across every training stage checked). Patience lowered
from 333's 10 to 3 here, matching this project's own stop_patience convention (config_300.yaml)
-- 3 is not a large fraction of a ~15-25 step settle, and 333's traces showed no noise that would
need a longer patience to reject.

Every train_step call publishes the settle steps it actually took to handle["diag"]["settle_
steps"]; a thin wrapper records that after every call, so "steps to converge" for a given dt is
now measured from the SAME real training runs the accuracy numbers come from, not a separate probe.

TWO TRAINING CONDITIONS, per (scenario, dt, seed) -- same fresh init shared across both, since
build(proto, "pc", seed, ...) reseeds init_params identically each call:
    joint       all ten classes trained together, no task switch (src.runner.run_joint). Sanity
                check that a low sequential number is forgetting, not "this dt can't learn at all".
    sequential  the actual continual run (src.runner.run_classil, task1 then task2) -- final
                task-1 accuracy (retention) and final task-2 accuracy, together.

Figure: one PNG, 2 rows (class_il, domain_il) x 2 columns (dt vs settle steps actually taken,
pooled across both conditions/seeds | dt vs final accuracy, FOUR series overlaid: joint, task-1
retention, task-2 final, crossover height). Crossover added because it is this project's own
designated primary metric (src.metrics.report_grid(primary="crossover")) and 310's whole headline
figure is built around it -- final-accuracy alone was an oversight, not a deliberate omission, and
it costs nothing extra since it is computed from the same curve run_classil already returns.
Domain-IL's shared output units are handled by wrapping train_step to remap labels through
Protocol.label_map before calling run_joint (which has no label_map parameter of its own);
run_classil does its own remapping and must NOT be double-wrapped (see _make_sequential_step's
docstring for the bug this was).
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
from src.runner import run_joint, run_classil
from src.metrics import crossover as crossover_fn

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv
_add_dt_arg = next((a for a in sys.argv if a.startswith("--add-dt=")), None)
ADD_DT = float(_add_dt_arg.split("=")[1]) if _add_dt_arg else None

METHOD = "pc"

# ---------------------------------------------------------------- script-local constants
# Criterion validated by 333 at dt=0.1: fires ~14-26 steps in, at every training stage checked,
# matching scripts 50/51/59's historical "worst case 18". Patience 3 (not 333's 10) per this
# project's own stop_patience convention -- 333's traces were smooth, no noise to reject.
SETTLE_DELTA_TOL = 1e-4
SETTLE_PATIENCE = 3
STOP_PATIENCE_JOINT = 5   # run_joint's own plateau patience (evals, not settle steps)
MIN_DELTA_JOINT = 1e-3

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, SEED_START = 2, 10
    DT_GRID = [0.05, 0.5]
    SETTLE_STEP_CAP = 60
    TRAIN_MAX_ITERS = 100
    EVAL_EVERY = 5
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = CFG["training"]["seeds"] // 2   # halved (10 -> 5): the first full run showed this
                                            # sweep's cost, and a dt control-selection pass
                                            # doesn't need the full headline seed count
    SEED_START = CFG["training"]["seed_start"]
    # Trimmed using 334's actual settle-trace results, not guessed: 334 (single seed, both
    # scenarios) found dt in {0.01,...,0.5} all converge to the SAME plateau (0.01 slowest, ~150+
    # settle steps/update; 0.5 fastest, ~15) and dt in {0.7,0.85,1.0,1.5} never converge (a
    # distinct, higher, oscillating plateau instead). Dropped 0.01 only (slowest to settle, and
    # the slowest-converging value is exactly what drove the ORIGINAL fixed-budget design's
    # blowup -- not worth its cost just to confirm it's a bad control choice). 1.0 kept despite
    # being unstable -- explicitly requested, and it's still useful to see what accuracy actually
    # does right at a dt that never settles. 0.85 kept as the boundary/edge reference. 0.6 added
    # to resolve the unsampled gap between 0.5 (converges, ~6 steps) and 0.85 (never converges) --
    # the transition itself hasn't been located yet, only bracketed.
    DT_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.85, 1.0]   # 0.3 and 0.4 both added via
    # --add-dt, merged into already-saved data rather than a full rerun -- see the ADD_DT branch
    # below. Resolution around the settle-steps trough (originally sampled only at 0.2 and 0.5)
    # was too coarse to be sure of the true minimum -- 0.3 turned out lower than both neighbours.
    # 100, not 200 -- direct check just run at patience=3 (real data, not the old patience=10
    # estimates): dt=0.02 (the slowest kept value) converges at ~85-86 steps, both seeds checked.
    # 100 clears that with a small margin while still cutting real cost off the two dt that never
    # converge (0.85, 1.0), which pay this cap on every single weight update.
    SETTLE_STEP_CAP = 100
    TRAIN_MAX_ITERS = 1500  # cut from config's 5000 for this sweep only -- see docstring
    EVAL_EVERY = CFG["evaluation"]["eval_every"]

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
    lr={METHOD: CFG["training"]["learning_rate"]["pc"]},
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


def array_path(scenario):
    return _array_path(__file__, _tag(scenario))


def _make_joint_step(train_step, handle, settle_log, lmap):
    """For run_joint, which has NO label_map parameter of its own -- wraps train_step to (a)
    remap labels/active for Domain-IL and (b) record the settle steps that call actually took,
    read live off handle["diag"] after each update."""
    def step(x, y, active=None):
        if lmap is not None:
            y = torch.tensor([lmap[int(v)] for v in y.tolist()], device=y.device)
            active = None if active is None else sorted({lmap[int(c)] for c in active})
        train_step(x, y, active=active)
        if "settle_steps" in handle["diag"]:
            settle_log.append(handle["diag"]["settle_steps"])
    return step


def _make_sequential_step(train_step, handle, settle_log):
    """For run_classil, which ALREADY remaps y/active itself when given label_map= -- this wrapper
    must NOT remap again. (Bug found live: 330's first adaptive-stop run wrapped this path with
    the joint-style remapper too, double-applying lmap to already-unit-space labels/active and
    silently colliding classes onto the wrong units under Domain-IL -- e.g. [0,1,2,3,4] ->
    [0,2,2,0,4]. That's why every Domain-IL seed was hitting the iteration cap regardless of dt:
    the model was training on scrambled targets, not struggling to learn. Class-IL was unaffected
    since label_map is None there, so the (buggy) remap was a no-op.) Only settle-step logging
    happens here."""
    def step(x, y, active=None):
        train_step(x, y, active=active)
        if "settle_steps" in handle["diag"]:
            settle_log.append(handle["diag"]["settle_steps"])
    return step


def _filter_eval(eval_xy, classes, lmap):
    x, y = eval_xy
    mask = torch.isin(y, torch.as_tensor(classes, device=y.device))
    x, y = x[mask], y[mask]
    if lmap is not None:
        y = torch.tensor([lmap[int(v)] for v in y.tolist()], device=y.device)
    return x, y


def run_joint_condition(proto, dt, seed, data, classes, lmap, settle_log):
    handle = {}
    train_step, predict = build(proto, METHOD, seed, handle=handle, dt=dt, steps=SETTLE_STEP_CAP,
                               stop_delta=SETTLE_DELTA_TOL, stop_patience=SETTLE_PATIENCE)
    ex, ey = _filter_eval(data.report_eval, classes, lmap)
    ts = _make_joint_step(train_step, handle, settle_log, lmap)
    _, accs = run_joint(ts, predict, classes, data.train, data.class_idx, ex, ey,
                        max_iters=TRAIN_MAX_ITERS, batch=proto.batch, eval_every=proto.eval_every,
                        device=proto.device, stop_patience=STOP_PATIENCE_JOINT,
                        min_delta=MIN_DELTA_JOINT, data_seed=seed)
    return float(accs[-1]) * 100 if len(accs) else float("nan")


def run_sequential_condition(proto, dt, seed, data, settle_log):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    handle = {}
    train_step, predict = build(proto, METHOD, seed, handle=handle, dt=dt, steps=SETTLE_STEP_CAP,
                               stop_delta=SETTLE_DELTA_TOL, stop_patience=SETTLE_PATIENCE)
    ts = _make_sequential_step(train_step, handle, settle_log)
    out = run_classil(ts, predict, tasks, data.train, data.class_idx,
                      report_eval=data.report_eval, stop_eval=data.stop_eval,
                      max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
                      eval_every=proto.eval_every, device=proto.device,
                      stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
                      data_seed=seed, label_map=lmap)
    curve = out["curves"]["argmax"]
    # crossover is undefined (NaN) if the curves never cross -- CENSORED, meaning task-1 never
    # fell below task-2, the BEST outcome (see metrics.py's CENSORED_IS_BEST). _agg's plain
    # nanmean/SEM treatment below is a simplification appropriate for this exploratory
    # control-selection plot, not the paired_sign-aware treatment the final headline figure uses.
    _, cx_height = crossover_fn(out["steps"], curve[:, 0] * 100, curve[:, 1] * 100,
                                after=out["switches"][0])
    return float(curve[-1, 0]) * 100, float(curve[-1, 1]) * 100, cx_height, all(out["reached"])


def _rows_for_dt(scenario, proto, data, dt):
    """Every seed's row for ONE dt value -- shared by the full sweep and by --add-dt, so a
    single new dt point can be merged into already-saved data without rerunning the rest."""
    classes_all = list(range(proto.n_classes))
    rows = []
    for seed in range(SEED_START, SEED_START + SEEDS):
        tasks = proto.tasks(seed)
        lmap = proto.label_map(tasks)
        settle_log = []
        joint_acc = run_joint_condition(proto, dt, seed, data, classes_all, lmap, settle_log)
        seq_t1, seq_t2, cx_height, reached_ok = run_sequential_condition(
            proto, dt, seed, data, settle_log)
        if not reached_ok:
            print(f"  WARNING: {scenario} dt={dt} seed={seed}: sequential run hit the cap")
        n_uncap = sum(1 for s in settle_log if s < SETTLE_STEP_CAP)
        rows.append((dt, seed, joint_acc, seq_t1, seq_t2, cx_height,
                    float(np.mean(settle_log)) if settle_log else float("nan"),
                    n_uncap, len(settle_log)))
    return rows


# ---------------------------------------------------------------- run or reload
if REPLOT and all(Path(array_path(s)).exists() for s in SCENARIOS):
    train_rows = {s: list(np.load(array_path(s), allow_pickle=True)["data"]) for s in SCENARIOS}
    print("--replot: redrawing from saved arrays, no training\n")
elif ADD_DT is not None:
    if not all(Path(array_path(s)).exists() for s in SCENARIOS):
        raise RuntimeError("--add-dt needs existing saved data to merge into -- run the full "
                          "sweep first")
    t0 = time.perf_counter()
    train_rows = {}
    for scenario in SCENARIOS:
        existing = [r for r in np.load(array_path(scenario), allow_pickle=True)["data"]
                   if r[0] != ADD_DT]   # drop any stale rows for this dt before re-adding
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        data = load(proto)
        new_rows = _rows_for_dt(scenario, proto, data, ADD_DT)
        train_rows[scenario] = existing + new_rows
        print(f"  {scenario:10s} added dt={ADD_DT}   [{time.perf_counter() - t0:5.0f}s]")
    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(train_rows[scenario], dtype=object))
    print("saved arrays (merged)")
else:
    t0 = time.perf_counter()
    train_rows = {s: [] for s in SCENARIOS}
    for scenario in SCENARIOS:
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        data = load(proto)
        for dt in DT_GRID:
            train_rows[scenario].extend(_rows_for_dt(scenario, proto, data, dt))
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]")

    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(train_rows[scenario], dtype=object))
    print("saved arrays")


# ---------------------------------------------------------------- aggregate mean/SEM per dt
def _agg(rows, col, dt_grid):
    means, sems = [], []
    for dt in dt_grid:
        vals = np.asarray([r[col] for r in rows if r[0] == dt], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            means.append(np.nan); sems.append(np.nan)
        elif vals.size == 1:
            means.append(vals[0]); sems.append(0.0)
        else:
            means.append(vals.mean()); sems.append(vals.std(ddof=1) / np.sqrt(vals.size))
    return np.array(means), np.array(sems)


# ---------------------------------------------------------------- figure
def _set_dt_ticks(ax, dts):
    """Log-scale default ticks are powers of ten, which don't land on the actual dt values
    tested (0.02, 0.6, 0.85, ...) -- explicit ticks so every point on the x-axis is readable."""
    ax.set_xscale("log")
    ax.set_xticks(dts)
    ax.set_xticklabels([str(dt) for dt in dts], rotation=45, fontsize=7)
    ax.minorticks_off()


# One accuracy panel, not split -- same units (accuracy %), same x-axis (dt), and the flat-line
# story is identical across all four series, which sit at clearly separated levels (final ~90%,
# joint ~77%, crossover ~65-76%, retention ~3-40%), so nothing overlaps confusingly.
ACC_SERIES = [
    ("joint", 2, "tab:purple", "o", "joint (all 10)"),
    ("seq_t2", 4, "tab:green", "s", "task-2 after task-1 (final)"),
    ("seq_t1", 3, "tab:red", "s", "task-1 after task-2 (retention)"),
    ("crossover", 5, "tab:blue", "d", "crossover height"),
]

fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(11, 4.5 * len(SCENARIOS)), squeeze=False)

for row, scenario in enumerate(SCENARIOS):
    ax_settle, ax_acc = axes[row]
    trows = train_rows[scenario]
    dts = sorted(set(r[0] for r in trows))

    m, se = _agg(trows, 6, dts)   # mean settle steps per run, pooled over conditions/seeds
    ax_settle.errorbar(dts, m, yerr=se, marker="o", color="tab:gray", capsize=3)
    for dt in dts:
        rows_dt = [r for r in trows if r[0] == dt]
        n_uncap = sum(r[7] for r in rows_dt)
        n_total = sum(r[8] for r in rows_dt)
        pct_uncap = 100 * n_uncap / n_total if n_total else float("nan")
        if pct_uncap < 99.5:
            ax_settle.annotate(f"{pct_uncap:.0f}% <cap", (dt, m[dts.index(dt)]),
                              fontsize=6, color="tab:red", xytext=(0, 8),
                              textcoords="offset points", ha="center")
    _set_dt_ticks(ax_settle, dts)
    ax_settle.set_yscale("log")
    ax_settle.set_xlabel("dt")
    ax_settle.set_ylabel("settle steps actually taken (mean)")
    ax_settle.set_title(f"{scenario.replace('_', '-')}: dt vs settle steps (adaptive stop)",
                        fontsize=9)
    ax_settle.grid(alpha=0.2, which="both")

    for key, col, color, marker, label in ACC_SERIES:
        m, se = _agg(trows, col, dts)
        ax_acc.errorbar(dts, m, yerr=se, marker=marker, color=color, label=label, capsize=3)
    _set_dt_ticks(ax_acc, dts)
    ax_acc.set_ylim(0, 100)
    ax_acc.set_xlabel("dt")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_title(f"{scenario.replace('_', '-')}: dt vs accuracy "
                     f"(settling stops itself, cap={SETTLE_STEP_CAP})", fontsize=9)
    ax_acc.grid(alpha=0.2)
    if row == 0:
        ax_acc.legend(fontsize=7, loc="lower left")

fig.tight_layout()
fig.savefig(figure_path(), dpi=130, bbox_inches="tight")
print(f"saved {figure_path()}")
