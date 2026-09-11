"""Across hidden widths from near the capacity floor (H=4) to double the established H=32, does
PC's retention/crossover difference from backprop hold, shrink, or reverse -- with replay
alongside, swept at the same widths, as the "what remembering actually looks like" reference?

Report series 2, decade 341. Second of the three-part BP-vs-PC comparison (narrative_plan.md's
S2/340 series), after 340 (lr). Unlike 340, replay is NOT held at a fixed reference here -- width
is a shared architecture parameter, not a rule-specific tuning knob, so all three methods
(backprop, replay, pc) are trained at every width point.

WIDTHS = [4, 8, 16, 32, 64] -- a subset of script 100's own capacity grid ([2,4,8,...,256]),
bracketing the established H=32 on both sides. H=4 is expected to sit AT OR BELOW the capacity
floor (script 100/41's capacity sweep), so some seeds there may never reach the 90% stop threshold
and will run the full max_iters_per_task budget on one or both tasks -- this is the point of
including it, and is reported via `reached`, not silently averaged in as if convergent.

Depth is held at 1 (established). lr is held at each rule's own established default from
config_300.yaml (backprop 0.01, pc 0.02); replay is not in that dict, so it keeps its own
METHOD_DEFAULTS lr (0.05) automatically. 340 (run first, on purpose) is the check for whether
those lr values are actually each rule's best operating point for retention/crossover -- this
script does not re-litigate that, it holds the established pairing fixed and varies width only.

PC's settle control (dt=0.4, adaptive stop: stop_delta=1e-4, stop_patience=3, steps=100 cap) is
held fixed at its H=32/depth=1-validated value (330-334) -- NOT re-verified per width here. If a
width cell shows PC failing to reach threshold at a rate out of line with backprop/replay at the
SAME width, that is a candidate sign the settle control itself needs revisiting at that scale, not
an assumption this script makes silently -- `reached` makes it visible either way.

FULL max_iters_per_task budget (config_300.yaml's 5000, not trimmed) -- same reasoning as 340: the
capacity-floor question needs the real established budget to mean anything.

SEEDS = 5 (10-14). --add-seeds=LO-HI merges 15-19 in later without rerunning existing cells.
--add-width=X merges one new width in (330's dt-merge pattern, generalised, as in 340's --add-lr).

Metrics, per seed, computed directly (ragged curves under matched-competence stopping -- see
332's convention): retention, task-2 final accuracy, crossover height and its step (relative to
the task switch), steps for task 1 and task 2 to each reach 90%, S&B's mean test error
(src.metrics.mean_test_error) for both tasks. Paired statistics (paired_diff, paired_sign,
paired_wilcoxon; control=backprop) at every width, on retention and crossover height. Replay's
own retention/crossover is printed as a benchmark, not paired-tested -- it is the positive
control, not a hypothesis under test here.

The H=32 cell is trained fresh (not reused from 310/332/340) -- agreement across independently-
coded scripts on the same cell is a cheap correctness check, not redundant compute.

Two figures: 341_width_sweep_accuracy.png (retention, crossover height) and
341_width_sweep_speed.png (task-2 steps to 90%, crossover step), each 2 scenario columns x 2
metric rows, box-and-whisker per width, one box per method (backprop, replay, pc).

--smoke: tiny budget, not meaningful. --replot: redraw from saved arrays, no training.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, replace, load, build, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import run_classil, _acc
from src.metrics import crossover as crossover_fn, mean_test_error, paired_diff, paired_sign, \
    paired_wilcoxon

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv
_add_seeds_arg = next((a for a in sys.argv if a.startswith("--add-seeds=")), None)
ADD_SEEDS = None
if _add_seeds_arg:
    lo, hi = _add_seeds_arg.split("=")[1].split("-")
    ADD_SEEDS = list(range(int(lo), int(hi) + 1))
_add_width_arg = next((a for a in sys.argv if a.startswith("--add-width=")), None)
ADD_WIDTH = int(_add_width_arg.split("=")[1]) if _add_width_arg else None
_extend_arg = next((a for a in sys.argv if a.startswith("--extend-cap=")), None)
EXTEND_CAP = int(_extend_arg.split("=")[1]) if _extend_arg else None

# ---------------------------------------------------------------- settled control (330-334)
OPTIMAL_DT = 0.2   # not 0.4 -- see 345/346/347: dt=0.4 oscillates (not slow, genuinely wrong
                   # fixed point) at Class-IL H=4; dt=0.2 is the only value confirmed stable
                   # there AND at every depth (347), so it's used uniformly here too
SETTLE_STEP_CAP = 100
SETTLE_DELTA_TOL = 1e-4
SETTLE_PATIENCE = 3

METHODS = ["backprop", "replay", "pc"]

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, SEED_START = 2, 10
    WIDTHS = [4, 32]
    MAX_ITERS = 200
    EVAL_EVERY, LOG_EVERY = 5, 5
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = 5
    SEED_START = CFG["training"]["seed_start"]
    WIDTHS = [4, 8, 16, 32, 64]
    MAX_ITERS = CFG["training"]["max_iters_per_task"]   # full budget -- see docstring
    EVAL_EVERY = CFG["evaluation"]["eval_every"]
    LOG_EVERY = CFG["evaluation"]["log_every"]

SCENARIOS = CFG["data"]["scenarios"]
THRESHOLD = CFG["training"]["stop_threshold"]

BASE_PROTOCOL = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    n_layers=CFG["architecture"]["n_layers"],   # depth held at 1 -- see docstring
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    lr={"backprop": CFG["training"]["learning_rate"]["backprop"],
        "pc": CFG["training"]["learning_rate"]["pc"]},   # replay keeps its own METHOD_DEFAULTS lr
    stop_threshold=CFG["training"]["stop_threshold"],
    stop_patience=CFG["training"]["stop_patience"],
    max_iters_per_task=MAX_ITERS, seeds=SEEDS,
    eval_per_class=CFG["evaluation"]["eval_per_class"], eval_every=EVAL_EVERY,
    device=CFG["evaluation"]["device"],
)


def _tag(suffix):
    return suffix + ("_SMOKE" if SMOKE else "")


def figure_path(suffix):
    return _figure_path(__file__, _tag(suffix))


def array_path(scenario):
    return _array_path(__file__, _tag(scenario))


# ---------------------------------------------------------------- one run, one (method, width, seed)
def _row_for(proto, method, seed, data):
    """(retention, t2_final, cx_height, cx_step_rel, t1_steps, t2_steps, reached_t1, reached_t2,
    mean_err_t1, mean_err_t2) -- all in accuracy PERCENT except the step counts and reached flags.
    """
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    classes = sorted({c for t in tasks for c in t})
    pos = {c: i for i, c in enumerate(classes)}

    kw = {}
    if method == "pc":
        kw.update(dt=OPTIMAL_DT, steps=SETTLE_STEP_CAP,
                 stop_delta=SETTLE_DELTA_TOL, stop_patience=SETTLE_PATIENCE)
    train_step, predict = build(proto, method, seed, **kw)

    report_x, report_y = data.report_eval
    a0 = _acc(predict(report_x), report_y, classes, lmap)
    init_row = [float(np.mean([a0[pos[c]] for c in t])) for t in tasks]

    out = run_classil(
        train_step, predict, tasks, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, log_every=LOG_EVERY, device=proto.device,
        stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
        data_seed=seed, label_map=lmap,
    )
    steps = np.concatenate([[0], out["steps"]])
    curve01 = np.concatenate([[init_row], out["curves"]["argmax"]], axis=0)   # 0-1 scale
    t1, t2 = curve01[:, 0] * 100.0, curve01[:, 1] * 100.0
    switch0, switch1 = out["switches"]

    cx_step, cx_height = crossover_fn(steps, t1, t2, after=switch0)
    cx_step_rel = float(cx_step - switch0) if cx_step is not None else float("nan")
    err = mean_test_error(steps, curve01, after=switch0)   # fraction, per task

    return (float(t1[-1]), float(t2[-1]), cx_height, cx_step_rel,
           int(switch0), int(switch1 - switch0),
           bool(out["reached"][0]), bool(out["reached"][1]),
           float(err[0]) * 100.0, float(err[1]) * 100.0)


def _rows_for_seeds(proto, data, method, width, seeds):
    return [(method, width, seed) + _row_for(proto, method, seed, data) for seed in seeds]


def _seed_range():
    return list(range(SEED_START, SEED_START + SEEDS))


# ---------------------------------------------------------------- run or reload
if REPLOT and all(Path(array_path(s)).exists() for s in SCENARIOS):
    train_rows = {s: list(np.load(array_path(s), allow_pickle=True)["data"]) for s in SCENARIOS}
    print("--replot: redrawing from saved arrays, no training\n")

elif ADD_SEEDS is not None:
    if not all(Path(array_path(s)).exists() for s in SCENARIOS):
        raise RuntimeError("--add-seeds needs existing saved data to merge into -- run the full "
                          "sweep first")
    t0 = time.perf_counter()
    train_rows = {}
    for scenario in SCENARIOS:
        existing = [r for r in np.load(array_path(scenario), allow_pickle=True)["data"]
                   if r[2] not in ADD_SEEDS]
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        new_rows = []
        for width in WIDTHS:
            p = replace(proto, hidden=width)
            data = load(p)
            for method in METHODS:
                new_rows.extend(_rows_for_seeds(p, data, method, width, ADD_SEEDS))
        train_rows[scenario] = existing + new_rows
        print(f"  {scenario:10s} added seeds {ADD_SEEDS}   [{time.perf_counter() - t0:5.0f}s]")
    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(train_rows[scenario], dtype=object))
    print("saved arrays (merged)")

elif ADD_WIDTH is not None:
    if not all(Path(array_path(s)).exists() for s in SCENARIOS):
        raise RuntimeError("--add-width needs existing saved data to merge into -- run the full "
                          "sweep first")
    t0 = time.perf_counter()
    train_rows = {}
    for scenario in SCENARIOS:
        loaded = list(np.load(array_path(scenario), allow_pickle=True)["data"])
        existing = [r for r in loaded if r[1] != ADD_WIDTH]
        seeds_present = sorted(set(r[2] for r in loaded)) if loaded else _seed_range()
        proto = replace(BASE_PROTOCOL, scenario=scenario, hidden=ADD_WIDTH)
        data = load(proto)
        new_rows = []
        for method in METHODS:
            new_rows.extend(_rows_for_seeds(proto, data, method, ADD_WIDTH, seeds_present))
        train_rows[scenario] = existing + new_rows
        print(f"  {scenario:10s} added width={ADD_WIDTH}   [{time.perf_counter() - t0:5.0f}s]")
    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(train_rows[scenario], dtype=object))
    print("saved arrays (merged)")

elif EXTEND_CAP is not None:
    # Re-runs ONLY the (method, width, seed) cells that never reached threshold on task 1 or
    # task 2 within MAX_ITERS, at a raised budget -- everything else is left byte-identical.
    # A capped cell is not "converged at a lower number", it is a truncated run; extending it
    # is not the same experiment continued, it is the same cell run properly.
    if not all(Path(array_path(s)).exists() for s in SCENARIOS):
        raise RuntimeError("--extend-cap needs existing saved data to extend -- run the full "
                          "sweep first")
    t0 = time.perf_counter()
    train_rows = {}
    for scenario in SCENARIOS:
        loaded = list(np.load(array_path(scenario), allow_pickle=True)["data"])
        capped = [r for r in loaded if not r[9] or not r[10]]
        keep = [r for r in loaded if r[9] and r[10]]
        proto = replace(BASE_PROTOCOL, scenario=scenario, max_iters_per_task=EXTEND_CAP)
        redone = []
        for width in sorted({r[1] for r in capped}):
            p = replace(proto, hidden=width)
            data = load(p)
            for method in sorted({r[0] for r in capped if r[1] == width}):
                seeds = sorted({r[2] for r in capped if r[1] == width and r[0] == method})
                redone.extend(_rows_for_seeds(p, data, method, width, seeds))
        train_rows[scenario] = keep + redone
        still_capped = sum(1 for r in redone if not r[9] or not r[10])
        print(f"  {scenario:10s} re-ran {len(capped)} capped cells at max_iters={EXTEND_CAP}, "
              f"{still_capped} still capped   [{time.perf_counter() - t0:5.0f}s]")
    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(train_rows[scenario], dtype=object))
    print("saved arrays (extended)")

else:
    t0 = time.perf_counter()
    train_rows = {s: [] for s in SCENARIOS}
    for scenario in SCENARIOS:
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        for width in WIDTHS:
            p = replace(proto, hidden=width)
            data = load(p)
            for method in METHODS:
                train_rows[scenario].extend(_rows_for_seeds(p, data, method, width, _seed_range()))
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]")
    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(train_rows[scenario], dtype=object))
    print("saved arrays")


# ---------------------------------------------------------------- column indices into a row
# (method, width, seed, retention, t2_final, cx_height, cx_step_rel, t1_steps, t2_steps,
#  reached_t1, reached_t2, mean_err_t1, mean_err_t2)
COL = dict(retention=3, t2_final=4, cx_height=5, cx_step_rel=6, t1_steps=7, t2_steps=8,
          reached_t1=9, reached_t2=10, mean_err_t1=11, mean_err_t2=12)


def _vals(rows, method, width, col):
    # Seed-sorted, not insertion-order: --extend-cap patches individual (method, seed) cells
    # back in at the END of the row list rather than in their original position, which broke
    # the implicit "both methods iterate seeds in the same order" assumption paired_diff/
    # paired_sign/paired_wilcoxon all depend on. Sorting by seed here fixes pairing regardless
    # of how the rows were assembled -- box plots are unaffected since order doesn't matter there.
    d = {int(r[2]): r[col] for r in rows if r[0] == method and r[1] == width}
    return np.array([d[s] for s in sorted(d)], dtype=float)


def _cap_pct(rows, method, width, reached_col):
    v = [r[reached_col] for r in rows if r[0] == method and r[1] == width]
    return 100.0 * (1.0 - np.mean(v)) if v else float("nan")


# ---------------------------------------------------------------- console report
WIDTHS_SEEN = WIDTHS if not (REPLOT or ADD_SEEDS is not None or ADD_WIDTH is not None) else \
    sorted(set(r[1] for r in train_rows[SCENARIOS[0]]))

print("\npaired comparison (pc vs backprop control), by width -- replay shown as a benchmark:")
for scenario in SCENARIOS:
    rows = train_rows[scenario]
    print(f"\n  {scenario}")
    for width in WIDTHS_SEEN:
        bp_ret, pc_ret = _vals(rows, "backprop", width, COL["retention"]), \
                         _vals(rows, "pc", width, COL["retention"])
        bp_cx, pc_cx = _vals(rows, "backprop", width, COL["cx_height"]), \
                      _vals(rows, "pc", width, COL["cx_height"])
        rp_ret = _vals(rows, "replay", width, COL["retention"])
        rp_cx = _vals(rows, "replay", width, COL["cx_height"])
        d_ret, se_ret, _ = paired_diff(pc_ret, bp_ret)
        w_ret, l_ret, t_ret, p_ret = paired_sign(pc_ret, bp_ret)
        _, pw_ret, n_ret = paired_wilcoxon(pc_ret, bp_ret)
        d_cx, se_cx, _ = paired_diff(pc_cx, bp_cx)
        w_cx, l_cx, t_cx, p_cx = paired_sign(pc_cx, bp_cx, censored_is_best=True)
        _, pw_cx, n_cx = paired_wilcoxon(pc_cx, bp_cx)
        cap_bp = _cap_pct(rows, "backprop", width, COL["reached_t2"])
        cap_pc = _cap_pct(rows, "pc", width, COL["reached_t2"])
        print(f"    H={width:<4}  retention pc-bp {d_ret:+6.1f} +-{se_ret:4.1f}"
             f"   sign {w_ret}W-{l_ret}L-{t_ret}T p={p_ret:.3f}   wilcoxon p={pw_ret:.3f} (n={n_ret})"
             f"   replay {np.nanmean(rp_ret):5.1f}%   [cap-hit: bp {cap_bp:.0f}% pc {cap_pc:.0f}%]")
        print(f"    {'':7s}crossover {d_cx:+6.1f} +-{se_cx:4.1f}"
             f"   sign {w_cx}W-{l_cx}L-{t_cx}T p={p_cx:.3f}   wilcoxon p={pw_cx:.3f} (n={n_cx})"
             f"   replay {np.nanmean(rp_cx):5.1f}%")

# ---------------------------------------------------------------- figures
METHOD_COLOR = {"backprop": "0.35", "replay": "tab:green", "pc": "tab:orange"}
METHOD_OFFSET = {"backprop": -0.22, "replay": 0.0, "pc": 0.22}


def _box_panel(ax, rows, col, ylabel, title):
    positions = np.arange(len(WIDTHS_SEEN))
    for method in METHODS:
        data_by_width = [_vals(rows, method, w, col) for w in WIDTHS_SEEN]
        data_by_width = [d[np.isfinite(d)] for d in data_by_width]
        ax.boxplot(data_by_width, positions=positions + METHOD_OFFSET[method], widths=0.18,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor=METHOD_COLOR[method], alpha=0.6),
                  medianprops=dict(color="black"))
        # Connect each box's own median across the sweep -- see 340's identical fix; boxes alone
        # don't read as a trend at a glance. NaN gaps break the line rather than interpolating.
        medians = [float(np.median(d)) if d.size else np.nan for d in data_by_width]
        ax.plot(positions + METHOD_OFFSET[method], medians, color=METHOD_COLOR[method],
               lw=1.4, alpha=0.9, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(w) for w in WIDTHS_SEEN])
    ax.set_xlabel("hidden width H")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2, axis="y")


handles = [plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLOR[m], alpha=0.6) for m in METHODS]

fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(5.5 * len(SCENARIOS), 8), squeeze=False)
for col_i, scenario in enumerate(SCENARIOS):
    rows = train_rows[scenario]
    _box_panel(axes[0][col_i], rows, COL["retention"], "task-1 retention (%)",
              f"{scenario.replace('_', '-')}: retention vs width")
    _box_panel(axes[1][col_i], rows, COL["cx_height"], "crossover height (%)",
              f"{scenario.replace('_', '-')}: crossover vs width")
axes[0][0].legend(handles, METHODS, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(figure_path("accuracy"), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path('accuracy')}")

fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(5.5 * len(SCENARIOS), 8), squeeze=False)
for col_i, scenario in enumerate(SCENARIOS):
    rows = train_rows[scenario]
    _box_panel(axes[0][col_i], rows, COL["t2_steps"], "steps for task 2 to reach 90%",
              f"{scenario.replace('_', '-')}: task-2 training length vs width")
    _box_panel(axes[1][col_i], rows, COL["cx_step_rel"], "steps into task 2 at crossover",
              f"{scenario.replace('_', '-')}: crossover step vs width")
axes[0][0].legend(handles, METHODS, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(figure_path("speed"), dpi=130, bbox_inches="tight")
print(f"saved {figure_path('speed')}")
