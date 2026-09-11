"""At a SHARED learning-rate grid (same absolute values for every rule), does backprop's
established 0.01 and PC's established 0.02 actually sit at each rule's own best point for
retention and crossover, or does a direct comparison justify different values -- and what does
that cost in training length (steps to reach crossover, steps for task 2 to reach 90%)?

Report series 2, decade 340. First of the three-part BP-vs-PC comparison (narrative_plan.md's
S2/340 series) -- run BEFORE the width (341) and depth (342) sweeps on purpose: those hold lr at
"the established per-rule default", and if this script shows that default is not actually each
rule's best operating point for retention/crossover, 341/342 would need rebuilding on top of a
corrected value anyway.

WHY NOT ONE lr PER RULE (the historical 0.01/0.02 pairing): a learning rate is an internal
parameter of each rule's own update mechanism, not a shared external stimulus -- so forcing them
onto DIFFERENT numbers by an unexamined convention isn't more controlled, it's just unexamined.
This script uses the SAME absolute grid for every rule and lets each rule's own response curve say
where its best point is, rather than assuming the inherited pairing is right. The historical 0.02
for PC was grid-searched in 102 for LEARNING SPEED, a different objective -- never for
retention/crossover, which is what this project actually reports (see 300_series_log.md).

REPLAY IS SWEPT TOO (second version of this script -- the first held it at a fixed reference lr).
That was wrong on its own terms: replay's train_step IS backprop's update rule plus a buffer, so
its lr is in the SAME units as backprop's, unlike PC's settled error signal. There was no
principled reason to treat it differently, and drawing it as a flat line while backprop/pc were
swept looked like it meant something it didn't.

LR_GRID = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16] -- extended (from an original 4-point run) after
that run showed BOTH rules' response curves still moving at the top of the range (class-il
crossover) and several paired comparisons sitting exactly at the 5-seed sign test's ceiling
(p=0.062, the best obtainable when all 5 seeds agree) -- not enough to tell a real effect from an
unlucky-but-unanimous small sample. 0.08 and 0.16 extend the top; SEEDS below extends the sample.

PC's settle dynamics are FIXED at the settled control (330-334: dt=0.4, adaptive stop,
stop_delta=1e-4, stop_patience=3, steps=100 as a generous cap) -- not re-swept here, already
settled.

FULL max_iters_per_task budget (config_300.yaml's 5000, not trimmed) -- this sweep exists partly
to find where lr is "too low to reach 90% in reasonable time", so the budget defining "reasonable"
must be the real established one, not a shortened one that would manufacture false failures. Cells
that hit this cap are FLAGGED (`reached`), not silently included as if they were genuine
matched-competence reads -- a cell where a rule hit the cap on most seeds is measuring "how far did
it get before running out of budget", a different and weaker quantity than retention at matched
competence, and the plots say so directly (cap-hit % annotated on every box that isn't 0%).

SEEDS = 10 (10-19), the full established count -- raised from an initial 5 after that run showed
several comparisons sitting at the 5-seed sign test's structural ceiling.

BACKFILL, NOT A SEPARATE MERGE FLAG: earlier versions of this script had --add-seeds=LO-HI and
--add-lr=X flags (330's dt-merge pattern). Replacing them here with a single mechanism: for every
(method, lr, seed) cell the current LR_GRID/SEEDS ask for, reuse it if it is already saved,
compute it if not. This does everything the two flags did (extend seeds, extend the grid) with one
code path instead of three, and it is also what correctly handles replay's redesign: its old rows
(saved at a single fixed lr, under the first version of this script) do not match any (method, lr)
pair in the new grid, so they are dropped and replay is recomputed fresh across the whole grid,
while backprop/pc's already-valid cells at the original 4 lr values and 5 seeds are kept as-is.

Metrics, per seed, computed directly from the run's own (ragged) curve -- matched-competence
stopping means runs finish at different absolute steps, so there is no single grid to pool into
(see 332's convention): retention (task-1 accuracy at the seed's own stop point), task-2 final
accuracy (so retention is never read alongside "learned less" without saying so), crossover
height and the step it occurs at (relative to the task switch), steps for task 1 and task 2 to
each reach 90%, and S&B's mean test error (src.metrics.mean_test_error) for both tasks. Paired
statistics (paired_diff, paired_sign, paired_wilcoxon; control=backprop) at every lr, on
retention and crossover height.

The H=32/depth=1 baseline cell here (lr=0.01 backprop, lr=0.02 pc) is NOT reused from 310/332's
saved arrays -- it is trained fresh, and agreement with those scripts is a cheap correctness
check (330's double-remap bug was found exactly this way), not redundant compute.

Three figures:
    340_lr_sweep_accuracy.png  retention, crossover height -- box-and-whisker per lr, one box per
        method (backprop, replay, pc). Y-axis is FIXED per metric row, shared across BOTH scenario
        columns (computed from that metric's actual data range, not 0-100) -- so Domain-IL and
        Class-IL panels are on a directly comparable scale rather than each auto-scaling to
        whatever happens to be in it (which, for Class-IL retention, previously meant the panel
        auto-scaled around replay's distant reference line while every rule's actual, near-zero
        variation sat invisibly in a one-pixel band at the bottom). Cap-hit % annotated above any
        box where a rule didn't reach 90% on every seed. Each method's best lr (by mean) marked.
    340_lr_sweep_speed.png     steps for task 2 to reach 90%, crossover step -- same layout.
    340_lr_sweep_diff.png      pc-vs-backprop paired difference (mean +- SEM) against lr, retention
        and crossover, both scenarios -- what the box-and-whisker overlap only lets you eyeball.
        Replay-vs-backprop shown alongside, fainter, as a "here is what a real intervention buys on
        this same scale" anchor.

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

# ---------------------------------------------------------------- settled control (330-334)
OPTIMAL_DT = 0.4
SETTLE_STEP_CAP = 100
SETTLE_DELTA_TOL = 1e-4
SETTLE_PATIENCE = 3

METHODS = ["backprop", "replay", "pc"]

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, SEED_START = 2, 10
    LR_GRID = [0.01, 0.02]
    MAX_ITERS = 200
    EVAL_EVERY, LOG_EVERY = 5, 5
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = 10
    SEED_START = CFG["training"]["seed_start"]
    LR_GRID = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]   # shared grid, every rule -- see docstring
    MAX_ITERS = CFG["training"]["max_iters_per_task"]   # full budget, not trimmed -- see docstring
    EVAL_EVERY = CFG["evaluation"]["eval_every"]
    LOG_EVERY = CFG["evaluation"]["log_every"]

SCENARIOS = CFG["data"]["scenarios"]
THRESHOLD = CFG["training"]["stop_threshold"]

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


def _seed_range():
    return list(range(SEED_START, SEED_START + SEEDS))


# ---------------------------------------------------------------- one run, one (method, lr, seed)
def _row_for(proto, method, lr, seed, data):
    """(retention, t2_final, cx_height, cx_step_rel, t1_steps, t2_steps, reached_t1, reached_t2,
    mean_err_t1, mean_err_t2) -- all in accuracy PERCENT except the step counts and reached flags.
    """
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    classes = sorted({c for t in tasks for c in t})
    pos = {c: i for i, c in enumerate(classes)}

    kw = dict(lr=lr)
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


# ---------------------------------------------------------------- run or reload (backfill only
# what's missing -- see docstring for why this replaced the earlier --add-seeds/--add-lr flags)
if REPLOT and all(Path(array_path(s)).exists() for s in SCENARIOS):
    train_rows = {s: list(np.load(array_path(s), allow_pickle=True)["data"]) for s in SCENARIOS}
    print("--replot: redrawing from saved arrays, no training\n")
else:
    t0 = time.perf_counter()
    train_rows = {}
    for scenario in SCENARIOS:
        existing = []
        if Path(array_path(scenario)).exists():
            loaded = list(np.load(array_path(scenario), allow_pickle=True)["data"])
            # drop any row whose lr is not in the CURRENT grid -- catches replay's old rows from
            # the fixed-reference design (a single lr not in LR_GRID), so they get recomputed
            # fresh across the real grid rather than silently kept under the old scheme.
            existing = [r for r in loaded if r[1] in LR_GRID]
        have = {(r[0], r[1], r[2]) for r in existing}
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        data = load(proto)
        new_rows = []
        n_skipped = 0
        for method in METHODS:
            for lr in LR_GRID:
                for seed in _seed_range():
                    if (method, lr, seed) in have:
                        n_skipped += 1
                        continue
                    new_rows.append((method, lr, seed) + _row_for(proto, method, lr, seed, data))
        train_rows[scenario] = existing + new_rows
        print(f"  {scenario:10s} done -- {len(new_rows)} new, {n_skipped} reused"
             f"   [{time.perf_counter() - t0:5.0f}s]")
    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(train_rows[scenario], dtype=object))
    print("saved arrays")


# ---------------------------------------------------------------- column indices into a row
# (method, lr, seed, retention, t2_final, cx_height, cx_step_rel, t1_steps, t2_steps,
#  reached_t1, reached_t2, mean_err_t1, mean_err_t2)
COL = dict(retention=3, t2_final=4, cx_height=5, cx_step_rel=6, t1_steps=7, t2_steps=8,
          reached_t1=9, reached_t2=10, mean_err_t1=11, mean_err_t2=12)


def _vals(rows, method, lr, col):
    return np.array([r[col] for r in rows if r[0] == method and r[1] == lr], dtype=float)


def _cap_pct(rows, method, lr, reached_col):
    v = [r[reached_col] for r in rows if r[0] == method and r[1] == lr]
    return 100.0 * (1.0 - np.mean(v)) if v else float("nan")


# ---------------------------------------------------------------- console report: paired stats
print("\npaired comparison (pc vs backprop control), by lr:")
for scenario in SCENARIOS:
    rows = train_rows[scenario]
    print(f"\n  {scenario}")
    for lr in LR_GRID:
        bp_ret, pc_ret = _vals(rows, "backprop", lr, COL["retention"]), \
                         _vals(rows, "pc", lr, COL["retention"])
        bp_cx, pc_cx = _vals(rows, "backprop", lr, COL["cx_height"]), \
                      _vals(rows, "pc", lr, COL["cx_height"])
        d_ret, se_ret, _ = paired_diff(pc_ret, bp_ret)
        w_ret, l_ret, t_ret, p_ret = paired_sign(pc_ret, bp_ret)
        _, pw_ret, n_ret = paired_wilcoxon(pc_ret, bp_ret)
        d_cx, se_cx, _ = paired_diff(pc_cx, bp_cx)
        w_cx, l_cx, t_cx, p_cx = paired_sign(pc_cx, bp_cx, censored_is_best=True)
        _, pw_cx, n_cx = paired_wilcoxon(pc_cx, bp_cx)
        cap_bp = _cap_pct(rows, "backprop", lr, COL["reached_t2"])
        cap_pc = _cap_pct(rows, "pc", lr, COL["reached_t2"])
        cap_rp = _cap_pct(rows, "replay", lr, COL["reached_t2"])
        print(f"    lr={lr:<6}  retention pc-bp {d_ret:+6.1f} +-{se_ret:4.1f}"
             f"   sign {w_ret}W-{l_ret}L-{t_ret}T p={p_ret:.3f}   wilcoxon p={pw_ret:.3f} (n={n_ret})"
             f"   [cap-hit: bp {cap_bp:.0f}% pc {cap_pc:.0f}% replay {cap_rp:.0f}%]")
        print(f"    {'':13s}crossover {d_cx:+6.1f} +-{se_cx:4.1f}"
             f"   sign {w_cx}W-{l_cx}L-{t_cx}T p={p_cx:.3f}   wilcoxon p={pw_cx:.3f} (n={n_cx})")

# ---------------------------------------------------------------- figures
METHOD_COLOR = {"backprop": "0.35", "replay": "tab:green", "pc": "tab:orange"}
METHOD_OFFSET = {"backprop": -0.22, "replay": 0.0, "pc": 0.22}


def _shared_ylim(col, margin_frac=0.08):
    """One y-range per metric, spanning its ACTUAL data across every scenario/method/lr -- not
    0-100 -- so the two scenario columns share a directly comparable scale (see docstring)."""
    vals = np.concatenate([_vals(train_rows[s], m, lr, col)
                           for s in SCENARIOS for m in METHODS for lr in LR_GRID])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * margin_frac if hi > lo else 1.0
    return lo - pad, hi + pad


def _box_panel(ax, rows, col, ylabel, title, ylim=None, cap_col=None, mark_best=True):
    positions = np.arange(len(LR_GRID))
    for method in METHODS:
        series = [_vals(rows, method, lr, col) for lr in LR_GRID]
        finite = [d[np.isfinite(d)] for d in series]
        ax.boxplot(finite, positions=positions + METHOD_OFFSET[method], widths=0.18,
                  patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor=METHOD_COLOR[method], alpha=0.6),
                  medianprops=dict(color="black"))
        # Connect each box's own median across the sweep -- boxes alone don't read as a trend at
        # a glance, and the median (not the mean used for "best") is what the box already shows,
        # so the line extends the same statistic rather than introducing a second one. NaN gaps
        # (e.g. a fully-censored lr point) break the line rather than being interpolated over.
        medians = [float(np.median(d)) if d.size else np.nan for d in finite]
        ax.plot(positions + METHOD_OFFSET[method], medians, color=METHOD_COLOR[method],
               lw=1.4, alpha=0.9, zorder=3)
        if mark_best:
            means = [d.mean() if d.size else np.nan for d in finite]
            if np.any(np.isfinite(means)):
                best_i = int(np.nanargmax(means))
                y = finite[best_i].max() if finite[best_i].size else 0.0
                ax.annotate("best", (positions[best_i] + METHOD_OFFSET[method], y),
                          fontsize=6, color=METHOD_COLOR[method], xytext=(0, 4),
                          textcoords="offset points", ha="center", fontweight="bold")
        if cap_col is not None:
            for i, lr in enumerate(LR_GRID):
                cap = _cap_pct(rows, method, lr, cap_col)
                if cap > 0:
                    y = finite[i].max() if finite[i].size else (ylim[1] if ylim else 0.0)
                    ax.annotate(f"{cap:.0f}%cap", (positions[i] + METHOD_OFFSET[method], y),
                              fontsize=5.5, color="crimson", xytext=(0, 13),
                              textcoords="offset points", ha="center", rotation=90)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(lr) for lr in LR_GRID])
    ax.set_xlabel("learning rate (shared grid)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    if ylim is not None:
        ax.set_ylim(*ylim)


handles = [plt.Rectangle((0, 0), 1, 1, facecolor=METHOD_COLOR[m], alpha=0.6) for m in METHODS]

# figure A: accuracy (retention, crossover height) -- shared y-range per row, cap-hit annotated
ret_ylim = _shared_ylim(COL["retention"])
cx_ylim = _shared_ylim(COL["cx_height"])
fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(5.5 * len(SCENARIOS), 8.5), squeeze=False)
for col_i, scenario in enumerate(SCENARIOS):
    rows = train_rows[scenario]
    _box_panel(axes[0][col_i], rows, COL["retention"], "task-1 retention (%)",
              f"{scenario.replace('_', '-')}: retention vs lr", ylim=ret_ylim,
              cap_col=COL["reached_t2"])
    _box_panel(axes[1][col_i], rows, COL["cx_height"], "crossover height (%)",
              f"{scenario.replace('_', '-')}: crossover vs lr", ylim=cx_ylim,
              cap_col=COL["reached_t2"])
axes[0][0].legend(handles, METHODS, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(figure_path("accuracy"), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path('accuracy')}")

# figure B: training length (task-2 steps to 90%, crossover step relative to switch)
steps_ylim = _shared_ylim(COL["t2_steps"])
cxstep_ylim = _shared_ylim(COL["cx_step_rel"])
fig, axes = plt.subplots(2, len(SCENARIOS), figsize=(5.5 * len(SCENARIOS), 8.5), squeeze=False)
for col_i, scenario in enumerate(SCENARIOS):
    rows = train_rows[scenario]
    _box_panel(axes[0][col_i], rows, COL["t2_steps"], "steps for task 2 to reach 90%",
              f"{scenario.replace('_', '-')}: task-2 training length vs lr", ylim=steps_ylim,
              cap_col=COL["reached_t2"], mark_best=False)
    _box_panel(axes[1][col_i], rows, COL["cx_step_rel"], "steps into task 2 at crossover",
              f"{scenario.replace('_', '-')}: crossover step vs lr", ylim=cxstep_ylim,
              cap_col=COL["reached_t2"], mark_best=False)
axes[0][0].legend(handles, METHODS, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(figure_path("speed"), dpi=130, bbox_inches="tight")
print(f"saved {figure_path('speed')}")

# figure C: paired difference vs lr -- what the box overlap in figure A only lets you eyeball
DIFF_METRICS = [("retention", COL["retention"], False), ("cx_height", COL["cx_height"], True)]


def _diff_series(rows, col, control, treatment, censored):
    means, sems, ps = [], [], []
    for lr in LR_GRID:
        t, c = _vals(rows, treatment, lr, col), _vals(rows, control, lr, col)
        m, se, _ = paired_diff(t, c)
        _, _, _, p = paired_sign(t, c, censored_is_best=censored)
        means.append(m); sems.append(se); ps.append(p)
    return np.array(means), np.array(sems), np.array(ps)


fig, axes = plt.subplots(len(DIFF_METRICS), len(SCENARIOS),
                        figsize=(5.5 * len(SCENARIOS), 4.2 * len(DIFF_METRICS)), squeeze=False)
for col_i, scenario in enumerate(SCENARIOS):
    rows = train_rows[scenario]
    for row_i, (name, col, censored) in enumerate(DIFF_METRICS):
        ax = axes[row_i][col_i]
        ax.axhline(0, color="black", lw=0.8)
        m_pc, se_pc, p_pc = _diff_series(rows, col, "backprop", "pc", censored)
        ax.errorbar(LR_GRID, m_pc, yerr=se_pc, marker="o", color="tab:orange", capsize=3,
                   label="pc - backprop")
        for x, y, p in zip(LR_GRID, m_pc, p_pc):
            if p <= 0.1:
                ax.annotate(f"p={p:.3f}", (x, y), fontsize=6, color="tab:orange",
                          xytext=(0, 8), textcoords="offset points", ha="center")
        m_rp, se_rp, _ = _diff_series(rows, col, "backprop", "replay", censored)
        ax.errorbar(LR_GRID, m_rp, yerr=se_rp, marker="s", color="tab:green", alpha=0.5,
                   capsize=3, label="replay - backprop", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(LR_GRID); ax.set_xticklabels([str(v) for v in LR_GRID], fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("learning rate (shared grid)")
        ax.set_ylabel(f"{name} difference (pp)")
        ax.set_title(f"{scenario.replace('_', '-')}: {name} difference vs lr", fontsize=9)
        ax.grid(alpha=0.2)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(figure_path("diff"), dpi=130, bbox_inches="tight")
print(f"saved {figure_path('diff')}")
