"""How far do each rule's synapses travel to reach where they end up, and does wandering more
mean forgetting more?

THE QUANTITY, AND WHOSE IT IS
    [R31] Li & van Rossum measure a metabolic cost: M = sum_t |w(t) - w(t-1)|, the L1 PATH LENGTH
    each synapse actually travels, counting every reversal. Divided by |w(T) - w(0)|, how far it
    needed to travel, that is INEFFICIENCY: 1.0 is a straight line, higher is more wandering.

    It is NOT a forgetting metric and is not reported as one. It belongs to "why do the rules
    differ", which is the question left open by the A series: script 54 measured PC's update as
    0.814 aligned with backprop's on W2 and EqProp's as 0.197, yet neither rule retained more.
    Different credit assignment that buys nothing needs a description, and path length is one --
    it distinguishes a rule that goes somewhere different from one that goes to the same place
    by a worse route.

WHY IT IS SPLIT AT THE TASK SWITCH
    Total path over a whole run conflates learning task 1 with learning task 2. The interesting
    number is task 2's: how much movement does a rule spend, and how much net displacement does
    it have to show for it, while task 1 is sitting underneath being damaged. Path is therefore
    accumulated separately either side of the switch, against the net displacement of that same
    block.

WHY PER LAYER
    Scripts 42/43 established that drift in W1 is what damages task 1, and 55 that PC diverges
    from backprop at the OUTPUT layer while W1 stays backprop-like (cos 0.952 at depth 3). So
    the layers answer different questions and a whole-network average would hide both.

PRE-COMMITTED READINGS
    PC ~ backprop on W1, different on W2   -> agrees with 55; PC reconfigures the output end and
                                              leaves the part that matters alone. This is what
                                              the A-series null predicts, so it is a check on
                                              the story rather than a new claim.
    EqProp markedly higher everywhere      -> 54's near-orthogonal updates show up as wandering,
                                              i.e. it reaches a similar place by a worse route.
                                              That would make "different credit assignment" and
                                              "worse credit assignment" the same statement here.
    Inefficiency correlates with forgetting across seeds -> a mechanism worth pursuing.
    Inefficiency flat across rules         -> path length does not distinguish them, and 54's
                                              cosine is the instrument that does. Report it and
                                              stop, rather than looking for a third measure.

DEVIATIONS FROM THE PROTOCOL, STATED
    * Fixed budgets per task, as script 52, so accuracy is comparable with it.
    * Path length is a SUM OVER UPDATES, so it scales with the number of updates taken. Fixed
      budgets are what make it comparable across rules at all; under accuracy stopping a slower
      rule would accumulate more path for that reason alone. Inefficiency divides most of that
      out, but the raw path is reported too, and must not be compared across different budgets.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import (PROTOCOL, load, build, replace,
                          figure_path as _figure_path, array_path as _array_path)
from src.runner import run_classil
from src.probes import weight_path_probe
from src.metrics import metric_grid, report_grid, paired_diff, inefficiency
from src.plotting import plot_learning_curves

SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):      # noqa: F811
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):       # noqa: F811
    return _array_path(f, _tag(f, suffix))


# ---------------------------------------------------------------- settings
METHODS = ["backprop", "replay", "pc", "eqprop"]
SEEDS = 5
EVAL_EVERY = 10
COLORS = {"backprop": "tab:gray", "replay": "tab:brown",
          "pc": "tab:red", "eqprop": "tab:green"}
TASK_COLORS = ["tab:orange", "tab:blue"]

HIDDEN = int(np.load(_array_path(str(ROOT / "experiments" /
                                     "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(_array_path(str(ROOT / "experiments" /
                              "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
SETTLE_TOL, EQ_MAX_STEPS = float(z51["settle_tol"]), int(z51["eq_max_steps"])
PC_STEPS = int(z51["pc_steps"])
T = float(z51["target_steps"])
ITERS = [int(2 * T), int(1.5 * T)]

if SMOKE:
    SEEDS, ITERS = 2, [40, 30]
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} | domain-IL | {ITERS[0]}+{ITERS[1]} updates | {SEEDS} seeds")
print("  question: do the rules differ in how far their synapses travel per unit of progress?\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il", stop_threshold=None,
               max_iters_per_task=ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)
LAYERS = ["W1", "W2"]


def settle_kw(method):
    if method == "pc":
        return dict(steps=PC_STEPS)
    if method == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    steps, switches = z["steps"], list(z["switches"])
    curves = {m: z[f"argmax_{m}"] for m in METHODS}
    ineff = {m: {L: z[f"ineff_{m}_{L}"] for L in LAYERS} for m in METHODS}
    pathlen = {m: {L: z[f"path_{m}_{L}"] for L in LAYERS} for m in METHODS}
    netdisp = {m: {L: z[f"net_{m}_{L}"] for L in LAYERS} for m in METHODS}
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(base)
    curves = {m: [] for m in METHODS}
    ineff = {m: {L: [] for L in LAYERS} for m in METHODS}
    pathlen = {m: {L: [] for L in LAYERS} for m in METHODS}
    netdisp = {m: {L: [] for L in LAYERS} for m in METHODS}
    t0 = time.perf_counter()
    for m in METHODS:
        proto = replace(base, lr={m: LR[m]})
        for seed in range(SEEDS):
            tasks = proto.tasks(seed)
            lmap = proto.label_map(tasks)
            handle = {}
            train_step, predict = build(proto, m, seed, handle=handle, **settle_kw(m))
            params = handle["params"]
            wrapped, path = weight_path_probe(train_step, params)

            snap = {}

            def take(tag, _p=params, _s=snap):
                _s[tag] = {k: v.detach().clone() for k, v in _p.named().items() if v is not None}

            take("init")
            mark = {}

            def on_task_end(ti, step, _path=path, _mark=mark):
                # path accumulates from the start of the RUN, so the task-2 contribution is the
                # difference between its value at the end and at the switch. Snapshotting rather
                # than zeroing keeps the whole-run number available too.
                _mark[ti] = {k: v.detach().clone() for k, v in _path.items()}
                take(f"end{ti}")

            out = run_classil(
                wrapped, predict, tasks, data.train, data.class_idx,
                report_eval=data.report_eval, stop_eval=data.stop_eval,
                max_iters_per_task=ITERS, batch=proto.batch, eval_every=EVAL_EVERY,
                device=proto.device, stop_threshold=None, data_seed=seed,
                label_map=lmap, on_task_end=on_task_end)
            curves[m].append(out["curves"]["argmax"])

            for L in LAYERS:
                # TASK 2 ONLY: path travelled after the switch, against net displacement after
                # the switch. Both restricted to the same block, so the ratio is a property of
                # task-2 learning and not of how task 1 happened to go.
                p2 = (mark[1][L] - mark[0][L]).cpu().numpy()
                n2 = (snap["end1"][L] - snap["end0"][L]).cpu().numpy()
                r = inefficiency(p2, n2)
                ineff[m][L].append(float(np.nanmedian(r)))
                pathlen[m][L].append(float(np.abs(p2).sum()))
                netdisp[m][L].append(float(np.abs(n2).sum()))
        steps, switches = out["steps"], out["switches"]
        curves[m] = np.array(curves[m], dtype=float)
        for L in LAYERS:
            for d in (ineff, pathlen, netdisp):
                d[m][L] = np.array(d[m][L], dtype=float)
        print(f"  {m:10s} done  [{time.perf_counter() - t0:6.0f}s]")

sw = switches[0]

# ---------------------------------------------------------------- readings
# The median over synapses, not the mean: inefficiency is a ratio with a small denominator for
# synapses that barely moved, so its distribution has a long right tail and the mean tracks that
# tail rather than the typical synapse. metrics.inefficiency already returns NaN for the
# smallest denominators; the median handles the rest.
print(f"\n  TASK 2 ONLY -- median inefficiency (path / net displacement, per synapse)")
print(f"  {'rule':10s} " + " ".join(f"{L:>22s}" for L in LAYERS))
for m in METHODS:
    cells = []
    for L in LAYERS:
        v = ineff[m][L]
        if m == "backprop":
            cells.append(f"{v.mean():14.2f}         ")
        else:
            d, se, n = paired_diff(v, ineff["backprop"][L])
            cells.append(f"{v.mean():8.2f} {d:+6.2f} {n:4.1f}sem")
    print(f"  {m:10s} " + " ".join(f"{c:>22s}" for c in cells))

print(f"\n  the two ingredients, summed over synapses (task 2 only):")
print(f"  {'rule':10s} " + " ".join(f"{'path ' + L:>12s} {'net ' + L:>11s}" for L in LAYERS))
for m in METHODS:
    cells = []
    for L in LAYERS:
        cells.append(f"{pathlen[m][L].mean():12.2f} {netdisp[m][L].mean():11.2f}")
    print(f"  {m:10s} " + " ".join(cells))

grid = {m: metric_grid(steps, curves[m], sw) for m in METHODS}

print(f"\n  does wandering predict forgetting?  across seeds, within each rule:")
print(f"  {'rule':10s} {'layer':>6s} {'r(ineff, t1 kept)':>19s}   n")
for m in METHODS:
    for L in LAYERS:
        x, y = ineff[m][L], grid[m]["final_t1"]
        ok = np.isfinite(x) & np.isfinite(y)
        r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else float("nan")
        print(f"  {m:10s} {L:>6s} {r:19.3f}  {int(ok.sum()):2d}")
print("  With 5 seeds a correlation here is descriptive only -- it says whether the question is")
print("  worth a properly powered run, not whether the relationship holds. Script 60 needed 24.")

report_grid(grid, METHODS, control="backprop", primary="crossover")

# ================================================================ THE ILLUSTRATION
# One figure showing WHAT IS BEING MEASURED, before the figure showing the result. The scalar
# above is a ratio of two lengths; this draws both of them, on real weights, for one seed.
#
# HONESTY ABOUT THE PROJECTION: the path lives in 196x32 dimensions and is drawn in 2. The
# projection is PCA fitted on ALL rules' task-2 trajectories together, so the rules share one
# set of axes and can be compared; the explained variance is printed on the panel. Every NUMBER
# annotated -- path length, net displacement, their ratio -- is computed in the FULL space, not
# in the projection, so the picture can mislead about shape but not about the quantity.
ILLUS_SEED, ILLUS_EVERY = 0, 5
# BOTH layers, because the result is a DISAGREEMENT between them: PC is more efficient than
# backprop on W2 and slightly worse on W1. A figure showing only W1 would illustrate the
# quantity while hiding the finding.
print(f"\n  drawing the mechanism: {' and '.join(LAYERS)} trajectories, seed {ILLUS_SEED}, "
      f"every {ILLUS_EVERY} updates")
from src.probes import weight_trace_probe

data_i = load(base)
traces, marks = {}, {}
for m in METHODS:
    proto = replace(base, lr={m: LR[m]})
    tasks = proto.tasks(ILLUS_SEED)
    lmap = proto.label_map(tasks)
    handle = {}
    ts, pr = build(proto, m, ILLUS_SEED, handle=handle, **settle_kw(m))
    wrapped, tr = weight_trace_probe(ts, handle["params"], keys=tuple(LAYERS),
                                     every=ILLUS_EVERY)
    at_switch = {}
    run_classil(wrapped, pr, tasks, data_i.train, data_i.class_idx,
                report_eval=data_i.report_eval, stop_eval=data_i.stop_eval,
                max_iters_per_task=ITERS, batch=proto.batch, eval_every=10 ** 9,
                device=proto.device, stop_threshold=None, data_seed=ILLUS_SEED,
                label_map=lmap,
                on_task_end=lambda ti, step, _t=tr, _a=at_switch: _a.setdefault(
                    ti, len(_t[LAYERS[0]])))
    traces[m] = {L: np.stack(tr[L]) for L in LAYERS}
    marks[m] = at_switch.get(0, 0)

# BOTH BLOCKS, as two separate paths: init -> switch, then switch -> end. The scalar reported
# above is task 2 only, but drawing task 2 alone hides where it started from and how far the
# weights had already travelled to learn task 1 -- and whether the two blocks have the same
# character at all. Each block gets its own chord and its own ratio.
#
# The PCA is fitted on the WHOLE trajectory anchored at INITIALISATION, so both blocks live in
# one projection and the switch point is a real location on the picture rather than an origin
# imposed by the plotting. Each LAYER gets its own PCA: they have different dimensions and
# wildly different scales, and one shared projection would be dominated by W1 while W2
# collapsed to a dot.
full, mus, bases, evrs = {}, {}, {}, {}
for L in LAYERS:
    full[L] = {m: traces[m][L] - traces[m][L][0] for m in METHODS}
    allpts = np.concatenate([full[L][m] for m in METHODS], axis=0)
    mus[L] = allpts.mean(0)
    _, S, Vt = np.linalg.svd(allpts - mus[L], full_matrices=False)
    bases[L] = Vt[:2]
    evrs[L] = float((S[:2] ** 2).sum() / (S ** 2).sum())


def _path_net(a):
    """L1 path length and L1 net displacement of one block, in the FULL space."""
    return float(np.abs(np.diff(a, axis=0)).sum()), float(np.abs(a[-1] - a[0]).sum())

figI, axesI = plt.subplots(len(LAYERS), len(METHODS),
                           figsize=(4.0 * len(METHODS), 4.2 * len(LAYERS)), squeeze=False)
# ONE set of axis limits for every panel. A shared projection is not a shared picture: with
# per-panel autoscaling, a rule that moved half as far fills its panel just as completely and
# the eye reads the two as equivalent. Limits are fixed from all rules together, so panel area
# means the same thing everywhere.
for row, L in enumerate(LAYERS):
    proj = {m: (full[L][m] - mus[L]) @ bases[L].T for m in METHODS}
    # EACH PANEL IS SCALED SO ITS OWN init->end DISPLACEMENT HAS UNIT LENGTH. Without this,
    # EqProp's W2 excursion is several times the others' and squeezes three of the four panels
    # to a dot -- the figure then shows that EqProp moves more, which is true but is not the
    # quantity. The quantity is a RATIO of two lengths and is scale-invariant, so rescaling each
    # panel discards nothing being measured and makes the shape -- which IS the ratio -- legible
    # in all four. Absolute sizes are printed in each panel's annotation instead.
    for m in METHODS:
        P = proj[m]
        d = float(np.linalg.norm(P[-1] - P[0]))
        proj[m] = (P - P[0]) / (d if d > 0 else 1.0)
    allP = np.concatenate(list(proj.values()), axis=0)
    # np.ptp(a, ...), not a.ptp(...): numpy 2 removed the ndarray method.
    pad = 0.12 * max(np.ptp(allP, axis=0).max(), 1e-12)
    half = max(np.ptp(allP[:, 0]), np.ptp(allP[:, 1])) / 2 + pad   # equal aspect, no distortion
    cx = (allP[:, 0].min() + allP[:, 0].max()) / 2
    cy = (allP[:, 1].min() + allP[:, 1].max()) / 2
    for col, m in enumerate(METHODS):
        ax = axesI[row][col]
        P, k, A = proj[m], marks[m], full[L][m]
        # THE ROUTE, in two blocks
        for lab, Pb, alpha in (("task 1", P[:k + 1], 0.40), ("task 2", P[k:], 1.0)):
            if len(Pb) < 2:
                continue
            ax.fill(np.concatenate([Pb[:, 0], [Pb[0, 0]]]),
                    np.concatenate([Pb[:, 1], [Pb[0, 1]]]),
                    color=COLORS[m], alpha=0.09 * (1 + alpha), lw=0)
            ax.plot(Pb[:, 0], Pb[:, 1], color=COLORS[m], lw=1.7, alpha=alpha,
                    label=f"route, {lab}")
        # THREE NET DISPLACEMENTS. The third is the point: task 2 partly UNDOES task 1, so the
        # overall net (init -> end) is not the sum of the two blocks, and the gap between them
        # is retraced ground -- movement spent going somewhere and then coming back.
        p1, n1 = _path_net(A[:k + 1]) if k >= 1 else (0.0, 0.0)
        p2, n2 = _path_net(A[k:]) if len(A) - k >= 2 else (0.0, 0.0)
        pf, nf = _path_net(A)
        for (i, j), ls, c, lab in ((( 0, k), ":", "tab:blue", "net, init->switch"),
                                   ((k, -1), "--", "k", "net, switch->end"),
                                   (( 0, -1), "-", "tab:purple", "net, init->end")):
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=c, lw=1.9, ls=ls,
                    alpha=0.9, label=lab, zorder=4)
        ax.plot(P[0, 0], P[0, 1], "o", color="k", ms=7, label="init")
        ax.plot(P[k, 0], P[k, 1], "*", color="k", ms=14, label="task switch")
        ax.plot(P[-1, 0], P[-1, 1], "s", color="k", ms=7, label="end")
        r1 = p1 / n1 if n1 > 0 else float("nan")
        r2 = p2 / n2 if n2 > 0 else float("nan")
        rf = pf / nf if nf > 0 else float("nan")
        ax.set_title(f"{m if row == 0 else ''}\n{L}   path/net:  "
                     f"t1 {r1:.2f}x   t2 {r2:.2f}x   FULL {rf:.2f}x", fontsize=9.5)
        # how much of task 1's displacement did task 2 undo? n1 + n2 - nf is the L1 distance
        # travelled and then given back; 0 means the two blocks moved in unrelated directions.
        ax.annotate(f"net: {n1:.0f} + {n2:.0f} = {n1 + n2:.0f}\nbut init->end = {nf:.0f}\n"
                    f"retraced {max(n1 + n2 - nf, 0):.0f}",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.5, va="bottom",
                    color="dimgray")
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(alpha=0.25)
    axesI[row][0].set_ylabel(f"{L}   (2-D PCA, {evrs[L]:.0%} of variance,\n"
                             f"each panel scaled to unit init->end)", fontsize=9)
axesI[0][0].legend(fontsize=7.5, loc="best")
figI.suptitle(
    f"WHAT IS BEING MEASURED: each layer's route through weight space, seed {ILLUS_SEED}. "
    f"The wiggly line is the route actually taken -- faint over task 1, solid over task 2.\n"
    f"THREE net displacements are drawn on the same axes: init->switch (blue dotted), "
    f"switch->end (black dashed), and init->end (purple solid). Shaded = area between a route "
    f"and its chord, the wandering the ratio charges for.\n"
    f"The purple line is the point: task 2 partly UNDOES task 1, so the overall net is shorter "
    f"than the two blocks summed, and the difference is ground retraced -- movement spent going "
    f"somewhere and then coming back.\n"
    f"CAVEAT: the annotated ratios are [R31]'s L1 (sum of |dw|, their metabolic cost) computed "
    f"in the FULL weight space; the drawing is Euclidean in 2-D PCA holding the stated variance. "
    f"Trust the numbers, read the picture for shape.", fontsize=9.5)
figI.tight_layout()
figI.savefig(figure_path(__file__, "mechanism"), dpi=130, bbox_inches="tight")
print(f"saved {figure_path(__file__, 'mechanism')}")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, len(LAYERS) + 1, figsize=(5.0 * (len(LAYERS) + 1), 4.4))
for ax, L in zip(axes, LAYERS):
    for i, m in enumerate(METHODS):
        v = ineff[m][L]
        ax.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, "o",
                color=COLORS[m], ms=6, alpha=0.75)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=COLORS[m], lw=2.6)
    ax.axhline(1.0, color="k", lw=0.9, ls=":")
    ax.annotate("1.0 = straight line", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS, fontsize=8)
    ax.set_title(f"{L} -- median inefficiency, task 2", fontsize=10)
    ax.set_ylabel("path length / net displacement")
    ax.grid(alpha=0.25, axis="y")
ax = axes[-1]
for m in METHODS:
    ax.plot(ineff[m][LAYERS[0]], grid[m]["final_t1"], "o", color=COLORS[m], ms=7, label=m)
ax.set_xlabel(f"{LAYERS[0]} inefficiency (task 2)")
ax.set_ylabel("task 1 kept (%)")
ax.set_title("does wandering cost retention?", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.suptitle("Synaptic path efficiency [R31]. Higher = more movement per unit of progress. "
             f"Domain-IL, H={HIDDEN}, {SEEDS} seeds.", fontsize=10)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

plot_learning_curves(
    steps, curves, METHODS, figure_path(__file__, "accuracy"),
    blocks=[(0, sw, 0), (sw, steps[-1], 1)], ncols=2, task_colors=TASK_COLORS,
    task_labels=["task 1", "task 2"], crossover_after=sw,
    title=f"Accuracy for the same runs -- {base.describe()}",
    legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False))

# ---------------------------------------------------------------- save
if not SMOKE:
    np.savez(array_path(__file__), steps=steps, switches=switches,
             methods=np.array(METHODS),
             **{f"argmax_{m}": curves[m] for m in METHODS},
             **{f"ineff_{m}_{L}": ineff[m][L] for m in METHODS for L in LAYERS},
             **{f"path_{m}_{L}": pathlen[m][L] for m in METHODS for L in LAYERS},
             **{f"net_{m}_{L}": netdisp[m][L] for m in METHODS for L in LAYERS})
    print(f"saved {array_path(__file__)}")
