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
