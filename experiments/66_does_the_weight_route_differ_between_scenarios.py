"""Does the route through weight space differ between Class-IL and Domain-IL?

WHY THE SCENARIOS SHOULD BE COMPARED IN WEIGHT SPACE, NOT JUST IN ACCURACY
    The two scenarios forget for different reasons. Class-IL forgetting is dominated by
    OUTPUT-LAYER SUPPRESSION -- absent classes are pushed down until they cannot win an argmax
    (42/43: NCM holds ~80% while argmax reads 0.2%). Domain-IL cannot suppress at all, because
    every output unit is a target for some class, which leaves representation drift.

    Those are claims about WHERE the damage lands, and they have only ever been tested through
    accuracy and probes. The weight path is a direct view: if Class-IL forgetting is the output
    layer being rewritten, W2's route should differ sharply between scenarios while W1's does
    not. Script 65 measured Domain-IL only, so the comparison has never been made.

THE FULL-RUN RATIO, WHICH 65 DID NOT REPORT
    65 reported path/net for TASK 2 ALONE. That number cannot see task 2 undoing task 1: each
    block can be individually direct while the run as a whole doubles back. 65's own mechanism
    figure showed exactly that -- backprop's W1 blocks displace 47 and 43 but init->end is only
    52, so 38 units are RETRACED, and the full-run ratio (2.86x) is far worse than either block
    (1.77x, 1.53x). Retraced distance is forgetting expressed in weight-space units rather than
    in accuracy, so it is reported here as a first-class quantity:

        r_t1    path/net over task 1        how directly the first solution was found
        r_t2    path/net over task 2        65's number, kept for comparability
        r_full  path/net over the WHOLE run the one that sees task 2 undoing task 1
        retraced (net_t1 + net_t2 - net_full) / (net_t1 + net_t2)
                                            the fraction of all displacement given back

PRE-COMMITTED READINGS
    W2 route differs between scenarios, W1 does not  -> confirms the suppression/drift split
        from the weight side, independently of the NCM probe. The cleanest possible version of
        the 42/43 story.
    Both layers differ                               -> the scenarios differ globally and
        "suppression vs drift" is too clean a description of what separates them.
    Neither differs                                  -> the scenarios produce the same weight
        motion and their different accuracy behaviour is a readout effect, which would be a
        strong and surprising claim about the argmax rather than about learning.
    PC's W2 advantage (65: 1.70 vs backprop's 3.28 in Domain-IL) should be LARGER in Class-IL
        if it is really about the output layer, since that is where Class-IL does its damage.

DEVIATIONS FROM THE PROTOCOL, STATED
    * Both scenarios in one script, which is the comparison. Everything else is held fixed.
    * Fixed budgets per task, as 52 and 65, so accuracy stays comparable with them.
    * EqProp included despite the cost: 65 found it the extreme case (4.56/8.85 inefficiency),
      so dropping it would remove the range the comparison is read against.
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
from src.probes import weight_path_probe, weight_trace_probe
from src.metrics import metric_grid, report_grid, paired_diff

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
SCENARIOS = ["domain_il", "class_il"]
LAYERS = ["W1", "W2"]
SEEDS, EVAL_EVERY = 5, 10
ILLUS_SEED, ILLUS_EVERY = 0, 5
COLORS = {"backprop": "tab:gray", "replay": "tab:brown",
          "pc": "tab:red", "eqprop": "tab:green"}

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

print(f"H = {HIDDEN} | {' vs '.join(SCENARIOS)} | {ITERS[0]}+{ITERS[1]} updates | {SEEDS} seeds")
print("  question: does the weight-space route differ between the two scenarios?\n")


def settle_kw(m):
    if m == "pc":
        return dict(steps=PC_STEPS)
    if m == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


def base_for(scn):
    return replace(PROTOCOL, hidden=HIDDEN, scenario=scn, stop_threshold=None,
                   max_iters_per_task=ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)


def _pn(a):
    """L1 path length and L1 net displacement of a block, in the full weight space."""
    return float(np.abs(np.diff(a, axis=0)).sum()), float(np.abs(a[-1] - a[0]).sum())


# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()
KEYS = ["r_t1", "r_t2", "r_full", "retraced"]

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    stat = {(s, m, L, k): z[f"{k}_{s}_{m}_{L}"]
            for s in SCENARIOS for m in METHODS for L in LAYERS for k in KEYS}
    curves = {(s, m): z[f"argmax_{s}_{m}"] for s in SCENARIOS for m in METHODS}
    steps = {s: z[f"steps_{s}"] for s in SCENARIOS}
    switches = {s: list(z[f"switches_{s}"]) for s in SCENARIOS}
    print("--replot: aggregate from saved arrays; illustration recomputed\n")
else:
    stat, curves, steps, switches = {}, {}, {}, {}
    t0 = time.perf_counter()
    for scn in SCENARIOS:
        base = base_for(scn)
        data = load(base)
        for m in METHODS:
            proto = replace(base, lr={m: LR[m]})
            rows = {(L, k): [] for L in LAYERS for k in KEYS}
            cur = []
            for seed in range(SEEDS):
                tasks = proto.tasks(seed)
                lmap = proto.label_map(tasks)
                handle = {}
                ts, pr = build(proto, m, seed, handle=handle, **settle_kw(m))
                params = handle["params"]
                wrapped, path = weight_path_probe(ts, params)
                snap, mark = {"init": {k: v.detach().clone()
                                       for k, v in params.named().items() if v is not None}}, {}

                def on_task_end(ti, step, _p=params, _pa=path, _s=snap, _mk=mark):
                    _s[f"end{ti}"] = {k: v.detach().clone()
                                      for k, v in _p.named().items() if v is not None}
                    _mk[ti] = {k: v.detach().clone() for k, v in _pa.items()}

                out = run_classil(
                    wrapped, pr, tasks, data.train, data.class_idx,
                    report_eval=data.report_eval, stop_eval=data.stop_eval,
                    max_iters_per_task=ITERS, batch=proto.batch, eval_every=EVAL_EVERY,
                    device=proto.device, stop_threshold=None, data_seed=seed,
                    label_map=lmap, on_task_end=on_task_end)
                cur.append(out["curves"]["argmax"])
                for L in LAYERS:
                    p1 = mark[0][L].cpu().numpy()
                    p2 = (mark[1][L] - mark[0][L]).cpu().numpy()
                    n1 = (snap["end0"][L] - snap["init"][L]).cpu().numpy()
                    n2 = (snap["end1"][L] - snap["end0"][L]).cpu().numpy()
                    nf = (snap["end1"][L] - snap["init"][L]).cpu().numpy()
                    a1, a2 = np.abs(p1).sum(), np.abs(p2).sum()
                    b1, b2, bf = np.abs(n1).sum(), np.abs(n2).sum(), np.abs(nf).sum()
                    rows[(L, "r_t1")].append(a1 / b1 if b1 > 0 else np.nan)
                    rows[(L, "r_t2")].append(a2 / b2 if b2 > 0 else np.nan)
                    rows[(L, "r_full")].append((a1 + a2) / bf if bf > 0 else np.nan)
                    rows[(L, "retraced")].append(
                        (b1 + b2 - bf) / (b1 + b2) if (b1 + b2) > 0 else np.nan)
            curves[(scn, m)] = np.array(cur, dtype=float)
            for L in LAYERS:
                for k in KEYS:
                    stat[(scn, m, L, k)] = np.array(rows[(L, k)], dtype=float)
            steps[scn], switches[scn] = out["steps"], out["switches"]
            print(f"  {scn:10s} {m:10s} done  [{time.perf_counter() - t0:6.0f}s]")

# ---------------------------------------------------------------- readings
for L in LAYERS:
    print(f"\n  {L}  --  path / net displacement, paired against backprop WITHIN each scenario")
    print(f"  {'':10s} " + " ".join(f"{k:>22s}" for k in KEYS))
    for scn in SCENARIOS:
        print(f"  {scn}")
        for m in METHODS:
            cells = []
            for k in KEYS:
                v = stat[(scn, m, L, k)]
                if m == "backprop":
                    cells.append(f"{np.nanmean(v):12.3f}          ")
                else:
                    d, se, n = paired_diff(v, stat[(scn, "backprop", L, k)])
                    cells.append(f"{np.nanmean(v):8.3f} {d:+7.3f} {n:4.1f}s")
            print(f"    {m:8s} " + " ".join(f"{c:>22s}" for c in cells))

print(f"\n  DOES THE SCENARIO CHANGE THE ROUTE?  class_il - domain_il, per rule, unpaired")
print(f"  (seeds share a split within a scenario but the scenarios have different out_dim,")
print(f"   so these are group means, not paired differences.)")
for L in LAYERS:
    print(f"    {L}")
    for m in METHODS:
        a = np.nanmean(stat[("class_il", m, L, "r_full")])
        b = np.nanmean(stat[("domain_il", m, L, "r_full")])
        ra = np.nanmean(stat[("class_il", m, L, "retraced")])
        rb = np.nanmean(stat[("domain_il", m, L, "retraced")])
        print(f"      {m:10s} r_full {b:6.2f} -> {a:6.2f}  ({a - b:+6.2f})   "
              f"retraced {rb:5.1%} -> {ra:5.1%}")

for scn in SCENARIOS:
    print(f"\n{'=' * 70}\n  metric grid, {scn}\n{'=' * 70}")
    report_grid({m: metric_grid(steps[scn], curves[(scn, m)], switches[scn][0])
                 for m in METHODS}, METHODS, control="backprop", primary="crossover")

# ---------------------------------------------------------------- the illustration
# One seed's actual route, for BOTH scenarios and BOTH layers, so the scenario difference can be
# seen rather than only tabulated. Four rows: domain W1, domain W2, class W1, class W2.
print(f"\n  drawing the mechanism: seed {ILLUS_SEED}, both scenarios, both layers")
traces, marks = {}, {}
for scn in SCENARIOS:
    base = base_for(scn)
    data_i = load(base)
    for m in METHODS:
        proto = replace(base, lr={m: LR[m]})
        tasks = proto.tasks(ILLUS_SEED)
        lmap = proto.label_map(tasks)
        handle = {}
        ts, pr = build(proto, m, ILLUS_SEED, handle=handle, **settle_kw(m))
        wrapped, tr = weight_trace_probe(ts, handle["params"], keys=tuple(LAYERS),
                                         every=ILLUS_EVERY)
        at = {}
        run_classil(wrapped, pr, tasks, data_i.train, data_i.class_idx,
                    report_eval=data_i.report_eval, stop_eval=data_i.stop_eval,
                    max_iters_per_task=ITERS, batch=proto.batch, eval_every=10 ** 9,
                    device=proto.device, stop_threshold=None, data_seed=ILLUS_SEED,
                    label_map=lmap,
                    on_task_end=lambda ti, step, _t=tr, _a=at: _a.setdefault(
                        ti, len(_t[LAYERS[0]])))
        traces[(scn, m)] = {L: np.stack(tr[L]) for L in LAYERS}
        marks[(scn, m)] = at.get(0, 0)

rows_spec = [(s, L) for s in SCENARIOS for L in LAYERS]
figI, axesI = plt.subplots(len(rows_spec), len(METHODS),
                           figsize=(4.0 * len(METHODS), 4.0 * len(rows_spec)), squeeze=False)
for row, (scn, L) in enumerate(rows_spec):
    # own PCA per row: layers differ in dimension and scale, scenarios differ in out_dim
    fullt = {m: traces[(scn, m)][L] - traces[(scn, m)][L][0] for m in METHODS}
    allpts = np.concatenate([fullt[m] for m in METHODS], axis=0)
    mu = allpts.mean(0)
    _, S, Vt = np.linalg.svd(allpts - mu, full_matrices=False)
    basis, evr = Vt[:2], float((S[:2] ** 2).sum() / (S ** 2).sum())
    proj = {}
    for m in METHODS:                       # each panel scaled to unit init->end; ratio is
        P = (fullt[m] - mu) @ basis.T       # scale-invariant, so nothing measured is lost
        d = float(np.linalg.norm(P[-1] - P[0]))
        proj[m] = (P - P[0]) / (d if d > 0 else 1.0)
    allP = np.concatenate(list(proj.values()), axis=0)
    pad = 0.12 * max(np.ptp(allP, axis=0).max(), 1e-12)
    half = max(np.ptp(allP[:, 0]), np.ptp(allP[:, 1])) / 2 + pad
    cx = (allP[:, 0].min() + allP[:, 0].max()) / 2
    cy = (allP[:, 1].min() + allP[:, 1].max()) / 2
    for col, m in enumerate(METHODS):
        ax = axesI[row][col]
        P, k, A = proj[m], marks[(scn, m)], fullt[m]
        for lab, Pb, alpha in (("task 1", P[:k + 1], 0.40), ("task 2", P[k:], 1.0)):
            if len(Pb) < 2:
                continue
            ax.fill(np.concatenate([Pb[:, 0], [Pb[0, 0]]]),
                    np.concatenate([Pb[:, 1], [Pb[0, 1]]]),
                    color=COLORS[m], alpha=0.09 * (1 + alpha), lw=0)
            ax.plot(Pb[:, 0], Pb[:, 1], color=COLORS[m], lw=1.7, alpha=alpha,
                    label=f"route, {lab}")
        for (i, j), ls, c, lab in (((0, k), ":", "tab:blue", "net, init->switch"),
                                   ((k, -1), "--", "k", "net, switch->end"),
                                   ((0, -1), "-", "tab:purple", "net, init->end")):
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], color=c, lw=1.9, ls=ls,
                    alpha=0.9, label=lab, zorder=4)
        ax.plot(P[0, 0], P[0, 1], "o", color="k", ms=7, label="init")
        ax.plot(P[k, 0], P[k, 1], "*", color="k", ms=14, label="task switch")
        ax.plot(P[-1, 0], P[-1, 1], "s", color="k", ms=7, label="end")
        p1, n1 = _pn(A[:k + 1]) if k >= 1 else (0.0, 0.0)
        p2, n2 = _pn(A[k:]) if len(A) - k >= 2 else (0.0, 0.0)
        pf, nf = _pn(A)
        ax.set_title((f"{m}\n" if row == 0 else "")
                     + f"t1 {p1 / n1 if n1 else np.nan:.2f}x  t2 "
                       f"{p2 / n2 if n2 else np.nan:.2f}x  FULL {pf / nf if nf else np.nan:.2f}x",
                     fontsize=9.5)
        ax.annotate(f"retraced {max(n1 + n2 - nf, 0) / max(n1 + n2, 1e-9):.0%}",
                    xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8, color="dimgray")
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.25)
    axesI[row][0].set_ylabel(f"{scn}\n{L}  ({evr:.0%} var, unit init->end)", fontsize=9)
axesI[0][0].legend(fontsize=7, loc="best")
figI.suptitle(
    "WHAT IS BEING MEASURED, AND HOW THE SCENARIOS DIFFER: each layer's route through weight "
    f"space, seed {ILLUS_SEED}.\nWiggly line = route taken, faint over task 1 and solid over "
    "task 2. Three net displacements per panel: init->switch (blue dotted), switch->end (black "
    "dashed), init->end (purple solid).\nShaded = area between a route and its chord. "
    "'retraced' = the fraction of all displacement given back, i.e. forgetting measured in "
    "weight-space distance rather than in accuracy.\nCAVEAT: ratios are [R31]'s L1 in the FULL "
    "weight space; the drawing is Euclidean in 2-D PCA, each panel scaled to unit init->end.",
    fontsize=9.5)
figI.tight_layout()
figI.savefig(figure_path(__file__, "mechanism"), dpi=120, bbox_inches="tight")
print(f"saved {figure_path(__file__, 'mechanism')}")

# ---------------------------------------------------------------- aggregate figure
fig, axes = plt.subplots(len(LAYERS), 2, figsize=(13.5, 4.4 * len(LAYERS)), squeeze=False)
for row, L in enumerate(LAYERS):
    for col, k in enumerate(["r_full", "retraced"]):
        ax = axes[row][col]
        w = 0.35
        for si, scn in enumerate(SCENARIOS):
            xs = np.arange(len(METHODS)) + (si - 0.5) * w
            mu = [np.nanmean(stat[(scn, m, L, k)]) for m in METHODS]
            se = [np.nanstd(stat[(scn, m, L, k)], ddof=1)
                  / np.sqrt(max(np.isfinite(stat[(scn, m, L, k)]).sum(), 1)) for m in METHODS]
            ax.bar(xs, mu, w, yerr=se, capsize=3, label=scn,
                   color=["tab:cyan", "tab:olive"][si], alpha=0.9,
                   edgecolor=[COLORS[m] for m in METHODS], lw=1.6)
        ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, fontsize=9)
        ax.set_title(f"{L} — {'full-run path/net' if k == 'r_full' else 'fraction retraced'}",
                     fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        if row == 0 and col == 0:
            ax.legend(fontsize=8)
fig.suptitle("Weight-space route by scenario. Bar colour = scenario, outline = rule.",
             fontsize=10)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"saved {figure_path(__file__)}")

if not SMOKE and not REPLOT:
    np.savez(array_path(__file__), methods=np.array(METHODS),
             scenarios=np.array(SCENARIOS), layers=np.array(LAYERS), hidden=HIDDEN,
             **{f"{k}_{s}_{m}_{L}": stat[(s, m, L, k)]
                for s in SCENARIOS for m in METHODS for L in LAYERS for k in KEYS},
             **{f"argmax_{s}_{m}": curves[(s, m)] for s in SCENARIOS for m in METHODS},
             **{f"steps_{s}": steps[s] for s in SCENARIOS},
             **{f"switches_{s}": np.array(switches[s]) for s in SCENARIOS})
    print(f"saved {array_path(__file__)}")
