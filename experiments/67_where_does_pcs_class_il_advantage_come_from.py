"""PC's only positive result is in Class-IL. Is it output-layer suppression that it avoids?

THE CLAIM BEING TESTED, AND WHY IT IS CURRENTLY ONLY A HYPOTHESIS
    Scripts 56/57 found PC's one advantage over backprop anywhere in this project: crossover
    +1.29 (3.7 sem, 5W-0L across seeds) at a fixed budget and +3.00 (2.2 sem, 4W-1L) at matched
    competence, in CLASS-IL and nowhere else. Domain-IL gives -0.01 and +0.49, and 59 finds
    nothing in 8 of 8 depth x width cells.

    The standing explanation is that Class-IL forgetting is dominated by OUTPUT-LAYER
    SUPPRESSION -- absent classes' units are pushed down until they cannot win an argmax -- and
    that PC, whose updates diverge from backprop's most at the output layer (55: cos 0.623 on the
    last layer against 0.952 on W1; 65: output-layer path 31% shorter while displacing 22%
    further), suppresses less. Domain-IL has no such mechanism to avoid, because every unit is a
    target for some class.

    **That explanation has never been measured.** Scripts 42/43 established the suppression
    account of Class-IL forgetting -- NCM holding ~80% while argmax reads 0.2%, the collapse
    living entirely in W2 -- but they ran BACKPROP ONLY. Nothing in the project shows how PC's
    Class-IL forgetting decomposes, so the one positive result rests on an untested inference.

WHAT THIS SCRIPT DOES
    Runs 43's decomposition -- the same four conditions, the same matched-competence reading --
    for all four rules instead of one:

        control          the rule as it is
        masked           absent classes get no gradient. Removes suppression BY FIAT, so it is
                         an ORACLE (it needs task identity), not a method. It is the reference
                         for "how much of the forgetting was suppression".
        frozen hidden    W1, b1 held at the switch. Removes representation drift.
        masked + frozen  both. The ceiling.

    Reading it. If suppression is what PC avoids, PC's control should sit CLOSER to its own
    masked condition than backprop's does -- i.e. masking should buy PC less, because PC was
    already not suppressing much. The quantity is therefore the GAP (masked - control) per rule,
    paired per seed, and the prediction is gap(pc) < gap(backprop).

PRE-COMMITTED READINGS
    gap(pc) < gap(backprop), NCM-argmax gap smaller for PC  -> the hypothesis holds; PC's
        Class-IL advantage is reduced output suppression, and its absence in Domain-IL follows
        because there is no suppression there to avoid. This closes the story.
    gaps equal                                              -> PC suppresses as much as backprop
        and the Class-IL advantage comes from somewhere else. The hypothesis is REFUTED and the
        +1.29/+3.00 needs a new explanation -- representation drift is the remaining candidate,
        and the frozen-hidden column measures it.
    gap(pc) > gap(backprop)                                 -> PC suppresses MORE and its
        advantage is despite that, not because of it. Least expected, most interesting.
    EqProp is the control on the control: 54 measured its updates as nearly orthogonal to
        backprop's, and it is WORSE in Class-IL on every seed (0W-5L). If its gap is also
        unremarkable, then "how much a rule suppresses" is not what separates these rules at all.

DEVIATIONS FROM THE PROTOCOL, STATED
    * Class-IL, deliberately -- this is a question about the scenario where the effect exists.
    * 43's thresholds and caps, unchanged, so the backprop column is directly comparable with it.
    * Four rules x four conditions x 5 seeds. EqProp is included despite the cost because it is
      the discriminating control described above, not for completeness.
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
from src.probes import live_ncm_fn, prototype_images
from src.metrics import metric_grid, report_grid, paired_diff, value_when, crossover, half_life

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
CONDITIONS = {"control": (False, False), "masked": (True, False),
              "frozen hidden": (False, True), "masked + frozen": (True, True)}
COLORS = {"backprop": "tab:gray", "replay": "tab:brown",
          "pc": "tab:red", "eqprop": "tab:green"}

# 43's values, unchanged, so its backprop column is directly comparable with this one.
T1_THRESHOLD, T2_THRESHOLD, STOP_PATIENCE = 0.90, 0.50, 3
CAP1, CAP2, EVAL_EVERY, SEEDS = 3000, 3000, 10, 5

cap = np.load(_array_path(str(ROOT / "experiments" / "41_capacity_vs_hidden_width.py")))
HIDDEN = int(cap["chosen"])
z51 = np.load(_array_path(str(ROOT / "experiments" /
                              "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
SETTLE_TOL, EQ_MAX_STEPS = float(z51["settle_tol"]), int(z51["eq_max_steps"])
PC_STEPS = int(z51["pc_steps"])

if SMOKE:
    SEEDS, CAP1, CAP2 = 2, 120, 90
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} | CLASS-IL | 4 rules x 4 conditions | {SEEDS} seeds")
print(f"  task 1 to {T1_THRESHOLD:.0%}, task 2 read at {T2_THRESHOLD:.0%}")
print("  question: is PC's Class-IL advantage reduced output-layer suppression?\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="class_il", eval_every=EVAL_EVERY,
               seeds=SEEDS, stop_threshold=[T1_THRESHOLD, T2_THRESHOLD],
               stop_patience=STOP_PATIENCE, max_iters_per_task=[CAP1, CAP2])


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
    kept = {(m, c): z[f"kept_{m}_{c}"] for m in METHODS for c in CONDITIONS}
    ncm = {(m, c): z[f"ncm_{m}_{c}"] for m in METHODS for c in CONDITIONS}
    t2 = {(m, c): z[f"t2_{m}_{c}"] for m in METHODS for c in CONDITIONS}
    xh = {(m, c): z[f"xh_{m}_{c}"] for m in METHODS for c in CONDITIONS}
    reached = {(m, c): int(z[f"reached_{m}_{c}"]) for m in METHODS for c in CONDITIONS}
    print("--replot: from saved arrays, no training\n")
else:
    data = load(base)
    kept, ncm, t2, xh, reached = {}, {}, {}, {}, {}
    t0 = time.perf_counter()
    for m in METHODS:
        for cname, (mask, freeze) in CONDITIONS.items():
            rows_k, rows_n, rows_2, rows_x, n_ok = [], [], [], [], 0
            for seed in range(SEEDS):
                proto = replace(base, mask=mask, lr={m: LR[m]})
                tasks = proto.tasks(seed)
                handle = {}
                train_step, predict = build(proto, m, seed, handle=handle, **settle_kw(m))
                px, py = prototype_images(data.train, data.class_idx,
                                          sorted(tasks[0] + tasks[1]), per_class=50,
                                          device=proto.device, seed=seed)

                def on_task_end(ti, step, _f=freeze, _h=handle):
                    if ti == 0 and _f:
                        _h["freeze"].update({"W1", "b1"})

                out = run_classil(
                    train_step, predict, tasks, data.train, data.class_idx,
                    report_eval=data.report_eval, stop_eval=data.stop_eval,
                    readouts={"argmax": predict,
                              "ncm": live_ncm_fn(handle["features"], px, py)},
                    max_iters_per_task=[CAP1, CAP2], batch=proto.batch,
                    eval_every=EVAL_EVERY, device=proto.device, data_seed=seed,
                    stop_threshold=[T1_THRESHOLD, T2_THRESHOLD],
                    stop_patience=STOP_PATIENCE, on_task_end=on_task_end)

                s = out["steps"]
                ca, cn = out["curves"]["argmax"] * 100, out["curves"]["ncm"] * 100
                sw = out["switches"][0]
                # read at the MATCHED moment, exactly as 43 does: the first eval at which task 2
                # has held the threshold, so no rule is credited for simply learning task 2 less
                rows_k.append(value_when(s, ca[:, 1], T2_THRESHOLD * 100, ca[:, 0],
                                         after=sw, patience=STOP_PATIENCE))
                rows_n.append(value_when(s, ca[:, 1], T2_THRESHOLD * 100, cn[:, 0],
                                         after=sw, patience=STOP_PATIENCE))
                rows_2.append(ca[-1, 1])
                rows_x.append(crossover(s, ca[:, 0], ca[:, 1], after=sw)[1])
                n_ok += int(out["reached"][1])
            kept[(m, cname)] = np.array(rows_k, dtype=float)
            ncm[(m, cname)] = np.array(rows_n, dtype=float)
            t2[(m, cname)] = np.array(rows_2, dtype=float)
            xh[(m, cname)] = np.array(rows_x, dtype=float)
            reached[(m, cname)] = n_ok
        print(f"  {m:10s} done   [{time.perf_counter() - t0:6.0f}s]")

# ---------------------------------------------------------------- readings
print(f"\n  task 1 retained when task 2 first reached {T2_THRESHOLD:.0%}   "
      f"(argmax %, mean over {SEEDS} seeds; n = seeds that reached the threshold)")
print(f"  {'rule':10s} " + " ".join(f"{c:>17s}" for c in CONDITIONS))
for m in METHODS:
    cells = [f"{np.nanmean(kept[(m, c)]):8.1f}  {reached[(m, c)]}/{SEEDS}" for c in CONDITIONS]
    print(f"  {m:10s} " + " ".join(f"{x:>17s}" for x in cells))

# THE QUANTITY. masked - control, per seed. How much of this rule's Class-IL forgetting was
# output-layer suppression that an oracle mask could remove? A rule that already suppresses
# little has little left for the mask to buy.
print(f"\n  THE GAP: masked - control, paired per seed. How much of the forgetting was")
print(f"  suppression the oracle mask could remove. SMALLER means the rule suppresses less.")
print(f"  {'rule':10s} {'gap':>18s} {'vs backprop':>22s}")
gaps = {m: kept[(m, "masked")] - kept[(m, "control")] for m in METHODS}
for m in METHODS:
    g = gaps[m]
    line = f"  {m:10s} {np.nanmean(g):10.1f} +-{np.nanstd(g, ddof=1) / np.sqrt(np.isfinite(g).sum()):5.1f}"
    if m != "backprop":
        d, se, n = paired_diff(g, gaps["backprop"])
        line += f"   {d:+10.1f} +-{se:5.1f} {n:4.1f}sem"
    print(line)

print(f"\n  DRIFT for contrast: frozen hidden - control, paired per seed.")
for m in METHODS:
    g = kept[(m, "frozen hidden")] - kept[(m, "control")]
    print(f"  {m:10s} {np.nanmean(g):10.1f}")

print(f"\n  NCM minus argmax on task 1 at the matched moment. LARGE means the hidden code")
print(f"  survived and the output layer is what failed -- the signature of suppression.")
for m in METHODS:
    d = ncm[(m, "control")] - kept[(m, "control")]
    print(f"  {m:10s} NCM {np.nanmean(ncm[(m, 'control')]):5.1f}  "
          f"argmax {np.nanmean(kept[(m, 'control')]):5.1f}  gap {np.nanmean(d):+6.1f}")

gp, gb = np.nanmean(gaps["pc"]), np.nanmean(gaps["backprop"])
dp = paired_diff(gaps["pc"], gaps["backprop"])
print("\n  READING (pre-committed in the docstring):")
if dp[2] > 2 and dp[0] < 0:
    print(f"    PC has LESS to gain from the oracle mask than backprop ({gp:.1f} vs {gb:.1f},")
    print(f"    {dp[0]:+.1f} +- {dp[1]:.1f}, {dp[2]:.1f} sem), so it was suppressing less to begin")
    print("    with. The Class-IL advantage IS reduced output suppression, and its absence in")
    print("    Domain-IL follows -- there is no suppression there to avoid.")
elif dp[2] > 2 and dp[0] > 0:
    print(f"    PC has MORE to gain from the mask than backprop ({gp:.1f} vs {gb:.1f}). Its")
    print("    Class-IL advantage exists DESPITE suppressing more, not because of it. The")
    print("    standing hypothesis is refuted and inverted.")
else:
    print(f"    PC and backprop gain the same from the mask ({gp:.1f} vs {gb:.1f}, {dp[2]:.1f} sem).")
    print("    PC suppresses as much as backprop does, so reduced suppression does NOT explain")
    print("    its Class-IL advantage. The hypothesis is refuted; the frozen-hidden column is")
    print("    the remaining candidate.")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
ax = axes[0]
w, xs = 0.2, np.arange(len(CONDITIONS))
for i, m in enumerate(METHODS):
    mu = [np.nanmean(kept[(m, c)]) for c in CONDITIONS]
    se = [np.nanstd(kept[(m, c)], ddof=1) / np.sqrt(max(np.isfinite(kept[(m, c)]).sum(), 1))
          for c in CONDITIONS]
    ax.bar(xs + (i - 1.5) * w, mu, w, yerr=se, capsize=3, color=COLORS[m], label=m, alpha=0.9)
ax.set_xticks(xs); ax.set_xticklabels(list(CONDITIONS), fontsize=9)
ax.set_ylabel(f"task 1 kept at matched task-2 = {T2_THRESHOLD:.0%}  (%)")
ax.set_title("the decomposition, per rule", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")

ax = axes[1]
for i, m in enumerate(METHODS):
    g = gaps[m]
    ax.bar(i, np.nanmean(g),
           yerr=np.nanstd(g, ddof=1) / np.sqrt(max(np.isfinite(g).sum(), 1)),
           capsize=4, color=COLORS[m], alpha=0.9)
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, fontsize=9)
ax.set_ylabel("masked - control  (points)")
ax.set_title("how much the oracle mask buys\nSMALLER = already suppressing less", fontsize=10)
ax.grid(alpha=0.25, axis="y")

ax = axes[2]
for m in METHODS:
    ax.plot(kept[(m, "control")], ncm[(m, "control")], "o", ms=7, color=COLORS[m], label=m)
lim = [0, max(1.0, np.nanmax([np.nanmax(ncm[(m, "control")]) for m in METHODS]) * 1.1)]
ax.plot(lim, lim, ls=":", color="k", lw=1)
ax.annotate("above the line = hidden code survived,\noutput layer failed", xy=(0.04, 0.86),
            xycoords="axes fraction", fontsize=8)
ax.set_xlabel("argmax on task 1 (%)"); ax.set_ylabel("NCM on task 1 (%)")
ax.set_title("is the damage in the output layer?", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.25)

fig.suptitle(f"Where does Class-IL forgetting live, per learning rule?  {base.describe()}",
             fontsize=10)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

if not SMOKE:
    np.savez(array_path(__file__), methods=np.array(METHODS),
             conditions=np.array(list(CONDITIONS)), hidden=HIDDEN,
             t1_threshold=T1_THRESHOLD, t2_threshold=T2_THRESHOLD,
             **{f"kept_{m}_{c}": kept[(m, c)] for m in METHODS for c in CONDITIONS},
             **{f"ncm_{m}_{c}": ncm[(m, c)] for m in METHODS for c in CONDITIONS},
             **{f"t2_{m}_{c}": t2[(m, c)] for m in METHODS for c in CONDITIONS},
             **{f"xh_{m}_{c}": xh[(m, c)] for m in METHODS for c in CONDITIONS},
             **{f"reached_{m}_{c}": reached[(m, c)] for m in METHODS for c in CONDITIONS})
    print(f"saved {array_path(__file__)}")
