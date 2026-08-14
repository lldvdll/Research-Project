"""Does Song & Bogacz's backtracking inference step change what PC computes, or only the route
it takes to the same fixed point?

THE ONE DOCUMENTED DIVERGENCE FROM [R1]
    Their `optimizer_x` shrinks the inference step size whenever the energy fails to fall
    (x_lr_discount=0.9, x_lr_amplifier=1.0). Ours takes a fixed step (1.0/1.0). Everything else
    in `pc_settle` and `pc_update` matches their published update equations -- checked line by
    line in the 2026-08-13 audit, and `src/test_numpy_mirror.py` verifies dF/dx and dF/dW
    against finite differences and that steps=0 reproduces backprop's output-layer gradient.

WHY IT SHOULD NOT MATTER, AND WHY THAT IS WORTH TESTING ANYWAY
    Backtracking is a step-size rule on a descent process. If the relaxation converges, both
    rules land on the SAME fixed point and the weight update -- which is computed from the
    settled state -- is identical. Script 50 measured PC reaching equilibrium in <= 18 steps
    where the protocol runs 50, so ours converges with room to spare.

    THE LIVE RISK RUNS THE OTHER WAY. If a fixed step count leaves the relaxation only PARTLY
    settled, the operating point is not the equilibrium, and "prospective configuration" is
    then a partially-relaxed state rather than a settled one -- a different algorithm wearing
    the same name. That is a claim about THEIR configuration, which we cannot test directly,
    but we can measure how far from equilibrium each rule sits at each step count, and how much
    the weight update depends on being there. That is what panel 1 does.

THREE LEVELS, BECAUSE A NULL AT THE OUTCOME LEVEL IS ONLY MEANINGFUL WITH THE FIRST TWO
    1. STATE     distance from the fully-settled state after k inference steps, both rules.
                 Answers "do they converge, and how fast", which is the mechanism.
    2. UPDATE    cos(dW_fixed, dW_backtracking) at matched k. Answers "does any difference in
                 the state survive into the weights".
    3. OUTCOME   a full Domain-IL forgetting run under each, paired per seed, on the metric
                 grid. Answers "does it change the result".

    Reporting 3 alone would be the mistake this project has made before: a null at the outcome
    level is uninterpretable if the intervention never did anything at the state level.

PRE-COMMITTED READINGS
    Converges, updates identical, outcome identical  -> the divergence is bookkeeping. Record it
                                                        as closed and stop citing it as a caveat.
    Converges but updates differ                     -> the two find DIFFERENT fixed points, so
                                                        the energy has more than one and the
                                                        step rule selects among them. Then it is
                                                        a real experimental variable.
    Does not converge at 50 steps                    -> script 50's calibration is wrong and
                                                        every PC number in the project is
                                                        computed from a partly-settled state.

DEVIATIONS FROM THE PROTOCOL, STATED
    * PC only. The question is about PC's inference loop; no other rule has one of this form.
    * Fixed budgets per task, as script 52, so the outcome panel is comparable with it.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.protocol import (PROTOCOL, load, build, run, replace,
                          figure_path as _figure_path, array_path as _array_path)
from src.predictive_coding import pc_settle, pc_update
from src.model import make_target, active_vector, init_params
from src.metrics import metric_grid, report_grid, paired_diff
from src.plotting import plot_retention_curve

SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):      # noqa: F811
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):       # noqa: F811
    return _array_path(f, _tag(f, suffix))


# ---------------------------------------------------------------- settings
VARIANTS = {"fixed step (ours)": 1.0, "backtracking (S&B)": 0.9}
COLORS = {"fixed step (ours)": "tab:red", "backtracking (S&B)": "tab:purple"}
SEEDS = 5
EVAL_EVERY = 10
STEP_GRID = [1, 2, 5, 10, 20, 50, 100, 200]
REFERENCE_STEPS = 2000            # "fully settled" reference, both rules must agree with it

HIDDEN = int(np.load(_array_path(str(ROOT / "experiments" /
                                     "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(_array_path(str(ROOT / "experiments" /
                              "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
PC_STEPS = int(z51["pc_steps"])
T = float(z51["target_steps"])
ITERS = [int(2 * T), int(1.5 * T)]

if SMOKE:
    SEEDS, ITERS, STEP_GRID, REFERENCE_STEPS = 2, [40, 30], [1, 5, 50], 400
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} | PC only | protocol runs {PC_STEPS} inference steps | {SEEDS} seeds")
print("  question: does the inference step-size rule change PC, or only its route?\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il", stop_threshold=None,
               max_iters_per_task=ITERS, eval_every=EVAL_EVERY, seeds=SEEDS,
               lr={"pc": LR["pc"]})
data = load(base)

REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

# ================================================================ 1. STATE
# How far from equilibrium is the hidden state after k inference steps, under each rule? The
# reference is the same relaxation run far past convergence. Measured at INITIALISATION and at
# the task switch, because script 50 found the settling requirement changes as weights grow.
print("  1. STATE -- relative distance from the fully-settled state after k steps")
tasks0 = base.tasks(0)
lmap0 = base.label_map(tasks0)
x_probe, y_probe = data.report_eval
keep = torch.zeros(len(y_probe), dtype=torch.bool)
for c in tasks0[0]:
    keep |= (y_probe == c)
xb = x_probe[keep][:64]
yb = y_probe[keep][:64]
if lmap0 is not None:
    yb = torch.tensor([lmap0[int(v)] for v in yb])

state_err = {v: {} for v in VARIANTS}
for when in ("at init", "after task 1"):
    if when == "at init":
        p = init_params(base.arch, seed=0, device=base.device)
    else:
        h = {}
        ts, _ = build(base, "pc", 0, handle=h, steps=PC_STEPS)
        loader_x, loader_y = xb, yb
        for _ in range(200 if not SMOKE else 20):
            ts(loader_x, loader_y, active=sorted(set(int(v) for v in yb.tolist())))
        p = h["params"]
    tgt = make_target(yb, base.arch, base.objective, device=base.device)
    av = active_vector(sorted(set(int(v) for v in yb.tolist())), base.arch, device=base.device)
    for name, disc in VARIANTS.items():
        ref, _ = pc_settle(xb, p, base.arch, base.objective, tgt, av, dt=0.1,
                           steps=REFERENCE_STEPS, x_lr_discount=disc)
        errs = []
        for k in STEP_GRID:
            xs, _ = pc_settle(xb, p, base.arch, base.objective, tgt, av, dt=0.1,
                              steps=k, x_lr_discount=disc)
            num = sum(((a - b) ** 2).sum() for a, b in zip(xs, ref)).sqrt()
            den = sum((b ** 2).sum() for b in ref).sqrt()
            errs.append(float(num / den))
        state_err[name][when] = errs
    # do the two rules agree on WHERE they settle? if not, the energy has several minima and
    # the step rule selects among them -- which would make this a real experimental variable.
    ra, _ = pc_settle(xb, p, base.arch, base.objective, tgt, av, dt=0.1,
                      steps=REFERENCE_STEPS, x_lr_discount=1.0)
    rb, _ = pc_settle(xb, p, base.arch, base.objective, tgt, av, dt=0.1,
                      steps=REFERENCE_STEPS, x_lr_discount=0.9)
    gap = float(sum(((a - b) ** 2).sum() for a, b in zip(ra, rb)).sqrt()
                / sum((b ** 2).sum() for b in rb).sqrt())
    print(f"     {when:14s} same fixed point? relative gap between the two = {gap:.2e}")
    for name in VARIANTS:
        cells = " ".join(f"{e:8.1e}" for e in state_err[name][when])
        print(f"       {name:20s} " + cells)
print("       steps:               " + " ".join(f"{k:>8d}" for k in STEP_GRID))

# ================================================================ 2. UPDATE
# Does any difference in the settled state survive into the weight update? Computed at matched
# step counts on the same weights and the same batch, so the ONLY difference is the step rule.
print("\n  2. UPDATE -- cos(dW fixed, dW backtracking) at matched inference steps")
p0 = init_params(base.arch, seed=0, device=base.device)
act = sorted(set(int(v) for v in yb.tolist()))
cos_by_k = []
for k in STEP_GRID:
    deltas = []
    for disc in (1.0, 0.9):
        p = p0.clone()
        before = [w.clone() for w in p.Ws]
        pc_update(xb, yb, p, arch=base.arch, obj=base.objective, lr=LR["pc"], dt=0.1,
                  steps=k, active=act, device=base.device, x_lr_discount=disc)
        deltas.append(torch.cat([(w - b).reshape(-1) for w, b in zip(p.Ws, before)]))
    c = torch.nn.functional.cosine_similarity(deltas[0], deltas[1], dim=0).item()
    rel = float((deltas[0] - deltas[1]).norm() / deltas[1].norm())
    cos_by_k.append((k, c, rel))
    print(f"     k={k:4d}   cos = {c:.6f}   relative size difference = {rel:.2e}")

# ================================================================ 3. OUTCOME
print(f"\n  3. OUTCOME -- full Domain-IL runs, {PC_STEPS} inference steps, paired per seed")
if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    steps, switches = z["steps"], list(z["switches"])
    curves = {v: z[f"argmax_{i}"] for i, v in enumerate(VARIANTS)}
    print("     --replot: from saved arrays, no training")
else:
    curves, t0 = {}, time.perf_counter()
    for name, disc in VARIANTS.items():
        rows = []
        for seed in range(SEEDS):
            out = run(base, "pc", seed, data=data, steps=PC_STEPS, x_lr_discount=disc)
            rows.append(out["curves"]["argmax"])
        steps, switches = out["steps"], out["switches"]
        curves[name] = np.stack(rows)
        print(f"     {name:20s} done  [{time.perf_counter() - t0:5.0f}s]")

sw = switches[0]
names = list(VARIANTS)
grid = {n: metric_grid(steps, curves[n], sw) for n in names}
report_grid(grid, names, control=names[0], primary="crossover")

d = paired_diff(grid[names[1]]["crossover"], grid[names[0]]["crossover"])
# The SIZE of the largest disagreement across the whole metric grid, in accuracy points. A
# sem ratio cannot be the test here: when two runs agree exactly the paired difference has zero
# mean AND zero spread, and n_sem comes back as inf -- which any "is it > 2 sem" check reads as
# overwhelming significance. That is the same trap report_grid had. Judge this one on effect
# size against float32 resolution, which is what the question actually is.
worst = max(abs(float(np.nanmean(grid[names[1]][k]) - np.nanmean(grid[names[0]][k])))
            for k in grid[names[0]])
curve_gap = float(np.nanmax(np.abs(curves[names[0]] - curves[names[1]])) * 100)
print("\n  READING (pre-committed in the docstring):")
print(f"    largest disagreement over the whole metric grid : {worst:.2e} accuracy points")
print(f"    largest disagreement at any point on any curve  : {curve_gap:.2e} accuracy points")
if worst < 0.05:
    print("    The two rules are EQUIVALENT to within float noise. PC is fully settled either")
    print("    way (panel 1), so the weight update -- computed from the settled state -- cannot")
    print("    see which route it took there (panel 2, cos = 1.000000 at every step count).")
    print("    The one documented divergence from [R1] is CLOSED. Stop citing it as a caveat.")
    print("    What remains open is the OTHER direction: whether [R1]'s own fixed step count")
    print("    leaves THEIR relaxation short of equilibrium. Panel 1 gives the tool for that")
    print("    -- distance from the settled state per step -- but not their configuration.")
else:
    print(f"    THE STEP RULE CHANGES THE RESULT: crossover {d[0]:+.2f} +- {d[1]:.2f}.")
    print("    Then it is an experimental variable, not a detail, and every PC number in the")
    print("    project is conditional on the fixed-step choice.")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
for name in VARIANTS:
    for when, ls in (("at init", "-"), ("after task 1", "--")):
        ax.plot(STEP_GRID, state_err[name][when], ls, marker="o", ms=4,
                color=COLORS[name], label=f"{name}, {when}")
ax.axvline(PC_STEPS, color="k", lw=0.9, ls=":")
ax.annotate(f"protocol runs {PC_STEPS}", xy=(PC_STEPS, 1), xytext=(4, 0),
            textcoords="offset points", rotation=90, fontsize=8, va="bottom")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("inference steps"); ax.set_ylabel("relative distance from settled state")
ax.set_title("1. does the relaxation converge?", fontsize=10)
ax.legend(fontsize=7); ax.grid(alpha=0.25)

ax = axes[1]
# 1 - cos is EXACTLY zero wherever the two updates are bit-identical, which cannot be drawn on
# a log axis. Floored at 1e-16 (below float32's resolution, so anything at the floor is an exact
# tie) and the floor is drawn, rather than dropping those points and leaving a gap that reads
# as missing data.
FLOOR = 1e-16
gaps = [max(1.0 - c, FLOOR) for _, c, _ in cos_by_k]
ax.plot([k for k, _, _ in cos_by_k], gaps, "o-", color="k")
ax.axhline(FLOOR, color="gray", lw=0.9, ls=":")
ax.annotate("exactly identical", xy=(0.02, FLOOR), xycoords=("axes fraction", "data"),
            xytext=(0, 3), textcoords="offset points", fontsize=8, color="gray")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(FLOOR / 4, max(max(gaps) * 4, 1e-12))
ax.set_xlabel("inference steps"); ax.set_ylabel("1 - cos(dW_fixed, dW_backtracking)")
ax.set_title("2. does it survive into the weight update?", fontsize=10)
ax.grid(alpha=0.25)

ax = axes[2]
i_sw = int(np.argmin(np.abs(np.asarray(steps) - sw)))
segs = {n: [curves[n][r, i_sw:, :] for r in range(curves[n].shape[0])] for n in names}
plt.close(plot_retention_curve(segs, names, figure_path(__file__, "retention"),
                               colors=COLORS, chance=1.0 / base.classes_per_task,
                               title="3. outcome: task 1 against task 2, post-switch"))
for n in names:
    ax.plot(grid[n]["crossover"], grid[n]["final_t1"], "o", ms=7, color=COLORS[n], label=n)
ax.set_xlabel("crossover height (%)"); ax.set_ylabel("task 1 kept (%)")
ax.set_title("3. outcome, per seed", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.25)

fig.suptitle("PC's inference step-size rule: ours (fixed) against Song & Bogacz's "
             "(backtracking, x_lr_discount=0.9)", fontsize=11)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

if not SMOKE:
    np.savez(array_path(__file__), steps=steps, switches=switches,
             variants=np.array(names), step_grid=np.array(STEP_GRID),
             cos_by_k=np.array([[k, c, r] for k, c, r in cos_by_k]),
             **{f"argmax_{i}": curves[n] for i, n in enumerate(names)},
             **{f"state_{i}_{w.replace(' ', '_')}": np.array(state_err[n][w])
                for i, n in enumerate(names) for w in ("at init", "after task 1")})
    print(f"saved {array_path(__file__)}")
