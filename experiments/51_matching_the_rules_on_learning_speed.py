"""What learning rate makes the four rules learn task 1 at the same speed?

Methods slide for the rule comparison. Nothing here is a result about forgetting; it is the
calibration that makes the comparison in scripts 52-53 a comparison of rules rather than a
comparison of learning rates.

WHY THIS HAS TO EXIST
    Every rule is run at plain SGD with a single learning rate, and the four rules do not
    respond to the same number the same way -- PC applies its update as `W += lr * (a^T e)`
    with e computed from a settled state, EqProp divides a finite difference by beta before
    the optimiser sees it. So an identical `lr` across rules is NOT a control; it is an
    arbitrary choice that happens to favour whichever rule it suits.

    What must be matched is the thing a reader would otherwise attribute the difference to:
    HOW FAST EACH RULE LEARNS. If backprop reaches competence on task 1 in 80 updates and PC
    takes 600, then their forgetting curves are being read at different points of two
    different processes, and any difference between them is unattributable.

WHAT IS MATCHED, AND WHY THIS QUANTITY
    updates to first hold >= THRESHOLD on task 1, for STOP_PATIENCE consecutive evals.

    Measured on TASK 1 ALONE, before task 2 exists. That matters: it means the calibration
    cannot launder the result. Tuning each rule's learning rate until its CROSSOVER sat in a
    convenient place would be tuning on the dependent variable -- the crossover is a forgetting
    measurement, and choosing learning rates by it would remove the very difference the
    comparison is meant to detect.

    The common target is TARGET_STEPS, chosen for legibility rather than from theory. At ~300
    updates to competence a 600-update task-2 block spreads the transition across half the
    axis; the earlier scripts ran transitions of ~200 updates inside 4000 and the curves read
    as step functions.

EXPERIMENT 12's LEARNING RATES CANNOT BE CARRIED FORWARD
    They were set under the legacy specification, where backprop ran ReLU with cross-entropy
    and EqProp ran a hinge loss on +-1 targets -- a different nonlinearity and a different loss
    per rule (see methods.legacy and model.LEGACY_SPEC). All four now run tanh, a linear
    output, squared error and one-hot targets. The gradient magnitudes are not comparable
    across that boundary, so the old numbers are void rather than approximately right.

THE SECOND QUESTION THIS FIGURE ANSWERS
    If the plots still look cramped, the cheap fix is to divide EVERY rule's learning rate by
    the same factor and keep the matching. That works only if steps-to-threshold responds to
    learning rate the same way for every rule -- i.e. if the four curves are PARALLEL on the
    log-log axes of the left panel. If they have different slopes, a global divide breaks the
    matching and the grid has to be re-run at a slower target instead. The figure shows which,
    so the shortcut is checked rather than assumed.

SETTLING: BOTH RULES ARE VERIFIED FULLY SETTLED, AND THE SETTINGS COME FROM SCRIPT 50
    Script 50 measured how many relaxation steps each rule needs before its hidden state stops
    moving, at four points during task-1 training:

        PC       needs <= 18 steps, runs 50 fixed        margin 2.8x to 6.3x
        EqProp   needs 47 steps trained, 529 at          stopping rule replaced; see below.
                 initialisation -- a 9x range            Tolerance calibrated by script 50.

    Under the settings this script uses, the weight update of each rule is computed from a
    genuinely settled state -- script 50 asserts that, and so does the check below. The
    concern that PC's fixed 50 steps make it a truncated relaxation -- which would pull it
    toward backprop (Millidge et al. 2022) and weaken the comparison before it starts -- does
    not apply at this size. The asymmetry that PC stops on a step count and EqProp stops on a
    patience test is therefore about compute and exposition, not about what is being compared.

    A fixed budget for EqProp was considered and rejected on 50's numbers. Its requirement
    spans 9x and peaks at INITIALISATION, so a fixed budget is either sized for the worst case
    (~600 steps, 4x more expensive than the patience rule it would replace) or sized for the
    trained regime (~150, wrong for the opening ~50 updates of every run).

    EqProp's stopping rule was CHANGED as a result of script 50, and the reason matters for
    reading any EqProp number in this project. It previously stopped once per-step movement
    had failed to improve for 30 consecutive steps, which fired on a temporary plateau at
    initialisation -- one seed needed 529 steps and it stopped at 178 -- and overshot by 3-4x
    once trained. It now stops when one step displaces the state by less than `settle_tol` of
    its own norm, and script 50 calibrates that tolerance against its own measurement of what
    is genuinely required. SETTLE_TOL below is read from 50's output, not chosen here.

READINGS COMMITTED BEFORE RUNNING
    Steps-to-threshold falls as learning rate rises, then the curve turns up or the run stops
    reaching threshold at all as the rate becomes unstable. The chosen learning rate must sit
    on the falling part, not near the turn -- a rule calibrated at the edge of instability
    would show extra forgetting for a reason that has nothing to do with credit assignment.
    The right panel is the check: final task-1 accuracy must be at its plateau at the chosen
    rate, not already falling.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, build, replace, figure_path, array_path
from src.runner import run_classil

# ---------------------------------------------------------------- settings
THRESHOLD = 0.90        # task-1 competence, against a 94.3% Domain-IL joint ceiling (script 41)
STOP_PATIENCE = 3
TARGET_STEPS = 300      # the common learning speed every rule is calibrated to
MAX_ITERS = 900         # cap at 3x the target: a rule slower than that is not a candidate
EVAL_EVERY = 20         # steps-to-threshold IS the measurement, but EqProp pays 0.25 s an eval
SEEDS = 3

METHODS = ["backprop", "replay", "pc", "eqprop"]
COLORS = {"backprop": "tab:gray", "replay": "tab:brown",
          "pc": "tab:red", "eqprop": "tab:green"}
GRIDS = {
    "backprop": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "replay":   [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "pc":       [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "eqprop":   [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02],
}

# EqProp costs ~185 ms per update here, so the full grid is ~45 minutes. `--smoke` runs every
# code path in under a minute; it proves the script executes, and nothing else.
if "--smoke" in sys.argv:
    SEEDS, MAX_ITERS, TARGET_STEPS = 1, 60, 40
    GRIDS = {m: g[2:4] for m, g in GRIDS.items()}
    print("--smoke: tiny grid, results are NOT meaningful\n")

HIDDEN = int(np.load(array_path(str(ROOT / "experiments" /
                                    "41_capacity_vs_hidden_width.py")))["chosen"])

# ---- settling, read from script 50 ----------------------------------------
# Nothing here is chosen by hand. Script 50 measures how many relaxation steps each rule needs
# and calibrates EqProp's tolerance against that measurement; this reads its output.
PC_STEPS = 50           # library default, kept: script 50 measured a worst case of 18
z50 = np.load(array_path(str(ROOT / "experiments" /
                             "50_do_the_settling_loops_converge.py")))
need = {r: float(np.nanmax(z50[f"settled_{r}"])) for r in ["pc", "eqprop"]}
SETTLE_TOL = float(z50["settle_tol"])
EQ_MAX_STEPS = int(np.ceil(1.5 * need["eqprop"]))    # cap only; the tolerance does the work

# Asserted rather than assumed: if anything later makes a rule need more settling than it is
# given, this stops the run instead of silently comparing a settled rule against a truncated
# one. That failure is invisible in the output curves, which is why it is checked here.
assert PC_STEPS > need["pc"], f"PC settles in {need['pc']:.0f} steps but is given {PC_STEPS}"
assert np.isfinite(SETTLE_TOL), "script 50 found no safe tolerance; re-run it before this"
print(f"H = {HIDDEN} from script 41")
print(f"settling, worst case over script 50's checkpoints: "
      + ", ".join(f"{r} needs {v:.0f}" for r, v in need.items())
      + f" | pc runs {PC_STEPS} fixed, eqprop stops at settle_tol {SETTLE_TOL:.0e} "
        f"under a {EQ_MAX_STEPS} cap\n")


def settle_kw(method):
    """This rule's settling configuration. Empty for the two rules that do not settle."""
    if method == "pc":
        return dict(steps=PC_STEPS)
    if method == "eqprop":
        return dict(max_steps=EQ_MAX_STEPS, settle_tol=SETTLE_TOL)
    return {}


base = replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il", stop_threshold=THRESHOLD,
               stop_patience=STOP_PATIENCE, max_iters_per_task=MAX_ITERS,
               eval_every=EVAL_EVERY, seeds=SEEDS)

# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    steps_to = {m: z[f"steps_{m}"] for m in METHODS}
    final = {m: z[f"final_{m}"] for m in METHODS}
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(base)
    steps_to = {m: np.full((len(GRIDS[m]), SEEDS), np.nan) for m in METHODS}
    final = {m: np.full((len(GRIDS[m]), SEEDS), np.nan) for m in METHODS}
    t0 = time.perf_counter()

    for m in METHODS:
        for li, lr in enumerate(GRIDS[m]):
            for seed in range(SEEDS):
                proto = replace(base, lr={m: lr})
                pair = proto.tasks(seed)                  # same split for every rule at this seed
                lmap = proto.label_map([pair[0]])
                train_step, predict = build(proto, m, seed, **settle_kw(m))
                out = run_classil(
                    train_step, predict, [pair[0]], data.train, data.class_idx,
                    report_eval=data.report_eval, stop_eval=data.stop_eval,
                    max_iters_per_task=MAX_ITERS, batch=proto.batch, eval_every=EVAL_EVERY,
                    device=proto.device, data_seed=seed, label_map=lmap,
                    stop_threshold=THRESHOLD, stop_patience=STOP_PATIENCE,
                )
                # `switches[0]` is where the task stopped, which under a threshold IS
                # steps-to-threshold. `reached` distinguishes that from hitting the cap.
                steps_to[m][li, seed] = out["switches"][0] if out["reached"][0] else np.nan
                final[m][li, seed] = out["curves"]["argmax"][-1, 0] * 100
            n_ok = int(np.isfinite(steps_to[m][li]).sum())
            s = np.nanmean(steps_to[m][li])
            print(f"  {m:9s} lr={lr:<7g} steps to {THRESHOLD:.0%}: "
                  + (f"{s:6.0f}" if n_ok else "  none")
                  + f"   reached {n_ok}/{SEEDS}   final {np.nanmean(final[m][li]):5.1f}%"
                  f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- choose
# Nearest in LOG steps, so being 2x too fast and 2x too slow are penalised equally. Only
# learning rates that reached threshold on EVERY seed are eligible: one seed failing to reach
# competence is an unstable rate, not a fast one.
chosen, achieved = {}, {}
for m in METHODS:
    mean = np.array([np.nanmean(steps_to[m][li]) for li in range(len(GRIDS[m]))])
    ok = np.array([np.isfinite(steps_to[m][li]).all() for li in range(len(GRIDS[m]))])
    if not ok.any():
        chosen[m], achieved[m] = np.nan, np.nan
        continue
    d = np.where(ok, np.abs(np.log(mean / TARGET_STEPS)), np.inf)
    li = int(np.argmin(d))
    chosen[m], achieved[m] = GRIDS[m][li], mean[li]

print(f"\n  matched at ~{TARGET_STEPS} updates to {THRESHOLD:.0%} on task 1\n")
print(f"  {'rule':10s} {'lr':>9s} {'steps':>8s} {'final acc':>11s}")
for m in METHODS:
    li = GRIDS[m].index(chosen[m]) if np.isfinite(chosen[m]) else None
    f = np.nanmean(final[m][li]) if li is not None else float("nan")
    print(f"  {m:10s} {chosen[m]:9g} {achieved[m]:8.0f} {f:10.1f}%")

spread = np.nanmax(list(achieved.values())) / max(1e-9, np.nanmin(list(achieved.values())))
print(f"\n  spread in learning speed after matching: {spread:.2f}x "
      + ("- acceptable" if spread < 1.5 else "- LARGE, the grid is too coarse near the target"))

# Is a global learning-rate divide safe? It is, if steps-to-threshold responds to learning rate
# with the same exponent for every rule. Fit log(steps) = a + b*log(lr) per rule and compare b.
slopes = {}
for m in METHODS:
    x = np.log(np.asarray(GRIDS[m], dtype=float))
    y = np.log(np.array([np.nanmean(steps_to[m][li]) for li in range(len(GRIDS[m]))]))
    ok = np.isfinite(y)
    slopes[m] = np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() >= 2 else np.nan
sl = np.array(list(slopes.values()))
print(f"\n  d log(steps) / d log(lr): "
      + ", ".join(f"{m} {slopes[m]:+.2f}" for m in METHODS))
print("  " + ("curves are parallel -- dividing every lr by the same factor keeps the matching"
              if np.nanmax(sl) - np.nanmin(sl) < 0.3 else
              "curves are NOT parallel -- a global lr divide breaks the matching; re-run this "
              "grid at a larger TARGET_STEPS instead"))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

for m in METHODS:
    g = np.asarray(GRIDS[m], dtype=float)
    mu = np.array([np.nanmean(steps_to[m][li]) for li in range(len(GRIDS[m]))])
    se = np.array([np.nanstd(steps_to[m][li], ddof=1) / np.sqrt(SEEDS)
                   for li in range(len(GRIDS[m]))])
    ax[0].errorbar(g, mu, yerr=se, marker="o", capsize=3, lw=2, color=COLORS[m], label=m)
    ax[1].errorbar(g, [np.nanmean(final[m][li]) for li in range(len(GRIDS[m]))],
                   marker="o", capsize=3, lw=2, color=COLORS[m], label=m)
    if np.isfinite(chosen[m]):
        ax[0].plot([chosen[m]], [achieved[m]], "*", color=COLORS[m], ms=18, mec="k", mew=0.6)

ax[0].axhline(TARGET_STEPS, color="k", ls="--", lw=1)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("learning rate")
ax[0].set_ylabel(f"updates to reach {THRESHOLD:.0%} on task 1")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.25, which="both")

ax[1].axhline(THRESHOLD * 100, color="k", ls="--", lw=1)
ax[1].set_xscale("log")
ax[1].set_xlabel("learning rate")
ax[1].set_ylabel("task 1 accuracy at the end (%)")
ax[1].set_ylim(0, 100)
ax[1].grid(alpha=0.25, which="both")

fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

np.savez(array_path(__file__), target_steps=TARGET_STEPS, threshold=THRESHOLD, hidden=HIDDEN,
         methods=np.asarray(METHODS), pc_steps=PC_STEPS,
         settle_tol=SETTLE_TOL, eq_max_steps=EQ_MAX_STEPS,
         lr=np.asarray([chosen[m] for m in METHODS]),
         achieved=np.asarray([achieved[m] for m in METHODS]),
         slope=np.asarray([slopes[m] for m in METHODS]),
         **{f"grid_{m}": np.asarray(GRIDS[m], dtype=float) for m in METHODS},
         **{f"steps_{m}": steps_to[m] for m in METHODS},
         **{f"final_{m}": final[m] for m in METHODS})
print(f"saved {array_path(__file__)}")
