"""Re-run experiment 12 exactly as it was written. Does the current code still produce its result?

WHY THIS IS THE DECISIVE TEST
    Experiment 12 reported that PC forgets less than backprop, read by eye off the crossover of
    its trajectory plots. Nothing since has reproduced an effect of that size. Every explanation
    tried so far has been eliminated -- the scenario (56/57 give the same answer as 52/53), the
    depth and width (59), the measurement point (fixed budget and matched competence agree), and
    the output standardisation (58, though that run was itself void).

    That leaves one possibility nobody has tested: THE CODE CHANGED. This runs 12's own
    parameters through today's library. If the old result comes back, the algorithms are intact
    and the difference is in how 12 was set up. If it does not come back, something in `src`
    behaves differently from when 12 was run, and finding what becomes the priority.

WHAT 12 ACTUALLY DID, which turns out to matter more than the learning rule
    ITERS = 100 updates per task. One hundred. Scripts 52-59 give each task 600-2500, so 12
    measured the EARLY TRANSIENT, before any rule had converged. That alone could account for a
    difference in crossover: the curves are steep and far from their asymptotes there.

    BP_LR = PC_LR = 0.05, the SAME learning rate for both, not matched on learning speed. Under
    the legacy specification backprop runs cross-entropy and PC runs squared error, and
    `output_error` returns `target - softmax(out)` for cross-entropy -- bounded by 1 -- against
    `target - out` for squared error, which is not. So the same nominal rate is a smaller
    effective step for backprop, and backprop learns more slowly for a reason that has nothing
    to do with credit assignment. Steps-to-competence is printed below to measure exactly that.

    N_RUNS = 10 random digit pairings from seed 0. More than the 5 used since, which matters:
    the pairing accounts for a 40-point range in retention, so 12's result is at least not one
    lucky draw.

EVERY PARAMETER BELOW IS COPIED FROM 12. Nothing is inherited from the protocol, nothing is
calibrated. That is the point: this is a reproduction, not a comparison.

TWO THINGS CANNOT BE REPRODUCED EXACTLY, AND BOTH ARE NAMED RATHER THAN QUIETLY ABSORBED
    replay   12 passed no `replay_frac`, so it appended the replay batch and trained on 64
             examples per step against everyone else's 32. That default has since changed to
             0.5. It is pinned back to None here, so replay is 12's replay -- including its
             advantage of seeing twice the data.
    eqprop   12 used the patience stopping rule, which experiment 50 found truncating the
             relaxation to a third of what it needed at initialisation, and which has been
             replaced by a relative-residual test. The old rule is gone and is not coming back.
             EqProp's numbers here are therefore NOT 12's; PC, backprop and replay are.
             Since the claim under test is about PC, that is acceptable -- but it means this
             script cannot speak to 12's EqProp result at all.

    backprop and pc are bit-identical to what 12 would run: `make_backprop`, `make_pc`,
    `pc_settle` and `pc_update` have not changed since.

WHAT IS MEASURED
    Crossover height, paired per pairing against backprop -- the metric 12 was read on, and the
    one that is a property of the trade-off curve rather than of where training stopped.
    Final task-1 and task-2 accuracy alongside, because 12's own plots show PC beating replay on
    crossover while sitting far below it on final task-1 accuracy, which is precisely why one
    number is not enough.

READINGS COMMITTED BEFORE RUNNING
    If PC's crossover advantage over backprop comes back at roughly 12's size, the algorithms
    are intact and 12's result was a property of its 100-update budget and unmatched learning
    rates -- both of which later scripts deliberately removed, which is why the effect shrank.
    If it does not come back, `src` has changed behaviour since 12 and the next step is a
    line-by-line audit of PC against the version 12 ran.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np

from src.protocol import PROTOCOL, load, replace, figure_path as _figure_path, \
    array_path as _array_path
from src.methods import build_method, legacy
from src.runner import run_classil
from src.metrics import crossover, paired_diff
from src.plotting import plot_learning_curves, plot_retention_curve

SMOKE = "--smoke" in sys.argv


def _tag(f, s):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (s + "_SMOKE").lstrip("_") if (SMOKE and own) else s


def figure_path(f, s=""):
    return _figure_path(f, _tag(f, s))


def array_path(f, s=""):
    return _array_path(f, _tag(f, s))


# ============ EXPERIMENT 12's CONSTANTS, copied verbatim ============
IMG_SIZE, BASE_SEED, N_RUNS = 14, 0, 10
ITERS, BATCH, EVAL_EVERY, EVAL_PER_CLASS = 100, 32, 1, 100
BP_LR = 0.05
RP_LR, RP_PER_CLASS = 0.05, 20
EQP_LR, EQP_BETA, EQP_DT, EQP_MAX_STEPS = 0.005, 0.3, 0.3, 500
PC_LR, PC_DT, PC_STEPS = 0.05, 0.1, 50
METHODS = ["backprop", "replay", "eqprop", "pc"]
# ====================================================================

COLORS = {"backprop": "tab:gray", "replay": "tab:brown",
          "eqprop": "tab:green", "pc": "tab:red"}
TASK_COLORS = ["tab:orange", "tab:blue"]

if SMOKE:
    N_RUNS, ITERS, EVAL_EVERY = 2, 20, 5
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"experiment 12 as written: class-IL 2x5, {ITERS} updates per task, {N_RUNS} pairings, "
      f"seed {BASE_SEED}")
print(f"  lr: backprop {BP_LR}, replay {RP_LR}, pc {PC_LR}, eqprop {EQP_LR}  "
      f"(NOT matched on learning speed -- 12 did not match them)\n")

# 12 ran Class-IL 2x5 at 14x14 with the legacy per-rule specification.
base = replace(PROTOCOL, img_size=IMG_SIZE, hidden=64, scenario="class_il",
               eval_per_class=EVAL_PER_CLASS, batch=BATCH, eval_every=EVAL_EVERY,
               stop_threshold=None, max_iters_per_task=ITERS, seeds=N_RUNS)
data = load(base)

KW = {"backprop": dict(lr=BP_LR),
      # replay_frac=None is 12's behaviour: the replay batch is APPENDED, so replay trains on 64
      # examples per step where the others get 32. Pinned, not inherited -- the library default
      # is now 0.5, and letting that through would silently change what replay means here.
      "replay":   dict(lr=RP_LR, per_class=RP_PER_CLASS, replay_frac=None),
      "eqprop":   dict(lr=EQP_LR, beta=EQP_BETA, dt=EQP_DT, max_steps=EQP_MAX_STEPS),
      "pc":       dict(lr=PC_LR, dt=PC_DT, steps=PC_STEPS)}

REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()
if REPLOT:
    z = np.load(array_path(__file__))
    curves = {m: z[f"argmax_{m}"] for m in METHODS}
    steps, switches = z["steps"], list(z["switches"])
    print("--replot: redrawing from saved arrays, no training\n")
else:
    curves, t0 = {}, time.perf_counter()
    for m in METHODS:
        rows = []
        for r in range(N_RUNS):
            seed = BASE_SEED + r
            tasks = base.tasks(seed)                    # a random digit pairing per run, as in 12
            train_step, predict = build_method(
                m, in_dim=base.in_dim, hidden=64, out_dim=base.out_dim, seed=seed,
                device=base.device, train_data=data.train, class_idx=data.class_idx,
                **legacy(m), **KW[m])
            out = run_classil(train_step, predict, tasks, data.train, data.class_idx,
                              report_eval=data.report_eval, stop_eval=data.stop_eval,
                              max_iters_per_task=ITERS, batch=BATCH, eval_every=EVAL_EVERY,
                              device=base.device, data_seed=seed)
            rows.append(out["curves"]["argmax"])
        steps, switches = out["steps"], out["switches"]
        curves[m] = np.stack(rows)
        A = curves[m] * 100
        print(f"  {m:9s} task 1 at switch {A[:, len(steps)//2, 0].mean():5.1f}%"
              f"  ->  {A[:, -1, 0].mean():5.1f}%   task 2 {A[:, -1, 1].mean():5.1f}%"
              f"   [{time.perf_counter()-t0:5.0f}s]")

sw = switches[0]

# ---------------------------------------------------------------- readings
def xh(m, i):
    c = curves[m][i] * 100
    r = crossover(steps, c[:, 0], c[:, 1], after=sw)
    return r[1] if r[1] is not None else np.nan


print(f"\n  CROSSOVER HEIGHT -- the metric experiment 12 was read on. Paired per pairing.\n")
print(f"  {'rule':10s} {'crossover':>11s} {'vs backprop':>26s} {'final t1':>10s} "
      f"{'final t2':>10s}")
bp = [xh("backprop", i) for i in range(N_RUNS)]
summary = {}
for m in METHODS:
    v = [xh(m, i) for i in range(N_RUNS)]
    d, s, n = paired_diff(v, bp) if m != "backprop" else (0.0, 0.0, 0.0)
    summary[m] = dict(xh=float(np.nanmean(v)), d=d, sem=s, nsem=n,
                      t1=float(curves[m][:, -1, 0].mean() * 100),
                      t2=float(curves[m][:, -1, 1].mean() * 100))
    cmp_ = "--" if m == "backprop" else (f"{d:+6.2f} +-{s:5.2f} {n:4.1f}sem"
                                         + ("*" if n > 2 else " "))
    print(f"  {m:10s} {summary[m]['xh']:10.1f}% {cmp_:>26s} {summary[m]['t1']:9.1f}% "
          f"{summary[m]['t2']:9.1f}%")
print("   * = separated at 2 sem")

pc = summary["pc"]
print(f"\n  DOES EXPERIMENT 12's RESULT COME BACK?")
if pc["nsem"] > 2 and pc["d"] > 0:
    print(f"    YES -- PC's crossover is {pc['d']:+.2f} points above backprop's ({pc['nsem']:.1f} "
          f"sem).\n    The algorithms are intact. 12's result was a property of its setup: 100"
          f" updates per\n    task, and learning rates that were never matched on speed. Later"
          f" scripts removed both,\n    which is why the effect shrank rather than vanished.")
else:
    print(f"    NO -- PC's crossover is {pc['d']:+.2f} points ({pc['nsem']:.1f} sem), not the "
          f"advantage 12\n    reported. Something in `src` behaves differently from when 12 ran."
          f" Audit PC against\n    the version 12 used before anything else.")
print(f"\n  Note PC vs replay on the two metrics -- 12's plots show the same tension:")
print(f"    crossover  pc {summary['pc']['xh']:.1f}%  replay {summary['replay']['xh']:.1f}%")
print(f"    final t1   pc {summary['pc']['t1']:.1f}%  replay {summary['replay']['t1']:.1f}%")
print(f"  One metric does not tell the whole story, which is why both are reported.")

# ---------------------------------------------------------------- figures
plot_learning_curves(steps, curves, METHODS, figure_path(__file__),
                     blocks=[(0, sw, 0), (sw, steps[-1], 1)], ncols=2,
                     task_colors=TASK_COLORS, task_labels=["task 1", "task 2"],
                     crossover_after=sw,
                     legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False))

i_sw = int(np.argmin(np.abs(np.asarray(steps) - sw)))
plot_retention_curve({m: [curves[m][i, i_sw:, :] for i in range(N_RUNS)] for m in METHODS},
                     METHODS, figure_path(__file__, "retention"), colors=COLORS,
                     chance=1.0 / base.n_classes,
                     title="Experiment 12 re-run — task 1 against task-2 progress")

np.savez(array_path(__file__), steps=np.asarray(steps), switches=np.asarray(switches),
         methods=np.asarray(METHODS), iters=ITERS, n_runs=N_RUNS,
         **{f"argmax_{m}": curves[m] for m in METHODS},
         **{f"{k}_{m}": summary[m][k] for m in METHODS for k in summary[m]})
print(f"\nsaved {array_path(__file__)}")
