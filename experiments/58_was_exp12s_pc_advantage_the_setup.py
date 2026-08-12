"""Was experiment 12's PC advantage a property of PC, or of the setup it was measured in?

THE DISCREPANCY THIS CLOSES
    Experiment 12 -- pre-protocol, on code this project no longer trusts -- ran Class-IL and
    reported that PC forgets less than backprop. Scripts 52 and 55 find nothing: PC is
    indistinguishable from backprop on retention, and their retention curves are superimposed at
    every depth tried. Something changed between the two, and until we know what, exp 12 sits
    over the project as an unexplained contradiction.

    THREE things changed at once, and no experiment so far separates them:

      1. the SCENARIO           Class-IL then, Domain-IL now.  Scripts 56/57 test this.
      2. the OUTPUT STRUCTURE   Each rule used to get its OWN nonlinearity and loss -- backprop
                                on ReLU with cross-entropy, PC on tanh with squared error,
                                EqProp on a hinge over +-1 targets. Under that specification a
                                difference between two rules confounds the rule with its output
                                structure. All four now share tanh, a linear output, squared
                                error and one-hot targets.
      3. the LEARNING RATES     Unmatched then, matched on task-1 learning speed now (51).

    THIS SCRIPT SEPARATES 2 FROM 3, in Class-IL so that 1 is held at exp 12's value.

WHY IT MATTERS WHICH ONE IT WAS
    If the advantage was the OUTPUT STRUCTURE, exp 12 never measured a property of predictive
    coding at all -- it measured PC's squared error against backprop's cross-entropy, and the
    honest conclusion is that the old comparison was confounded and its result is void. That is
    a methodological finding and it justifies the whole standardisation effort.

    If it was the LEARNING RATES, exp 12 measured PC being slower, which is the "forgot less =
    learned less" confound this project has hit repeatedly -- most recently in script 55, where
    PC appeared to retain 14 points more at depth 3 purely by travelling less far along a shared
    curve.

    If NEITHER reproduces it, the difference is the scenario or the old code, and 56/57 are
    where the answer is.

THE THREE CELLS
      legacy spec, legacy rates    as close to exp 12 as trusted code can get
      legacy spec, matched rates   removes the learning-rate difference only
      unified spec, matched rates  the current protocol -- should reproduce script 56

    A fourth cell (unified spec, legacy rates) is omitted deliberately: the legacy rates were
    never tuned for the unified specification, so it would measure an arbitrary pairing.

    backprop, replay and PC only. EqProp costs ~350x backprop per update and the claim in
    question is about PC; if a difference appears, EqProp can be added to the cell that shows it.

READ THE RETENTION CURVE, NOT THE ENDPOINT
    Script 55 is the cautionary case: its endpoint table showed PC keeping 14 points more than
    backprop, and the retention curve showed the two tracing the same path with PC stopping
    earlier. The endpoint would have been reported as a win. Here the endpoint numbers are
    printed with the task-2 accuracy beside them, and the curve is the figure.

READINGS COMMITTED BEFORE RUNNING
    If PC beats backprop in the legacy cells and not in the unified cell, exp 12's result was
    the setup and is void -- report it as such and stop treating it as evidence.
    If PC beats backprop in every cell, the setup was not the cause and the scenario or the old
    code is; 56/57 decide which.
    If PC beats backprop in NO cell, exp 12 is not reproducible on trusted code at all, which is
    the strongest reason yet to hold to the evidence standard the project already adopted.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np

from src.protocol import PROTOCOL, load, run, replace, figure_path as _figure_path, \
    array_path as _array_path
from src.model import LEGACY_SPEC
from src.methods import METHOD_DEFAULTS
from src.plotting import plot_learning_curves, plot_retention_curve

SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):
    return _array_path(f, _tag(f, suffix))


# ---------------------------------------------------------------- settings
METHODS = ["backprop", "replay", "pc"]
SEEDS = 5
EVAL_EVERY = 10
COLORS = {"backprop": "tab:gray", "replay": "tab:brown", "pc": "tab:red"}
TASK_COLORS = ["tab:orange", "tab:blue"]

HIDDEN = int(np.load(array_path(str(ROOT / "experiments" /
                                    "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(array_path(str(ROOT / "experiments" /
                             "51_matching_the_rules_on_learning_speed.py")))
MATCHED_LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
LEGACY_LR = {m: METHOD_DEFAULTS[m]["lr"] for m in METHODS}      # the pre-unification defaults
PC_STEPS = int(z51["pc_steps"])
T = float(z51["target_steps"])
ITERS = [int(2 * T), int(1.5 * T)]                              # same budgets as scripts 52/56

CELLS = {                                                        # name -> (use legacy spec, lr)
    "legacy spec, legacy lr":   (True, LEGACY_LR),
    "legacy spec, matched lr":  (True, MATCHED_LR),
    "unified spec, matched lr": (False, MATCHED_LR),
}

if SMOKE:
    SEEDS, ITERS = 1, [40, 30]
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} | class_il | {ITERS[0]}+{ITERS[1]} updates | {SEEDS} seeds")
print(f"  legacy rates  {LEGACY_LR}")
print(f"  matched rates { {m: MATCHED_LR[m] for m in METHODS} }\n")

base = replace(PROTOCOL, hidden=HIDDEN, scenario="class_il", stop_threshold=None,
               max_iters_per_task=ITERS, eval_every=EVAL_EVERY, seeds=SEEDS)

# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    cells = list(z["cells"])
    curves = {(c, m): z[f"argmax_{ci}_{m}"] for ci, c in enumerate(cells) for m in METHODS}
    steps, switches = z["steps"], list(z["switches"])
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(base)
    cells = list(CELLS)
    curves = {}
    t0 = time.perf_counter()
    for cname, (use_legacy, lrs) in CELLS.items():
        print(f"  {cname}")
        for m in METHODS:
            # The legacy specification is applied through the PROTOCOL rather than passed to the
            # builder, because `protocol.build` fills arch and obj from the protocol and would
            # collide. Same effect, and it keeps the specification visible in the object that is
            # supposed to hold every controlled setting.
            #
            # Under the legacy cells each rule genuinely gets a different nonlinearity AND a
            # different loss -- backprop ReLU + cross-entropy, PC tanh + squared error. That is
            # the confound being tested, reproduced deliberately rather than inherited.
            proto = replace(base, lr={m: lrs[m]})
            if use_legacy:
                a, o = LEGACY_SPEC[m]
                proto = replace(proto, act=a.act, bias=a.bias, init=a.init,
                                loss=o.loss, target=o.target)
            kw = dict(steps=PC_STEPS) if m == "pc" else {}
            rows = []
            for seed in range(SEEDS):
                out = run(proto, m, seed, data=data, **kw)
                rows.append(out["curves"]["argmax"])
            steps, switches = out["steps"], out["switches"]
            curves[(cname, m)] = np.stack(rows)
            A = curves[(cname, m)] * 100
            i_sw = int(np.argmin(np.abs(np.asarray(steps) - switches[0])))
            print(f"    {m:9s} task 1 {A[:, i_sw, 0].mean():5.1f}% -> {A[:, -1, 0].mean():5.1f}%"
                  f"  (task 2 {A[:, -1, 1].mean():5.1f}%)   [{time.perf_counter()-t0:5.0f}s]")

i_sw = int(np.argmin(np.abs(np.asarray(steps) - switches[0])))

# ---------------------------------------------------------------- readings
print(f"\n  task 1 kept, and task 2 reached, at the end of the budget. mean +- SEM over "
      f"{SEEDS} seeds.")
print(f"  A rule that kept more AND reached less did not retain better -- it learned less.\n")
print(f"  {'cell':26s} " + " ".join(f"{m:>22s}" for m in METHODS))
verdict = {}
for c in cells:
    row = []
    for m in METHODS:
        A = curves[(c, m)] * 100
        k, t2 = A[:, -1, 0], A[:, -1, 1]
        row.append(f"{k.mean():5.1f}+-{k.std(ddof=1)/np.sqrt(SEEDS):3.1f} / {t2.mean():4.1f}")
    print(f"  {c:26s} " + " ".join(f"{r:>22s}" for r in row))
    bp, pc = curves[(c, "backprop")][:, -1, 0] * 100, curves[(c, "pc")][:, -1, 0] * 100
    d = pc.mean() - bp.mean()
    se = np.hypot(bp.std(ddof=1), pc.std(ddof=1)) / np.sqrt(SEEDS)
    verdict[c] = (d, se, d > 2 * se)
    print(f"  {'':26s} pc - backprop = {d:+5.1f} points, SE {se:4.1f}  ->  "
          + ("PC RETAINS MORE" if d > 2 * se else
             "PC RETAINS LESS" if -d > 2 * se else "not separated"))

# A CELL IS ONLY A COMPARISON IF ITS RULES REACHED COMPARABLE COMPETENCE ON TASK 2. This has to
# be checked before anything is said about retention, and the first run of this script proved it:
# under the legacy specification backprop reached only 50% on task 2 where PC reached 89%, so it
# "kept" 89% of task 1 by never learning the new task. The verdict logic read the retention
# column, ignored the task-2 column printed beside it, and announced a conclusion.
print(f"\n  IS EACH CELL A VALID COMPARISON? (task-2 accuracy must be comparable across rules)")
valid = {}
for c in cells:
    t2 = {m: curves[(c, m)][:, -1, 1].mean() * 100 for m in METHODS}
    gap = max(t2.values()) - min(t2.values())
    valid[c] = gap <= 10.0
    print(f"    {c:26s} task 2: " + ", ".join(f"{m} {v:.0f}%" for m, v in t2.items())
          + f"   spread {gap:.0f} pts  " + ("ok" if valid[c] else "INVALID, not a comparison"))

print(f"\n  WHAT THIS SAYS ABOUT EXPERIMENT 12")
if not all(valid.values()):
    bad = [c for c in cells if not valid[c]]
    print(f"    {len(bad)} of {len(cells)} cells are not comparisons: "
          + "; ".join(bad) + ".")
    print("    In those cells one rule failed to learn task 2 at all, so its high retention is")
    print("    'learned less', not 'forgot less'. Nothing about PC can be concluded from them.")
    print("    What they DO show is that the legacy specification is badly unbalanced -- it")
    print("    leaves backprop unable to learn task 2 while PC learns it fine -- which is an")
    print("    argument for the standardisation, not evidence about either rule.")
    print("    To test exp 12 properly the learning rates must be calibrated WITHIN the legacy")
    print("    specification, or both tasks stopped on accuracy so competence is matched.")
leg, mat, uni = (verdict[c][2] and valid[c] for c in cells)
if leg and not uni:
    print("    PC wins under the legacy setup and not under the unified one, so exp 12's result"
          "\n    was the SETUP, not the rule. Which part: "
          + ("the OUTPUT STRUCTURE -- it survives matching the learning rates."
             if mat else
             "the LEARNING RATES -- it disappears once they are matched."))
    print("    Either way exp 12 measured a confound and should not be cited as evidence "
          "about PC.")
elif leg and uni:
    print("    PC wins in every cell, so the setup is NOT what changed. The scenario or the old"
          "\n    code is responsible -- scripts 56 and 57 decide which.")
else:
    print("    PC does not beat backprop in ANY cell, including the one built to reproduce exp"
          "\n    12's conditions. Exp 12 is not reproducible on trusted code.")

# ---------------------------------------------------------------- figures
names = [f"{c} | {m}" for c in cells for m in METHODS]
plot_learning_curves(
    steps, {f"{c} | {m}": curves[(c, m)] for c in cells for m in METHODS}, names,
    figure_path(__file__), blocks=[(0, switches[0], 0), (switches[0], steps[-1], 1)], ncols=3,
    task_colors=TASK_COLORS, task_labels=["task 1", "task 2"],
    legend_kw=dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False),
)

# The figure that decides it. Script 55's endpoint table claimed a 14-point PC advantage that
# this plot showed to be PC stopping earlier on a shared curve; the same trap is live here.
for c in cells:
    plot_retention_curve(
        {m: [curves[(c, m)][i, i_sw:, :] for i in range(SEEDS)] for m in METHODS},
        METHODS, figure_path(__file__, "retention_" + c.split(",")[0].replace(" ", "_")
                             + ("_legacylr" if "legacy lr" in c else "_matchedlr")),
        colors=COLORS, chance=1.0 / base.n_classes,
        title=f"{c}\nTask 1 retained against task-2 progress. Curves apart = a real difference.",
    )

np.savez(array_path(__file__), steps=np.asarray(steps), switches=np.asarray(switches),
         hidden=HIDDEN, cells=np.asarray(cells), methods=np.asarray(METHODS),
         **{f"argmax_{ci}_{m}": curves[(c, m)]
            for ci, c in enumerate(cells) for m in METHODS})
print(f"\nsaved {array_path(__file__)}")
