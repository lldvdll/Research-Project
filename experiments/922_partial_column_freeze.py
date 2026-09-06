"""Does masking work by sparing the task-1 output weights?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing.

MODE A -- two PNGs, composed with `subfigure`.

THE CONTRADICTION THIS RESOLVES. Masking the absent classes recovers a great deal (120, and
again here). Freezing the whole output layer recovers nothing (130, +0.36 +- 0.43). Both spare
the task-1 output weights during task 2, so on the naive reading they should agree. 806 separates
them by adding the condition that was missing: freeze ONLY the task-1 columns of W2 and the
matching entries of b2 -- masking's intervention expressed in weight space -- leaving task 2's
readout free to learn.

⚠ THE ANSWER IS THE SECOND OF THE THREE PRE-COMMITTED READINGS, so it cannot be rationalised
after the fact. 806's docstring recorded, before the run, that
    freeze_w2_t1 ~ mask      -> the mechanism IS the task-1 readout weights being driven down
    freeze_w2_t1 ~ control   -> masking works some other way and the readout account is wrong
    in between               -> both effects are real, report their relative size

Final task-1 accuracy, Class-IL, ten seeds:

    condition                       backprop    pc
    control                              4.8   6.0
    freeze all of W2                     3.2   4.3
    freeze task-1 columns of W2 + b2     6.7   6.3
    mask                                50.9  44.7

Masking recovers about +46 points. Freezing exactly the weights masking spares recovers about
+1.9. That is the second branch: MASKING DOES NOT WORK BY SPARING THOSE WEIGHTS. Whatever it
does, it does through something the weight-space freeze does not reproduce -- the remaining
candidate being that masking changes what W1 learns, the trunk reshaped to serve an objective
with no competing targets. That is not a readout account, and R5 is written accordingly.

NOTE FREEZING THE WHOLE OUTPUT LAYER IS WORSE THAN THE CONTROL (3.2 against 4.8). It spares the
task-1 columns too, but it also blocks task 2 from learning through the readout at all, forcing
that learning into W1 and damaging the shared trunk. The three freeze conditions therefore
bracket the effect rather than merely repeating it.

⚠ CENSORING, NOT MISSING DATA. Under masking, backprop's crossover is undefined on 3 of 10 seeds
because task 1 NEVER FELL BELOW task 2 -- the best outcome the metric can encounter, and the
reason `paired_sign(censored_is_best=True)` exists. Panel (b) draws crossover and annotates the
censored count rather than averaging over the seven that produced a number, which would report
masking using only its three worst runs.

WHY BOTH PANELS, AND THE DISSOCIATION BETWEEN THEM. Showing only (a) would quietly switch to a
metric the rest of the report does not use, for the one result where the usual metric says
little; showing only (b) would hide a 46-point effect. Reporting both exposes something neither
shows alone -- the two interventions act on DIFFERENT parts of the curve:

    paired vs control          final task-1            crossover
    freeze task-1 columns    +1.87 +- 1.03 (1.8)   +1.16 +- 0.23 (5.0 sem, 10W-0L)
    mask                    +46.11 +- 4.52 (10.2)  +0.20 +- 1.96 (0.1 sem, 3 censored)

Freezing the task-1 readout produces a small but highly consistent shift in WHERE THE CURVES
CROSS and almost nothing at the end. Masking produces an enormous shift in WHERE THE CURVES END
and nothing at the crossing. They are not weak and strong versions of one mechanism; they act at
different times. That is a second, independent reason to reject "masking works by sparing the
task-1 output weights" -- if it did, the two rows would differ in size but agree in shape.

PROVENANCE
    806  Class-IL only -- Domain-IL has no absent classes, so neither masking nor a task-1 column
         freeze is definable there, which is the same structural asymmetry R1 rests on.
         backprop and pc, seeds 10-19, config_800.yaml, matched competence. Freezes are applied
         at the END of task 1, so task 1 itself trains unhindered in every condition.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff, paired_sign, sem
from src import style

SOURCE = ROOT / "experiments" / "806_partial_column_freeze.npz"
CONDITIONS = ["control", "freeze_w2", "freeze_w2_t1", "mask"]
NICE = {"control": "control",
        "freeze_w2": "freeze all of $W_2$",
        "freeze_w2_t1": "freeze task-1\ncolumns of $W_2$",
        "mask": "mask absent\nclasses"}
RULES = ["backprop", "pc"]
LABEL = {"backprop": "backprop", "pc": "PC"}

style.apply()


def cells(field):
    """{(condition, rule): array over seeds} for one saved scalar."""
    d = np.load(SOURCE, allow_pickle=True)
    conds, meths, seeds = d["conditions"], d["methods"], d["seeds"]
    out = {}
    for i in range(len(conds)):
        out.setdefault((str(conds[i]), str(meths[i])), []).append(
            (int(seeds[i]), float(d[f"{field}_{i}"])))
    # Sorted by seed so paired comparisons line up row for row.
    return {k: np.array([v for _, v in sorted(vs)]) for k, vs in out.items()}


def panel(field, suffix, ylabel, paired):
    """`paired=False` draws absolute values; `paired=True` draws the difference from control.

    Which form is right depends on the range. Final task-1 accuracy runs 4.8 to 50.9, so absolute
    bars carry the result. Crossover runs 65.2 to 67.0 -- bars from zero would spend 95% of the
    panel below any data and hide a 1.16-point effect at 5 sem, so it is drawn paired against
    control, which is this project's default comparison anyway.
    """
    data = cells(field)
    conds = CONDITIONS[1:] if paired else CONDITIONS
    fig, ax = plt.subplots(figsize=style.size("col", 0.70))
    x = np.arange(len(conds))
    width = 0.36
    if paired:
        ax.axhline(0, color=style.ZERO_LINE, lw=0.7)
    for k, rule in enumerate(RULES):
        mu, se = [], []
        for cond in conds:
            v = data[(cond, rule)]
            if paired:
                m, s, _ = paired_diff(v, data[("control", rule)])
            else:
                m, s = sem(v)                   # sem() drops non-finite, i.e. censored seeds
            mu.append(m)
            se.append(s)
        ax.bar(x + (k - 0.5) * width, mu, width, yerr=se, color=style.RULE[rule],
               alpha=0.85, label=LABEL[rule], error_kw=dict(lw=0.9))
        for j, cond in enumerate(conds):
            v = data[(cond, rule)]
            n = int((~np.isfinite(v)).sum())
            if n:
                # A censored crossover means task 1 never fell below task 2 -- the BEST outcome.
                # It cannot enter paired_diff, so the bar above is computed on the seeds that
                # produced a number, i.e. this rule's worst. Saying so is not optional.
                ax.annotate(f"{n}/{v.size} censored\n(excluded from bar)",
                            (x[j] + (k - 0.5) * width, mu[j]),
                            xytext=(0, 10), textcoords="offset points", ha="center",
                            fontsize=5.5, color=style.CAPPED, linespacing=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[c] for c in conds], fontsize=6.2)
    ax.set_ylabel(ylabel)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo if paired else 0, hi * 1.16)
    ax.legend(fontsize=6.5, handlelength=1.2, borderpad=0.15,
              loc="upper left" if paired else "best")
    out = figure_path(__file__, suffix)
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    panel("final_t1", "a", "final task-1 accuracy (%)", paired=False)
    panel("crossover", "b", "$\Delta$ crossover vs control (pp)", paired=True)

    for field in ("final_t1", "crossover"):
        data = cells(field)
        print(f"\n  {field}, paired against the control:")
        for rule in RULES:
            base = data[("control", rule)]
            for cond in CONDITIONS[1:]:
                v = data[(cond, rule)]
                m, se_, nsem = paired_diff(v, base)
                w, l, t, p = paired_sign(v, base, censored_is_best=(field == "crossover"))
                cen = int((~np.isfinite(v)).sum())
                print(f"    {rule:9s} {cond:13s} {np.nanmean(v):6.2f}   "
                      f"vs control {m:+7.2f}+-{se_:5.2f} ({nsem:4.1f} sem)   "
                      f"sign {w}W-{l}L-{t}T p={p:.3f}"
                      + (f"   [{cen} censored]" if cen else ""))
