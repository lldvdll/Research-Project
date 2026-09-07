"""Is Class-IL forgetting the task-1 output weights being driven down, or the competition at argmax?

900-series PLOT SCRIPT. Loads a saved array, trains nothing. Results R5 -- the decisive experiment
for the contradiction at the centre of the section.

ONE PLOT, one grid cell, form from 007's SK_PFREEZE: the four conditions as bars against a zero
reference, with the two that ought to agree marked.

THE CONTRADICTION IT RESOLVES. Masking the absent classes recovers a great deal; freezing the
whole output layer recovers nothing (+0.36 ± 0.43, script 130). Both spare the task-1 output
weights during task 2, so naively they should agree. They differ in exactly one way:

    masking        is TRAIN-TIME ONLY. `active_vector` zeroes the error inside `output_error`, so
                   the task-1 units get no gradient -- while `predict` still argmaxes over all ten
                   units and task 2's units keep learning normally.
    freeze all W2  spares the task-1 columns too, but ALSO blocks task 2 from learning through the
                   readout at all, forcing it into W1 and damaging the shared trunk.

806 adds the condition that separates them: freeze ONLY the task-1 columns of W2 and the matching
entries of b2 -- masking's intervention expressed in weight space, with task 2's readout left free.

⚠ THE ANSWER IS THE SECOND OF THREE PRE-COMMITTED READINGS, recorded in 806's docstring BEFORE the
run, so it cannot be rationalised afterwards:
    freeze_w2_t1 ≈ mask      -> the mechanism IS the task-1 readout weights being driven down
    freeze_w2_t1 ≈ control   -> masking works another way; the readout account needs rewriting
    in between               -> both effects are real; report their relative size

    final task-1 accuracy, Class-IL, ten seeds        backprop   pc
        control                                            4.8   6.0
        freeze all of W2                                   3.2   4.3
        freeze task-1 columns of W2 + b2                   6.7   6.3
        mask absent classes                               50.9  44.7

    paired against control          final task-1              crossover
        freeze task-1 columns     +1.87 ± 1.03 (1.8 sem)   +1.16 ± 0.23 (5.0 sem, 10W-0L)
        mask                     +46.11 ± 4.52 (10.2 sem)  +0.20 ± 1.96 (0.1 sem, 3 censored)

Masking recovers about +46; freezing exactly the weights it spares recovers about +1.9. Twenty-five
times smaller, so MASKING DOES NOT WORK BY SPARING THOSE WEIGHTS. The remaining candidate is that
masking changes what W1 learns -- the trunk reshaped to serve an objective with no competing
targets -- which is not a readout account at all.

AND THEY DO NOT EVEN ACT AT THE SAME TIME. Freezing shifts where the curves CROSS (+1.16, 5.0 sem)
and barely moves the endpoint; masking shifts the ENDPOINT by +46 and not the crossing. One
mechanism at two strengths would differ in size but agree in shape. A second, independent reason
to reject the readout account.

⚠ FREEZING ALL OF W2 IS WORSE THAN THE CONTROL (3.2 against 4.8), because it also blocks task 2's
readout path. The three freeze conditions bracket the effect rather than repeating it.

⚠ CENSORING. Under masking, backprop's crossover is undefined on 3/10 seeds -- task 1 never fell
below task 2, the best outcome the metric can encounter. Annotated, never dropped.

Class-IL only: Domain-IL has no absent classes, so neither masking nor a task-1 column freeze is
definable there. That asymmetry is the same structural fact R1 rests on.

Styling follows the 300-series scripts: tab: colours, dpi 120, bbox_inches="tight", 9pt labels.
No shared style module.

PROVENANCE
    806_partial_column_freeze.npz   Class-IL, backprop and pc, seeds 10-19, config_800.yaml,
    matched competence. Freezes applied at the END of task 1, so task 1 trains unhindered in every
    condition.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff, paired_sign, sem

SOURCE = ROOT / "experiments" / "806_partial_column_freeze.npz"
CONDITIONS = ["control", "freeze_w2", "freeze_w2_t1", "mask"]
NICE = {"control": "control", "freeze_w2": "freeze all\nof $W_2$",
        "freeze_w2_t1": "freeze task-1\ncolumns of $W_2$", "mask": "mask absent\nclasses"}
RULES = ["backprop", "pc"]
METHOD_COLOR = {"backprop": "0.35", "pc": "tab:orange"}      # as 340 uses


def cells(field):
    d = np.load(SOURCE, allow_pickle=True)
    out = {}
    for i in range(len(d["conditions"])):
        out.setdefault((str(d["conditions"][i]), str(d["methods"][i])), []).append(
            (int(d["seeds"][i]), float(d[f"{field}_{i}"])))
    return {k: np.array([v for _, v in sorted(vs)]) for k, vs in out.items()}


if __name__ == "__main__":
    final, cross = cells("final_t1"), cells("crossover")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(CONDITIONS))
    width = 0.36

    for k, rule in enumerate(RULES):
        mu = [sem(final[(c, rule)])[0] for c in CONDITIONS]
        se = [sem(final[(c, rule)])[1] for c in CONDITIONS]
        ax.bar(x + (k - 0.5) * width, mu, width, yerr=se, capsize=3,
               color=METHOD_COLOR[rule], alpha=0.85, label=rule)

    # the pre-committed comparison, drawn: the two conditions that ought to agree, and do not
    ax.annotate("", xy=(2, 12), xytext=(3, 12),
                arrowprops=dict(arrowstyle="<->", lw=1.4, color="crimson"))
    ax.annotate("these spare the SAME weights\nand differ 25-fold", (2.5, 13.5),
                ha="center", fontsize=8, color="crimson", linespacing=1.25)

    for i, c in enumerate(CONDITIONS[1:], start=1):
        for k, rule in enumerate(RULES):
            d, se_, nsem = paired_diff(final[(c, rule)], final[("control", rule)])
            ax.annotate(f"{d:+.1f}", (x[i] + (k - 0.5) * width, sem(final[(c, rule)])[0]),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([NICE[c] for c in CONDITIONS], fontsize=8.5)
    ax.set_ylabel("final task-1 accuracy (%)", fontsize=9)
    ax.set_ylim(0, 62)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Class-IL — masking recovers +46; freezing exactly the weights it spares "
                 "recovers +1.9", fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    for field, data in (("final_t1", final), ("crossover", cross)):
        print(f"\n  {field}, paired against control:")
        for rule in RULES:
            base = data[("control", rule)]
            for c in CONDITIONS[1:]:
                v = data[(c, rule)]
                m, se_, nsem = paired_diff(v, base)
                w, l, t_, p = paired_sign(v, base, censored_is_best=(field == "crossover"))
                cen = int((~np.isfinite(v)).sum())
                print(f"    {rule:9s} {c:13s} {np.nanmean(v):6.2f}   vs control "
                      f"{m:+7.2f} ± {se_:5.2f} ({nsem:4.1f} sem)   sign {w}W-{l}L-{t_}T "
                      f"p={p:.3f}" + (f"   [{cen} censored]" if cen else ""))
