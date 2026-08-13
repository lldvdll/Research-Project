"""What does every comparison in this project look like under the FULL metric grid?

NOT AN EXPERIMENT -- a re-analysis. It trains nothing. It loads the saved arrays from the
comparisons already run and re-reports each one under every metric the project uses, paired
against backprop.

WHY THIS EXISTS
    The A series was written up on ONE metric at a time, and the choice changed the conclusion.
    Reported on endpoint retention, PC was indistinguishable from backprop everywhere and the
    project's answer was a flat no. Reported on CROSSOVER HEIGHT -- the metric experiment 12 was
    actually read on -- PC separates positively in Class-IL, +1.29 (3.7 sem) and +3.00 (2.2 sem).
    Same runs, same seeds, opposite conclusion.

    That is not a reason to prefer crossover and move on. It is a reason to report every metric
    every time, because the one that gets quoted should not be the one that was chosen after
    seeing the numbers. So this prints the whole grid for every run, and new comparisons print
    it too.

WHAT EACH METRIC IS BLIND TO is documented on `metrics.metric_grid`; the short version is that
crossover ignores the asymptote, endpoint metrics inherit the budget, half-life ignores where
the curve settles, and [R1]'s mean-test-error cannot separate "learned faster" from "forgot
less". None of them is sufficient alone, which is the point of a grid.

READING IT
    Every number is a PAIRED difference against backprop at the same seed, because the rules
    share the class split and the initialisation. BETTER/worse is flagged only past 2 sem, and
    the direction is corrected per metric -- for mean error and forgetting, lower is better.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.metrics import metric_grid, report_grid, HIGHER_IS_BETTER
from src.protocol import figure_path

RUNS = [
    ("52_do_the_rules_differ_fixed_budget.npz",        "52  Domain-IL, fixed budget"),
    ("53_do_the_rules_differ_at_matched_competence.npz", "53  Domain-IL, matched competence"),
    ("56_do_the_rules_differ_class_il.npz",            "56  Class-IL, fixed budget"),
    ("57_do_the_rules_differ_class_il_matched.npz",    "57  Class-IL, matched competence"),
    ("61_rerun_exp12_as_written.npz",                  "61  experiment 12, as written"),
]

here = Path(__file__).parent
summary = {}
for fname, label in RUNS:
    path = here / fname
    if not path.exists():
        print(f"\n=== {label}  -- not run yet, skipped")
        continue
    z = np.load(path)
    methods = [str(m) for m in z["methods"]]
    # 52/56/61 store absolute steps with a switch; 53/57 store a switch-relative grid with
    # switch = 0. Both are handled by reading whichever key is present.
    switch = 0 if "switch" in z.files else float(np.asarray(z["switches"]).ravel()[0])
    grid = {m: metric_grid(z["steps"], z[f"argmax_{m}"], switch) for m in methods}
    print(f"\n{'=' * 78}\n{label}   ({z[f'argmax_{methods[0]}'].shape[0]} runs)\n{'=' * 78}")
    report_grid(grid, methods, control="backprop", primary="crossover")
    summary[label] = grid

# ---------------------------------------------------------------- one figure, all runs
# A grid of numbers is hard to compare across experiments, so the paired difference for every
# metric is drawn on one axes per metric, with the experiments along x. Bars above zero favour
# the rule; the sign is already corrected so "up is better" holds for every panel.
if summary:
    import matplotlib.pyplot as plt
    keys = [k for k in HIGHER_IS_BETTER if k in next(iter(summary.values()))["backprop"]]
    labels = list(summary)
    rules = [m for m in next(iter(summary.values())) if m != "backprop"]
    COL = {"replay": "tab:brown", "pc": "tab:red", "eqprop": "tab:green"}
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        sign = 1.0 if HIGHER_IS_BETTER[k] else -1.0
        for j, m in enumerate(rules):
            xs, ys, es = [], [], []
            for i, lab in enumerate(labels):
                if m not in summary[lab]:
                    continue
                from src.metrics import paired_diff
                d, se, _ = paired_diff(summary[lab][m][k], summary[lab]["backprop"][k])
                xs.append(i + (j - 1) * 0.22)
                ys.append(sign * d)
                es.append(se)
            ax.errorbar(xs, ys, yerr=es, fmt="o", ms=5, capsize=3, lw=1.6,
                        color=COL.get(m, "k"), label=m)
        ax.axhline(0, color="k", lw=1)
        ax.set_title(k + ("" if HIGHER_IS_BETTER[k] else "  (sign flipped)"), fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.split()[0] for l in labels], fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Paired difference against backprop. Up is better in every panel.", fontsize=11)
    fig.tight_layout()
    fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
    print(f"\nsaved {figure_path(__file__)}")
