"""Of everything we can add, what actually recovers retention -- and does it recover it the same
way for both rules, or is the rule itself part of the story here too?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Discussion D1.

REGROUPED FROM THE ORIGINAL VERSION, which listed "predictive coding" as ONE ROW among the
interventions -- a category error, since PC is not an intervention applied on top of a base
rule the way freezing/masking/EWC/SI/k-WTA are, it IS one of the two base rules. Every
intervention below is now measured against its OWN rule's control, backprop AND pc separately,
so the plot answers two questions at once: does the intervention help (bar vs zero), and does
the RULE it's applied to change how much it helps (backprop-bar vs pc-bar, same intervention).
That second question is the one this project keeps finding "backprop-only" answers for without
checking PC -- 918 was wrong to skip it, 923 turned out to be fine; this figure was wrong too.

TWO VERTICAL PANELS (Class-IL, Domain-IL), not one flat bar chart -- adjacent bars are
backprop-base vs pc-base for the SAME intervention, so both the "does it help" and "does the rule
matter" readings sit in one glance instead of needing the rule buried in a legend.

⚠ MASKING HAS NO DOMAIN-IL BAR. Not zero -- absent. Masking requires task-identity information
(which classes are absent) that the Domain-IL setting structurally does not have, so there is no
cell to measure, and a missing bar says that; a zero bar would say "measured and found nothing".

⚠ EWC'S LAMBDA GRID NEVER BRACKETED ITS OPTIMUM. Its number is a LOWER BOUND, not a tuned result,
drawn hatched to say so.

⚠ K-WTA IS SHOWN AT ITS STRONGEST GATING (k=3), NOT ITS BEST. It is harmful at every genuine k
and monotonically worse as gating tightens; "best k" would report its least-bad cell and bury the
finding that it is uniformly harmful, and worse for PC than backprop.

⚠ THE INTERVENTIONS ARE NOT COMPARABLE IN COST, ONLY IN EFFECT. Replay stores and re-presents
task-1 data -- the exact resource a continual-learning rule is meant not to need. The axis
measures effect size under one fixed protocol; the text supplies the cost. Replay itself has no
"base rule" to vary (it is not applied on top of backprop or PC, it IS its own rule), so it is
drawn once per scenario as a reference, not paired by rule like the rest.

PROVENANCE
    341_width_sweep_{scenario}.npz  replay at H=32, seeds 10-19, current protocol
    806_partial_column_freeze.npz   masking and column-freeze, Class-IL only, backprop AND pc
    130_freeze_factorial.npz        freeze-W2, both scenarios, backprop AND pc
    200_ewc_lambda_sweep.npz        backprop AND pc, lambda grid did NOT bracket the optimum
    210_si_lambda_sweep.npz         backprop AND pc, swept
    220_kwta_k_sweep.npz            backprop AND pc, swept
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff

EXP = ROOT / "experiments"
SCEN = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
RULES = ["backprop", "pc"]
RULE_COLOR = {"backprop": "0.35", "pc": "tab:red"}
CX = 5


def replay_at(scenario):
    rows = np.load(EXP / f"341_width_sweep_{scenario}.npz", allow_pickle=True)["data"]
    g = lambda m: np.array([float(r[CX]) for r in
                            sorted((r for r in rows if r[0] == m and r[1] == 32),
                                   key=lambda r: int(r[2]))], dtype=float)
    return paired_diff(g("replay"), g("backprop"))[:2]


def freeze_w2(scenario, rule):
    z = np.load(EXP / "130_freeze_factorial.npz", allow_pickle=True)
    treat = np.asarray(z[f"crossover_{scenario}_{rule}_freeze_w2"], float)
    ctrl = np.asarray(z[f"crossover_{scenario}_{rule}_control"], float)
    return paired_diff(treat, ctrl)[:2]


def mask_806(rule):
    """Class-IL only -- 806's own control/mask columns, both rules."""
    d = np.load(EXP / "806_partial_column_freeze.npz", allow_pickle=True)
    conds, methods = d["conditions"], d["methods"]
    def vals(cond):
        idx = sorted((i for i in range(len(conds)) if conds[i] == cond and methods[i] == rule),
                     key=lambda i: int(d["seeds"][i]) if "seeds" in d.files else i)
        return np.array([float(d[f"crossover_{i}"]) for i in idx], dtype=float)
    return paired_diff(vals("mask"), vals("control"))[:2]


def sweep_cell(fname, key, scenario, rule, pick="best", skip=()):
    z = np.load(EXP / fname, allow_pickle=True)
    ctl = np.load(EXP / "130_freeze_factorial.npz", allow_pickle=True)
    grid = z["k_values"] if "k_values" in z.files else z["lambdas"]
    arr = np.asarray(z[f"crossover_{scenario}_{rule}_{key}"], float)
    base = np.asarray(ctl[f"crossover_{scenario}_{rule}_control"], float)
    cand = [(paired_diff(arr[i], base)[:2], g) for i, g in enumerate(grid) if g not in skip]
    if pick == "best":
        return max(cand, key=lambda t: -np.inf if np.isnan(t[0][0]) else t[0][0])[0]
    return next(c for c, g in cand if g == pick)


# (label, fn(scenario, rule) -> (mean, sem) or None if not applicable, is_lower_bound)
ROWS = [
    ("freeze $W_2$",      lambda s, r: freeze_w2(s, r), False),
    ("mask",              lambda s, r: mask_806(r) if s == "class_il" else None, False),
    ("SI (best $\\lambda$)",  lambda s, r: sweep_cell("210_si_lambda_sweep.npz", "si", s, r), False),
    ("EWC (best $\\lambda$)", lambda s, r: sweep_cell("200_ewc_lambda_sweep.npz", "ewc", s, r), True),
    ("k-WTA ($k=3$)",     lambda s, r: sweep_cell("220_kwta_k_sweep.npz", "kwta", s, r,
                                                    pick=3, skip=(32,)), False),
]

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 8.4), sharex=False)
    width = 0.36

    for ax, scenario in zip(axes, SCEN):
        labels = ["replay"] + [lab for lab, _, _ in ROWS]
        x = np.arange(len(labels))
        rep_m, rep_se = replay_at(scenario)
        ax.bar([0], [rep_m], width * 2, yerr=[rep_se], capsize=3, color="tab:green",
               alpha=0.85, label="replay (own rule, reference)")
        for k, rule in enumerate(RULES):
            mus, ses, present = [], [], []
            for lab, fn, lb in ROWS:
                r = fn(scenario, rule)
                if r is None:
                    mus.append(0.0); ses.append(0.0); present.append(False)
                else:
                    mus.append(r[0]); ses.append(r[1]); present.append(True)
            xi = x[1:] + (k - 0.5) * width
            bars = ax.bar(xi, mus, width, yerr=ses, capsize=3, color=RULE_COLOR[rule],
                          alpha=0.85, label=f"{rule} base")
            for i, (ok, (lab, _, lb)) in enumerate(zip(present, ROWS)):
                if not ok:
                    bars[i].set_alpha(0.0)
                elif lb:
                    bars[i].set_alpha(0.4)
                    bars[i].set_hatch("//")
            print(f"  {NICE[scenario]:10s} {rule:9s} " + "  ".join(
                f"{lab}: {m:+.2f}±{e:.2f}" if ok else f"{lab}: n/a"
                for (lab, _, _), m, e, ok in zip(ROWS, mus, ses, present)))

        ax.axhline(0, color="k", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(f"{NICE[scenario]}\nΔ crossover vs own-rule control (pp)", fontsize=8.5)
        ax.grid(alpha=0.25, axis="y")
        ax.legend(fontsize=7.5, loc="upper right")

    axes[0].annotate("hatched = EWC, whose λ grid never bracketed its optimum: a LOWER BOUND",
                     (0.02, 0.92), xycoords="axes fraction", fontsize=7.5, color="crimson")
    axes[1].annotate("mask has no bar here — not zero, not applicable (no task-identity signal "
                     "in Domain-IL)", (0.02, 0.92), xycoords="axes fraction", fontsize=7.5,
                     color="crimson")
    fig.suptitle("Preliminary evaluation, both rules — effect size under one protocol, not cost",
                fontsize=10)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
