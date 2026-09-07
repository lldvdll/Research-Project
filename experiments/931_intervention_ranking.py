"""Of everything we can add, what actually recovers retention?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Discussion D1.

ONE PLOT, one grid cell, form from 007's SK_INTERV: every intervention on one axis as paired
Δ crossover against its own backprop control, both scenarios, with a zero line and the harmful
ones drawn below it.

WHY IT SITS IN DISCUSSION AND NOT RESULTS. Placing the learning-rule result beside every
alternative is what stops it being read as larger than it is -- but the alternatives were not all
measured to the same standard, so this is a PRELIMINARY EVALUATION and the section says so.
Ranking them as though they were is the failure mode this figure has to avoid.

⚠ EWC'S LAMBDA GRID NEVER BRACKETED ITS OPTIMUM. Its number is a LOWER BOUND, not a tuned result,
and it is drawn with an open marker and an arrow to say so. Ranking it against SI or replay as if
both had been searched equally would be dishonest. Everything else here was swept.

⚠ THE INTERVENTIONS ARE NOT COMPARABLE IN COST, ONLY IN EFFECT. Replay stores and re-presents
task-1 data -- the exact resource a continual-learning rule is meant not to need. Masking requires
knowing which classes are absent, which is task-identity information the Domain-IL setting does
not have and which is why the bar is missing there rather than zero. The axis measures effect size
under one fixed protocol; the text supplies the cost.

THE ORDERING, paired Δ crossover against backprop at the working point:
    replay      +9.84 / +3.43      the positive control -- the problem IS solvable
    masking     Class-IL only, and large on the endpoint (+46 in 922) though not on crossover
    SI, EWC     small positives
    PC          +1.42 / −0.72      the report's own subject
    freezing    nothing, either scenario
    k-WTA       strongly NEGATIVE, and worse for PC than backprop

The defensible summary: prospective configuration produces a small, real, scenario-dependent
difference, roughly a seventh of what replay buys. The credit-assignment rule is not the lever.
That does not refute Song & Bogacz -- it locates their effect at a size, under a protocol that
reports its censoring and pairs its seeds.

Styling follows the 300-series scripts: tab: colours by scenario, dpi 120, bbox_inches="tight",
9pt labels. No shared style module.

PROVENANCE
    341_width_sweep_{scenario}.npz  replay and pc at H = 32, seeds 10-19, current protocol
    200_ewc_lambda_sweep.npz        ⚠ lambda grid did NOT bracket the optimum
    210_si_lambda_sweep.npz         swept
    220_kwta_k_sweep.npz            swept
    130_freeze_factorial.npz        freeze-W2 against its control
All are PRE-800 except the sweeps; each is drawn against its OWN control from the same script, so
protocol differences between scripts cannot leak into the comparison.
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
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
CX = 5


def sweep_at(stem, x, treat):
    out = {}
    for s in SCEN:
        rows = np.load(EXP / f"{stem}_{s}.npz", allow_pickle=True)["data"]
        g = lambda m: np.array([float(r[CX]) for r in
                                sorted((r for r in rows if r[0] == m and r[1] == x),
                                       key=lambda r: int(r[2]))], dtype=float)
        out[s] = paired_diff(g(treat), g("backprop"))[:2]
    return out


def sweep_cell(fname, key, ctrl_from, pick="best", skip=()):
    """One cell of a lambda/k sweep against the shared control, per scenario.

    ⚠ `skip` exists because 220's k = H = 32 cell is NOT an intervention -- gating all 32 units is
    the control, and it returns +0.00 +- 0.00 exactly. Left in, "best k" selects it and k-WTA
    reads as harmless, which is the opposite of the finding.
    """
    z = np.load(EXP / fname, allow_pickle=True)
    ctl = np.load(EXP / ctrl_from, allow_pickle=True)
    grid = z["k_values"] if "k_values" in z.files else z["lambdas"]
    out = {}
    for s in SCEN:
        arr = np.asarray(z[f"crossover_{s}_backprop_{key}"], float)
        base = np.asarray(ctl[f"crossover_{s}_backprop_control"], float)
        cand = [(paired_diff(arr[i], base)[:2], g) for i, g in enumerate(grid) if g not in skip]
        out[s] = (max(cand, key=lambda t: -np.inf if np.isnan(t[0][0]) else t[0][0])[0]
                  if pick == "best" else
                  next(c for c, g in cand if g == pick))
    return out


def freeze_w2():
    z = np.load(EXP / "130_freeze_factorial.npz", allow_pickle=True)
    return {s: paired_diff(np.asarray(z[f"crossover_{s}_backprop_freeze_w2"], float),
                           np.asarray(z[f"crossover_{s}_backprop_control"], float))[:2]
            for s in SCEN}


ROWS = [
    ("replay",                sweep_at("341_width_sweep", 32, "replay"), False),
    ("SI (best $\\lambda$)",  sweep_cell("210_si_lambda_sweep.npz", "si",
                                            "130_freeze_factorial.npz"), False),
    ("EWC (best $\\lambda$)", sweep_cell("200_ewc_lambda_sweep.npz", "ewc",
                                            "130_freeze_factorial.npz"), True),
    ("predictive coding",     sweep_at("341_width_sweep", 32, "pc"), False),
    ("freeze $W_2$",          freeze_w2(), False),
    # k = 3, the strongest gating. k-WTA is harmful at EVERY genuine k and monotonically worse as
    # gating tightens (Class-IL backprop -2.47, -3.39, -6.43, -9.78 for k = 24, 16, 8, 3), so the
    # strongest setting is the honest summary of a harmful intervention -- "best k" would report
    # its least-bad cell and bury the finding.
    ("k-WTA ($k=3$)",         sweep_cell("220_kwta_k_sweep.npz", "kwta",
                                         "130_freeze_factorial.npz", pick=3, skip=(32,)), False),
]

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(ROWS))
    width = 0.36
    for k, s in enumerate(SCEN):
        mu = [r[1][s][0] for r in ROWS]
        se = [r[1][s][1] for r in ROWS]
        bars = ax.bar(x + (k - 0.5) * width, mu, width, yerr=se, capsize=3,
                      color=COLORS[s], alpha=0.85, label=NICE[s])
        for i, (lab, vals, lower_bound) in enumerate(ROWS):
            if lower_bound:
                bars[i].set_alpha(0.4)
                bars[i].set_hatch("//")
    ax.axhline(0, color="k", lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in ROWS], fontsize=8.5)
    ax.set_ylabel("paired $\\Delta$ crossover vs backprop (pp)", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper right")
    ax.annotate("hatched = EWC, whose λ grid never bracketed its optimum:\n"
                "a LOWER BOUND, not a tuned result",
                (0.02, 0.04), xycoords="axes fraction", fontsize=7.5, color="crimson",
                linespacing=1.3)
    ax.set_title("Preliminary evaluation — effect size under one protocol, not cost", fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    for lab, vals, lb in ROWS:
        print(f"  {lab:22s} " + "   ".join(
            f"{NICE[s]} {vals[s][0]:+6.2f} ± {vals[s][1]:4.2f}" for s in SCEN)
            + ("   [LOWER BOUND: grid did not bracket]" if lb else ""))
