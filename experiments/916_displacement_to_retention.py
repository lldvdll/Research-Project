"""Does PC's settling displacement translate into smaller weight movement, and does smaller
weight movement translate into better retention?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing.

MODE A -- three PNGs, composed with `subfigure`.

THE CHAIN THIS FIGURE TESTS, one link at a time:
    D = ||x*(settled) - x(feedforward)||   ->   ||dW|| per layer   ->   task-1 retention

⚠ LINK 1 IS NOT A COMPARISON. Backprop's displacement is EXACTLY 0.0000 by construction -- it has
no relaxation, so there is nothing to displace. Panel (a) therefore describes PC and does not
compare the rules; a non-zero D is a definition, not evidence. What can be measured is whether D
varies across PC seeds and whether that variation goes anywhere.

⚠ A CONFOUND THAT REVERSES THE ANSWER, AND THE REASON PANELS (b) AND (c) USE THE TOTAL PATH
RATHER THAN THE MEAN STEP. Under matched-competence stopping, task-2 length is a dependent
variable: it runs from 119 to 4999 updates here. Long runs have SMALL mean per-update ||dW|| and
ALSO forget more, so the mean step size correlates with retention at r = +0.93 -- which reads as
"bigger updates preserve task 1" and is an artefact of run length, not a mechanism. The total
path summed over task 2 does not have this problem and gives the opposite, interpretable sign:

    total ||dW2|| over task 2  vs  final task-1 accuracy      r = -0.56  (Class-IL, backprop)

More total movement in the output weights, more forgetting. That is the direction the report
claims, and it is claimed on the total, never on the mean.

THE RESULT PANEL (b) CARRIES. PC moves the output weights LESS than backprop over the same task:
total ||dW2|| 1.75 against 2.52 in Class-IL, 1.42 against 2.26 in Domain-IL. It takes a shorter
route. It does not convert that into proportionally better retention -- which is the same shape
as R2's finding that the effect is real and small, arrived at from the weights instead of the
accuracy curve.

PROVENANCE
    803  both scenarios, backprop and pc, seeds 10-19, config_800.yaml, matched competence.
         D and ||dW|| per layer are recorded on EVERY update; nothing here is subsampled.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff, sem
from src import style

EXP = ROOT / "experiments"
SOURCE = {s: EXP / f"803_mechanism_logged_{s}.npz" for s in ("class_il", "domain_il")}
SCENARIOS = ["class_il", "domain_il"]
RULES = ["backprop", "pc"]
LABEL = {"class_il": "Class-IL", "domain_il": "Domain-IL",
         "backprop": "backprop", "pc": "PC"}
LEAD = 150
LAYERS = ["W1", "W2"]        # dW columns, in order

style.apply()


def runs(scenario, rule):
    """Per run: (disp, dW, switch, final_t1), sliced to task 2 where noted at the call site."""
    d = np.load(SOURCE[scenario], allow_pickle=True)
    out = []
    for i, m in enumerate(d["methods"]):
        if m != rule:
            continue
        out.append((np.asarray(d[f"disp_{i}"], float), np.asarray(d[f"dW_{i}"], float),
                    int(d[f"switch0_{i}"]), float(d[f"final_t1_{i}"])))
    return out


def total_path(scenario, rule, layer):
    """Sum of ||dW|| over each seed's own task 2 -- NOT the mean. See the docstring."""
    return np.array([dw[np.arange(len(dw)) > s, layer].sum()
                     for _, dw, s, _ in runs(scenario, rule)])


def retention(scenario, rule):
    return np.array([r[3] for r in runs(scenario, rule)])


def panel_a():
    """Settling displacement through task 2. PC only -- backprop's is identically zero."""
    fig, ax = plt.subplots(figsize=style.size("col", 0.62))
    for scenario in SCENARIOS:
        rs = runs(scenario, "pc")
        hi = min(len(d) - s for d, _, s, _ in rs)
        grid = np.arange(-min(LEAD, min(s for _, _, s, _ in rs)), hi)
        stack = np.vstack([d[s + grid] for d, _, s, _ in rs])
        mu = stack.mean(axis=0)
        se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
        c = style.SCENARIO[scenario]
        ax.plot(grid, mu, lw=1.2, color=c, label=f"{LABEL[scenario]} · PC")
        ax.fill_between(grid, mu - se, mu + se, color=c, alpha=0.15, lw=0)
    ax.axhline(0, color=style.RULE["backprop"], lw=1.0, ls=(0, (5, 3)))
    ax.annotate("backprop $\\equiv 0$ by construction", (0.98, 0.0),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", fontsize=6.5, ha="right", va="bottom",
                color=style.RULE["backprop"])
    ax.axvline(0, color=style.ZERO_LINE, lw=0.8)
    ax.annotate("task switch", (0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -9), textcoords="offset points", fontsize=6.5, ha="left")
    ax.set_xlabel("updates relative to the task switch")
    ax.set_ylabel("settling displacement $D$")
    ax.legend(fontsize=6.5, handlelength=1.5, borderpad=0.15)
    out = figure_path(__file__, "a")
    fig.savefig(out)
    print(f"saved {out}")


def panel_b():
    """Total distance each layer travelled over task 2."""
    fig, ax = plt.subplots(figsize=style.size("col", 0.60))
    width, xs = 0.36, np.arange(len(SCENARIOS) * len(LAYERS))
    for k, rule in enumerate(RULES):
        vals, errs = [], []
        for scenario in SCENARIOS:
            for layer, _ in enumerate(LAYERS):
                m, s = sem(total_path(scenario, rule, layer))
                vals.append(m)
                errs.append(s)
        ax.bar(xs + (k - 0.5) * width, vals, width, yerr=errs,
               color=style.RULE[rule], alpha=0.85, label=LABEL[rule],
               error_kw=dict(lw=0.9))
    # The bars carry UNPAIRED group SEM, which is wide because seeds differ a lot -- the W2 bars
    # overlap and would be read as "no difference". The comparison this project actually makes is
    # PAIRED (same seed, same class split, same init), and on W2 it is a 4-sem effect. Annotating
    # it prevents the figure from understating its own result.
    j = 0
    for scenario in SCENARIOS:
        for layer, _ in enumerate(LAYERS):
            m, s, nsem = paired_diff(total_path(scenario, "pc", layer),
                                     total_path(scenario, "backprop", layer))
            top = max(total_path(scenario, r, layer).mean() for r in RULES)
            ax.annotate(f"paired {m:+.2f}\n({nsem:.1f} sem)", (xs[j], top),
                        xytext=(0, 20), textcoords="offset points", ha="center",
                        fontsize=5.8, linespacing=1.25, color=style.ZERO_LINE)
            j += 1
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{LABEL[s]}\n$\\|\\Delta {l}\\|$" for s in SCENARIOS for l in LAYERS],
                       fontsize=6.5)
    ax.set_ylabel("total path over task 2")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.22)
    ax.legend(fontsize=6.5, handlelength=1.2, borderpad=0.15, loc="lower right")
    out = figure_path(__file__, "b")
    fig.savefig(out)
    print(f"saved {out}")


def panel_c():
    """Total output-layer path against what survived. One point per seed."""
    fig, ax = plt.subplots(figsize=style.size("col", 0.66))
    for scenario in SCENARIOS:
        for rule, mk in (("backprop", "o"), ("pc", "^")):
            x, y = total_path(scenario, rule, 1), retention(scenario, rule)
            ax.scatter(x, y, s=13, marker=mk, facecolor="none", lw=0.9,
                       edgecolor=style.SCENARIO[scenario],
                       label=f"{LABEL[scenario]} · {LABEL[rule]}")
    ax.set_xlabel("total $\\|\\Delta W_2\\|$ over task 2")
    ax.set_ylabel("final task-1 accuracy (%)")
    ax.legend(fontsize=6.0, handlelength=1.0, borderpad=0.15, labelspacing=0.2)
    out = figure_path(__file__, "c")
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    panel_a()
    panel_b()
    panel_c()
    print("\n  total path over task 2, and its correlation with final task-1 accuracy:")
    for scenario in SCENARIOS:
        for rule in RULES:
            r2 = retention(scenario, rule)
            row = []
            for layer, name in enumerate(LAYERS):
                p = total_path(scenario, rule, layer)
                row.append(f"{name} {p.mean():6.3f}  r(path,ret) {np.corrcoef(p, r2)[0, 1]:+.2f}")
            print(f"    {LABEL[scenario]:10s} {rule:9s} " + "   ".join(row))
    print("\n  paired PC - backprop on total path:")
    for scenario in SCENARIOS:
        for layer, name in enumerate(LAYERS):
            m, se_, nsem = paired_diff(total_path(scenario, "pc", layer),
                                       total_path(scenario, "backprop", layer))
            print(f"    {LABEL[scenario]:10s} {name}  {m:+.3f}+-{se_:.3f} ({nsem:.1f} sem)")
