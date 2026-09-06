"""How large is PC's effect compared with an intervention that is known to work?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing.

MODE A -- two separate PNGs, composed in the report with `subfigure` and one subcaption each.

WHY THIS IS A SEPARATE FIGURE FROM 912 AND NOT A THIRD LINE ON IT. 912's y-axis spans about
+-4 points, which is the range PC's effect occupies. Putting replay on that axis would either
clip it or expand the range until PC's entire result -- the thing 912 exists to show -- collapsed
into a band a few pixels high. Both figures are needed and they must be read in order: 912 asks
whether the PC effect is real, 913 asks whether it is large. The answers are yes and no.

REPLAY IS THE POSITIVE CONTROL, NOT A COMPETITOR. It is not a biologically-motivated learning
rule and the report does not propose it. It is here to establish that the problem is solvable at
all -- without it, every small or null effect in this project could be read as "2x5 MNIST at H=32
simply cannot be improved", and no figure would contradict that. Replay contradicts it.

    (a)  the headline, at the working configuration: paired change in crossover against backprop,
         for PC and for replay, both scenarios. One number each, with its SEM.
    (b)  the same three sweeps as 912, replotted with replay included so the two effects share a
         y-axis. Replay stays positive and roughly flat across every lr, width and depth, so its
         advantage is not a tuning artefact -- and PC's line is the flat one near zero.

WHAT (b) IS NOT SAYING. Replay being flat across the sweeps does not make it free: it stores and
re-presents task-1 data, which is exactly the resource a continual-learning rule is supposed not
to need. The comparison is of EFFECT SIZE under a fixed protocol, not of cost, and the report says
so where it cites this figure.

PROVENANCE -- config_300.yaml, seeds 10-19, 90% matched competence, replay arm present in all
three sweeps:
    340  lr sweep, shared absolute grid   341  width sweep   342  depth sweep
The working point in (a) is taken from 341 at H = 32 (identical to 342 at depth 1: the same
configuration, trained independently in the two scripts -- agreement between them is a correctness
check, and they agree to 0.01 pp on backprop's mean crossover). Each rule runs at its established
lr there, backprop 0.01 and PC 0.02, which is the pairing 314 licenses.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff, paired_sign
from src import style

EXP = ROOT / "experiments"
SWEEPS = [
    ("340_lr_sweep",    "learning rate",   "log"),
    ("341_width_sweep", "hidden width $H$", "log"),
    ("342_depth_sweep", "hidden layers",   "linear"),
]
SCENARIOS = ["class_il", "domain_il"]
LABEL = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
COL_CX = 5                                   # crossover height, per the sweep row layout
WORKING = ("341_width_sweep", 32)            # H = 32, depth 1 -- the report's configuration
TREATMENTS = ["pc", "replay"]
TLABEL = {"pc": "predictive coding", "replay": "replay"}

style.apply()


def load(stem, scenario):
    return np.load(EXP / f"{stem}_{scenario}.npz", allow_pickle=True)["data"]


def by_seed(rows, method, x):
    d = {int(r[2]): float(r[COL_CX]) for r in rows if r[0] == method and r[1] == x}
    return np.array([d[s] for s in sorted(d)], dtype=float)


def panel_a():
    """The headline: two interventions, two scenarios, one number each."""
    stem, x = WORKING
    fig, ax = plt.subplots(figsize=style.size("col", 0.62))
    ax.axvline(0, color=style.ZERO_LINE, lw=0.7, zorder=1)

    ypos, labels = [], []
    y = 0.0
    for scenario in SCENARIOS:
        rows = load(stem, scenario)
        bp = by_seed(rows, "backprop", x)
        for treatment in TREATMENTS:
            m, se, _ = paired_diff(by_seed(rows, treatment, x), bp)
            ax.errorbar(m, y, xerr=se, marker="o", ms=3.4, lw=0, elinewidth=1.3,
                        color=style.RULE[treatment], capsize=1.8)
            ax.annotate(f"{m:+.2f}", (m, y), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=6.5, color=style.RULE[treatment])
            ypos.append(y)
            labels.append(TLABEL[treatment])
            y -= 1.0
        y -= 0.6                              # gap between the two scenario blocks

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_ylim(min(ypos) - 0.8, max(ypos) + 0.8)
    ax.set_xlabel("$\\Delta$ crossover vs backprop (pp)")
    # Scenario names as block headings on the right, so the y-axis stays a single clean list.
    # Centred on the block, not on its first row -- otherwise the label reads as belonging to
    # the PC row alone.
    n = len(TREATMENTS)
    for i, scenario in enumerate(SCENARIOS):
        block = ypos[i * n:(i + 1) * n]
        ax.text(1.005, float(np.mean(block)), LABEL[scenario], transform=
                ax.get_yaxis_transform(), va="center", ha="left", fontsize=7, rotation=270)
    ax.spines["right"].set_visible(False)
    out = figure_path(__file__, "a")
    fig.savefig(out)
    print(f"saved {out}")


def panel_b():
    """Replay and PC on one y-axis, across every swept configuration."""
    w = style.WIDTH["page"] / 3.0
    fig, axes = plt.subplots(1, 3, figsize=(3 * w, w * 0.92), sharey=True)
    for c, (stem, xlabel, xscale) in enumerate(SWEEPS):
        ax = axes[c]
        ax.axhline(0, color=style.ZERO_LINE, lw=0.7, zorder=1)
        for scenario in SCENARIOS:
            rows = load(stem, scenario)
            grid = np.asarray(sorted({r[1] for r in rows}), dtype=float)
            for treatment, ls in (("replay", "-"), ("pc", "--")):
                ms, ses = [], []
                for g in sorted({r[1] for r in rows}):
                    bp = by_seed(rows, "backprop", g)
                    d, se, _ = paired_diff(by_seed(rows, treatment, g), bp)
                    ms.append(d)
                    ses.append(se)
                ax.errorbar(grid, ms, yerr=ses, ls=ls, lw=1.2, marker="o", ms=2.6,
                            color=style.SCENARIO[scenario], alpha=1.0 if treatment == "replay"
                            else 0.55)
        ax.set_xscale(xscale)
        if xscale == "log":
            ax.set_xticks(sorted({float(r[1]) for r in load(stem, SCENARIOS[0])}))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.minorticks_off()
        ax.set_xlabel(xlabel)
        style.panel_label(ax, "abc"[c], dx=-0.22)
    axes[0].set_ylabel("$\\Delta$ crossover vs backprop (pp)")
    handles = [plt.Line2D([], [], color=style.SCENARIO[s], ls=ls, lw=1.2,
                          alpha=1.0 if t == "replay" else 0.55)
               for s in SCENARIOS for t, ls in (("replay", "-"), ("pc", "--"))]
    # Lower left: the only corner of panel (a) that no series passes through. Upper left sits on
    # the Class-IL replay line, which is the series the figure is about.
    axes[0].legend(handles, [f"{LABEL[s]} · {TLABEL[t]}" for s in SCENARIOS
                             for t in ("replay", "pc")],
                   loc="lower left", fontsize=5.8, handlelength=1.5, borderpad=0.15,
                   labelspacing=0.2)
    out = figure_path(__file__, "b")
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    panel_a()
    panel_b()
    stem, x = WORKING
    print(f"\n  working point: {stem} at {x}")
    for scenario in SCENARIOS:
        rows = load(stem, scenario)
        bp = by_seed(rows, "backprop", x)
        line = [f"{LABEL[scenario]:10s} backprop {np.nanmean(bp):5.2f}"]
        for treatment in TREATMENTS:
            v = by_seed(rows, treatment, x)
            m, se, _ = paired_diff(v, bp)
            w_, l_, t_, p = paired_sign(v, bp, censored_is_best=True)
            line.append(f"{treatment} {m:+5.2f}+-{se:.2f} ({w_}W-{l_}L-{t_}T p={p:.3f})")
        print("   " + "   ".join(line))
