"""Does PC's updates aim at the target more directly than backprop's -- and does that help the
task it is not training on?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing.

MODE A -- two PNGs, composed with `subfigure`. They are separate files because the two panels'
values differ by an order of magnitude (about +0.27 against about -0.02); a shared y-axis would
flatten panel (b) into the zero line, and panel (b) is the one that bears on forgetting.

THIS IS A DIRECT TEST OF SONG & BOGACZ'S OWN CREDITED MECHANISM, and it belongs in Results
whichever way it falls. Their Fig. 3b reports target alignment,

    alignment = cos(target - out_before,  out_after - out_before)

-- of the movement an update produced in the output, how much of it went toward where the target
actually was. 1.0 is straight at it, 0 is sideways, negative is away. An earlier draft of this
report demoted alignment to an appendix negative. That was wrong: a negative result on the
mechanism a paper credits is a result about that paper, not an inconvenience.

    (a)  ON THE BATCH BEING TRAINED -- the replication of their measurement.
    (b)  ON A FIXED TASK-1 BATCH THAT IS NEVER TRAINED ON, measured during task 2. This is the
         quantity that bears on forgetting and the source paper does not report it. Negative
         means each update actively pushes task-1 outputs away from their own targets: forgetting
         caught per update, rather than inferred from an endpoint. Because it is a RATE it does
         not inherit the training budget, unlike every accuracy metric in this report.

THE COMMON WINDOW, and why the curves stop where they do. Matched-competence stopping ends each
seed at a different update -- task-2 lengths run from 115 to 4995 -- so averaging over the full
length would compute the tail from a shrinking sample, the failure 911's docstring refuses for
the same reason. Panels (a) and (b) are therefore drawn over the window EVERY seed reached, 115
updates into task 2. That is short, and it is not arbitrary: 901 puts the crossover about 57
updates into task 2, so the common window contains the event the whole report measures.

Panel (c) exists because that window still throws data away. It reports each seed's mean over
its OWN full task 2, paired PC - backprop, so the quantitative claim uses every update recorded
and does not depend on where the window was cut. Read the shape from (a) and (b); read the claim
from (c).

PROVENANCE
    803  both scenarios, backprop and pc, seeds 10-19, config_800.yaml, matched competence.
         Alignment sampled every 5th update (config_800 `mechanism.alignment_every`); the
         reference batch is 40 images per task-1 class, drawn from train and never trained on.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff
from src import style

EXP = ROOT / "experiments"
SOURCE = {s: EXP / f"803_mechanism_logged_{s}.npz" for s in ("class_il", "domain_il")}
SCENARIOS = ["class_il", "domain_il"]
RULES = ["backprop", "pc"]
LABEL = {"class_il": "Class-IL", "domain_il": "Domain-IL",
         "backprop": "backprop", "pc": "PC"}
# align[:, 0] is the update index, align[:, 1] the mean cosine over the batch.
IDX, VAL = 0, 1

style.apply()


def runs(scenario, rule):
    """Every (align, align_ref, switch, final_t1) for one scenario and rule."""
    d = np.load(SOURCE[scenario], allow_pickle=True)
    out = []
    for i, m in enumerate(d["methods"]):
        if m != rule:
            continue
        out.append((np.asarray(d[f"align_{i}"], float),
                    np.asarray(d[f"align_ref_{i}"], float),
                    int(d[f"switch0_{i}"]), float(d[f"final_t1_{i}"])))
    return out


LEAD = 150          # updates of task-1 context drawn before the switch


def common_window(traces, switches):
    """Grid of updates relative to the switch that EVERY run actually reached.

    Taking the intersection rather than the union is the whole point: a union grid would let the
    tail of the mean be computed from whichever seeds happened to run longest, which is a
    different set of runs at every x and not a curve any network traced.

    Clipped to `-LEAD` at the low end. Task 1 runs far longer than the common part of task 2
    (up to 1400 updates against 115), so an unclipped window spends 90% of the panel on the
    phase the figure is not about.
    """
    lo = min(int((t[:, IDX] - s).min()) for t, s in zip(traces, switches))
    hi = min(int((t[:, IDX] - s).max()) for t, s in zip(traces, switches))
    return np.arange(max(lo, -LEAD), hi + 1, 5)


def band(ax, scenario, rule, which, ls):
    rs = runs(scenario, rule)
    traces = [r[which] for r in rs]
    switches = [r[2] for r in rs]
    grid = common_window(traces, switches)
    stack = np.vstack([np.interp(grid, t[:, IDX] - s, t[:, VAL])
                       for t, s in zip(traces, switches)])
    mu = stack.mean(axis=0)
    se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
    c = style.SCENARIO[scenario]
    ax.plot(grid, mu, ls=ls, lw=1.2, color=c,
            label=f"{LABEL[scenario]} · {LABEL[rule]}")
    ax.fill_between(grid, mu - se, mu + se, color=c, alpha=0.15, lw=0)
    return grid


def panel(which, suffix, ylabel, zero_line):
    fig, ax = plt.subplots(figsize=style.size("col", 0.66))
    if zero_line:
        ax.axhline(0, color=style.ZERO_LINE, lw=0.7, zorder=1)
    lo = []
    for scenario in SCENARIOS:
        for rule, ls in (("backprop", "-"), ("pc", "--")):
            lo.append(band(ax, scenario, rule, which, ls))
    ax.axvline(0, color=style.ZERO_LINE, lw=0.8)
    ax.annotate("task switch", (0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -9), textcoords="offset points", fontsize=6.5, ha="left")
    ax.set_xlim(min(g[0] for g in lo), min(g[-1] for g in lo))
    ax.set_xlabel("updates relative to the task switch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6.0, handlelength=1.6, borderpad=0.15, labelspacing=0.2)
    out = figure_path(__file__, suffix)
    fig.savefig(out)
    print(f"saved {out}")


def task2_means(scenario, rule, which):
    """Per-seed mean alignment over that seed's OWN full task 2 -- no window truncation."""
    return np.array([r[which][r[which][:, IDX] > r[2], VAL].mean()
                     for r in runs(scenario, rule)])


def panel_c():
    """The claim, over every update recorded rather than the common window."""
    fig, ax = plt.subplots(figsize=style.size("col", 0.55))
    ax.axvline(0, color=style.ZERO_LINE, lw=0.7, zorder=1)
    ypos, labels, y = [], [], 0.0
    for which, name in ((0, "trained batch"), (1, "task-1 batch")):
        for scenario in SCENARIOS:
            m, se, _ = paired_diff(task2_means(scenario, "pc", which),
                                   task2_means(scenario, "backprop", which))
            ax.errorbar(m, y, xerr=se, marker="o", ms=3.4, lw=0, elinewidth=1.3,
                        color=style.SCENARIO[scenario], capsize=1.8)
            ypos.append(y)
            labels.append(f"{LABEL[scenario]} · {name}")
            y -= 1.0
        y -= 0.5
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_ylim(min(ypos) - 0.8, max(ypos) + 0.8)
    ax.set_xlabel("$\\Delta$ alignment over task 2, PC $-$ backprop")
    out = figure_path(__file__, "c")
    fig.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    panel(0, "a", "target alignment, trained batch", zero_line=False)
    panel(1, "b", "target alignment, task-1 batch", zero_line=True)
    panel_c()

    # The numbers the caption is written from: each seed's own full task 2, paired.
    print("\n  mean alignment over task 2 (each seed's own full task 2):")
    for scenario in SCENARIOS:
        for which, name in ((0, "trained batch"), (1, "task-1 batch")):
            bp, pc = (task2_means(scenario, r, which) for r in RULES)
            m, se, nsem = paired_diff(pc, bp)
            print(f"    {LABEL[scenario]:10s} {name:14s} "
                  f"backprop {bp.mean():+.4f}   pc {pc.mean():+.4f}   "
                  f"pc-bp {m:+.4f}+-{se:.4f} ({nsem:.1f} sem)")
