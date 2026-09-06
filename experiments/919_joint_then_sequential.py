"""If the network is handed a solution that already holds both tasks, does sequential training
still destroy it?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing.

MODE A -- two PNGs, composed with `subfigure`.

WHY THIS IS WORTH A FIGURE. Every other result here measures a network that has never seen a
joint solution, so "forgetting" could always be read as a failure to FIND one -- an optimisation
problem rather than an interference problem. This arm removes that reading: train on the joint
distribution to convergence FIRST, then run the ordinary sequential protocol from those weights.
The joint solution is not hypothetical, it is where the run starts.

    Class-IL   scratch  4.8  ->  joint-start 19.3      still collapsing from ~80
    Domain-IL  scratch 37.6  ->  joint-start 44.3

Joint pre-training protects PARTIALLY and nowhere near completely. The solution can be handed to
the network and it still leaves. So forgetting here is not mainly a failure to find a joint
solution; it is a failure to STAY at one.

⚠ THE ARMS DO NOT TRAIN TASK 1 FOR THE SAME NUMBER OF UPDATES, AND PANEL (b) EXISTS BECAUSE OF
IT. Both arms stop task 1 at the same competence threshold -- but the joint-initialised arm is
ALREADY above threshold when its task-1 phase begins, so that phase ends almost immediately
(Class-IL seed 10: 240 updates from scratch against 30 from joint). A shorter task-1 phase means
less to lose, so part of any apparent protection is bookkeeping rather than mechanism. Panel (b)
draws the two phase lengths so the size of the confound is visible next to the size of the
effect, and the console prints the correlation between the phase-length difference and the
retention difference. This is a limitation the figure states, not one it corrects: the two cannot
be separated within this design, because equalising the phase lengths would mean stopping the two
arms at different competences, which breaks the protocol every other result uses.

⚠ A DEFECT IN 805'S SAVED ARRAYS. The joint arm's per-eval accuracy CURVES were lost to a key
collision -- the scalar `joint_t1` (task-1 accuracy after the joint phase, before any sequential
training) and the curve `joint_t1` were written under the same name, so only the scalar survives.
Its endpoint scalars (final_t1, final_t2, crossover, switch0) are intact and are what this figure
uses, but no trajectory can be drawn for that arm. Fixing it means renaming one of the two in 805
and re-running, about 22 minutes.

PROVENANCE
    805  both scenarios, backprop and pc, seeds 10-19, config_800.yaml, matched competence.
         Both arms share one build per (rule, seed), so the joint arm's sequential phase starts
         from exactly the weights its joint phase ended at, and the scratch arm is rebuilt from
         the same seed -- the two are paired by construction, not merely by label.
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
SOURCE = {s: EXP / f"805_joint_then_sequential_{s}.npz" for s in ("class_il", "domain_il")}
SCENARIOS = ["class_il", "domain_il"]
RULES = ["backprop", "pc"]
ARMS = ["scratch", "joint"]
LABEL = {"class_il": "Class-IL", "domain_il": "Domain-IL",
         "backprop": "backprop", "pc": "PC",
         "scratch": "from scratch", "joint": "joint pre-trained"}

style.apply()


def field(scenario, rule, arm, name):
    """One saved scalar per seed, ordered by seed so the arms pair row for row."""
    d = np.load(SOURCE[scenario], allow_pickle=True)
    vals = [(int(d["seeds"][i]), float(d[f"{arm}_{name}_{i}"]))
            for i in range(len(d["methods"])) if str(d["methods"][i]) == rule]
    return np.array([v for _, v in sorted(vals)])


def grouped_bars(ax, value_fn, ylabel):
    """Four groups (scenario x rule), two bars each (arm)."""
    groups = [(s, r) for s in SCENARIOS for r in RULES]
    x = np.arange(len(groups))
    width = 0.36
    for k, arm in enumerate(ARMS):
        mu, se = [], []
        for scenario, rule in groups:
            m, s = sem(value_fn(scenario, rule, arm))
            mu.append(m)
            se.append(s)
        ax.bar(x + (k - 0.5) * width, mu, width,
               color=style.NEUTRAL if arm == "scratch" else style.SCENARIO["class_il"],
               alpha=0.85, label=LABEL[arm], yerr=se, error_kw=dict(lw=0.9))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LABEL[s]}\n{LABEL[r]}" for s, r in groups], fontsize=6.2)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.16)
    ax.legend(fontsize=6.5, handlelength=1.2, borderpad=0.15)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=style.size("col", 0.70))
    grouped_bars(ax, lambda s, r, a: field(s, r, a, "final_t1"),
                 "final task-1 accuracy (%)")
    out = figure_path(__file__, "a")
    fig.savefig(out)
    print(f"saved {out}")

    fig, ax = plt.subplots(figsize=style.size("col", 0.70))
    grouped_bars(ax, lambda s, r, a: field(s, r, a, "switch0"),
                 "task-1 phase length (updates)")
    ax.set_title("the confound, not a result", fontsize=7)
    out = figure_path(__file__, "b")
    fig.savefig(out)
    print(f"saved {out}")

    print("\n  final task-1 accuracy, joint minus scratch (paired on seed):")
    for scenario in SCENARIOS:
        for rule in RULES:
            sc = field(scenario, rule, "scratch", "final_t1")
            jt = field(scenario, rule, "joint", "final_t1")
            m, se_, nsem = paired_diff(jt, sc)
            # The confound: does the protection track how much SHORTER the joint arm's task-1
            # phase was? A strong correlation would mean the effect is largely bookkeeping.
            dphase = (field(scenario, rule, "joint", "switch0")
                      - field(scenario, rule, "scratch", "switch0"))
            r = np.corrcoef(dphase, jt - sc)[0, 1]
            print(f"    {LABEL[scenario]:10s} {rule:9s} scratch {sc.mean():5.1f} -> joint "
                  f"{jt.mean():5.1f}   {m:+6.2f}+-{se_:5.2f} ({nsem:4.1f} sem)   "
                  f"phase {field(scenario, rule, 'scratch', 'switch0').mean():5.0f} -> "
                  f"{field(scenario, rule, 'joint', 'switch0').mean():4.0f} updates   "
                  f"r(dphase, dretention) {r:+.2f}")
