"""Is forgetting about where you start in weight space?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R4.

ONE PLOT, one grid cell, form from 007's SK_JOINT: train on the JOINT distribution first, then
run the ordinary sequential protocol from those weights, against sequential-from-scratch.

WHY IT SETTLES SOMETHING NOTHING ELSE DOES. Every other result measures a network that has never
seen a joint solution, so "forgetting" can always be read as a failure to FIND one -- an
optimisation problem rather than an interference problem. Here the joint solution is not
hypothetical; it is where the run starts.

    final task-1 accuracy, joint-pretrained vs from scratch, paired on seed
        Class-IL   backprop   4.8 → 16.2    +11.35 ± 3.67  (3.1 sem)
        Class-IL   pc         6.0 → 15.5     +9.52 ± 3.73  (2.5 sem)
        Domain-IL  backprop  37.6 → 44.3     +6.70 ± 2.02  (3.3 sem)
        Domain-IL  pc        37.0 → 41.0     +4.02 ± 0.78  (5.1 sem)

Protection is real and PARTIAL. A network that already solves both tasks still collapses on task 1
-- so forgetting here is not mainly a failure to find a joint solution, it is a failure to STAY at
one. That is the sentence this figure exists to license.

⚠ THE ARMS DO NOT TRAIN TASK 1 FOR THE SAME NUMBER OF UPDATES, AND THE RIGHT PANEL DRAWS IT.
Both stop task 1 at the same competence threshold, but the joint-initialised arm is ALREADY above
threshold when its task-1 phase begins, so that phase ends almost at once (402 → 182 updates,
Class-IL backprop). A shorter phase means less to lose, so part of any apparent protection is
bookkeeping. The correlation between the phase-length difference and the retention difference says
how much:

    Class-IL   backprop r = −0.64   pc r = −0.62      confound present
    Domain-IL  backprop r = −0.09   pc r = +0.51      backprop arm is clean

LEAN ON THE DOMAIN-IL BACKPROP ARM. The two effects cannot be separated within this design --
equalising the phase lengths would mean stopping the arms at different competences, which breaks
the protocol every other result uses -- so the figure states the limitation rather than correcting
it.

⚠ A DEFECT IN 805'S SAVED ARRAYS. The joint arm's per-eval accuracy CURVES were lost to a key
collision: the scalar `joint_t1` (task-1 accuracy after the joint phase) and the curve `joint_t1`
were written under one name, so only the scalar survives. Endpoint scalars are intact and are what
this figure uses; no trajectory can be drawn for that arm. Renaming and re-running costs ~22 min.

Styling follows the 300-series scripts: tab: colours, dpi 120, bbox_inches="tight", 9pt labels.
No shared style module.

PROVENANCE
    805_joint_then_sequential_{scenario}.npz   both scenarios, backprop and pc, seeds 10-19,
    config_800.yaml, matched competence. Both arms share one build per (rule, seed), so the joint
    arm's sequential phase starts from exactly the weights its joint phase ended at and the
    scratch arm is rebuilt from the same seed -- paired by construction, not by label.
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
SCENARIOS = ["class_il", "domain_il"]
RULES = ["backprop", "pc"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
ARM_COLOR = {"scratch": "0.45", "joint": "tab:red"}


def field(scenario, rule, arm, name):
    d = np.load(EXP / f"805_joint_then_sequential_{scenario}.npz", allow_pickle=True)
    vals = [(int(d["seeds"][i]), float(d[f"{arm}_{name}_{i}"]))
            for i in range(len(d["methods"])) if str(d["methods"][i]) == rule]
    return np.array([v for _, v in sorted(vals)])


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    groups = [(s, r) for s in SCENARIOS for r in RULES]
    x = np.arange(len(groups))
    width = 0.36

    # left: the result -- paired, per seed, both arms
    for k, arm in enumerate(("scratch", "joint")):
        mu = [np.mean(field(s, r, arm, "final_t1")) for s, r in groups]
        se = [np.std(field(s, r, arm, "final_t1"), ddof=1) / np.sqrt(10) for s, r in groups]
        ax1.bar(x + (k - 0.5) * width, mu, width, yerr=se, capsize=3,
                color=ARM_COLOR[arm], alpha=0.85,
                label="from scratch" if arm == "scratch" else "joint pre-trained")
    for i, (s, r) in enumerate(groups):
        d, se_, nsem = paired_diff(field(s, r, "joint", "final_t1"),
                                   field(s, r, "scratch", "final_t1"))
        ax1.annotate(f"{d:+.1f}\n({nsem:.1f} sem)",
                     (x[i], max(np.mean(field(s, r, a, "final_t1")) for a in ("scratch", "joint"))),
                     xytext=(0, 14), textcoords="offset points", ha="center", fontsize=7,
                     linespacing=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{NICE[s]}\n{r}" for s, r in groups], fontsize=8)
    ax1.set_ylabel("final task-1 accuracy (%)", fontsize=9)
    ax1.set_ylim(0, max(np.mean(field(s, r, "joint", "final_t1")) for s, r in groups) * 1.6)
    ax1.grid(alpha=0.25, axis="y")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("Handing the network a joint solution protects it — partially", fontsize=9)

    # right: the confound, drawn rather than hidden
    for i, (s, r) in enumerate(groups):
        dphase = field(s, r, "joint", "switch0") - field(s, r, "scratch", "switch0")
        dret = field(s, r, "joint", "final_t1") - field(s, r, "scratch", "final_t1")
        rr = np.corrcoef(dphase, dret)[0, 1]
        ax2.scatter(dphase, dret, s=32, color=COLORS[s],
                    marker="s" if r == "backprop" else "o", alpha=0.85,
                    label=f"{NICE[s]} · {r}  (r = {rr:+.2f})")
        print(f"  {NICE[s]:10s} {r:9s} scratch {np.mean(field(s, r, 'scratch', 'final_t1')):5.1f} "
              f"-> joint {np.mean(field(s, r, 'joint', 'final_t1')):5.1f}   "
              f"phase {np.mean(field(s, r, 'scratch', 'switch0')):5.0f} -> "
              f"{np.mean(field(s, r, 'joint', 'switch0')):4.0f}   r(dphase,dret) {rr:+.2f}")
    ax2.axhline(0, color="k", lw=1.0)
    ax2.set_xlabel("task-1 phase length, joint $-$ scratch (updates)", fontsize=9)
    ax2.set_ylabel("retention gain, joint $-$ scratch (pp)", fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.set_title("⚠ the confound: the joint arm trains task 1 for far fewer updates,\n"
                  "so it has less to lose — clean only for Domain-IL backprop", fontsize=9)

    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
