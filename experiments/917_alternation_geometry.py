"""Under repeated alternation, does the network converge on a joint solution?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R4.

ONE PLOT, one grid cell, two panels side by side as 007's SK_REPEAT asks: task-1 accuracy against
task-2 accuracy, colour = time, Class-IL beside Domain-IL on identical axes. A loop and a spiral
are different KINDS of behaviour, not different amounts of one, so the two scenarios have to be
readable against each other or the comparison says nothing.

⚠ THE PREDICTION IS REFUTED AND THE FIGURE IS DRAWN TO SHOW IT. 007 predicted Class-IL would
close a LOOP -- ping-ponging between incompatible solutions, learning nothing cumulative -- while
Domain-IL spiralled inward. NEITHER closes a loop. Both spiral in; they differ in where they stop.
Task-1 accuracy measured at the end of each task-2 block, ten seeds:

    Class-IL   backprop   4.8  21.9  30.8  32.4  35.9      rises, plateaus near 35
    Class-IL   pc         6.0  23.0  28.4  32.8  34.1      indistinguishable from backprop
    Domain-IL  backprop  37.6  49.5  54.9  58.8  63.1      still climbing at the fifth block
    Domain-IL  pc        37.0  42.9  44.5  47.1  46.4      plateaus near 47

A closed loop would be FLAT in block number. Nothing here is flat. Class-IL recovers from ~5%
after a single switch to a ~35% ceiling; Domain-IL from ~38% to 63% and rising.

⚠ AND A SEPARATION THAT A TWO-TASK PROTOCOL CANNOT SEE. At a single switch PC - backprop is
-0.72 ± 0.19 (912). At the fifth task-2 block:

    Domain-IL  PC - backprop  -16.71 ± 2.45  (6.8 sem)
    Class-IL   PC - backprop   -1.78 ± 3.09  (0.6 sem, nothing)

PC does not merely fail to help in Domain-IL under repetition -- it stops improving while backprop
keeps going, and one switch understates that by more than twenty times. This is the strongest
empirical separation between the scenarios in the project, and it is an argument about the
BENCHMARK as much as about the rule.

MATCHED COMPETENCE PER BLOCK MAKES BLOCK LENGTH A DEPENDENT VARIABLE, which 007's card notes is
itself the "gradual relearn" result: blocks shorten from ~400 to ~50 updates in Class-IL and
settle near 150-200 in Domain-IL. The console prints them.

Styling follows the 300-series scripts: viridis for time, dpi 120, bbox_inches="tight", 9pt
labels. No shared style module.

PROVENANCE
    804_repeated_alternation_{scenario}.npz   both scenarios, backprop and pc, seeds 10-19,
    config_800.yaml, MATCHED COMPETENCE PER BLOCK over 5 repeats = 10 blocks. Supersedes legacy
    68, which was Domain-IL only, 5 seeds, fixed budget.
    ⚠ Class-IL PC seed 13 hit the 5000-update cap on two blocks and reached 8 of 10, ending at
    task-1 0.2 where its neighbours end near 45. Drawn, not dropped; the capped count is reported.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
RULES = ["backprop", "pc"]
SHOW_RULE, SHOW_SEED = "backprop", 10


def runs(scenario, rule):
    d = np.load(EXP / f"804_repeated_alternation_{scenario}.npz", allow_pickle=True)
    out = []
    for i, m in enumerate(d["methods"]):
        if m != rule:
            continue
        out.append(dict(seed=int(d["seeds"][i]), steps=np.asarray(d[f"steps_{i}"]),
                        t1=np.asarray(d[f"t1_{i}"], float), t2=np.asarray(d[f"t2_{i}"], float),
                        switches=np.asarray(d[f"switches_{i}"]),
                        reached=np.asarray(d[f"reached_{i}"], bool)))
    return out


def end_of_block(r):
    return np.array([r["t1"][max(0, int(np.searchsorted(r["steps"], s, side="right")) - 1)]
                     for s in r["switches"]])


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), sharex=True, sharey=True)
    norm = mpl.colors.Normalize(1, 10)
    cmap = mpl.colormaps["viridis"]

    for ax, s in zip(axes, SCENARIOS):
        r = next(x for x in runs(s, SHOW_RULE) if x["seed"] == SHOW_SEED)
        bounds = [0] + [int(np.searchsorted(r["steps"], w, side="right")) for w in r["switches"]]
        for b in range(len(bounds) - 1):
            lo, hi = bounds[b], min(bounds[b + 1] + 1, len(r["steps"]))
            ax.plot(r["t1"][lo:hi], r["t2"][lo:hi], color=cmap(norm(b + 1)), lw=1.2)
        ax.plot([0, 100], [0, 100], ls=":", lw=1.0, color="0.6")
        ax.scatter([r["t1"][-1]], [r["t2"][-1]], s=45, marker="*", color=cmap(norm(10)),
                   zorder=6, edgecolor="k", linewidth=0.5)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_xlabel("task-1 accuracy (%)", fontsize=9)
        ax.grid(alpha=0.2)

        eob = np.vstack([end_of_block(x) for x in runs(s, SHOW_RULE)])
        k = np.arange(1, eob.shape[1], 2)
        ax.set_title(f"{NICE[s]} — spirals inward, does not close\n"
                     f"task-1 at end of task-2 blocks: "
                     + " → ".join(f"{np.mean(eob[:, j]):.0f}" for j in k), fontsize=9)
    axes[0].set_ylabel("task-2 accuracy (%)", fontsize=9)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, label="alternation block", pad=0.015)
    fig.suptitle(f"Repeated alternation, {SHOW_RULE}, seed {SHOW_SEED} — the predicted Class-IL "
                 f"closed loop does not occur", fontsize=9)
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    for s in SCENARIOS:
        per = {}
        for rule in RULES:
            rs = runs(s, rule)
            eob = np.vstack([end_of_block(x) for x in rs])
            k = np.arange(1, eob.shape[1], 2)
            per[rule] = eob[:, k]
            blocks = np.mean([np.diff(np.concatenate([[0], x["switches"]])) for x in rs], axis=0)
            capped = sum(int((~x["reached"]).any()) for x in rs)
            print(f"  {NICE[s]:10s} {rule:9s} end-of-block task-1: "
                  + "  ".join(f"{np.mean(eob[:, j]):5.1f}" for j in k)
                  + f"   block lengths: " + " ".join(f"{b:.0f}" for b in blocks)
                  + (f"   [{capped}/{len(rs)} hit the cap]" if capped else ""))
        m, se, nsem = paired_diff(per["pc"][:, -1], per["backprop"][:, -1])
        print(f"  {NICE[s]:10s} PC - backprop at the FIFTH task-2 block: "
              f"{m:+.2f} ± {se:.2f} ({nsem:.1f} sem)")
