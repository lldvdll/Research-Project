"""Under repeated alternation, does the network return to where it started, or does it converge
on a solution that holds both tasks?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing.

MODE B -- one file, four panels, ONE LaTeX caption. Shared axes are the whole point: "spiral"
and "closed loop" are claims about the SHAPE of a trajectory, and two panels on independently
scaled axes cannot be compared for shape at all.

⚠ THE PRE-REGISTERED PREDICTION IS REFUTED, AND THIS FIGURE IS DRAWN TO SHOW THAT.
The plan predicted Class-IL would trace a CLOSED LOOP under repeated alternation -- returning to
the same place after every pair of blocks, learning nothing cumulative -- while Domain-IL would
SPIRAL IN toward a joint solution. NEITHER scenario traces a loop. Task-1 accuracy measured at
the end of each task-2 block, over ten seeds:

    Class-IL   backprop   4.8  21.9  30.8  32.4  35.9      rises steeply, plateaus near 35
    Class-IL   pc         6.0  23.0  28.4  32.8  34.1      indistinguishable from backprop
    Domain-IL  backprop  37.6  49.5  54.9  58.8  63.1      still climbing at the fifth block
    Domain-IL  pc        37.0  42.9  44.5  47.1  46.4      plateaus near 47

Both scenarios converge; the difference is where they stop. Class-IL recovers from ~5% after a
single switch to a ~35% ceiling, Domain-IL from ~38% to 63% and rising. Reporting the prediction
and its refutation together is the point -- quietly restating the prediction as if it had been
confirmed is the failure this docstring exists to prevent.

⚠ AND A RESULT THAT IS NOT IN R2 AT ALL: repeated alternation MULTIPLIES the rule difference in
Domain-IL. At a single switch PC - backprop is -0.72 +- 0.19 crossover points (912). At the fifth
task-2 block it is

    Domain-IL  PC - backprop  -16.71 +- 2.45  (6.8 sem)
    Class-IL   PC - backprop   -1.78 +- 3.09  (0.6 sem, i.e. nothing)

PC does not merely fail to help in Domain-IL under repetition -- it stops improving while
backprop keeps going. A single switch understates this by more than twenty times, which is an
argument about the PROTOCOL as much as about the rule: a two-task benchmark cannot see it.

    (a), (b)  the trajectory itself, one seed, coloured by block index. A closed loop retraces
              the same path in the same place; a converging system draws a path that contracts
              toward a point. Read the SHAPE here.
    (c), (d)  the quantity behind the shape, over all ten seeds: task-1 accuracy at the END of
              each task-2 block -- how much of task 1 survives the k-th relearning of task 2. A
              closed loop is FLAT in k. A system finding a joint solution RISES in k. Read the
              CLAIM here, because (a) and (b) are one seed each.

WHY END-OF-BLOCK AND NOT THE WHOLE CURVE. Within a block the accuracy of the task not being
trained falls and then partially recovers, so a mean over the block mixes the transient with the
level. The end of a task-2 block is the one moment per cycle at which the comparison "how much
task 1 is left once task 2 has been fully retrained" is well posed -- and it is the same moment
the single-switch result in R1 is read at.

PROVENANCE
    804  both scenarios, backprop and pc, seeds 10-19, config_800.yaml, MATCHED COMPETENCE PER
         BLOCK (68 ran a fixed budget) over 5 repeats = 10 blocks. Block length is therefore a
         DEPENDENT variable, not a setting.
    ⚠ Class-IL PC seed 13 hit the 5000-update cap on two blocks and reached only 8 of 10; it
      ends at task-1 0.2 where its neighbours end near 45. It is drawn, not dropped, and the
      count of capped runs is annotated -- a run that ran out of budget is not a run that
      converged somewhere different.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import sem, paired_diff
from src import style

EXP = ROOT / "experiments"
SOURCE = {s: EXP / f"804_repeated_alternation_{s}.npz" for s in ("class_il", "domain_il")}
SCENARIOS = ["class_il", "domain_il"]
RULES = ["backprop", "pc"]
LABEL = {"class_il": "Class-IL", "domain_il": "Domain-IL",
         "backprop": "backprop", "pc": "PC"}
SHOW_SEED = 10          # the seed drawn in (a) and (b); it has a weight trace too, for 918

style.apply()


def runs(scenario, rule):
    d = np.load(SOURCE[scenario], allow_pickle=True)
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
    """Task-1 accuracy at the end of each block, indexed by block number.

    `switches` holds the step each block ENDED at, so the last eval at or before switch k is the
    state the block finished in. Task-2 blocks are the odd indices.
    """
    return np.array([r["t1"][max(0, int(np.searchsorted(r["steps"], s, side="right")) - 1)]
                     for s in r["switches"]])


def trajectory_panel(ax, scenario, rule, letter):
    r = next(x for x in runs(scenario, rule) if x["seed"] == SHOW_SEED)
    bounds = [0] + [int(np.searchsorted(r["steps"], s, side="right")) for s in r["switches"]]
    cols = style.sweep(len(bounds) - 1)
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], min(bounds[b + 1] + 1, len(r["steps"]))
        ax.plot(r["t1"][lo:hi], r["t2"][lo:hi], color=cols[b], lw=0.9)
    ax.plot([0, 100], [0, 100], ls=":", lw=0.8, color=style.NEUTRAL, zorder=1)
    ax.scatter([r["t1"][-1]], [r["t2"][-1]], s=14, marker="s",
               color=cols[-1], zorder=5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_title(f"{LABEL[scenario]} · {LABEL[rule]}, seed {SHOW_SEED}")
    style.panel_label(ax, letter, dx=-0.16)
    # Without a key the colour gradient is decoration. It is the block index, and the direction
    # of travel is the entire claim -- a closed loop would retrace, a converging system does not.
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(1, len(bounds) - 1))
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.045, pad=0.03, aspect=18)
    cb.set_label("block", fontsize=6.5)
    cb.set_ticks([1, len(bounds) - 1])
    cb.ax.tick_params(labelsize=6)


def block_panel(ax, scenario, letter):
    for rule, ls in (("backprop", "-"), ("pc", "--")):
        rs = runs(scenario, rule)
        stack = np.vstack([end_of_block(r) for r in rs])
        # Odd block indices are the ends of task-2 blocks -- the moment the comparison is posed.
        k = np.arange(1, stack.shape[1], 2)
        mu = np.array([sem(stack[:, j])[0] for j in k])
        se = np.array([sem(stack[:, j])[1] for j in k])
        x = np.arange(1, len(k) + 1)
        ax.errorbar(x, mu, yerr=se, ls=ls, lw=1.2, marker="o", ms=2.8,
                    color=style.SCENARIO[scenario],
                    alpha=1.0 if rule == "backprop" else 0.6,
                    label=f"{LABEL[rule]}")
        capped = sum(int((~r["reached"]).any()) for r in rs)
        if capped:
            ax.annotate(f"{capped}/{len(rs)} {LABEL[rule]} runs hit the cap",
                        (0.5, 0.04 if rule == "pc" else 0.12), xycoords="axes fraction",
                        fontsize=5.8, ha="center", color=style.CAPPED)
    ax.set_xticks(np.arange(1, 6))
    ax.set_xlabel("task-2 block number")
    style.panel_label(ax, letter, dx=-0.16)


if __name__ == "__main__":
    w = style.WIDTH["page"] / 2.0
    fig, axes = plt.subplots(2, 2, figsize=(2 * w, 2 * w * 0.80))
    for c, scenario in enumerate(SCENARIOS):
        trajectory_panel(axes[0][c], scenario, "backprop", "ab"[c])
        block_panel(axes[1][c], scenario, "cd"[c])
    axes[0][0].set_ylabel("task-2 accuracy (%)")
    for c in range(2):
        axes[0][c].set_xlabel("task-1 accuracy (%)")
    axes[1][0].set_ylabel("task-1 accuracy at\nend of block (%)")
    # (c) and (d) must share a scale or "Class-IL plateaus lower" cannot be read off the page.
    # sharey= at construction would also tie them to the accuracy axes of (a) and (b), which are
    # a different quantity, so the bottom row is matched explicitly here instead.
    lo = min(ax.get_ylim()[0] for ax in axes[1])
    hi = max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(lo, hi)
    print(f"  bottom row y-limits: {[tuple(round(v, 1) for v in ax.get_ylim()) for ax in axes[1]]}")
    axes[1][0].legend(fontsize=6.5, handlelength=1.6, borderpad=0.15, loc="upper left")

    out = figure_path(__file__)
    fig.savefig(out)
    print(f"saved {out}")
    for scenario in SCENARIOS:
        per = {}
        for rule in RULES:
            rs = runs(scenario, rule)
            stack = np.vstack([end_of_block(r) for r in rs])
            k = np.arange(1, stack.shape[1], 2)
            per[rule] = stack[:, k]
            vals = "  ".join(f"{np.mean(stack[:, j]):5.1f}" for j in k)
            print(f"  {LABEL[scenario]:10s} {rule:9s} task-1 at end of task-2 blocks: {vals}")
        # The comparison R4 reports: does repeated alternation widen the rule gap? At a single
        # switch it is +1.4 / -0.7 crossover points (912).
        m, se_, nsem = paired_diff(per["pc"][:, -1], per["backprop"][:, -1])
        print(f"  {LABEL[scenario]:10s} PC - backprop at the FIFTH task-2 block: "
              f"{m:+.2f}+-{se_:.2f} ({nsem:.1f} sem)")
