"""Is the code still readable when the readout fails?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R5.

TWO PLOTS, ONE PER GRID CELL, matching 007's two cards:
    923_probe_vs_argmax_a.png   TASK 1 -- the question
    923_probe_vs_argmax_b.png   TASK 2 -- the control that keeps the probe honest

Form is 007's SK_PROBE: a refit linear probe against argmax through training, switch marked, with
the two readings named on the plot.

    argmax LOW, probe HIGH   the information is in the hidden code and the OUTPUT misreads it
    argmax LOW, probe LOW    the code itself is gone

⚠ THE RANDOM-INIT FLOOR IS NOT OPTIONAL, AND IT IS WHY NCM WAS REPLACED. A linear probe on 32 tanh
units is a strong classifier even with RANDOM weights. Measured here on the untrained network of
the same seed, the probe already reads 80.2% on the task-1 classes. So of the 82.6% it reports
mid-run, roughly 80 points are available before any training and only ~2.4 are attributable to
what the trunk learned. Drawn without that line this figure says "the representation survives"
when what it mostly shows is that the probe barely needed one. Every panel draws its own floor,
and the claim is made on the DISTANCE ABOVE IT.

    at each seed's last checkpoint, backprop
        Class-IL  task 1   probe 82.6  argmax 21.1  gap +61.4  floor 80.2  above floor +2.4
        Class-IL  task 2   probe 83.1  argmax 72.3  gap +10.9  floor 79.7  above floor +3.4
        Domain-IL task 1   probe 81.8  argmax 47.8  gap +34.0  floor 78.0  above floor +3.8
        Domain-IL task 2   probe 81.5  argmax 83.8  gap  −2.3  floor 77.1  above floor +4.3

THE TASK-2 PANEL IS THE CONTROL. There the gap REVERSES -- argmax beats the probe -- which rules
out "the probe is simply a better classifier". 007's card keeps exactly this control from the NCM
version (70.8 against 91.3) while replacing the probe itself, and it survives the replacement.

⚠ THE GAP IS LARGE IN BOTH SCENARIOS (+61.4 and +34.0) even though output suppression is
structurally impossible in Domain-IL. 920 draws that as one of the measurements that does NOT
separate. It is the honest limit on the readout account.

⚠ A DEFECT IN 803'S CHECKPOINT SPACING. Its interval is (2 * max_iters_per_task) // CHECKPOINTS =
10000 // 20 = 500 updates, which divides the BUDGET, not the run. Matched competence finishes runs
in 700-2500 updates, so each gets 2-5 checkpoints instead of ~20 and the LAST sits a median ~175
updates before the end. argmax on task 1 reads 21.1 there against a true final of 4.8, so this
figure UNDERSTATES the end-of-training gap. Every number is labelled "last checkpoint" for that
reason. Fixing it means re-running 803 with the interval derived from run length -- about 8 min.

Styling follows the 300-series scripts: tab: colours, dpi 120, bbox_inches="tight", 9pt labels.
No shared style module.

PROVENANCE
    803_mechanism_logged_{scenario}.npz   both scenarios, backprop and pc, seeds 10-19,
    config_800.yaml, matched competence. The probe is a ridge least-squares readout (ridge 1e-3)
    refitted at each checkpoint on `stop_eval` and scored on `report_eval` -- FIT AND SCORE SPLITS
    ARE DISJOINT -- predicting output UNITS exactly as `predict` does, so the two are scored alike.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
STEP, P1, P2, A1, A2 = 0, 1, 2, 3, 4
RULE = "backprop"          # 912 establishes the rules barely differ; one is drawn for legibility


def runs(scenario):
    d = np.load(EXP / f"803_mechanism_logged_{scenario}.npz", allow_pickle=True)
    return [(np.asarray(d[f"probe_{i}"], float), np.asarray(d[f"probe_floor_{i}"], float),
             int(d[f"switch0_{i}"]))
            for i, m in enumerate(d["methods"]) if m == RULE]


def draw(task, tag):
    pcol, acol = (P1, A1) if task == 0 else (P2, A2)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, s in zip(axes, SCENARIOS):
        rs = runs(s)
        rel = [p[:, STEP] - sw for p, _, sw in rs]
        grid = np.linspace(max(r.min() for r in rel), min(r.max() for r in rel), 60)
        probe = np.vstack([np.interp(grid, r, p[:, pcol]) for r, (p, _, _) in zip(rel, rs)])
        argmax = np.vstack([np.interp(grid, r, p[:, acol]) for r, (p, _, _) in zip(rel, rs)])
        floor = np.array([f[task] for _, f, _ in rs])

        for stack, c, lab in ((probe, COLORS[s], "refit linear probe"),
                              (argmax, "0.35", "argmax (the network's own readout)")):
            mu = stack.mean(axis=0)
            se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
            ax.plot(grid, mu, lw=2.0, color=c, label=lab)
            ax.fill_between(grid, mu - se, mu + se, color=c, alpha=0.2)
        ax.axhline(floor.mean(), ls="--", lw=1.4, color="tab:red")
        ax.annotate(f"probe floor, UNTRAINED net ({floor.mean():.0f}%)",
                    (grid[-1], floor.mean()), xytext=(-4, 5), textcoords="offset points",
                    ha="right", fontsize=7.5, color="tab:red")
        ax.axvline(0, color="0.25", lw=1.2)
        ax.annotate("switch", (0, 2), xytext=(4, 0), textcoords="offset points", fontsize=8)

        fp = np.array([p[-1, pcol] for p, _, _ in rs]).mean()
        fa = np.array([p[-1, acol] for p, _, _ in rs]).mean()
        ax.annotate(f"last checkpoint: probe {fp:.0f}, argmax {fa:.0f}\n"
                    f"gap {fp - fa:+.1f} · probe above floor {fp - floor.mean():+.1f}",
                    (0.98, 0.04), xycoords="axes fraction", ha="right", fontsize=7.5,
                    linespacing=1.3)
        ax.set_xlabel("updates relative to the task switch", fontsize=9)
        ax.set_ylim(-2, 102)
        ax.set_title(f"{NICE[s]} · task {task + 1}", fontsize=9)
        ax.grid(alpha=0.25)
        print(f"  {NICE[s]:10s} task {task + 1}   probe {fp:5.1f}  argmax {fa:5.1f}  "
              f"gap {fp - fa:+6.1f}  floor {floor.mean():5.1f}  above floor {fp - floor.mean():+5.1f}")
    axes[0].set_ylabel("accuracy (%)", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower left")
    title = ("The code stays linearly readable while argmax collapses — but barely above what an "
             "UNTRAINED trunk already gives" if task == 0 else
             "The control: on the task just trained the gap REVERSES, so the probe is not simply "
             "a better classifier")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__, tag)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    draw(0, "a")
    draw(1, "b")
