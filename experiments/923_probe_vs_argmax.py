"""Does the hidden code for task 1 survive when argmax says task 1 is gone?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing.

MODE B -- one file, four panels (task x scenario), ONE LaTeX caption. Shared axes because the
whole reading is a COMPARISON ACROSS PANELS: the argmax-minus-probe gap is large in Class-IL,
where output suppression is available, and also large in Domain-IL, where it is structurally
impossible. Four independently scaled panels could not support that.

WHAT A LINEAR PROBE IS FOR. `predict` reads the network's own output units through argmax. A
probe refits a fresh linear readout on the hidden layer, so it asks a different question:

    argmax LOW, probe HIGH   the information is still in the hidden code and the OUTPUT is
                             misreading it -- a recalibration failure
    argmax LOW, probe LOW    the code itself is gone -- a representation failure

⚠ THE FLOOR IS NOT OPTIONAL, AND THIS IS THE WHOLE REASON NCM WAS REPLACED AND THEN THE
REPLACEMENT NEARLY REPEATED ITS MISTAKE. A linear probe on 32 tanh units is a strong classifier
even with RANDOM weights. Measured here, Class-IL, task-1 classes: the probe reads 82.4% on the
UNTRAINED network and 86.2% after task 1. So of the ~84% it reports mid-run, about 82 points are
available before any training has happened and only ~4 points are attributable to what the trunk
learned. Drawn without that line, this figure says "the representation survives" when what it
mostly shows is that the probe barely needed one. Every panel therefore carries its own
random-init floor, and the claim is made on the DISTANCE ABOVE IT, never on the raw height.

NCM is excluded for the same reason -- it is a centroid-only probe with almost no dynamic range
here -- and the one control it does provide is kept: on task 2 its sign reverses (NCM 70.8
against argmax 91.3), which rules out "the probe is simply a better classifier".

⚠ A DEFECT IN 803'S CHECKPOINT SPACING, AND WHAT IT COSTS THIS FIGURE. 803 sets its checkpoint
interval to (2 * max_iters_per_task) // CHECKPOINTS = 10000 // 20 = 500 updates. That divides the
BUDGET, not the run -- and under matched competence the runs finish in 700-2500 updates, so each
one gets only 2 to 5 checkpoints instead of ~20, and the LAST checkpoint sits a median of ~175
updates before the run actually ends. The consequence is quantitative and must not be papered
over: argmax on task 1 reads 21.1 at the last checkpoint but 4.8 at the true end (Class-IL,
backprop). So this figure reports the gap SHORTLY BEFORE the end of training, not at it, and the
real end-of-training gap is LARGER than what is drawn. Every number here is labelled "last
checkpoint" for that reason. Fixing it means re-running 803 with the interval derived from the
run length rather than the budget; that costs about 8 minutes and would also sharpen 915, 916
and 924.

PROVENANCE
    803  both scenarios, backprop and pc, seeds 10-19, config_800.yaml, matched competence.
         The probe is a ridge least-squares readout (ridge = 1e-3) refitted at each checkpoint on
         `stop_eval` and scored on `report_eval` -- FIT AND SCORE SPLITS ARE DISJOINT. It
         predicts output UNITS, exactly as `predict` does, so the two are scored identically.
         2-5 checkpoints per run -- see the defect note above, NOT the ~20 intended.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import sem
from src import style

EXP = ROOT / "experiments"
SOURCE = {s: EXP / f"803_mechanism_logged_{s}.npz" for s in ("class_il", "domain_il")}
SCENARIOS = ["class_il", "domain_il"]
LABEL = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
# probe rows are (step, probe_t1, probe_t2, argmax_t1, argmax_t2)
STEP, P1, P2, A1, A2 = 0, 1, 2, 3, 4
RULE_SHOWN = "backprop"     # 912 already establishes the rules barely differ; see below

style.apply()


def runs(scenario, rule):
    d = np.load(SOURCE[scenario], allow_pickle=True)
    out = []
    for i, m in enumerate(d["methods"]):
        if m != rule:
            continue
        out.append((np.asarray(d[f"probe_{i}"], float),
                    np.asarray(d[f"probe_floor_{i}"], float),
                    int(d[f"switch0_{i}"])))
    return out


def series(scenario, rule, task):
    """Probe and argmax against updates relative to the switch, on a common grid.

    Checkpoints fall at different absolute steps in every run (matched competence sets the
    budget), so each run is interpolated onto the grid every run reached -- the same
    intersection-not-union rule 915 uses, for the same reason.
    """
    rs = runs(scenario, rule)
    rel = [p[:, STEP] - sw for p, _, sw in rs]
    grid = np.linspace(max(r.min() for r in rel), min(r.max() for r in rel), 60)
    pcol, acol = (P1, A1) if task == 0 else (P2, A2)
    probe = np.vstack([np.interp(grid, r, p[:, pcol]) for r, (p, _, _) in zip(rel, rs)])
    argmax = np.vstack([np.interp(grid, r, p[:, acol]) for r, (p, _, _) in zip(rel, rs)])
    floor = np.array([f[task] for _, f, _ in rs])
    return grid, probe, argmax, floor


def final_checkpoint(scenario, rule, task):
    """Probe, argmax and floor at each seed's OWN last checkpoint.

    The common grid in `series` ends where the SHORTEST run ended, which is early in most runs --
    argmax there is still mid-collapse. The claim R5 makes is about the end of training, so it is
    computed per seed and reported separately rather than read off the end of the shared window.
    """
    pcol, acol = (P1, A1) if task == 0 else (P2, A2)
    rs = runs(scenario, rule)
    return (np.array([p[-1, pcol] for p, _, _ in rs]),
            np.array([p[-1, acol] for p, _, _ in rs]),
            np.array([f[task] for _, f, _ in rs]))


def panel(ax, scenario, task, letter):
    grid, probe, argmax, floor = series(scenario, RULE_SHOWN, task)
    for stack, c, name in ((probe, style.SCENARIO[scenario], "linear probe"),
                           (argmax, style.RULE["backprop"], "argmax")):
        mu = stack.mean(axis=0)
        se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
        ax.plot(grid, mu, lw=1.3, color=c, label=name)
        ax.fill_between(grid, mu - se, mu + se, color=c, alpha=0.18, lw=0)
    f, fse = sem(floor)
    ax.axhline(f, color=style.CAPPED, lw=0.9, ls=(0, (4, 3)))
    ax.annotate(f"probe floor, untrained net ({f:.0f}%)", (0.98, f), xytext=(0, 3),
                xycoords=("axes fraction", "data"), textcoords="offset points",
                fontsize=6.0, ha="right", va="bottom", color=style.CAPPED)
    ax.axvline(0, color=style.ZERO_LINE, lw=0.8)
    # The curves stop where the SHORTEST run stopped. These are each seed's own last checkpoint
    # -- still a median ~175 updates before its run ended, so they understate the final gap.
    fp, fa, ff = final_checkpoint(scenario, RULE_SHOWN, task)
    ax.text(0.98, 0.06,
            f"last checkpoint: probe {fp.mean():.0f}, argmax {fa.mean():.0f}\n"
            f"probe above floor {fp.mean() - ff.mean():+.1f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0, linespacing=1.35)
    ax.set_ylim(-2, 102)
    ax.set_title(f"{LABEL[scenario]} · task {task + 1}")
    style.panel_label(ax, letter, dx=-0.17)


if __name__ == "__main__":
    w = style.WIDTH["page"] / 2.0
    fig, axes = plt.subplots(2, 2, figsize=(2 * w, 2 * w * 0.62), sharex=True, sharey=True)
    for r, task in enumerate((0, 1)):
        for c, scenario in enumerate(SCENARIOS):
            panel(axes[r][c], scenario, task, "abcd"[r * 2 + c])
    for c in range(2):
        axes[1][c].set_xlabel("updates relative to the task switch")
    for r in range(2):
        axes[r][0].set_ylabel("accuracy (%)")
    axes[0][0].legend(loc="lower left", fontsize=6.5, handlelength=1.5, borderpad=0.15)

    out = figure_path(__file__)
    fig.savefig(out)
    print(f"saved {out}")
    print("\n  at each seed's OWN last checkpoint, mean over seeds (backprop):")
    for scenario in SCENARIOS:
        for task in (0, 1):
            p, a, f = (v.mean() for v in final_checkpoint(scenario, RULE_SHOWN, task))
            print(f"    {LABEL[scenario]:10s} task {task + 1}   probe {p:5.1f}   "
                  f"argmax {a:5.1f}   gap {p - a:+6.1f}   floor {f:5.1f}   "
                  f"probe ABOVE floor {p - f:+5.1f}")
