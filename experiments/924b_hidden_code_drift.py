"""Where does Domain-IL's damage sit -- does the hidden code drift, and for which units?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R5.

ONE PLOT, one grid cell, form from 007's SK_DRIFT: hidden-code displacement through task 2,
resolved by whether a unit is task-1 or task-2 selective, against a frozen-trunk baseline.

⚠ THIS CLOSES WHAT 007 CALLS THE LARGEST GENUINE GAP IN THE PROJECT. Output suppression is
structurally impossible in Domain-IL, so its forgetting must be representational -- and until now
freezing only said where it is NOT. No saved array held hidden codes (803 logs the probe's
accuracy, not the code), so run 807 was written to log them.

WHAT IS MEASURED, on a fixed task-1 batch never trained on, every 10 updates through task 2:
    rel_l2   mean ||now − at-switch|| / ||at-switch||, per image. 0 = unchanged.
    split by unit selectivity, labelled AT THE SWITCH by which task each hidden unit responds to
    more strongly. The pre-registered expectation was that if the damage is representational, the
    units that carried task 1 should move MORE than the ones that did not.

⚠ THEY DO NOT. Drift is essentially uniform across the two groups:
        Class-IL   task-1 units 0.439   task-2 units 0.448   all 0.429
        Domain-IL  task-1 units 0.375   task-2 units 0.398   all 0.381
    The code moves a long way -- about 40% of its own norm -- but it does not move preferentially
    where task 1 lived. So "task-1 units are overwritten" is NOT supported by the measurement that
    was built to test it. The drift is real and undifferentiated.

THE FROZEN-TRUNK ARM IS THE FLOOR, not a comparison: W1 and b1 both held, so the code cannot move
and drift is exactly 0.0000. It proves the measurement reads the trunk rather than eval noise.
⚠ Freezing W1 alone is NOT enough -- the hidden bias keeps learning and the code still drifts
(rel_l2 0.14 against 0.56 unfrozen). 807 freezes both for that reason.

⚠ AND THE FROZEN ARM CARRIES THE RESULT THE FIGURE TURNS ON. With the trunk completely frozen --
drift exactly 0.0000, the representation provably untouched -- Class-IL task-1 accuracy still falls
to 0.0%, against 4.8% when the trunk is free. So in Class-IL the readout ALONE is sufficient to
destroy task 1, and preventing all representational change does not merely fail to help, it is
slightly WORSE. Domain-IL's frozen arm keeps 30.4% against 37.6% free -- also worse.

THE HONEST CONCLUSION, read with 922 (freezing the readout recovers nothing either). Forgetting
here is not localised to a layer. Either layer left free is enough to destroy task-1 argmax in
Class-IL, which is exactly why protecting one of them never helped, and freezing a layer costs
more than it saves because the remaining layer must absorb all of task 2. That is a negative
result about localisation rather than the positive localisation 007 hoped for -- and it is a
answer, where before there was none.

Styling follows the 300-series scripts: tab: colours by scenario, dpi 120, bbox_inches="tight",
9pt labels. No shared style module.

PROVENANCE
    807_code_drift_{scenario}.npz   NEW 800-series run. Both scenarios, backprop and pc, seeds
    10-19, config_800.yaml, matched competence, plus a frozen-trunk backprop arm. Codes sampled
    every 10 updates on 40 images per task-1 class.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
SCEN = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
# drift columns: step, cosine, rel_l2, rel_l2 on task-1 units, rel_l2 on task-2 units
STEP, COS, REL, REL_T1, REL_T2 = 0, 1, 2, 3, 4


def arms(scenario):
    d = np.load(EXP / f"807_code_drift_{scenario}.npz", allow_pickle=True)
    out = {}
    for i in range(len(d["methods"])):
        key = (str(d["methods"][i]), bool(d["frozen"][i]))
        out.setdefault(key, []).append(
            (np.asarray(d[f"drift_{i}"], float), float(d[f"final_t1_{i}"]),
             int(d[f"n_t1_units_{i}"])))
    return out


def common(traces, col):
    """Mean over seeds on the window every seed reached -- the intersection, as 915 uses."""
    n = min(len(t) for t in traces)
    return np.vstack([t[:n, col] for t in traces]).mean(axis=0), traces[0][:n, STEP]


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, s in zip(axes, SCEN):
        a = arms(s)
        tr = [t for t, _, _ in a[("backprop", False)]]
        for col, lab, ls in ((REL_T1, "task-1 selective units", "-"),
                             (REL_T2, "task-2 selective units", "--")):
            mu, steps = common(tr, col)
            ax.plot(np.arange(len(mu)) * 10, mu, ls=ls, lw=2.0, color=COLORS[s], label=lab)
        mu_all, _ = common(tr, REL)
        ax.plot(np.arange(len(mu_all)) * 10, mu_all, lw=1.2, color="0.35", label="all units")
        frozen = [t for t, _, _ in a[("backprop", True)]]
        mu_f, _ = common(frozen, REL)
        ax.plot(np.arange(len(mu_f)) * 10, mu_f, lw=2.0, color="tab:red",
                label="frozen trunk (floor)")

        n1 = np.mean([n for _, _, n in a[("backprop", False)]])
        fin = np.mean([f for _, f, _ in a[("backprop", False)]])
        finf = np.mean([f for _, f, _ in a[("backprop", True)]])
        ax.annotate(f"{n1:.0f}/32 units task-1 selective at the switch\n"
                    f"final task-1: {fin:.1f}% free, {finf:.1f}% frozen trunk",
                    (0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=7.5,
                    linespacing=1.35)
        ax.set_xlabel("updates into task 2", fontsize=9)
        ax.set_title(NICE[s], fontsize=9)
        ax.grid(alpha=0.25)
        print(f"  {NICE[s]:10s} end drift  all {mu_all[-1]:.3f}   "
              f"task-1 units {common(tr, REL_T1)[0][-1]:.3f}   "
              f"task-2 units {common(tr, REL_T2)[0][-1]:.3f}   "
              f"frozen {mu_f[-1]:.3f}   final t1 free {fin:.1f} frozen {finf:.1f}")
    axes[0].set_ylabel("hidden-code drift from the switch (rel. $L_2$)", fontsize=9)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("The code does drift — and in Class-IL a frozen trunk still loses task 1 "
                 "entirely, so drift is not what destroys it", fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
