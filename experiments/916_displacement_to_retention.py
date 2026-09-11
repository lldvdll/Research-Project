"""Does how far PC's state moves during settling predict how much is forgotten?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R3.

TWO PLOTS, ONE PER GRID CELL, matching 007's two cards:
    916_displacement_to_retention_a.png   the bridge, form from SK_DX
    916_displacement_to_retention_b.png   where settling reaches the weights -- 344 regenerated,
                                          which 007's card asks to fold in as a panel of the bridge

THE CHAIN THIS TESTS, link by link:
    PC dynamics  ->  internal configuration D  ->  weight change  ->  forgetting

where D = ||x*(settled) - x(feedforward)||, the distance between the relaxed hidden state and the
one the feedforward pass would have produced.

⚠ D IS ZERO FOR BACKPROP BY CONSTRUCTION -- it has no relaxation, so there is nothing to displace.
Panel (a) therefore describes PC and is not a rule comparison; a non-zero D is a definition, not
evidence. What CAN be measured is whether D varies across PC seeds and whether that variation
goes anywhere. It does:

    r(D, total ||dW2|| over task 2)   Class-IL −0.85   Domain-IL −0.62
    r(D, final task-1 accuracy)       Class-IL +0.84   Domain-IL +0.29

More displacement, less total movement in the output weights, more retention. ⚠ But the second
link SURVIVES ONLY IN CLASS-IL: controlling for task-2 length the partial correlation is +0.85 in
Class-IL and +0.12 in Domain-IL. The mechanism has explanatory power exactly where PC helps, and
none where it does not, which is consistent with R2's sign flip rather than an answer to it.

⚠ THE CONFOUND THAT MADE ME USE TOTALS. Matched competence makes task-2 length a dependent
variable (119-4999 updates here). Long runs have a small MEAN per-update ||dW|| and also forget
more, so mean step size correlates with retention at r = +0.93 -- which reads as "bigger updates
preserve task 1" and is an artefact of run length. The TOTAL path summed over task 2 has no such
problem and gives the interpretable sign. Every weight claim here is on totals.

THIRD SUBPLOT ADDED TO (a), AFTER 808: does the D -> retention link strengthen under sigmoid,
where PC actually has something to explain in Domain-IL (+0.74 crossover, d=1.25)? If the
mechanism is real, sigmoid/Domain-IL should look more like tanh/Class-IL (partial r ~+0.85) than
tanh/Domain-IL (partial r ~+0.12).

WHAT (b) ADDS, AND WHY IT REFUTED ITS OWN PREDICTION. 344 predicted PC's damping would show in W1,
whose error signal comes through settling, and not in W2, which uses the same direct target error
for both rules. The opposite happened. Log-log slope of realised ||dW|| against nominal lr:

    W1   backprop 0.93   pc 0.90     no differential damping
    W2   backprop 0.99   pc 0.78     PC's output step scales sub-linearly

PC's output update multiplies the same error against the SETTLED hidden activity, so settling
reaches W2 by a route the hypothesis did not anticipate. It belongs in the argument because it
refuted its own pre-registration.

Styling follows the 300-series scripts: tab: colours, dpi 120, bbox_inches="tight", 9pt labels.
No shared style module.

PROVENANCE
    803_mechanism_logged_{scenario}.npz   D and ||dW|| per layer on EVERY update, seeds 10-19,
                                          config_800.yaml, matched competence. Nothing subsampled.
    344_weight_step_vs_lr.npz             PRE-800. Realised mean |dW| per layer across the shared
                                          lr grid, backprop and pc, 5 seeds.
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
METHOD_COLOR = {"backprop": "0.35", "pc": "tab:orange"}     # as 344/340 use


def pc_runs(scenario, path=None):
    """Per PC seed: (mean displacement over task 2, total ||dW2|| over task 2, retention)."""
    d = np.load(path or EXP / f"803_mechanism_logged_{scenario}.npz", allow_pickle=True)
    out = []
    for i, m in enumerate(d["methods"]):
        if m != "pc":
            continue
        sw = int(d[f"switch0_{i}"])
        dp, dw = np.asarray(d[f"disp_{i}"], float), np.asarray(d[f"dW_{i}"], float)
        post = np.arange(len(dp)) > sw
        out.append((dp[post].mean(), dw[post, 1].sum(), float(d[f"final_t1_{i}"])))
    return np.array(out)


def partial(x, y, z):
    rxy, rxz, ryz = (np.corrcoef(a, b)[0, 1] for a, b in ((x, y), (x, z), (y, z)))
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def panel_a():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for s in SCENARIOS:
        D, W2, R = pc_runs(s).T
        r1, r2 = np.corrcoef(D, W2)[0, 1], np.corrcoef(D, R)[0, 1]
        ax1.scatter(D, W2, s=34, color=COLORS[s], alpha=0.85,
                    label=f"{NICE[s]}  (r = {r1:+.2f})")
        ax2.scatter(D, R, s=34, color=COLORS[s], alpha=0.85,
                    label=f"{NICE[s]}  (r = {r2:+.2f})")
        for ax, y in ((ax1, W2), (ax2, R)):
            b = np.polyfit(D, y, 1)
            xs = np.linspace(D.min(), D.max(), 20)
            ax.plot(xs, np.polyval(b, xs), lw=1.3, color=COLORS[s], alpha=0.6)
    ax1.set_xlabel("mean settling displacement $D$ over task 2", fontsize=9)
    ax1.set_ylabel("total $\\|\\Delta W_2\\|$ over task 2", fontsize=9)
    ax1.set_title("link 1: more displacement, less total\noutput-weight movement", fontsize=9)
    ax2.set_xlabel("mean settling displacement $D$ over task 2", fontsize=9)
    ax2.set_ylabel("final task-1 accuracy (%)", fontsize=9)
    ax2.set_title("link 2: and more retention — but only in Class-IL\n"
                  "(partial r | task-2 length: +0.85 vs +0.12)", fontsize=9)

    # Third subplot: sigmoid, Domain-IL (808) -- does the link strengthen where PC actually wins?
    D8, W28, R8 = pc_runs("domain_il", path=EXP / "808_sigmoid_domain_mechanism_domain_il.npz").T
    r8 = np.corrcoef(D8, R8)[0, 1]
    ax3.scatter(D8, R8, s=34, color="tab:green", marker="^", alpha=0.85,
                label=f"Domain-IL · sigmoid  (r = {r8:+.2f})")
    b8 = np.polyfit(D8, R8, 1)
    xs8 = np.linspace(D8.min(), D8.max(), 20)
    ax3.plot(xs8, np.polyval(b8, xs8), lw=1.3, color="tab:green", alpha=0.6)
    d8 = np.load(EXP / "808_sigmoid_domain_mechanism_domain_il.npz", allow_pickle=True)
    L8 = np.array([len(d8[f"disp_{i}"]) - int(d8[f"switch0_{i}"])
                   for i, m in enumerate(d8["methods"]) if m == "pc"], dtype=float)
    p8 = partial(D8, R8, L8)
    ax3.set_xlabel("mean settling displacement $D$ over task 2", fontsize=9)
    ax3.set_ylabel("final task-1 accuracy (%)", fontsize=9)
    ax3.set_title(f"sigmoid: does D predict retention where PC\nactually wins? "
                  f"(partial r | task-2 len: {p8:+.2f})", fontsize=9)

    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("PC only — backprop's displacement is identically zero by construction",
                 fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__, "a")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    for s in SCENARIOS:
        D, W2, R = pc_runs(s).T
        # TASK-2 length, not total run length -- the confound being controlled for is how long
        # task 2 ran, and total length would fold task 1's own variable phase into it.
        d = np.load(EXP / f"803_mechanism_logged_{s}.npz", allow_pickle=True)
        L = np.array([len(d[f"disp_{i}"]) - int(d[f"switch0_{i}"])
                      for i, m in enumerate(d["methods"]) if m == "pc"], dtype=float)
        print(f"  {NICE[s]:10s} r(D,W2) {np.corrcoef(D, W2)[0, 1]:+.2f}   "
              f"r(D,ret) {np.corrcoef(D, R)[0, 1]:+.2f}   "
              f"partial r(D,ret | task-2 len) {partial(D, R, L):+.2f}")
    print(f"  domain_il·sigmoid (808)  r(D,ret) {r8:+.2f}   partial r(D,ret | task-2 len) {p8:+.2f}")


def panel_b():
    rows = np.load(EXP / "344_weight_step_vs_lr.npz", allow_pickle=True)["data"]
    lrs = sorted({float(r[1]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, (col, lab) in zip(axes, ((3, "$W_1$ (hidden)"), (4, "$W_2$ (output)"))):
        for m in ("backprop", "pc"):
            y = [np.mean([float(r[col]) for r in rows
                          if r[0] == m and float(r[1]) == g]) for g in lrs]
            slope = np.polyfit(np.log(lrs), np.log(y), 1)[0]
            ax.plot(lrs, y, marker="o", ms=5, lw=1.8, color=METHOD_COLOR[m],
                    label=f"{m}   slope {slope:.2f}")
            print(f"  {lab:16s} {m:9s} log-log slope {slope:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(lrs)
        ax.set_xticklabels([str(g) for g in lrs], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("nominal learning rate", fontsize=9)
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("realised mean $|\\Delta W|$ per step", fontsize=9)
    fig.suptitle("Damping shows in the OUTPUT layer, not the hidden one — the opposite of "
                 "344's prediction", fontsize=9)
    fig.tight_layout()
    out = figure_path(__file__, "b")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    panel_a()
    panel_b()
