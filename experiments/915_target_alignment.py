"""Does PC aim its updates toward the solution better than backprop, and does that track forgetting?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R3.

ONE PLOT, one grid cell, two panels, as 007's card asks: alignment through training, and alignment
against retention. Form is 007's SK_TA.

THE MEASURE IS SONG & BOGACZ'S OWN (their Fig. 3b):

    alignment = cos(target - out_before,  out_after - out_before)

-- of the movement an update produced in the output, how much went toward where the target
actually was. 1 is straight at it, 0 sideways, negative away. Testing the mechanism a paper
credits is not optional, and the result belongs in Results whichever way it falls.

TWO VERSIONS ARE DRAWN, AND THE SECOND IS THE ONE THAT BEARS ON FORGETTING:
    trained batch   the replication of their measurement, on the data being learned.
    task-1 batch    the SAME cosine on a fixed task-1 batch that is never trained on, measured
                    during task 2. Negative means each update actively pushes task-1 outputs away
                    from their own targets -- forgetting caught per update rather than inferred
                    from an endpoint, and because it is a rate it does not inherit the budget.
                    The source paper does not report this.

⚠ THE LEGACY FINDING DOES NOT REPLICATE, AND THAT IS THE RESULT. 007's card records the earlier
measurement (5 seeds, four rules): "alignment does not track forgetting, and ranks the rules
opposite to retention". Re-run at 10 seeds under the current protocol as the card asks, it does
track it, strongly:

    r(mean post-switch alignment, final task-1)   Class-IL  +0.96 bp  +0.96 pc
                                                  Domain-IL +0.74 bp  +0.71 pc

⚠ and this is NOT the matched-competence run-length artefact that caught 916. Controlling for
task-2 length the partial correlation is still +0.95 to +0.97 in Class-IL. Length is not the
mediator.

THE RULE RANKING IS THE HONEST QUALIFIER. PC aligns better than backprop on the trained batch in
both scenarios (+0.0245 ± 0.0046 Class-IL, 5.4 sem). Retention ranks the same way in Class-IL
(PC ahead) and the OPPOSITE way in Domain-IL (PC behind despite aligning better). So alignment
predicts retention across seeds within a scenario, and does not explain the between-rule
difference in the scenario where that difference is negative.

THIRD PANEL, ADDED AFTER 808: under sigmoid, Domain-IL is the one place PC's crossover advantage
is real AND large (+0.74 +/- 0.19, Cohen's d = 1.25) -- the single best chance for the credited
mechanism to show up. It does not: interference-alignment (task-1 reference batch, POST-SWITCH
ONLY, matching this figure's own convention -- a rougher whole-run average gives ~0.015/0.014,
still the same conclusion) is +0.0050 (backprop) vs +0.0032 (pc), both small and barely distinct.
So target alignment fails to explain PC's advantage even in the one condition engineered to give
it the best possible shot -- a stronger negative than the tanh result alone, which had no
advantage to explain in the first place.

Styling follows the 300-series scripts: tab: colours, dpi 120, bbox_inches="tight", 9pt labels.
No shared style module.

PROVENANCE
    803_mechanism_logged_{scenario}.npz   both scenarios, backprop and pc, seeds 10-19,
    config_800.yaml, matched competence. Alignment sampled every 5th update; the reference batch
    is 40 images per task-1 class, drawn from train and never trained on.
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
MARKER = {"backprop": "s", "pc": "o"}
LS = {"backprop": "-", "pc": "--"}
IDX, VAL = 0, 1
LEAD = 150          # updates of task-1 context drawn before the switch


def runs(scenario, rule):
    d = np.load(EXP / f"803_mechanism_logged_{scenario}.npz", allow_pickle=True)
    out = []
    for i, m in enumerate(d["methods"]):
        if m != rule:
            continue
        out.append((np.asarray(d[f"align_{i}"], float), np.asarray(d[f"align_ref_{i}"], float),
                    int(d[f"switch0_{i}"]), float(d[f"final_t1_{i}"])))
    return out


def mean_post(scenario, rule, which):
    """Per-seed mean alignment over that seed's own full task 2."""
    return np.array([r[which][r[which][:, IDX] > r[2], VAL].mean() for r in runs(scenario, rule)])


def runs_808(rule):
    """808's sigmoid + Domain-IL run -- same save schema as 803, different file."""
    d = np.load(EXP / "808_sigmoid_domain_mechanism_domain_il.npz", allow_pickle=True)
    out = []
    for i, m in enumerate(d["methods"]):
        if m != rule:
            continue
        out.append((np.asarray(d[f"align_{i}"], float), np.asarray(d[f"align_ref_{i}"], float),
                    int(d[f"switch0_{i}"]), float(d[f"final_t1_{i}"])))
    return out


def mean_post_808(rule, which):
    return np.array([r[which][r[which][:, IDX] > r[2], VAL].mean() for r in runs_808(rule)])


if __name__ == "__main__":
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.2))

    # ---- panel 1: alignment through training, over the window every seed reached
    for scenario in SCENARIOS:
        for rule in RULES:
            rs = runs(scenario, rule)
            for which, alpha, lw in ((0, 1.0, 1.8), (1, 0.45, 1.3)):
                tr = [r[which] for r in rs]
                sws = [r[2] for r in rs]
                lo = max(-LEAD, min(int((t[:, IDX] - s).min()) for t, s in zip(tr, sws)))
                hi = min(int((t[:, IDX] - s).max()) for t, s in zip(tr, sws))
                g = np.arange(lo, hi + 1, 5)
                st = np.vstack([np.interp(g, t[:, IDX] - s, t[:, VAL]) for t, s in zip(tr, sws)])
                ax1.plot(g, st.mean(axis=0), ls=LS[rule], lw=lw, alpha=alpha,
                         color=COLORS[scenario],
                         label=(f"{NICE[scenario]} · {rule}" if which == 0 else None))
    ax1.axhline(0, color="k", lw=1.0)
    ax1.axvline(0, color="0.3", lw=1.2)
    ax1.annotate("switch", (0, 0.52), xytext=(4, 0), textcoords="offset points", fontsize=8)
    ax1.annotate("solid: trained batch\nfaint: fixed task-1 batch", (0.98, 0.04),
                 xycoords="axes fraction", ha="right", fontsize=7.5, linespacing=1.3)
    ax1.set_xlabel("updates relative to the task switch", fontsize=9)
    ax1.set_ylabel("target alignment", fontsize=9)
    ax1.set_title("On the task being trained PC aims better; on the task it is not,\n"
                  "both rules go negative and PC more so", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7.5, loc="upper left")

    # ---- panel 2: alignment against retention, one point per seed
    for scenario in SCENARIOS:
        for rule in RULES:
            a, r = mean_post(scenario, rule, 0), np.array([x[3] for x in runs(scenario, rule)])
            rr = np.corrcoef(a, r)[0, 1]
            ax2.scatter(a, r, s=30, marker=MARKER[rule], color=COLORS[scenario],
                        alpha=0.8, label=f"{NICE[scenario]} · {rule}  (r = {rr:+.2f})")
            b = np.polyfit(a, r, 1)
            xs = np.linspace(a.min(), a.max(), 20)
            ax2.plot(xs, np.polyval(b, xs), lw=1.2, ls=LS[rule], color=COLORS[scenario], alpha=0.6)
    ax2.set_xlabel("mean target alignment over task 2 (trained batch)", fontsize=9)
    ax2.set_ylabel("final task-1 accuracy (%)", fontsize=9)
    ax2.set_title("It DOES track retention at 10 seeds — the 5-seed\n"
                  "legacy result (flat) does not replicate", fontsize=9)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7.5, loc="upper left")

    # ---- panel 3: the direct test -- sigmoid, Domain-IL, where the advantage is real and large
    bp_ref = mean_post_808("backprop", 1)
    pc_ref = mean_post_808("pc", 1)
    bp_ret = np.array([x[3] for x in runs_808("backprop")])
    pc_ret = np.array([x[3] for x in runs_808("pc")])
    xpos = {"backprop": 0, "pc": 1}
    for rule, vals in (("backprop", bp_ref), ("pc", pc_ref)):
        ax3.scatter(np.full(vals.shape, xpos[rule]), vals, s=30, marker=MARKER[rule],
                    color="tab:green", alpha=0.7)
        ax3.scatter([xpos[rule]], [vals.mean()], s=120, marker="_", color="k", lw=2.5, zorder=5)
    ax3.axhline(0, color="k", lw=1.0)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["backprop", "pc"])
    ax3.set_xlim(-0.5, 1.5)
    ax3.set_ylabel("interference-alignment (task-1 ref, during task 2)", fontsize=9)
    ax3.set_title("Sigmoid, Domain-IL: crossover advantage is real (d=1.25)\n"
                  "but alignment barely differs -- not the mechanism, even here", fontsize=9)
    ax3.grid(alpha=0.25, axis="y")
    print(f"\n  sigmoid/domain_il (808): interference-alignment  bp {bp_ref.mean():+.4f}  "
          f"pc {pc_ref.mean():+.4f}   retention bp {bp_ret.mean():.1f}  pc {pc_ret.mean():.1f}")

    fig.tight_layout()
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    for scenario in SCENARIOS:
        for which, name in ((0, "trained batch"), (1, "task-1 batch")):
            bp, pc = mean_post(scenario, "backprop", which), mean_post(scenario, "pc", which)
            m, se, nsem = paired_diff(pc, bp)
            print(f"  {NICE[scenario]:10s} {name:14s} bp {bp.mean():+.4f}  pc {pc.mean():+.4f}  "
                  f"pc-bp {m:+.4f}±{se:.4f} ({nsem:.1f} sem)")
