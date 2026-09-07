"""How large is PC's effect next to an intervention that is known to work?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R2.

ONE PLOT, one grid cell. 007's card asks for this trimmed to "backprop / PC / replay at the
working point, with replay given its own scale -- its magnitude currently crushes PC's line flat".
That is exactly the amendment 340_lr_sweep_diff needed, and it is why this is a separate figure
from 912: on 912's +-4pp axis replay would compress the entire PC result into a few pixels.

REPLAY IS THE POSITIVE CONTROL, NOT A COMPETITOR. It is not biologically motivated and the report
does not propose it. It is here so that every small or null effect has a scale: without it, "PC
gives +1.42" could be read as "2x5 MNIST at H=32 simply cannot be improved", and no figure would
contradict that. Replay contradicts it.

    paired Δ crossover vs backprop, H = 32, depth 1
        Class-IL    PC +1.42 ± 0.39      replay +9.84 ± 0.77       replay is 6.9× PC
        Domain-IL   PC −0.72 ± 0.19      replay +3.43 ± 0.50       replay positive where PC is not

BROKEN AXIS, AND WHY. The two effects differ by roughly an order of magnitude, so a single linear
axis either hides PC's or squashes replay's. The axis is split so both are legible at their own
resolution, and the break is drawn rather than implied.

WHAT THIS IS NOT SAYING. Replay stores and re-presents task-1 data -- exactly the resource a
continual-learning rule is supposed not to need. The comparison is of EFFECT SIZE under one fixed
protocol, not of cost, and the text says so where it cites this figure.

Styling follows 340, whose arrays these are: tab: colours matching its METHOD_COLOR, dpi 120,
bbox_inches="tight", 9pt labels. No shared style module.

PROVENANCE
    341_width_sweep_{scenario}.npz at H = 32 -- identical configuration to 342 at depth 1, trained
    independently in the two scripts; they agree to 0.01 pp on backprop's mean crossover, which is
    a cheap correctness check rather than redundancy. Each rule runs at its established lr
    (backprop 0.01, PC 0.02), the pairing 314 licenses. Seeds 10-19, matched competence.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path
from src.metrics import paired_diff, paired_sign

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
METHOD_COLOR = {"pc": "tab:orange", "replay": "tab:green"}      # as 340 already uses
STEM, WORKING = "341_width_sweep", 32
CX = 5
BREAK = 5.0        # the axis break: everything below it left, above it right


def by_seed(rows, method):
    d = {int(r[2]): float(r[CX]) for r in rows if r[0] == method and r[1] == WORKING}
    return np.array([d[s] for s in sorted(d)], dtype=float)


if __name__ == "__main__":
    # Broken axis: PC's effect lives in 0-4pp, replay's in 3-11pp.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True,
                                   gridspec_kw=dict(width_ratios=(1, 1), wspace=0.06))
    ypos, labels = [], []
    y = 0.0
    for s in SCENARIOS:
        rows = np.load(EXP / f"{STEM}_{s}.npz", allow_pickle=True)["data"]
        bp = by_seed(rows, "backprop")
        for m in ("pc", "replay"):
            v = by_seed(rows, m)
            d, se, nsem = paired_diff(v, bp)
            w, l, t_, p = paired_sign(v, bp, censored_is_best=True)
            # Each point is drawn ONCE, on whichever side of the break it falls -- drawing on
            # both panels duplicates any value inside the overlap and reads as two measurements.
            host = axL if d < BREAK else axR
            host.errorbar(d, y, xerr=se, marker="o", ms=8, capsize=4, lw=2.2,
                          color=METHOD_COLOR[m])
            host.annotate(f"{d:+.2f}", (d, y), xytext=(0, 11), textcoords="offset points",
                          ha="center", fontsize=8, color=METHOD_COLOR[m])
            ypos.append(y)
            labels.append(f"{NICE[s]} · {m}")
            y -= 1.0
            print(f"  {NICE[s]:10s} {m:7s} {d:+6.2f} ± {se:.2f} ({nsem:.1f} sem)  "
                  f"sign {w}W-{l}L-{t_}T p={p:.3f}")
        y -= 0.55

    for ax in (axL, axR):
        ax.axvline(0, color="k", lw=1.2)
        ax.grid(alpha=0.25, axis="x")
        ax.set_ylim(min(ypos) - 0.8, max(ypos) + 0.8)
    axL.set_xlim(-1.6, BREAK)        # PC both scenarios, plus Domain-IL replay
    axR.set_xlim(8.6, 11.2)          # Class-IL replay alone, far out
    axL.set_yticks(ypos)
    axL.set_yticklabels(labels, fontsize=8.5)
    axR.tick_params(labelleft=False)

    # draw the break
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    for ax, xx in ((axL, 1.0), (axR, 0.0)):
        ax.plot([xx, xx], [0, 1], transform=ax.transAxes, clip_on=False,
                color="k", lw=1.0, ls=(0, (3, 3)))
    fig.supxlabel("paired $\\Delta$ crossover vs backprop (pp)", fontsize=9)
    fig.suptitle("Replay is ~7× PC's effect where PC wins, and positive where PC is negative",
                 fontsize=9)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.86, bottom=0.16)
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
