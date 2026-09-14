"""If the learning rule is not what sets retention, what is?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R3 (close), or Discussion.

THE COMPANION TO 930, AND THE ANSWER TO IT. 930 asks whether the mechanism Song & Bogacz
credit moves where PC's advantage moves, and finds that it does, on sign. This figure asks the
prior question that 930 cannot: how much of the outcome is the rule deciding at all? The answer
is very little, and the three panels are three independent ways of showing the same thing.

    (a)  THE SEED DECIDES THE OUTCOME. Backprop's task-1 retention against PC's, one point per
         seed, three conditions, with the identity line drawn. r = +0.97 / +0.97 / +0.99. The
         two rules see the same class split and the same initialisation at a given seed, and
         they land in the same place to within a point or two. A seed that is hard for backprop
         is hard for PC.

    (b)  THE ONE CORRELATION THIS PROJECT LEANED ON IS MOSTLY THAT SAME SEED VARIANCE. PC's
         settling displacement D predicts its retention at +0.80 in Class-IL, which is where
         the report's own proposed mechanism came from. But D also predicts BACKPROP's
         retention at +0.78 -- and backprop does no settling whatsoever, so that correlation
         cannot be causal in backprop and D must therefore be reading something about the
         split. Partial backprop's retention out, and +0.80 falls to +0.30; under sigmoid it
         REVERSES to -0.25. Raw bar beside partialled bar, per condition.

    (c)  THE SCALE COMPARISON, in one unit. Left cloud: each seed's backprop retention as a
         deviation from the condition mean -- how far the seed moves the outcome. Right cloud:
         the paired PC-minus-backprop retention at the same seeds -- how far the RULE moves it.
         Same axis, same units, points of task-1 accuracy. The rule effect is inside the
         marker size of the seed effect.

WHY (a) IS NOT CIRCULAR, since it is the objection a reader will raise. The two rules are not
being compared to each other here; each is being compared to the split it was handed. If the
learning rule were the dominant term, the points would scatter off the diagonal in the
direction of whichever rule was better. They sit on it.

⚠ WHAT THIS FIGURE DOES NOT SAY. It does not say the rule effect is zero -- 912 measures it and
it is real, significant and paired. It says the rule effect is SMALL RELATIVE TO the variance it
has to be measured against, which is the licence for paired seeds (906) and the reason an
unpaired comparison of these rules reports nothing at all.

⚠ THE NUMBER IN (b) IS A CORRELATION AMONG TEN SEEDS. It is used to REMOVE a claim, not to
establish one, which is the direction of inference n=10 can support. Nothing downstream should
cite +0.30 as an effect size.

COLOURS. The panels carry scenario without a rule split, so the standard gives Class-IL PURPLE
and Domain-IL GREEN. The standard assigns nothing to ACTIVATION, and this figure needs a third
condition (Domain-IL under sigmoid) -- drawn in the same green with an OPEN marker and a
lighter face, so the scenario reading survives and the activation is a secondary mark.
⚠ Raise this with the user rather than treating it as settled convention.

RETENTION, not crossover, is the quantity throughout, because all three panels need one metric
measured on both rules at the same seed and 803/808 log final task-1 accuracy directly. 930
uses crossover for the benefit; the two metrics agree on sign in every condition here (912).

PROVENANCE
    803_mechanism_logged_class_il.npz             tanh, Class-IL, seeds 10-19, both rules
    803_mechanism_logged_domain_il.npz            tanh, Domain-IL, same seeds
    808_sigmoid_domain_mechanism_domain_il.npz    the same instrumentation under sigmoid
    All three log final_t1 (retention), final_t2, and disp (per-update settling displacement,
    identically zero for backprop by construction). D is the mean of disp over the updates
    AFTER the task switch -- the interference phase, which is the only part that can affect
    what is retained. Averaging the whole run instead gives +0.40 in Class-IL rather than
    +0.80, because it dilutes the task-2 phase with task-1 training that cannot forget anything.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"

# (label, file, colour, filled marker?)
# The tick label is the scenario over the activation on two lines. One line ran the third
# condition's label into the neighbouring panel at this figure width.
CONDITIONS = [
    ("Class-IL\ntanh", "803_mechanism_logged_class_il.npz", "tab:purple", True),
    ("Domain-IL\ntanh", "803_mechanism_logged_domain_il.npz", "tab:green", True),
    ("Domain-IL\nsigmoid", "808_sigmoid_domain_mechanism_domain_il.npz", "tab:green", False),
]


def per_seed(fname):
    """Retention and post-switch settling displacement, per seed, per rule.

    `disp` is logged once per WEIGHT UPDATE while `steps`/`t1` are logged every tenth, so the
    switch index from `switch0` indexes disp directly and must not be looked up through steps.
    """
    d = np.load(EXP / fname, allow_pickle=True)
    out = {}
    for i, (m, s) in enumerate(zip(d["methods"], d["seeds"])):
        sw = int(d[f"switch0_{i}"])
        disp = np.asarray(d[f"disp_{i}"], float)
        post = np.arange(1, len(disp) + 1) > sw
        out.setdefault(int(s), {})[str(m)] = (
            float(d[f"final_t1_{i}"]),
            float(disp[post].mean()) if post.any() else np.nan,
        )
    seeds = sorted(k for k, v in out.items() if "backprop" in v and "pc" in v)
    bp = np.array([out[s]["backprop"][0] for s in seeds])
    pc = np.array([out[s]["pc"][0] for s in seeds])
    D = np.array([out[s]["pc"][1] for s in seeds])
    return seeds, bp, pc, D


def r(a, b):
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def partial_r(x, y, z):
    """Correlation of x and y with z held constant -- here, with split difficulty removed."""
    rxy, rxz, ryz = r(x, y), r(x, z), r(y, z)
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))


def panel_a(ax, data):
    lo, hi = 0, 0
    for (lab, _, col, filled), (_, bp, pc, _) in zip(CONDITIONS, data):
        ax.scatter(bp, pc, s=26, color=col if filled else "none",
                   edgecolor=col, linewidth=1.1, alpha=0.85 if filled else 1.0,
                   zorder=3, label=f"{lab.replace(chr(10), ' · ')}   r = {r(bp, pc):+.2f}")
        hi = max(hi, bp.max(), pc.max())
    pad = 0.06 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.55", lw=0.9, ls="--", zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("backprop retention (%)", fontsize=8)
    ax.set_ylabel("PC retention (%)", fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.18)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9, handletextpad=0.4,
              borderpad=0.35)


def panel_b(ax, data):
    x = np.arange(len(CONDITIONS))
    w = 0.34
    for k, ((lab, _, col, filled), (_, bp, pc, D)) in enumerate(zip(CONDITIONS, data)):
        raw, par = r(D, pc), partial_r(D, pc, bp)
        ax.bar(x[k] - w / 2, raw, w, color=col, alpha=0.85 if filled else 0.35,
               edgecolor=col, linewidth=1.1, zorder=3)
        ax.bar(x[k] + w / 2, par, w, color="none", edgecolor=col, linewidth=1.4,
               hatch="///", zorder=3)
        for xx, v in [(x[k] - w / 2, raw), (x[k] + w / 2, par)]:
            ax.annotate(f"{v:+.2f}", (xx, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -9), ha="center", fontsize=6.0)
    ax.axhline(0, color="black", lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CONDITIONS], fontsize=6.0)
    ax.set_ylabel("r (settling displacement,\nPC retention)", fontsize=8)
    ax.set_ylim(-0.45, 1.0)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(alpha=0.18, axis="y")
    # solid = raw, hatched = split difficulty removed. Two entries, drawn colourless so the
    # legend explains the FORM and the colours keep meaning scenario.
    ax.bar(0, 0, 0, color="0.45", label="raw")
    ax.bar(0, 0, 0, color="none", edgecolor="0.45", hatch="///", label="split removed")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9, handletextpad=0.5)


def panel_c(ax, data, rng):
    """Seed effect and rule effect on one axis, both as differences in retention points."""
    # Seed and rule are told apart by MARKER SHAPE, not by a tick label each: six tick labels
    # do not fit across a third of the page, and the condition names have to own that row.
    ymax = 0.0
    for k, ((lab, _, col, filled), (_, bp, pc, _)) in enumerate(zip(CONDITIONS, data)):
        seed_eff = bp - bp.mean()          # how far this seed moves the outcome
        rule_eff = pc - bp                 # how far the rule moves it, same seeds
        for vals, xoff, mk in [(seed_eff, -0.20, "o"), (rule_eff, 0.20, "D")]:
            xc = k + xoff
            ax.scatter(xc + rng.uniform(-0.07, 0.07, len(vals)), vals, s=13, marker=mk,
                       color=col if filled else "none", edgecolor=col,
                       linewidth=0.9, alpha=0.8, zorder=3)
            ax.plot([xc - 0.12, xc + 0.12], [vals.mean()] * 2, color=col, lw=1.8, zorder=4)
            ymax = max(ymax, abs(vals).max())
        # placed above the RULE cloud, which is the short one, so it never lands on a point
        ratio = seed_eff.std(ddof=1) / abs(rule_eff.mean())
        ax.annotate(f"{ratio:.0f}$\\times$", (k + 0.20, rule_eff.max()),
                    textcoords="offset points", xytext=(0, 6), ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=col)
    ax.axhline(0, color="black", lw=1.0, zorder=2)
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels([c[0] for c in CONDITIONS], fontsize=6.0)
    ax.set_xlim(-0.5, len(CONDITIONS) - 0.5)
    ax.set_ylim(-1.85 * ymax, 1.28 * ymax)
    ax.scatter([], [], s=13, marker="o", color="0.45", label="seed")
    ax.scatter([], [], s=13, marker="D", color="0.45", label="rule (PC $-$ backprop)")
    ax.legend(fontsize=6, loc="lower left", framealpha=0.9, handletextpad=0.3,
              borderpad=0.3)
    ax.set_ylabel("effect on retention\n(percentage points)", fontsize=8)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(alpha=0.18, axis="y")


if __name__ == "__main__":
    data = [per_seed(f) for _, f, _, _ in CONDITIONS]
    rng = np.random.RandomState(0)

    for (lab, _, _, _), (seeds, bp, pc, D) in zip(CONDITIONS, data):
        dif = pc - bp
        print(f"  {lab:20s} n={len(seeds)}")
        print(f"     (a) r(bp ret, pc ret)          {r(bp, pc):+.3f}")
        print(f"     (b) r(D, pc ret) raw           {r(D, pc):+.3f}   "
              f"r(D, bp ret) {r(D, bp):+.3f}   partial {partial_r(D, pc, bp):+.3f}")
        print(f"     (c) bp retention {bp.min():5.1f}-{bp.max():5.1f}  "
              f"sd {bp.std(ddof=1):5.2f}   paired pc-bp {dif.mean():+5.2f} "
              f"+- {dif.std(ddof=1) / np.sqrt(len(dif)):.2f}   "
              f"ratio {bp.std(ddof=1) / abs(dif.mean()):.0f}x")

    # (a) is square by construction and needs less width than the two categorical panels,
    # whose three two-line tick labels were running into their neighbours at equal widths.
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45),
                             gridspec_kw=dict(width_ratios=[0.90, 1.03, 1.07]))
    panel_a(axes[0], data)
    panel_b(axes[1], data)
    panel_c(axes[2], data, rng)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    out = figure_path(__file__)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
