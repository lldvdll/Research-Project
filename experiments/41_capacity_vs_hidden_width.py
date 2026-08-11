"""Does the network have the capacity to hold both tasks at once, and at what width?

Slide 7, first half. Deviation from the protocol (presentation_plan.md §2): joint training, no
task sequence, and the hidden width is swept. Backprop only -- capacity is a property of the
architecture, not of the learning rule, and whether each RULE can reach the ceiling is script 43.

WHY THIS RUNS BEFORE ANYTHING COMPARATIVE
    If the network cannot represent all ten classes at once, no learning rule can hold both
    tasks, and forgetting measured at that width is confounded with plain underfitting. Every
    later result would be uninterpretable. This fixes H, and H is inherited from nothing.

THE BUDGET IS MEASURED, NOT GUESSED -- and this is the whole experiment
    "Accuracy after N updates" answers this question only if N is past convergence. Otherwise
    the curve flattens because the BUDGET binds at every width, and that flat region gets
    mistaken for a capacity ceiling. A first attempt at 2500 updates (1.3 epochs) showed every
    width above 8 sitting at ~87% and would have concluded H=16. Run to convergence the same
    widths reach 91.7 / 93.5 / 94.0% at H=16 / 64 / 256 -- a real and ordered difference that
    the short budget had entirely hidden.

    A separate probe at one seed established: larger H converges LATER and higher, and the
    largest width tried needs ~22k updates. MAX_ITERS below is set from that, and every width
    gets the same budget.

MATCHED CONDITIONS
    For a given seed, every width sees the same class split, the same data order and the same
    initialisation seed. Only H differs. The same holds for the Domain-IL check.

ONE SCENARIO SWEPT, THE OTHER CHECKED AT A POINT
    Class-IL is the primary scenario, so it is swept. Domain-IL is checked only at the chosen H,
    to confirm the width is not capacity-limiting there before script 45 relies on it. That is
    enough because a full Domain-IL sweep showed it plateaus at a SMALLER width than Class-IL
    and at a similar level -- five shared outputs is a coarser discrimination than ten, so it is
    not the binding constraint. Sweeping both cost twice the compute to establish that.

    Note the two scenarios have different chance levels -- 1/10 against 1/5 -- so their absolute
    accuracies are not directly comparable. Each scenario's ceiling is the reference for its own
    retention numbers, and nothing else.

HOW H IS CHOSEN, stated before looking
    the smallest width whose converged accuracy is within TOLERANCE of the best width tried.
    The whole curve is plotted and the gain from each doubling is printed, so the choice is
    visible and arguable rather than asserted.

READINGS COMMITTED BEFORE RUNNING
    Accuracy rises with width and then flattens. If it is still climbing at the largest width,
    the problem is capacity-limited everywhere below and the sweep must be extended before
    anything else runs. If it flattens, the onset is H, there is room to hold both tasks, and
    forgetting seen later is a property of the rule rather than of the architecture.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, build, replace, figure_path, array_path
from src.runner import run_classil

# ---------------------------------------------------------------- settings
WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
MAX_ITERS = 25000       # from the convergence probe: the largest width needs ~22k
EVAL_EVERY = 250
PLATEAU_EVALS = 6       # average the last few evals rather than taking a lucky maximum
SEEDS = 5
TOLERANCE = 1.0         # percentage points from the best width
RISE_TOL = 0.5          # a curve still gaining more than this over the last quarter is not converged

base = replace(PROTOCOL,
               stop_threshold=None,                # no early stopping: we want the ceiling
               max_iters_per_task=MAX_ITERS,
               eval_every=EVAL_EVERY,
               seeds=SEEDS)


def joint(proto, seed, data):
    """One joint-training run: all ten classes in a single block. Returns the accuracy curve."""
    pair = proto.tasks(seed)                       # same split for every width at this seed
    train_step, predict = build(proto, "backprop", seed)
    out = run_classil(
        train_step, predict, [pair[0] + pair[1]], data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=MAX_ITERS, batch=proto.batch, eval_every=EVAL_EVERY,
        device=proto.device, data_seed=seed,       # same data order for every width
        label_map=proto.label_map(pair),
    )
    return out["steps"], out["curves"]["argmax"][:, 0] * 100.0


# ---------------------------------------------------------------- sweep, class-IL
# `python 41_... --replot` redraws from the saved arrays without retraining. The sweep is 25k
# updates x 8 widths x 5 seeds and takes ~22 minutes; adjusting a label should not cost that.
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

SCENARIOS = ["class_il", "domain_il"]

if REPLOT:
    z = np.load(array_path(__file__))
    acc = {s: z[f"acc_{s}"] for s in SCENARIOS}
    rise = {s: z[f"rise_{s}"] for s in SCENARIOS}
    learning = {s: {w: z[f"curve_{s}_{w}"] for w in WIDTHS} for s in SCENARIOS}
    steps, data = z["steps"], None
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    data = load(replace(base, hidden=1))            # hidden is irrelevant to loading
    acc = {s: np.zeros((len(WIDTHS), SEEDS)) for s in SCENARIOS}
    rise = {s: np.zeros((len(WIDTHS), SEEDS)) for s in SCENARIOS}
    learning = {s: {} for s in SCENARIOS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        for wi, width in enumerate(WIDTHS):
            proto = replace(base, hidden=width, scenario=scen)
            curves = []
            for seed in range(SEEDS):
                steps, c = joint(proto, seed, data)
                curves.append(c)
                k = PLATEAU_EVALS
                acc[scen][wi, seed] = c[-k:].mean()
                # Convergence is tested as "is it still RISING", not "is the best above the
                # final". Small widths oscillate several points around a flat mean, and a
                # best-minus-final test calls that undertrained when it is only noisy.
                rise[scen][wi, seed] = c[-k:].mean() - c[-3 * k:-2 * k].mean()
            learning[scen][width] = np.mean(curves, axis=0)
            print(f"  {scen:10s} H={width:4d}  {acc[scen][wi].mean():5.1f} ± "
                  f"{acc[scen][wi].std(ddof=1)/np.sqrt(SEEDS):.1f}%"
                  f"   still rising {rise[scen][wi].mean():+.2f} pts"
                  f"   [{time.perf_counter()-t0:5.0f}s]")

mean = {s: acc[s].mean(1) for s in SCENARIOS}
sem = {s: acc[s].std(1, ddof=1) / np.sqrt(SEEDS) for s in SCENARIOS}

# ---------------------------------------------------------------- choose H
# H must be adequate in BOTH scenarios, so the criterion is applied to both and the larger wins.
per_scen = {s: next((w for w, m in zip(WIDTHS, mean[s]) if m >= mean[s].max() - TOLERANCE),
                    WIDTHS[-1]) for s in SCENARIOS}
chosen = max(per_scen.values())
print(f"\n  H = {chosen}   (smallest width within {TOLERANCE:.1f} points of that scenario's best: "
      + ", ".join(f"{s} {per_scen[s]}" for s in SCENARIOS) + ")")
for s in SCENARIOS:
    print(f"  {s:10s} best {mean[s].max():.1f}% at H={WIDTHS[int(np.argmax(mean[s]))]}"
          f" | gain per doubling: "
          + ", ".join(f"{a}->{b}:{mean[s][i+1]-mean[s][i]:+.1f}"
                      for i, (a, b) in enumerate(zip(WIDTHS[:-1], WIDTHS[1:]))))
    r = rise[s].mean(1).max()
    print(f"  {s:10s} {'WARNING: still gaining' if r > RISE_TOL else 'converged; largest gain'}"
          f" {r:+.2f} points over the last quarter")

COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
CHANCE = {"class_il": 10, "domain_il": 20}

# ---------------------------------------------------------------- main figure
fig, ax = plt.subplots(figsize=(7.6, 5))
for s in SCENARIOS:
    ax.errorbar(WIDTHS, mean[s], yerr=sem[s], marker="o", capsize=3, lw=2, color=COLORS[s],
                label=f"{s.replace('_', '-')}  ({10 if s == 'class_il' else 5} outputs)")
    ax.axhline(CHANCE[s], color=COLORS[s], ls=":", lw=1, alpha=0.6)
    ax.text(WIDTHS[0], CHANCE[s] + 1.5, f"chance {CHANCE[s]}%", fontsize=7, color=COLORS[s])
ax.axvline(chosen, color="k", ls="--", lw=1)
ax.annotate(f"H = {chosen}", xy=(chosen, 55), xytext=(8, 0), textcoords="offset points",
            fontsize=12, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(WIDTHS); ax.set_xticklabels(WIDTHS)
ax.set_xlabel("hidden units")
ax.set_ylabel("accuracy on all 10 classes (%)")
ax.set_ylim(0, 100)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ------------------------------------------------- supporting figure: is the budget enough?
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
cmap = plt.get_cmap("viridis")
for a, s in zip(ax2, SCENARIOS):
    for i, width in enumerate(WIDTHS):
        a.plot(steps, learning[s][width], lw=1.6, color=cmap(i / max(1, len(WIDTHS) - 1)),
               label=f"{width}")
    a.set_xlabel("training step"); a.set_ylim(0, 100); a.grid(alpha=0.25)
    a.set_title(s.replace("_", "-"))
ax2[0].set_ylabel("accuracy (%)")
ax2[1].legend(fontsize=7, ncol=2, loc="lower right", title="hidden units")
fig2.suptitle("Diagnostic: curves must be flat at the right edge, "
              "or the sweep is measuring the training budget")
fig2.tight_layout()
fig2.savefig(figure_path(__file__, "convergence"), dpi=120, bbox_inches="tight")
print(f"saved {figure_path(__file__, 'convergence')}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__),
         widths=np.asarray(WIDTHS), steps=np.asarray(steps), chosen=chosen, max_iters=MAX_ITERS,
         **{f"acc_{s}": acc[s] for s in SCENARIOS},
         **{f"rise_{s}": rise[s] for s in SCENARIOS},
         **{f"curve_{s}_{w}": learning[s][w] for s in SCENARIOS for w in WIDTHS})
print(f"saved {array_path(__file__)}")
