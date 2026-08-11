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

if REPLOT:
    z = np.load(array_path(__file__))
    acc, rise, dom, steps = z["acc_class_il"], z["rise"], z["acc_domain_il"], z["steps"]
    learning = {w: z[f"curve_{w}"] for w in WIDTHS}
    data = None
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    data = load(replace(base, hidden=1))            # hidden is irrelevant to loading
    acc = np.zeros((len(WIDTHS), SEEDS))
    rise = np.zeros((len(WIDTHS), SEEDS))
    learning = {}
    t0 = time.perf_counter()

    for wi, width in enumerate(WIDTHS):
        proto = replace(base, hidden=width, scenario="class_il")
        curves = []
        for seed in range(SEEDS):
            steps, c = joint(proto, seed, data)
            curves.append(c)
            k = PLATEAU_EVALS
            acc[wi, seed] = c[-k:].mean()
            # Convergence must be tested as "is it still RISING", not "is the best above the
            # final". Small widths oscillate several points around a flat mean, and a
            # best-minus-final test calls that undertrained when it is only noisy.
            rise[wi, seed] = c[-k:].mean() - c[-3 * k:-2 * k].mean()
        learning[width] = np.mean(curves, axis=0)
        print(f"  H={width:4d}  {acc[wi].mean():5.1f} ± {acc[wi].std(ddof=1)/np.sqrt(SEEDS):.1f}%"
              f"   still rising {rise[wi].mean():+.2f} pts"
              f"   [{time.perf_counter()-t0:5.0f}s]")

mean, sem = acc.mean(1), acc.std(1, ddof=1) / np.sqrt(SEEDS)

# ---------------------------------------------------------------- choose H
chosen = next((w for w, m in zip(WIDTHS, mean) if m >= mean.max() - TOLERANCE), WIDTHS[-1])
print(f"\n  H = {chosen}  (smallest width within {TOLERANCE:.1f} points of the best, "
      f"{mean.max():.1f}% at H={WIDTHS[int(np.argmax(mean))]})")
print("  gain from each doubling: " +
      ", ".join(f"{a}->{b}: {mean[i+1]-mean[i]:+.1f}" for i, (a, b) in
                enumerate(zip(WIDTHS[:-1], WIDTHS[1:]))))
if rise.mean(1).max() > RISE_TOL:
    w = WIDTHS[int(np.argmax(rise.mean(1)))]
    print(f"  WARNING: H={w} still gaining {rise.mean(1).max():+.2f} points — raise MAX_ITERS")
else:
    print(f"  converged: largest remaining gain {rise.mean(1).max():+.2f} points")

# ---------------------------------------------------------------- domain-IL, at the chosen H only
if not REPLOT:
    dom_proto = replace(base, hidden=chosen, scenario="domain_il")
    dom = np.array([joint(dom_proto, s, data)[1][-PLATEAU_EVALS:].mean() for s in range(SEEDS)])
print(f"\n  domain-IL at H={chosen}: {dom.mean():.1f} ± {dom.std(ddof=1)/np.sqrt(SEEDS):.1f}%"
      f"  (chance 20%, against 10% for class-IL — not directly comparable)")

# ---------------------------------------------------------------- main figure
fig, ax = plt.subplots(figsize=(7.2, 5))
ax.errorbar(WIDTHS, mean, yerr=sem, marker="o", capsize=3, lw=2,
            color="tab:purple", label="class-IL (10 outputs)")
ax.errorbar([chosen], [dom.mean()], yerr=[dom.std(ddof=1) / np.sqrt(SEEDS)], marker="D",
            capsize=3, ms=8, color="tab:green", ls="none",
            label=f"domain-IL (5 shared outputs), at H={chosen}")
ax.axhline(10, color="tab:purple", ls=":", lw=1, alpha=0.7)
ax.axhline(20, color="tab:green", ls=":", lw=1, alpha=0.7)
ax.text(WIDTHS[0], 11, "chance, class-IL", fontsize=7, color="tab:purple")
ax.text(WIDTHS[0], 21, "chance, domain-IL", fontsize=7, color="tab:green")
ax.axvline(chosen, color="k", ls="--", lw=1)
ax.annotate(f"H = {chosen}", xy=(chosen, 62), xytext=(8, 0), textcoords="offset points",
            fontsize=12, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(WIDTHS)
ax.set_xticklabels(WIDTHS)
ax.set_xlabel("hidden width H")
ax.set_ylabel("joint accuracy on all 10 classes (%)")
ax.set_ylim(0, 100)
ax.set_title("Is there room to hold both tasks?\n"
             f"joint training to convergence, {MAX_ITERS:,} updates, mean ± SEM over {SEEDS} seeds")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ------------------------------------------------- supporting figure: is the budget enough?
fig2, ax2 = plt.subplots(figsize=(7.2, 4.4))
cmap = plt.get_cmap("viridis")
for i, width in enumerate(WIDTHS):
    ax2.plot(steps, learning[width], lw=1.6,
             color=cmap(i / max(1, len(WIDTHS) - 1)), label=f"H={width}")
ax2.set_xlabel("training step")
ax2.set_ylabel("joint accuracy (%)")
ax2.set_ylim(0, 100)
ax2.set_title("Diagnostic: is the budget past convergence?\n"
              "curves must be flat at the right edge, or the sweep measures the budget")
ax2.legend(fontsize=7, ncol=2, loc="lower right")
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(figure_path(__file__, "convergence"), dpi=120, bbox_inches="tight")
print(f"saved {figure_path(__file__, 'convergence')}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__),
         widths=np.asarray(WIDTHS), steps=np.asarray(steps), chosen=chosen,
         acc_class_il=acc, rise=rise, acc_domain_il=dom, max_iters=MAX_ITERS,
         **{f"curve_{w}": learning[w] for w in WIDTHS})
print(f"saved {array_path(__file__)}")
