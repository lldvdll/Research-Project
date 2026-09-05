"""Does the network have the capacity to hold both tasks at once, and at what width?

Fresh run under the report-series-2 protocol (see `claude_code_management/now.md`): 10 seeds
(was 5), width sweep capped at 128 -- the plateau already found at H=32 makes 256 pointless to
recheck -- backprop only (capacity is a property of the architecture, not the rule), no plot
titles, one figure with both scenarios overlaid.

Method unchanged from the original capacity script (old 41): joint training -- all ten classes
in a single block, no task sequence -- run to measured convergence rather than a fixed budget,
because a short budget produces a flat region indistinguishable from a real capacity ceiling.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, build, replace, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import run_classil

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv


def _tag(f, suffix):
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):
    return _array_path(f, _tag(f, suffix))


# ---------------------------------------------------------------- settings
WIDTHS = [2, 4, 8, 16, 32, 64, 128]
MAX_ITERS = 25000       # from the original convergence probe: the largest width needs ~22k
EVAL_EVERY = 250
PLATEAU_EVALS = 6        # average the last few evals rather than taking a lucky maximum
SEEDS = 10
TOLERANCE = 1.0          # percentage points from the best width
RISE_TOL = 0.5           # a curve still gaining more than this over the last quarter is not converged
SCENARIOS = ["class_il", "domain_il"]
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
CHANCE = {"class_il": 10, "domain_il": 20}

if SMOKE:
    WIDTHS, SEEDS, MAX_ITERS, EVAL_EVERY = [2, 32], 2, 400, 50
    print("--smoke: tiny budget, results are NOT meaningful\n")

base = replace(PROTOCOL,
               stop_threshold=None,                # no early stopping: we want the ceiling
               max_iters_per_task=MAX_ITERS,
               eval_every=EVAL_EVERY,
               seeds=SEEDS,
               lr={"backprop": 0.01})              # BUG FIX: was missing, silently fell back to
                                                    # methods.METHOD_DEFAULTS["backprop"] = 0.05


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


if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__))
    acc = {s: z[f"acc_{s}"] for s in SCENARIOS}
    rise = {s: z[f"rise_{s}"] for s in SCENARIOS}
    learning = {s: {w: z[f"curve_{s}_{w}"] for w in WIDTHS} for s in SCENARIOS}
    steps = z["steps"]
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
                rise[scen][wi, seed] = c[-k:].mean() - c[-3 * k:-2 * k].mean()
            learning[scen][width] = np.mean(curves, axis=0)
            print(f"  {scen:10s} H={width:4d}  {acc[scen][wi].mean():5.1f} +- "
                  f"{acc[scen][wi].std(ddof=1)/np.sqrt(SEEDS):.1f}%"
                  f"   still rising {rise[scen][wi].mean():+.2f} pts"
                  f"   [{time.perf_counter()-t0:5.0f}s]")

mean = {s: acc[s].mean(1) for s in SCENARIOS}
sem = {s: acc[s].std(1, ddof=1) / np.sqrt(SEEDS) for s in SCENARIOS}

# ---------------------------------------------------------------- choose H
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

# ---------------------------------------------------------------- main figure: both scenarios, one axis
fig, ax = plt.subplots(figsize=(7.2, 5))
for s in SCENARIOS:
    ax.errorbar(WIDTHS, mean[s], yerr=sem[s], marker="o", capsize=3, lw=2, color=COLORS[s],
                label=s.replace("_", "-"))
    ax.axhline(CHANCE[s], color=COLORS[s], ls=":", lw=1, alpha=0.6)
ax.axvline(chosen, color="k", ls="--", lw=1)
ax.annotate(f"H = {chosen}", xy=(chosen, 55), xytext=(8, 0), textcoords="offset points",
            fontsize=11, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(WIDTHS); ax.set_xticklabels(WIDTHS)
ax.set_xlabel("hidden units")
ax.set_ylabel("accuracy (%)")
ax.set_ylim(0, 100)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ------------------------------------------------- diagnostic: is the budget enough? (per scenario, no titles)
cmap = plt.get_cmap("viridis")
for s in SCENARIOS:
    fig2, ax2 = plt.subplots(figsize=(6, 4.2))
    for i, width in enumerate(WIDTHS):
        ax2.plot(steps, learning[s][width], lw=1.6, color=cmap(i / max(1, len(WIDTHS) - 1)),
                 label=f"{width}")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7, ncol=2, loc="lower right", title="hidden units")
    fig2.tight_layout()
    fig2.savefig(figure_path(__file__, f"convergence_{s}"), dpi=120, bbox_inches="tight")
    print(f"saved {figure_path(__file__, f'convergence_{s}')}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__),
         widths=np.asarray(WIDTHS), steps=np.asarray(steps), chosen=chosen, max_iters=MAX_ITERS,
         **{f"acc_{s}": acc[s] for s in SCENARIOS},
         **{f"rise_{s}": rise[s] for s in SCENARIOS},
         **{f"curve_{s}_{w}": learning[s][w] for s in SCENARIOS for w in WIDTHS})
print(f"saved {array_path(__file__)}")
