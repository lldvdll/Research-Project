"""Is the joint problem hard enough that training W1 matters, or would a frozen random
projection plus a trained linear head solve it just as well?

Fresh run under the report-series-2 protocol. Freezes W1 (and b1) at their random draw --
`scaled_normal`, the same init distribution real training starts from -- and trains only the
output layer (W2, b2) by backprop, at the width exp100 selects. 5 data-split seeds x 5 W1-init
seeds (25 runs/scenario, 50 total), joint training as in exp100 (all ten classes, one block, no
early stopping -- we want the ceiling, not a threshold crossing). The 25-run distribution is
compared against exp100's fully-trained accuracy at the same width: if the random-projection
ceiling sits close to the trained ceiling, W1 was not doing much work, and later analyses that
locate forgetting in W1 need to be read with that in mind.

Why the SAME init distribution for all 5 draws, not a scale sweep: the question is "does training
W1 matter, starting from where it actually starts" -- so the frozen baseline has to start from
that same distribution, not an invented one. A scale sweep answers a different question (does
random-feature scale matter) and adds a parameter this experiment does not need.

Boundary check, stated before running: `scaled_normal` scales by 1/sqrt(fan_in) specifically to
keep a layer's pre-activation variance ~O(1) regardless of width -- neither collapsed near zero
(the degenerate regime that would make this test meaningless) nor saturating tanh. That is
asserted here, not just cited: the realised pre-activation std and tanh-saturation fraction, on
real training images, under the actual frozen draws used, are printed and saved alongside the
result.
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
from src.model import hidden_pre

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
DATA_SEEDS = list(range(5))
INIT_SEEDS = list(range(5))
MAX_ITERS = 25000
EVAL_EVERY = 250
PLATEAU_EVALS = 6
RISE_TOL = 0.5
SCENARIOS = ["class_il", "domain_il"]
COLORS = {"class_il": "tab:purple", "domain_il": "tab:green"}
EXP100 = Path(__file__).parent / "100_capacity_vs_width.py"

if SMOKE:
    DATA_SEEDS, INIT_SEEDS, MAX_ITERS, EVAL_EVERY = [0, 1], [0, 1], 400, 50
    print("--smoke: tiny budget, results are NOT meaningful\n")

# H comes from exp100; fall back to 32 (the expected answer) if 100 hasn't been run yet.
if Path(_array_path(EXP100)).exists():
    H = int(np.load(_array_path(EXP100))["chosen"])
    print(f"H = {H}, read from exp100")
else:
    H = 32
    print("exp100's array not found -- falling back to H = 32; re-run after 100 completes")

base = replace(PROTOCOL, hidden=H, stop_threshold=None,   # no early stopping: we want the ceiling
               max_iters_per_task=MAX_ITERS, eval_every=EVAL_EVERY,
               lr={"backprop": 0.01})   # BUG FIX: was missing, silently used the raw default 0.05


def frozen_w1_run(proto, data_seed, init_seed, data):
    """Freeze W1, b1 at their random draw; train only the output layer, joint (all ten classes)."""
    pair = proto.tasks(data_seed)
    handle = {}
    train_step, predict = build(proto, "backprop", init_seed, handle=handle)
    handle["freeze"].add("W1")
    handle["freeze"].add("b1")
    out = run_classil(
        train_step, predict, [pair[0] + pair[1]], data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=MAX_ITERS, batch=proto.batch, eval_every=EVAL_EVERY,
        device=proto.device, data_seed=data_seed, label_map=proto.label_map(pair),
    )
    acc = out["curves"]["argmax"][:, 0] * 100.0
    k = PLATEAU_EVALS
    final = float(acc[-k:].mean())
    rise = float(acc[-k:].mean() - acc[-3 * k:-2 * k].mean())

    p, arch = handle["params"], handle["arch"]
    xb = data.train.x[:512].reshape(512, -1).float()
    pre = hidden_pre(xb, p, arch).detach()
    pre_std = float(pre.std())
    sat_frac = float((pre.tanh().abs() > 0.99).float().mean())
    return final, rise, pre_std, sat_frac


if REPLOT and Path(array_path(__file__)).exists():
    z = np.load(array_path(__file__))
    final = {s: z[f"final_{s}"] for s in SCENARIOS}
    pre_std = {s: z[f"pre_std_{s}"] for s in SCENARIOS}
    sat_frac = {s: z[f"sat_frac_{s}"] for s in SCENARIOS}
    print(f"--replot: redrawing from {array_path(__file__)}, no training\n")
else:
    data = load(base)
    final = {s: np.zeros((len(DATA_SEEDS), len(INIT_SEEDS))) for s in SCENARIOS}
    pre_std = {s: np.zeros((len(DATA_SEEDS), len(INIT_SEEDS))) for s in SCENARIOS}
    sat_frac = {s: np.zeros((len(DATA_SEEDS), len(INIT_SEEDS))) for s in SCENARIOS}
    t0 = time.perf_counter()

    for scen in SCENARIOS:
        proto = replace(base, scenario=scen)
        for di, ds in enumerate(DATA_SEEDS):
            for ii, iseed in enumerate(INIT_SEEDS):
                f, r, ps, sf = frozen_w1_run(proto, ds, iseed, data)
                final[scen][di, ii] = f
                pre_std[scen][di, ii] = ps
                sat_frac[scen][di, ii] = sf
                if r > RISE_TOL:
                    print(f"  WARNING: {scen} data={ds} init={iseed} still rising {r:+.2f}pts")
        print(f"  {scen:10s} frozen-W1 final: {final[scen].mean():5.1f} +- "
              f"{final[scen].std(ddof=1)/np.sqrt(final[scen].size):.1f}%"
              f"   pre-act std {pre_std[scen].mean():.2f}, saturated {sat_frac[scen].mean()*100:.1f}%"
              f"   [{time.perf_counter()-t0:5.0f}s]")

# ---------------------------------------------------------------- exp100's trained ceiling, for the vertical lines
if Path(_array_path(EXP100)).exists():
    z100 = np.load(_array_path(EXP100))
    widths100 = z100["widths"].tolist()
    wi = widths100.index(H)
    trained = {s: float(z100[f"acc_{s}"][wi].mean()) for s in SCENARIOS}
else:
    trained = None
    print("exp100's array not found -- vertical lines omitted; re-run after 100 completes")

# ---------------------------------------------------------------- figure: one thin panel per scenario, filled, bin width 2
all_vals = np.concatenate(
    [final[s].ravel() for s in SCENARIOS] + ([np.asarray(list(trained.values()))] if trained else []))
lo = float(np.floor(np.nanmin(all_vals) / 2) * 2 - 2)
hi = float(np.ceil(np.nanmax(all_vals) / 2) * 2 + 2)
bins = np.arange(lo, hi + 2, 2)

fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(7.2, 4.2), sharex=True)
for ax, s in zip(axes, SCENARIOS):
    v = final[s].ravel()
    ax.hist(v, bins=bins, color=COLORS[s], alpha=0.85, label=s.replace("_", "-"))
    if trained is not None:
        ax.axvline(trained[s], color="k", ls="--", lw=1.8)
    ax.set_ylabel("count", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
axes[0].set_xlim(lo, hi)
axes[-1].set_xlabel("accuracy (%)")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# ---------------------------------------------------------------- save
np.savez(array_path(__file__), H=H, data_seeds=np.asarray(DATA_SEEDS),
         init_seeds=np.asarray(INIT_SEEDS),
         **{f"final_{s}": final[s] for s in SCENARIOS},
         **{f"pre_std_{s}": pre_std[s] for s in SCENARIOS},
         **{f"sat_frac_{s}": sat_frac[s] for s in SCENARIOS})
print(f"saved {array_path(__file__)}")
