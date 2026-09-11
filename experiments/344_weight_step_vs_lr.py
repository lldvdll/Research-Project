"""Does PC's realised per-step weight movement scale sub-linearly with nominal lr relative to
backprop's (a direct mechanistic test of "PC is more robust to lr overshoot"), and is that
confined to W1 -- the hidden-layer weights, whose error signal comes through settling -- while W2
-- the output weights, which use the same direct target-error signal for both rules -- scales the
same way for both, as a built-in control?

Report series 2, decade 344. Motivated by 343: the accuracy-based evidence that PC degrades more
gently than backprop past their shared lr optimum is real in some cells (class-il both metrics,
domain-il crossover) but not others (domain-il retention actually degrades FASTER for pc) -- a
mixed, metric/scenario-confounded picture. This tests the underlying MECHANISM directly, on the
update rule itself, independent of accuracy, forgetting, or crossover's own measurement-resolution
problems (which 343 showed are real at high lr).

DESIGN: fixed-step JOINT training (no continual learning, no early stopping -- a stated deviation
from src.protocol.run/run_classil, needed because early stopping would let different lr/method
combinations see different numbers of updates, contaminating the per-step measurement) on all 10
Class-IL classes at once. N=300 steps, identical across every (method, lr, seed) cell. Same seed
gives both rules an IDENTICAL initial W1/W2 (src.model.init_params keys off seed, not method), so
the only thing that can differ is what each rule's update does to those same starting weights.

MEASUREMENT: src.probes.weight_path_probe (already built, used by scripts 47-49/65 for synaptic
path efficiency) accumulates sum|dw| per synapse over the run. Divided by N, that is each
synapse's mean per-step movement; averaged over the layer, one scalar per layer per run. Fit
log(mean per-step movement) against log(lr): a rule applying update = lr * (error signal) with an
error signal that does not itself depend on lr should show a slope of ~1 (directly proportional).
A slope below 1 means the realised step shrinks relative to what the nominal lr alone would
predict -- the direct signature of a damping mechanism.

lr grid matches 340/343 ([0.005 .. 0.16]) for direct comparability. PC's settle control fixed at
its settled value (dt=0.4, adaptive stop) -- not re-swept here, already established (330-334).
5 seeds (10-14) -- this is a mechanism check, not a headline comparison; cheap to extend if needed.

One figure: 344_weight_step_vs_lr.png, 2 columns (W1, W2) -- mean per-step |dW| vs lr, log-log,
backprop and pc, with fitted slopes annotated.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, replace, load, build, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import _loader
from src.probes import weight_path_probe

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

OPTIMAL_DT = 0.4
SETTLE_STEP_CAP = 100
SETTLE_DELTA_TOL = 1e-4
SETTLE_PATIENCE = 3

METHODS = ["backprop", "pc"]

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, SEED_START = 2, 10
    LR_GRID = [0.01, 0.02]
    N_STEPS = 20
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = 5
    SEED_START = CFG["training"]["seed_start"]
    LR_GRID = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
    N_STEPS = 300

BASE_PROTOCOL = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    scenario="class_il",   # out_dim=10, plain 10-way -- no label remap needed for joint training
    hidden=CFG["architecture"]["hidden"], n_layers=CFG["architecture"]["n_layers"],
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    device=CFG["evaluation"]["device"],
)


def figure_path():
    return _figure_path(__file__, "SMOKE" if SMOKE else "")


def array_path():
    return _array_path(__file__, "SMOKE" if SMOKE else "")


def _row_for(proto, method, lr, seed, data):
    """(mean_per_step_W1, mean_per_step_W2) -- accumulated path length / N_STEPS, averaged over
    each layer's synapses."""
    handle = {}
    kw = dict(lr=lr)
    if method == "pc":
        kw.update(dt=OPTIMAL_DT, steps=SETTLE_STEP_CAP,
                 stop_delta=SETTLE_DELTA_TOL, stop_patience=SETTLE_PATIENCE)
    train_step, predict = build(proto, method, seed, handle=handle, **kw)

    wrapped, path = weight_path_probe(train_step, handle["params"])

    classes = list(range(proto.n_classes))
    loader = _loader(data.train, data.class_idx, classes, proto.batch, seed=seed)
    it = iter(loader)
    for _ in range(N_STEPS):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        wrapped(x.to(proto.device), y.to(proto.device), active=classes)

    w1 = float((path["W1"] / N_STEPS).mean())
    w2 = float((path["W2"] / N_STEPS).mean())
    return w1, w2


if REPLOT and Path(array_path()).exists():
    rows = list(np.load(array_path(), allow_pickle=True)["data"])
    print("--replot: redrawing from saved arrays, no training\n")
else:
    rows = []
    data = load(BASE_PROTOCOL)   # single load, reused across every cell -- dataset doesn't vary
    for method in METHODS:
        for lr in LR_GRID:
            for seed in range(SEED_START, SEED_START + SEEDS):
                w1, w2 = _row_for(BASE_PROTOCOL, method, lr, seed, data)
                rows.append((method, lr, seed, w1, w2))
        print(f"  {method:9s} done")
    np.savez(array_path(), data=np.array(rows, dtype=object))
    print("saved arrays")


def _vals(col, method, lr):
    return np.array([r[col] for r in rows if r[0] == method and r[1] == lr], dtype=float)


METHOD_COLOR = {"backprop": "0.35", "pc": "tab:orange"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
print("\nfitted slope: log(mean per-step |dW|) vs log(lr) -- ~1.0 = proportional to lr, "
     "<1.0 = damped relative to lr")
for ax, (name, col) in zip(axes, [("W1 (hidden -- settled error, pc only)", 3),
                                  ("W2 (output -- direct error, both rules)", 4)]):
    print(f"\n  {name}")
    for method in METHODS:
        means = np.array([_vals(col, method, lr).mean() for lr in LR_GRID])
        sems = np.array([_vals(col, method, lr).std(ddof=1) / np.sqrt(_vals(col, method, lr).size)
                        if _vals(col, method, lr).size > 1 else 0.0 for lr in LR_GRID])
        ax.errorbar(LR_GRID, means, yerr=sems, marker="o", color=METHOD_COLOR[method],
                   capsize=3, label=method)
        slope, intercept = np.polyfit(np.log(LR_GRID), np.log(means), 1)
        print(f"    {method:9s} slope = {slope:+5.2f}")
        ax.annotate(f"slope={slope:.2f}", (LR_GRID[-1], means[-1]), fontsize=8,
                   color=METHOD_COLOR[method], xytext=(5, 0), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(LR_GRID); ax.set_xticklabels([str(v) for v in LR_GRID], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("learning rate")
    ax.set_ylabel("mean per-step |dW| (log scale)")
    ax.set_title(name, fontsize=9)
    ax.grid(alpha=0.2, which="both")
axes[0].legend(fontsize=8)

fig.tight_layout()
fig.savefig(figure_path(), dpi=130, bbox_inches="tight")
print(f"\nsaved {figure_path()}")
