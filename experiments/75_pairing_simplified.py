"""Slide asset: does class-pair similarity predict crossover height? One clean scatter per
scenario, from already-run data (60, 69) -- no retraining.

CORRECTED: the first version used sim_paired for Domain-IL (mean cosine of the 5 digit pairs
that share an output unit) but sim_cross for Class-IL (mean cosine over all 25 cross-task pairs)
-- two DIFFERENT quantities, so the two panels were not answering the same question. tasks(seed)
uses identical machinery in both scenarios: one permutation of the 10 digits, task 1 = the first
5, task 2 = the last 5, digit i of task 1 paired with digit i of task 2 by index. That pairing
exists in Class-IL too -- it just doesn't happen to share an output unit there. So the SAME
paired measure is recomputed for Class-IL here (tasks(seed) is a deterministic function of the
seed alone, independent of scenario -- no retraining needed, just re-drawing the same
permutations 69's run used, seeds 0..23 in order).

Simplified from 60/69's own figures per direct request: drop the third bar-chart panel (sem
ratio -- a different question, different units), drop "task 1 kept" in favour of crossover (the
metric used everywhere else in this deck), backprop and PC only (replay's crossover is undefined
on many seeds -- censored, not missing -- which is a real result but not one this slide needs to
explain).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, replace, array_path, figure_path
from src.metrics import metric_grid

METHODS = ["backprop", "pc"]
COLORS = {"backprop": "tab:gray", "pc": "tab:red"}
E = str(ROOT / "experiments") + "\\"
N_SEEDS_CLASS_IL = 24     # 69 ran seeds 0..23, in order -- matches the npz row order


def class_means(train):
    """Identical to 60's -- global-mean-centred class mean image per digit, flattened."""
    x = train.x.reshape(len(train.x), -1).float()
    y = train.targets
    raw = np.stack([x[y == c].mean(0).numpy() for c in range(10)])
    return raw - raw.mean(0, keepdims=True)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


z60 = np.load(E + "60_why_does_the_seed_matter_so_much.npz", allow_pickle=True)
z69 = np.load(E + "69_does_the_data_split_confound_the_comparison.npz", allow_pickle=True)

sw60 = float(np.asarray(z60["switches"]).ravel()[0])
g60 = {m: metric_grid(z60["steps"], z60[f"argmax_{m}"], sw60) for m in METHODS}
g69 = {m: metric_grid(z69["steps"], z69[f"argmax_{m}"], float(z69["switch"])) for m in METHODS}

# Reconstruct the SAME paired-index similarity for Class-IL, using the exact seeds 69 used.
base = replace(PROTOCOL, scenario="class_il")   # tasks() does not depend on hidden/threshold/etc
data = load(base)
cen = class_means(data.train)
sim_paired_class = np.array([
    np.mean([cos(cen[t1], cen[t2]) for t1, t2 in zip(*base.tasks(seed))])
    for seed in range(N_SEEDS_CLASS_IL)
])

PANELS = [("Domain IL", z60["sim_paired"], g60), ("Class IL", sim_paired_class, g69)]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
for ax, (title, sim, g) in zip(axes, PANELS):
    for m in METHODS:
        v = g[m]["crossover"]
        ok = np.isfinite(v)
        ax.plot(sim[ok], v[ok], "o", ms=6, color=COLORS[m], label=m, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("class-pair similarity")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("crossover (%)")
axes[0].legend(fontsize=9)
fig.suptitle("Does which digits are paired predict crossover height?", fontsize=11)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"saved {figure_path(__file__)}")
