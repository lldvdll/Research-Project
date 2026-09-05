"""Does a seed's task-pair digit similarity predict how much task-1 accuracy survives, and is
the relevant pairing Domain-IL's shared-output-unit pairs or Class-IL's overall task overlap?

Report series 2, decade 311. Identical to 301 except ONE stated deviation: reads 310's saved
runs (seeds 10-19) instead of 300's (seeds 0-9) -- pure re-analysis, no new training. Checks
whether 301's findings (Domain-IL: real, robust; Class-IL: no reliable relationship) replicate
on the independent seed block.

Domain-IL's mechanism is SPECIFIC: label_map makes digit task1[i] and digit task2[i] share one
output unit (i = 0..4), so only those 5 index-matched pairs can reinforce or overwrite each
other. Class-IL has no shared units at all -- if pairing matters there, it would have to be
through a different, more speculative route (representational overlap in the trunk), so the
similarity measure used per scenario differs on purpose:

    Domain-IL : mean similarity of the 5 SHARED-UNIT pairs  (task1[i], task2[i])
    Class-IL  : mean similarity over ALL 25 cross pairs      (task1[i], task2[j])

Similarity = cosine similarity between per-class mean images (14x14, training set) -- a cheap,
fixed property of the 10 MNIST digits, independent of scenario, rule or seed.

One figure, two panels (one per scenario): x = that scenario's similarity measure, y = final
task-1 accuracy from 310's saved arrays.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, replace, figure_path as _figure_path, array_path as _array_path
from src.data import load_mnist

SCENARIOS = ["class_il", "domain_il"]
SEEDS = 10
SEED_START = 10                       # the one stated deviation from 301
OUTLIER_SEED = 9                      # array index (not absolute seed) -- see 301
MARKER_COLOR = {"class_il": "tab:blue", "domain_il": "tab:green"}

# ---------------------------------------------------------------- 310's saved arrays (required)
_310 = {}
for s in SCENARIOS:
    p = Path(__file__).parent / f"310_forgetting_by_scenario_{s}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found -- run 310_forgetting_by_scenario.py first")
    _310[s] = np.load(p, allow_pickle=True)

# ---------------------------------------------------------------- fixed digit-pair similarity matrix
train, _ = load_mnist(size=14, root=str(ROOT / "data"))
x = train.x.reshape(len(train), -1)
y = train.targets
class_means = np.stack([x[y == c].mean(dim=0).numpy() for c in range(10)])   # [10, 196]
norm = class_means / np.linalg.norm(class_means, axis=1, keepdims=True)
SIM = norm @ norm.T                                                          # [10, 10] cosine similarity

# ---------------------------------------------------------------- per-seed task split (scenario-independent)
proto = replace(PROTOCOL, hidden=32)   # only .tasks()/.n_classes used here -- hidden is irrelevant
splits = [proto.tasks(seed) for seed in range(SEED_START, SEED_START + SEEDS)]

# Shared-unit (5 index-matched pairs) similarity -- a property of the DIGIT SPLIT alone, which
# is identical across scenarios (Protocol.tasks(seed) never reads scenario), so this single array
# is the correct x-axis for EITHER scenario's y-values, not just domain_il's.
shared_sim = np.array([np.mean([SIM[t1[i], t2[i]] for i in range(5)]) for t1, t2 in splits])
domain_sim = shared_sim
classil_sim = np.array([np.mean([[SIM[a, b] for b in t2] for a in t1])
                        for t1, t2 in splits])

domain_finals = _310["domain_il"]["finals"]
classil_finals = _310["class_il"]["finals"]
domain_crossovers = _310["domain_il"]["crossovers"]
classil_crossovers = _310["class_il"]["crossovers"]

# ---------------------------------------------------------------- figure: similarity vs final task-1 accuracy
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

for ax, scen, sim, finals, xlabel in [
    (axes[0], "domain_il", domain_sim, domain_finals, "shared-unit pair similarity (5 index-matched pairs)"),
    (axes[1], "class_il", classil_sim, classil_finals, "overall task-pair similarity (all 25 cross pairs)"),
]:
    ax.scatter(sim, finals, color=MARKER_COLOR[scen], s=40, zorder=3)
    ax.scatter(sim[OUTLIER_SEED], finals[OUTLIER_SEED], facecolors="none",
              edgecolors="black", s=90, linewidths=1.5, zorder=4)
    for i in range(SEEDS):
        ax.annotate(str(i), (sim[i], finals[i]), fontsize=7, xytext=(3, 3),
                   textcoords="offset points")
    r = np.corrcoef(sim, finals)[0, 1]
    ax.text(0.03, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("final task-1 accuracy (%)")
    ax.text(0.5, 1.03, scen.replace("_", "-"), fontsize=9, ha="center", transform=ax.transAxes)
    ax.grid(alpha=0.2)

fig.tight_layout()
out = _figure_path(__file__)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")

np.savez(_array_path(__file__), sim_matrix=SIM, domain_sim=domain_sim, classil_sim=classil_sim,
        domain_finals=domain_finals, classil_finals=classil_finals)
print(f"saved {_array_path(__file__)}")


# ---------------------------------------------------------------- combined-axes comparison + crossover + reliability
# Both scenarios plotted on the SAME shared_sim x-axis, so the shape of the trend (not just its
# r) is directly comparable. Each panel also reports r with seed 9 excluded: a correlation driven
# by one leverage point (one seed far out on both axes, the rest clustered) should collapse
# toward 0 once that point is dropped; a real distributed trend should survive losing one seed.
def r_with_without(sim, val, exclude=OUTLIER_SEED):
    r_all = np.corrcoef(sim, val)[0, 1]
    mask = np.arange(len(sim)) != exclude
    r_excl = np.corrcoef(sim[mask], val[mask])[0, 1]
    return r_all, r_excl


fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, val_domain, val_classil, ylabel in [
    (axes2[0], domain_finals, classil_finals, "final task-1 accuracy (%)"),
    (axes2[1], domain_crossovers, classil_crossovers, "crossover accuracy (%)"),
]:
    ax.scatter(shared_sim, val_domain, color=MARKER_COLOR["domain_il"], s=40,
              label="domain-il", zorder=3)
    ax.scatter(shared_sim, val_classil, color=MARKER_COLOR["class_il"], s=40,
              label="class-il", zorder=3)
    ax.scatter(shared_sim[OUTLIER_SEED], val_domain[OUTLIER_SEED], facecolors="none",
              edgecolors="black", s=90, linewidths=1.5, zorder=4)
    ax.scatter(shared_sim[OUTLIER_SEED], val_classil[OUTLIER_SEED], facecolors="none",
              edgecolors="black", s=90, linewidths=1.5, zorder=4)
    r_d_all, r_d_excl = r_with_without(shared_sim, val_domain)
    r_c_all, r_c_excl = r_with_without(shared_sim, val_classil)
    print(f"{ylabel}: domain-il r={r_d_all:.2f} (excl. seed9: {r_d_excl:.2f})   "
         f"class-il r={r_c_all:.2f} (excl. seed9: {r_c_excl:.2f})")
    ax.text(0.03, 0.95, f"domain-il r = {r_d_all:.2f}  (excl. seed9: {r_d_excl:.2f})",
           transform=ax.transAxes, fontsize=8, va="top", color=MARKER_COLOR["domain_il"])
    ax.text(0.03, 0.88, f"class-il  r = {r_c_all:.2f}  (excl. seed9: {r_c_excl:.2f})",
           transform=ax.transAxes, fontsize=8, va="top", color=MARKER_COLOR["class_il"])
    ax.set_xlabel("shared-unit-style pair similarity (5 index-matched pairs)", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
axes2[0].legend(fontsize=8, loc="lower right")
fig2.tight_layout()
out2 = _figure_path(__file__, "combined")
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"saved {out2}")
