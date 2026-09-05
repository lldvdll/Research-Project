"""Class-IL finals split roughly into two groups (~0% and ~15-30%) -- what distinguishes the
seeds in each group?

Report series 2, decade 312. Pure re-analysis of 300's (seeds 0-9) and 310's (seeds 10-19)
saved Class-IL arrays -- no new training. One figure: all 20 seeds as rows, one column per
digit (0-9) coloured by which task it was assigned to that seed, plus retention/crossover/
runtime columns, ordered by runtime (task-1 phase length, switch0) so any pattern is visible
at a glance.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.protocol import PROTOCOL, replace

TASK1_COLOR = "tab:orange"
TASK2_COLOR = "tab:blue"

proto = replace(PROTOCOL, hidden=32)

seeds, finals, crossovers, runtimes = [], [], [], []
for prefix, seed_range in [("300", range(0, 10)), ("310", range(10, 20))]:
    z = np.load(Path(__file__).parent / f"{prefix}_forgetting_by_scenario_class_il.npz",
               allow_pickle=True)
    for i, seed in enumerate(seed_range):
        seeds.append(seed)
        finals.append(z["finals"][i])
        crossovers.append(z["crossovers"][i])
        runtimes.append(z["switch0"][i])

seeds = np.array(seeds)
finals = np.array(finals)
crossovers = np.array(crossovers)
runtimes = np.array(runtimes)

order = np.argsort(runtimes)
seeds, finals, crossovers, runtimes = seeds[order], finals[order], crossovers[order], runtimes[order]

task_grid = np.zeros((len(seeds), 10), dtype=int)
for row, seed in enumerate(seeds):
    t1, t2 = proto.tasks(int(seed))
    for d in t2:
        task_grid[row, d] = 1   # 0 = task-1 (orange), 1 = task-2 (blue)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(task_grid, cmap=ListedColormap([TASK1_COLOR, TASK2_COLOR]), aspect="auto",
         extent=(-0.5, 9.5, len(seeds) - 0.5, -0.5))

for row in range(len(seeds)):
    ax.text(10.3, row, f"{finals[row]:5.1f}%", va="center", fontsize=8, family="monospace")
    ax.text(12.2, row, f"{crossovers[row]:5.1f}%", va="center", fontsize=8, family="monospace")
    ax.text(14.3, row, f"{int(runtimes[row]):5d}", va="center", fontsize=8, family="monospace")

ax.set_xticks(range(10))
ax.set_xticklabels(range(10))
ax.set_yticks(range(len(seeds)))
ax.set_yticklabels([f"seed {s}" for s in seeds])
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(len(seeds) - 0.5, -0.5)

for x, label in [(10.3, "retention"), (12.2, "crossover"), (14.3, "runtime")]:
    ax.text(x, -1.0, label, fontsize=8, ha="left", rotation=30)

ax.set_xlabel("digit (orange = task-1, blue = task-2)")
ax.set_title("Class-IL, all 20 seeds, ordered by runtime (task-1 phase length)", fontsize=10)

fig.tight_layout()
out = str(Path(__file__).parent / "312_class_il_digit_table.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
