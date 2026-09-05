"""Slide asset: does depth rescue PC? One summary chart, all 8 cells, from 59 (already run)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.protocol import array_path, figure_path
from src.metrics import paired_diff

z = np.load(array_path(str(ROOT / "experiments" / "59_does_pc_help_in_a_bigger_network.py")),
            allow_pickle=True)
cells = z["cells"]
labels = [f"{d}×{w}" for d, w in cells]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(cells))
w = 0.35
for i, m in enumerate(["replay", "pc"]):
    ds, ses = [], []
    for d, wd in cells:
        key = f"xh_{d}x{wd}"
        diff, se, _ = paired_diff(z[f"{key}_{m}"], z[f"{key}_backprop"])
        ds.append(diff)
        ses.append(se)
    ax.bar(x + (i - 0.5) * w, ds, w, yerr=ses, capsize=3,
           color="tab:brown" if m == "replay" else "tab:red", label=m)

ax.axhline(0, color="k", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("depth × width")
ax.set_ylabel("crossover height, minus backprop")
ax.set_title("Does depth rescue PC? All 8 cells, Domain-IL.\nReplay separates in every cell. PC never does.")
ax.legend()
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=130, bbox_inches="tight")
print(f"saved {figure_path(__file__)}")
