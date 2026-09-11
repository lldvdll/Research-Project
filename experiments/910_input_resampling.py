"""What does downsampling MNIST to 14x14 actually cost the input?

900-series PLOT SCRIPT. Loads MNIST, trains nothing. Methods M2, supporting the data row of
the control-parameter table.

WHY THE FIGURE EXISTS. The report runs at 14x14 rather than the usual 28x28, and that is a
choice a reader is entitled to challenge: a quarter of the pixels is a real reduction, and if
it destroyed class structure then every downstream claim would be made on a degraded problem.
The honest answer is visual -- the digits remain plainly legible -- so the figure simply shows
the same images at both resolutions and lets the reader judge. It asserts nothing about
accuracy; 810 and 811 carry that.

ONE ROW PER RESOLUTION, ONE COLUMN PER CLASS, so the comparison is within-column and the eye
does not have to hold two grids apart. The first test-set example of each digit is used --
not a hand-picked one, and not an average, which would hide exactly the fine strokes the
downsampling might be expected to lose.

MINIMAL FURNITURE, per the report's figure convention: no title, no axes, no per-panel text.
The two row labels are the only words in the figure.

PROVENANCE
    MNIST test split via src.data.load_mnist, at size=28 and size=14. The 14x14 path is the
    same call the whole project trains on, so this shows the actual input, not a re-derivation
    of it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.data import load_mnist
from src.protocol import figure_path, DATA_ROOT

SIZES = [28, 14]
CLASSES = list(range(10))


def first_example_per_class(size):
    """The first test image of each class, at `size`. Same images at both resolutions, because
    the test split order does not depend on the resampling."""
    _, test = load_mnist(size=size, root=DATA_ROOT, fashion=False)
    # TensorMNIST holds the split as tensors: .x is (N, size*size), .targets is (N,)
    x = test.x.detach().cpu().numpy()
    y = test.targets.detach().cpu().numpy().reshape(-1)
    out = []
    for c in CLASSES:
        idx = int(np.flatnonzero(y == c)[0])
        out.append(x[idx].reshape(size, size))
    return out


if __name__ == "__main__":
    rows = {s: first_example_per_class(s) for s in SIZES}

    # Height is set from the WIDTH so the cells stay square and no dead band opens between
    # the rows: ten columns across 6.6in is 0.66in a cell, so two rows want ~1.4in plus labels.
    cell = 6.6 / len(CLASSES)
    fig, axes = plt.subplots(len(SIZES), len(CLASSES),
                             figsize=(6.6, cell * len(SIZES) + 0.12))
    for r, size in enumerate(SIZES):
        for c in range(len(CLASSES)):
            ax = axes[r][c]
            ax.imshow(rows[size][c], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color("0.75")
        axes[r][0].set_ylabel(f"{size}x{size}", fontsize=9, rotation=90, va="center")

    fig.subplots_adjust(wspace=0.06, hspace=0.06, left=0.055, right=0.995,
                        top=0.995, bottom=0.005)
    out = figure_path(__file__)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")
    for s in SIZES:
        print(f"  {s}x{s}: {len(rows[s])} classes, {s * s} input features")
