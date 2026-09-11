"""Are the weights converging, orbiting, or drifting under repeated alternation?

900-series PLOT SCRIPT. Loads saved arrays, trains nothing. Results R4.

ONE PLOT, form from 007's SK_PCA: the weight trajectory projected onto its own first two
principal components, W1 and W2 SEPARATELY, both scenarios AND BOTH RULES (2 layers x 4
scenario-rule columns). Originally backprop-only ("912 establishes the rules barely differ, one
is drawn for legibility") -- that assumption doesn't hold up as well as it did: 802/808 found a
real, large, activation-dependent PC-vs-backprop effect in Domain-IL, and 923 made the same
backprop-only call for the same reason. Showing both rules here costs four more panels and
answers a question this project can no longer assume the answer to: does PC's weight trajectory
contract the same way backprop's does, per layer, per scenario.

WHY THIS IS THE STRONGER VERSION OF 917. Accuracy can look settled while the parameters wander --
two different weight vectors can score the same on both tasks. 917 shows the accuracy trajectory
spiralling inward; this asks whether the WEIGHTS do, and it separates the trunk from the readout,
which is the question R5 then takes up.

THE SCALAR THAT STOPS IT BEING A PRETTY PICTURE. Per-block CONTRACTION is measured directly: the
distance in PCA space between the state at the end of block k and at the end of block k-2 -- the
same phase of the previous cycle. A closed loop returns to where it was, so that distance stays
FLAT. A converging system returns nearer each time, so it FALLS. The figure prints the ratio of
the last such distance to the first.

READ THE TWO LAYERS SEPARATELY. The question 007 poses is whether the trunk contracts while the
readout keeps orbiting. Drawing W1 and W2 on one axis would make that unanswerable, since they
have different dimensions (196x32 against 32x10) and different natural scales.

⚠ PCA IS FITTED PER RUN, so the axes are NOT comparable between panels -- each is that run's own
basis. Only the SHAPE within a panel and the contraction number are meaningful. Saying so matters:
a reader who compares PC1 across panels is comparing two different directions.

ORIENTED AND SHARED-SCALE, so panels ARE now visually comparable despite the point above. PCA
sign is arbitrary -- flipping an axis is a valid transform of the same result, since explained
variance depends only on squared projections and is unchanged by a sign flip -- so each panel is
oriented by two concrete, checkable criteria rather than a vague "up and to the right": PC2 is
flipped, if needed, so the trajectory travels UP in PC2 over the course of block 1; PC1 is
flipped, if needed, so the trajectory ends further right in PC1 than where it started (P1 at the
end of the last block > P1 at the very start). Both are REFLECTIONS, not a change to the
trajectory's actual shape, so this does not manufacture the resemblance or difference between
panels; it only removes the arbitrary sign so a real resemblance or difference is visible instead
of hidden by coincidental axis orientation. Axis
LIMITS are then fixed per LAYER ROW (shared across all 4 scenario/rule columns within W1, and
separately within W2) so within-layer panels sit on an identical grid -- not fixed across both
rows, since W1 (196x32) and W2 (32x5 or 32x10) have different natural scales and the figure's own
design already treats the two layers as separate questions (see "READ THE TWO LAYERS
SEPARATELY" above); forcing both onto one global range would just shrink one row to a sliver.
Explained variance is now also annotated directly on each panel (top-left) rather than only in
the axis labels, so the PC1-dominance difference between rules is immediately visible: PC's W2
panels run close to 90% on PC1 alone, backprop's sit closer to 70% -- worth noticing on its own.

Styling follows the 300-series scripts: viridis for time, dpi 120, bbox_inches="tight", 9pt
labels. No shared style module.

PROVENANCE
    804_repeated_alternation_{scenario}.npz   weight traces stored for ALL 10 seeds (10-19) of
    each rule (TRACE_SEEDS = 10, raised from 3 by explicit request so the seed-robustness checks
    below rest on the full seed set, not a 3-seed sample), every 10th eval, W1 (6272 = 196x32)
    and W2 (320 = 32x10) both saved. Legacy 68 stored W1 only, Domain-IL only, at a fixed budget;
    this supersedes it.

Two optional diagnostic flags, both born from the same question -- is the main figure's seed-10
shape (in particular Domain-IL/pc/W2's non-oscillating excursion) a real, seed-general effect or
a one-seed artefact? Neither changes the default (no-flag) invocation, which still produces only
the single seed-10 report figure exactly as before.

--per-seed: writes one grid per traced seed (10-19) -- `918_weight_space_pca_seed{N}.png` --
same layout as the main figure. Uses ONE global axis range shared across EVERY panel (both
layers, all four scenario/rule columns, all ten seeds), a deliberate departure from the main
figure's per-layer-row ranges (see "ORIENTED AND SHARED-SCALE" above) requested explicitly: W2's
smaller excursions read as visually compressed relative to W1 here, which is the direct, honest
consequence of one shared range, not a bug.

--consolidated: writes FOUR files, one grid each, W1 on top and W2 below (row label at far
left), columns backprop/Class-IL, backprop/Domain-IL, pc/Class-IL, pc/Domain-IL (rule grouped
before scenario, unlike the main figure, header only on the top row), plus two summary columns
showing what PC1 and PC2 actually ARE in that layer, both boxed and coloured by condition (tab10)
to match the trajectory panels and IDENTICAL across all four files: a 5th column, the
participation ratio (`1 / (N * sum(v_i^4))` on the unit-normalised PC1 loading vector -- 1.0 =
spread evenly over all N weights, ~1/N = one weight does everything); a 6th column, the
distribution across seeds of explained variance, grouped PC1+PC2 / PC1 / PC2 (three box-groups,
one box per condition each, legend gives the colour mapping). Neither boxplot column jitters
individual seeds as points -- tried, and with up to 12 boxes per axis it read as noise rather
than signal; the box (median, IQR, whiskers) and mean marker carry the distribution instead.
CAVEAT throughout: each line's PC1/PC2 is its own independently-fitted, per-seed basis (see "PCA
IS FITTED PER RUN" above) -- valid for comparing shape and scale, never for comparing what a
given direction means between two lines. The four trajectory columns differ only in how the ten
seeds are drawn, and in how each is coloured:
    `918_weight_space_pca_consolidated.png` -- all 10 overlaid on one shared per-panel axis
    (scaled to that panel's own data, not a common range), coloured by alternation block
    (viridis), with a shared colorbar.
    `918_weight_space_pca_consolidated_flat.png` -- same overlay, but every seed in a condition
    drawn in ONE flat colour matched to that condition's box in columns 5 and 6 -- no colorbar,
    since colour no longer encodes time.
    `918_weight_space_pca_consolidated_grid.png` -- overlaying ten seeds on one axis hides which
    one is odd, so this instead gives the first 9 traced seeds (10-18) their own 3x3 sub-region
    within the panel: each seed's trajectory is independently min-max normalised to [0,1]^2 (so
    shape is comparable, absolute scale is not) and shifted into its cell, coloured by
    alternation block as in the first file.
    `918_weight_space_pca_consolidated_grid_flat.png` -- the same 3x3 layout, but every seed's
    line drawn in the condition's flat colour (matching columns 5/6) instead of by block.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt

from src.protocol import figure_path

EXP = ROOT / "experiments"
SCENARIOS = ["class_il", "domain_il"]
NICE = {"class_il": "Class-IL", "domain_il": "Domain-IL"}
LAYERS = [("trace_W1", "$W_1$ (hidden)"), ("trace_W2", "$W_2$ (output)")]
METHODS = ["backprop", "pc"]
SHOW_SEED = 10
SWEEP_SEEDS = list(range(10, 20))


def trace(scenario, rule, seed, key):
    d = np.load(EXP / f"804_repeated_alternation_{scenario}.npz", allow_pickle=True)
    for i, m in enumerate(d["methods"]):
        if m == rule and int(d["seeds"][i]) == seed and f"{key}_{i}" in d.files:
            return (np.asarray(d[f"{key}_{i}"], float), np.asarray(d[f"steps_{i}"]),
                    np.asarray(d[f"switches_{i}"]))
    return None, None, None


def pca2(X):
    """Project onto the trajectory's own first two PCs. Fitted per run -- see the docstring."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD: the trajectory has far fewer points than weights, so this is cheap.
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    return U[:, :2] * S[:2], var[:2]


def orient(P, block1_end):
    """Flip each axis (a valid sign transform of the same PCA result -- variance depends only on
    squared projections, so a sign flip changes nothing about how much each PC explains) so every
    trajectory travels UP in PC2 across block 1, and ends further right in PC1 than it started
    (P1 at the end of the last block > P1 at the very start). A REFLECTION, not a change to the
    trajectory's actual shape, so this does not manufacture resemblance or difference between
    panels; it only removes the arbitrary sign so a real resemblance or difference is visible
    instead of hidden by coincidental axis orientation."""
    P = P.copy()
    if P[block1_end, 1] < P[0, 1]:
        P[:, 1] *= -1
    if P[-1, 0] < P[0, 0]:
        P[:, 0] *= -1
    return P


def contraction(P, idx):
    """Distance between the same phase of successive cycles: |p_k - p_{k-2}| over block ends."""
    pts = P[idx]
    return np.array([np.linalg.norm(pts[k] - pts[k - 2]) for k in range(2, len(pts))])


def compute_panel(scenario, rule, seed, key):
    X, steps, switches = trace(scenario, rule, seed, key)
    if X is None:
        return None
    P, var = pca2(X)
    bounds = [0] + [int(np.searchsorted(steps, w, side="right")) for w in switches]
    ends = [min(b, len(P) - 1) for b in bounds[1:]]
    P = orient(P, ends[0])
    d = contraction(P, ends)
    return dict(P=P, var=var, bounds=bounds, ends=ends, d=d)


def draw_panel(ax, pan, title, xlim, ylim, norm, cmap):
    P, var, bounds, ends, d = pan["P"], pan["var"], pan["bounds"], pan["ends"], pan["d"]
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], min(bounds[b + 1] + 1, len(P))
        ax.plot(P[lo:hi, 0], P[lo:hi, 1], color=cmap(norm(b + 1)), lw=1.2)
    ax.scatter(P[ends, 0], P[ends, 1], s=18, facecolor="none", edgecolor="k", lw=0.7, zorder=5)
    ratio = d[-1] / d[0] if len(d) and d[0] > 0 else np.nan
    ax.set_title(f"{title}\ncycle distance {d[0]:.2f} → {d[-1]:.2f}  (×{ratio:.2f})", fontsize=8)
    ax.set_xlabel(f"PC1 ({100*var[0]:.0f}% var)", fontsize=8)
    ax.set_ylabel(f"PC2 ({100*var[1]:.0f}% var)", fontsize=8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    total = var[0] + var[1]
    ax.annotate(f"{100*total:.0f}% var shown", (0.03, 0.97),
               xycoords="axes fraction", ha="left", va="top", fontsize=8,
               color="0.2" if var[0] < 0.8 else "crimson", fontweight="bold")
    ax.grid(alpha=0.2)


def run_per_seed_sweep(columns, norm, cmap):
    """Diagnostic sweep: same grid as the main figure, once per traced seed, all panels sharing
    ONE global axis range (see the --per-seed note in the module docstring)."""
    panels = {}
    all_pts = []
    for r, (key, llab) in enumerate(LAYERS):
        for seed in SWEEP_SEEDS:
            for c, (s, rule) in enumerate(columns):
                pan = compute_panel(s, rule, seed, key)
                panels[(seed, r, c)] = pan
                if pan is not None:
                    all_pts.append(pan["P"])
    allP = np.concatenate(all_pts, axis=0)
    pad_x = 0.08 * (allP[:, 0].max() - allP[:, 0].min())
    pad_y = 0.08 * (allP[:, 1].max() - allP[:, 1].min())
    xlim = (allP[:, 0].min() - pad_x, allP[:, 0].max() + pad_x)
    ylim = (allP[:, 1].min() - pad_y, allP[:, 1].max() + pad_y)

    for seed in SWEEP_SEEDS:
        fig, axes = plt.subplots(len(LAYERS), len(columns), figsize=(18.0, 9.4))
        for r, (key, llab) in enumerate(LAYERS):
            for c, (s, rule) in enumerate(columns):
                ax = axes[r][c]
                pan = panels[(seed, r, c)]
                if pan is None:
                    ax.set_axis_off()
                    continue
                draw_panel(ax, pan, f"{NICE[s]} · {rule} · {llab}", xlim, ylim, norm, cmap)
        fig.subplots_adjust(hspace=0.42, wspace=0.35, top=0.90, right=0.90)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=axes, label="alternation block", pad=0.02)
        fig.suptitle(f"Weight-space trajectory, seed {seed} — ONE global axis range shared "
                     f"across every panel (both layers, all seeds, all conditions)", fontsize=10)
        out = figure_path(__file__, suffix=f"seed{seed}")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved {out}")


def participation(v):
    """Inverse participation ratio of a unit-normalised loading vector: 1/(N*sum(v_i^4)).
    1.0 = spread evenly over all N entries; ~1/N = one entry does everything."""
    v = v / np.linalg.norm(v)
    return 1.0 / (len(v) * np.sum(v ** 4))


CONSOLIDATED_COLUMNS = [("class_il", "backprop"), ("domain_il", "backprop"),
                        ("class_il", "pc"), ("domain_il", "pc")]


def run_consolidated_diagnostic(mode="block"):
    """See the --consolidated note in the module docstring. W1 (top row) and W2 (bottom row);
    columns backprop/Class-IL, backprop/Domain-IL, pc/Class-IL, pc/Domain-IL, then a 5th column
    showing the participation ratio of PC1's loading vector and a 6th showing the distribution
    of explained variance (PC1+PC2, PC1, PC2) across seeds -- both for the same four conditions,
    coloured to match, IDENTICAL across all four modes below.

    mode="block" (`..._consolidated.png`): all 10 seeds overlaid on one shared per-panel axis,
    each seed's line coloured by alternation block (viridis), a shared colorbar on the right.
    mode="flat" (`..._consolidated_flat.png`): same overlay, but every seed in a condition drawn
    in ONE flat colour, matched to that condition's box in columns 5 and 6 (same tab10 index) --
    no colorbar, since colour no longer encodes time.
    mode="grid" / "grid_flat" (`..._consolidated_grid.png` / `..._grid_flat.png`): overlaying ten
    seeds on one axis hides which one is odd -- this instead gives the first 9 traced seeds
    (10-18) their OWN 3x3 sub-region within the panel. Each seed's trajectory is independently
    min-max normalised to [0,1]^2 (so shape is comparable, absolute scale is not -- "just let
    them fit the sub-axis"), then scaled by 1/3 and shifted into its grid cell -- a deliberate
    axis-breaking transform for spotting an outlier shape by eye, not a shared coordinate system.
    "grid" colours by alternation block (viridis); "grid_flat" colours by condition instead,
    matching columns 5/6."""
    norm = mpl.colors.Normalize(1, 10)
    cmap = mpl.colormaps["viridis"]
    box_colors = mpl.colormaps["tab10"]
    ncols = len(CONSOLIDATED_COLUMNS) + 2

    fig, axes = plt.subplots(len(LAYERS), ncols, figsize=(3.6 * ncols, 4.0 * len(LAYERS)))
    for r, (key, llab) in enumerate(LAYERS):
        axes[r][0].text(-0.32, 0.5, llab, transform=axes[r][0].transAxes, fontsize=13,
                         fontweight="bold", ha="center", va="center", rotation=90)
        pr_by_cond = {}
        var_by_cond = {}
        for c, (s, rule) in enumerate(CONSOLIDATED_COLUMNS):
            ax = axes[r][c]
            panels_this = []
            seeds_this = []
            prs = []
            for seed in SWEEP_SEEDS:
                pan = compute_panel(s, rule, seed, key)
                if pan is None:
                    continue
                panels_this.append(pan)
                seeds_this.append(seed)
                X, _, _ = trace(s, rule, seed, key)
                Xc = X - X.mean(axis=0, keepdims=True)
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                prs.append(participation(Vt[0]))
            pr_by_cond[(s, rule)] = prs
            var_by_cond[(s, rule)] = [pan["var"] for pan in panels_this]
            if mode == "block":
                print(f"  {llab:16s} {NICE[s]:10s} {rule:9s} n_seeds={len(prs)}  "
                      f"participation {np.mean(prs):.3f}±{np.std(prs):.3f}")
            if not panels_this:
                ax.set_axis_off()
                continue

            if mode in ("grid", "grid_flat"):
                for k, (pan, seed) in enumerate(zip(panels_this[:9], seeds_this[:9])):
                    P, bounds = pan["P"], pan["bounds"]
                    pmin, pmax = P.min(axis=0), P.max(axis=0)
                    prange = pmax - pmin
                    prange[prange == 0] = 1.0
                    Pn = (P - pmin) / prange
                    grow, gcol = divmod(k, 3)
                    shift = np.array([gcol / 3.0, (2 - grow) / 3.0])
                    Pg = Pn / 3.0 + shift
                    for b in range(len(bounds) - 1):
                        lo, hi = bounds[b], min(bounds[b + 1] + 1, len(Pg))
                        line_color = cmap(norm(b + 1)) if mode == "grid" else box_colors(c)
                        ax.plot(Pg[lo:hi, 0], Pg[lo:hi, 1], color=line_color, lw=0.8,
                                alpha=0.85)
                    ax.text(shift[0] + 0.02, shift[1] + 0.30, f"s{seed}", fontsize=5,
                            color="0.3", alpha=0.9)
                for g in (1 / 3, 2 / 3):
                    ax.axvline(g, color="0.85", lw=0.6, zorder=0)
                    ax.axhline(g, color="0.85", lw=0.6, zorder=0)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                allP = np.concatenate([p["P"] for p in panels_this], axis=0)
                pad_x = 0.08 * (allP[:, 0].max() - allP[:, 0].min())
                pad_y = 0.08 * (allP[:, 1].max() - allP[:, 1].min())
                ax.set_xlim(allP[:, 0].min() - pad_x, allP[:, 0].max() + pad_x)
                ax.set_ylim(allP[:, 1].min() - pad_y, allP[:, 1].max() + pad_y)
                for pan in panels_this:
                    P, bounds = pan["P"], pan["bounds"]
                    for b in range(len(bounds) - 1):
                        lo, hi = bounds[b], min(bounds[b + 1] + 1, len(P))
                        line_color = cmap(norm(b + 1)) if mode == "block" else box_colors(c)
                        ax.plot(P[lo:hi, 0], P[lo:hi, 1], color=line_color, lw=1.0, alpha=0.75)
            if r == 0:
                ax.set_title(f"{rule} · {NICE[s]}", fontsize=10)
            ax.grid(alpha=0.2)

        ax = axes[r][-2]
        labels = [f"{rule}\n{NICE[s]}" for s, rule in CONSOLIDATED_COLUMNS]
        data = [pr_by_cond[c] for c in CONSOLIDATED_COLUMNS]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        for patch, colidx in zip(bp["boxes"], range(len(CONSOLIDATED_COLUMNS))):
            patch.set_facecolor(box_colors(colidx))
            patch.set_alpha(0.5)
        ax.set_ylabel("participation ratio\nof PC1 loading", fontsize=7)
        if r == 0:
            ax.set_title("participation ratio", fontsize=10)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(alpha=0.25)

        ax = axes[r][-1]
        n_cond = len(CONSOLIDATED_COLUMNS)
        group_labels = ["PC1+PC2", "PC1", "PC2"]
        positions, data, box_col = [], [], []
        for gi in range(3):
            for ci, cond in enumerate(CONSOLIDATED_COLUMNS):
                positions.append(gi * (n_cond + 1) + ci + 1)
                vlist = var_by_cond[cond]
                if gi == 0:
                    vals = [v[0] + v[1] for v in vlist]
                elif gi == 1:
                    vals = [v[0] for v in vlist]
                else:
                    vals = [v[1] for v in vlist]
                data.append(vals)
                box_col.append(box_colors(ci))
        bp2 = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True,
                          showmeans=True)
        for patch, col in zip(bp2["boxes"], box_col):
            patch.set_facecolor(col)
            patch.set_alpha(0.5)
        group_centers = [gi * (n_cond + 1) + (n_cond + 1) / 2 for gi in range(3)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels, fontsize=8)
        ax.set_xlim(0, 3 * (n_cond + 1))
        handles = [mpl.patches.Patch(facecolor=box_colors(ci), alpha=0.5, label=labels[ci])
                   for ci in range(n_cond)]
        ax.legend(handles=handles, fontsize=6, loc="upper right")
        ax.set_ylabel("fraction of variance explained", fontsize=7)
        if r == 0:
            ax.set_title("explained variance", fontsize=10)
        ax.grid(alpha=0.25)

    fig.subplots_adjust(hspace=0.15, wspace=0.28, top=0.90, right=0.98, left=0.05, bottom=0.04)
    if mode == "block":
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=axes[:, :-2], label="alternation block", pad=0.02, fraction=0.03)
    suffix = {"block": "consolidated", "flat": "consolidated_flat",
              "grid": "consolidated_grid", "grid_flat": "consolidated_grid_flat"}[mode]
    out = figure_path(__file__, suffix=suffix)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-seed", action="store_true",
                         help="also write one diagnostic grid per traced seed (10-19), "
                              "all panels on one shared global axis range")
    parser.add_argument("--consolidated", action="store_true",
                         help="also write the four consolidated grids (block/flat/grid/"
                              "grid_flat modes): W1/W2 rows, all 10 seeds per condition, plus "
                              "participation-ratio and explained-variance columns")
    args = parser.parse_args()

    columns = [(s, m) for s in SCENARIOS for m in METHODS]
    fig, axes = plt.subplots(len(LAYERS), len(columns), figsize=(18.0, 9.4))
    norm = mpl.colors.Normalize(1, 10)
    cmap = mpl.colormaps["viridis"]

    # Pass 1: compute every panel's oriented trajectory first, so axis limits can be shared
    # across a row before anything is drawn.
    panels = {}
    for r, (key, llab) in enumerate(LAYERS):
        row_pts = []
        for c, (s, rule) in enumerate(columns):
            pan = compute_panel(s, rule, SHOW_SEED, key)
            panels[(r, c)] = pan
            if pan is None:
                continue
            row_pts.append(pan["P"])
        allP = np.concatenate(row_pts, axis=0)
        pad_x = 0.08 * (allP[:, 0].max() - allP[:, 0].min())
        pad_y = 0.08 * (allP[:, 1].max() - allP[:, 1].min())
        row_xlim = (allP[:, 0].min() - pad_x, allP[:, 0].max() + pad_x)
        row_ylim = (allP[:, 1].min() - pad_y, allP[:, 1].max() + pad_y)
        for c in range(len(columns)):
            if panels[(r, c)] is not None:
                panels[(r, c)]["xlim"] = row_xlim
                panels[(r, c)]["ylim"] = row_ylim

    # Pass 2: draw, now that every panel in a row shares the same grid.
    for r, (key, llab) in enumerate(LAYERS):
        for c, (s, rule) in enumerate(columns):
            ax = axes[r][c]
            pan = panels[(r, c)]
            if pan is None:
                ax.set_axis_off()
                continue
            P, var, bounds, ends, d = pan["P"], pan["var"], pan["bounds"], pan["ends"], pan["d"]
            for b in range(len(bounds) - 1):
                lo, hi = bounds[b], min(bounds[b + 1] + 1, len(P))
                ax.plot(P[lo:hi, 0], P[lo:hi, 1], color=cmap(norm(b + 1)), lw=1.2)
            ax.scatter(P[ends, 0], P[ends, 1], s=18, facecolor="none", edgecolor="k", lw=0.7,
                       zorder=5)
            ratio = d[-1] / d[0] if len(d) and d[0] > 0 else np.nan
            ax.set_title(f"{NICE[s]} · {rule} · {llab}\n"
                         f"cycle distance {d[0]:.2f} → {d[-1]:.2f}  (×{ratio:.2f})", fontsize=8)
            ax.set_xlabel(f"PC1 ({100*var[0]:.0f}% var)", fontsize=8)
            ax.set_ylabel(f"PC2 ({100*var[1]:.0f}% var)", fontsize=8)
            ax.set_xlim(pan["xlim"])
            ax.set_ylim(pan["ylim"])
            total = var[0] + var[1]
            ax.annotate(f"{100*total:.0f}% var shown", (0.03, 0.97),
                       xycoords="axes fraction", ha="left", va="top", fontsize=8,
                       color="0.2" if var[0] < 0.8 else "crimson", fontweight="bold")
            ax.grid(alpha=0.2)
            print(f"  {NICE[s]:10s} {rule:9s} {llab:16s} var {100 * var[0]:4.1f}/{100 * var[1]:4.1f}%   "
                  f"cycle distance {d[0]:.3f} -> {d[-1]:.3f}  ratio {ratio:.2f}")

    # Two-line titles need room, or the top row's x-label lands on the bottom row's title.
    fig.subplots_adjust(hspace=0.42, wspace=0.35, top=0.90, right=0.90)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, label="alternation block", pad=0.02)
    fig.suptitle(f"Weight-space trajectory, seed {SHOW_SEED}, both rules — a closed loop would "
                 f"keep the cycle-to-cycle distance flat", fontsize=10)
    out = figure_path(__file__)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")

    if args.per_seed:
        run_per_seed_sweep(columns, norm, cmap)
    if args.consolidated:
        run_consolidated_diagnostic(mode="block")
        run_consolidated_diagnostic(mode="flat")
        run_consolidated_diagnostic(mode="grid")
        run_consolidated_diagnostic(mode="grid_flat")
