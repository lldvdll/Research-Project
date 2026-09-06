"""Global plot style for the 900-series report figures.

ONE RESULT, ONE PLOT, ONE FILE -- by default. A 900 script writes one PNG per
result and the report composes them with LaTeX `subfigure`, each with its own
subcaption. The exception is a figure whose panels must SHARE AN AXIS for the
comparison to be readable (911, 912, 917, 921, 923): those stay in one file with
sharex/sharey, take one LaTeX caption, and are lettered in-figure by
`panel_label`. Hand-matching limits across separate files is how inconsistency
gets in, so where comparability is the point, matplotlib should enforce it.

SIZES ARE FINAL. Save at the physical width the panel occupies in the document
and include it at that width with no scaling. This is why `savefig.bbox` is
"standard" and NOT "tight", with constrained layout doing the fitting instead:
tight crops to drawn content, so the output is not the size requested (measured,
matplotlib 3.11: a 3.44 in request saves as 3.11 in) and the crop varies with
content. LaTeX would then scale each panel of a grid by a different factor and
the font sizes would drift apart -- the exact problem this module exists to
prevent. Constrained layout returns exactly 3.440 in whatever the labels say.

Nothing here draws anything. It sets rcParams and names colours, so a font or
palette change is one edit that re-renders every figure -- it is not a harness.
"""
import matplotlib as mpl

# ---------------------------------------------------------------- physical sizes, inches
# revtex4-2 aps twocolumn: \columnwidth = 3.404, \textwidth = 7.06.
# report/main.tex already defines \singlefigure = 0.45\textwidth = 3.18.
WIDTH = {
    "half_col":  1.55,   # two panels across INSIDE one column -- too small for axes, avoid
    "col":       3.18,   # one panel filling a column (= \singlefigure)
    "half_page": 3.44,   # two panels across a figure* spanning both columns -- grid default
    "page":      7.06,   # one panel spanning both columns
}


def size(name, ratio=0.72):
    """(width, height) in inches for one PANEL. `ratio` is height/width.

    For a Mode B figure holding an n x m grid, multiply: a 2x2 at "half_page" is
    figsize=(2 * WIDTH["half_page"], 2 * WIDTH["half_page"] * ratio).
    """
    w = WIDTH[name]
    return (w, w * ratio)


# ---------------------------------------------------------------- colour groupings
# Matched to the figures already in the repo (340/341/342) so new and legacy agree.
RULE = {"backprop": "#5c5c5c", "pc": "#d1682a", "replay": "#3f7d3a"}
TASK = {0: "#d1682a", 1: "#2b6ca3"}                  # task 1 orange, task 2 blue

# Domain-IL is TEAL, not green: 340/341/342 already spend green on replay, and a
# figure carrying both rules and scenarios would otherwise be ambiguous. Script
# 112's existing PNG uses green here and disagrees until it is redrawn as 908.
SCENARIO = {"class_il": "#6a4c93", "domain_il": "#1b7f79"}

NEUTRAL = "#8a8a8a"      # a series that is context, not the comparison
ZERO_LINE = "#333333"    # the y = 0 reference on any paired-difference plot
CAPPED = "#a33121"       # annotation for runs that hit the iteration cap / censored cells


def sweep(n, dark=False):
    """`n` ordered colours for a swept axis (lr, width, depth, dt)."""
    # mpl.cm.get_cmap was removed in matplotlib 3.9; this repo runs 3.11.
    cmap = mpl.colormaps["magma" if dark else "viridis"]
    return [cmap(i / max(1, n - 1) * 0.88) for i in range(n)]


# ---------------------------------------------------------------- rcParams
def apply(base=8.0):
    """Set global style. `base` is the final ON-PAGE font size in points.

    Call once at the top of a 900 script, before creating any figure.
    """
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.labelpad": 2.5,
        "axes.titlelocation": "left",
        "lines.linewidth": 1.3, "lines.markersize": 3.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "grid.linewidth": 0.4, "grid.alpha": 0.25,
        "errorbar.capsize": 1.8,
        "figure.dpi": 150, "savefig.dpi": 400,
        "figure.constrained_layout.use": True,      # honours figsize exactly
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "savefig.bbox": "standard",                 # NOT "tight" -- see module docstring
    })


def panel_label(ax, letter, dx=-0.16, dy=1.04):
    """Mode B only: draw (a), (b), ... in-figure.

    A shared-axes figure takes ONE LaTeX caption, so the panel letters have to
    come from matplotlib. Mode A panels are separate files and are labelled by
    \\subcaption instead -- do not use this there, or they will be lettered twice.
    """
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=mpl.rcParams["font.size"], va="bottom", ha="left")
