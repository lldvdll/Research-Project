"""Across the WHOLE dt range in one plot: does the settle trace's convergence step shrink
smoothly as dt grows, and at what dt does it stop converging at all and start oscillating
around a different plateau instead?

Report series 2, decade 334 -- companion to 330/333. 330 sweeps dt and reports the RESULT
(accuracy, mean settle-steps-taken); this script shows the actual per-step displacement TRACE
for every dt on one set of axes, so the transition from "converges, just slower" to "does not
converge" is visible directly, not just inferred from a mean. Same criterion as 330/333
(|disp[t]-disp[t-1]| < 1e-4 for 3 consecutive steps) is used only to MARK where each trace would
have been called converged -- the relaxation itself always runs the full step cap here, since the
point is to see the whole shape, including whatever happens after a trace fails to settle.

DT_GRID -- 330's 7 core values (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0), so this plot reads directly
against 330's own dt-vs-settle-steps panel, PLUS three extra points (0.7, 0.85, 1.5) added
specifically to resolve the stable/unstable transition: 330's grid jumps straight from 0.5
(stable, converges) to 1.0 (unstable, oscillates at a different plateau -- found earlier, ~0.29
vs the stable ~0.073), so the boundary between them is currently a guess. 1.5 is included beyond
1.0 as a clearly-diverged reference point (found earlier: displacement climbs to ~5, no plateau).

STEP_CAP=500 -- generous margin beyond 333's validated 300, since a couple of the new finer-grid
points near the transition are untested and might need longer to reveal their character.
SEEDS=1 -- ONE run per dt, not several. Multiple seeds per dt on the same axes, in the same
colour, was indistinguishable clutter rather than informative spread -- this plot is about the
SHAPE of the phenomenon, not a mean over seeds, so a single representative trace per dt is the
right amount of data, not less rigorous. Fresh init, one representative batch spanning all ten
classes, same method as 330/333's diagnostic snapshots.

Two figures, same data, different files: log-y (..._log.png) and linear-y (..._linear.png).
Log makes the wide dynamic range (1e-3 to ~1) readable at once; linear shows the oscillation
amplitude and the converged plateau's actual scale without log compression -- worth having both
since they emphasise different things and neither is strictly more correct here.

One panel per scenario, ALL dt values overlaid on the same axes (colour = dt, viridis, log-scaled
colour mapping regardless of which figure) -- not split into subplots per dt. Each trace gets a
marker at its own detected convergence step, if any, so "converges but slow" vs "never converges"
is visible from the same criterion 330 uses, not a different one.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.protocol import PROTOCOL, replace, load, build, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import _loader
from src.predictive_coding import pc_settle
from src.model import make_target

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

METHOD = "pc"
SETTLE_DELTA_TOL = 1e-4   # same criterion as 330/333 -- used here only to MARK convergence
SETTLE_PATIENCE = 3       # matches 330's (project stop_patience convention), not 333's original 10

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, SEED_START = 2, 10
    DT_GRID = [0.05, 0.5, 1.0]
    STEP_CAP = 80
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = 1   # ONE run per dt -- see docstring
    SEED_START = CFG["training"]["seed_start"]
    # Full grid restored -- the clutter was multiple seeds per dt, not the number of dt values.
    # 330's 7 core values (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0) plus three added to resolve the
    # stable/unstable transition: 0.7, 0.85 (the scenario-dependent boundary point -- converges
    # under domain-il, not class-il), 1.5 (clearly diverged reference).
    DT_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 0.85, 1.0, 1.5]
    STEP_CAP = 500

SCENARIOS = CFG["data"]["scenarios"]

BASE_PROTOCOL = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    hidden=CFG["architecture"]["hidden"], n_layers=CFG["architecture"]["n_layers"],
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    lr={METHOD: CFG["training"]["learning_rate"]["pc"]},
    device=CFG["evaluation"]["device"],
    seeds=SEEDS,
)


def figure_path(suffix):
    tag = suffix if not SMOKE else f"{suffix}_SMOKE"
    return _figure_path(__file__, tag)


def array_path(scenario):
    return _array_path(__file__, scenario + ("_SMOKE" if SMOKE else ""))


def _converged_at(disp):
    """Same criterion as 330/333: first index where |disp[i]-disp[i-1]| stays below
    SETTLE_DELTA_TOL for SETTLE_PATIENCE consecutive steps. None if it never does."""
    delta = np.abs(np.diff(disp))
    below = delta < SETTLE_DELTA_TOL
    for i in range(len(below) - SETTLE_PATIENCE + 1):
        if below[i:i + SETTLE_PATIENCE].all():
            return i + 1
    return None


def settle_trace(proto, dt, seed, data):
    """One relaxation, fresh init, one batch spanning all classes, full trace, no early exit --
    the whole point here is to see the entire shape, including whatever a non-converging dt does
    for the rest of the step budget."""
    handle = {}
    build(proto, METHOD, seed, handle=handle, dt=dt, steps=1)   # steps=1 unused -- only need fresh params
    p, arch, obj = handle["params"], handle["arch"], handle["obj"]
    classes = list(range(proto.n_classes))
    loader = _loader(data.train, data.class_idx, classes, proto.batch, seed=seed)
    x, y = next(iter(loader))
    x, y = x.to(proto.device), y.to(proto.device)
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    if lmap is not None:
        y = torch.tensor([lmap[int(v)] for v in y.tolist()], device=proto.device)
    target = make_target(y, arch, obj, device=proto.device)
    _, _, disp = pc_settle(x, p, arch, obj, target, dt=dt, steps=STEP_CAP, trace=True)
    return np.asarray(disp)


# ---------------------------------------------------------------- run or reload
if REPLOT and all(Path(array_path(s)).exists() for s in SCENARIOS):
    results = {}
    for s in SCENARIOS:
        z = np.load(array_path(s), allow_pickle=True)
        rows = list(z["data"])
        # DT_GRID/SEEDS may be a subset of what was originally computed (e.g. this run reduced
        # 3 seeds/dt to 1) -- filter rather than recompute when every requested (dt, seed) pair
        # is already present in the saved data.
        wanted_seeds = set(range(SEED_START, SEED_START + SEEDS))
        filtered = [r for r in rows if r[0] in DT_GRID and r[1] in wanted_seeds]
        if len(filtered) < len(DT_GRID) * SEEDS:
            raise RuntimeError(
                f"{s}: saved data doesn't cover the requested dt/seed combination -- "
                f"delete {array_path(s)} and rerun without --replot"
            )
        results[s] = filtered
    print("--replot: redrawing from saved arrays, no relaxation\n")
else:
    t0 = time.perf_counter()
    results = {}   # scenario -> list of (dt, seed, disp)
    for scenario in SCENARIOS:
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        data = load(proto)
        rows = []
        for dt in DT_GRID:
            for seed in range(SEED_START, SEED_START + SEEDS):
                disp = settle_trace(proto, dt, seed, data)
                rows.append((dt, seed, disp))
        results[scenario] = rows
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]"
             f"   ({len(DT_GRID)} dt x {SEEDS} seeds = {len(rows)} traces)")
    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(results[scenario], dtype=object))
    print("saved arrays")

# ---------------------------------------------------------------- figure(s)
# Stacked (one row per scenario, not side by side) so each panel gets the full figure width --
# the oscillation structure in the unstable dt band needs horizontal resolution to read. X-axis
# clipped to 200 steps for display: everything of interest (convergence, or the onset/shape of
# oscillation) happens well before then, and the flat tail out to STEP_CAP=500 was mostly wasted
# space. The full 500-step trace is still what's saved/computed -- only the displayed range changes.
XLIM = 200
norm = mcolors.LogNorm(vmin=min(DT_GRID), vmax=max(DT_GRID))
cmap = cm.viridis


def make_figure(log_scale, suffix):
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(11, 4.5 * len(SCENARIOS)), squeeze=False)
    axes = axes[:, 0]
    for ax, scenario in zip(axes, SCENARIOS):
        rows = results[scenario]
        n_converged = {dt: 0 for dt in DT_GRID}
        n_total = {dt: 0 for dt in DT_GRID}
        for dt, seed, disp in rows:
            color = cmap(norm(dt))
            ax.plot(np.arange(1, len(disp) + 1), disp, color=color, lw=0.9, alpha=0.8)
            conv = _converged_at(disp)
            n_total[dt] += 1
            if conv is not None and conv <= len(disp):
                ax.scatter([conv], [disp[conv - 1]], marker="x", color="black", s=25, zorder=5)
                n_converged[dt] += 1
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlim(0, XLIM)
        ax.set_xlabel("settle step")
        ax.set_ylabel("displacement")
        ax.set_title(scenario.replace("_", "-"), fontsize=10)
        ax.grid(alpha=0.2, which="both")
        never = [dt for dt in DT_GRID if n_converged[dt] == 0]
        if never:
            ax.text(0.98, 0.03, f"never converged: dt={never}", transform=ax.transAxes,
                   fontsize=7, color="tab:red", ha="right", va="bottom")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label="dt", pad=0.02)
    cbar.set_ticks(DT_GRID)
    cbar.set_ticklabels([str(dt) for dt in DT_GRID])

    scale_name = "log-y" if log_scale else "linear-y"
    fig.suptitle(f"PC settle trace across the whole dt range ({scale_name}) -- x = convergence "
                f"marker (criterion: |{chr(0x394)}disp|<{SETTLE_DELTA_TOL:.0e} for "
                f"{SETTLE_PATIENCE} steps), {SEEDS} run/dt, cap={STEP_CAP}", fontsize=9)
    path = figure_path(suffix)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"saved {path}")


make_figure(log_scale=True, suffix="log")
make_figure(log_scale=False, suffix="linear")
