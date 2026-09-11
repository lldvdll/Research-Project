"""Is there ONE dt that settles cleanly at every depth 1-4 (in which case depth can stay the only
varying axis, no confound to weigh), or does depth>1 need its own, smaller dt -- and if it does,
is accuracy/forgetting still flat across THAT depth's own stable dt sub-range, the way 330 found
at depth=1 (in which case picking each depth's own smallest-stable dt is a fair, motivated choice,
not a confound -- the same reasoning that already applies to lr)?

Report series 2, decade 347. Direct follow-up to 346, which found dt=0.4 (the established
depth=1 control) oscillates, not converges, at depth 2-4. This is 334's own method (dt sweep,
settle trace, classify_settle) generalised across BOTH axes at once -- dt AND depth -- rather than
dt alone. Width is not swept here: 346 already found dt=0.4 stable at every width tested
(4-128) except one already-flagged cell (class-il, H=4), not a systematic problem the way depth is.

STAGE 1 ONLY: this script answers "which dt is stable at which depth" via settle traces (fresh
init, single batch, full trace, no training) -- it does NOT yet check whether accuracy/forgetting
are flat across a depth's own stable dt range. That is a separate, necessarily more expensive
check (needs real training, not just a settle trace) and is only worth designing once this script
shows what each depth's stable dt range actually looks like.

DT_GRID = 334's own depth=1 grid (0.02, 0.05, 0.1, 0.2, 0.3, 0.4) -- same values, so this reads
directly against 334's own dt-vs-depth=1 result. DEPTH_GRID = [1, 2, 3, 4] (1 included as the
known-good reference, not because it needs re-checking). STEP_CAP=500, matching 334/346.

Same method as 334/346: fresh init, one batch spanning all classes, full trace, no early exit,
SEEDS=1 per (dt, depth, scenario) cell -- this is about the SHAPE of the phenomenon, not a mean.

Figure: 347_pc_settle_dt_by_depth.png, 2 rows (scenarios) x 4 columns (depth), each panel a
dt-coloured settle-trace overlay (334's own single-axis figure design, one panel per depth
instead of one figure for the whole dt axis) -- so 334's own depth=1 panel should be visually
reproducible here as a sanity check.
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
from src.metrics import classify_settle

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

METHOD = "pc"
SETTLE_DELTA_TOL = 1e-4
SETTLE_PATIENCE = 3

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEED_START = 10
    DT_GRID, DEPTH_GRID = [0.1, 0.4], [1, 2]
    STEP_CAP = 80
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEED_START = CFG["training"]["seed_start"]
    DT_GRID = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]   # 334's own depth=1 grid -- see docstring
    DEPTH_GRID = [1, 2, 3, 4]
    STEP_CAP = 500

SCENARIOS = CFG["data"]["scenarios"]

BASE_PROTOCOL = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    hidden=CFG["architecture"]["hidden"],
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    lr={METHOD: CFG["training"]["learning_rate"]["pc"]},
    device=CFG["evaluation"]["device"],
    seeds=1,
)


def figure_path(suffix=""):
    tag = suffix if not SMOKE else f"{suffix}_SMOKE".lstrip("_")
    return _figure_path(__file__, tag)


def array_path(scenario):
    return _array_path(__file__, scenario + ("_SMOKE" if SMOKE else ""))


def settle_trace(proto, dt, seed, data):
    handle = {}
    build(proto, METHOD, seed, handle=handle, dt=dt, steps=1)
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


if REPLOT and all(Path(array_path(s)).exists() for s in SCENARIOS):
    results = {}
    for s in SCENARIOS:
        z = np.load(array_path(s), allow_pickle=True)
        results[s] = list(z["data"])
    print("--replot: redrawing from saved arrays, no relaxation\n")
else:
    t0 = time.perf_counter()
    results = {}   # scenario -> list of (depth, dt, seed, disp)
    for scenario in SCENARIOS:
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        rows = []
        for depth in DEPTH_GRID:
            p = replace(proto, n_layers=depth)
            data = load(p)
            for dt in DT_GRID:
                disp = settle_trace(p, dt, SEED_START, data)
                rows.append((depth, dt, SEED_START, disp))
        results[scenario] = rows
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]")
    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(results[scenario], dtype=object))
    print("saved arrays")

# ---------------------------------------------------------------- classification summary
print(f"\nclassify_settle: dt x depth (cap={STEP_CAP})")
stable_by_depth = {}
for scenario in SCENARIOS:
    print(f"\n  {scenario}")
    stable_by_depth[scenario] = {}
    for depth in DEPTH_GRID:
        stable = []
        for dt in DT_GRID:
            row = next(r for r in results[scenario] if r[0] == depth and r[1] == dt)
            disp = np.asarray(row[3])
            label, conv_step = classify_settle(disp, tol=SETTLE_DELTA_TOL, patience=SETTLE_PATIENCE)
            if label == "converged":
                stable.append(dt)
            print(f"    depth={depth} dt={dt:<5} -> {label}"
                 f"{f' (step {conv_step})' if conv_step is not None else ''}")
        stable_by_depth[scenario][depth] = stable

print("\nstable dt values per depth:")
for scenario in SCENARIOS:
    print(f"  {scenario}: " + ", ".join(f"depth={d}:{stable_by_depth[scenario][d]}"
                                       for d in DEPTH_GRID))
common = set(DT_GRID)
for scenario in SCENARIOS:
    for depth in DEPTH_GRID:
        common &= set(stable_by_depth[scenario][depth])
print(f"\ndt values stable at EVERY depth, BOTH scenarios: {sorted(common) if common else 'NONE'}")

# ---------------------------------------------------------------- figure
XLIM = 300
norm = mcolors.LogNorm(vmin=min(DT_GRID), vmax=max(DT_GRID))
cmap = cm.viridis

fig, axes = plt.subplots(len(SCENARIOS), len(DEPTH_GRID),
                        figsize=(4.2 * len(DEPTH_GRID), 4.2 * len(SCENARIOS)), squeeze=False)
for row_i, scenario in enumerate(SCENARIOS):
    for col_i, depth in enumerate(DEPTH_GRID):
        ax = axes[row_i][col_i]
        for dt in DT_GRID:
            row = next(r for r in results[scenario] if r[0] == depth and r[1] == dt)
            disp = np.asarray(row[3])
            label, conv_step = classify_settle(disp, tol=SETTLE_DELTA_TOL, patience=SETTLE_PATIENCE)
            color = cmap(norm(dt))
            ax.plot(np.arange(1, len(disp) + 1), disp, color=color, lw=0.9, alpha=0.85)
            if conv_step is not None and conv_step <= len(disp):
                ax.scatter([conv_step], [disp[conv_step - 1]], marker="x", color="black", s=20,
                          zorder=5)
        ax.set_yscale("log")
        ax.set_xlim(0, XLIM)
        ax.set_xlabel("settle step")
        if col_i == 0:
            ax.set_ylabel(f"{scenario.replace('_', '-')}\ndisplacement")
        ax.set_title(f"depth={depth}", fontsize=9)
        ax.grid(alpha=0.2, which="both")

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, label="dt", pad=0.01)
cbar.set_ticks(DT_GRID)
cbar.set_ticklabels([str(dt) for dt in DT_GRID])

fig.suptitle("PC settle trace: dt x depth (x = converged, criterion as 330/333/334/346)",
            fontsize=9)
path = figure_path()
fig.savefig(path, dpi=130, bbox_inches="tight")
print(f"saved {path}")
