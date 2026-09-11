"""At the established dt=0.4, does PC's settle trace actually CONVERGE at greater depth (just
more slowly, needing a bigger cap) or does it OSCILLATE around a fixed point or DIVERGE outright
(cases a bigger cap cannot fix, which would call for a smaller dt instead) -- and does anything
comparable happen across width, which might help explain the width sweep's own non-monotonic
pattern?

Report series 2, decade 346. Direct companion to 334 (which swept dt at fixed depth=1/width=32)
and to 345 (which found settle steps balloon with depth and, at class-il H=4 specifically, with
width -- but only ever recorded step COUNTS, never the trace shape, so it could not distinguish
"slow but converging" from "not converging at all"). This is the missing piece: full displacement
TRACES, classified with src.metrics.classify_settle (converged / slow / oscillating / diverging),
not just counted.

Same method as 334: fresh init, one batch spanning all classes, full trace, no early exit -- the
relaxation always runs the whole step cap so the shape after any trouble starts is visible, not
just where trouble starts. SEEDS=1 per condition (334's own reasoning: this is about the SHAPE of
the phenomenon, not a mean over seeds).

DEPTH_GRID=[1,2,3,4] at width=32 (342's own grid). WIDTH_GRID=[4,8,16,32,64] at depth=1 (341's own
grid). dt FIXED at 0.4 (the established control, 330-334) -- this script asks whether that fixed
dt is still doing its job as depth/width change, not whether some other dt would do better.
STEP_CAP=500, matching 334's own generous margin (345's cap of 300 was already shown insufficient
at depth>=2 -- traces there were still pinned at the ceiling, not settling).

Two figures, matching 334's convention: 346_..._log.png and 346_..._linear.png. Each has 2 rows
(scenarios) x 2 columns (depth axis, width axis), one trace per grid value, coloured by grid value,
each trace's classify_settle() label shown in the legend/annotation rather than just a convergence
marker -- so "slow" and "oscillating"/"diverging" read differently even though both fail to reach
the marked convergence point.
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
FIXED_DT = 0.4   # the established control (330-334) -- held fixed, not re-swept here
SETTLE_DELTA_TOL = 1e-4
SETTLE_PATIENCE = 3

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEED_START = 10
    DEPTH_GRID, WIDTH_GRID = [1, 2], [4, 32]
    STEP_CAP = 80
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEED_START = CFG["training"]["seed_start"]
    DEPTH_GRID = [1, 2, 3, 4]
    WIDTH_GRID = [4, 8, 16, 32, 64, 128]   # 128 added: checked before trusting it in 341
    STEP_CAP = 500

SCENARIOS = CFG["data"]["scenarios"]

BASE_PROTOCOL = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    lr={METHOD: CFG["training"]["learning_rate"]["pc"]},
    device=CFG["evaluation"]["device"],
    seeds=1,
)


def figure_path(suffix):
    tag = suffix if not SMOKE else f"{suffix}_SMOKE"
    return _figure_path(__file__, tag)


def array_path(scenario):
    return _array_path(__file__, scenario + ("_SMOKE" if SMOKE else ""))


def settle_trace(proto, seed, data):
    """One relaxation at FIXED_DT, fresh init, one batch spanning all classes, full trace, no
    early exit -- same method as 334."""
    handle = {}
    build(proto, METHOD, seed, handle=handle, dt=FIXED_DT, steps=1)   # steps=1 unused -- fresh params only
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
    _, _, disp = pc_settle(x, p, arch, obj, target, dt=FIXED_DT, steps=STEP_CAP, trace=True)
    return np.asarray(disp)


# ---------------------------------------------------------------- run or reload
if REPLOT and all(Path(array_path(s)).exists() for s in SCENARIOS):
    results = {}
    for s in SCENARIOS:
        z = np.load(array_path(s), allow_pickle=True)
        results[s] = list(z["data"])
    print("--replot: redrawing from saved arrays, no relaxation\n")
else:
    t0 = time.perf_counter()
    results = {}   # scenario -> list of (axis, value, seed, disp)
    for scenario in SCENARIOS:
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        rows = []
        for axis, grid in [("depth", DEPTH_GRID), ("width", WIDTH_GRID)]:
            for v in grid:
                p = replace(proto, hidden=32, n_layers=1)
                p = replace(p, n_layers=v) if axis == "depth" else replace(p, hidden=v)
                data = load(p)
                disp = settle_trace(p, SEED_START, data)
                rows.append((axis, v, SEED_START, disp))
        results[scenario] = rows
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]")
    for scenario in SCENARIOS:
        np.savez(array_path(scenario), data=np.array(results[scenario], dtype=object))
    print("saved arrays")

# ---------------------------------------------------------------- classification summary
LABEL_COLOR = {"converged": "tab:green", "slow": "tab:blue",
              "oscillating": "tab:orange", "diverging": "tab:red", "too_short": "0.5"}

print(f"\nclassify_settle summary (dt={FIXED_DT} fixed, cap={STEP_CAP}):")
for scenario in SCENARIOS:
    print(f"\n  {scenario}")
    for axis, grid in [("depth", DEPTH_GRID), ("width", WIDTH_GRID)]:
        for v in grid:
            row = next(r for r in results[scenario] if r[0] == axis and r[1] == v)
            disp = np.asarray(row[3])
            label, conv_step = classify_settle(disp, tol=SETTLE_DELTA_TOL, patience=SETTLE_PATIENCE)
            extra = f" (step {conv_step})" if conv_step is not None else ""
            print(f"    {axis}={v:<4} -> {label}{extra}   "
                 f"final disp={disp[-1]:.4f}   max disp={disp.max():.4f}")

# ---------------------------------------------------------------- figure(s)
XLIM = 300


def make_figure(log_scale, suffix):
    fig, axes = plt.subplots(len(SCENARIOS), 2, figsize=(13, 4.5 * len(SCENARIOS)), squeeze=False)
    for row_i, scenario in enumerate(SCENARIOS):
        rows = results[scenario]
        for col_i, (axis, grid, xlabel) in enumerate([("depth", DEPTH_GRID, "depth"),
                                                       ("width", WIDTH_GRID, "width")]):
            ax = axes[row_i][col_i]
            norm = mcolors.Normalize(vmin=min(grid), vmax=max(grid))
            cmap = cm.viridis
            for v in grid:
                row = next(r for r in rows if r[0] == axis and r[1] == v)
                disp = np.asarray(row[3])
                label, conv_step = classify_settle(disp, tol=SETTLE_DELTA_TOL,
                                                   patience=SETTLE_PATIENCE)
                color = cmap(norm(v))
                ax.plot(np.arange(1, len(disp) + 1), disp, color=color, lw=1.2, alpha=0.85,
                       label=f"{xlabel}={v} [{label}]")
                if conv_step is not None and conv_step <= len(disp):
                    ax.scatter([conv_step], [disp[conv_step - 1]], marker="x", color="black",
                              s=30, zorder=5)
                else:
                    ax.scatter([len(disp)], [disp[-1]], marker="o", s=30, zorder=5,
                              color=LABEL_COLOR[label], edgecolor="black", linewidth=0.5)
            if log_scale:
                ax.set_yscale("log")
            ax.set_xlim(0, XLIM)
            ax.set_xlabel("settle step")
            ax.set_ylabel("displacement")
            ax.set_title(f"{scenario.replace('_', '-')}: settle trace vs {xlabel}"
                        f" (dt={FIXED_DT} fixed)", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper right")
            ax.grid(alpha=0.2, which="both")

    scale_name = "log-y" if log_scale else "linear-y"
    fig.suptitle(f"PC settle trace vs depth/width at fixed dt={FIXED_DT} ({scale_name}) -- "
                f"x = converged (criterion fired), o = did not (colour = classify_settle label), "
                f"cap={STEP_CAP}", fontsize=9)
    fig.tight_layout()
    path = figure_path(suffix)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"saved {path}")


make_figure(log_scale=True, suffix="log")
make_figure(log_scale=False, suffix="linear")
