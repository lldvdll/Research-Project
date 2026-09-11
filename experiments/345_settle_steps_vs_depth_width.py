"""Does PC's settle step count grow with depth (a candidate explanation for 342's unusually long
runtime, since PC's settle control -- dt=0.4, adaptive stop -- was only validated at depth=1), and
does anything similarly unexpected show up across width (which might help explain the width
sweep's own non-monotonic pattern, e.g. H=8 not reaching significance while H=16/32/64 did)?

Report series 2, decade 345. Diagnostic, not a headline result -- mirrors 334's design (settle
steps under the established adaptive-stop criterion, read off LIVE training rather than a
separate probe) but sweeps depth and width instead of dt, which 334 already covered.

CHEAP APPROXIMATION, stated: a full matched-competence run at every depth/width (like 341/342
themselves) would cost exactly what this is trying to explain cheaply. Instead: N_STEPS=200 of
FIXED joint training (all classes at once, no task switch, no early stopping -- same simplification
344 used) per depth/width point, settle steps read off handle["diag"]["settle_steps"] after every
update (330's pattern). This does not reproduce 341/342 exactly, but it directly tests the
mechanism in question -- does settling itself cost more steps -- much more cheaply.

SETTLE_STEP_CAP here is 300, not the production 100 -- deliberately generous, so a depth/width
point that would be truncated in production shows up as a real, uncapped number here instead of
silently hitting the same ceiling. The production cap (100) is also checked against directly: the
fraction of updates that would have exceeded it is reported, since that fraction -- not just the
mean -- is what would make 341/342 slow (every capped update pays the full 100-step cost).

DEPTHS = [1,2,3,4] at width=32 (342's own grid). WIDTHS = [4,8,16,32,64] at depth=1 (341's own
grid). lr fixed at pc's established default (0.02). 3 seeds -- diagnostic, not a headline number.

One figure: 345_settle_steps_vs_depth_width.png, 2 rows (settle steps: mean+max; wall-clock for
the 200-step window) x 2 columns (depth axis, width axis), both scenarios overlaid per panel.
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

from src.protocol import PROTOCOL, replace, load, build, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import _loader

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

OPTIMAL_DT = 0.4
SETTLE_DELTA_TOL = 1e-4
SETTLE_PATIENCE = 3
PRODUCTION_CAP = 100   # what 341/342 actually use -- compared against here, not re-imposed

with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, SEED_START = 2, 10
    DEPTHS, WIDTHS = [1, 2], [4, 32]
    N_STEPS = 20
    SETTLE_STEP_CAP = 60
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = 3
    SEED_START = CFG["training"]["seed_start"]
    DEPTHS = [1, 2, 3, 4]
    WIDTHS = [4, 8, 16, 32, 64]
    N_STEPS = 200
    SETTLE_STEP_CAP = 300   # generous -- see docstring

SCENARIOS = CFG["data"]["scenarios"]
PC_LR = CFG["training"]["learning_rate"]["pc"]

BASE_PROTOCOL = replace(
    PROTOCOL,
    img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
    act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
    optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
    device=CFG["evaluation"]["device"],
)


def figure_path():
    return _figure_path(__file__, "SMOKE" if SMOKE else "")


def array_path():
    return _array_path(__file__, "SMOKE" if SMOKE else "")


def _row_for(proto, seed, data):
    """(mean_settle_steps, max_settle_steps, frac_over_production_cap, wallclock_seconds) over
    N_STEPS of fixed joint training."""
    handle = {}
    train_step, predict = build(proto, "pc", seed, handle=handle, lr=PC_LR, dt=OPTIMAL_DT,
                                steps=SETTLE_STEP_CAP, stop_delta=SETTLE_DELTA_TOL,
                                stop_patience=SETTLE_PATIENCE)
    settle_log = []

    # Domain-IL shares output units across the (here, single joint) class group -- label_map
    # remaps the 10 raw class ids onto the 5 output units train_step/pc_update actually expect.
    # None for Class-IL (out_dim == n_classes, so raw ids already are the right target index).
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)

    def wrapped(x, y, active=None):
        if lmap is not None:
            y = torch.tensor([lmap[int(v)] for v in y.tolist()], device=y.device)
            active = None if active is None else sorted({lmap[int(c)] for c in active})
        train_step(x, y, active=active)
        if "settle_steps" in handle["diag"]:
            settle_log.append(handle["diag"]["settle_steps"])

    classes = list(range(proto.n_classes))
    loader = _loader(data.train, data.class_idx, classes, proto.batch, seed=seed)
    it = iter(loader)
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        wrapped(x.to(proto.device), y.to(proto.device), active=classes)
    wallclock = time.perf_counter() - t0

    s = np.array(settle_log, dtype=float)
    frac_over = float(np.mean(s >= PRODUCTION_CAP)) if s.size else float("nan")
    return float(s.mean()), float(s.max()), frac_over, wallclock


if REPLOT and Path(array_path()).exists():
    rows = list(np.load(array_path(), allow_pickle=True)["data"])
    print("--replot: redrawing from saved arrays, no training\n")
else:
    rows = []
    t0 = time.perf_counter()
    for scenario in SCENARIOS:
        proto = replace(BASE_PROTOCOL, scenario=scenario)
        for axis, values in [("depth", DEPTHS), ("width", WIDTHS)]:
            for v in values:
                p = replace(proto, hidden=32, n_layers=1)
                p = replace(p, n_layers=v) if axis == "depth" else replace(p, hidden=v)
                data = load(p)
                for seed in range(SEED_START, SEED_START + SEEDS):
                    mean_s, max_s, frac_over, wc = _row_for(p, seed, data)
                    rows.append((scenario, axis, v, seed, mean_s, max_s, frac_over, wc))
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]")
    np.savez(array_path(), data=np.array(rows, dtype=object))
    print("saved arrays")


def _vals(scenario, axis, v, col):
    return np.array([r[col] for r in rows
                     if r[0] == scenario and r[1] == axis and r[2] == v], dtype=float)


SCEN_COLOR = {"class_il": "tab:blue", "domain_il": "tab:red"}
COL = dict(mean_s=4, max_s=5, frac_over=6, wallclock=7)

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
for col_i, (axis, values, xlabel) in enumerate([("depth", DEPTHS, "hidden layers (depth)"),
                                               ("width", WIDTHS, "hidden width H")]):
    ax_settle, ax_wall = axes[0][col_i], axes[1][col_i]
    for scenario in SCENARIOS:
        means = [_vals(scenario, axis, v, COL["mean_s"]).mean() for v in values]
        maxes = [_vals(scenario, axis, v, COL["max_s"]).max() for v in values]
        ax_settle.plot(values, means, marker="o", color=SCEN_COLOR[scenario],
                      label=f"{scenario} mean")
        ax_settle.plot(values, maxes, marker="x", ls="--", color=SCEN_COLOR[scenario],
                      label=f"{scenario} max", alpha=0.6)
        for v, m in zip(values, means):
            frac = _vals(scenario, axis, v, COL["frac_over"]).mean()
            if frac > 0:
                ax_settle.annotate(f"{frac:.0%}>cap", (v, m), fontsize=6, color="crimson",
                                  xytext=(0, 6), textcoords="offset points", ha="center")

        wall = [_vals(scenario, axis, v, COL["wallclock"]).mean() for v in values]
        ax_wall.plot(values, wall, marker="o", color=SCEN_COLOR[scenario], label=scenario)

    ax_settle.axhline(PRODUCTION_CAP, color="black", ls=":", lw=1,
                      label=f"production cap ({PRODUCTION_CAP})")
    ax_settle.set_xlabel(xlabel); ax_settle.set_ylabel("settle steps")
    ax_settle.set_title(f"settle steps vs {axis}", fontsize=9)
    ax_settle.grid(alpha=0.2)
    if col_i == 0:
        ax_settle.legend(fontsize=6.5)

    ax_wall.set_xlabel(xlabel); ax_wall.set_ylabel(f"wall-clock for {N_STEPS} steps (s)")
    ax_wall.set_title(f"wall-clock vs {axis}", fontsize=9)
    ax_wall.grid(alpha=0.2)
    if col_i == 0:
        ax_wall.legend(fontsize=7)

fig.tight_layout()
fig.savefig(figure_path(), dpi=130, bbox_inches="tight")
print(f"saved {figure_path()}")

print("\nsummary (mean settle steps -> production-cap fraction):")
for scenario in SCENARIOS:
    print(f"  {scenario}")
    for axis, values in [("depth", DEPTHS), ("width", WIDTHS)]:
        for v in values:
            m = _vals(scenario, axis, v, COL["mean_s"]).mean()
            frac = _vals(scenario, axis, v, COL["frac_over"]).mean()
            wc = _vals(scenario, axis, v, COL["wallclock"]).mean()
            print(f"    {axis}={v:<4} mean settle steps={m:6.1f}   >cap frac={frac:5.1%}"
                 f"   wallclock/{N_STEPS}steps={wc:5.1f}s")
