"""Does backprop's canonical headline figure (310) change substantively if backprop's lr moves
from the historical 0.01 to 0.02 -- the shared value 340's lr sweep found close to jointly optimal
for both rules, adopted (provisionally, pending 340/341's full-seed results) as the project's
single shared lr going forward?

Report series 2, decade 314. Exact copy of 310 in every respect except ONE stated deviation:
lr=0.02 for backprop (310 used config_300.yaml's historical 0.01). Same seeds (10-19), same
protocol, same figure design -- the comparison is only clean if nothing else moves.

This is a ROBUSTNESS CHECK on adopting lr=0.02 project-wide, not a new finding in its own right:
310 is the canonical §1 figure ("forgetting happens, and its shape differs by scenario") and is
cited elsewhere by that number. If switching backprop's lr changes it substantively, that is a
reason to reconsider fixing lr=0.02 before it propagates further (341/342 already used the
per-rule historical pair; 340 is what raised the question in the first place). If it does not,
310 stands as-is and lr=0.02 can be adopted without redoing anything upstream.

Paired comparison against 310's own saved arrays (same seeds, so paired_diff/paired_sign apply
directly) is printed at the end, on both final retention and crossover height, for both scenarios.
310 must have already been run (its .npz files present) for this to work.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, replace, load, build, figure_path as _figure_path, \
    array_path as _array_path
from src.runner import run_classil, _acc
from src.metrics import crossover, paired_diff, paired_sign

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

SEED_START = 10

BACKPROP_LR = 0.02   # the ONE stated deviation from 310 -- see docstring

# ---------------------------------------------------------------- plot-only choices (not experiment parameters -- see config_300.yaml for those)
TASK1_COLOR = "tab:orange"
TASK2_COLOR = "tab:blue"
CROSSOVER_COLOR = "tab:green"
FINAL_COLOR = "tab:red"

# ---------------------------------------------------------------- load every control parameter explicitly
with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, EVAL_EVERY, LOG_EVERY, MAX_ITERS = 2, 5, 1, 200
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS = CFG["training"]["seeds"]
    EVAL_EVERY = CFG["evaluation"]["eval_every"]
    LOG_EVERY = CFG["evaluation"]["log_every"]
    MAX_ITERS = CFG["training"]["max_iters_per_task"]

SCENARIOS = CFG["data"]["scenarios"]
METHOD = "backprop"

BASE_PROTOCOL = replace(
    PROTOCOL,
    # -- data --
    img_size=CFG["data"]["img_size"],
    n_tasks=CFG["data"]["n_tasks"],
    classes_per_task=CFG["data"]["classes_per_task"],
    fashion=CFG["data"]["fashion"],
    # -- architecture --
    hidden=CFG["architecture"]["hidden"],
    n_layers=CFG["architecture"]["n_layers"],
    act=CFG["architecture"]["activation"],
    bias=CFG["architecture"]["bias"],
    init=CFG["architecture"]["init"],
    # -- output layer --
    loss=CFG["output_layer"]["loss"],
    target=CFG["output_layer"]["target"],
    mask=CFG["output_layer"]["mask"],
    reduction=CFG["output_layer"]["reduction"],
    # -- training --
    optimizer=CFG["training"]["optimizer"],
    batch=CFG["training"]["batch_size"],
    lr={METHOD: BACKPROP_LR},   # the one deviation -- see docstring
    stop_threshold=CFG["training"]["stop_threshold"],
    stop_patience=CFG["training"]["stop_patience"],
    max_iters_per_task=MAX_ITERS,
    seeds=SEEDS,
    # -- evaluation --
    eval_per_class=CFG["evaluation"]["eval_per_class"],
    eval_every=EVAL_EVERY,
    device=CFG["evaluation"]["device"],
)
THRESHOLD = CFG["training"]["stop_threshold"]


def _tag(scenario):
    return scenario + ("_SMOKE" if SMOKE else "")


def figure_path(scenario):
    return _figure_path(__file__, _tag(scenario))


def array_path(scenario):
    return _array_path(__file__, _tag(scenario))


def _310_array_path(scenario):
    """310's own saved array -- same seeds, different lr, for the paired comparison."""
    return _array_path(Path(__file__).parent / "310_forgetting_by_scenario.py", scenario)


# ---------------------------------------------------------------- one run, unclipped from step 0
def run_one(proto, seed, data):
    tasks = proto.tasks(seed)
    lmap = proto.label_map(tasks)
    classes = sorted({c for t in tasks for c in t})
    pos = {c: i for i, c in enumerate(classes)}

    train_step, predict = build(proto, METHOD, seed)

    report_x, report_y = data.report_eval
    a0 = _acc(predict(report_x), report_y, classes, lmap)
    init_row = [float(np.mean([a0[pos[c]] for c in t])) for t in tasks]

    out = run_classil(
        train_step, predict, tasks, data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, log_every=LOG_EVERY, device=proto.device,
        stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
        data_seed=seed, label_map=lmap,
    )
    steps = np.concatenate([[0], out["steps"]])
    curve = np.concatenate([[init_row], out["curves"]["argmax"]], axis=0)
    return steps, curve[:, 0], curve[:, 1], out["switches"][0], out["reached"]


def run_scenario(scenario):
    proto = replace(BASE_PROTOCOL, scenario=scenario)
    data = load(proto)
    seed_runs = []
    for seed in range(SEED_START, SEED_START + SEEDS):
        steps, t1, t2, switch0, reached = run_one(proto, seed, data)
        if not all(reached):
            print(f"  WARNING: {scenario} seed {seed}: hit the cap before reaching threshold")
        seed_runs.append((steps, t1, t2, switch0))
    return seed_runs


# ---------------------------------------------------------------- run or reload
if REPLOT and all(Path(array_path(s)).exists() for s in SCENARIOS):
    results = {}
    for s in SCENARIOS:
        z = np.load(array_path(s), allow_pickle=True)
        results[s] = [(z["steps"][i], z["t1"][i], z["t2"][i], int(z["switch0"][i]))
                      for i in range(z["t1"].shape[0])]
    print(f"--replot: redrawing from saved arrays, no training\n")
else:
    results = {}
    t0 = time.perf_counter()
    for scenario in SCENARIOS:
        seed_runs = run_scenario(scenario)
        results[scenario] = seed_runs
        print(f"  {scenario:10s} done   [{time.perf_counter() - t0:5.0f}s]")

# ---------------------------------------------------------------- per-scenario figure
for scenario in SCENARIOS:
    seed_runs = results[scenario]

    crossovers, finals = [], []
    for steps, t1, t2, switch0 in seed_runs:
        _, cx_height = crossover(steps, t1 * 100, t2 * 100, after=switch0)
        crossovers.append(cx_height)
        finals.append(t1[-1] * 100)
    crossovers = np.asarray(crossovers)
    finals = np.asarray(finals)

    FIG_W, FIG_H = 7.0, 6.8
    PAD_L, PAD_B, PAD_R, PAD_T = 0.9, 0.7, 0.15, 0.15
    HIST_THICK, GAP, S = 0.9, 0.05, 5.0

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([PAD_L / FIG_W, PAD_B / FIG_H, S / FIG_W, S / FIG_H])
    ax_top = fig.add_axes([PAD_L / FIG_W, (PAD_B + S + GAP) / FIG_H, S / FIG_W, HIST_THICK / FIG_H],
                          sharex=ax)
    ax_right = fig.add_axes([(PAD_L + S + GAP) / FIG_W, PAD_B / FIG_H, HIST_THICK / FIG_W, S / FIG_H],
                            sharey=ax)

    MARGIN = 5
    LO, HI = -MARGIN, 100 + MARGIN
    ax.plot([LO, HI], [LO, HI], color=CROSSOVER_COLOR, ls="--", lw=1.2)
    ax.text(20, 20, "crossover accuracy", color=CROSSOVER_COLOR, fontsize=8, rotation=45,
           ha="center", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.axhline(THRESHOLD * 100, color=FINAL_COLOR, ls="--", lw=1.2)
    ax.text(HI - 2, THRESHOLD * 100, f"task-2 stop ({THRESHOLD:.0%})",
           color=FINAL_COLOR, fontsize=8, ha="right", va="center",
           bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

    for steps, t1, t2, switch0 in seed_runs:
        idx = int(np.searchsorted(steps, switch0))
        ax.plot(t1[:idx + 1] * 100, t2[:idx + 1] * 100, color=TASK1_COLOR, lw=0.8, alpha=0.6)
        ax.plot(t1[idx:] * 100, t2[idx:] * 100, color=TASK2_COLOR, lw=0.8, alpha=0.6)
    ax.scatter(crossovers, crossovers, marker="s", color=CROSSOVER_COLOR, s=28, zorder=5)
    ax.scatter(finals, [s[2][-1] * 100 for s in seed_runs], marker="o", color=FINAL_COLOR,
              s=28, zorder=5)

    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI)
    ax.set_xlabel("task-1 accuracy (%)"); ax.set_ylabel("task-2 accuracy (%)")
    ax.grid(alpha=0.2)

    mean_final, mean_cross = float(np.mean(finals)), float(np.mean(crossovers))

    ax_top.hist(finals, bins=20, range=(0, 100), color=FINAL_COLOR, alpha=0.7)
    ax_top.axvline(mean_final, color="black", ls="--", lw=1.2)
    ax_top.text(mean_final + 1, ax_top.get_ylim()[1] * 0.5, f"mean={mean_final:.1f}%",
               color="black", fontsize=7, rotation=90, ha="left", va="center")
    ax_top.set_xlim(LO, HI)
    ax_top.tick_params(labelbottom=False, length=0)
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    ax_right.hist(crossovers, bins=20, range=(0, 100), color=CROSSOVER_COLOR, alpha=0.7,
                 orientation="horizontal")
    ax_right.axhline(mean_cross, color="black", ls="--", lw=1.2)
    ax_right.text(ax_right.get_xlim()[1] * 0.5, mean_cross + 1, f"mean={mean_cross:.1f}%",
                 color="black", fontsize=7, rotation=0, ha="center", va="bottom")
    ax_right.set_ylim(LO, HI)
    ax_right.tick_params(labelleft=False, length=0)
    ax_right.set_xticks([])
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    fig.savefig(figure_path(scenario), dpi=130, bbox_inches="tight")
    print(f"saved {figure_path(scenario)}")

    np.savez(array_path(scenario),
            steps=np.array([s[0] for s in seed_runs], dtype=object),
            t1=np.array([s[1] for s in seed_runs], dtype=object),
            t2=np.array([s[2] for s in seed_runs], dtype=object),
            switch0=np.array([s[3] for s in seed_runs]),
            crossovers=crossovers, finals=finals)
    print(f"saved {array_path(scenario)}")

# ---------------------------------------------------------------- paired check against 310
print("\npaired check: 314 (backprop lr=0.02) vs 310 (backprop lr=0.01), same seeds")
for scenario in SCENARIOS:
    p310 = _310_array_path(scenario)
    if not Path(p310).exists():
        print(f"  {scenario:10s} 310's array not found at {p310} -- run 310 first, skipping check")
        continue
    z310 = np.load(p310, allow_pickle=True)
    finals_310, cx_310 = z310["finals"], z310["crossovers"]
    finals_314 = np.array([s[1][-1] * 100 for s in results[scenario]])
    cx_314 = np.array([crossover(s[0], s[1] * 100, s[2] * 100, after=s[3])[1]
                       for s in results[scenario]])
    if finals_314.shape != finals_310.shape:
        print(f"  {scenario:10s} seed count mismatch (314={finals_314.shape[0]}, "
             f"310={finals_310.shape[0]}) -- skipping paired check")
        continue
    d_f, se_f, _ = paired_diff(finals_314, finals_310)
    w_f, l_f, t_f, p_f = paired_sign(finals_314, finals_310)
    d_c, se_c, _ = paired_diff(cx_314, cx_310)
    w_c, l_c, t_c, p_c = paired_sign(cx_314, cx_310, censored_is_best=True)
    print(f"  {scenario:10s} retention (lr0.02 - lr0.01) {d_f:+6.1f} +-{se_f:4.1f}"
         f"   sign {w_f}W-{l_f}L-{t_f}T p={p_f:.3f}")
    print(f"  {'':10s} crossover  (lr0.02 - lr0.01) {d_c:+6.1f} +-{se_c:4.1f}"
         f"   sign {w_c}W-{l_c}L-{t_c}T p={p_c:.3f}")
