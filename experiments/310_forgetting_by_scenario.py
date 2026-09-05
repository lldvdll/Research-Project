"""Does backprop's sequential forgetting differ in shape between Class-IL and Domain-IL?

Report series 2, decade 310. Identical to 300 in every respect except ONE stated deviation:
seeds 10-19 instead of 0-9 -- an independent seed block, checking whether 300's story (and the
seed-9 switch0 outlier found in 302) replicates or was a one-off draw. Every control parameter
is read from experiments/config_300.yaml -- nothing here relies on src/protocol.py's own defaults.

ONE deviation from every earlier script in this project: no windowing, no common-window crop, no
relative-to-switch alignment. The whole trajectory is plotted from initialisation (step 0,
evaluated before any training) to the run's own end, per seed, unclipped.

Two report figures, saved separately so they can be placed independently in the LaTeX report:
    310_forgetting_by_scenario_class_il.png
    310_forgetting_by_scenario_domain_il.png

Each is a task-1-vs-task-2 accuracy trajectory (x = task-1 acc, y = task-2 acc): 10 seed lines,
coloured by which task is currently being trained (task-1 = orange, task-2 = blue). No mean line
-- with matched-competence stopping, seeds end at different absolute steps, and a step-aligned
mean across them was found to misrepresent the individual trajectories more than it clarified.

The dashed green diagonal marks x=y ("crossover accuracy"); each seed's crossing of it is marked
with a green square, and the distribution of crossing heights is a thin vertical histogram on the
right, with its mean marked. The dashed red horizontal line marks task-2's own stop threshold;
each seed's final point (final task-1 accuracy) is marked with a red circle, and that
distribution is a thin horizontal histogram on top, with its mean marked.

Logs every update (log_every=1) while the stop CRITERION stays on its original eval_every=10 /
patience=3 cadence, unchanged -- see src/runner.py's run_classil and config_300.yaml.
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
from src.metrics import crossover

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

SEED_START = 10   # the one stated deviation from 300

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
    lr={METHOD: CFG["training"]["learning_rate"][METHOD]},
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

    # Explicit axes geometry in inches, rather than gridspec ratios + sharex/sharey alone: the
    # main plot's box is forced EXACTLY square (so a data-diagonal really renders at 45 degrees)
    # and the two histogram axes are placed flush against its actual left/width or bottom/height,
    # so they align with it by construction instead of by hoping aspect and gridspec agree.
    FIG_W, FIG_H = 7.0, 6.8
    PAD_L, PAD_B, PAD_R, PAD_T = 0.9, 0.7, 0.15, 0.15
    HIST_THICK, GAP, S = 0.9, 0.05, 5.0

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([PAD_L / FIG_W, PAD_B / FIG_H, S / FIG_W, S / FIG_H])
    ax_top = fig.add_axes([PAD_L / FIG_W, (PAD_B + S + GAP) / FIG_H, S / FIG_W, HIST_THICK / FIG_H],
                          sharex=ax)
    ax_right = fig.add_axes([(PAD_L + S + GAP) / FIG_W, PAD_B / FIG_H, HIST_THICK / FIG_W, S / FIG_H],
                            sharey=ax)

    MARGIN = 5                              # 5% all round, so edge/corner points aren't clipped
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

    # Runs are ragged (matched-competence stopping ends each seed at a different absolute step),
    # so steps/t1/t2 are saved per-seed as object arrays rather than one shared grid.
    np.savez(array_path(scenario),
            steps=np.array([s[0] for s in seed_runs], dtype=object),
            t1=np.array([s[1] for s in seed_runs], dtype=object),
            t2=np.array([s[2] for s in seed_runs], dtype=object),
            switch0=np.array([s[3] for s in seed_runs]),
            crossovers=crossovers, finals=finals)
    print(f"saved {array_path(scenario)}")
