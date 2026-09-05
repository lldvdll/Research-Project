"""For each seed's task-1 digit split, is being slow to reach 90% a property of the digit set
itself, or of that seed's own particular weight initialisation?

Report series 2, decade 300, third script. Follows directly from discussing 300's Class-IL
outlier (seed 9, final=32.3%): its task-1 phase took 3,290 steps against everyone else's
280-670, in BOTH scenarios (task splits are scenario-independent, `Protocol.tasks(seed)` doesn't
read scenario at all), which rules out task-2 interaction as the cause -- something makes THIS
DIGIT SET slow before task-2 ever starts. But `build(proto, method, seed)` and the split
(`proto.tasks(seed)`) share one seed value in every other script, so "this digit set is hard"
and "this seed's init happened to be unusual" are confounded in 300's own data and cannot be
told apart there.

This decouples them directly: train task-1 ALONE (no task-2 at all -- a single-element tasks
list passed straight to run_classil, so the stopping definition is identical to 300's own
switch0) on each of the 10 splits, under 5 FRESH init seeds unrelated to the split's own seed,
plus the split's own ORIGINAL (paired) seed for comparison. If split 9 is slow under the fresh
inits too, the digit set itself is hard. If only its own original init is slow while fresh inits
on the same split look ordinary, the init was the confound.

Uses Class-IL's 10-unit output structure arbitrarily (a task-1-alone run has no second task to
share units with, so scenario only affects 5 idle, never-trained output units -- `active`
restricts gradient to task-1's own 5 units either way, so this shouldn't bias the comparison).

One figure: x = split index (0-9), y = steps to reach 90%. 5 fresh-init draws per split (small
grey dots) plus the split's own paired-seed result (red diamond) at each x position.
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
from src.runner import run_classil

SMOKE = "--smoke" in sys.argv
REPLOT = "--replot" in sys.argv

OUTLIER_SEED = 9
FRESH_SEED_BASE = 1000     # fresh init seeds are FRESH_SEED_BASE + split*10 + k -- outside 0-9


def _tag(suffix=""):
    return (suffix + "_SMOKE").lstrip("_") if SMOKE else suffix


def figure_path():
    return _figure_path(__file__, _tag())


def array_path():
    return _array_path(__file__, _tag())


with open(Path(__file__).parent / "config_300.yaml") as f:
    CFG = yaml.safe_load(f)

if SMOKE:
    SEEDS, N_FRESH, MAX_ITERS = 3, 2, 200
    print("--smoke: tiny budget, results are NOT meaningful\n")
else:
    SEEDS, N_FRESH = 10, 5
    MAX_ITERS = CFG["training"]["max_iters_per_task"]

METHOD = "backprop"
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
    lr={METHOD: CFG["training"]["learning_rate"][METHOD]},
    stop_threshold=CFG["training"]["stop_threshold"],
    stop_patience=CFG["training"]["stop_patience"],
    max_iters_per_task=MAX_ITERS, seeds=SEEDS,
    eval_per_class=CFG["evaluation"]["eval_per_class"], eval_every=CFG["evaluation"]["eval_every"],
    device=CFG["evaluation"]["device"],
    scenario="class_il",
)


def steps_to_threshold(proto, task1, init_seed, data):
    train_step, predict = build(proto, METHOD, init_seed)
    out = run_classil(
        train_step, predict, [task1], data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=proto.max_iters_per_task, batch=proto.batch,
        eval_every=proto.eval_every, device=proto.device,
        stop_threshold=proto.stop_threshold, stop_patience=proto.stop_patience,
        data_seed=init_seed,
    )
    return out["switches"][0], out["reached"][0]


if REPLOT and Path(array_path()).exists():
    z = np.load(array_path(), allow_pickle=True)
    paired_steps, fresh_steps = z["paired_steps"], z["fresh_steps"]
    print(f"--replot: redrawing from {array_path()}, no training\n")
else:
    data = load(BASE_PROTOCOL)
    paired_steps = np.full(SEEDS, np.nan)
    fresh_steps = np.full((SEEDS, N_FRESH), np.nan)
    t0 = time.perf_counter()
    for split in range(SEEDS):
        task1 = BASE_PROTOCOL.tasks(split)[0]

        s, reached = steps_to_threshold(BASE_PROTOCOL, task1, split, data)
        paired_steps[split] = s
        if not reached:
            print(f"  WARNING: split {split} paired seed: hit the cap")

        for k in range(N_FRESH):
            fresh_seed = FRESH_SEED_BASE + split * 10 + k
            s, reached = steps_to_threshold(BASE_PROTOCOL, task1, fresh_seed, data)
            fresh_steps[split, k] = s
            if not reached:
                print(f"  WARNING: split {split} fresh seed {k}: hit the cap")
        print(f"  split {split}   done   [{time.perf_counter() - t0:5.0f}s]")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7.5, 5))
xpos = np.arange(SEEDS)
for split in range(SEEDS):
    ax.scatter([xpos[split]] * fresh_steps.shape[1], fresh_steps[split], color="0.6", s=20,
              zorder=3, label="fresh init" if split == 0 else None)
ax.scatter(xpos, paired_steps, marker="D", color="tab:red", s=50, zorder=4,
          label="original (paired) init")
if OUTLIER_SEED < SEEDS:
    ax.axvspan(OUTLIER_SEED - 0.4, OUTLIER_SEED + 0.4, color="tab:red", alpha=0.08)
ax.set_xticks(xpos)
ax.set_xlabel("task split (= seed whose task-1 digits were used)")
ax.set_ylabel("steps to reach 90% (task-1 alone)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.2)
fig.tight_layout()

fig.savefig(figure_path(), dpi=130, bbox_inches="tight")
print(f"saved {figure_path()}")

np.savez(array_path(), paired_steps=paired_steps, fresh_steps=fresh_steps)
print(f"saved {array_path()}")
