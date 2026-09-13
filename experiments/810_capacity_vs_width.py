"""Does hidden width bound what either rule can learn, and does H=32 sit off that bound?

800-series TRAINING RUN. Arrays only; the figure is 902 (panel b), Methods M2.

WHY THIS REPLACES THE LEGACY ARRAY RATHER THAN REDRAWING IT. 100_capacity_vs_width ran
BACKPROP ONLY. The report uses its H=32 choice for both rules, so a backprop-only capacity
curve leaves the PC half of that claim untested -- if PC were capacity-bound at 32 where
backprop is not, every PC-vs-backprop number downstream would be confounded by width rather
than by the rule. This runs both rules on one grid so the claim covers both.

JOINT TRAINING, NOT SEQUENTIAL. All ten classes in a single block. The question is the
ceiling the architecture can reach, not what it retains, so there is no task switch here and
nothing to forget.

NO EARLY STOPPING, ON PURPOSE. stop_threshold=None and a flat 25k-iteration budget, because
the measurement IS the plateau. 100's own header records the trap this avoids: a capacity
sweep cut short produces a flat region that cannot be told apart from a capacity ceiling, and
an earlier 2500-iteration attempt would have picked H=16 for that reason. The last
PLATEAU_EVALS evaluations are averaged rather than taking the maximum, so a lucky eval does
not set the ceiling.

REDUCED GRID, BY DECISION. Six widths and five seeds against 100's seven and ten. The claim
this has to support is only that H=32 is off the bottleneck for both rules; resolving the
knee finely is not needed, and PC costs ~2.6x backprop per update once settling is counted.
Width 2 is dropped -- 100 already shows it far below everything else in both scenarios, and
it contributes nothing except axis range.

PROVENANCE
    Trains from scratch. config_800.yaml supplies every control parameter; the PC settle
    block is dt=0.4 adaptive, which is the correct setting at depth 1 (347 -- dt=0.4 is
    unstable only at depth >= 2, and this sweep never leaves depth 1).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
import numpy as np

from src.protocol import PROTOCOL, replace, load, build, array_path
from src.runner import run_classil

CFG = yaml.safe_load(open(Path(__file__).parent / "config_800.yaml"))

METHODS = ["backprop", "pc"]
SCENARIOS = CFG["data"]["scenarios"]
SEED_START = CFG["training"]["seed_start"]

WIDTHS = [4, 8, 16, 32, 64, 128]
SEEDS = 5
MAX_ITERS = 25000
EVAL_EVERY = 250
PLATEAU_EVALS = 6

SMOKE = "--smoke" in sys.argv
if SMOKE:
    WIDTHS, SEEDS, MAX_ITERS, EVAL_EVERY = [4, 32], 2, 400, 100
    print("--smoke: tiny budget, results are NOT meaningful\n")

# dt = 0.2, NOT the config default of 0.4. config_800.yaml's own comment reads "H=32, depth=1
# -- everywhere except a width/depth sweep", and this IS a width sweep: 346 found Class-IL
# H = 4 oscillates at dt = 0.4, sitting 3-7x above the true fixed point, so the PC arm at the
# narrow end would be measured under a settle that never converged. The project has already
# paid for this twice in re-run sweeps; 341/342 use 0.2 for the same reason.
SWEEP_DT = 0.2

PC_KW = dict(dt=SWEEP_DT,
             steps=CFG["predictive_coding"]["settle_step_cap"],
             stop_delta=CFG["predictive_coding"]["stop_delta"],
             stop_patience=CFG["predictive_coding"]["stop_patience"])


def base(scenario, width):
    return replace(
        PROTOCOL,
        img_size=CFG["data"]["img_size"], n_tasks=CFG["data"]["n_tasks"],
        classes_per_task=CFG["data"]["classes_per_task"], fashion=CFG["data"]["fashion"],
        hidden=width, n_layers=CFG["architecture"]["n_layers"],
        act=CFG["architecture"]["activation"], bias=CFG["architecture"]["bias"],
        init=CFG["architecture"]["init"],
        loss=CFG["output_layer"]["loss"], target=CFG["output_layer"]["target"],
        mask=CFG["output_layer"]["mask"], reduction=CFG["output_layer"]["reduction"],
        optimizer=CFG["training"]["optimizer"], batch=CFG["training"]["batch_size"],
        lr=dict(CFG["training"]["learning_rate"]),
        stop_threshold=None,                 # ceiling measurement -- see docstring
        max_iters_per_task=MAX_ITERS, eval_every=EVAL_EVERY, seeds=SEEDS,
        eval_per_class=CFG["evaluation"]["eval_per_class"],
        scenario=scenario, device=CFG["evaluation"]["device"])


def joint_run(proto, method, seed, data):
    """One joint-training run over all ten classes. Returns the plateau accuracy."""
    kw = dict(PC_KW) if method == "pc" else {}
    train_step, predict = build(proto, method, seed, **kw)
    pair = proto.tasks(seed)                     # same split at this seed for every width
    # label_map is required in Domain-IL, where ten classes share five output units; without
    # it make_target indexes past out_dim and the run dies on the first batch.
    out = run_classil(
        train_step, predict, [pair[0] + pair[1]], data.train, data.class_idx,
        report_eval=data.report_eval, stop_eval=data.stop_eval,
        max_iters_per_task=MAX_ITERS, batch=proto.batch, eval_every=EVAL_EVERY,
        device=proto.device, stop_threshold=None, data_seed=seed,
        label_map=proto.label_map(pair))
    curve = np.asarray(out["curves"]["argmax"], dtype=float)[:, 0] * 100.0
    return float(np.mean(curve[-PLATEAU_EVALS:])), curve


if __name__ == "__main__":
    t0 = time.perf_counter()
    acc = {s: {m: np.zeros((len(WIDTHS), SEEDS)) for m in METHODS} for s in SCENARIOS}
    curves = {}
    for scenario in SCENARIOS:
        for wi, width in enumerate(WIDTHS):
            proto = base(scenario, width)
            data = load(proto)
            for method in METHODS:
                for si in range(SEEDS):
                    seed = SEED_START + si
                    plateau, curve = joint_run(proto, method, seed, data)
                    acc[scenario][method][wi, si] = plateau
                    if si == 0:
                        curves[f"curve_{scenario}_{method}_{width}"] = curve
                m = acc[scenario][method][wi]
                print(f"  {scenario:10s} {method:9s} H={width:4d}  "
                      f"{m.mean():5.1f} +- {m.std(ddof=1)/np.sqrt(SEEDS):4.1f}%"
                      f"   [{time.perf_counter()-t0:6.0f}s]", flush=True)

    out = array_path(__file__) if not SMOKE else array_path(__file__, "SMOKE")
    np.savez_compressed(
        out, widths=np.array(WIDTHS), seeds=np.arange(SEED_START, SEED_START + SEEDS),
        methods=np.array(METHODS), max_iters=MAX_ITERS,
        **{f"acc_{s}_{m}": acc[s][m] for s in SCENARIOS for m in METHODS},
        **curves)
    print(f"saved {out}")
    print(f"done in {time.perf_counter()-t0:.0f}s")
