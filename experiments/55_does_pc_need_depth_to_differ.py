"""Does predictive coding stop looking like backprop once the network has depth?

THE CLAIM THIS TESTS, WHICH CAME OUT OF SCRIPT 54
    54 measured each rule's actual weight update against the update backprop would make from
    identical weights, and found PC's hidden-layer update 0.985 aligned with backprop's -- flat
    across training and flat across settling amount. The proposed explanation was structural:

        with ONE hidden layer, PC's W1 update is x0^T e1, and the relaxation displacement e1 is
        driven by (e_out W2^T) * f'(x1), which IS backprop's hidden error. There is no
        intermediate layer for prospective configuration to reconfigure, so PC lands on
        backprop's direction and differs only in scale.

    That was an inference from a single-layer measurement, and it makes a sharp prediction:
    ADD A HIDDEN LAYER AND THE COSINE SHOULD FALL. If it does not, the explanation is wrong,
    PC is close to backprop for some other reason, and script 52's null result needs a
    different account.

    Testing it converts that inference into a measurement, which is the point.

WHY THIS MATTERS BEYOND TIDINESS
    If PC only differs from backprop at depth, then every retention result in this project so
    far has been collected at the one depth where PC has no room to differ, and the whole
    comparison has been run in the least informative place available. The remedy is cheap --
    PC costs about 8.5 ms an update -- which is why this is worth settling before more
    single-layer comparisons are run.

WHAT IS MEASURED
    The same instrument as 54, at n_layers = 1, 2, 3:
      cos(dW_l, backprop's dW_l) for every weight matrix, from the rule's own weights on its own
      trajectory, with backprop's update obtained by copying those weights into a backprop
      harness so nothing is re-derived by hand.
    EqProp is dropped. It is 350x backprop per update, 54 already showed it is genuinely
    different (cos 0.316), and the question here is specifically about PC's mechanism.

    Alongside, and as illustration rather than as the result: task-1 and task-2 accuracy curves
    for backprop and PC OVERLAID at each depth, so any difference in the shape of learning or
    forgetting is visible directly rather than inferred from a summary number.

A LIMITATION, STATED RATHER THAN HIDDEN
    The learning rates come from script 51, which calibrated them at ONE hidden layer. They are
    not re-calibrated per depth. The cosine is scale-invariant so the headline result does not
    depend on this, but the accuracy curves do -- steps-to-threshold is printed per depth so any
    mismatch is visible rather than assumed away. Task 1 stops on ACCURACY so that every depth
    enters task 2 equally competent whatever its learning rate; task 2 then runs a fixed budget
    so the panels share an x-axis. That mix is acceptable here because retention is not the
    result; in scripts 52 and 53, where it is, both tasks stop the same way.

READINGS COMMITTED BEFORE RUNNING
    If cos(dW1) falls clearly below its single-layer value as depth increases, 54's explanation
    holds, and the rule comparison should be re-run at depth before any conclusion is drawn
    about whether PC helps.
    If it stays near 0.98 at every depth, the explanation is wrong. PC would then be close to
    backprop for a reason we have not identified, and finding that reason becomes the priority.
    If the deepest layers diverge but W1 does not, prospective configuration is reconfiguring
    the middle of the network while leaving the input mapping alone -- which would predict no
    retention benefit even at depth, since drift in W1 is what damages task 1.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # make `src` importable from anywhere

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.protocol import PROTOCOL, load, run, build, replace, figure_path as _figure_path, array_path as _array_path
from src.plotting import plot_retention_curve

# --smoke writes NOTHING. Its outputs would otherwise overwrite the real .npz that later
# scripts read their settings from -- script 53 takes its task-2 threshold from 52's measured
# ceilings, and a smoke-sized 52 silently poisons it.
SMOKE = "--smoke" in sys.argv


def _tag(f, suffix):
    """Suffix only THIS script's own outputs, never the .npz files it reads from others."""
    own = Path(f).resolve() == Path(__file__).resolve()
    return (suffix + "_SMOKE").lstrip("_") if (SMOKE and own) else suffix


def figure_path(f, suffix=""):      # noqa: F811  - shadows the import, on purpose
    return _figure_path(f, _tag(f, suffix))


def array_path(f, suffix=""):       # noqa: F811
    return _array_path(f, _tag(f, suffix))

# ---------------------------------------------------------------- settings
DEPTHS = [1, 2, 3]
METHODS = ["backprop", "pc"]
SEEDS = 3
EVAL_EVERY = 10
CHECKPOINTS = [0, 100, 250, 420]        # updates into task 1, as in script 54
PROBE_BATCH = 32
T1_STOP = 0.90                          # task 1 on accuracy, so every depth enters task 2 equal
ITERS_TASK2 = 630                       # fixed, so the panels share an x-axis
MAX_T1 = 1500
COLORS = {"backprop": "tab:gray", "pc": "tab:red"}
DEPTH_LS = {1: "-", 2: "--", 3: ":"}
TASK_COLORS = ["tab:orange", "tab:blue"]

HIDDEN = int(np.load(array_path(str(ROOT / "experiments" /
                                    "41_capacity_vs_hidden_width.py")))["chosen"])
z51 = np.load(array_path(str(ROOT / "experiments" /
                             "51_matching_the_rules_on_learning_speed.py")))
LR = {m: float(v) for m, v in zip(z51["methods"], z51["lr"])}
PC_STEPS = int(z51["pc_steps"])

if "--smoke" in sys.argv:
    DEPTHS, SEEDS, CHECKPOINTS, ITERS_TASK2, MAX_T1 = [1, 2], 1, [0, 20], 40, 60
    print("--smoke: tiny budget, results are NOT meaningful\n")

print(f"H = {HIDDEN} per layer | depths {DEPTHS} | pc settles {PC_STEPS} steps | {SEEDS} seeds")
print(f"  learning rates from 51 (calibrated at depth 1): "
      + ", ".join(f"{m} {LR[m]:g}" for m in METHODS) + "\n")


def settle_kw(m):
    return dict(steps=PC_STEPS) if m == "pc" else {}


def one_update(proto, rule, weights, x, y, active, seed):
    """The weight change `rule` makes from exactly `weights`, on exactly this batch."""
    h = {}
    train_step, _ = build(proto, rule, seed, handle=h, **settle_kw(rule))
    named = h["params"].named()
    for k, v in weights.items():
        named[k].data.copy_(v)
    before = {k: named[k].detach().clone() for k in weights}
    train_step(x, y, active=active)
    return {k: (named[k].detach() - before[k]) for k in weights}


# ---------------------------------------------------------------- run
REPLOT = "--replot" in sys.argv and Path(array_path(__file__)).exists()

if REPLOT:
    z = np.load(array_path(__file__), allow_pickle=True)
    cos = {d: z[f"cos_d{d}"] for d in DEPTHS}
    curves = {(m, d): z[f"curve_{m}_d{d}"] for m in METHODS for d in DEPTHS}
    t1_steps = {(m, d): z[f"t1steps_{m}_d{d}"] for m in METHODS for d in DEPTHS}
    steps = z["steps"]
    print("--replot: redrawing from saved arrays, no training\n")
else:
    data = load(replace(PROTOCOL, hidden=HIDDEN, scenario="domain_il"))
    cos = {d: np.full((SEEDS, len(CHECKPOINTS), d + 1), np.nan) for d in DEPTHS}
    curves, t1_steps = {}, {}
    t0 = time.perf_counter()

    # ---- part A: how different is PC's update from backprop's, at each depth?
    for d in DEPTHS:
        proto = replace(PROTOCOL, hidden=HIDDEN, n_layers=d, scenario="domain_il",
                        stop_threshold=None, max_iters_per_task=max(CHECKPOINTS),
                        eval_every=200, seeds=SEEDS, lr={"pc": LR["pc"]})
        for seed in range(SEEDS):
            pair = proto.tasks(seed)
            lmap = proto.label_map([pair[0]])
            active = sorted({lmap[int(c)] for c in pair[0]})
            handle = {}

            g = torch.Generator().manual_seed(3000 + seed)
            pool = torch.cat([torch.as_tensor(data.class_idx[c]) for c in pair[0]])
            pick = pool[torch.randperm(len(pool), generator=g)[:PROBE_BATCH]]
            px = torch.stack([data.train[i][0] for i in pick.tolist()])
            py = torch.tensor([lmap[int(data.train[i][1])] for i in pick.tolist()])

            def measure(ci):
                w = {k: v.detach().clone() for k, v in handle["params"].named().items()
                     if v is not None}
                d_pc = one_update(proto, "pc", w, px, py, active, seed)
                d_bp = one_update(replace(proto, lr={"backprop": LR["backprop"]}),
                                  "backprop", w, px, py, active, seed)
                for li in range(d + 1):
                    k = f"W{li + 1}"
                    a, b = d_pc[k].flatten(), d_bp[k].flatten()
                    cos[d][seed, ci, li] = float(
                        torch.nn.functional.cosine_similarity(a, b, dim=0)) \
                        if a.norm() > 0 and b.norm() > 0 else 0.0

            n = [0]

            def wrap(train_step):
                measure(0)

                def wrapped(x, y, active=None):
                    train_step(x, y, active=active)
                    n[0] += 1
                    if n[0] in CHECKPOINTS:
                        measure(CHECKPOINTS.index(n[0]))
                return wrapped

            run(proto, "pc", seed, data=data, handle=handle, tasks=[pair[0]], wrap=wrap,
                **settle_kw("pc"))
        print(f"  depth {d}: cos(dW, backprop) per layer at the last checkpoint = "
              + ", ".join(f"W{i+1} {np.nanmean(cos[d][:, -1, i]):.3f}" for i in range(d + 1))
              + f"   [{time.perf_counter()-t0:5.0f}s]")

    # ---- part B: what the learning actually looks like, both rules, every depth
    for d in DEPTHS:
        for m in METHODS:
            proto = replace(PROTOCOL, hidden=HIDDEN, n_layers=d, scenario="domain_il",
                            stop_threshold=[T1_STOP, None],
                            max_iters_per_task=[MAX_T1, ITERS_TASK2],
                            eval_every=EVAL_EVERY, seeds=SEEDS, lr={m: LR[m]})
            rows, t1s = [], []
            for seed in range(SEEDS):
                out = run(proto, m, seed, data=data, **settle_kw(m))
                sw = out["switches"][0]
                i_sw = int(np.argmin(np.abs(np.asarray(out["steps"]) - sw)))
                # switch-aligned: every run's task-2 block starts at index 0
                rows.append(out["curves"]["argmax"][i_sw:i_sw + ITERS_TASK2 // EVAL_EVERY])
                t1s.append(sw)
            k = min(len(r) for r in rows)
            curves[(m, d)] = np.stack([r[:k] for r in rows])
            t1_steps[(m, d)] = np.asarray(t1s, dtype=float)
            print(f"  depth {d} {m:9s} task 1 reached {T1_STOP:.0%} in "
                  f"{np.mean(t1s):5.0f} updates   task 1 kept "
                  f"{curves[(m, d)][:, -1, 0].mean()*100:5.1f}%   task 2 "
                  f"{curves[(m, d)][:, -1, 1].mean()*100:5.1f}%"
                  f"   [{time.perf_counter()-t0:5.0f}s]")
    steps = np.arange(curves[(METHODS[0], DEPTHS[0])].shape[1]) * EVAL_EVERY

# ---------------------------------------------------------------- readings
print(f"\n  cos(PC's update, backprop's update), by depth and layer, at the task switch\n")
print(f"  {'depth':>6s} " + " ".join(f"{'W'+str(i+1):>8s}" for i in range(max(DEPTHS) + 1))
      + f" {'min':>8s}")
for d in DEPTHS:
    v = [np.nanmean(cos[d][:, -1, i]) for i in range(d + 1)]
    print(f"  {d:6d} " + " ".join(f"{x:8.3f}" for x in v)
          + "         " * (max(DEPTHS) + 1 - len(v)) + f" {min(v):8.3f}")

base_cos = np.nanmean(cos[DEPTHS[0]][:, -1, 0])
deep_cos = np.nanmean(cos[DEPTHS[-1]][:, -1, 0])
print(f"\n  VERDICT on script 54's explanation")
print(f"    cos(dW1) at depth {DEPTHS[0]}: {base_cos:.3f}   at depth {DEPTHS[-1]}: {deep_cos:.3f}"
      f"   change {deep_cos - base_cos:+.3f}")
print("    " + ("depth DOES separate PC from backprop -- the single-layer comparisons were run "
                "where PC has\n    no room to differ, and the rule comparison should be repeated "
                "at depth"
                if base_cos - deep_cos > 0.05 else
                "depth does NOT separate them. 54's structural explanation is WRONG, and PC's "
                "closeness to\n    backprop needs a different account before more is built on "
                "it."))
inner = [np.nanmean(cos[d][:, -1, 1:d]) for d in DEPTHS if d >= 2]
if inner and np.nanmin(inner) < base_cos - 0.05 and deep_cos > base_cos - 0.05:
    print("    NOTE: the INNER layers diverge while W1 does not. PC would then be reconfiguring "
          "the middle\n    of the network and leaving the input mapping alone -- which predicts "
          "no retention benefit\n    even at depth, since drift in W1 is what damages task 1.")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(7.2, 5.0))
for li in range(max(DEPTHS) + 1):
    xs = [d for d in DEPTHS if li <= d]
    ys = [np.nanmean(cos[d][:, -1, li]) for d in xs]
    es = [np.nanstd(cos[d][:, -1, li], ddof=1) / np.sqrt(SEEDS) for d in xs]
    if xs:
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=2, label=f"W{li + 1}")
ax.axhline(1.0, color="k", lw=1)
ax.set_xticks(DEPTHS)
ax.set_xlabel("hidden layers")
ax.set_ylabel("cos(PC's update, backprop's update)")
ax.set_ylim(-0.1, 1.1)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, title="weight matrix", title_fontsize=8)
ax.set_title("1.0 means PC is doing what backprop does", fontsize=10)
fig.tight_layout()
fig.savefig(figure_path(__file__), dpi=120, bbox_inches="tight")
print(f"\nsaved {figure_path(__file__)}")

# supporting: the curves themselves, both rules overlaid, one panel per depth
figc, axc = plt.subplots(1, len(DEPTHS), figsize=(5.2 * len(DEPTHS), 4.4),
                         sharex=True, sharey=True)
axc = np.atleast_1d(axc)
for a, d in zip(axc, DEPTHS):
    for m in METHODS:
        A = curves[(m, d)] * 100
        for t, tc in enumerate(TASK_COLORS):
            a.plot(steps, np.nanmean(A[:, :, t], axis=0), lw=2.2, color=COLORS[m],
                   ls="-" if t == 0 else "--",
                   label=f"{m}, task {t + 1}" if d == DEPTHS[0] else None)
    a.set_title(f"{d} hidden layer{'s' if d > 1 else ''}")
    a.set_xlabel("updates since the task switch")
    a.set_ylim(-2, 103)
    a.grid(alpha=0.25)
axc[0].set_ylabel("accuracy (%)")
axc[0].legend(fontsize=8, loc="center right")
figc.tight_layout()
figc.savefig(figure_path(__file__, "curves"), dpi=120, bbox_inches="tight")
print(f"saved {figure_path(__file__, 'curves')}")

# supporting: retention against task-2 progress, every rule and depth on one axes
segs = {f"{m} d{d}": [curves[(m, d)][i] for i in range(curves[(m, d)].shape[0])]
        for m in METHODS for d in DEPTHS}
plot_retention_curve(
    segs, list(segs), figure_path(__file__, "retention"),
    colors={f"{m} d{d}": COLORS[m] for m in METHODS for d in DEPTHS},
    chance=1.0 / PROTOCOL.classes_per_task,
    title="Task 1 retained against task-2 progress, by rule and depth.",
)

np.savez(array_path(__file__), depths=np.asarray(DEPTHS), hidden=HIDDEN,
         checkpoints=np.asarray(CHECKPOINTS), steps=steps,
         **{f"cos_d{d}": cos[d] for d in DEPTHS},
         **{f"curve_{m}_d{d}": curves[(m, d)] for m in METHODS for d in DEPTHS},
         **{f"t1steps_{m}_d{d}": t1_steps[(m, d)] for m in METHODS for d in DEPTHS})
print(f"saved {array_path(__file__)}")
