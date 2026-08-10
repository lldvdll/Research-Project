"""22_does_hidden_drift_matter  (CORRECTED)

ONE QUESTION
    Does it make any difference whether the network keeps its internal representation?

WHAT WAS WRONG IN THE FIRST VERSION
    Task 1 and task 2 were run in two separate calls to run_classil. The second call was given
    a ONE-element task list, so its `curves` array had a single column -- task 2's accuracy --
    which the script then plotted with a "task 1" label. The top row of the old figure was
    task-2 learning curves. The printed table and the bar chart were computed separately and
    were correct, so the CONCLUSION did not change; only the picture was wrong.

    Fix: one run_classil call covering both tasks, using a per-task budget list. Task 1 trains
    to a competence threshold; task 2 gets a fixed number of updates, identical in every
    condition, so no condition can appear to "forget less" merely by training less.

WHAT IS BEING COMPARED
    Backprop only. 2 tasks x 5 classes, MNIST 14x14. During TASK 2 ONLY:

      normal      nothing held fixed
      freeze W1   input->hidden weights held fixed   -> the hidden code CANNOT change
      freeze W2   hidden->output weights held fixed  -> the output readout CANNOT change
      masked      absent classes contribute no gradient -> the downward push on classes not in
                  the current task is switched off

    Run at hidden = 16 and hidden = 64.

HOW IT ANSWERS THE QUESTION
    "freeze W1 minus normal" is a CAUSAL upper bound. If eliminating hidden-layer change
    entirely gives back 3 percentage points, then no method that works by reducing
    hidden-layer change -- prospective configuration included -- can give back more than 3
    points at that width.

RESULT FROM THE FIRST RUN (not changed by this fix)
      hidden=16:  freeze W1 +3.3   freeze W2 +0.1   masked +30.2
      hidden=64:  freeze W1 +0.1   freeze W2 +0.0   masked +46.9

    Note especially that freezing W2 changes nothing while task 2 still reaches ~90%. With the
    readout pinned, the network learns task 2 by rewriting the hidden code instead. Blocking
    one route diverts the change through the other; only switching off the downward push on
    absent classes helps.

FIGURE
    Row 1: task-1 accuracy through task 2 -- now genuinely task 1.
    Row 2: task-2 accuracy over the same window, so it is visible that every condition really
           did learn task 2. Without this, "freeze W1 forgot less" could just mean "learnt
           less", which is the confound this whole design exists to remove.
    Right: a small table of final values. Three of four conditions sit at zero, so a bar chart
           is the wrong instrument -- it draws three empty boxes and says nothing. The table
           states the numbers plainly and the curves carry the shape information.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data import load_mnist, class_indices, make_eval_split
from src.model import Arch, Objective, replace
from src.methods import build_method
from src.runner import run_classil
from src.metrics import half_life, area_retained
from src.probes import restricted_argmax_fn

# ============================ constants ============================
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR  = ROOT / "data"
FIG       = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE  = 14
BASE_SEED = 0
N_RUNS    = 10

WIDTHS            = [16, 64]
CLASSES_PER_TASK  = 5
TASK1_THRESHOLD   = 0.70
STOP_PATIENCE     = 3
TASK1_MAX_ITERS   = 800
TASK2_ITERS       = 300
BATCH             = 32
EVAL_EVERY        = 5
EVAL_PER_CLASS    = 100

ARCH_BASE = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=64, out_dim=10,
                 act="tanh", bias=True, init="scaled_normal")
OBJ_FULL   = Objective(loss="mse", target="onehot", mask=False)
OBJ_MASKED = replace(OBJ_FULL, mask=True)
BP_LR = 0.1     # exp 24: backprop's best under linear+SE 1/0. Was 0.05, worth about 1 point.

CONDITIONS = {
    "normal":    (OBJ_FULL,   set()),
    "freeze W1": (OBJ_FULL,   {"W1", "b1"}),
    "freeze W2": (OBJ_FULL,   {"W2", "b2"}),
    "masked":    (OBJ_MASKED, set()),
}
COLOURS = {"normal": "tab:red", "freeze W1": "tab:blue",
           "freeze W2": "tab:green", "masked": "tab:purple"}
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)

C1 = {(w, c): [] for w in WIDTHS for c in CONDITIONS}
C2 = {(w, c): [] for w in WIDTHS for c in CONDITIONS}
peak = {(w, c): [] for w in WIDTHS for c in CONDITIONS}
hl = {(w, c): [] for w in WIDTHS for c in CONDITIONS}
ar = {(w, c): [] for w in WIDTHS for c in CONDITIONS}
missed = {(w, c): 0 for w in WIDTHS for c in CONDITIONS}
t0 = time.time()

for width in WIDTHS:
    arch = replace(ARCH_BASE, hidden=width)
    for run in range(N_RUNS):
        seed = BASE_SEED + run
        d = np.random.default_rng(seed).permutation(10).tolist()
        k = CLASSES_PER_TASK
        tasks = [sorted(d[:k]), sorted(d[k:2 * k])]
        classes = sorted({c for t in tasks for c in t})
        stop_ev, rep_ev = make_eval_split(test, classes, EVAL_PER_CLASS, DEVICE, seed=seed)

        for cname, (obj, frozen_names) in CONDITIONS.items():
            handle = {}
            step_fn, predict = build_method("backprop", in_dim=arch.in_dim, hidden=width,
                                            out_dim=arch.out_dim, arch=arch, obj=obj,
                                            lr=BP_LR, seed=seed, device=DEVICE, handle=handle)
            params, freeze = handle["params"], handle["freeze"]
            snap = {}

            def on_task_end(ti, step, _p=params, _fz=freeze, _fn=frozen_names, _s=snap):
                if ti == 0:
                    _s["W1"] = _p.W1.detach().clone()
                    _s["W2"] = _p.W2.detach().clone()
                    _fz.update(_fn)

            out = run_classil(
                step_fn, predict, tasks, train, cidx,
                report_eval=rep_ev, stop_eval=stop_ev,
                readouts={"a": restricted_argmax_fn(predict, classes)},
                max_iters_per_task=[TASK1_MAX_ITERS, TASK2_ITERS],
                stop_threshold=[TASK1_THRESHOLD, None],
                stop_patience=STOP_PATIENCE, batch=BATCH,
                eval_every=EVAL_EVERY, device=DEVICE, data_seed=seed,
                on_task_end=on_task_end)

            steps, cur, sw = out["steps"], out["curves"]["a"], out["switches"][0]
            missed[(width, cname)] += int(not out["reached"][0])
            pk = float(cur[steps <= sw][-1, 0])
            peak[(width, cname)].append(pk)
            after = cur[steps > sw]
            # Index 0 is the PRE-SWITCH PEAK. Without it the curve starts at the first eval
            # after the switch (EVAL_EVERY updates in), by which point an unmasked run has
            # already fallen a long way -- so panels appeared to start at different heights.
            pk2 = float(cur[steps <= sw][-1, 1])
            C1[(width, cname)].append(np.concatenate([[pk], after[:, 0]]))
            C2[(width, cname)].append(np.concatenate([[pk2], after[:, 1]]))
            hl[(width, cname)].append(half_life(steps, cur[:, 0], after=sw, peak=pk))
            ar[(width, cname)].append(area_retained(steps, cur[:, 0], after=sw, peak=pk))

            if "W1" in frozen_names:
                assert torch.equal(params.W1, snap["W1"]), "W1 moved despite being frozen"
            if "W2" in frozen_names:
                assert torch.equal(params.W2, snap["W2"]), "W2 moved despite being frozen"

    print(f"h={width:>4} | " + " | ".join(
        f"{c}: t1 {np.mean([x[-1] for x in C1[(width, c)]]) * 100:5.1f}%"
        f" t2 {np.mean([x[-1] for x in C2[(width, c)]]) * 100:5.1f}%" for c in CONDITIONS)
        + f" | {time.time() - t0:5.0f}s")

# ------------------------------- table -------------------------------
print("\n" + "=" * 104)
print(f"After {TASK2_ITERS} fixed updates on task 2  ({N_RUNS} runs, task 1 trained to "
      f"{TASK1_THRESHOLD:.0%}, lr {BP_LR})")
print(f"{'hidden':>7}{'condition':>12}{'t1 peak':>10}{'t1 final':>13}{'recovered':>11}"
      f"{'t2 final':>10}{'half-life':>11}{'area kept':>11}")
for w in WIDTHS:
    base = np.mean([x[-1] for x in C1[(w, "normal")]]) * 100
    for c in CONDITIONS:
        f1 = np.array([x[-1] for x in C1[(w, c)]]) * 100
        f2 = np.array([x[-1] for x in C2[(w, c)]]) * 100
        # Runs that never halved are CENSORED at the window end, not dropped. Dropping them
        # biases the mean downward and hides the best-performing condition entirely.
        h = [TASK2_ITERS if v is None else v for v in hl[(w, c)]]
        ncens = sum(v is None for v in hl[(w, c)])
        rec = "" if c == "normal" else f"{f1.mean() - base:+11.1f}"
        print(f"{w:>7}{c:>12}{np.mean(peak[(w, c)]) * 100:>10.1f}"
              f"{f1.mean():>9.1f}+/-{f1.std():<3.1f}{rec}"
              f"{f2.mean():>10.1f}{np.mean(h):>10.0f}{'*' if ncens else ' '}"
              f"{np.mean(ar[(w, c)]):>11.2f}")
        if missed[(w, c)]:
            print(f"{'':>19}WARNING: task 1 missed the threshold in {missed[(w, c)]} run(s)")
        if np.mean(peak[(w, c)]) < TASK1_THRESHOLD:
            print(f"{'':>19}CAVEAT: mean peak {np.mean(peak[(w, c)]) * 100:.1f}% is below the "
                  f"{TASK1_THRESHOLD:.0%} criterion -- retention here is not comparable")

print("\nREAD IT LIKE THIS")
print("  'recovered' = percentage points of task-1 accuracy the intervention gives back,")
print("     compared with doing nothing. 'freeze W1' recovered is the CAUSAL CEILING on what")
print("     any method that works by stabilising the hidden layer could win at that width.")
print("  't2 final' should be similar across conditions, or the comparison is unfair.")
print("  '*' after half-life = some runs never halved and were censored at the window end.")
print("  'half-life' = updates for task-1 accuracy to fall to half its pre-switch peak. Use")
print("     this, not the final value, whenever every condition ends near zero.")

# ------------------------------- figure -------------------------------
ncol = len(WIDTHS) + 1
fig, axes = plt.subplots(2, ncol, figsize=(5.8 * len(WIDTHS) + 4.6, 8),
                         gridspec_kw={"width_ratios": [1] * len(WIDTHS) + [0.8]})
for j, w in enumerate(WIDTHS):
    n = min(min(len(x) for x in C1[(w, c)]) for c in CONDITIONS)
    xs = np.arange(n) * EVAL_EVERY
    for row, (store, lab) in enumerate([(C1, "task 1 (old)"), (C2, "task 2 (new)")]):
        ax = axes[row, j]
        for c in CONDITIONS:
            A = np.stack([x[:n] for x in store[(w, c)]]) * 100
            ax.plot(xs, A.mean(0), color=COLOURS[c], lw=2.2, label=c)
            ax.fill_between(xs, A.mean(0) - A.std(0), A.mean(0) + A.std(0),
                            color=COLOURS[c], alpha=0.12)
        ax.axhline(100 / (2 * CLASSES_PER_TASK), color="k", lw=0.8, ls="-.", alpha=0.5)
        if row == 0:
            ax.axhline(np.mean(peak[(w, "normal")]) * 100, color="gray", ls=":", lw=1.2)
        ax.set_ylim(-2, 103); ax.grid(alpha=0.2)
        ax.set_title(f"hidden = {w}:  {lab}")
        ax.set_xlabel("updates into task 2")
        if j == 0:
            ax.set_ylabel("accuracy (%)")
            ax.legend(fontsize=8, loc="center right")

for row, w in enumerate(WIDTHS[:2]):
    ax = axes[row, ncol - 1]; ax.axis("off")
    base = np.mean([x[-1] for x in C1[(w, "normal")]]) * 100
    rows = [[c, f"{np.mean(peak[(w, c)]) * 100:.1f}",
             f"{np.mean([x[-1] for x in C1[(w, c)]]) * 100:.1f}",
             "-" if c == "normal" else f"{np.mean([x[-1] for x in C1[(w, c)]]) * 100 - base:+.1f}",
             f"{np.mean([x[-1] for x in C2[(w, c)]]) * 100:.1f}"] for c in CONDITIONS]
    tb = ax.table(cellText=rows,
                  colLabels=["condition", "t1 peak", "t1 final", "gained", "t2 final"],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.6)
    ax.set_title(f"hidden = {w}", fontsize=10)

fig.suptitle("Does hidden-layer drift matter? Causal test by freezing (backprop, 2x5 MNIST)\n"
             f"task 2 given the same {TASK2_ITERS}-update budget in every condition")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}")
np.savez(FIG.with_suffix(".npz"),
         **{f"t1|{w}|{c}": np.array([x[-1] for x in v]) for (w, c), v in C1.items()},
         **{f"t2|{w}|{c}": np.array([x[-1] for x in v]) for (w, c), v in C2.items()},
         widths=WIDTHS, task2_iters=TASK2_ITERS)
