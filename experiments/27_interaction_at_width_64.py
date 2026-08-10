"""27_interaction_at_width_64

ONE QUESTION
    Once the downward push on absent classes is switched off, what is still lost -- and can a
    different learning rule recover any of it?

    "2x2 factorial" means a 2-by-2 grid of CONDITIONS, not 2 classes per task. Every run here
    uses all ten digits, 5 in task 1 and 5 in task 2, exactly as before.

WHY THIS IS THE RIGHT NEXT EXPERIMENT
    Experiment 22 measured three interventions but never combined them, and that left the one
    interesting cell unmeasured. At hidden 16 the masked condition ends at 30% against a
    pre-switch peak of about 79%. Something is costing roughly 49 points that masking does not
    address. There are two candidates and they need separating:

        (a) the hidden code drifts, so the frozen output weights no longer read what they were
            trained to read
        (b) no gradient ever compared a task-1 class against a task-2 class, so the boundary
            between them was never learned and is whatever the geometry happens to give

    Freezing the hidden layer ON TOP of masking distinguishes them. If masked+frozen recovers
    most of those 49 points, cause (a) dominates and prospective configuration has a real
    budget to compete for after all. If it recovers almost none, cause (b) dominates, which no
    learning rule can touch, and the thesis result is settled.

THE GRID
                            W1 free                    W1 frozen during task 2
      unmasked        baseline: total collapse     hidden pathway blocked (exp 22: +0.1 to +3.3)
      masked          push removed (+30 to +47)    BOTH removed  <- the missing cell

    crossed with three learning rules: backprop, predictive coding, equilibrium propagation.
    12 cells. Same architecture everywhere, biases on, and each rule at the learning rate
    experiment 24 selected for THIS output structure (linear + squared error, one-hot 1/0).

WHY THE OUTPUT STRUCTURE IS FIXED HERE
    Experiment 24 showed the three rules land within 2.2 points of each other under linear +
    squared error with a 1/0 target (84.0 / 82.4 / 84.6), which is the tightest agreement of
    any structure. That makes it the fairest common ground: a forgetting difference cannot
    then be blamed on one rule simply learning better. Softmax is excluded because equilibrium
    propagation cannot use it (56% and total saturation).

METRICS -- and why the ones used so far were not enough
    Final accuracy saturated at zero in experiment 25 for eleven of twelve cells, so it could
    not separate anything. Three measures are recorded instead:
      final       task-1 accuracy at the end                (saturates, kept for continuity)
      half-life   updates until task-1 falls to half its peak   (works even if all end at zero)
      area kept   mean task-1 accuracy over task 2, as a fraction of its peak (never undefined)
    Also recorded: task-2 final accuracy, which must be similar across cells or the comparison
    is unfair, and the pre-switch peak, so "forgot less" can be separated from "learnt less".

EXPECTED
    * masked + frozen W1 lands much closer to masked than to the pre-switch peak, i.e. cause
      (b) dominates. Prediction: recovery of well under half the missing 49 points.
    * the three rules are within a few points of each other in every cell.
    * predictive coding shows its advantage in half-life and area kept rather than in final
      accuracy -- a gentler slope to the same place.

IF IT COMES OUT DIFFERENTLY
    masked + frozen W1 recovers most of the gap
        -> hidden drift matters a great deal once the output pathway is quiet, experiment 22
           underestimated it because the unmasked collapse swamped everything, and the
           learning-rule comparison becomes the centre of the report.
    predictive coding clearly beats backprop in any cell
        -> the first positive result in the project. Follow it with the depth sweep.
    equilibrium propagation beats backprop
        -> check the settling-convergence flags and beta before believing it; theory says a
           finite-difference estimate of a gradient should not beat the gradient.

RUNTIME
    Equilibrium propagation dominates: roughly 0.2 s per update at these settling settings. At
    N_RUNS=8 expect somewhere near an hour. To cut it, lower N_RUNS first, then TASK2_ITERS.

FIGURE
    Panel A: task-1 curves through task 2, one sub-panel per condition, one line per rule.
    Panel B: the 2x2 condition grid as a heatmap of task-1 accuracy retained, averaged over
             rules -- the decomposition, readable at a glance.
    Panel C: half-life by rule within each condition, which is where a rule difference will
             show up if it exists anywhere.
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

HIDDEN            = 64      # exp 26 ran at 16; a marker can call 16 units crippled.
                            # Same design at a defensible width: does the INTERACTION hold?
CLASSES_PER_TASK  = 5       # all ten digits, 5 + 5
TASK1_THRESHOLD   = 0.70
STOP_PATIENCE     = 3
TASK1_MAX_ITERS   = 600
TASK2_ITERS       = 250
BATCH             = 32
EVAL_EVERY        = 5
EVAL_PER_CLASS    = 100

ARCH = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=HIDDEN, out_dim=10,
            act="tanh", bias=True, init="scaled_normal")
OBJ_FULL   = Objective(loss="mse", target="onehot", mask=False)
OBJ_MASKED = replace(OBJ_FULL, mask=True)

# the 2x2 grid of conditions: (objective, parameters frozen during task 2)
CONDITIONS = {
    "unmasked, W1 free":   (OBJ_FULL,   set()),
    "unmasked, W1 frozen": (OBJ_FULL,   {"W1", "b1"}),
    "masked, W1 free":     (OBJ_MASKED, set()),
    "masked, W1 frozen":   (OBJ_MASKED, {"W1", "b1"}),
}

# learning rates from experiment 24, row "linear + SE, 1/0"
LR = {"backprop": 0.1, "pc": 0.1, "eqprop": 0.075}   # eqprop 0.075 from the widened exp-24 grid
RULES = list(LR)
EXTRA = {"pc": dict(dt=0.1, steps=50),
         "eqprop": dict(beta=0.1, dt=0.3, max_steps=200, settle_patience=15),
         "backprop": {}}
COL = {"backprop": "tab:red", "pc": "tab:blue", "eqprop": "tab:green"}
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)

K = [(c, r) for c in CONDITIONS for r in RULES]
C1 = {k: [] for k in K}
peak = {k: [] for k in K}
fin = {k: [] for k in K}
t2 = {k: [] for k in K}
hl = {k: [] for k in K}
ar = {k: [] for k in K}
missed = {k: 0 for k in K}
t0 = time.time()

for run in range(N_RUNS):
    seed = BASE_SEED + run
    d = np.random.default_rng(seed).permutation(10).tolist()
    k = CLASSES_PER_TASK
    tasks = [sorted(d[:k]), sorted(d[k:2 * k])]
    classes = sorted({c for t in tasks for c in t})
    stop_ev, rep_ev = make_eval_split(test, classes, EVAL_PER_CLASS, DEVICE, seed=seed)

    for cname, (obj, frozen_names) in CONDITIONS.items():
        for rule in RULES:
            handle = {}
            step_fn, predict = build_method(rule, in_dim=ARCH.in_dim, hidden=HIDDEN,
                                            out_dim=ARCH.out_dim, arch=ARCH, obj=obj,
                                            lr=LR[rule], seed=seed, device=DEVICE,
                                            handle=handle, **EXTRA[rule])
            params, freeze = handle["params"], handle["freeze"]
            snap = {}

            def on_task_end(ti, step, _p=params, _fz=freeze, _fn=frozen_names, _s=snap):
                if ti == 0:
                    _s["W1"] = _p.W1.detach().clone()
                    _fz.update(_fn)

            out = run_classil(
                step_fn, predict, tasks, train, cidx,
                report_eval=rep_ev, stop_eval=stop_ev,
                readouts={"a": restricted_argmax_fn(predict, classes)},
                max_iters_per_task=[TASK1_MAX_ITERS, TASK2_ITERS],
                stop_threshold=[TASK1_THRESHOLD, None],
                stop_patience=STOP_PATIENCE, batch=BATCH, eval_every=EVAL_EVERY,
                device=DEVICE, data_seed=seed, on_task_end=on_task_end)

            steps, cur, sw = out["steps"], out["curves"]["a"], out["switches"][0]
            key = (cname, rule)
            missed[key] += int(not out["reached"][0])
            pk = float(cur[steps <= sw][-1, 0])
            peak[key].append(pk)
            after = cur[steps > sw]
            # Index 0 is the PRE-SWITCH PEAK -- see the note in exp 22. Without it, panels
            # start at whatever the accuracy had already fallen to EVAL_EVERY updates in.
            C1[key].append(np.concatenate([[pk], after[:, 0]]))
            fin[key].append(float(after[-1, 0]))
            t2[key].append(float(after[-1, 1]))
            hl[key].append(half_life(steps, cur[:, 0], after=sw, peak=pk))
            ar[key].append(area_retained(steps, cur[:, 0], after=sw, peak=pk))

            if "W1" in frozen_names:
                assert torch.equal(params.W1, snap["W1"]), "W1 moved despite being frozen"
    print(f"run {run + 1}/{N_RUNS}  ({time.time() - t0:5.0f}s)")

# ------------------------------- table -------------------------------
print("\n" + "=" * 110)
print(f"hidden={HIDDEN}, 2x5 MNIST, {N_RUNS} runs, task 1 to {TASK1_THRESHOLD:.0%}, "
      f"task 2 fixed at {TASK2_ITERS} updates, linear+SE 1/0")
print(f"{'condition':>22}{'rule':>10}{'t1 peak':>9}{'t1 final':>12}{'t2 final':>10}"
      f"{'half-life':>11}{'area kept':>11}")
for cname in CONDITIONS:
    for rule in RULES:
        key = (cname, rule)
        f1 = np.array(fin[key]) * 100
        # Censor rather than drop: a run that never halved is the BEST outcome, and dropping
        # it silently removes the winning condition from the mean.
        h = [TASK2_ITERS if v is None else v for v in hl[key]]
        ncens = sum(v is None for v in hl[key])
        print(f"{cname:>22}{rule:>10}{np.mean(peak[key]) * 100:>9.1f}"
              f"{f1.mean():>8.1f}+/-{f1.std():<3.1f}{np.mean(t2[key]) * 100:>10.1f}"
              f"{np.mean(h):>10.0f}{'*' if ncens else ' '}{np.nanmean(ar[key]):>11.2f}")
        if missed[key]:
            print(f"{'':>32}WARNING: task 1 missed threshold in {missed[key]} run(s)")
        if np.mean(peak[key]) < TASK1_THRESHOLD:
            print(f"{'':>32}CAVEAT: mean peak {np.mean(peak[key]) * 100:.1f}% is below the "
                  f"{TASK1_THRESHOLD:.0%} criterion -- retention here is not comparable")

print("\nTHE DECOMPOSITION (averaged over rules, percentage points of task-1 accuracy)")
pk_all = np.mean([np.mean(peak[(c, r)]) for c in CONDITIONS for r in RULES]) * 100
g = {c: np.mean([np.mean(fin[(c, r)]) for r in RULES]) * 100 for c in CONDITIONS}
print(f"  pre-switch peak                              {pk_all:6.1f}")
for c in CONDITIONS:
    print(f"  {c:<42} {g[c]:6.1f}")
print(f"\n  removing the downward push alone gained      "
      f"{g['masked, W1 free'] - g['unmasked, W1 free']:+6.1f}")
print(f"  freezing the hidden layer ON TOP gained      "
      f"{g['masked, W1 frozen'] - g['masked, W1 free']:+6.1f}   <- the answer")
print(f"  still missing at the end                     "
      f"{pk_all - g['masked, W1 frozen']:6.1f}   <- no learning rule can reach this")

print("\nRULE COMPARISON, paired within condition (same seeds and pairings)")
for cname in CONDITIONS:
    b = np.array(fin[(cname, "backprop")]) * 100
    line = f"  {cname:<22}"
    for rule in ("pc", "eqprop"):
        d = np.array(fin[(cname, rule)]) * 100 - b
        line += f"  {rule} - backprop: {d.mean():+5.1f} (wins {int((d > 0).sum())}/{N_RUNS})"
    print(line)

# ------------------------------- figure -------------------------------
fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.95])
names = list(CONDITIONS)
for i, cname in enumerate(names):
    ax = fig.add_subplot(gs[0, i])
    n = min(min(len(x) for x in C1[(cname, r)]) for r in RULES)
    xs = np.arange(n) * EVAL_EVERY
    for r in RULES:
        A = np.stack([x[:n] for x in C1[(cname, r)]]) * 100
        ax.plot(xs, A.mean(0), color=COL[r], lw=2.2, label=r)
        ax.fill_between(xs, A.mean(0) - A.std(0), A.mean(0) + A.std(0),
                        color=COL[r], alpha=0.12)
    ax.axhline(pk_all, color="gray", ls=":", lw=1.2)
    ax.axhline(100 / (2 * CLASSES_PER_TASK), color="k", lw=0.8, ls="-.", alpha=0.5)
    ax.set_ylim(-2, 103); ax.grid(alpha=0.2); ax.set_title(cname, fontsize=10)
    ax.set_xlabel("updates into task 2")
    if i == 0:
        ax.set_ylabel("task-1 accuracy (%)"); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, :2])
M = np.array([[g["unmasked, W1 free"], g["unmasked, W1 frozen"]],
              [g["masked, W1 free"], g["masked, W1 frozen"]]])
im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(pk_all, M.max()))
fig.colorbar(im, ax=ax, label="task-1 accuracy retained (%)")
ax.set_xticks([0, 1]); ax.set_xticklabels(["W1 free", "W1 frozen"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["unmasked", "masked"])
for a in range(2):
    for b_ in range(2):
        ax.text(b_, a, f"{M[a, b_]:.1f}", ha="center", va="center", fontsize=13,
                color="white" if M[a, b_] < M.max() * 0.6 else "black")
ax.set_title(f"The decomposition (mean over rules). Pre-switch peak = {pk_all:.1f}%")

ax = fig.add_subplot(gs[1, 2:])
xs = np.arange(len(names)); w = 0.26
def _hl(c, r):
    """Mean half-life with never-halved runs CENSORED at the window end, plus the censor
       count. The old version dropped them, so a condition where accuracy never halved -- the
       best possible result -- was drawn as a missing bar."""
    raw = hl[(c, r)]
    return float(np.mean([TASK2_ITERS if v is None else v for v in raw])), \
        sum(v is None for v in raw)

for i, r in enumerate(RULES):
    stats = [_hl(c, r) for c in names]
    bars = ax.bar(xs + (i - 1) * w, [s[0] for s in stats], w, color=COL[r], alpha=0.85, label=r)
    for b, (_, cens) in zip(bars, stats):
        if cens:
            b.set_hatch("//")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + TASK2_ITERS * 0.015,
                    f"{cens}/{N_RUNS}\nnever\nhalved", ha="center", fontsize=6)
ax.axhline(TASK2_ITERS, color="k", lw=0.8, ls="-.", alpha=0.6)
ax.set_ylim(0, TASK2_ITERS * 1.25)
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8, rotation=12)
ax.set_ylabel("half-life (updates); hatched = censored at window end")
ax.legend(fontsize=8); ax.grid(alpha=0.2, axis="y")
ax.set_title("Where a learning-rule difference would show up\n(higher = slower forgetting)")

fig.suptitle("What remains once the downward push on absent classes is removed? "
             f"(2x5 MNIST, hidden={HIDDEN}, all ten digits)")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}")
np.savez(FIG.with_suffix(".npz"),
         **{f"final|{c}|{r}": np.array(v) for (c, r), v in fin.items()},
         **{f"peak|{c}|{r}": np.array(v) for (c, r), v in peak.items()},
         **{f"area|{c}|{r}": np.array(v) for (c, r), v in ar.items()},
         hidden=HIDDEN, task2_iters=TASK2_ITERS)
