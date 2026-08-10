"""25_output_structure_and_forgetting

ONE QUESTION
    How much of the difference in forgetting between the three learning rules is caused by the
    OUTPUT STRUCTURE rather than by the credit-assignment rule?

WHY IT MATTERS FOR THE REPORT
    Every earlier comparison varied three things at once: the rule, the loss, and the label
    coding. Backprop ran softmax + cross-entropy; predictive coding ran squared error with a
    one-hot 1/0 target; equilibrium propagation ran a hinge with a +-1 target. Knowledge base
    section 4.5 predicts these differ sharply in how hard they push an ABSENT class down:

        target 0  (squared error)  push stops once the output reaches 0
                                   -> the weight vector ends ORTHOGONAL to current features,
                                      keeping whatever structure lives in other directions
        target -1 (squared error)  push continues to -1, one whole unit further
        target -1 (hinge)          same destination, and the gradient is a constant, so the
                                   push does not weaken as it gets there
        softmax                    relative, with NO stopping point at all: the absent output
                                   is driven down without limit relative to the winner

    If that analysis is right, label coding alone should reorder the methods -- which would
    mean the published ordering in experiments 11, 12 and 15 was partly an artefact of our own
    configuration choices.

TEST
    Run experiment 24's surviving cells continually. 2 tasks x 5 classes, MNIST 14x14, at the
    operating point chosen from experiments 22 and 23. Every cell uses the SAME architecture
    (biases on, tanh) and the per-rule learning rate that experiment 24 selected for that
    output structure -- so within a row, the only thing changing is the rule, and within a
    column, the only thing changing is the output structure.

    Skip any cell that experiment 24 showed cannot learn: a forgetting number for a model that
    never learned task 1 is meaningless.

    Headline metric: task-1 accuracy after a FIXED task-2 budget, with task 1 trained to a
    common threshold first. Same protocol as experiment 22, so the numbers are comparable.

PROJECT DECISIONS THIS SETTLES
    1. Whether the standardised output structure is the right one to carry into the main
       comparison, on forgetting grounds rather than just learnability grounds.
    2. How much of the earlier "equilibrium propagation is worst" result was the +-1 hinge.
    3. Whether the report can claim a rule effect at all, or must report a rule effect
       conditional on output structure.

EXPECTED
    * Within every rule: "linear + SE, 1/0" forgets LEAST, "softmax + CE" forgets MOST, and the
      two +-1 codings sit in between, with hinge worse than squared error.
    * The spread ACROSS output structures within one rule is comparable to, or larger than, the
      spread across rules within one output structure. That is the finding.
    * Equilibrium propagation's deficit shrinks substantially once it is off the +-1 hinge and
      onto a properly tuned learning rate.
    * Predictive coding retains a small advantage over backprop in every column, since its
      mechanism acts on the hidden layer and is orthogonal to the output structure.

WHAT THAT WOULD DEMONSTRATE
    That the output structure is a first-class experimental variable in class-incremental
    learning, and that any paper comparing learning rules without matching it is reporting a
    confounded result. It also gives the report a clean, defensible justification for the
    standard we adopt, backed by our own numbers.

IF IT COMES OUT DIFFERENTLY
    Output structure barely matters, rules separate cleanly
        -> section 4.5's analysis is too strong and should be weakened; the good news is that
           the method comparison is then robust and the earlier results are largely salvageable.
    Predictive coding's advantage disappears once the output structure is matched
        -> then the earlier advantage WAS the output structure, and the central claim of the
           thesis becomes negative. Still a real result, and exp 22's freeze-W1 ceiling tells
           you whether it was ever available.
    Equilibrium propagation overtakes backprop under a matched structure
        -> would contradict the theory in section 3.7 (a finite-difference estimator of
           backprop should not beat it). Check beta and the settling convergence flags first.

FIGURE
    25_output_structure_and_forgetting.png
    Panel A: grouped bars, final task-1 accuracy, x = output structure, colour = rule, with the
             collapse floor and each cell's pre-switch peak marked.
    Panel B: the same cells as a rule-vs-structure heatmap of forgetting (peak minus final), so
             the relative size of the two effects is directly readable.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data import load_mnist, class_indices, make_eval_split
from src.model import Arch, Objective
from src.methods import build_method
from src.runner import run_classil
from src.probes import restricted_argmax_fn

# ============================ constants ============================
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data"
FIG      = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE = 14
BASE_SEED = 0
N_RUNS   = 10

HIDDEN            = 16       # <-- SET FROM EXPERIMENTS 22 AND 23 before running
CLASSES_PER_TASK  = 5
TASK1_THRESHOLD   = 0.70
STOP_PATIENCE     = 3
TASK1_MAX_ITERS   = 800
TASK2_ITERS       = 300      # fixed budget, identical in every cell (protocol from exp 22)
BATCH             = 32
EVAL_EVERY        = 5
EVAL_PER_CLASS    = 100

ARCH = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=HIDDEN, out_dim=10,
            act="tanh", bias=True, init="scaled_normal")

STRUCTURES = {
    "softmax + CE, 1/0":   Objective(loss="ce",    target="onehot"),
    "linear + SE, 1/0":    Objective(loss="mse",   target="onehot"),
    "linear + SE, +-1":    Objective(loss="mse",   target="pm1"),
    "linear + hinge, +-1": Objective(loss="hinge", target="pm1"),
}
RULES = ["backprop", "pc", "eqprop"]

# <-- FILL IN FROM EXPERIMENT 24. Set a value to None to SKIP that cell (it could not learn).
LR = {
    ("softmax + CE, 1/0",   "backprop"): 0.05,
    ("softmax + CE, 1/0",   "pc"):       0.05,
    ("softmax + CE, 1/0",   "eqprop"):   0.05,
    ("linear + SE, 1/0",    "backprop"): 0.05,
    ("linear + SE, 1/0",    "pc"):       0.05,
    ("linear + SE, 1/0",    "eqprop"):   0.05,
    ("linear + SE, +-1",    "backprop"): 0.05,
    ("linear + SE, +-1",    "pc"):       0.05,
    ("linear + SE, +-1",    "eqprop"):   0.05,
    ("linear + hinge, +-1", "backprop"): 0.05,
    ("linear + hinge, +-1", "pc"):       0.05,
    ("linear + hinge, +-1", "eqprop"):   0.05,
}
EQP_KW = dict(beta=0.1, dt=0.3, max_steps=300, settle_patience=20)
PC_KW = dict(dt=0.1, steps=50)
COL = {"backprop": "tab:red", "pc": "tab:blue", "eqprop": "tab:green"}
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)

peak = {(s, r): [] for s in STRUCTURES for r in RULES}
final = {(s, r): [] for s in STRUCTURES for r in RULES}
t2fin = {(s, r): [] for s in STRUCTURES for r in RULES}
missed = {(s, r): 0 for s in STRUCTURES for r in RULES}
t0 = time.time()

for run in range(N_RUNS):
    seed = BASE_SEED + run
    d = np.random.default_rng(seed).permutation(10).tolist()
    k = CLASSES_PER_TASK
    tasks = [sorted(d[:k]), sorted(d[k:2 * k])]
    classes = sorted({c for t in tasks for c in t})
    stop_ev, rep_ev = make_eval_split(test, classes, EVAL_PER_CLASS, DEVICE, seed=seed)

    for sname, obj in STRUCTURES.items():
        for rule in RULES:
            lr = LR.get((sname, rule))
            if lr is None:
                continue
            kw = dict(EQP_KW) if rule == "eqprop" else (dict(PC_KW) if rule == "pc" else {})
            handle = {}
            step_fn, predict = build_method(rule, in_dim=ARCH.in_dim, hidden=HIDDEN,
                                            out_dim=ARCH.out_dim, arch=ARCH, obj=obj, lr=lr,
                                            seed=seed, device=DEVICE, handle=handle, **kw)
            ra = restricted_argmax_fn(predict, classes)

            o1 = run_classil(step_fn, predict, [tasks[0]], train, cidx,
                             report_eval=rep_ev, stop_eval=stop_ev, readouts={"a": ra},
                             max_iters_per_task=TASK1_MAX_ITERS, batch=BATCH,
                             eval_every=EVAL_EVERY, device=DEVICE,
                             stop_threshold=TASK1_THRESHOLD, stop_patience=STOP_PATIENCE,
                             data_seed=seed)
            missed[(sname, rule)] += sum(not r for r in o1["reached"])
            peak[(sname, rule)].append(o1["curves"]["a"][-1, 0])

            run_classil(step_fn, predict, [tasks[1]], train, cidx,
                        report_eval=rep_ev, stop_eval=stop_ev, readouts={"a": ra},
                        max_iters_per_task=TASK2_ITERS, batch=BATCH, eval_every=EVAL_EVERY,
                        device=DEVICE, stop_threshold=None, data_seed=seed + 7777)

            pred = ra(rep_ev[0])
            acc = {c: (pred[rep_ev[1] == c] == c).float().mean().item() for c in classes}
            final[(sname, rule)].append(float(np.mean([acc[c] for c in tasks[0]])))
            t2fin[(sname, rule)].append(float(np.mean([acc[c] for c in tasks[1]])))
    print(f"run {run+1}/{N_RUNS} done ({time.time()-t0:5.0f}s)")

# ------------------------------- table -------------------------------
print("\n" + "=" * 104)
print(f"FINAL TASK-1 ACCURACY (%) after {TASK2_ITERS} fixed updates on task 2   "
      f"hidden={HIDDEN}, {N_RUNS} runs")
print(f"{'output structure':<23}" + "".join(f"{r:>26}" for r in RULES))
for s in STRUCTURES:
    row = f"{s:<23}"
    for r in RULES:
        v = np.array(final[(s, r)]) * 100
        pk = np.array(peak[(s, r)]) * 100
        row += ("".rjust(26) if v.size == 0
                else f"{v.mean():>12.1f}+/-{v.std():<4.1f}(peak {pk.mean():4.1f})")
    print(row)
    for r in RULES:
        if missed[(s, r)]:
            print(f"{'':<23}  WARNING {r}: task 1 never reached threshold in "
                  f"{missed[(s, r)]} run(s)")

print("\nEFFECT SIZES -- this is the comparison the report needs")
sp_struct = []
for r in RULES:
    vs = [np.mean(final[(s, r)]) * 100 for s in STRUCTURES if final[(s, r)]]
    if len(vs) > 1:
        sp_struct.append(max(vs) - min(vs))
        print(f"  spread ACROSS output structures, {r:>9}: {max(vs) - min(vs):5.1f} points")
sp_rule = []
for s in STRUCTURES:
    vs = [np.mean(final[(s, r)]) * 100 for r in RULES if final[(s, r)]]
    if len(vs) > 1:
        sp_rule.append(max(vs) - min(vs))
        print(f"  spread ACROSS rules, {s:>21}: {max(vs) - min(vs):5.1f} points")
if sp_struct and sp_rule:
    print(f"\n  mean structure effect {np.mean(sp_struct):.1f} pts   "
          f"vs   mean rule effect {np.mean(sp_rule):.1f} pts")
    print("  If the structure effect is the larger of the two, then any learning-rule")
    print("  comparison that does not match the output structure is confounded, and that")
    print("  sentence belongs in the report.")

# ------------------------------- figure -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5),
                         gridspec_kw={"width_ratios": [1.35, 1.0]})
names = list(STRUCTURES)
xs = np.arange(len(names)); w = 0.26
for i, r in enumerate(RULES):
    m = [np.mean(final[(s, r)]) * 100 if final[(s, r)] else np.nan for s in names]
    e = [np.std(final[(s, r)]) * 100 if final[(s, r)] else np.nan for s in names]
    axes[0].bar(xs + (i - 1) * w, m, w, yerr=e, capsize=3, color=COL[r], alpha=0.85, label=r)
for i, s in enumerate(names):
    pk = [np.mean(peak[(s, r)]) * 100 for r in RULES if peak[(s, r)]]
    if pk:
        axes[0].hlines(np.mean(pk), i - 1.6 * w, i + 1.6 * w, color="gray", ls=":", lw=1.4)
axes[0].axhline(100 / (2 * CLASSES_PER_TASK), color="k", lw=0.9, ls="-.",
                label="collapse floor")
axes[0].set_xticks(xs); axes[0].set_xticklabels(names, fontsize=8, rotation=12)
axes[0].set_ylabel("final task-1 accuracy (%)"); axes[0].set_ylim(0, 103)
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2, axis="y")
axes[0].set_title("A. Retention by output structure and rule\n(dotted = pre-switch peak)")

M = np.array([[(np.mean(peak[(s, r)]) - np.mean(final[(s, r)])) * 100 if final[(s, r)] else np.nan
               for r in RULES] for s in names])
im = axes[1].imshow(M, cmap="magma_r", aspect="auto")
fig.colorbar(im, ax=axes[1], label="forgetting: peak minus final (points)")
axes[1].set_xticks(range(len(RULES))); axes[1].set_xticklabels(RULES)
axes[1].set_yticks(range(len(names))); axes[1].set_yticklabels(names, fontsize=8)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isfinite(M[i, j]):
            axes[1].text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9,
                         color="white" if M[i, j] > np.nanmean(M) else "black")
axes[1].set_title("B. Forgetting (lower is better)")
fig.suptitle(f"How much of the method difference is the output structure? "
             f"(2x5 class-IL, hidden={HIDDEN}, matched architecture and per-cell learning rate)")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}")
np.savez(FIG.with_suffix(".npz"),
         **{f"final|{s}|{r}": np.array(v) for (s, r), v in final.items()},
         **{f"peak|{s}|{r}": np.array(v) for (s, r), v in peak.items()},
         hidden=HIDDEN, task2_iters=TASK2_ITERS)
