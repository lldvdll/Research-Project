"""24_output_structure_can_it_learn

ONE QUESTION
    For each combination of learning rule and output structure, does the network learn at all,
    and at what learning rate and ceiling?

    JOINT training only -- all ten classes at once, no continual learning. This is the safety
    gate before any forgetting result involving these combinations can be interpreted, and it
    is where the "did we break a method by standardising it" worry gets settled once.

WHY THIS EXISTS
    We are standardising every method onto one architecture and one output structure so that a
    method difference is a method difference. But standardising can silently cripple a method:
      * Predictive coding's energy IS squared prediction error. Handing it a cross-entropy
        output error makes it a hybrid -- relaxation with a non-Gaussian output -- not
        canonical predictive coding. It may still work; Pinchetti et al. (2022) treat this
        properly. Worth knowing empirically.
      * Equilibrium propagation's output units are free variables with a self-decay term, so in
        the free phase they settle to some natural magnitude. If that magnitude is around 0.1
        and we now demand a target of 1.0, the loss will drive large weights, tanh saturates,
        f'(h) collapses, and the nudge can no longer reach the hidden layer. We would have
        broken it while believing we had standardised it. The script measures the free-phase
        output scale BEFORE training, so this is checked rather than assumed.

TEST
    3 rules x 4 output structures x a learning-rate grid, joint 10-class MNIST 14x14.

      rules             backprop, predictive coding, equilibrium propagation
      output structures softmax + cross-entropy, one-hot 1/0
                        linear + squared error,   one-hot 1/0        <- the proposed standard
                        linear + squared error,   +-1
                        linear + hinge,           +-1                <- EqProp's old default
      architecture      IDENTICAL everywhere: 196 -> 64 -> 10, tanh on the way out, biases on

    Per cell: best accuracy within the budget, and the learning rate that achieved it. Every
    cell also reports a saturation fraction and, for equilibrium propagation, the free-phase
    output magnitude.

PROJECT DECISIONS THIS SETTLES
    1. Whether the proposed standard (squared error, one-hot 1/0) is safe for all three rules.
       If any rule cannot learn under it, the standard changes -- better to find out now.
    2. The per-rule learning rate for every later experiment. Exp 14 tried this and failed
       because its target accuracy of 90% was above the 10-class ceiling of ~87%; the target
       here is set from exp 13 at 0.75 and the ceiling is reported rather than assumed.
    3. Whether biases are harmless for predictive coding and equilibrium propagation. This is
       the gate on the bias decision.

EXPECTED
    * Backprop learns under all four; best under softmax + cross-entropy.
    * Predictive coding learns well under both squared-error codings, and comparably or a
      little worse under cross-entropy. Roughly matches backprop.
    * Equilibrium propagation learns best under hinge +-1 (what it was designed around) and
      squared error +-1, and WORSE under one-hot 1/0 -- because the 1/0 target is asymmetric
      relative to the free-phase equilibrium, which sits near zero.
    * Best learning rates cluster around 0.05 for backprop and predictive coding, and 0.05-0.1
      for equilibrium propagation -- an order of magnitude above the 0.005 used in
      experiments 11, 12 and 15, which is the confound that made it look worst.

WHAT THAT WOULD DEMONSTRATE
    That the choice of output structure is a substantive design decision with a measurable
    cost per rule, not a formality; and that the reported ordering of methods in earlier work
    was partly a statement about hyperparameters rather than about credit assignment.

IF IT COMES OUT DIFFERENTLY
    EqProp fails under one-hot 1/0 at every learning rate
        -> keep +-1 for EqProp and report the asymmetry honestly as an intrinsic constraint of
           a Hopfield-style energy, rather than pretending the comparison is fully matched.
           The comparison then varies the label coding for one method, and that must be stated.
    Predictive coding fails under cross-entropy
        -> fine, and expected on theoretical grounds. It just means the softmax arm of
           experiment 25 is backprop-only.
    Everything works everywhere
        -> the output structure matters less than the analysis in the knowledge base suggests,
           and section 4.5 needs weakening. Note it explicitly.

FIGURE
    24_output_structure_can_it_learn.png
    Heatmap grid: rows = output structure, columns = learning rule, cell = best joint accuracy,
    annotated with the learning rate that achieved it. A second panel shows accuracy against
    learning rate, so the shape of each rule's sensitivity is visible rather than just its peak.
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data import load_mnist, class_indices, make_eval_set
from src.model import Arch, Objective, init_params
from src.methods import build_method
from src.runner import run_joint
from src.probes import saturation
from src.eqprop import eqprop_free_output_scale

# ============================ constants ============================
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data"
FIG      = Path(__file__).resolve().with_suffix(".png")
IMG_SIZE = 14
SEED     = 0

CLASSES    = list(range(10))
MAX_ITERS  = 600
BATCH      = 32
EVAL_EVERY = 20
EVAL_PER_CLASS = 50

ARCH = Arch(in_dim=IMG_SIZE * IMG_SIZE, hidden=64, out_dim=10,
            act="tanh", bias=True, init="scaled_normal")

STRUCTURES = {
    "softmax + CE, 1/0":   Objective(loss="ce",    target="onehot"),
    "linear + SE, 1/0":    Objective(loss="mse",   target="onehot"),   # proposed standard
    "linear + SE, +-1":    Objective(loss="mse",   target="pm1"),
    "linear + hinge, +-1": Objective(loss="hinge", target="pm1"),      # EqProp's old default
}
# Grids widened after the first run. Reasons, from that run:
#   * backprop and pc PEAKED AT THE EDGE (lr 0.3) under softmax+CE and under hinge, so the
#     true optimum was outside the grid. Extended to 0.5 and 1.0.
#   * under squared error they DIVERGED at 0.3 (accuracy fell to 10%, i.e. one-class collapse),
#     so the extra points there are wasted but harmless -- divergence is itself a reportable
#     property of the objective and worth having on the curve.
#   * eqprop peaked at 0.05 and collapsed to 10% at 0.1 everywhere except softmax. The grid is
#     refined between those two rather than extended past them.
LRS = {
    "backprop": [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
    "pc":       [0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
    "eqprop":   [0.005, 0.01, 0.02, 0.035, 0.05, 0.075],
}
RULES = list(LRS)
EQP_KW = dict(beta=0.1, dt=0.3, max_steps=300, settle_patience=20)   # beta 0.1, not 0.3:
# the numpy mirror measured ~42% gradient-estimator error at beta=0.3 and ~27% at 0.1.
PC_KW = dict(dt=0.1, steps=50)
# ==================================================================

train, test = load_mnist(size=IMG_SIZE, root=str(DATA_DIR))
cidx = class_indices(train)
eval_x, eval_y = make_eval_set(test, CLASSES, EVAL_PER_CLASS, DEVICE)

best = np.full((len(STRUCTURES), len(RULES)), np.nan)
best_lr = np.empty((len(STRUCTURES), len(RULES)), dtype=object)
sweep = {}
notes = []
t0 = time.time()

# ---- pre-flight: EqProp free-phase output magnitude (the safety gate) --------------
p0 = init_params(ARCH, seed=SEED, device=DEVICE).requires_grad_(True)
scale = eqprop_free_output_scale(eval_x[:64], p0, ARCH, dt=0.3, max_steps=300,
                                 settle_patience=20, device=DEVICE)
print(f"EqProp free-phase output magnitude at init: mean|y| = {scale['mean_abs']:.3f}, "
      f"max|y| = {scale['max_abs']:.3f}")
print("  -> if mean|y| is far below 1.0, a one-hot 1/0 target is asking the network to move a")
print("     long way and saturation is the risk to watch in the 1/0 rows.\n")

for i, (sname, obj) in enumerate(STRUCTURES.items()):
    for j, rule in enumerate(RULES):
        accs = []
        for lr in LRS[rule]:
            kw = dict(EQP_KW) if rule == "eqprop" else (dict(PC_KW) if rule == "pc" else {})
            handle = {}
            step_fn, predict = build_method(rule, in_dim=ARCH.in_dim, hidden=ARCH.hidden,
                                            out_dim=ARCH.out_dim, arch=ARCH, obj=obj, lr=lr,
                                            seed=SEED, device=DEVICE, handle=handle, **kw)
            s, a = run_joint(step_fn, predict, CLASSES, train, cidx, eval_x, eval_y,
                             max_iters=MAX_ITERS, batch=BATCH, eval_every=EVAL_EVERY,
                             device=DEVICE, data_seed=SEED)
            sat = saturation(handle["features"], eval_x[:200])
            accs.append(a.max() if a.size else np.nan)
            print(f"  {sname:<21}{rule:>9} lr {lr:<6} best {a.max()*100 if a.size else 0:5.1f}%"
                  f"  saturated {sat*100:4.1f}%  ({time.time()-t0:5.0f}s)")
            if sat > 0.5:
                notes.append(f"{sname} / {rule} / lr={lr}: {sat*100:.0f}% of hidden units saturated")
        sweep[(sname, rule)] = np.array(accs, dtype=float)
        k = int(np.nanargmax(sweep[(sname, rule)]))
        best[i, j] = sweep[(sname, rule)][k]
        best_lr[i, j] = LRS[rule][k]

# ------------------------------- table -------------------------------
print("\n" + "=" * 92)
print(f"BEST JOINT 10-CLASS ACCURACY (%) within {MAX_ITERS} updates, with the winning lr")
print(f"{'output structure':<23}" + "".join(f"{r:>21}" for r in RULES))
for i, sname in enumerate(STRUCTURES):
    print(f"{sname:<23}" + "".join(
        f"{best[i, j]*100:>12.1f} (lr {best_lr[i, j]})" for j in range(len(RULES))))
print(f"\nreference: exp 13 gives the converged 10-class ceiling at hidden=64 as ~86.7%")
if notes:
    print("\nSATURATION WARNINGS (f'(h) -> 0, the nudge/error cannot reach the hidden layer):")
    for n in notes:
        print("  " + n)

print("\nDECISIONS TO TAKE FROM THIS TABLE")
print("  1. the standard output structure = the row where all three rules are closest to the")
print("     86.7% ceiling. If that is not 'linear + SE, 1/0', change the standard.")
print("  2. the per-rule learning rate for experiments 25 onward = the winning lr in that row.")
print("  3. if eqprop's winning lr is far above 0.005, every earlier eqprop result is void.")

# ------------------------------- figure -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5),
                         gridspec_kw={"width_ratios": [1.05, 1.4]})
im = axes[0].imshow(best * 100, cmap="viridis", aspect="auto", vmin=0, vmax=90)
fig.colorbar(im, ax=axes[0], label="best joint accuracy (%)")
axes[0].set_xticks(range(len(RULES))); axes[0].set_xticklabels(RULES)
axes[0].set_yticks(range(len(STRUCTURES))); axes[0].set_yticklabels(list(STRUCTURES), fontsize=9)
for i in range(len(STRUCTURES)):
    for j in range(len(RULES)):
        v = best[i, j] * 100
        axes[0].text(j, i, f"{v:.1f}\nlr {best_lr[i, j]}", ha="center", va="center",
                     fontsize=8, color="white" if v < 45 else "black")
axes[0].set_title("Can it learn? (joint 10-class)")

# Legend fix: the first version labelled only the backprop series, so a low-scoring eqprop
# curve looked as though it contradicted the heatmap. Colour = output structure, LINE STYLE =
# rule, and both are now in the legend.
LS = {"backprop": "-", "pc": "--", "eqprop": ":"}
MK = {"backprop": "o", "pc": "s", "eqprop": "^"}
COL = {k: c for k, c in zip(STRUCTURES, ["tab:red", "tab:blue", "tab:green", "tab:purple"])}
for sname in STRUCTURES:
    for rule in RULES:
        axes[1].plot(LRS[rule], sweep[(sname, rule)] * 100, marker=MK[rule], ls=LS[rule],
                     color=COL[sname], lw=1.6, ms=5, alpha=0.85)
from matplotlib.lines import Line2D
handles = ([Line2D([], [], color=COL[s], lw=2.4, label=s) for s in STRUCTURES]
           + [Line2D([], [], color="k", ls=LS[r], marker=MK[r], label=r) for r in RULES])
axes[1].axhline(86.7, color="k", ls=":", lw=1.2)
axes[1].axvline(0.005, color="gray", ls="--", lw=1.0)
axes[1].set_xscale("log"); axes[1].set_xlabel("learning rate")
axes[1].set_ylabel("best joint accuracy (%)"); axes[1].set_ylim(0, 100)
axes[1].grid(alpha=0.2)
handles += [Line2D([], [], color="k", ls=":", lw=1.2, label="exp-13 ceiling (~86.7%)")]
axes[1].legend(handles=handles, fontsize=7, ncol=2, loc="lower left")
axes[1].set_title("Sensitivity to learning rate\ncolour = output structure, line style = rule")
fig.suptitle("Output structure x learning rule: does the standardised setup break any method?")
fig.tight_layout(); fig.savefig(FIG, dpi=120, bbox_inches="tight")
print(f"\nsaved {FIG.name}")
np.savez(FIG.with_suffix(".npz"), best=best,
         best_lr=np.array([[str(x) for x in row] for row in best_lr]),
         structures=list(STRUCTURES), rules=RULES)
