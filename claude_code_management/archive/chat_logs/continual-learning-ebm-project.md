# Continual Learning & Energy-Based Models — Project Handoff

*MSc dissertation. Context transfer document: everything needed to pick this up in a fresh workspace.*

---

## 1. Main goal

**Core question:** Does an energy-based model (EBM) with a biologically plausible local learning rule reduce catastrophic forgetting compared to backprop — and if not, why not, and can it be made to?

**The authoritative scope is the advisor's four-point outline** (received after a period of scope sprawl; he explicitly said *"everything below can/should wait"* about the wider wishlist):

1. **Pick one EBM.** Be clear which and acknowledge others exist. Reviewing some others is desirable, not essential.
2. **Compare catastrophic forgetting in that model vs a backprop model.**
3. **Try to understand why they differ.** Can "trivial" differences — coding sparsity, network size — explain the difference?
4. **Try to reduce catastrophic forgetting in the EBM.** EBMs are *predictive*, so it should be possible to find the nodes that differ most in their prediction and select only those for learning new stimuli. Does this work?

Deferred by the advisor (do not chase now): generative/synthetic replay, VAE example-ordering, the 3×3 scenario×dataset grid, other EBM families, efficiency comparisons.

---

## 2. Key decisions

### Model / problem
| Decision | Value | Reason |
|---|---|---|
| Dataset | MNIST downsampled to **14×14** (196 inputs), scaled [0,1] | EqProp settling is the runtime bottleneck; 14×14 keeps MNIST intact (~97% backprop ceiling) while being ~4× cheaper. 8×8 degrades classes. |
| Architecture | MLP **196 → 64 → 10**, one hidden layer | Same for every method so comparisons isolate the learning rule. No CNNs — avoids inductive priors so observations generalise. |
| Optimiser | **plain SGD** everywhere | Exact interference identity; no cross-task momentum confound; matches EqProp's own SGD updates. |
| Scenario | **Class-IL** (single 10-way output, no task ID at test) | Hardest scenario; where softmax models collapse and where EBM claims matter. |
| Splits used | 10×1, 5×2, 2×2 (`TASKS` list) | 10×1 cleanest for forgetting; 2×2 best for seeing a single crossing in detail. |
| EBM #1 | **Equilibrium Propagation (EqProp)** — supervisor's `gem_lazyep.py`, rewritten | This is the "pick one EBM" choice. |
| EBM #2 | **Predictive Coding (PC)** — added later | **Critical:** the "energy-based learning reduces interference" claim in the project's reading (Song & Bogacz, prospective configuration) is *predictive coding's* claim, **not EqProp's**. To test the literature's claim you need PC. |

### Working practice (adopted after a period of overwhelm — these matter)
- **One question per experiment**, written as a single sentence before running.
- **Controls on every forgetting experiment**: backprop (negative control — should forget) and replay (positive control — should fix it). This is what stops "maybe forgetting is just inevitable" spirals: if replay works, the problem is provably solvable.
- **A doubt gets one scheduled test, then it's closed.** Doubts that arise mid-run go on an *open questions* list, not chased immediately.
- **One script → one figure, named identically.** Figure name is derived from `__file__`.
- Keep code minimal. Repeated over-engineering (wandb, Bayesian HPO, class hierarchies, a `harness.py` abstraction) was tried and **deliberately deleted**. Don't reintroduce.

### Explicitly abandoned
- wandb / Optuna / persistent HPO infrastructure — too much machinery for the science question.
- `harness.py` shared run+plot module — run/plot now lives *in* each experiment script so each can vary.
- Per-class output heads — investigated and rejected: with task ID it's Task-IL (which doesn't forget for *any* method); without it, calibration fails equally for all methods. Not a route to an EBM advantage.
- Contrastive/conditional-energy EBM (Li, Du, van de Ven, Mordatch 2022) — this *is* the EBM proven to beat replay in Class-IL, but the advisor's plan supersedes it. Keep as a "mentioned alternative" for point 1.

---

## 3. Current status

### Working and validated
- **EqProp trains**: ~91% validation on joint 14×14 MNIST, ~6 min for 3 epochs on CPU.
- **PC implemented**: all three gradients (∂F/∂x₁, ∂F/∂W₁, ∂F/∂W₂) verified against finite differences to ~1e-9; a numpy mirror of the update rule learns a toy 3-class problem to 100%. Torch version written but **not yet executed end-to-end**.
- **Code structure settled** (`src/` + `experiments/`), see §5.

### Results so far

**Class-IL 10×1** (10 tasks, 1 digit each, 100 iters/task):
| method | final mean acc | reading |
|---|---|---|
| backprop | ~10% (collapse floor) | forgets completely |
| eqprop | ~10% | forgets completely |
| replay | ~64% | positive control works — problem IS solvable |

**Class-IL 5×2**: backprop ~20%, eqprop ~20%, replay ~60% (note: 5×2 ran only 500 total updates vs 1000 for 10×1 — set `ITERS=200` for a matched budget).

**Class-IL 2×2** (tasks [0,1] then [2,3], 100 iters/task, batch 32) — the most informative run:
- **backprop**: task 1 holds at 100% until ~step 110, then collapses to 0 as task 2 rises. Clean sequential trade.
- **replay**: both tasks end ~95%. Retains.
- **eqprop**: **forgets before it learns** — task 1 collapses at the switch, task 2 only rises afterwards. Worst crossing of the four.
- **pc**: **forgets more slowly** — task 1 decays gradually over ~100 steps while task 2 rises, but ends ~15%. Better decay shape than backprop; crossover point lower than backprop's; nowhere near replay.

### Interpretation of these results (important)

**Why everything forgets in Class-IL:** forgetting here is driven by **output competition**, not by the learning rule. A single shared output layer means training class *c* actively pushes every other class's output down. A local/energy-based learning rule changes *how credit is assigned*, not *that the outputs compete*. This is why EqProp forgetting is an expected, honest result rather than a failure.

**Why EqProp is worst:** its hinge target is **+1 for the true class and −1 for all others**. Every single example actively drives all nine other outputs down — even more aggressive suppression than softmax. Hence "forgets before it learns."

**Why PC decays more gently:** its target is one-hot (0 for others, not −1), so the suppression is weaker; and the settle-then-update dynamic means the weight change needed is smaller. This is the *shape* difference worth quantifying — and it's the closest thing so far to supporting the prospective-configuration claim.

**The confound to rule out (advisor point 3):** learning rates differ across methods (BP 0.05, EqProp 0.005, PC 0.05). Comparing forgetting speed while learning speed differs measures the learning rate, not the learning rule. **Match `to_learn` across methods before attributing `to_forget` differences to the rule.**

### Immediate next step
`experiments/11_consolidate_pairs_4methods.py` is **written and compile-checked but not yet run.** It runs all four methods over 10 random digit pairings (2 tasks × 2 classes), producing 2×2 accuracy grids, 2×2 trajectory grids (thin lines = runs, thick = mean), a pairing table, and a summary table with crossover accuracy, final accuracies, steps-to-learn-T2 and steps-to-forget-T1.

### Then, in order
1. **Consolidate** (run 11) — establishes what's known with variance across pairings.
2. **Advisor point 3** — rule out trivial explanations: sweep hidden width (32/64/128/256) and measure activation sparsity per method. Does network size or sparsity explain the EqProp/PC/backprop differences?
3. **Advisor point 4** — node gating. Already implemented as `eqprop_update_gated` / `make_eqprop_gated` (§5), **untested**. Also worth a PC version, since PC has an explicit per-node prediction error which is a more natural "which nodes differ most in their prediction" signal than EqProp's free-vs-nudged shift.
4. Optional, already coded but unrun: `make_eqprop_replay`, `make_eqprop_synthetic` (generative replay).

---

## 4. Important constraints

### Hardware / runtime
- **CPU only, no GPU.** A GPU helps less than expected: each settling step is a tiny matmul (batch×64), so per-step overhead dominates. Parallel CPU processes beat one GPU for sweeps.
- Settling is the bottleneck. EqProp ≫ PC ≫ backprop in cost (PC prediction is a plain feedforward pass, no settling).
- Keep sweeps cheap: subset the data (10k), 1 epoch, fewer settle steps. Confirm only the winner on full data.

### EqProp failure modes (hard-won)
- **Saturation is the killer.** As weights grow, hidden pre-activations grow, `tanh` flattens, `tanh'(h) → 0`, and the feedback path carrying the nudge to the hidden layer is severed → learning stalls. Track `% of |tanh(h)| > 0.95` as a first-class diagnostic. Low `lr` is the main control.
- **The nudged phase never reaches an absolute tolerance.** The hinge keeps pushing while the margin is unmet, so per-step movement plateaus at a non-zero floor. Hence settling stops on **patience** (movement stops improving), not on a fixed `tol`. This is a genuine property worth a sentence in the writeup.
- **Warm-start the nudged phase from the free equilibrium** — otherwise it re-settles from scratch and dominates runtime.
- **Batch size 1 destroys EqProp.** With ±1 targets and no batch to average over, every update reconfigures the network to the most recent image. Slow training down with the **learning rate**, keep batch ≥16.
- The energy has **no ½x² self-term**, so during generation `x` has no restoring force and pins at the clamp bounds → weak generator. Inspect samples before trusting synthetic replay.

### Metric pitfalls (all previously hit)
- **Never report train-batch accuracy.** An earlier bug did this and invalidated a day of sweeps. Always evaluate on the held-out test set (`make_eval_set` uses the test split).
- **`cur%` is degenerate at 1 class/task** — a model that predicts one class always scores 100% on that class. Use `seen%` or per-task accuracy.
- **`seen%` has a changing denominator** across tasks, so it isn't comparable over a run. **Prefer per-task accuracy with fixed class sets.**
- **Accuracy is a threshold readout.** After a task switch nothing appears to happen for ~20 steps while logits climb toward the crossing, then it flips fast. **Log raw outputs** (`predict(x, raw=True)`) to see the underlying continuous dynamics.
- The flat line at exactly 10% (10×1) / 20% (5×2) / 25% (2×2) is not chance — it's the **collapse floor**: `100 / n_classes`, the score of a model that predicts one class for everything.
- Don't fit sigmoids to accuracy curves — they're step-like and noisy. Use **threshold crossings** and the **ACC1-vs-ACC2 trajectory** instead.

### Personal working constraints
- Plain language, no flowery metaphors. Concept first (ELI5 + graduate), then decisions with trade-offs, then code.
- One stage at a time; small increments; minimal machinery. Over-complication has repeatedly caused loss of momentum.
- Motivation dips have occurred. The engaging threads are: *what happens inside the network when a new class arrives* (do units get overwritten, reused, or newly allocated) and *the EBM as its own replay generator*. Keep those visible.

---

## 5. Code

### Layout
```
project/
├── data/                       # MNIST downloads here
├── src/
│   ├── __init__.py             # empty
│   ├── data.py                 # load_mnist, class_indices, make_eval_set
│   ├── eqprop.py               # eqprop_init/energy/settle/update/predict
│   │                           #   + eqprop_update_gated, eqprop_generate
│   ├── predictive_coding.py    # pc_init/forward/settle/update/predict
│   └── methods.py              # make_backprop, make_replay, make_eqprop, make_pc,
│                               #   make_eqprop_gated, make_eqprop_replay, make_eqprop_synthetic
└── experiments/
    ├── 09_eqprop_learning_vs_forgetting.py
    ├── 10_pc_learning_vs_forgetting.py
    └── 11_consolidate_pairs_4methods.py    # written, NOT YET RUN
```

**Interface contract:** every `make_*` returns `(train_step, predict)`.
`train_step(x, y)` does one update. `predict(x, raw=False)` returns class indices, or raw pre-argmax outputs when `raw=True`.
Adding a model = one new `make_*` function. Experiment scripts change only the `methods` dict.

### `src/eqprop.py` — core
```python
def eqprop_energy(x, h, y, W1, W2):
    state = 0.5 * (h ** 2).sum() + 0.5 * (y ** 2).sum()
    align = (h * (x @ W1)).sum() + (y * (torch.tanh(h) @ W2)).sum()
    return state - align

def eqprop_settle(x, W1, W2, target=None, beta=0.0, dt=0.3, max_steps=500,
                  settle_patience=30, min_delta=1e-4, h0=None, y0=None, device="cpu"):
    """Relax (h, y) until per-step movement stops improving for `settle_patience` steps."""
    with torch.enable_grad():
        x = x.reshape(x.size(0), -1)
        h = (torch.zeros(x.size(0), W1.size(1), device=device) if h0 is None else h0.clone()).requires_grad_(True)
        y = (torch.zeros(x.size(0), W2.size(1), device=device) if y0 is None else y0.clone()).requires_grad_(True)
        best, since = float("inf"), 0
        for _ in range(max_steps):
            gh, gy = torch.autograd.grad(eqprop_energy(x, h, y, W1, W2), [h, y])
            if target is not None:                      # hinge nudge, added by hand (== autograd of the hinge term)
                gy = gy + beta * torch.where(1 - target * y > 0, -target, torch.zeros_like(target))
            move = (dt * (gh.pow(2).sum() + gy.pow(2).sum()).sqrt()).item()
            h.data -= dt * gh; y.data -= dt * gy
            if move < best - min_delta: best, since = move, 0
            else: since += 1
            if since >= settle_patience: break
    return h.detach(), y.detach()

def eqprop_update(x, y_labels, W1, W2, opt, beta=0.3, dt=0.3, max_steps=500, settle_patience=30, device="cpu"):
    x = x.reshape(x.size(0), -1)
    target = torch.full((x.size(0), W2.size(1)), -1.0, device=device)   # +1 true class, -1 all others
    target.scatter_(1, y_labels.unsqueeze(1), 1.0)
    h_f, y_f = eqprop_settle(x, W1, W2, dt=dt, max_steps=max_steps,
                             settle_patience=settle_patience, device=device)              # free phase
    h_n, y_n = eqprop_settle(x, W1, W2, target, beta, dt, max_steps, settle_patience,
                             h0=h_f, y0=y_f, device=device)                               # nudged, warm-started
    opt.zero_grad()
    gW1_f, gW2_f = torch.autograd.grad(eqprop_energy(x, h_f, y_f, W1, W2), [W1, W2])
    gW1_n, gW2_n = torch.autograd.grad(eqprop_energy(x, h_n, y_n, W1, W2), [W1, W2])
    W1.grad = (gW1_n - gW1_f) / (beta * x.size(0))      # contrastive: difference of the two equilibria
    W2.grad = (gW2_n - gW2_f) / (beta * x.size(0))
    opt.step()
```

### `src/eqprop.py` — node gating (advisor point 4, untested)
```python
def eqprop_update_gated(x, y_labels, W1, W2, opt, beta=0.3, dt=0.3, max_steps=500,
                        settle_patience=30, gate_frac=0.3, device="cpu"):
    """Update only the `gate_frac` hidden nodes that move MOST under the nudge; freeze the rest."""
    # ... free + nudged settle as above ...
    shift = (h_n - h_f).abs().mean(0)                   # per-node responsibility [hidden]
    k = max(1, int(gate_frac * shift.numel()))
    mask = torch.zeros_like(shift); mask[torch.topk(shift, k).indices] = 1.0
    W1.grad = ((gW1_n - gW1_f) / (beta * x.size(0))) * mask.unsqueeze(0)   # gate W1 columns [in, hidden]
    W2.grad = ((gW2_n - gW2_f) / (beta * x.size(0))) * mask.unsqueeze(1)   # gate W2 rows    [hidden, out]
    opt.step()
```
*Works only if different digits recruit different nodes.* If the same nodes are most responsive for every class, gating won't protect anything — which is itself the answer, and points to adding a freeze on nodes claimed by earlier tasks.

### `src/predictive_coding.py` — core (gradients finite-difference verified)
```
Layers:  x0 (input, clamped) -> x1 (hidden, free) -> x2 (output, clamped to target while training)
    mu1 = x0 @ W1 ;  e1 = x1 - mu1
    mu2 = tanh(x1) @ W2 ;  e2 = x2 - mu2
    F   = ½|e1|² + ½|e2|²                     (energy = total squared prediction error)
Inference = relax x1 to reduce F with the target clamped.
Learning  = dW1 = x0ᵀ e1 , dW2 = tanh(x1)ᵀ e2   (local: pre-activity × post-error)
```
```python
def pc_settle(x0, W1, W2, target, dt=0.1, steps=50):
    """Infer hidden activities with the output clamped. Starts from the feedforward value (e1 = 0)."""
    x0 = x0.reshape(x0.size(0), -1)
    mu1 = x0 @ W1
    x1 = mu1.clone()
    for _ in range(steps):
        e1 = x1 - mu1
        e2 = target - torch.tanh(x1) @ W2
        dx1 = e1 - (1 - torch.tanh(x1) ** 2) * (e2 @ W2.t())     # dF/dx1
        x1 = x1 - dt * dx1
    return x1

def pc_update(x, y_labels, W1, W2, lr=0.05, dt=0.1, steps=50, device="cpu"):
    x0 = x.reshape(x.size(0), -1)
    target = torch.zeros(x0.size(0), W2.size(1), device=device)
    target.scatter_(1, y_labels.unsqueeze(1), 1.0)               # ONE-HOT (0 for others, not -1)
    x1 = pc_settle(x0, W1, W2, target, dt=dt, steps=steps)       # activities settle FIRST
    e1 = x1 - x0 @ W1
    e2 = target - torch.tanh(x1) @ W2
    W1 += lr * (x0.t() @ e1) / x0.size(0)                        # local updates, no autograd
    W2 += lr * (torch.tanh(x1).t() @ e2) / x0.size(0)

def pc_predict(x, W1, W2, raw=False):
    """Test time: nothing clamped, so the equilibrium is just the feedforward pass."""
    x0 = x.reshape(x.size(0), -1)
    out = torch.tanh(x0 @ W1) @ W2
    return out if raw else out.argmax(1)
```

### Hyperparameters currently in use
```python
IMG_SIZE = 14 ; IN_DIM = 196 ; HIDDEN = 64 ; OUT = 10
BATCH = 32 ; ITERS = 100 (per task) ; EVAL_EVERY = 1..5 ; EVAL_PER_CLASS = 100

BP_LR = 0.05
RP_LR = 0.05 ; RP_PER_CLASS = 20
EQP_LR = 0.005 ; EQP_BETA = 0.3 ; EQP_DT = 0.3 ; EQP_MAX_STEPS = 500 ; EQP_SETTLE_PAT = 30
PC_LR  = 0.05  ; PC_DT = 0.1 ; PC_STEPS = 50
```
*(Joint-training EqProp config that reached ~91%: lr 0.03–0.1, beta 0.3–0.5, dt 0.3–0.5, batch 32–64.)*

### Experiment script template
```python
import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
# ... imports from src ...

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = ROOT / "data"
FIG      = Path(__file__).resolve().with_suffix(".png")   # figure auto-named after the script
TASKS    = [[0, 1], [2, 3]]                               # THE experiment definition lives here
CLASSES  = sorted({c for t in TASKS for c in t})
CLS_POS  = {c: i for i, c in enumerate(CLASSES)}
COLLAPSE = 100 / len(CLASSES)                             # collapse floor, NOT chance
# ... constants ...

eval_x, eval_y = make_eval_set(test, classes=CLASSES, per_class=EVAL_PER_CLASS, device=DEVICE)
methods = { "backprop": make_backprop(...), "replay": make_replay(...), "<under test>": ... }
# run loop + plots inline (per-experiment, deliberately not abstracted)
```

### Key metrics implemented
```python
def crossover(steps, t1, t2, switch):
    """Accuracy value where task-1 and task-2 curves cross after the switch (linear interp).
       HIGH = held both tasks at once; LOW = pure trade of one for the other."""

def first_cross(steps, series, thresh, switch, rising):
    """Steps after the switch until a series crosses `thresh` (rising = learning, falling = forgetting)."""
```
Plus the **ACC1-vs-ACC2 trajectory plot** (advisor's whiteboard sketch): plot `task1_acc` vs `task2_acc` over time. A forgetting model travels along the anti-diagonal from (100,0) to (0,100); a retaining model bends up-right toward (100,100). Removes time from the picture entirely — the most robust forgetting metric found so far.

---

## 6. Reference: how the three learning rules differ

All three answer *"the output was wrong — which weights do I change?"* The difficulty is that hidden units have no target.

**The sharpest distinction:** in backprop the hidden activities are **fixed by the weights**. In PC and EqProp they are **variables that get optimised first**, and only then do the weights change. That is the energy-based family in one sentence.

| | backprop | predictive coding | EqProp |
|---|---|---|---|
| hidden activities | fixed by weights | **variables — settle to a target** | **variables — settle twice** |
| credit assignment | chain rule from above | inferred by relaxation | difference of two equilibria |
| weight update | global backward pass | local: pre-activity × post-error | local: free vs nudged difference |
| passes per update | 1 fwd + 1 bwd | 1 settling | 2 settlings |
| gradient | exact | exact at equilibrium | approximate (β-biased) |
| target at output | supervision signal | clamped | a perturbation, not a target |
| non-target classes | softmax suppression | one-hot → 0 | **hinge → −1 (strongest suppression)** |

---

## 7. Key references

- **van de Ven, Tuytelaars & Tolias (2022)**, *Three types of incremental learning*, Nat Mach Intell 4:1185 — defines Task-IL / Domain-IL / Class-IL; the baseline the project reproduces.
- **Song, Bogacz et al. (2024)**, prospective configuration / predictive coding, Nat Neurosci — **the source of the "energy-based learning reduces interference" claim.** This is PC's claim, not EqProp's.
- **Scellier & Bengio (2017)**, *Equilibrium Propagation*, Front. Comput. Neurosci. — the chosen EBM.
- **Kirkpatrick et al. (2017)**, PNAS — EWC (fails in Class-IL, ~20% ≈ none; reproduced in earlier notebook).
- **Li, Du, van de Ven & Mordatch (2022)**, *Energy-Based Models for Continual Learning*, CoLLAs, arXiv 2011.12216 — the conditional-energy EBM that **does** beat replay in Class-IL, via contrastive divergence (no softmax normalisation over all classes). Code: `github.com/ShuangLI59/ebm-continual-learning`. **Deferred**, but cite as the acknowledged alternative for advisor point 1.
- **van de Ven, Siegelmann & Tolias (2020)**, *Brain-inspired replay*, Nat Commun 11:4069 — generative replay; hippocampus as generative network.
- **Kendall et al. (2020)** arXiv 2006.01981; **Martin et al. (2021)** EqSpike arXiv 2010.07859 — EqProp on analog/neuromorphic hardware. Explains why settling is slow in silicon but free in physics. This is EqProp's real value proposition and the natural framing for its chapter.

*A fuller literature review exists as `ebm_literature_review.md` (biological plausibility by brain region, hardware requirements, which EBMs have demonstrated CF mitigation and on which IL tasks).*

---

## 8. Earlier work (previous notebooks, still valid)

From `00_mnist_baseline.ipynb` at 28×28 with Adam, before the move to scripts:
- **Task-IL does not forget** for backprop (shared trunk + per-task heads: probe and head both ~99%). An earlier apparent forgetting was a **polarity artifact** of a balanced 2-way head — fold with `max(acc, 1-acc)`.
- **Class-IL forgets catastrophically**: ~21.6% final mean, tasks collapse to ~0% (active overwriting, not decay to chance).
- **Gradient interference measured**: cosine between current-task and prior-task gradients is predominantly **negative** during Class-IL. Occasional positive (cooperative) cosine coincides with slower new-task learning. Sign↔Δloss identity is exact under SGD.
- **Replay works**: ~78% vs ~21%. Not equivalent to joint training (memory-limited).
- **EWC fails in Class-IL**: ~20% ≈ baseline, reproducing van de Ven Table 2. Reason: (a) deadlock — protecting old weights prevents learning new classes, learning new ones suppresses old logits; (b) preserving each task's function cannot create discrimination between classes never seen together. Unresolved hypothesis: EWC's Fisher-importance distribution resembles replay's yet its accuracy matches the baseline → failure is in the **readout/arbitration, not the features**. Test = linear probe on the EWC trunk (predict: probe high, head low). Never run.
- **Multi-pass caution**: >1 pass = cyclic revisiting, not longer continual learning. Baseline mean drifts ~20%→~30% then plateaus (residual output structure once every class has been trained), not learning-to-not-forget.
