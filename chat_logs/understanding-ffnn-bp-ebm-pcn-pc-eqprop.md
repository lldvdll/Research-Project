# Context Transfer: FFNN vs BP vs EBM vs PCN vs PC vs EqProp

**Purpose of this doc:** hand-off summary so a new workspace can pick up mid-project without re-deriving anything. Written after a conversation that (a) rebuilt the conceptual understanding from primary sources, (b) verified the existing experiment code, (c) corrected two claims that were wrong, and (d) began laying out a separated experiment plan.

---

## 1. Main goal / task

**Thesis project:** an evaluation of energy-based / predictive-coding learning rules against backpropagation for **catastrophic forgetting (CF)** in continual learning. MSc in computational neuroscience & AI, on a tight deadline.

**Four methods under comparison**, all on a 2-task class-incremental split of MNIST (14×14), 64 hidden units:

| Method | What it is |
|---|---|
| `backprop` | baseline / lower bound |
| `replay` | backprop + stored-example buffer — upper bound |
| `eqprop` | Equilibrium Propagation (two-phase contrastive, Hopfield-style energy) |
| `pc` | Predictive Coding / Prospective Configuration (settle activities, then local Hebbian update) |

**Goal of the conversation specifically:** get a *mechanical, non-metaphorical* understanding of how these architectures and learning rules relate, grounded in the actual papers, and then design experiments that test the forgetting claims rather than assert them.

---

## 2. Key decisions and corrections

### 2.1 Two corrections to earlier (wrong) explanations — important

**Correction A — "PCN error nodes are basically backprop's errors" is only true in specific regimes.**
There are three regimes, and only the first two look like backprop:

1. **Partial relaxation / infinitesimal nudging → approximates backprop.** Whittington & Bogacz (2017) is literally titled "*An approximation* of the error backpropagation algorithm…". Song et al. note that updating activities for only the *first few steps* makes the PC weight update **equal** backprop's. Millidge et al. (2022) unify PC, EqProp, and contrastive Hebbian learning as all reducing to backprop in the infinitesimal inference limit.
2. **Engineered exact equivalence** (Song et al., NeurIPS 2020) — but the Dong & Wu commentary states this equivalence "is not general": it needs specific initialisation, a precise layer-wise update schedule, and particular inference settings.
3. **Full relaxation to equilibrium → prospective configuration, genuinely ≠ backprop.** This is the actual contribution of Song et al. 2024.

**So: "same result, different route" is true only in the small-step limit.** Turning settling all the way up is what makes it a different algorithm. This is a citable, useful sentence for the methods chapter.

**Correction B — the "zero error → weights don't move" mechanism does NOT explain cross-task forgetting mitigation.**
This was challenged and the challenge was correct. The property ("an already-correct output has zero error, so its weights barely move") is a **within-a-single-forward-pass** statement — it is what Bogacz Fig 1 shows (one input, two outputs, one already correct; PC fixes the wrong one without disturbing the right one *on that same example*). During task-2 training you clamp **task-2 inputs and task-2 targets**, so the error is task-2's error and it drives weight change through whatever units task-2's settling implicates — including units task 1 relied on. The argument gives **no protection to task 1**.

### 2.2 What the results actually show (this is the honest headline)

From the attached plots (`11_consolidate_pairs_4methods.png` / `_traj.png`, 10 random digit pairings):

- **PC does *not* retain task 1 in this setup.** Final task-1 accuracy ≈ **8–10%**. Backprop and EqProp ≈ **0%**. Only `replay` ends genuinely up-and-right (final task-1 ≈ **68%**).
- What PC buys here is **graceful degradation, not durable retention**: task-1 accuracy decays as a *slope* rather than a *cliff*, and the trajectory bows **above** the diagonal during the transition (higher crossover: ~75% PC vs ~65% BP; replay ~85%).
- **Interpretation to use:** PC reduces *interference per update*; it does not *anchor* past knowledge. Durable retention requires an explicit anchor (replayed data, or a consolidation penalty). PC has none, so it forgets — just politely.

### 2.3 The refined, *testable* mechanism (replaces the wrong one)

In the PC implementation, after settling:

```
e1 = x1 - x0 @ W1     # how far the hidden layer moved from its feedforward value
ΔW1 ∝ x0ᵀ e1
```

So **PC changes a weight in proportion to the activity displacement its settling required.** The hypothesis becomes:

> PC interferes with task 1 only to the extent that satisfying task-2's target *forces movement in the hidden units task 1 depends on*.

If tasks used disjoint hidden units, PC's updates would concentrate on task-2's units and leave task-1's alone. In the current setup (64 hidden units, overlapping digit inputs, single head) representational overlap is high, so task-1 units get overwritten anyway.

**This reframes the question from "is the error zero?" (no) to "where does the weight movement go?" (measurable).** That is the core interpretability experiment.

### 2.4 Other decisions

- **Reproduce Bogacz's claimed result *before* dissecting the mechanism.** Confirm the outcome in the regime/setup where it's claimed, then explain it.
- **Separate the experiments.** One research question per script, so each result can be internalised before moving on. No combined mega-experiments.
- **Metaphors are banned.** Springs / rubber sheets / corporate analogies obscure control flow. Explanations must map to variables, order of operations, and code lines.
- **Claims must be separated from hypotheses**, with figure-level citations for anything called a claim.

---

## 3. Current status and next steps

### 3.1 Status

- ✅ Conceptual framework rebuilt and verified against primary sources (see §6).
- ✅ Existing code (`predictive_coding.py`, `eqprop.py`, `methods.py`, `data.py`, `11_consolidate_pairs_4methods.py`) **reviewed and verified correct** — see §5.2 for the equation→code map.
- ✅ **One confound identified** in the current 4-way comparison (see §4.1).
- ✅ `src/plotting.py` written (reusable plotting utilities, refactored out of script 11) — see §5.4.
- ⬜ Reproduction script — **not yet written** (was in progress when the conversation ended).
- ⬜ Separated interpretability experiments — planned only.

### 3.2 Immediate next step: the Bogacz reproduction

**Research question:** *On an alternating 5+5 class-incremental split, with PC and backprop matched in architecture and loss, does PC forget less and relearn faster?*

Target: **Song & Bogacz 2024, Fig 4d–e**. Their setup: task 1 = five randomly selected Fashion-MNIST classes, task 2 = the remaining five, **trained alternately** (not a single switch).

Faithfulness points that matter:
- **5 + 5 split** (current script uses 2+2 — change this).
- **Alternating schedule** — task1, task2, task1, task2… The current script does a *single* switch, which cannot show the "relearning" half of their claim.
- **Matched tanh + squared error** between PC and BP (see §4.1).
- Hidden 64; PC relaxation γ ≈ 0.1, ~50 steps.
- Run once on **MNIST**, once on **Fashion-MNIST** (dataset as a variable, same script).
- Learning rate should be **grid-searched per rule** — the paper selects the optimal LR independently for each learning rule, and comparing at a single shared LR is not a fair reproduction.

⚠️ This reproduces the **structure** of their experiment, not their exact hyperparameters. Their GitHub has the exact LR grid, epoch counts, and LeakyReLU choice if a closer match is wanted. The verified tanh PC implementation was kept rather than switching to LeakyReLU, to keep the codebase consistent.

### 3.3 Planned experiment sequence (proposed — one question per script)

| # | Research question |
|---|---|
| 12 | Under a matched setup on alternating 5+5, does PC forget less / relearn faster than BP? *(reproduction; run on MNIST and Fashion-MNIST)* |
| 14 | Does task 1 live in a small subset of hidden units? *(ablation sweep + Fisher diagonal ranking; keep top-k, zero the rest, measure retained accuracy)* |
| 15 | Is unit/weight **importance** correlated with weight **magnitude**? *(directly tests the "large weights preserve task 1 for a while" hypothesis — do not assume large = important)* |
| 16 | **Where does the weight movement go?** During task-2 training, does PC concentrate updates *away* from task-1-important weights more than BP does? *(the core mechanism test)* |
| 17 | Freezing control: freeze the top-k task-1-important weights during task 2 — how much forgetting disappears? *(causal evidence; also a hard-consolidation mini-EWC)* |
| 18 | Does EWC stack usefully on PC, even though it failed on vanilla BP? *(contained: add penalty to the weight update)* |
| 19 | Metaplasticity / "kinematic consolidation" augmentation *(see §4.4 — only worth building if 16 shows the mechanism is real)* |

**Metrics for #16 (the decisive one):**
- Per-weight `|Δw|` accumulated during task 2, split by task-1 importance rank.
- Overlap between "task-1-important" and "task-2-heavily-updated" weight sets.
- Representational drift of task-1 inputs (CKA or activation overlap, before vs after task 2).

**Confirm vs refute:**
- PC's task-2 updates concentrate *away* from task-1-important weights while BP's overlap heavily → **supports** the displacement/locality mechanism.
- Both overlap equally yet PC still forgets less → **refutes** locality; the advantage is then something else (e.g. simply less erratic updates, which Song et al. also claim via Supplementary Fig. 7).

Either outcome is a defensible thesis result.

**Controls to hold fixed throughout:** per-rule optimal learning rate; **matched final task-2 accuracy** (otherwise "forgot less" is confounded with "learned less"); batch size; multiple seeds with error bars.

---

## 4. Important constraints

### 4.1 ⚠️ Confound in the current 4-way comparison — fix before drawing conclusions

`make_backprop` / `make_replay` use **ReLU + CrossEntropyLoss**; `pc` and `eqprop` use **tanh + squared error**. The current experiment therefore varies **three** things at once: algorithm, nonlinearity, and loss. Bogacz deliberately holds nonlinearity and loss fixed so a PC-vs-BP gap isolates the *algorithm*.

**Matched BP control should be** (exactly PC's function class, no biases):
```
x1  = x0 @ W1
out = tanh(x1) @ W2
loss = ½ |target − out|²        # one-hot target, SGD
```

### 4.2 ⚠️ Scenario taxonomy — this probably explains the EWC failure

Per van de Ven, Tuytelaars & Tolias (2022, *Nat. Mach. Intell.* — in project files):

- **Task-incremental (Task-IL):** task identity known at test → multi-head output.
- **Domain-incremental (Domain-IL).**
- **Class-incremental (Class-IL):** identity must be *inferred* → single head over all classes.

Their empirical finding: **EWC and SI perform near the upper bound on Task-IL, degrade on Domain-IL, and fail essentially completely on Class-IL** (down to the no-defence baseline). **Replay holds up across all three.**

The current setup is **Class-IL** (single head, `make_eval_set` over all four classes). So EWC failing is *expected*, not an implementation bug. Before concluding "EWC doesn't work", run the Task-IL (multi-head) version as the fair test. This also explains why replay dominates.

### 4.3 Working-style constraints (respect these when generating code)

- **One research question per script**, named `NN_short_question_description.py`, with the question stated in the module docstring.
- **All constants at the top of the file**, in an obvious block, easy to edit.
- **Minimal, linear, easy-to-follow scripts.** Large or sprawling codebases cause a stall — keep each script small enough to hold in working memory.
- **Reusable logic goes in `src/`**, experiment scripts stay thin.
- Avoid spawning parallel variants of experiments; finish and interpret one before starting the next.
- Compute: full resolution is acceptable, but downsample + adjust parameters if runtime becomes painful (EqProp dominates runtime — it runs two settling loops of up to 500 steps each).

### 4.4 Idea quarantine (do not build yet)

**"Kinematic consolidation" / metaplasticity idea:** stiffness triggered by a *sharp velocity drop* in a weight's trajectory (the "right-angle elbow", a second-derivative signal), with the stiffness decaying over training time.

Where it sits in the literature — it is **a genuine variant of none of these exactly**:
- **EWC** (Kirkpatrick 2017): importance = diagonal Fisher; quadratic penalty; needs task boundaries.
- **Synaptic Intelligence** (Zenke, Poole & Ganguli 2017): importance accumulated *online along the trajectory* — a path integral of each synapse's contribution to decreasing the loss. Closest relative, but **first-order** (path length / velocity), and it does **not** decay.
- **Synaptic metaplasticity** (Laborieux, Ernoult, Hirtzlin & Querlioz 2021, *Nat. Commun.*): "the plasticity itself is plastic"; hidden weights as metaplastic variables; works **without task boundaries and without replay**. Keys on weight history/magnitude, not the acceleration→deceleration elbow.
- **Benna & Fusi (2016)**: cascade / complex-synapse models — the biological grounding (multiple timescales in a single synapse).

**Sequencing:** only build this *after* experiment 16. If PC already leaves task-1-critical weights alone, a stiffness term adds little; if it doesn't, stiffness is exactly the missing piece. Also: in Class-IL, *any* parameter-regularisation method (EWC, SI, this idea) is expected to underperform replay — combining stiffness *with* generative replay is more likely to move the needle than stiffness alone.

---

## 5. Reference material: the maths, and where it lives in the code

### 5.1 The taxonomy (two axes + one umbrella)

|  | **Architecture** (what units exist) | **Learning rule** (how weights change) |
|---|---|---|
| Standard DL | **FFNN** — layers of value units, one-way | **Backprop** — chain-rule gradient of an output loss |
| Energy-based | **Hopfield net**, **PCN** — recurrent; PCN adds error units | **EqProp**, **Prospective Configuration** |

- **EBM** — umbrella: a scalar energy `E(x; w)` over the state; both inference and learning minimise it.
- **Hopfield network** — ancestral EBM: homogeneous units, symmetric recurrent weights, pairwise-interaction energy, settles into attractors (associative memory). **No explicit error variable.**
- **PCN** — hierarchical EBM with **value nodes** `x` and **error nodes** `ε`; energy = sum of squared prediction errors.
- **Prospective Configuration** — the learning rule on a PCN: settle activities to a target-consistent state first, then one local weight update.
- **EqProp** — learning rule for EBMs: two equilibria (free, nudged), update from their difference.

**The unifying idea:** all the energy-based methods run an **EM-like two-step** — E-step: activities relax to low energy; M-step: weights move to make that state more probable (Dong & Wu commentary states this explicitly). **Backprop collapses the two**: activities are computed once and frozen; only weights move toward the target.

### 5.2 Key clarifications settled in the conversation

**Why an FFNN's loss isn't already an EBM energy → free variables.**
In an FFNN, activities are a *deterministic function* of input and weights — there is nothing to minimise over in `x`. In an EBM, `x` is a **free variable** and inference *is* `x ← argmin_x E(x, w)`, done before `w` moves.

Exact statement: take the PCN energy `E = Σ_l ½(x_l − w_{l−1} f(x_{l−1}))²`. **Clamp every hidden `x_l` to its feedforward value** and every error is zero except at the output — `E` reduces to **exactly the FFNN output loss**. *The FFNN is the PCN with its internal state frozen to the forward pass.*

**Where `f` sits / is `x` pre- or post-activation.**
Song–Bogacz convention: prediction of layer `l` is `w_{l−1} f(x_{l−1})` — nonlinearity **then** weight. So `x` is the raw node state (membrane-potential-like), `f(x)` is the rate it sends onward. Not the textbook `a_l = f(W a_{l−1})`. Visible in the weight update, where the presynaptic factor is `f(x_l)`, not `x_l`. It's a modelling convention — different PC papers place `f` differently; don't read deep meaning into it.

**Do EBMs use activation functions the same way?**
Same *role*, different *entry point*. In an FFNN `f` appears once, in the forward substitution. In a PCN, `f` **and its derivative `f'`** live inside the relaxation dynamics.

**Is the global energy ever actually used? Does it break biological plausibility?**
**No — it is never computed, stored, or transmitted by the network.** It is a Lyapunov function *we* use to prove convergence. Because the energy is a **sum of local terms**, its gradient w.r.t. any local variable is itself a purely local expression. Analogy: a ball rolling downhill has no "potential-energy register"; it responds only to the local slope. This is precisely the objection that sinks *backprop's* plausibility (nonlocal backward pass) and that EBMs escape.

**Two timescales.** Fast inner loop over settling steps `t` (activities move, weights frozen), inside a slow outer loop over learning steps `k` (weights move). **Backprop has no inner loop at all** — that is the structural gap.

**β vs α vs γ.**
- `α` — weight learning rate: how far *weights* step, once, per learning update.
- `γ` — settling rate: how far *activities* move per relaxation iteration.
- `β` — EqProp's **nudge**, **not** a learning rate: the size of a deliberate *perturbation* used to estimate a gradient by finite difference between two equilibria. It appears **in the denominator** (`Δw ∝ (1/β)(nudged − free)`), and the estimate becomes exact as `β → 0`.

**"Hopfield-style nodes" and why explicit error nodes matter.**
Hopfield-style = one homogeneous population, symmetric recurrent weights, one scalar per unit, pairwise-interaction energy, **no separate error variable**. That is exactly why EqProp *needs two phases*: with no explicit error, the only way to recover the learning signal is to **subtract two equilibria**.

Making the error an explicit node buys: (1) a **one-shot local** weight update, `Δw ∝ ε_post · f(x_pre)`, read straight off a neuron; (2) a **directed hierarchy** (predictions one way, errors the other) instead of a homogeneous recurrent soup; (3) **one relaxation instead of two**.

### 5.3 Equation → code map (verified)

**Predictive coding — `src/predictive_coding.py`**

Energy: `F = ½|e1|² + ½|e2|²`, with `e1 = x1 − x0·W1`, `e2 = target − tanh(x1)·W2`.

*Inference* — `pc_settle`. Starts `x1 = mu1 = x0 @ W1` (so `e1 = 0` initially), then relaxes:
```python
dx1 = e1 - (1 - torch.tanh(x1)**2) * (e2 @ W2.t())      # dF/dx1
x1  = x1 - dt * dx1
```
This is exactly `∂F/∂x1 = e1 − f'(x1) ⊙ (W2ᵀ e2)` — the hidden state reducing its own error while absorbing the top-down output error, weighted by the local tanh slope. **Derivative verified by hand; matches.** (The file's own note "gradients verified against finite differences" is accurate.)

*Learning* — `pc_update`:
```python
W1 += lr * (x0.t() @ e1) / N                # presynaptic input × hidden error
W2 += lr * (torch.tanh(x1).t() @ e2) / N    # presynaptic rate  × output error
```
These are `Δw = −α ∂F/∂w`. **Sign verified:** `∂F/∂W1 = −x0ᵀe1`, so `+=` is gradient *descent* on energy. Purely local Hebbian.

**Key subtlety for experiment 16:** because `x1` is initialised to `mu1`, `e1` after settling **is literally the displacement of the hidden layer from its feedforward value**. So `ΔW1` is proportional to that displacement. This is the measurable quantity in the refined mechanism.

Also note: no bias terms; `mu1 = x0 @ W1` is **linear** (tanh appears only on hidden→output). The matched BP control must have the same function class.

`pc_predict` uses the feedforward pass — correct, since with the output unclamped and `e1 = 0` that *is* the equilibrium for this 1-hidden-layer architecture.

**Equilibrium propagation — `src/eqprop.py`**

Hopfield-style energy (`eqprop_energy`):
```
E = ½|h|² + ½|y|² − hᵀ(x·W1) − yᵀ(tanh(h)·W2)
```
quadratic leak terms minus bilinear interactions.

`eqprop_settle` descends `∂E/∂h`, `∂E/∂y` via autograd — the relaxation. Free phase starts from `h=0, y=0`; the nudged phase **warm-starts from the free state** (`h0=h_f, y0=y_f`), which is correct and saves time.

The nudge:
```python
gy = gy + beta * torch.where(1 - target*y > 0, -target, torch.zeros_like(target))
```
is `β · ∂/∂y` of a hinge cost `max(0, 1 − target·y)` — it only pushes outputs that haven't reached margin. Correct EqProp-with-hinge.

*Learning* — `eqprop_update`:
```python
W1.grad = (gW1_n - gW1_f) / (beta * x.size(0))       # (nudged − free) / β
```
**Verified:** since `∂E/∂W1 = −xᵀh`, this yields effective `ΔW1 ∝ +xᵀ(h_n − h_f)/β` after the SGD step — the standard contrastive rule. **Implementation is correct**; EqProp's noisiness and task-1 collapse in the plots are EqProp being EqProp (the finite-nudging gradient bias that stops it scaling past MNIST — Laborieux et al. 2021), not a bug.

**Replay — `src/methods.py`, `make_replay`**
It *is* backprop. The only addition: the first time each class is seen, store `per_class` examples in `mem_x`/`mem_y`; every batch gets an equal-sized replay sample concatenated in before the normal backward pass. **No new learning rule** — it just re-shows old data. That is exactly the anchor the energy-based methods lack, and why it wins.

### 5.4 `src/plotting.py` (already written)

Refactors the two figures from script 11 into reusable functions. Takes `curves` as `{method_name: array[runs, evals, n_tasks]}`.

```python
plot_learning_curves(steps, curves, methods, out_path, title="",
                     switches=None, ncols=2, task_labels=None)
    # accuracy of each task vs training step; thin = runs, thick = mean;
    # `switches` draws dashed vlines at task boundaries (supports the
    # alternating schedule needed for the Bogacz reproduction)

plot_trajectory(curves, methods, out_path, title="", ncols=2)
    # 2-task only: path through (task1, task2) accuracy space,
    # with the equal-tradeoff diagonal and a marker at the final point
```

### 5.5 Current experiment configuration (script 11)

```python
IMG_SIZE = 14;  N_RUNS = 10;  ITERS = 100 per task;  BATCH = 32
BP_LR = 0.05
RP_LR, RP_PER_CLASS      = 0.05, 20
EQP_LR, EQP_BETA, EQP_DT = 0.005, 0.3, 0.3;  EQP_MAX_STEPS, EQP_SETTLE_PAT = 500, 30
PC_LR, PC_DT, PC_STEPS   = 0.05, 0.1, 50
# tasks: 2 classes each, single switch at step 100, Class-IL (single head)
```

### 5.6 Empirical results so far (10 random digit pairings)

| Method | Crossover | Final task 1 | Final task 2 | Shape |
|---|---|---|---|---|
| backprop | ~65% | ~0% | ~97% | vertical cliff at the switch |
| replay | ~85% | **~68%** | ~96% | dips then **recovers**; only method ending up-right |
| eqprop | — | ~0% | ~95% | noisy throughout; forgets before it learns |
| pc | ~75% | ~8–10% | ~97% | **slope, not cliff**; bows above the diagonal, still ends top-left |

**Read this as: PC = graceful degradation, replay = actual retention.**

---

## 6. References (verified)

**Core (in project files):**
- **Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz (2024).** "Inferring neural activity before plasticity: a foundation for learning beyond backpropagation." *Nature Neuroscience.* — **Fig 1** (interference within a single association); **Fig 4d–e** (continual learning, Fashion-MNIST 5+5 alternating — the reproduction target); **Fig 4f–g** (concept drift, largest advantage); **Supplementary Fig 6** (detecting which weights to modify); **Supplementary Fig 7** (less erratic updates).
- **Dong & Wu** — commentary on Song & Bogacz. EM framing (E-step inference / M-step plasticity); **weak clamp (EqProp) vs strong clamp (PC)**; exact-backprop equivalence "is not general"; costs — expensive relaxation phase, requires (approximately) symmetric weights.
- **Kirkpatrick et al. (2017).** EWC, *PNAS.* Diagonal Fisher importance; biological motivation from dendritic-spine persistence (enlarged spines survive later learning; erasing them causes forgetting).
- **van de Ven, Tuytelaars & Tolias (2022).** "Three types of incremental learning," *Nature Machine Intelligence.* Task-/Domain-/Class-IL taxonomy; EWC & SI fail on Class-IL; replay robust across all three.

**Predictive coding in ML:**
- Whittington & Bogacz (2017), *Neural Computation* — PC **approximates** backprop with local Hebbian plasticity.
- Millidge, Salvatori, Song, Bogacz & Lukasiewicz (2022), IJCAI survey — "Predictive Coding: Towards a Future of Deep Learning Beyond Backpropagation?" arXiv:2202.09467. PC as a general-purpose algorithm using only local computations; motivation is parallelisability / neuromorphic hardware.
- Millidge et al. (2022), arXiv:2206.02629 — "Backpropagation at the Infinitesimal Inference Limit of Energy-Based Models: Unifying Predictive Coding, Equilibrium Propagation, and Contrastive Hebbian Learning."
- Millidge, Tschantz & Buckley (2022), *Neural Computation* — PC approximates backprop along arbitrary computation graphs.
- Song et al. (2020), NeurIPS — "Can the brain do backpropagation? Exact implementation of backpropagation in predictive coding networks."
- Salvatori et al. (2024), ICLR — "A stable, fast, and fully automatic learning algorithm for predictive coding networks."

**Energy-based models in mainstream ML:**
- Ramsauer et al. (2021), ICLR — "Hopfield Networks is All You Need," arXiv:2008.02217. Transformer self-attention **is** the update rule of a modern continuous-state Hopfield network.
- Bai, Kolter & Koltun (2019), NeurIPS — Deep Equilibrium Models. "Settle to a fixed point, then differentiate implicitly" as mainstream ML.
- Scellier & Bengio (2017) — Equilibrium Propagation.
- Laborieux et al. (2021), *Front. Neurosci.* — "Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing Its Gradient Estimator Bias." Finite-nudging bias is why vanilla EqProp doesn't scale past MNIST.
- Li, Du, van de Ven & Mordatch (2020) — "Energy-based models for continual learning."

**Consolidation / metaplasticity:**
- Zenke, Poole & Ganguli (2017), ICML — Synaptic Intelligence.
- Laborieux, Ernoult, Hirtzlin & Querlioz (2021), *Nature Communications* — "Synaptic metaplasticity in binarized neural networks."
- Benna & Fusi (2016) — cascade / complex-synapse models of memory.
- McCloskey & Cohen (1989) — original catastrophic-interference result.

---

## 7. Suggested opening prompt for the new workspace

> I'm working on an MSc thesis comparing backprop, replay, Equilibrium Propagation, and Predictive Coding (prospective configuration) on catastrophic forgetting. Attached is a context summary plus my `src/` code. Next task: write experiment 12, a minimal reproduction of Song & Bogacz 2024 Fig 4d–e — alternating 5+5 class-incremental split, PC vs backprop **matched on tanh + squared error**, run on MNIST and Fashion-MNIST, per-rule LR grid search. Follow the conventions in §4.3 of the summary: one research question per script, constants at the top, thin script with reusable logic in `src/`.
