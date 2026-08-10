# Research Plan — Continual Learning & Energy-Based Models
*A from-scratch build guide: what each notebook does, in what order, with which code. Companion file: `research_log.md` (results & observations). Numbering: experiments = `section.notebook.experiment` (e.g. `1.2.3`); code = `file.function` (e.g. `data.load`).*

---

## 0. Overview
**Question:** How does catastrophic forgetting arise when a network is trained on tasks in sequence, and how well do different mitigations — replay, weight regularisation (EWC), and energy-based models — reduce it?

**Setup:** sequential task training on one small MLP (no CNNs → no inductive prior → general observations). Methods compared across a 3×3 grid.
- **Scenarios (3)** [van de Ven]: **Task-IL** (task id given at test) · **Domain-IL** (same labels, shifting input) · **Class-IL** (all classes, no task id — hardest).
- **Datasets (3)**, 28×28 grayscale, increasing difficulty: **MNIST · FashionMNIST · KMNIST**.
- **Methods:** none (baseline) · replay (ER) · EWC · EBM ×3 (predictive coding, equilibrium propagation, contrastive/energy-classifier).
- **Metrics:** accuracy (final & mean; ACC/BWT/FWT) · gradient cosine (interference) · Fisher importance · linear-probe decodability (features vs readout) · weight-space trajectory (EBM interpretability).
- **Optimiser:** SGD (clean interference identity; fair comparison with the EBMs' SGD updates).

---

## 1. Notebooks

### 1.0 — `00_baseline` — Foundations
**Aim:** build the data/model/training foundation; demonstrate forgetting and its cause on one dataset.
- 1.0.1 Data pipeline: load, split into sequential tasks, sanity-check.
- 1.0.2 Capacity: joint-training ceiling + width sweep → choose network size.
- 1.0.3 Baseline: sequential training, no mitigation → catastrophic forgetting.
- 1.0.4 Interference: gradient cosine, current vs prior tasks.
**Tools:** `data.load` `data.make_pairing` `data.context_indices` `models.make_mlp` `train.fit` `train.evaluate` `train.run_continual` `train.eval_task` `strategies.Strategy` `interference.measure` `metrics.accuracy_matrix` `viz.forgetting_curves`

### 1.1 — `01_forgetting_demonstration` — Forgetting across the grid
**Aim:** demonstrate forgetting across all 3 scenarios × 3 datasets with the baseline model.
- 1.1.1 Scenario construction: label/task-id schemes for Task-IL, Domain-IL, Class-IL.
- 1.1.2 Baseline across the 3×3 grid.
- 1.1.3 Seed-averaging → mean + spread.
- 1.1.4 Summarise per cell (ACC/BWT/FWT).
**Tools:** `data.load` `data.scenario` `models.make_mlp` `train.run_continual` `strategies.Strategy` `metrics.acc_bwt_fwt` `viz.grid`

### 1.2 — `02_replay_exploration` — Replay
**Aim:** explore replay variants/parameters; select the best replay across the grid.
- 1.2.1 Experience replay (replayed loss added to current loss).
- 1.2.2 Buffer-size sweep.
- 1.2.3 Replay-ratio sweep.
- 1.2.4 Sample selection: does *which* example matter? (random vs central vs boundary; position in latent space).
- 1.2.5 Fixed-total buffer (reservoir) control.
- 1.2.6 Best replay across the 3×3.
**Tools:** `strategies.Replay` `memory.Buffer` `train.run_continual` `metrics.acc_bwt_fwt` `viz.forgetting_curves`

### 1.3 — `03_ewc_exploration` — EWC
**Aim:** refine EWC; understand why it helps in some scenarios and not others.
- 1.3.1 EWC (per-task diagonal Fisher penalty).
- 1.3.2 λ sweep (retention vs plasticity).
- 1.3.3 Online EWC / Fisher-decay variant.
- 1.3.4 Why it fails in Class-IL: linear probe on trunk (features vs readout).
- 1.3.5 Importance distribution across tasks, boundaries, seeds.
- 1.3.6 Best EWC across the 3×3.
**Tools:** `strategies.EWC` `fisher.diagonal` `fisher.importance_map` `probes.decode` `train.run_continual` `metrics.acc_bwt_fwt` `viz.importance_maps`

### 1.4 — `04_ebm_eqprop` — EBM: Equilibrium Propagation
**Aim:** get EqProp converging, adapt to continual learning, find its best configuration.
- 1.4.1 Joint convergence (init, clipping, LR, settling diagnostics).
- 1.4.2 Adapt to sequential CL.
- 1.4.3 Sweeps (β, dt, steps, LR).
- 1.4.4 Interference/trajectory vs backprop.
**Tools:** `ebm_eqprop.energy` `ebm_eqprop.settle` `ebm_eqprop.update` `ebm_eqprop.run` `interference.measure` `landscape.trajectory` `metrics.acc_bwt_fwt` `viz.forgetting_curves`

### 1.5 — `05_ebm_pc` — EBM: Predictive Coding  *(build FIRST)*
**Aim:** implement predictive coding as the first working EBM; compare to baseline.
- 1.5.1 Joint convergence (inference steps, nonlinearity, energy form).
- 1.5.2 Adapt to sequential CL.
- 1.5.3 Sweeps.
- 1.5.4 Interference/trajectory vs backprop (does inference-first reduce interference?).
**Tools:** `ebm_pc.settle` `ebm_pc.update` `ebm_pc.run` `interference.measure` `landscape.trajectory` `metrics.acc_bwt_fwt` `viz.forgetting_curves`

### 1.6 — `06_ebm_contrastive` — EBM: Contrastive / energy-classifier
**Aim:** implement a per-class energy classifier (contrastive divergence); compare to baseline.
- 1.6.1 Energy model + negative sampling.
- 1.6.2 Joint convergence.
- 1.6.3 Adapt to sequential CL.
- 1.6.4 Sweeps.
**Tools:** `ebm_contrastive.energy` `ebm_contrastive.sample` `ebm_contrastive.run` `metrics.acc_bwt_fwt` `viz.forgetting_curves`

### 1.7 — `07_comparisons` — Main comparison
**Aim:** compare all methods across the 3×3 grid at their best configurations. **This is the main result.**
- 1.7.1 Assemble best config per method.
- 1.7.2 Full grid run (scenarios × datasets × methods × seeds).
- 1.7.3 Core metrics per cell (accuracy, ACC/BWT/FWT).
- 1.7.4 Interference/interpretability comparison (esp. EBM vs backprop).
- 1.7.5 Summary figures & tables.
**Tools:** all modules; `metrics.acc_bwt_fwt` `interference.measure` `viz.grid` `viz.comparison`

---

## 2. Code modules (`src/`)
*Principle: one job per function, few arguments, build incrementally. Cross-refs show where each is used.*

### 2.1 — `data.py` — datasets & task splitting
- `data.load(name)` — load train/test (name ∈ mnist|fashion|kmnist). → 1.0.1, 1.1.2, all
- `data.make_pairing(seed)` — assign classes to sequential contexts. → 1.0.1, 1.1.1
- `data.context_indices(dataset, pairing)` — per-context sample indices. → 1.0.1, 1.1.2
- `data.scenario(pairing, kind)` — labels/task-id scheme for task|domain|class-IL. → 1.1.1, 1.7

### 2.2 — `models.py` — networks
- `models.make_mlp(hidden, out_dim)` — flat MLP; `hidden=()` = linear. → 1.0.2, all backprop methods
- `models.MultiHeadMLP(hidden, n_tasks)` — shared trunk + per-task heads (Task-IL). → 1.1

### 2.3 — `train.py` — training & evaluation loops
- `train.fit(model, loader)` — standard (joint) training. → 1.0.2
- `train.evaluate(model, loader)` — accuracy on a loader. → 1.0.2
- `train.run_continual(model, sequence, strategy)` — sequential CL trainer; behaviour set by `strategy`. → 1.0.3, 1.1–1.3, 1.7
- `train.eval_task(model, task)` — per-task accuracy. → 1.0.3, all

### 2.4 — `strategies.py` — CL strategies (plug into `train.run_continual`)
- `strategies.Strategy` — base = none (no-op hooks: before_task / augment_batch / extra_loss / after_task). → 1.0.3, 1.1
- `strategies.Replay` — augment batch from buffer. → 1.2
- `strategies.EWC` — Fisher penalty + importance snapshot. → 1.3
- `strategies.Combined(*s)` — compose strategies. → 1.3, 1.7

### 2.5 — `memory.py` — replay buffer
- `memory.Buffer(policy)` — store/sample past examples (per-class or reservoir). → 1.2

### 2.6 — `fisher.py` — Fisher information / importance
- `fisher.diagonal(model, data)` — per-parameter empirical Fisher. → 1.3, 1.7
- `fisher.importance_map(fisher, layer)` — per-input-unit importance for viz. → 1.3.5

### 2.7 — `interference.py` — gradient conflict
- `interference.flat_grad(model, X, y)` — flattened loss gradient. → 1.0.4
- `interference.cosine(g1, g2)` — cosine between gradients. → 1.0.4
- `interference.measure(model, tasks)` — mean cosine vs prior tasks. → 1.0.4, 1.4.4, 1.5.4, 1.7.4

### 2.8 — `probes.py` — linear probes
- `probes.decode(model, data)` — linear decodability of frozen features. → 1.3.4, 1.7

### 2.9 — `ebm_pc.py` — predictive coding
- `ebm_pc.settle(...)` — infer activities by relaxation. → 1.5.1
- `ebm_pc.update(...)` — local weight update. → 1.5.1
- `ebm_pc.run(model, sequence)` — PC training (joint & CL). → 1.5

### 2.10 — `ebm_eqprop.py` — equilibrium propagation
- `ebm_eqprop.energy(...)` — energy function. → 1.4.1
- `ebm_eqprop.settle(...)` — free/nudged relaxation. → 1.4.1
- `ebm_eqprop.update(...)` — contrastive weight update. → 1.4.1
- `ebm_eqprop.run(model, sequence)` — EqProp training. → 1.4

### 2.11 — `ebm_contrastive.py` — energy-classifier
- `ebm_contrastive.energy(x, y)` — per-class energy. → 1.6.1
- `ebm_contrastive.sample(...)` — negative sampling (e.g. Langevin). → 1.6.1
- `ebm_contrastive.run(model, sequence)` — contrastive-divergence training. → 1.6

### 2.12 — `metrics.py` — CL metrics
- `metrics.accuracy_matrix(logs)` — per-task accuracy over time. → all
- `metrics.acc_bwt_fwt(matrix)` — ACC, backward & forward transfer [Lopez-Paz]. → 1.1.4, 1.7.3

### 2.13 — `landscape.py` — EBM trajectory/landscape interpretability
- `landscape.trajectory(checkpoints)` — weight-space path. → 1.4.4, 1.5.4
- `landscape.distance(a, b)` / `landscape.roughness(path)` — path geometry. → 1.4.4, 1.5.4, 1.7.4

### 2.14 — `viz.py` — plotting
- `viz.forgetting_curves(...)` — per-task + mean accuracy. → 1.0, 1.2–1.6
- `viz.grid(...)` — 3×3 scenario×dataset panels. → 1.1, 1.7
- `viz.importance_maps(...)` — Fisher maps + histograms. → 1.3.5
- `viz.comparison(...)` — methods on one axis. → 1.7

---

## 3. References & key concepts
**Scenarios & benchmark:** van de Ven, Tuytelaars & Tolias 2022, *Three types of incremental learning* (Nat Mach Intell).
**Replay:** Rebuffi 2017 (iCaRL) · Lopez-Paz & Ranzato 2017 (GEM; ACC/BWT/FWT) · Chaudhry 2019 (A-GEM).
**Regularisation:** Kirkpatrick 2017 (EWC) · Zenke 2017 (SI) · Schwarz 2018 (online EWC / Progress & Compress) · Mallya 2018 (PackNet) · Serra 2018 (HAT).
**EBMs:** Scellier & Bengio 2017 (Equilibrium Propagation) · Song & Bogacz (predictive coding / prospective configuration) · Li, Du, van de Ven & Mordatch 2022 (Energy-Based Models for Continual Learning).

**Key concepts:**
- *Catastrophic forgetting* — new learning overwrites old.
- *Interference* — opposing task gradients; measured by cosine (negative = conflict).
- *Stability–plasticity trade-off* — retain old vs learn new.
- *Fisher information* — per-weight importance ≈ curvature of the loss.
- *Contrastive divergence* (train an energy model: push energy down on data, up on model samples) ≠ *contrastive learning* (self-supervised representation method, e.g. SimCLR/InfoNCE).
