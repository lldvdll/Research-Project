# Research Log — Continual Learning & Energy-Based Models
*Results, observations, hypotheses, amendments. Companion: `research_plan.md`. Tags: **FACT** established · **HYP** hypothesis (unconfirmed) · **AMEND** plan change + reason · **Q** open question. Cross-refs use plan IDs (e.g. 1.0.3).*

---

## Amendments (project-level)
- **AMEND** Optimiser Adam → **SGD**. Exact interference identity (update = −η·g); no cross-task momentum confound; matches the EBMs' SGD updates for fair comparison. Adam only for paper-number reproduction.
- **AMEND** **No CNNs.** Avoid inductive priors so observations generalise. All datasets 28×28 grayscale, MLP only.
- **AMEND** EBM order: build **predictive coding first** (1.5 before 1.4). Most reliable to train + the project's conceptual core; de-risks EqProp.
- **AMEND** **Domain-IL cannot come from class-splitting** (1.1.1). Needs a consistent semantic across contexts (e.g. odd/even) or input shifts (rotations/permutations). Decide construction before building the grid.

---

## Findings

### Architecture (→ 1.0.2)
- **FACT** One hidden layer is sufficient on MNIST: a 2nd layer adds ~0.3% at large interpretability cost.
- **FACT** Accuracy-vs-width knee ~64–128. Carry two sizes: tight = 64 (~96%), roomy = 256 (~97.5%). Joint ceiling ~97–98%.
- **HYP** Extra width slightly reduces interference (more room for non-overlapping task solutions). [test across widths]

### Baseline forgetting (→ 1.0.3, 1.1)
- **FACT** Class-IL: catastrophic forgetting — each task ~99% in-band, then collapses to ~0% as later tasks train; final mean ~20% (vs ~96% joint). Collapse to ~0% (not 10% chance) = **active overwriting**, not passive decay.
- **FACT** Task-IL (per-task heads): **no forgetting** on easy data — features transfer, heads stay ~99% ("high–high"). Apparent forgetting with a single shared 2-unit head was a **polarity artifact** (balanced 2-way acc can't fall below 50%; fold with max(acc, 1−acc)).
- **FACT** Driver is the shared output layer + absence of task id, **not** trunk depth. Later tasks train slower / peak lower (crowded softmax); seed-dependent.

### Interference (→ 1.0.4)
- **FACT** Gradient cosine (current vs prior tasks) is predominantly **negative** (~−0.5–0) during Class-IL → new-task updates oppose prior tasks = the mechanism of forgetting.
- **FACT** Magnitude varies with per-pairing feature overlap; occasional **positive** (cooperative) cosine coincides with slower new-task learning.
- Note: sign↔Δloss identity is exact for SGD, approximate for Adam.

### Replay (→ 1.2)
- **FACT** Experience replay ≫ none: ~78% vs ~20% Class-IL (MNIST, 20/class). Mechanism: old-task loss enters the objective every step (counters interference directly).
- **FACT** Not equivalent to joint training (memory-limited): 20/class stored vs ~6000/class real; ends below the ~96% ceiling.
- **Q** Growing per-class buffer confounds cross-time comparison → add fixed-total (reservoir) control (1.2.5).
- **Q** If very small buffers work, does *which* example matter — centrality / class boundary / latent-space position? (1.2.4)

### EWC (→ 1.3)
- **FACT** EWC ≈ none in Class-IL (~20%), reproducing van de Ven Table 2. Task-boundary spikes grow with #tasks (transient resistance) but lose to the new-task gradient within a task.
- **FACT** Why it fails: (a) **deadlock** — protect old weights ⇒ can't learn new classes; learn new ⇒ softmax suppresses old-class logits; no λ escapes. (b) preserving each task's function doesn't create **discrimination between classes never seen together**.
- **HYP** Failure is in the **readout, not the features**: EWC's Fisher-importance distribution resembles replay's (distributed) yet accuracy ≈ none. TEST = linear probe on EWC trunk (predict: probe high, head low). (1.3.4) [not yet run]
- **FACT** Input-pixel Fisher maps are ~identical across none/replay/EWC → input importance ≈ which pixels discriminate digits (a dataset property). Method differences appear in the importance *distribution* and deeper layers.

### Multi-pass / cyclic revisiting (observational)
- **FACT** >1 pass = revisiting already-seen tasks (cyclic), **not** longer continual learning — changes what the curves mean; flag on any multi-pass run.
- **FACT** none mean drifts up over passes (~20 → ~30%) then plateaus (residual output structure once every class has been trained; not learning-to-not-forget). Replay plateaus ~75–80% once the buffer covers all classes.
- **HYP (parked → interpretability)** Cyclic replay may yield cleaner representations over passes — test by probe/CKA, **not** by a double-descent analogy (double descent = model size / training time in *stationary* training; N/A here).

---

## Open questions
- **Q** Does an energy-based *learning rule* (PC / EqProp) reduce interference vs backprop? Measure cosine + trajectory. (1.5.4, 1.4.4)
- **Q** Does an energy-based *classifier* (contrastive) beat replay/EWC in Class-IL, as reported by Li/Du/van de Ven? (1.6, 1.7)
- **Q** EWC dynamic-λ / Fisher-decay / capacity-allocation variants — worth exploring? (1.3.3; see plan refs Schwarz 2018, Mallya 2018, Serra 2018)

---

## Grid results
- **1.1+ pending** — per-cell forgetting, method comparisons, and metrics to be filled in as the grid is run.
