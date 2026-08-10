# Timeline

Sequential, append-only log of chats. One entry per objective; a chat with several
objectives is split into several entries. **Entries are never edited once added.**
Corrections go in `knowledge_base.md`, not here.

Format:

```
### NNN — YYYY-MM-DD — short title
**Objective:** …
**Outcome:** … (one sentence)
```

Entries 001–012 are **reconstructed** from the three context-transfer documents
(`continual-learning-ebm-project.md`, `understanding-ffnn-bp-ebm-pcn-pc-eqprop.md`,
`energy-based-memory-and-continual-learning.md`). Their dates are unknown and their
boundaries are inferred, so they are ordered logically rather than chronologically.
Everything from 013 onward is recorded live.

---

### 001 — (reconstructed) — MNIST continual-learning baselines
**Objective:** Establish whether backprop forgets catastrophically on split MNIST and whether standard defences fix it.
**Outcome:** Task-IL did not forget at all, Class-IL collapsed to ~21.6% mean with individual tasks falling to ~0%, replay recovered to ~78% and EWC failed at ~20%, reproducing van de Ven et al.'s scenario ranking.

### 002 — (reconstructed) — Gradient interference measurement
**Objective:** Test whether Class-IL forgetting corresponds to measurable gradient conflict between tasks.
**Outcome:** Cosine similarity between current-task and prior-task gradients was predominantly negative throughout Class-IL training, with the sign↔Δloss identity holding exactly under plain SGD.

### 003 — (reconstructed) — EqProp implementation and joint-training validation
**Objective:** Get Equilibrium Propagation training as the project's chosen energy-based model.
**Outcome:** EqProp reached ~91% validation on joint 14×14 MNIST in ~6 minutes for 3 epochs on CPU, with tanh saturation identified as its dominant failure mode.

### 004 — (reconstructed) — Scope reset against the advisor's four-point outline
**Objective:** Cut scope sprawl down to a deliverable dissertation plan.
**Outcome:** Adopted the advisor's four points (pick one EBM; compare forgetting against backprop; explain the difference; try to reduce it) and formally deferred generative replay, VAE example-ordering, the scenario×dataset grid, other EBM families and efficiency comparisons.

### 005 — (reconstructed) — First Class-IL sweeps across task splits
**Objective:** Characterise forgetting for backprop, EqProp and replay across 10×1, 5×2 and 2×2 Class-IL splits.
**Outcome:** Backprop and EqProp both sat at the collapse floor (~10% / ~20%) while replay reached ~64% / ~60%, establishing that the problem is solvable and that EqProp alone does not solve it.

### 006 — (reconstructed) — Predictive coding added as the second EBM
**Objective:** Implement the learning rule that actually makes the "energy-based learning reduces interference" claim.
**Outcome:** PC was implemented with all three gradients finite-difference verified to ~1e-9, on the corrected understanding that the interference claim belongs to prospective configuration rather than to EqProp.

### 007 — (reconstructed) — Energy-based memory / CLS hypothesis review
**Objective:** Assess whether a single EBM could carry both a sharp per-sample memory component and a smooth generalising component.
**Outcome:** Judged plausible and partially precedented — diffusion energies already interpolate the two regimes via noise scale, and VAE+MHN systems instantiate the split across two modules — with the single-model decomposed-energy version identified as the genuinely novel part.

### 008 — (reconstructed) — Continual-learning taxonomy consolidation
**Objective:** Name the project's target regime unambiguously against the competing taxonomies.
**Outcome:** Separated three orthogonal axes (data shift, IL scenario, stream carving) and fixed the target regime as Class-IL under a task-free stream.

### 009 — (reconstructed) — Mechanism analysis of Class-IL forgetting
**Objective:** Work out where forgetting actually lives in the architecture.
**Outcome:** Derived the two-pathology decomposition — logit suppression (a calibration failure, cheap to fix) versus absent inter-class discriminative signal (a representation failure, needs old-class information) — and identified the one-hot target rather than the softmax as the source of suppression.

### 010 — (reconstructed) — Four-method comparison interpreted
**Objective:** Interpret backprop / replay / EqProp / PC across 10 random digit pairings on split MNIST.
**Outcome:** PC gave the best trade-off path but no retention floor, replay gave a floor without reducing interference, and EqProp came out worst overall — read as EqProp being a noisy finite-difference estimator of backprop's gradient.

### 011 — (reconstructed) — Conceptual rebuild against primary sources
**Objective:** Obtain a mechanical, non-metaphorical account of FFNN / BP / EBM / PCN / PC / EqProp grounded in the papers.
**Outcome:** Established that an FFNN is a PCN with its internal state frozen to the forward pass, and corrected two earlier claims (PC ≈ BP only in the small-step limit; zero prediction error does not explain cross-task mitigation).

### 012 — (reconstructed) — Code review and separated experiment plan
**Objective:** Verify the existing codebase and lay out one-question-per-script experiments.
**Outcome:** Code verified correct, one confound identified (backprop/replay use ReLU + cross-entropy while PC/EqProp use tanh + squared error), and experiments 12–19 specified with the Song & Bogacz Fig 4d–e reproduction as the immediate next step.

### 013 — 2026-07-29 — Quarantine list triage
**Objective:** Assess the accumulated quarantine list of research ideas, group it into experiments, define high-level research plans, identify what the literature already answers, and order the work into a sequential story.
**Outcome:** Seventeen items were sorted into five research programmes, seven were judged already answered or superseded by cheaper reproductions, two were dropped as architecture-blocked or advisor-deferred, and a five-tier ordering was defined with the head-vs-trunk probe and the target-structure control identified as the highest-value next experiments.

### 014 — 2026-07-29 — Knowledge management set-up
**Objective:** Establish `timeline.md` and `knowledge_base.md` as the project's persistent record.
**Outcome:** Both files were created, with the timeline seeded from the three context-transfer documents and the knowledge base consolidating scope, decisions, results, mechanism analysis, the quarantine triage, constraints and a full reference list.
