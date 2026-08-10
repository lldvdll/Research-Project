# Continual Learning + Energy-Based Models — Project Plan

**Aim:** Use **Split MNIST** as a controlled testbed to (1) observe catastrophic *forgetting*, (2) measure *interference* with interpretability, (3) set up a fair backprop-vs-EBM comparison.

**How we work:** one stage at a time → concept (ELI5 + grad) → decisions + trade-offs → code cell-by-cell → refactor to `src/` only when a function earns it.

---

### Stage 0 — Data & setup *(gentlest start)*
- **Goal:** load MNIST, make seeded digit-pair splits, fix train/test, sanity-check.
- **Core:** Split MNIST protocol (van de Ven 2022).
- **Decide:** random vs fixed pairing; treat pairing as part of the seed.

### Stage 1 — How big a net does MNIST need? *(architecture)*
- **Goal:** pick the smallest MLP that solves MNIST, eyes open.
- **Core:** capacity sweep (accuracy vs params, find the knee); intrinsic dimension (TwoNN / PCA participation ratio); objective intrinsic dim (Li 2018, *optional*).
- **Decide:** tight vs roomy net; depth × width grid; how much rigour (sweep only, or + ID).
- **Watch:** "optimal for the task" ≠ "optimal for continual learning"; tight net = more visible interference + cheaper EBM relaxation later.

### Stage 2 — Incremental learning & forgetting *(behavioural baseline)*
- **Goal:** train pairs sequentially; log the full accuracy matrix over time.
- **Core:** scenarios — Task-IL vs Class-IL (van de Ven 2022); accuracy matrix R + ACC/BWT/FWT (Lopez-Paz & Ranzato 2017); "none"/"joint" baselines; EWC as a reference method (Kirkpatrick 2017).
- **Decide:** which scenario(s); fixed iterations vs convergence (→ fixed); reset optimizer state at task boundaries?; number of seeds.
- **Watch:** dual measurement — live-head accuracy **and** frozen-feature linear probes (separates "features forgot" from "readout collapsed"); log loss too; log every ~25–50 iters.

### Stage 3 — Interpretability: interference, drift, persistence
- **Goal:** open the box; tell *mechanistic interference* apart from *behavioural forgetting*.
- **Core:** update-direction / gradient cosine = "target alignment" (Song 2024); Fisher importance (EWC) + SI path-integral ω = "energy cost"; CKA/RSA for representational drift; ablation + unit selectivity for "circuits"; cross-ordering consistency + weight-graph modularity for order-agnostic structure.
- **Decide:** which tier to attempt (start Tier 1); checkpoint frequency.
- **Watch:** TDA / persistent homology = deferred & skeptical; real circuit work → CNN phase.

---

### Later (noted, not now)
- **Phase 4 — EBM swap:** predictive coding / prospective configuration (Song 2024) *or* EBM-as-classifier (Li/Du/van de Ven/Mordatch). Energy becomes literal; per-method LR tuning required; relaxation is costly (Dong/Wu 2025 commentary).
- **Phase 5 — CNNs:** kernels more legible → circuit analysis; watch BatchNorm cross-task leakage.

---

**Cross-cutting decisions (pick when each becomes relevant):** primary scenario · EBM flavour · compute budget (GPU?).