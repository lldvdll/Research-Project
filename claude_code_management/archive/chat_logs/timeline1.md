# Timeline

Sequential record of chats on this project.

**Rules for this file**
- One entry per *objective*. A chat with several objectives is split into several entries.
- Each entry: objective (what it set out to do) + outcome (what it produced), one sentence each.
- **Entries are append-only. Once written, an entry is never edited or deleted.** Corrections are made by adding a later entry that supersedes an earlier one, referencing it by ID.
- Loaded into context only when chronology matters. For "what do we currently believe", use `knowledge_base.md`.

**Label scheme.** `S#` groups entries by session (one past chat). `T###` is an entry, assigned in order of addition. These are the only labels used in this file. The knowledge base uses `D#` (discrepancy), `CTRL-#` (control) and `H#` (hypothesis) — see `knowledge_base.md` §0.3 for the full key.

*Note: session labels were `C0`–`C4` when first written; renamed to `S0`–`S4` to avoid a clash with the knowledge base's `C1`–`C6` controls, which are now `CTRL-1`–`CTRL-6`.*

**Chronology caveat:** sessions S0–S3 were reconstructed from three handoff documents, not from live logs. Ordering of S2 and S3 is *inferred* from internal evidence (S3 verifies code and issues corrections to claims made in S2). Treat the S2/S3 boundary as provisional.

---

## S0 — Baseline notebook phase (`00_mnist_baseline.ipynb`, 28×28, Adam)
*Source: `continual-learning-ebm-project.md` §8*

**T001**
- **Objective:** Establish whether catastrophic forgetting occurs on split MNIST, and in which incremental-learning scenario.
- **Outcome:** Task-IL showed no forgetting under backprop (an apparent early effect was traced to a polarity artefact of a balanced 2-way head, fixed with `max(acc, 1−acc)`), whereas Class-IL collapsed to ~21.6% final mean accuracy with individual tasks driven to ~0%.

**T002**
- **Objective:** Measure gradient interference directly during Class-IL.
- **Outcome:** Cosine similarity between current-task and prior-task gradients was predominantly negative, with occasional positive (cooperative) cosine coinciding with slower new-task learning, and the sign↔Δloss identity confirmed exact under plain SGD.

**T003**
- **Objective:** Verify that Class-IL forgetting is solvable at all, using replay as a positive control.
- **Outcome:** Replay reached ~78% versus ~21% for the naive baseline, establishing that the problem is solvable but that buffer-limited replay is not equivalent to joint training.

**T004**
- **Objective:** Test EWC in Class-IL.
- **Outcome:** EWC scored ~20%, indistinguishable from the no-defence baseline and reproducing van de Ven et al. (2022) Table 2, with the unresolved hypothesis that the failure lies in readout/arbitration rather than features — the proposed linear-probe test on the EWC trunk was never run.

**T005**
- **Objective:** Check whether training for more than one pass over the task sequence changes the forgetting picture.
- **Outcome:** More than one pass is cyclic revisiting rather than longer continual learning; baseline mean drifts from ~20% to ~30% and then plateaus, reflecting residual output structure once every class has been trained, not learning-to-not-forget.

---

## S1 — Project handoff: scope, substrate, first EBM results
*Source: `continual-learning-ebm-project.md`*

**T006**
- **Objective:** Resolve a period of scope sprawl and fix an authoritative project scope.
- **Outcome:** The advisor's four-point outline was adopted as authoritative (pick one EBM; compare its forgetting to backprop; explain why they differ, including trivial explanations such as sparsity and network size; try to reduce forgetting in the EBM by selecting the nodes whose predictions differ most), with generative/synthetic replay, VAE example-ordering, the 3×3 scenario×dataset grid, other EBM families, and efficiency comparisons all explicitly deferred.

**T007**
- **Objective:** Fix an experimental substrate that isolates the learning rule.
- **Outcome:** Settled on MNIST downsampled to 14×14 (196 inputs), a single-hidden-layer MLP 196→64→10, plain SGD everywhere, Class-IL evaluation with a single 10-way head, and three splits (10×1, 5×2, 2×2), with CNNs rejected to avoid inductive priors.

**T008**
- **Objective:** Implement Equilibrium Propagation as the chosen EBM and get it training.
- **Outcome:** EqProp reached ~91% validation on joint 14×14 MNIST (~6 min for 3 epochs on CPU), with four failure modes documented as hard-won constraints: tanh saturation severing the feedback path, the nudged phase never reaching an absolute tolerance (hence patience-based settling), collapse at batch size 1, and no ½x² self-term making it a weak generator.

**T009**
- **Objective:** Add Predictive Coding, on the grounds that the "energy-based learning reduces interference" claim in the reading is PC's claim and not EqProp's.
- **Outcome:** PC was implemented with all three gradients (∂F/∂x₁, ∂F/∂W₁, ∂F/∂W₂) verified against finite differences to ~1e-9 and a numpy mirror learning a toy 3-class problem to 100%, though the torch version had not been run end-to-end at the time of handoff.

**T010**
- **Objective:** Compare backprop, replay, EqProp and PC on Class-IL forgetting at three split granularities.
- **Outcome:** At 10×1 and 5×2 every method except replay collapsed to the floor (10% / 20% versus replay's ~64% / ~60%), while the 2×2 run proved most informative: backprop held task 1 until the switch then fell off a cliff, EqProp forgot *before* it learned, PC decayed as a slope, and replay retained both tasks at ~95%.

**T011**
- **Objective:** Explain why every learning rule forgets in Class-IL.
- **Outcome:** Forgetting was attributed to output competition in the shared head rather than to credit assignment — a local learning rule changes *how* credit is assigned, not *that* the outputs compete — with EqProp worst because its hinge target is −1 for every non-target class versus PC's one-hot 0, making its suppression more aggressive than softmax's.

**T012**
- **Objective:** Identify what would invalidate the comparison.
- **Outcome:** The learning rates were found to differ across methods (BP 0.05, EqProp 0.005, PC 0.05), meaning forgetting speed was being compared while learning speed differed, so `to_learn` must be matched before any `to_forget` difference can be attributed to the learning rule.

**T013**
- **Objective:** Adopt working practices that survive periods of overwhelm.
- **Outcome:** Fixed on one question per experiment written as a single sentence beforehand, backprop and replay controls on every forgetting experiment, doubts getting one scheduled test then closing, one script producing one identically-named figure, and the deliberate deletion of wandb, Optuna, class hierarchies and a shared `harness.py`.

---

## S2 — Energy-based memory, taxonomy, and forgetting mechanics
*Source: `energy-based-memory-and-continual-learning.md`*

**T014**
- **Objective:** Test whether a single EBM could carry a sharp/true-to-sample component (for generative replay) and a smooth/interpolated component (for discrimination) at the same time.
- **Outcome:** Judged plausible and partly precedented — the memorisation↔generalisation transition is controlled by inverse temperature β in dense associative memory, and a diffusion model is already a single energy carrying both regimes indexed by noise scale σ — with the project's novelty narrowed to a *single decomposed energy* rather than the usual two-module CLS instantiations (MHN + VAE, generator + classifier).

**T015**
- **Objective:** Get precise about continual-learning taxonomy so the target regime could be named unambiguously.
- **Outcome:** Three *orthogonal* axes were separated (data shift — Moreno-Torres; IL scenario — van de Ven; stream carving — Maltoni & Lomonaco), the taxonomies were shown to genuinely disagree (permuted MNIST is MT for Lomonaco but explicitly Domain-IL and *not* Task-IL for van de Ven), and the project's target regime was fixed as **Class-IL under a task-free stream**.

**T016**
- **Objective:** Work out where forgetting actually lives in the architecture.
- **Outcome:** Derived ∂L/∂z_o = p_o − 1[o=t] and identified the *active set* 𝒜 as the real knob (scenario → 𝒜 → which gradient paths exist → which methods can possibly work), then decomposed Class-IL forgetting into two distinct pathologies: logit suppression / task-recency bias, which is a calibration failure and cheap to fix without replay; and the absent inter-context discriminative signal, which is a representation failure and irreducible without information about old classes.

**T017**
- **Objective:** Test the proposal that replacing softmax with a linear or independently-activated output would remove normalisation-based forgetting.
- **Outcome:** Refuted as stated — the suppression comes from the one-hot target supplying zero for every absent class, not from softmax, so linear+MSE and sigmoid+BCE still push absent logits down — but a real partial gain was identified in that MSE's suppression has a fixed point at zero (driving w_o merely orthogonal to current features) whereas softmax's competition is unbounded, and the deeper tension was named: more output coupling buys calibration and costs suppression, and no single knob wins both.

**T018**
- **Objective:** Interpret the four-method run (backprop, replay, EqProp, PC × 10 random digit pairings).
- **Outcome:** A premise correction was issued — EqProp is *not* forgetting less, it is simply slower throughout, and the trajectory plot with time removed shows it as the worst panel — and two orderings were found to disagree: trade-off efficiency (area above the diagonal) ran pc > replay > backprop > eqprop while final task-1 retention ran replay ≫ pc > eqprop > backprop, with that divergence identified as the whole story.

**T019**
- **Objective:** Give each method's forgetting signature a mechanism.
- **Outcome:** Backprop's cliff was attributed to output-layer suppression plus cross-layer interference (Song & Bogacz's target alignment); replay's floor to replayed old samples arriving as *positives* that directly cancel the suppression term and restore minibatch co-occurrence; PC's above-diagonal path with no floor to it attacking trunk interference while doing nothing about one-hot zeros; and EqProp to being a finite-difference estimator of backprop's gradient — weak clamp with β→0 suppresses the very activity shift that *is* prospective configuration — so scoring below backprop is the expected result and not an anomaly.

---

## S3 — Mechanical understanding, code verification, corrections
*Source: `understanding-ffnn-bp-ebm-pcn-pc-eqprop.md`*

**T020**
- **Objective:** Rebuild a mechanical, non-metaphorical account of how FFNN, backprop, EBM, PCN, PC and EqProp relate, grounded in primary sources.
- **Outcome:** Settled three load-bearing statements — an FFNN *is* the PCN with its internal state clamped to the forward pass (clamp every hidden x_l to its feedforward value and the PCN energy reduces exactly to the FFNN output loss); every energy-based rule runs an EM-like two-step with a fast activity loop inside a slow weight loop, which backprop lacks entirely; and the global energy is never computed, stored or transmitted by the network, being only a Lyapunov function used by the analyst.

**T021**
- **Objective:** Verify the existing implementation against the equations before trusting any result from it.
- **Outcome:** `predictive_coding.py`, `eqprop.py`, `methods.py`, `data.py` and `11_consolidate_pairs_4methods.py` were all confirmed correct, including the sign of ∂F/∂W (so `+=` is descent on energy), the hinge nudge as β·∂/∂y of max(0, 1 − target·y), and the warm-started nudged phase — meaning EqProp's noisiness is EqProp being EqProp, not a bug.

**T022**
- **Objective:** Re-examine two explanations that had been given earlier for PC's advantage.
- **Outcome:** Both were corrected — (A) "PCN error nodes are basically backprop's errors" holds only under partial relaxation / infinitesimal nudging or under engineered exact equivalence with specific initialisation and update schedule, and is false at full relaxation, which is precisely Song et al.'s contribution; and (B) "an already-correct output has zero error so its weights don't move" is a within-a-single-example statement that gives task 1 no protection whatsoever during task-2 training, superseding the mechanism asserted in T019.

**T023**
- **Objective:** Replace the refuted mechanism with something testable.
- **Outcome:** Because `x1` is initialised to `mu1`, the residual `e1` after settling is literally the displacement of the hidden layer from its feedforward value, so ΔW1 is proportional to that displacement — reframing the question from "is the error zero?" (no) to "where does the weight movement go?" (measurable), and making the hypothesis *PC interferes with task 1 only to the extent that satisfying task 2's target forces movement in the hidden units task 1 depends on*.

**T024**
- **Objective:** Audit the four-method comparison for confounds beyond learning rate.
- **Outcome:** Found that `make_backprop` and `make_replay` use ReLU + cross-entropy while `pc` and `eqprop` use tanh + squared error, so the comparison varies algorithm, nonlinearity and loss simultaneously; a matched BP control was specified as `x1 = x0 @ W1; out = tanh(x1) @ W2; loss = ½|target − out|²`, with no biases.

**T025**
- **Objective:** Lay out a separated experiment sequence, one research question per script.
- **Outcome:** Queued script 12 (reproduce Song & Bogacz Fig 4d–e: alternating 5+5 class-incremental split, matched tanh + squared error, MNIST and Fashion-MNIST, per-rule LR grid search) followed by 14 (does task 1 live in a small subset of hidden units), 15 (is importance correlated with magnitude), 16 (where does the weight movement go — the decisive test), 17 (freezing control), 18 (does EWC stack on PC), 19 (metaplasticity), with the "kinematic consolidation" idea quarantined until 16 reports.

---

## S4 — Consolidation and triage
*This conversation.*

**T026**
- **Objective:** Diagnose the sense of information overload and of results that might conflict.
- **Outcome:** The confusion was traced to collapsing three independent axes that the literature treats as one — credit assignment (BP vs PC/EqProp), continual-learning mitigation (none / EWC / replay / ...), and IL scenario (Task/Domain/Class) — compounded by a metric mismatch (Song & Bogacz report *mean test error during training*, van de Ven reports *final average accuracy*, so their numbers were never comparable) and by numeric disagreements between the three handoff documents, catalogued as D1–D6 in the knowledge base.

**T027**
- **Objective:** Extend the reference base with work not yet in the project.
- **Outcome:** Added Pinchetti et al. (ICLR 2025, PCX) as the fair-comparison and scaling authority — PC rivals backprop on VGG-7-scale networks but degrades on 9-layer convnets and ResNets where backprop improves — plus Ororbia's sequential neural coding networks and BayesPCN as the closest direct prior work on PC for continual learning, and Lopez-Paz & Ranzato for formal backward-transfer metrics.

**T028**
- **Objective:** Create and populate the two maintained project files.
- **Outcome:** `timeline.md` and `knowledge_base.md` were written from the three consolidated chat documents plus the four in-project PDFs, with all numeric disagreements preserved and flagged rather than reconciled by guesswork.

**T029**
- **Objective:** Determine whether the four methods actually use different output-layer structures in the code, and whether that can be reduced to a single architecture.
- **Outcome:** Three simultaneous differences were confirmed from the recorded code review — activation function (ReLU on the way in versus tanh on the way out), output stage and loss (softmax + cross-entropy versus linear + squared error versus free settling + hinge), and label coding (0 versus −1 for non-target classes) — and a single shared specification was proposed (no biases, linear hidden pre-activation, tanh on the way out, one-hot 0/1 target, squared error, plain SGD), chosen because squared error is the only loss all three learning rules can accept without being altered.

**T030**
- **Objective:** Establish where catastrophic forgetting actually lives in this network — the hidden layer or the output layer — and whether the distinction matters.
- **Outcome:** Three independent observations already in our own data (Task-IL barely forgets; Class-IL task-1 accuracy falls to 0% rather than 50%; replay alone retains anything) point to the output layer as the dominant damage site in this set-up, which implies no learning rule can fix it and makes the nearest-class-mean probe a decision point rather than an optional extra, since it also determines whether the advisor's node-selection idea is aimed at the right layer.

**T031**
- **Objective:** Make the knowledge base easier to read and maintain — simpler language, defined terminology, consistent labels, and a visual map.
- **Outcome:** Added §0 (two Mermaid diagrams and a complete key to every label) and §14 (a full glossary in plain English), expanded the MT/SIT/MIT abbreviations, rewrote the parked metaplasticity idea in plain terms with a comparison table, renamed the confound labels `C1`–`C6` to `CTRL-1`–`CTRL-6` to remove a clash with the timeline's session labels, and confirmed by line-level comparison against a backup that no existing content was lost.
