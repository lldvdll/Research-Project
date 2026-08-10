# Knowledge Base

Consolidated, structured record of all knowledge and observations for the MSc project.
**Loaded into context at the start of each response.** New information is added, old entries amended, corrections and consolidations made in place. Reference list at the end (§12).

*Last consolidated: 2026-08-10 (chat #009).*

---

## 1. Project definition

**Thesis:** An evaluation of energy-based / predictive-coding learning rules against backpropagation for catastrophic forgetting (CF) in continual learning. MSc, computational neuroscience & AI.

**Core question:** Does an energy-based model with a biologically plausible local learning rule reduce catastrophic forgetting compared to backprop — and if not, why not, and can it be made to?

**Authoritative scope — the advisor's four points** [R-ADV]:
1. Pick one EBM. Be clear which; acknowledge others exist.
2. Compare CF in that model vs a backprop model.
3. Try to understand why they differ. Can "trivial" differences (coding sparsity, network size) explain it?
4. Try to reduce CF in the EBM. EBMs are *predictive*, so it should be possible to find the nodes that differ most in their prediction and select only those for learning new stimuli. Does this work?

**Explicitly deferred by the advisor:** generative/synthetic replay, VAE example-ordering, the full 3×3 scenario×dataset grid, other EBM families, efficiency comparisons.

**Deadline context (as of 2026-08-10):** 20-minute presentation due Friday. Working assumption for prioritisation: *"if we started now with 3 months, what would we do?"*

---

## 2. Scenario taxonomy — and the load-bearing observation

### 2.1 The three scenarios [R2]

| Scenario | Output layer | Task ID at test | Difficulty |
|---|---|---|---|
| **Task-IL** | multi-head, only current context's units active | yes | easy — does not forget for any method |
| **Domain-IL** | single head, *n* units reused across contexts | no | intermediate |
| **Class-IL** | single head, all classes have own unit, all active | no | hardest |

Biological analogues [R2]: Task-IL ≈ playing different sports/instruments (it is always clear which). Domain-IL ≈ recognising objects under different lighting, driving in different weather. Class-IL ≈ incrementally learning to discriminate a growing set of objects.

### 2.2 ⚑ Song & Bogacz Fig 4d is **Domain-IL**, not Class-IL

Verified directly from the paper's Methods [R1]: *task 1 = five randomly selected classes, task 2 = the remaining five, whole network shared, **the network only had five output neurons**, alternating 4 iterations each to 84 total.* Five outputs reused for two different class sets with no task ID = **Domain-IL** under [R2].

Structural consequence (stated as structure, **not** as a claim about our results):

- In **Domain-IL**, every output unit is a positive target for some examples in *both* tasks. Output-layer suppression is **symmetric** — no recency bias. The live forgetting mechanism is **trunk representation drift**.
- In **Class-IL**, two further pathologies appear (§4.2).

**Status (decided 2026-08-10, chat #009):** this was briefly promoted to the thesis of the presentation ("the claim has a scenario boundary"). **That framing is retired as premature** — we have no results in both scenarios yet. It is demoted to an open question (§9.1) to be revisited once E4/E5/E6 are in. Both scenarios are run; the slides report what is observed and attribute afterwards.

**Note on protocol:** [R1] Fig 4d is 2 tasks × 5 classes, alternating. We use **5 tasks × 2 classes, sequential, in both scenarios**, so that our two scenarios differ *only* in the output layer. This is a deliberate deviation from [R1] and is listed in §8.

---

## 3. Methods under comparison

| Method | Role | What it is |
|---|---|---|
| `backprop` | **negative control** — must forget | baseline |
| `replay` | **positive control** — must fix it | backprop + stored-example buffer |
| `pc` | **primary EBM** | Predictive Coding / prospective configuration: settle activities to equilibrium with output **strongly clamped**, then local Hebbian update |
| `eqprop` | **contrast EBM** | Equilibrium Propagation: two settlings (free + **weakly clamped**/nudged), update from their difference |

### 3.1 How the rules differ

The sharpest distinction: **in backprop the hidden activities are fixed by the weights; in PC and EqProp they are variables that get optimised first, and only then do the weights change.** That is the energy-based family in one sentence.

| | backprop | predictive coding | EqProp |
|---|---|---|---|
| hidden activities | fixed by weights | variables — settle once | variables — settle twice |
| credit assignment | chain rule from above | inferred by relaxation | difference of two equilibria |
| weight update | global backward pass | local: pre-activity × post-error | local: free vs nudged difference |
| passes per update | 1 fwd + 1 bwd | 1 settling | 2 settlings |
| gradient | exact | exact at equilibrium | approximate (β-biased) |
| target at output | supervision signal | **clamped (strong)** | **a perturbation (weak)** |
| non-target classes | softmax suppression | one-hot → 0 | hinge → −1 (strongest suppression) |

### 3.2 ⚑ EqProp's role: it is the control that isolates *which* property matters

Both PC and EqProp are energy-based and both settle. If PC shows the effect and EqProp does not, then the effect is **not "energy-based-ness" per se** — it is specifically **strong clamping / full relaxation to a prospective configuration**. Dong & Wu's strong-clamp vs weak-clamp framing [R3] is the citation for this. This rescues EqProp from being dead weight in the comparison and turns a weak result into a scientific argument.

### 3.3 When PC *is* backprop — three regimes

Only the first two look like backprop:
1. **Partial relaxation / infinitesimal nudging → approximates backprop.** Whittington & Bogacz (2017) is titled "*An approximation* of…" [R6]. Millidge et al. (2022) unify PC, EqProp and contrastive Hebbian learning as all reducing to backprop in the infinitesimal inference limit [R7].
2. **Engineered exact equivalence** (Song et al., NeurIPS 2020) [R8] — but Dong & Wu state this equivalence "is not general": it needs specific initialisation, a precise layer-wise update schedule and particular inference settings [R3].
3. **Full relaxation to equilibrium → prospective configuration, genuinely ≠ backprop.** This is the actual contribution of [R1].

**Usable sentence:** *turning settling all the way up is what makes it a different algorithm.*

---

## 4. Mechanics of forgetting

### 4.1 The one gradient that explains the output layer

With logits `z_o`, softmax over the **active set 𝒜**, true class `t`:

```
∂L/∂z_o = p_o − 1[o = t]
```

For the true class the gradient is negative (descent raises `z_t`); for **every other active unit** it is `p_o > 0` (descent lowers `z_o`). With `z_o = w_oᵀh + b_o`, an old class with no data present gets `∂L/∂w_o = p_o·h` and `∂L/∂b_o = p_o` — so `b_o` drifts monotonically **down** and `w_o` is pushed **anti-parallel to the current feature mean**. No old data is involved; the trunk may be entirely intact.

**The active set 𝒜 is the real knob.** Scenario → 𝒜 → which gradient paths exist → which forgetting mechanisms can fire → which method families can possibly work.

### 4.2 Two distinct pathologies in Class-IL

1. **Logit suppression / task-recency bias** — a **calibration** failure. Features are fine; per-class scale and offset are wrong. Cheap to fix, no replay needed.
2. **Absent inter-context discriminative signal** — a **representation** failure. A boundary must be placed between classes never co-observed, with no gradient ever comparing them. **Irreducible**; requires information about old classes from somewhere.

"Class-IL needs replay" is only true of pathology 2 — and even then "replay" can mean a generative model or class prototypes rather than a buffer.

### 4.3 Full forgetting-mechanism map

| Mechanism | Lives in | Bites in | Replay-free fix |
|---|---|---|---|
| Trunk representation drift | shared layers | all three | EWC/SI, gating, freezing, **prospective configuration** |
| Head–feature mismatch | old head ∘ new features | Task-IL, Domain-IL | trunk stabilisation |
| Logit suppression / recency bias | output layer | **Class-IL only** | masking, cosine classifier, weight alignment, NCM |
| Missing inter-context boundary | decision function | **Class-IL only** | prototypes, generative classifier — or replay |

### 4.4 Dropping the softmax — correction

Replacing softmax does **not** remove suppression. The source is the **one-hot target supplying zero for every absent class**, not the softmax. Linear+MSE gives `∂L/∂z_o = z_o` for absent classes; sigmoid+BCE gives `σ(z_o) > 0`. Units decouple in the forward pass; **the labels stay coupled.**

What is right about the intuition: MSE's suppression has a **fixed point at zero** — `w_o` is driven until it is *orthogonal* to the current features, so if old-class features occupy a different subspace `w_o` can remain informative. Softmax has no fixed point; `z_o` is driven toward −∞ relative to `z_t` without bound.

**The tension:** more coupling → better calibration, more suppression; less coupling → less suppression, worse calibration. *You cannot win both with the same knob.* A **generative classifier** resolves it by getting the common scale from every class model being a normalised density over the same input space, rather than from a shared denominator.

### 4.5 The corrected PC mechanism hypothesis

**Wrong (retired):** "an already-correct output has zero error, so its weights barely move." This is a **within-a-single-forward-pass** statement (Bogacz Fig 1). During task-2 training you clamp task-2 inputs *and* task-2 targets, so the error is task-2's and it drives change through whatever units task-2's settling implicates — including task-1's. **It gives task 1 no protection.**

**Current (testable):** after settling, `e1 = x1 − x0 @ W1` and `ΔW1 ∝ x0ᵀ e1`. So **PC changes a weight in proportion to the activity displacement its settling required.** Hence:

> PC interferes with task 1 only to the extent that satisfying task-2's target forces movement in the hidden units task 1 depends on.

This reframes the question from *"is the error zero?"* (no) to ***"where does the weight movement go?"*** (measurable). → experiment P5.

---

## 5. Metrics

| Metric | Definition | Use | Status |
|---|---|---|---|
| Per-task accuracy | acc on each task's fixed class set, held-out test split | primary | ✅ in use |
| **ACC1–ACC2 trajectory** | plot task-1 acc vs task-2 acc, time removed | **most robust forgetting metric found** | ✅ in use |
| **Area above diagonal** | integral of trajectory above the ACC1=ACC2 line | learning/forgetting *trade-off efficiency* | ✅ in use |
| Crossover height | acc at the point the two task curves cross | single-number trade-off summary | ✅ in use |
| Forgetting | max task-1 acc − final task-1 acc | standard CL metric, comparable to literature | ⬜ to lock |
| **Target alignment** [R1 Fig 3b] | `cos( target − out_before , out_after_no_target − out_before )` | direct interference measure; the literature's own metric | ⬜ **P4 — not yet run** |
| **Linear probe / NCM** | classify from penultimate features, head discarded | **separates pathology 1 (calibration) from pathology 2 (representation)** | ⬜ **P3 — never run** |
| Per-weight \|Δw\| by importance rank | accumulated during task 2, split by task-1 Fisher rank | mechanism: where movement goes | ⬜ P5 |

### 5.1 Measurement traps (hard-won — do not repeat)

- **Never evaluate on training data.** An earlier bug did this and invalidated a day of sweeps. Use the test split.
- **`cur%` is degenerate at 1 class/task** — a model predicting one class always scores 100%.
- **`seen%` has a changing denominator** across tasks, so it is not comparable over a run. Prefer per-task accuracy with fixed class sets.
- **Accuracy is a threshold readout.** After a task switch nothing appears to happen for ~20 steps while logits climb, then it flips fast. Log raw pre-argmax outputs to see the real dynamics.
- **The flat line at exactly 10% / 20% / 25% is not chance** — it is the **collapse floor** `100/n_classes`, the score of a model predicting one class for everything.
- **Do not fit sigmoids to accuracy curves** — they are step-like and noisy. Use threshold crossings and the trajectory plot.
- **Below-chance (0%, not 50%) task-1 accuracy** is only possible if argmax is systematically captured by task-2 units ⇒ confirms global argmax over a shared head, and points at suppression rather than representation loss.
- **Multi-pass caution:** >1 pass = cyclic revisiting, not longer continual learning.
- **Match final new-task accuracy** when claiming "forgot less", or the claim is confounded with "learned less".

---

## 6. Results to date

⚠️ **All figures below are from prior runs, reported second-hand in the handoff docs. None were reused for the presentation — every experiment in §11 is re-run from scratch.** Retained as priors for sanity-checking new results, not as evidence.

### 6.1 Class-IL, backprop vs controls
| Split | backprop | eqprop | replay |
|---|---|---|---|
| 10×1 (100 it/task) | ~10% (floor) | ~10% | ~64% |
| 5×2 | ~20% | ~20% | ~60% |
| 2×2 | task 1 holds ~100% to step ~110 then collapses to 0 | — | both ~95% |

### 6.2 Four-method comparison, 5×2 Class-IL, 10 seeds (exp 11)
- **Trade-off efficiency (area above diagonal): pc > replay > backprop > eqprop**
- **Final task-1 retention: replay (~37%) ≫ pc (~10%) > eqprop (~3%) > backprop (~0%)**
- **These two orderings differ, and the difference is the whole story.**
- Qualitative shapes: backprop = cliff (85→0 in ~35 steps); replay = drop then plateau, large spread (buffer-composition variance); eqprop = gradual monotone decay, noisy; pc = gradual decay, tightly clustered.
- Crossover: ~75% PC vs ~65% BP; replay ~85%.

**⚑ Honest headline:** *PC buys graceful degradation, not durable retention.* It reduces interference per update; it does not **anchor** past knowledge. Durable retention requires an explicit anchor (replayed data or a consolidation penalty). PC has none.

**⚑ Premise correction:** EqProp is **not** forgetting less. Its decay only looks gentle because everything is slower (lower peak on task 1, slower/lower task 2). On the trajectory plot — which removes time — EqProp is the **worst** panel, below the diagonal throughout.

### 6.3 Earlier findings (28×28 notebook era, still valid)
- **Task-IL does not forget** for backprop (probe and head both ~99%). Earlier apparent forgetting was a **polarity artifact** of a balanced 2-way head — fold with `max(acc, 1−acc)`.
- **Class-IL forgets catastrophically:** ~21.6% final mean; tasks collapse to ~0% (active overwriting, not decay to chance).
- **Gradient interference measured:** cosine between current-task and prior-task gradients is predominantly **negative** during Class-IL. Sign↔Δloss identity is exact under SGD.
- **Replay works** (~78% vs ~21%) but is not equivalent to joint training.
- **EWC fails in Class-IL** (~20% ≈ baseline), reproducing [R2] Table 2. Reason: (a) deadlock — protecting old weights prevents learning new classes; (b) preserving each task's function cannot create discrimination between classes never seen together. **Unresolved hypothesis:** EWC's Fisher-importance distribution resembles replay's yet accuracy matches baseline → failure is in the **readout/arbitration, not the features**. Test = linear probe on the EWC trunk (predict: probe high, head low). → folded into P3.

### 6.4 Reproduction attempt (exp 12) — status
Like-for-like reproduction of [R1] Fig 4d **was not achieved**. ~1 week spent. Working code was broken and the codebase became unreadable.
**Ruling (2026-08-10): stop chasing numeric match.** Reproduce the **protocol**, not the hyperparameters. The claim under test is qualitative: *does PC forget less than BP under matched conditions?* This is a **conceptual replication** and is the scientifically standard framing. See §8.

---

## 7. Working practice (adopted after a period of overwhelm — these matter)

- **One question per experiment**, written as a single sentence before running.
- **Controls on every forgetting experiment**: backprop (negative) and replay (positive). If replay works, the problem is provably solvable — this stops "maybe forgetting is inevitable" spirals.
- **A doubt gets one scheduled test, then it is closed.** Doubts arising mid-run go on an open-questions list, not chased immediately.
- **One script → one figure, named identically** (figure name derived from `__file__`).
- **Keep code minimal.** Repeated over-engineering (wandb, Bayesian HPO, class hierarchies, a `harness.py` abstraction) was tried and **deliberately deleted**. Do not reintroduce.
- **Metaphors are banned.** Explanations must map to variables, order of operations and code lines.
- **Claims separated from hypotheses**, with figure-level citations for anything called a claim.
- **Never repair a broken exploratory branch — freeze the last good commit and branch from it.**

### 7.1 Explicitly abandoned
- wandb / Optuna / persistent HPO infrastructure.
- `harness.py` shared run+plot module.
- Per-class output heads — with task ID it is Task-IL (which forgets for no method); without it, calibration fails equally for all methods. Not a route to an EBM advantage.
- Contrastive/conditional-energy EBM [R9] — this *is* the EBM proven to beat replay in Class-IL, but the advisor's plan supersedes it. Keep as the "acknowledged alternative" for advisor point 1.

### 7.2 Personal working constraints
- Plain language, no flowery metaphors. Concept first (ELI5 + graduate), then decisions with trade-offs, then code.
- One stage at a time; small increments; minimal machinery. Over-complication has repeatedly caused loss of momentum.
- Motivation dips have occurred. The engaging threads are: *what happens inside the network when a new class arrives* (units overwritten, reused, or newly allocated) and *the EBM as its own replay generator*. Keep those visible.

---

## 8. Deviations from the literature, and their justifications

| Axis | Song & Bogacz [R1] | Ours | Justification |
|---|---|---|---|
| Dataset | Fashion-MNIST 28×28 | **MNIST 14×14** (196 in) | EqProp settling is the runtime bottleneck; 14×14 is ~4× cheaper and keeps MNIST intact (~97% BP ceiling); 8×8 degrades classes. **Evidence required → P0.** |
| Depth | 15 layers (Fig 3f–h); unspecified for 4d | **1 hidden layer (196→64→10)** | Isolates the learning rule from a depth confound. Also the **conservative** choice: [R1] Fig 3e/4h claim PC's advantage *grows* with depth, so our result is a **lower bound**, not a ceiling. |
| Activation | LeakyReLU | **tanh** | Matched across all rules; the verified PC implementation uses tanh; [R1] chose LeakyReLU specifically because sigmoid fails in *deep* nets — not applicable at depth 1. |
| Loss | squared error | **squared error, matched** | Matched PC/BP to remove the loss confound (identified as the confound in the 4-way comparison). |
| Scenario | **Domain-IL** (5 shared outputs) | **both Domain-IL and Class-IL** | Class-IL is primary: largest dynamic range between methods [R2 Tables 2–3], standard ML benchmark, matches the common biological case, and exercises trunk *and* output layer so forgetting can be attributed. Domain-IL is also run because it is the scenario the interference claim was demonstrated in. |
| Protocol | 2 tasks × 5 classes, **alternating**, 4 iters each to 84 | **5 tasks × 2 classes, sequential**, both scenarios | Holds partition and schedule constant so the output layer is the only difference between our two scenarios; [R1]'s protocol varies task count and schedule alongside scenario. Also matches [R2]'s Split MNIST protocol, making our baselines comparable to a published table. |
| Optimiser | SGD | SGD | Matched. Plain SGD everywhere: exact interference identity, no cross-task momentum confound, matches EqProp's own updates. |
| Learning rate | per-rule grid search | **per-rule grid search** | Not a deviation — required for a fair comparison. |
| Batch size | 32 | 32 | Matched. |
| Seeds | 10 | ≥5 | Compute budget; report 68% CI as they do. |
| Architecture | FC | FC, no CNN | Avoids inductive priors so observations generalise across the method comparison. |

---

## 9. Open questions

### 9.1 Retired-pending-evidence
**Does the interference claim have a scenario boundary** — i.e. is PC's advantage present in Domain-IL and absent in Class-IL, while its *target alignment* advantage stays scenario-independent? This was briefly the presentation's thesis and was retired on 2026-08-10 as premature. Revisit only after E4, E5 and E6. Do not write slides that presuppose the answer.

### 9.2 Active
1. Is each method's failure calibration or representation? **(E2)**
2. Is the forgetting in the trunk (W1) or the head (W2)? **(E3)**
3. What is target alignment in our own networks, per rule and scenario, and does it track retention? **(E6)**
4. Does PC concentrate weight movement away from earlier-context-important weights, or is the advantage simply lower update variance? Both outcomes are defensible. **(E7)**
5. Does depth change the conclusion? **(E8)**
6. Does EWC stack usefully on PC even though it failed on vanilla BP?
7. Can E_sharp and E_smooth be trained under a single unified objective, or does the sharp component need a separate fast-weight update rule?
8. Advisor point 4: can per-node prediction error at settling time be used to gate which nodes learn new stimuli?
9. §4.1's output-layer suppression argument is derived for **softmax cross-entropy**. Our implementation uses **squared error**. The MSE version has a fixed point at zero (§4.4) — restate the argument for MSE before relying on it in slide 9.
10. The trajectory plot was designed for 2 tasks. With 5 contexts, decide the presentation: pairwise first-vs-last, or mean-of-seen vs current. **(E-metrics)**

---

## 10. Code

```
project/
├── data/                       # MNIST downloads here
├── src/
│   ├── data.py                 # load_mnist, class_indices, make_eval_set
│   ├── eqprop.py               # eqprop_init/energy/settle/update/predict (+ gated, generate)
│   ├── predictive_coding.py    # pc_init/forward/settle/update/predict
│   ├── plotting.py             # reusable plotting utilities
│   └── methods.py              # make_backprop, make_replay, make_eqprop, make_pc, …
└── experiments/
    ├── 09_eqprop_learning_vs_forgetting.py
    ├── 10_pc_learning_vs_forgetting.py
    ├── 11_consolidate_pairs_4methods.py     # ran previously, 10 seeds
    └── 12_*                                  # broken by the depth refactor + S&B protocol shift
```

⚠️ **This layout is from the handoff docs and has NOT been verified against the actual repo.** No code has been seen directly in the current workspace.

**Code policy (agreed 2026-08-10):** all work is committed; there is no earlier revision to revert to, and none is wanted. Move forward from current state.
- Keep both scenarios and the depth parameter. Do not remove functionality.
- Move conditionals out of the core functions in `eqprop.py` and `predictive_coding.py`. Scenario and depth resolve at construction time, not inside the update step.
- Once a core function is readable and tested, **freeze it**; change it only when an experiment cannot be expressed otherwise.
- One script per experiment, one figure per script, named identically.
- One change at a time, each tested on its own, with the previous experiment re-run to confirm the figure is unchanged.

**Interface contract:** every `make_*` returns `(train_step, predict)`. `train_step(x, y)` does one update. `predict(x, raw=False)` returns class indices, or raw pre-argmax outputs when `raw=True`. Adding a model = one new `make_*`. Experiment scripts change only the `methods` dict.

**Hyperparameters in use:** MLP 196→64→10; plain SGD; batch 32; PC relaxation γ ≈ 0.1, ~50 steps; tanh; squared error.

### 10.1 Reference configurations from [R2] (for comparability)
```
Split MNIST base net:  2 × 400 ReLU FC + softmax; 5 contexts × 2 digits;
                       2000 iters/context; batch 128; Adam lr 1e-3
Output by scenario:    Task-IL   → multi-head, only current context active
                       Domain-IL → single head, 2 units (MNIST)
                       Class-IL  → single head, 10 units, all active
```

### 10.2 Snippets to lock
```python
# Target alignment — [R1] Fig 3b
d_target  = target - out_before
d_learn   = out_after_no_target - out_before          # NB: measured WITHOUT the target provided
alignment = F.cosine_similarity(d_target, d_learn, dim=-1)

# NCM / linear probe — separates pathology 1 from pathology 2
h = model.trunk(x)                                    # discard the head entirely
prototypes = {c: mean_of_h_for_class_c}
pred = argmin_c ||h - prototypes[c]||
# If task-1 accuracy jumps from ~0% to substantial, the trunk survived
# and the output layer was the entire failure.
```

---

## 11. Current experiment plan (E-series)

Full detail in `presentation_plan.md`. All experiments re-run from scratch; no earlier results or plots are reused.

| ID | Question | Slides | Depends on |
|---|---|---|---|
| **E0** | Does 14×14 preserve the joint-training accuracy of 28×28? | 5, 6 | — |
| **E-metrics** | Metric definitions and plotting module | 7, all | — |
| **E1** | How much does backprop forget in each scenario, and does replay recover it? | 2, 8 | E-metrics |
| **E2** | Do the hidden features survive when the head does not? (NCM probe) | 9, 17 | E1 |
| **E3** | Is the forgetting in W1 or W2? (freeze one, then the other) | 9, 17 | E1 |
| **E4** | How do the four rules compare in Class-IL 5×2? | 12, 14 | E-metrics |
| **E5** | How do the four rules compare in Domain-IL 5×2? | 13, 14 | E4 |
| **E6** | What is the target alignment of each rule in each scenario? | 15 | E4, E5 |
| **E7** | Where does the weight movement go during later contexts? | 16 | E4 |
| **E8** | Does depth change the conclusion? | 18 | E4 |

Order: E-metrics → E1 → E4 → E5 → E2/E3 → E6 → E0 → E7 → E8.

---

## 12. References

### 12.1 In-project (PDFs in the workspace)

- **[R1] Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz (2024).** "Inferring neural activity before plasticity as a foundation for learning beyond backpropagation." *Nature Neuroscience* 27:348–358. `song_bogacz_24.pdf` · https://doi.org/10.1038/s41593-023-01514-1 · Code: https://github.com/YuhangSong/Prospective-Configuration
  *Introduces prospective configuration — settle activities to equilibrium before plasticity — and argues it reduces interference relative to backprop across continual learning, online learning, RL and human behavioural data.*
  Key figures: Fig 1 (interference example), Fig 2 (energy machine), Fig 3a–e (**target alignment**, and its degradation with depth), **Fig 4d–e (two alternating 5-class tasks — the protocol we mirror; note this is Domain-IL)**, Fig 4f–g (concept drift), Fig 4h (depth), Discussion (the EqProp remark).

- **[R2] van de Ven, Tuytelaars & Tolias (2022).** "Three types of incremental learning." *Nature Machine Intelligence* 4:1185–1197. `s42256022005683.pdf` · https://doi.org/10.1038/s42256-022-00568-3 · Code: https://github.com/GMvandeVen/continual-learning
  *Defines the Task-/Domain-/Class-IL taxonomy and benchmarks the major CL strategies in all three, showing parameter regularisation collapses to baseline in Class-IL while replay holds up everywhere.*
  Key: Table 1 (scenario mappings), Fig 2 (Split MNIST three ways), eq. (2) (active-unit softmax), Tables 2–3 (empirical comparison), Methods (architectures, generative classifier, iCaRL).

- **[R3] Dong, Peng & Wu (2025).** Commentary on Song et al. *Intelligent Computing.* `dong_wu_rev_song_bogacz.pdf` · https://doi.org/10.34133/icomputing.0244
  *Situates predictive coding networks in the EBM lineage as an EM procedure and re-examines the biological-plausibility and cost claims, noting the backprop-equivalence result "is not general".*
  Key: EBM lineage; **strong-clamp (PC) vs weak-clamp (EqProp) framing**; interference-regime caveat (batch size 1 / depth); expensive relaxation phase; requires approximately symmetric weights.

- **[R4] Kirkpatrick et al. (2017).** "Overcoming catastrophic forgetting in neural networks." *PNAS* 114:3521–3526. `kirkpatrick_17.pdf` · https://doi.org/10.1073/pnas.1611835114
  *Introduces Elastic Weight Consolidation, a diagonal-Fisher quadratic penalty anchoring weights important to previous tasks, motivated by dendritic-spine persistence.*

### 12.2 Core external references

- **[R5] Scellier & Bengio (2017).** "Equilibrium propagation: bridging the gap between energy-based models and backpropagation." *Front. Comput. Neurosci.* 11:24. https://doi.org/10.3389/fncom.2017.00024
  *Defines EqProp: a free phase and a weakly nudged phase, with the weight update given by the difference between the two equilibria — the chosen EBM for advisor point 1.*

- **[R6] Whittington & Bogacz (2017).** "An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity." *Neural Computation* 29(5):1229–1262. https://doi.org/10.1162/NECO_a_00949
  *Shows a predictive coding network with purely local Hebbian plasticity approximates backprop — the "approximation" regime that prospective configuration deliberately departs from.*

- **[R7] Millidge, Song, Salvatori, Lukasiewicz & Bogacz (2022).** "Backpropagation at the infinitesimal inference limit of energy-based models: unifying predictive coding, equilibrium propagation, and contrastive Hebbian learning." arXiv:2206.02629. https://arxiv.org/abs/2206.02629
  *Proves PC, EqProp and contrastive Hebbian learning all reduce to backprop in the infinitesimal inference limit — the formal basis for saying differences only appear at full relaxation.*

- **[R8] Song, Lukasiewicz, Xu & Bogacz (2020).** "Can the brain do backpropagation? Exact implementation of backpropagation in predictive coding networks." *NeurIPS* 33:22566–22579.
  *Constructs an exact backprop equivalence for PCNs under a specific schedule — the equivalence [R3] flags as non-general.*

- **[R9] Li, Du, van de Ven & Mordatch (2022).** "Energy-based models for continual learning." *CoLLAs.* arXiv:2011.12216 · Code: https://github.com/ShuangLI59/ebm-continual-learning
  *A conditional-energy EBM that beats replay in Class-IL by scoring class-conditional energies with contrastive divergence instead of normalising a softmax over all classes — the acknowledged alternative EBM, deferred by the advisor.*

- **[R10] van de Ven, Siegelmann & Tolias (2020).** "Brain-inspired replay for continual learning with artificial neural networks." *Nature Communications* 11:4069. https://doi.org/10.1038/s41467-020-17866-2
  *Generative replay with the hippocampus framed as a generative network, removing the need for a stored buffer.*

- **[R11] McCloskey & Cohen (1989).** "Catastrophic interference in connectionist networks: the sequential learning problem." *Psychology of Learning and Motivation* 24:109–165.
  *The original demonstration that sequentially trained connectionist networks abruptly lose earlier associations.*

- **[R12] Zenke, Poole & Ganguli (2017).** "Continual learning through synaptic intelligence." *ICML.* arXiv:1703.04200
  *Synaptic Intelligence — online per-weight importance accumulated during training rather than from a post-hoc Fisher estimate.*

- **[R13] Laborieux, Ernoult, Hirtzlin & Querlioz (2021).** "Synaptic metaplasticity in binarized neural networks." *Nature Communications* 12:2549. https://doi.org/10.1038/s41467-021-22768-y
  *Uses hidden weight magnitude as a metaplastic consolidation variable to mitigate forgetting without storing data — the template for the "reduce CF in the EBM" step.*

- **[R14] Benna & Fusi (2016).** "Computational principles of synaptic memory consolidation." *Nature Neuroscience* 19:1697–1706. https://doi.org/10.1038/nn.4401
  *Cascade / complex-synapse model in which multiple timescales within a synapse extend memory lifetime — the biological grounding for consolidation as a third mechanism.*

### 12.3 Supporting external references

- **[R15] Millidge, Salvatori, Song, Bogacz & Lukasiewicz (2022).** "Predictive coding: towards a future of deep learning beyond backpropagation?" *IJCAI survey.* arXiv:2202.09467 — *PC as a general-purpose local-computation algorithm; motivation is parallelisability and neuromorphic hardware.*
- **[R16] Salvatori et al. (2024).** "A stable, fast, and fully automatic learning algorithm for predictive coding networks." *ICLR.* — *Removes PC's sensitivity to relaxation hyperparameters.*
- **[R17] Laborieux et al. (2021).** "Scaling equilibrium propagation to deep ConvNets by drastically reducing its gradient estimator bias." *Front. Neurosci.* 15:633674 — *Finite-nudging bias is why vanilla EqProp does not scale past MNIST.*
- **[R18] Ramsauer et al. (2021).** "Hopfield networks is all you need." *ICLR.* arXiv:2008.02217 — *Transformer self-attention is the update rule of a modern continuous-state Hopfield network; EBMs are not a fringe framework.*
- **[R19] Bai, Kolter & Koltun (2019).** "Deep equilibrium models." *NeurIPS.* — *"Settle to a fixed point, then differentiate implicitly" as mainstream ML.*
- **[R20] Kendall et al. (2020).** arXiv:2006.01981; **Martin et al. (2021).** "EqSpike." arXiv:2010.07859 — *EqProp on analog/neuromorphic hardware; settling is slow in silicon but free in physics — EqProp's real value proposition.*
- **[R21] Hou et al. (2019)** LUCIR (cosine classifier); **Wu et al. (2019)** BiC; **Zhao et al. (2020)** WA; **Rebuffi et al. (2017)** iCaRL — *Class-IL bias-correction family; each targets pathology 1 (calibration) specifically.*
- **[R22] Maltoni & Lomonaco (2019).** "Continuous learning in single-incremental-task scenarios." *Neural Networks* 116:56–73 — *Alternative MT/SIT/MIT scenario taxonomy predating [R2].*
- **[R23] Aljundi et al. (2019)**; **Lee et al. (2020)** — *Task-free continual learning, where boundaries are not given.*
- **[R-ADV] Advisor's four-point outline** (personal communication) — *The authoritative scope document; everything outside it "can/should wait".*
