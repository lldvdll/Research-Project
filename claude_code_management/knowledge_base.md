# Knowledge Base

Consolidated, structured record of everything established, refuted and still open on this project.
**Loaded into context at the start of every session** via the import in `CLAUDE.md`.

Maintenance: new information is *added*; existing entries are *amended in place*; where a later
finding contradicts an earlier one, the old claim is **kept and marked corrected**, never silently
deleted. Reference keys `[Rn]` resolve in §12. Terms are defined in §13.

*Consolidated 2026-08-10 from fourteen superseded documents in `archive/chat_logs/`, then extended
the same day with experiments 13–34, which no source document recorded (§6.6).*

> ## A line has been drawn under all previous work
>
> **Nothing in §6 is evidence.** Prior results inform direction and experiment design; they do not
> support claims. Any statement that reaches a slide is backed by a **fresh experiment, designed for
> that slide, run under the new protocol** — not by a number recovered from here.
>
> This is deliberate. The prior work used an inconsistent array of setups (§6.6.2), several runs
> contradict each other or their own documentation (§6.6.3, §9.3), the reproduction failed with the
> cause unidentified (§6.5), and the code that produced all of it is not trusted (§6.7).
>
> Sections marked **DO NOT TRUST** carry sweeping conclusions that every archived document repeats
> and that later work contradicts. They are kept for their reasoning, not their verdicts.

---

## 1. Project definition

**Thesis:** an evaluation of energy-based / predictive-coding learning rules against
backpropagation for catastrophic forgetting (CF) in continual learning. MSc, computational
neuroscience & AI. Report target ~10,000 words.

**Core question:** does an energy-based model with a biologically plausible local learning rule
reduce catastrophic forgetting compared with backprop — and if not, why not, and can it be made to?

**The advisor's four points — guidance, not scope** [R-ADV]. They arrived ad hoc, in an email
responding to a set of results, and have already shifted once: EqProp and replay were each dropped
and then reinstated as results came in. Taken seriously; **not** treated as a specification to be
satisfied point by point. The project target is stated in `presentation_plan.md` §1.

1. Pick one EBM. Be clear which; acknowledge others exist.
2. Compare CF in that model vs a backprop model.
3. Try to understand why they differ. Can "trivial" differences (coding sparsity, network size)
   explain it?
4. Try to reduce CF in the EBM. EBMs are *predictive*, so find the nodes whose prediction differs
   most and select only those for learning new stimuli. Does this work?

**Explicitly deferred by the advisor:** generative/synthetic replay, VAE example-ordering, the full
3×3 scenario×dataset grid, other EBM families, efficiency comparisons.

**Chosen EBMs.** EqProp is the formal "pick one" answer (point 1). PC was added because **the
prospective-configuration interference claim is PC's claim, not EqProp's** — testing the
literature's claim requires PC. `[SETTLED]`

**Deadline context (as of 2026-08-10):** 20-minute presentation due Friday.

---

## 2. Scenario taxonomy

### 2.1 The three IL scenarios [R2] `[SETTLED]`

| | Task-IL | Domain-IL | Class-IL |
|---|---|---|---|
| Mapping | f: X × C → Y | f: X → Y | f: X → Y × C |
| Context ID at test | given | not given, not needed | not given, **must be inferred** |
| Output layer | multi-head | single, fixed | single, global, all active |
| Cross-context discrimination | never required | not required | **required** |
| Difficulty | easy — nothing forgets | intermediate | hardest |

Biological analogues [R2]: Task-IL ≈ playing different instruments (always clear which);
Domain-IL ≈ recognising objects under different lighting; Class-IL ≈ incrementally learning to
discriminate a growing set of categories.

**Training-time axis, frequently conflated with the above:** *task-free* = no boundaries during
training; *task-agnostic* = no context labels needed at any point. State both explicitly.

### 2.2 Two other taxonomies that are not the same thing `[SETTLED]`

**Data shift** [R24], via p(x,y) = p(y|x)·p(x) = p(x|y)·p(y): covariate (p(x) changes) lands in the
features; prior-probability / label shift (p(y) changes) lands in **output biases only**; concept
shift (p(y|x) changes) lands in output weights and is an unavoidable overwrite in a shared head.

**Stream carving** [R22]: MT (isolated tasks, multi-head) / SIT (one task extended over time,
single head) / MIT (several tasks, each extended). Inside SIT: NI (new instances) / NC (new
classes) / NIC (both). **SIT+NC ≈ Class-IL.**

> ⚠️ **The taxonomies genuinely disagree.** Permuted MNIST is **MT** for Maltoni & Lomonaco (tasks
> are isolated) but explicitly **Domain-IL, not Task-IL**, for van de Ven. Neither is wrong; they
> measure different things. Always check which axis a paper means by "multi-task".

### 2.3 ⚑ Song & Bogacz Fig 4d is **Domain-IL**, not Class-IL `[SETTLED]`

Verified directly from the paper's Methods [R1]: *task 1 = five randomly selected classes, task 2 =
the remaining five, whole network shared, **the network only had five output neurons**, alternating
4 iterations each to 84 total.* Five outputs reused for two disjoint class sets with no task ID is
**Domain-IL** under [R2].

Structural consequence, stated as structure and **not** as a claim about our results:

- In **Domain-IL** every output unit is a positive target for some examples in *both* tasks, so
  output-layer suppression is **symmetric** — no recency bias. The live forgetting mechanism is
  **trunk representation drift**.
- In **Class-IL** two further pathologies appear (§4.3).

**Status (decided 2026-08-10):** this observation was briefly promoted to the thesis of the
presentation ("the interference claim has a scenario boundary"). **That framing is retired as
premature** — there are no results in both scenarios yet. Demoted to an open question (§10.1),
revisited only after E4/E5/E6. Both scenarios are run; the slides report what is observed and
attribute afterwards.

### 2.4 Target regime

**Class-IL under a task-free stream** is the target regime and the primary scenario: the one cell
where parameter regularisation provably is not enough and where the deficit is specifically
inter-context discrimination. Domain-IL is also run (§8).

---

## 3. Methods under comparison

| Method | Role | What it is |
|---|---|---|
| `backprop` | **negative control** — must forget | baseline |
| `replay` | **positive control** — must fix it | backprop + stored-example buffer |
| `pc` | **primary EBM** | Predictive Coding / prospective configuration: settle activities with the output **strongly clamped**, then one local Hebbian update |
| `eqprop` | **contrast EBM** | Equilibrium Propagation: two settlings (free + **weakly clamped**), update from their difference |

### 3.1 How the rules differ `[SETTLED]`

The sharpest distinction: **in backprop the hidden activities are fixed by the weights; in PC and
EqProp they are variables that get optimised first, and only then do the weights change.** That is
the energy-based family in one sentence.

Exact statement: take the PCN energy E = Σ_l ½(x_l − w_{l−1} f(x_{l−1}))². **Clamp every hidden x_l
to its feedforward value** and every error is zero except at the output, so E reduces to *exactly*
the FFNN output loss. **The FFNN is the PCN with its internal state frozen to the forward pass.**

| | backprop | predictive coding | EqProp |
|---|---|---|---|
| hidden activities | fixed by weights | variables — settle once | variables — settle twice |
| credit assignment | chain rule from above | inferred by relaxation | difference of two equilibria |
| weight update | global backward pass | local: pre-activity × post-error | local: free vs nudged difference |
| passes per update | 1 fwd + 1 bwd | 1 settling | 2 settlings |
| gradient | exact | exact at equilibrium | approximate (β-biased) |
| clamping | — | **strong clamp** | **weak clamp** |
| target at output | supervision signal | clamped | a perturbation, not a target |
| non-target classes | softmax suppression | one-hot → 0 | **hinge → −1 (strongest suppression)** |

All energy-based rules run an **EM-like two-step** [R3]: E-step, activities relax to low energy;
M-step, weights move to make that state more probable. **Backprop collapses the two** — it has no
inner loop at all.

**Symbols — do not confuse.** `α` = weight learning rate (how far *weights* step per update).
`γ` / `dt` = settling rate (how far *activities* move per relaxation step). `β` = EqProp's **nudge**,
*not* a learning rate: a deliberate perturbation used to estimate a gradient by finite difference.
It appears **in the denominator** (Δw ∝ (1/β)(nudged − free)) and the estimate becomes exact as β→0.

**Where `f` sits.** Song–Bogacz convention: the prediction of layer l is w_{l−1} f(x_{l−1}) —
nonlinearity *then* weight. So x is the raw node state and f(x) is the rate it sends onward. A
modelling convention only; different PC papers place f differently.

**The global energy is not a plausibility problem** `[SETTLED]`: it is never computed, stored or
transmitted by the network. It is a Lyapunov function used by the analyst. Because the energy is a
sum of local terms, its gradient with respect to any local variable is itself purely local. This is
exactly the objection that sinks *backprop's* plausibility and that EBMs escape.

### 3.2 ⚑ EqProp's role: the control that isolates *which* property matters

Both PC and EqProp are energy-based and both settle. If PC shows the effect and EqProp does not,
the effect is **not "energy-based-ness" per se** — it is specifically **strong clamping / full
relaxation to a prospective configuration**. Dong & Wu's strong-clamp vs weak-clamp framing [R3] is
the citation. This turns a weak EqProp result into a scientific argument rather than dead weight.

### 3.3 The three regimes of the PC↔backprop relationship `[SETTLED]`

Only the first two look like backprop:

1. **Partial relaxation / infinitesimal nudging → *approximates* backprop.** Whittington & Bogacz
   (2017) is literally titled "*An approximation* of…" [R6]. Millidge et al. (2022) unify PC,
   EqProp and contrastive Hebbian learning as all reducing to backprop in the infinitesimal
   inference limit [R7].
2. **Engineered exact equivalence** (Song et al. 2020, Z-IL) [R8] — but [R3] states this equivalence
   "is not general": it needs specific initialisation, a precise layer-wise schedule and particular
   inference settings.
3. **Full relaxation to equilibrium → prospective configuration, genuinely ≠ backprop.** This is the
   actual contribution of [R1].

**Usable sentence:** *turning settling all the way up is what makes it a different algorithm.*

**Corollary for EqProp** `[SETTLED]`: EqProp's weak clamp with β→0 suppresses the very activity
shift that *is* prospective configuration. [R1]'s Discussion names equilibrium propagation
explicitly as the example of setting up energy-based networks to approximate backprop. EqProp is
therefore a **finite-difference estimator of backprop's gradient**, inheriting backprop's
interference and adding estimator variance plus finite-β bias. **EqProp scoring below backprop is
the expected result, not an anomaly.**

---

## 4. Mechanics of forgetting

### 4.1 The one gradient that explains the output layer `[SETTLED]`

```
Softmax + CE:   ∂L/∂z_o = p_o − 1[o=t]      (no fixed point; unbounded competition)
Linear + MSE:   ∂L/∂z_o = z_o − y_o          (fixed point at z_o = 0 ⇒ w_o ⟂ current features)
Sigmoid + BCE:  ∂L/∂z_o = σ(z_o) − y_o       (decoupled forward, still-coupled labels)
```

For the true class the gradient is negative (descent raises `z_t`); for **every other active unit**
it is positive (descent lowers `z_o`). With `z_o = w_oᵀh + b_o`, an old class with no data present
gets `∂L/∂w_o = p_o·h` and `∂L/∂b_o = p_o` — so `b_o` drifts monotonically **down** and `w_o` is
pushed **anti-parallel to the current feature mean**. No old data is involved; the trunk may be
entirely intact.

> ⚠️ **This derivation is for softmax cross-entropy. The implementation uses squared error.** The
> MSE version has a fixed point at zero (§4.4) and the argument must be restated before slide 9
> relies on it. Open item, §10.2.

### 4.2 The active set 𝒜 is the real knob `[SETTLED]`

Scenario → 𝒜 → which gradient paths exist → which forgetting mechanisms can fire → which method
families can possibly work.

- **Task-IL / multi-head:** 𝒜 = current context only. Old heads are *literally disconnected from
  the loss*. No output-layer interference by construction — which is why EWC/SI work here and why
  Separate Networks has exactly zero forgetting.
- **Domain-IL / single fixed head:** the head is shared and does receive gradients, but every class
  is positive in every context, so suppression is **symmetric** — no recency bias.
- **Class-IL / single global head:** only current classes ever appear as positives ⇒ **asymmetric
  suppression at full strength**.

### 4.3 Two distinct pathologies in Class-IL `[SETTLED]` — the most useful decomposition

1. **Logit suppression / task-recency bias** — a **calibration** failure. Features are fine; per-class
   scale and offset are wrong. **Cheap to fix, no replay needed.**
2. **Absent inter-context discriminative signal** — a **representation** failure. A boundary must be
   placed between classes never co-observed, with no gradient ever comparing them. **Irreducible**;
   requires information about old classes from somewhere.

"Class-IL needs replay" is true only of pathology 2 — and even then "replay" can mean a generative
model or class prototypes rather than a buffer.

**Full forgetting-mechanism map:**

| Mechanism | Lives in | Bites in | Replay-free fix |
|---|---|---|---|
| Trunk representation drift | shared layers | all three | EWC/SI, gating, freezing, prospective configuration |
| Head–feature mismatch (old head ∘ new features) | output layer | Task-IL, Domain-IL | stabilise the trunk |
| **Logit suppression / recency bias** | output layer | **Class-IL only** | masking, cosine classifier, weight alignment, NCM |
| **Missing inter-context boundary** | decision function | **Class-IL only** | prototypes, generative classifier — or replay |

### 4.4 Dropping the softmax does not remove suppression — correction on record `[SETTLED]`

The source is the **one-hot target supplying zero for every absent class**, not the softmax.
Softmax is the transport mechanism; the label is the source. Units decouple in the forward pass;
**the labels stay coupled.**

Three components that were being treated as one: the **normaliser** (couples classes, provides a
common scale — removing it loses calibration); the **negative targets in the label** (the actual
source of suppression — removing them requires masking); the **active set 𝒜** (which units the loss
may touch — *this is the part that actually controls the behaviour*).

**Where the intuition is right:** MSE's suppression has a **fixed point at zero** — `w_o` is driven
only until it is *orthogonal* to the current features, so if old-class features occupy a different
subspace `w_o` can remain informative. Softmax has no fixed point; `z_o` is driven toward −∞
relative to `z_t` and `w_o` accumulates an unbounded anti-`h` component.

**The tension:** more coupling → better calibration, more suppression; less coupling → less
suppression, worse calibration. *You cannot win both with the same knob.* A **generative classifier**
resolves it: p(x|y) per class, no negatives at all, common scale from every class model being a
normalised density over the same input space rather than from a shared denominator.

### 4.5 Where does forgetting live — trunk or head? `[EMPIRICAL, indirect]` — **DO NOT TRUST**

> ⚠️ **This section is a sweeping claim and is now contested. Do not carry it onto a slide.**
> Every archived document asserts it; **experiment 26 (§6.6.3) points the other way**, reporting
> that hidden-layer freezing recovers a large amount *once suppression is masked off*. Neither the
> claim below nor the contradiction is verified. If the talk needs an answer to "trunk or head", it
> gets a **fresh experiment designed for that slide**. Retained for the reasoning, not the verdict.

Three independent observations already in our own data point at the **output layer** as the dominant
damage site in this setup:

1. **Task-IL barely forgets** (probe and head both ≈99%) — only the trunk can drift there, so trunk
   drift is comparatively mild in this network.
2. **Class-IL task-1 accuracy falls to 0%, not 50%.** On a two-class task a merely degraded internal
   description would guess and land near 50%. Falling to 0% means the *wrong* answer is chosen every
   time — the new task's output units are capturing argmax. That is a scoring failure, not a
   description failure. *(Still to be confirmed against the eval code — §10.2.)*
3. **Replay is the only method that retains anything**, and it changes nothing about the learning
   rule; its entire effect is that an old class appears as a *positive* again, cancelling the
   downward push. If the trunk were the problem, twenty images per class would not rescue it.

**Why this matters.** A learning rule decides *how blame is shared among the hidden units*. It does
not decide *whether the output units compete*. If the dominant damage is in the output layer, **no
choice of learning rule can fix it** — which is exactly what the results show, and which also bears
on advisor point 4, since node selection is a hidden-layer intervention.

**The test that settles it: the NCM probe (E2).** Discard the head; classify by nearest class mean
over the 64 hidden features. If task-1 accuracy jumps from ≈0% to substantial, the representation
survived and the head was the entire problem — and PC's advantage should *shrink* under this
readout. This is a decision point, not an optional extra.

### 4.6 The corrected PC mechanism hypothesis `[HYPOTHESIS]` — H1

**Retired (see §9.1):** "an already-correct output has zero error, so its weights barely move."

**Current, testable:** after settling, `e1 = x1 − x0 @ W1` and `ΔW1 ∝ x0ᵀ e1`. Because `x1` is
initialised to `mu1`, `e1` after settling **is literally the displacement of the hidden layer from
its feedforward value**. So:

> PC changes a weight in proportion to the activity displacement its settling required, and
> therefore interferes with task 1 only to the extent that satisfying task-2's target forces
> movement in the hidden units task 1 depends on.

This reframes the question from *"is the error zero?"* (no) to ***"where does the weight movement
go?"*** (measurable). → **E7**. With 64 hidden units, overlapping digit inputs and a single head,
representational overlap is high, so task-1 units may be overwritten anyway — which is why the
experiment is decisive either way.

---

## 5. Metrics

| Metric | Definition | Use | Status |
|---|---|---|---|
| Per-task accuracy | accuracy on each task's fixed class set, held-out test split | primary | ✅ in use |
| **ACC1–ACC2 trajectory** | task-1 accuracy vs task-2 accuracy, time removed | **most robust forgetting metric found** | ✅ in use |
| **Area above diagonal** | integral of the trajectory above the ACC1 = ACC2 line | learning/forgetting *trade-off efficiency* | ✅ in use |
| Crossover height | accuracy where the two task curves cross after the switch | single-number trade-off summary | ✅ in use |
| Forgetting | max task-1 accuracy − final task-1 accuracy | standard CL metric, comparable to [R25] | ⬜ to lock |
| **Target alignment** [R1 Fig 3b] | `cos(target − out_before, out_after_no_target − out_before)` | the literature's own interference measure | ⬜ **E6 — never run** |
| **Linear probe / NCM** | classify from hidden features, head discarded | **separates calibration from representation failure** | ⬜ **E2 — never run** |
| Per-weight \|Δw\| by importance rank | accumulated during later contexts, split by earlier-context Fisher rank | mechanism: where movement goes | ⬜ **E7** |

### 5.1 Measurement traps — hard-won, do not repeat `[SETTLED]`

- **Never evaluate on training data.** An earlier bug did this and invalidated a day of sweeps. Use
  the held-out test split.
- **The flat line at exactly 100/n_classes is not chance** — it is the **collapse floor**, the score
  of a model predicting one class for everything.
- **Accuracy is a threshold readout.** After a task switch nothing appears to happen for ~20 steps
  while logits climb, then it flips fast. Always log raw pre-argmax outputs alongside accuracy.
- **Below-chance task-1 accuracy** (0%, not 50%) means argmax is systematically captured by later
  units — output-layer suppression, not representation loss.
- **Do not fit sigmoids to accuracy curves.** They are step-like and noisy. Use threshold crossings
  and the trajectory plot.
- **`cur%` is degenerate at one class per task**; **`seen%` has a changing denominator.** Prefer
  per-task accuracy on fixed class sets.
- **Report final new-task accuracy alongside retention.** "Forgot less" is not a result if it is
  confounded with "learned less".
- **Never put ReLU on the output layer.** Zero gradient below zero makes an old-class unit pushed
  negative *permanently dead* — it converts a recoverable calibration problem into an irreversible
  one.
- **More than one pass over the data is cyclic revisiting**, not longer continual learning. The
  baseline mean drifts ≈20%→≈30% then plateaus — residual output structure once every class has been
  trained, not learning-to-not-forget.
- **Standing rule:** every reported number comes from a logged array with a seed count and a
  confidence interval. Nothing is read off a plot into prose. **Every script saves its arrays to
  `.npz` next to its figure.** Experiment 11 did not, and the result is D2 (§9.3): two documents
  read two different retention figures off the same plot, and there is now no way to tell which is
  right. Thirteen later scripts do save arrays; make it universal.

---

## 6. Results to date `[EMPIRICAL]` — provisional

⚠️ **Every figure below is from prior runs, reported second-hand in the archived handoff documents.
None is reused for the presentation — every experiment in §11.3 is re-run from scratch.** Retained
as priors for sanity-checking new results, not as evidence. **Read §9.3 first.**

### 6.1 Class-IL, coarse splits (14×14, SGD)

| Split | backprop | eqprop | replay |
|---|---|---|---|
| 10×1 (100 iters/task) | ≈10% (floor) | ≈10% | ≈64% |
| 5×2 (100 iters/task) | ≈20% | ≈20% | ≈60% |

*5×2 ran only 500 total updates vs 1000 for 10×1 — `ITERS=200` gives a matched budget (CTRL-4).*

**2×2 ([0,1] then [2,3], 100 iters/task, batch 32)** — the most informative early run: backprop
holds task 1 at 100% until ≈step 110 then collapses to 0 (a cliff); replay ends ≈95% on both;
eqprop *forgets before it learns* (task 1 collapses at the switch, task 2 rises only afterwards);
pc decays as a slope over ≈100 steps, ending ≈15%.

### 6.2 The four-method comparison (experiment 11, 10 random digit pairings)

**Protocol, verified from the code on 2026-08-10** `[SETTLED]` — `11_consolidate_pairs_4methods.py`
draws `rng.permutation(10)[:4]` and forms `tasks = [[d0,d1],[d2,d3]]`: **2 tasks × 2 classes**,
10 random digit pairings, `ITERS = 100` per task, so 200 steps with a single switch at step 100.
Collapse floor 25%. **The "5×2" label attached to this run in two source documents — including the
most recent knowledge base — is wrong** (D1, resolved).

**Two irreconcilable versions of the results table exist in the sources. Both are recorded; neither
can be re-derived, because this script never logs its arrays — see D2 in §9.3.**

**Version A** (`energy-based-memory-and-continual-learning.md` §5.3, read off plots, ±3%)

| Method | Peak T1 | Final T1 | Final T2 | Sum | vs diagonal |
|---|---|---|---|---|---|
| backprop | ≈85% | ≈0% | ≈78% | 78 | below |
| replay | ≈84% | **≈37%** | ≈75% | 112 | on / slightly above |
| eqprop | ≈74% | ≈3% | ≈66% | 69 | **well below (worst)** |
| pc | ≈86% | ≈10% | ≈82% | 92 | **above for most of the path** |

**Version B** (`understanding-ffnn-bp-ebm-pcn-pc-eqprop.md` §5.6)

| Method | Crossover | Final T1 | Final T2 | Shape |
|---|---|---|---|---|
| backprop | ≈65% | ≈0% | ≈97% | vertical cliff at the switch |
| replay | ≈85% | **≈68%** | ≈96% | dips then recovers; only method ending up-right |
| eqprop | — | ≈0% | ≈95% | noisy throughout; forgets before it learns |
| pc | ≈75% | ≈8–10% | ≈97% | slope not cliff; bows above the diagonal, still ends top-left |

**Both versions agree on the qualitative shapes:** backprop = cliff; replay = drop then
plateau/recovery with a very wide thin-line spread (buffer-composition variance); eqprop = noisy,
scattered runs; pc = gradual decay, tightly clustered runs.

### 6.3 The two orderings that disagree — **DO NOT TRUST**

> ⚠️ **Sweeping claim, unverified, and produced by a script that has never been run in its current
> form** (§6.6.1). The method ordering below may also be an artefact of output structure rather than
> of the learning rules, which is exactly what experiment 25 was built to test. Keep it as a
> hypothesis about what to look for; do not present it.

- **Trade-off efficiency** (area above the diagonal): **pc > replay > backprop > eqprop**
- **Final task-1 retention:** **replay ≫ pc > eqprop > backprop**

**These orderings differ, and the difference is the whole story.**

> **Honest headline:** *PC buys graceful degradation, not durable retention.* It reduces
> interference per update; it does not **anchor** past knowledge. Durable retention requires an
> explicit anchor — replayed data or a consolidation penalty — and PC has none.

**Per-method mechanism:**

| Method | Clamping / update | Interference reduced? | Suppression cancelled? | Signature |
|---|---|---|---|---|
| backprop | forward pass, fixed | no | no | cliff to 0% |
| + replay | same | no | **yes** | floor, on-diagonal, high variance |
| eqprop | weak clamp, β→0 | no (≈ BP + noise) | no | slow, noisy, below diagonal |
| pc | strong clamp, full relaxation | **yes** | no | above diagonal, no floor |

That PC still decays to ≈10% is itself evidence that the dominant failure in this setup is
**output-layer suppression rather than representation drift**: fixing credit assignment buys a
better path but no floor; supplying old positives buys a floor. PC and replay attack different rows
of the mechanism table and should compose — the obvious missing cell (H4).

### 6.4 Earlier notebook-era findings (28×28, Adam) — still valid

- **Task-IL does not forget** for backprop (probe and head both ≈99%). Earlier apparent forgetting
  was a **polarity artefact** of a balanced 2-way head — fold with `max(acc, 1−acc)`.
- **Class-IL forgets catastrophically:** ≈21.6% final mean; individual tasks collapse to ≈0% —
  active overwriting, not decay to chance.
- **Gradient interference measured:** cosine between current-task and prior-task gradients is
  predominantly **negative** (≈−0.5–0) during Class-IL. Occasional positive (cooperative) cosine
  coincides with slower new-task learning. The sign↔Δloss identity is exact under SGD, approximate
  under Adam.
- **Replay works** (≈78% vs ≈21%) but is not equivalent to joint training (memory-limited).
- **EWC fails in Class-IL** (≈20% ≈ baseline), reproducing [R2] Table 2. Reasons: (a) deadlock —
  protecting old weights prevents learning new classes, learning new ones suppresses old logits, and
  no λ escapes; (b) preserving each task's function cannot create discrimination between classes
  never seen together. **Unresolved (H2):** EWC's Fisher-importance distribution resembles replay's
  yet its accuracy matches baseline ⇒ the failure may be in the **readout, not the features**.
  Test = linear probe on the EWC trunk. *Never run.*
- **Architecture:** one hidden layer is sufficient on MNIST (a second adds ≈0.3% at large
  interpretability cost); the accuracy-vs-width knee is ≈64–128; joint ceiling ≈97–98%.
- **Input-pixel Fisher maps are near-identical** across none/replay/EWC — input importance is a
  dataset property. Method differences appear in the importance *distribution* and deeper layers.

### 6.5 Reproduction attempt (experiment 12) — abandoned

A like-for-like reproduction of [R1] Fig 4d **was not achieved**. Roughly a week was spent on
hyperparameter matching; previously working experiment-12 code was broken and the codebase became
unreadable.

**Ruling (2026-08-10): stop chasing a numeric match.** Reproduce the **protocol**, not the
hyperparameters. The claim under test is qualitative — *does PC forget less than BP under matched
conditions?* — so this is a **conceptual replication**, which is the scientifically standard framing.
See §8.

**Second ruling (2026-08-10, after recovering §6.6): the reproduction is closed.** A later attempt
(`34_reproduce_bogacz_fig4d.py`) rebuilt the configuration from their published YAML rather than the
paper text and still produced results inconsistent with the paper. The cause was never found. It
costs hours per run. **Do not re-run it.** The forward path is to close the gap between our own
cheap 2×5 setup and their configuration one axis at a time, so their claims can be explored
efficiently. The axes are listed in §6.6.4.

---

## 6.6 Recovered experiment series 13–34 — **prior work, treated as unreliable**

⚠️ **Read this framing first.** None of the fourteen archived source documents mentions any of these
experiments; they were written after the last handoff was recorded. They were recovered by reading
`experiments/` directly on 2026-08-10.

**Status of everything in this section: informative, not evidential.** These runs used an
inconsistent array of setups (§6.6.2), several contradict each other (§6.6.3), the reproduction
failed, and the code they ran on is not trusted (§6.7). **They inform direction and experiment
design. No claim from them goes on a slide.** Anything worth asserting gets a fresh experiment,
designed for the slide that needs it, at the time that slide is built.

### 6.6.1 What was asked, and whether it ran

| # | Question it asks | Ran? |
|---|---|---|
| 11 | Four rules across random digit pairings, 2 tasks × 2 classes | **no** — only `_old.png` from a version no longer in the repo |
| 12 | The same on 2 tasks × 5 classes | **no** — `_old.png` only |
| 13 | Achievable accuracy vs hidden width and number of classes (the ceiling) | yes |
| 14 | Which EqProp learning rate matches backprop's learning *speed*; does β affect speed | yes |
| 15 | Forgetting when every method is trained to the *same accuracy standard* | yes |
| 20 | Is the damage in the hidden layer or the output layer (mask × NCM factorial) | yes |
| 21 | At what width does representation drift start to contribute | yes |
| 22 | Does keeping the internal representation make any difference (freeze W1 / W2 / mask) | yes |
| 23 | Is MNIST too easy — does a harder dataset widen the useful width range | yes |
| 24 | Can each rule learn at all under each output structure (joint, safety gate) | yes |
| 25 | How much of the rule difference is output *structure* rather than credit assignment | yes |
| 26 | What remains after masking — mask × freeze-W1 factorial × 3 rules, hidden 16 | yes |
| 27 | The same factorial at hidden 64 | **no** |
| 30, 32, 33 | Reproduction attempts, superseded | 30 only |
| 34 | Reproduction from their published YAML | yes — **failed to match the paper** |

### 6.6.2 ⚑ The setups are not comparable to each other

This is the central problem with the series and the reason a coherent story cannot be assembled from
it. Nothing here was held fixed across experiments:

| Axis | Values used across 13–34 |
|---|---|
| Classes per task | 2 (11, 15, 20) · 2 **and** 5 (21) · 5 (12, 22, 23, 25, 26, 27) |
| Hidden width | 64 (11, 15, 20, 24, 27) · 16 (25, 26) · sweep 2–128 (13, 21, 23) · 16 and 64 (22) |
| Biases | **off** (20, 21) · **on** (22, 23, 24, 25, 26, 27) |
| Backprop learning rate | 0.05 · **0.1** (22 only) |
| Task-1 stopping criterion | 0.80 (15, 20) · 0.75 (21) · 0.70 (22, 23, 25, 26, 27) |
| Task-2 budget | matched-accuracy early stop (15) · 300 fixed (22, 23) · 250 fixed (26, 27) |
| Dataset | MNIST 14×14 · both (23) · Fashion-MNIST (34) |
| Seeds | 8 or 10 |
| Output structure | ReLU+CE vs tanh+SE vs hinge (≤15) · standardised linear+SE 1/0 (≥24) |

**Consequence:** no two of these experiments can be placed on the same axis, and a number quoted
from one does not transfer to another. The fresh series must fix a single protocol first and vary
one thing at a time.

### 6.6.3 Recorded outcomes and the conflicts between them

Reported here so the work is not repeated blindly. **Every figure is unverified.**

- **Exp 26** (hidden 16, 2×5) reports that masking alone recovers task 1 to ≈30% while masking
  **plus** freezing W1 recovers it to ≈65%, against a pre-switch peak of ≈84%. If that held, the two
  pathologies would **interact** rather than being independent, and a learning rule would have a real
  budget to compete for — **contradicting §4.5 and §6.3, which every archived document asserts.** It
  also reports PC *behind* backprop in that cell, and PC *ahead* of backprop in the unmasked-frozen
  cell. **Unverified; both readings are provisional.**
- **Exp 22's prose contradicts its own saved arrays.** Its docstring quotes a freeze-W1 gain of
  +3.3 points at hidden 16 and +0.1 at 64; the saved `.npz` gives ≈+0.0 at both. The masked figures
  roughly agree. **Experiments 25 and 26 selected hidden = 16 on the strength of the quoted number.**
- **Exp 23 refuted its own hypothesis.** It expected Fashion-MNIST to widen the range of widths where
  the hidden layer matters. The saved arrays show the opposite — the freeze-W1 gap is smaller on
  Fashion at nearly every width, and trained-NCM-minus-untrained-NCM goes negative. **The proposed
  dataset switch is not supported by its own data.**
- **Exp 21 invalidated exp 20's probe.** It records that exp 20's NCM arm sat near ceiling
  throughout, including before task 2 was trained, so it had no dynamic range and could not have
  detected drift. Exp 21 added a random-init floor and frozen prototypes to fix this.
- **Exp 24** reports the three rules landing within ≈2 points of each other under linear + squared
  error with a 1/0 target — the tightest agreement of the four output structures tested — and reports
  EqProp failing under softmax with heavy saturation. This is the empirical case for the
  standardisation in §7.1. **Still unverified, but it is the most directly useful result recovered.**
- **Exp 34** records a possible error in the published source: [R1]'s `learn_code` slices
  `(outputs - target)[:, 0:-1]`, excluding the last output unit from the loss while still including
  it at test time. Reproducing that literally collapses the model. **This is a claim about a
  published paper and must be verified against the repository before it is repeated anywhere.**

### 6.6.4 Closing the gap to [R1]'s configuration

Recovered from the stored config in `34_reproduce_bogacz_fig4d.npz`, traced to
`base-shuffle-task-5-FashionMNIST.yaml`. These are the axes on which our cheap 2×5 setup differs
from theirs — the list to work through one at a time rather than by attempting the whole
reproduction again:

| Axis | Theirs | Ours (2×5) |
|---|---|---|
| Input | 784 (28×28) | 196 (14×14) |
| Depth | 3 hidden layers of 32 | 1 hidden layer |
| Activation | sigmoid | tanh |
| Biases | off | on (post-24) |
| Output layer | 5 shared units — **Domain-IL** | 10 units (Class-IL) or 5 (Domain-IL) |
| Training data | 600 images per class | full split |
| Batch | 500 | 32 |
| Loss reduction | **summed** over batch and outputs | mean |
| Schedule | alternating, 4 iterations per task, 84 analysed | single switch |
| Inference | T = 64 steps, x_lr 0.1, discount 0.9 | ~50 fixed steps |
| Dataset | Fashion-MNIST | MNIST |

The summed-vs-mean reduction alone explains why their learning rates (1e-4–5e-3) and ours
(5e-3–5e-1) were never comparable units.

## 6.7 Status of the codebase — **not trusted**

Before the reproduction work, `src/` handled **single-layer networks only**. The reproduction
required arbitrary depth, and the code was refactored to provide it — introducing the `Arch` /
`Objective` dataclasses and a `run_classil` runner that the archived documents do not describe.
**The reproduction then failed, and the cause was never identified, so the refactored code cannot be
trusted.**

Consequences: the interface contract recorded in §11.2 (`make_* → (train_step, predict)`) describes
the *pre-refactor* code and is stale. Any result in §6.6 was produced by the untrusted code.

**Pending work, ahead of or during the first experiment of the new series:** read the current
modules, check them, and simplify — removing complicated interdependencies that could break core
functions. Not now; this is a planning phase.

---

## 7. Controls: variables that must be matched `[SETTLED]`

A *control* here is a variable that currently differs between methods and must be equalised before
any difference in forgetting can be attributed to the learning rule.

| ID | Confound | Status | Fix |
|---|---|---|---|
| **CTRL-1** | **Learning rate not matched** (BP 0.05, EqProp 0.005, PC 0.05). Comparing forgetting speed at different learning speeds measures the learning rate, not the rule. | open | Grid-search LR **per rule**, as [R1] does; then match `to_learn`. |
| **CTRL-2** | **Loss and nonlinearity not matched.** `make_backprop`/`make_replay` use ReLU + cross-entropy; `pc`/`eqprop` use tanh + squared error — three things vary at once. | open | Matched BP control (§7.1). |
| **CTRL-3** | **Compute not matched.** PC runs ~50 settling steps and EqProp up to 2×500 per weight update; backprop runs one forward + one backward. | open | Report an equal-compute backprop arm alongside the equal-epoch one. Cannot be designed away. |
| **CTRL-4** | **Iteration budget not matched across splits** — 5×2 ran 500 total updates vs 1000 for 10×1. | known | Set `ITERS = 200` for 5×2. |
| **CTRL-5** | **Final new-task accuracy not matched** — "forgot less" is confounded with "learned less". This is exactly why EqProp's gentle-looking decay was a false impression. | open | Match final task-2 accuracy, or report the trajectory plot, which removes time. |
| **CTRL-6** | **Regime.** PC's claimed advantage is largest at **batch size 1** and with **depth** [R1 Fig 4a–c, R3]; the current net is 1 hidden layer at batch 32, so PC is measured near its weakest point. | known | Depth sweep (E8) and, optionally, a batch-size-1 arm — but note batch 1 destroys EqProp (§9.2), which is itself reportable. |

### 7.1 The single shared specification (closes CTRL-2)

```python
# no bias terms anywhere
x1     = x0 @ W1                    # hidden pre-activation: LINEAR
out    = torch.tanh(x1) @ W2        # activation applied on the way out
target = one_hot(y)                 # 1 for the true class, 0 for the rest
loss   = 0.5 * ((target - out) ** 2).sum(1).mean()
# plain SGD; prediction = argmax over the outputs
```

**Why squared error and not softmax + CE:** PC's energy *is* squared prediction error — give it a
softmax output and it stops being predictive coding. **Squared error is the only loss all three
rules can take unaltered**, so it is the one that must be shared, and backprop is the method that
should move. EqProp's original formulation also uses squared error; the ±1 hinge was **our** choice,
not part of the algorithm, and dropping it removes the harshest downward push in the comparison —
very likely why EqProp currently looks worst.

**The cost, accepted explicitly:** squared error with a linear output has no shared denominator, so
the class scores are not held on a common scale the way softmax guarantees. In Class-IL that matters
(§4.4), and absolute accuracies will not be comparable with [R2]'s published tables. Comparability
*between our own methods* is worth more than comparability to an external table. State this in the
methods chapter.

---

## 8. Deviations from the literature, and their justifications

| Axis | Song & Bogacz [R1] | Ours | Justification |
|---|---|---|---|
| Dataset | Fashion-MNIST 28×28 | **MNIST 14×14** (196 in) | EqProp settling is the runtime bottleneck; 14×14 is ≈4× cheaper and keeps MNIST intact (≈97% BP ceiling); 8×8 degrades classes. **Evidence required → E0.** |
| Depth | 15 layers (Fig 3f–h); unspecified for 4d | **1 hidden layer (196→64→10)** | Isolates the learning rule from a depth confound, and is the **conservative** choice: [R1] Fig 3e/4h claim PC's advantage *grows* with depth, so our result is a **lower bound**. |
| Activation | LeakyReLU | **tanh** | Matched across all rules; the verified PC implementation uses tanh; [R1] chose LeakyReLU because sigmoid fails in *deep* nets — not applicable at depth 1. |
| Loss | squared error | **squared error, matched** | Removes the loss confound identified as CTRL-2. |
| Scenario | **Domain-IL** (5 shared outputs) | **both Domain-IL and Class-IL** | Class-IL is primary: largest dynamic range between methods [R2 Tables 2–3], standard ML benchmark, matches the common biological case, and exercises trunk *and* output layer so forgetting can be attributed. Domain-IL is also run because it is the scenario the interference claim was demonstrated in, and it isolates trunk drift. |
| Protocol | 2 tasks × 5 classes, **alternating**, 4 iters each to 84 | **5 tasks × 2 classes, sequential**, both scenarios | Holds partition and schedule constant so the output layer is the only difference between our two scenarios; [R1]'s protocol varies task count and schedule alongside scenario. Also matches [R2]'s Split MNIST protocol, so baselines are comparable to a published table. |
| Optimiser | SGD | SGD | Matched. Plain SGD everywhere: exact interference identity, no cross-task momentum confound, matches EqProp's own updates. |
| Learning rate | per-rule grid search | **per-rule grid search** | Not a deviation — required for fairness (CTRL-1). |
| Batch size | 32 | 32 | Matched. |
| Seeds | 10 | ≥5 | Compute budget; report 68% CI as they do. |
| Architecture | FC | FC, no CNN | Avoids inductive priors so observations generalise across the method comparison. |

**Framing:** this is a **conceptual replication** — the protocol is reproduced, the hyperparameters
are not. Say so on the deviations slide.

---

## 9. Corrections, refutations, and unresolved contradictions

### 9.1 Refuted — recorded so they are not re-derived `[REFUTED]`

- **"PCN error nodes are basically backprop's errors."** True only in the partial-relaxation /
  infinitesimal limit, or under an engineered equivalence that [R3] says is not general. §3.3.
- **"An already-correct output has zero error, so its weights don't move — that is how PC avoids
  forgetting."** This is a within-a-single-forward-pass statement ([R1] Fig 1). During task-2
  training you clamp task-2 inputs *and* task-2 targets, so the error is task-2's and it drives
  change through whatever units task-2's settling implicates — including task-1's. **It gives task 1
  no protection at all.** Replaced by H1 (§4.6).
- **"EqProp forgets less than backprop."** Its decay only *looks* gentle because everything is
  slower — lower peak on task 1, slower and lower task 2. With time removed, the trajectory plot
  shows it as the worst panel.
- **"Replacing softmax removes the suppression."** §4.4.
- **"Per-class output heads are a route to an EBM advantage."** With task ID it is Task-IL (which
  forgets for no method); without it, calibration fails equally for all methods. Investigated and
  rejected.
- **"The interference claim's scenario boundary is the thesis of the talk."** Retired 2026-08-10 as
  premature — no results in both scenarios yet. Demoted to §10.1.

### 9.2 Constraints and failure modes `[SETTLED]`

**Hardware / runtime.** CPU only. A GPU helps less than expected — each settling step is a tiny
matmul, so per-step overhead dominates; parallel CPU processes beat one GPU for sweeps. Cost
ordering: **EqProp ≫ PC ≫ backprop** (PC's *prediction* is a plain feedforward pass, no settling).
Keep sweeps cheap: subset to 10k, 1 epoch, fewer settle steps; confirm only the winner on full data.

**EqProp failure modes.** *Saturation is the killer* — as weights grow tanh flattens, `tanh'(h)→0`,
and the feedback path carrying the nudge is severed; track `% of |tanh(h)| > 0.95` as a first-class
diagnostic, with low `lr` as the main control. *The nudged phase never reaches an absolute
tolerance* — the hinge keeps pushing while the margin is unmet, so settling stops on **patience**,
not a fixed `tol`; worth a sentence in the write-up. *Warm-start the nudged phase from the free
equilibrium*, or it dominates runtime. *Batch size 1 destroys EqProp* — with ±1 targets and no batch
to average over, every update reconfigures the network to the most recent image; keep batch ≥16.
*No ½x² self-term in the energy*, so during generation `x` has no restoring force and pins at the
clamp bounds — a weak generator.

**Conceptual limits of PC** [R3]. Requires approximately symmetric forward/backward weights — a
plausibility issue PCNs *share* with backprop, as are signed real-valued error signals; PCNs differ
from backprop only in that feedback influences activity during inference. Iterative relaxation is
expensive on von Neumann hardware. Architectural inflexibility: the energy is meticulously
structured, and multiplicative second-order interactions (a transformer's QKᵀ) would need
higher-order, hard-to-stabilise terms. No BPTT analogue — existing work achieves only a one-step
approximation. **Scaling:** PC rivals backprop on small/medium architectures but degrades on 9-layer
convnets and ResNets where backprop improves [R23] — "PC beats BP" and "PC loses to BP" are both
true, at different scales.

**Capacity.** Hopfield-type systems hit *blackout catastrophe* at saturation; a sharp store cannot
grow unbounded and needs a **consolidation pathway, not just storage**. [R4] notes EWC shares this:
past capacity it performs *worse* than plain gradient descent.

### 9.3 ⚑ Unresolved contradictions between the source documents

These are bookkeeping problems inherited from three parallel consolidations of the same handoffs.
**They are open and require adjudication; none has been resolved by guesswork.**

| ID | Contradiction | Status |
|---|---|---|
| **D0** | **Three rival consolidations exist.** `knowledge_base.md`+`timeline.md` (2026-08-10, "chat #009"), `knowledge_base1.md`+`timeline1.md` (undated, "session S4"), and `knowledge_base2.md`+`timeline2.md` (2026-07-29, "chats 013–014") each claim to be *the* consolidation of the same three handoff documents, each invented its own numbering, and none references the others. Only their relative order to the 2026-08-10 pair is certain. | **open** |
| **D1** | **Experiment 11's split.** `understanding-…md` §1/§5.5 said 2 tasks × 2 classes, single switch at step 100; `energy-based-memory-…md` §2.5 and the 2026-08-10 knowledge base both labelled the same run "5×2". | **RESOLVED 2026-08-10 from the code.** `11_consolidate_pairs_4methods.py:114–115` draws four digits and forms two tasks of two classes; the docstring says "Same 2-task Class-IL problem". **It is 2×2.** The oldest document was correct and the newest was wrong — which is why "keep the later claim" is not a safe default here. |
| **D2** | **Replay's final task-1 accuracy:** ≈37% (Version A) vs ≈68% (Version B). Backprop's final task-2: ≈78% vs ≈97%. The 2026-08-10 knowledge base adopted ≈37% without flagging the conflict. | **OPEN — and now known to be unresolvable from the archive.** Experiment 11 never writes an `.npz`; `curves` is held in memory, plotted and discarded. Both figures were therefore read off plots, which is exactly how they came to disagree. **Only a re-run can settle it** — and E4 supersedes it. Treat both numbers as uncitable. |
| **D3** | **PC's final task-1 accuracy:** ≈15% (handoff), ≈10% (Version A), ≈8–10% (Version B). Likely single-run vs 10-run mean. | **open** |
| **D4** | **EqProp's final task-1 accuracy:** ≈3% vs ≈0%. Same cause as D3. | **open** |
| **D5** | EqProp described both as "forgets before it learns / worst crossing" and as "gradual monotone decay". | **not a conflict** — both are true: decay begins at the switch while task 2 rises only afterwards. Merge into one description. |
| **D6** | PC's slope called "the closest thing to supporting the prospective-configuration claim" in one document and "PC does **not** retain" in another. | **not a conflict** — two different questions, *shape of the path* vs *endpoint*. State both explicitly, always; this distinction **is** the result (§6.3). |
| **D7** | **Slide counts.** `timeline.md` #008 records "20-slide plan (12 mandatory / 8 optional)"; `presentation_plan.md` and `current.txt` say 15 mandatory / 5 optional. | **resolved by recency** — 15/5 stands; #008 records an intermediate state later revised in #009. |
| **D8** | **Experiment naming.** `timeline.md` #008 records seven experiments P0–P7; the final plan has ten, E0/E-metrics/E1–E8. Earlier documents also use experiment numbers 12–19 and labels D1–D3/C1–C3/M1–M4/W1–W2 for the same underlying work. | **resolved by recency** — the E-series in `presentation_plan.md` is authoritative. Older labels appear only in the archive. |
| **D9** | **Scope.** `research_plan.md` describes a 3×3 scenario×dataset grid (Task-IL/Domain-IL/Class-IL × MNIST/Fashion/KMNIST) with EWC and a contrastive EBM; `research_log.md` records amendments narrowing it. | **resolved by recency** — superseded by the advisor's four points (§1). Kept as the record of what was cut. |
| **D10** | **Resolution and optimiser.** `research_plan.md`/`research_log.md` specify 28×28 and Adam; everything later specifies 14×14 and plain SGD. | **resolved** — amendments recorded in `research_log.md` itself (Adam→SGD, no CNNs); 14×14 + SGD stands. |
| **D11** | **EBM build order.** `research_log.md` amends the order to "predictive coding first (1.5 before 1.4)"; the handoffs record EqProp built first with PC added later. | **open, low stakes** — the plan was amended and then not followed. Historical only. |

---

## 10. Open questions

### 10.1 Retired pending evidence

**Does the interference claim have a scenario boundary** — is PC's advantage present in Domain-IL
and absent in Class-IL, while its *target alignment* advantage stays scenario-independent? Retired
2026-08-10 as premature. Revisit only after E4, E5 and E6. **Do not write slides that presuppose the
answer.**

### 10.2 Active

1. Is each method's failure calibration or representation? **(E2)**
2. Is the forgetting in the trunk (W1) or the head (W2)? **(E3)**
3. What is target alignment in our own networks, per rule and scenario, and does it track retention?
   **(E6)**
4. Does PC concentrate weight movement away from earlier-context-important weights, or is the
   advantage simply lower update variance? Both outcomes are defensible. **(E7 / H1)**
5. Does depth change the conclusion? **(E8 / CTRL-6)**
6. §4.1's suppression argument is derived for softmax cross-entropy but the implementation uses
   squared error. **Restate it for MSE before slide 9 relies on it.**
7. The trajectory plot was designed for 2 tasks. With 5 contexts, decide the axes — pairwise
   first-vs-last, or mean-of-seen vs current. **(E-metrics)**
8. Confirm the evaluation protocol is global argmax over a shared head, as the 0%-not-50% signature
   implies (§4.5).
9. Does the ±1 vs one-hot target structure explain the whole method ordering? (Swap EqProp's hinge
   for one-hot and PC's one-hot for ±1 — a one-line change, very high insight-to-effort.)
10. Does EWC stack usefully on PC even though it failed on vanilla BP? **(H2 / H4)**

### 10.3 Deferred — beyond the advisor's scope, for the discussion or future-work chapter

11. Can E_sharp (fast, pattern-separated) and E_smooth (slow, averaged) be trained under a single
    unified objective, or does the sharp component need a separate fast-weight rule? **(H5)**
12. What is the right readout for replay — sharpest energy (verbatim) or intermediate noise
    (recombinant)? The literature points to intermediate.
13. What is the consolidation pathway from sharp to smooth, and what triggers it in a task-free
    stream?
14. Does energy-based replay *selection* (high-energy/outlier samples first) survive in a decomposed
    energy? **(H6)**
15. Advisor point 4: can per-node prediction error at settling time gate which nodes learn new
    stimuli? Implemented as `eqprop_update_gated` (**untested**); a PC version is arguably more
    natural, since PC has an explicit per-node prediction error. **(H3)**
16. "Kinematic consolidation": stiffen a weight at the moment it stops changing, and let the
    stiffness decay. Closest relative is Synaptic Intelligence [R12] (path integral, first-order,
    non-decaying), not EWC; Benna & Fusi [R14] is the biological grounding. **Parked — gate on E7**:
    if PC already leaves task-1 weights alone, stiffness adds little.

---

## 11. Code and experiment plan

### 11.1 Policy (agreed 2026-08-10)

All work is committed; there is no earlier revision to revert to and none is wanted. Move forward
from the current state. Keep both scenarios and the depth parameter — nothing is removed. Move
conditionals **out of the core functions** in `eqprop.py` and `predictive_coding.py`; scenario and
depth resolve at construction time, not inside the update step. Once a core function is readable and
tested, **freeze it**; change it only when an experiment cannot be expressed otherwise. One script
per experiment, one figure per script, named from `__file__`. One change at a time, each tested on
its own, with the previous experiment re-run to confirm its figure is unchanged.

**Interface contract:** every `make_*` returns `(train_step, predict)`. `train_step(x, y)` does one
update. `predict(x, raw=False)` returns class indices, or raw pre-argmax outputs when `raw=True`.
Adding a model = one new `make_*`; experiment scripts change only the `methods` dict.

**Do not reintroduce:** wandb, Optuna, persistent HPO infrastructure, class hierarchies, a shared
`harness.py`. All were built, found to cost more momentum than they saved, and deliberately deleted.

### 11.2 Equations → code — **describes the pre-refactor codebase; see §6.7**

> ⚠️ **Stale.** This records the single-layer code as it stood before the depth refactor. The
> current `src/` uses `Arch` / `Objective` dataclasses and a `run_classil` runner, and the
> `make_* → (train_step, predict)` contract below no longer describes it. The equations are still
> the right reference for *what the rules do*; the interface description is not.

**Predictive coding** — `src/predictive_coding.py`, gradients finite-difference verified to ≈1e-9:

```
x0 (input, clamped) → x1 (hidden, free) → x2 (output, clamped to target during training)
mu1 = x0 @ W1 ;  e1 = x1 − mu1
mu2 = tanh(x1) @ W2 ;  e2 = x2 − mu2
F   = ½|e1|² + ½|e2|²
Inference: relax x1 to reduce F with the target clamped:  dx1 = e1 − f'(x1) ⊙ (W2ᵀ e2)
Learning : ΔW1 = x0ᵀ e1 ,  ΔW2 = tanh(x1)ᵀ e2            (local: pre-activity × post-error)
```

Sign verified: ∂F/∂W1 = −x0ᵀe1, so `+=` is gradient *descent* on energy. `pc_predict` uses the plain
feedforward pass — correct, since with the output unclamped and e1 = 0 that *is* the equilibrium for
a one-hidden-layer net. **`mu1 = x0 @ W1` is linear** (tanh appears only hidden→output), so the
matched BP control must have the same function class.

**EqProp** — `src/eqprop.py`, verified correct:

```
E = ½|h|² + ½|y|² − hᵀ(x·W1) − yᵀ(tanh(h)·W2)
free phase:   settle from h = 0, y = 0
nudged phase: warm-start from the free state; gy += β · ∂/∂y max(0, 1 − target·y)
W1.grad = (gW1_n − gW1_f)/(β·N)   ⇒   effective ΔW1 ∝ +xᵀ(h_n − h_f)/β
```

Targets are **+1 for the true class, −1 for all others**. EqProp's noisiness is EqProp being
EqProp — the finite-nudging gradient bias of [R17] — not a bug.

**Replay** — `src/methods.py`, `make_replay`: it *is* backprop. Stores `per_class` examples the
first time each class is seen and concatenates an equal-sized replay sample into every batch. **No
new learning rule** — it just re-shows old data. That is exactly the anchor the energy-based methods
lack.

**Hyperparameters carried from the archive** (pre-grid-search; CTRL-1 supersedes the learning rates):

```python
IMG_SIZE = 14 ; IN_DIM = 196 ; HIDDEN = 64 ; OUT = 10
BATCH = 32 ; ITERS = 100 per task ; EVAL_PER_CLASS = 100 ; N_RUNS = 10
BP_LR  = 0.05
RP_LR  = 0.05 ; RP_PER_CLASS = 20
EQP_LR = 0.005 ; EQP_BETA = 0.3 ; EQP_DT = 0.3 ; EQP_MAX_STEPS = 500 ; EQP_SETTLE_PAT = 30
PC_LR  = 0.05  ; PC_DT = 0.1 ; PC_STEPS = 50
```

*Joint-training EqProp config reaching ≈91%: lr 0.03–0.1, β 0.3–0.5, dt 0.3–0.5, batch 32–64.*

**Reference configuration from [R2]**, for comparability: Split MNIST base net 2×400 ReLU FC +
softmax; 5 contexts × 2 digits; 2000 iters/context; batch 128; Adam lr 1e-3. Output by scenario:
Task-IL multi-head with only the current context active; Domain-IL single head, 2 units; Class-IL
single head, 10 units, all active.

### 11.3 Current experiment plan (E-series)

Full detail in `presentation_plan.md`. All experiments are re-run from scratch; no earlier result or
plot is reused.

| ID | Question | Slides | Depends on |
|---|---|---|---|
| **E0** | Does 14×14 preserve the joint-training accuracy of 28×28? | 5, 6 | — |
| **E-metrics** | Lock the metric definitions and the plotting module | 7, all | — |
| **E1** | How much does backprop forget in each scenario, and does replay recover it? | 2, 8 | E-metrics |
| **E2** | Do the hidden features survive when the head does not? (NCM probe) | 9, 17 | E1 |
| **E3** | Is the forgetting in W1 or W2? (freeze one, then the other) | 9, 17 | E1 |
| **E4** | How do the four rules compare in Class-IL 5×2? | 12, 14 | E-metrics |
| **E5** | How do the four rules compare in Domain-IL 5×2? | 13, 14 | E4 |
| **E6** | What is the target alignment of each rule in each scenario? | 15 | E4, E5 |
| **E7** | Where does the weight movement go during later contexts? | 16 | E4 |
| **E8** | Does depth change the conclusion? | 18 | E4 |

**Order:** E-metrics → E1 → E4 → E5 → E2/E3 → E6 → E0 → E7 → E8.

### 11.4 Snippets to lock

```python
# Target alignment — [R1] Fig 3b
d_target  = target - out_before
d_learn   = out_after_no_target - out_before      # NB: measured WITHOUT the target provided
alignment = F.cosine_similarity(d_target, d_learn, dim=-1)

# NCM / linear probe — separates calibration failure from representation failure
h = model.trunk(x)                                # discard the head entirely
prototypes = {c: mean_of_h_for_class_c}
pred = argmin_c ||h - prototypes[c]||
# If task-1 accuracy jumps from ~0% to substantial, the trunk survived and the
# output layer was the entire failure.
```

---

## 12. References

Reference keys `[Rn]` are used throughout this file and in `presentation_plan.md`.
**Link status:** ✅ verified from a project PDF or handoff document. ⚠ identifier from general
knowledge — **confirm before citing in the report.**

### 12.1 In-project PDFs (`ref/`)

- **[R1] Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz (2024).** "Inferring neural activity
  before plasticity as a foundation for learning beyond backpropagation." *Nature Neuroscience*
  27:348–358. `song_bogacz_24.pdf` · ✅ https://doi.org/10.1038/s41593-023-01514-1 · code ✅
  https://github.com/YuhangSong/Prospective-Configuration
  *Introduces prospective configuration — settle activities to equilibrium before plasticity — and
  argues it reduces interference relative to backprop across continual learning, online learning, RL
  and human behavioural data.*
  Key: Fig 1 (interference within one association), Fig 2 (energy machine), Fig 3b–e (**target
  alignment**, and its degradation with depth), **Fig 4d–e (two alternating 5-class tasks — note
  this is Domain-IL, §2.3)**, Fig 4a–c (batch size), Fig 4f–g (concept drift, largest advantage),
  Fig 4h (depth), Supp. Fig 6 (which weights to modify), Supp. Fig 7 (less erratic updates),
  Discussion (**the EqProp remark**).

- **[R2] van de Ven, Tuytelaars & Tolias (2022).** "Three types of incremental learning." *Nature
  Machine Intelligence* 4:1185–1197. `s42256-022-00568-3.pdf` · ✅
  https://doi.org/10.1038/s42256-022-00568-3 · code ✅ https://github.com/GMvandeVen/continual-learning
  *Defines the Task-/Domain-/Class-IL taxonomy by whether context identity is known or must be
  inferred at test, and shows parameter regularisation collapses to the no-defence baseline in
  Class-IL while replay holds up everywhere.*
  Key: Table 1 (scenario mappings), Fig 2 (Split MNIST three ways), eq. (2) (active-unit softmax),
  Tables 2–3 (empirical comparison), Methods (architectures, generative classifier, iCaRL).

- **[R3] Dong, Peng & Wu (2025).** Commentary on Song et al. *Intelligent Computing* 4:0244.
  `dong_wu_rev_song_bogacz.pdf` · ✅ https://doi.org/10.34133/icomputing.0244
  *Places PCNs in the EBM lineage, frames training as an EM procedure, and argues PCNs are not "more
  biologically plausible" than backprop but a fundamentally different paradigm — while cataloguing
  computational overhead, architectural inflexibility and the absence of a BPTT analogue.*
  Key: EBM lineage; **strong-clamp (PC) vs weak-clamp (EqProp)**; "the exact-backprop equivalence is
  not general"; interference-regime caveat (batch size 1 / depth); the symmetric-weights problem.

- **[R4] Kirkpatrick et al. (2017).** "Overcoming catastrophic forgetting in neural networks." *PNAS*
  114:3521–3526. `kirkpatrick_17.pdf` · ✅ https://doi.org/10.1073/pnas.1611835114
  *Introduces Elastic Weight Consolidation — a diagonal-Fisher quadratic penalty anchoring weights
  important to previous tasks — motivated by dendritic-spine persistence, and notes it degrades below
  plain SGD once capacity is exceeded.*

### 12.2 Core external references

- **[R5] Scellier & Bengio (2017).** "Equilibrium propagation: bridging the gap between energy-based
  models and backpropagation." *Front. Comput. Neurosci.* 11:24. ⚠
  https://doi.org/10.3389/fncom.2017.00024 — *Defines EqProp: a free phase and a weakly nudged phase,
  with the weight update given by the difference between the two equilibria. The chosen EBM for
  advisor point 1.*
- **[R6] Whittington & Bogacz (2017).** "An approximation of the error backpropagation algorithm in a
  predictive coding network with local Hebbian synaptic plasticity." *Neural Computation*
  29:1229–1262. ⚠ https://doi.org/10.1162/NECO_a_00949 — *PC with purely local Hebbian plasticity
  approximates backprop — the "approximation" regime prospective configuration departs from.*
- **[R7] Millidge, Song, Salvatori, Lukasiewicz & Bogacz (2022).** "Backpropagation at the
  infinitesimal inference limit of energy-based models." arXiv:2206.02629 ✅
  https://arxiv.org/abs/2206.02629 — *Proves PC, EqProp and contrastive Hebbian learning all reduce to
  backprop in the infinitesimal inference limit — the formal basis for "differences appear only at
  full relaxation".*
- **[R8] Song, Lukasiewicz, Xu & Bogacz (2020).** "Can the brain do backpropagation? Exact
  implementation of backpropagation in predictive coding networks." *NeurIPS* 33:22566–22579. ⚠ —
  *The engineered exact-equivalence result (Z-IL) that [R3] flags as non-general.*
- **[R9] Li, Du, van de Ven & Mordatch (2022).** "Energy-based models for continual learning."
  *CoLLAs.* ✅ https://arxiv.org/abs/2011.12216 · code ✅
  https://github.com/ShuangLI59/ebm-continual-learning — *A conditional-energy EBM that beats replay
  in Class-IL by scoring class-conditional energies with contrastive divergence instead of
  normalising a softmax over all classes. **The acknowledged alternative EBM**, deliberately deferred.*
- **[R10] van de Ven, Siegelmann & Tolias (2020).** "Brain-inspired replay for continual learning
  with artificial neural networks." *Nature Communications* 11:4069. ⚠
  https://doi.org/10.1038/s41467-020-17866-2 — *Generative replay with the hippocampus framed as a
  generative network, removing the need for a stored buffer.*
- **[R11] McCloskey & Cohen (1989).** "Catastrophic interference in connectionist networks: the
  sequential learning problem." *Psychology of Learning and Motivation* 24:109–165. — *The original
  demonstration that sequentially trained connectionist networks abruptly lose earlier associations.*
- **[R12] Zenke, Poole & Ganguli (2017).** "Continual learning through synaptic intelligence." *ICML.*
  ⚠ https://arxiv.org/abs/1703.04200 — *Per-weight importance accumulated online as a path integral
  along the trajectory; first-order and non-decaying.*
- **[R13] Laborieux, Ernoult, Hirtzlin & Querlioz (2021).** "Synaptic metaplasticity in binarized
  neural networks." *Nature Communications* 12:2549. ⚠ https://doi.org/10.1038/s41467-021-22768-y —
  *Hidden weight magnitude as a metaplastic consolidation variable, working without task boundaries
  or replay — the template for "reduce CF in the EBM".*
- **[R14] Benna & Fusi (2016).** "Computational principles of synaptic memory consolidation."
  *Nature Neuroscience* 19:1697–1706. ⚠ https://doi.org/10.1038/nn.4401 — *Complex-synapse model with
  multiple timescales inside one synapse; the biological grounding for consolidation as a third
  mechanism.*

### 12.3 Supporting external references

- **[R15] Millidge, Salvatori, Song, Bogacz & Lukasiewicz (2022).** "Predictive coding: towards a
  future of deep learning beyond backpropagation?" IJCAI survey. ✅ https://arxiv.org/abs/2202.09467 —
  *PC as a general-purpose local-computation algorithm; motivated by parallelisability and
  neuromorphic hardware.*
- **[R16] Salvatori et al. (2024).** "A stable, fast, and fully automatic learning algorithm for
  predictive coding networks." *ICLR.* ⚠ — *Incremental PC removes sensitivity to relaxation
  hyperparameters and answers the relaxation-cost objection.*
- **[R17] Laborieux et al. (2021).** "Scaling equilibrium propagation to deep ConvNets by drastically
  reducing its gradient estimator bias." *Front. Neurosci.* 15:633674. ⚠ — *Finite-nudging bias is
  why vanilla EqProp does not scale past MNIST — the explanation for our EqProp results.*
- **[R18] Ramsauer et al. (2021).** "Hopfield networks is all you need." *ICLR.* ✅
  https://arxiv.org/abs/2008.02217 — *Transformer self-attention is the update rule of a modern
  continuous-state Hopfield network; EBMs are not a fringe framework.*
- **[R19] Bai, Kolter & Koltun (2019).** "Deep equilibrium models." *NeurIPS.* ⚠ — *"Settle to a
  fixed point, then differentiate implicitly" as mainstream ML.*
- **[R20] Kendall et al. (2020).** ✅ https://arxiv.org/abs/2006.01981; **Martin et al. (2021).**
  "EqSpike." ✅ https://arxiv.org/abs/2010.07859 — *EqProp on analog and neuromorphic hardware;
  settling is slow in silicon but free in physics. EqProp's real value proposition.*
- **[R21] Hou et al. (2019)** LUCIR (cosine classifier); **Wu et al. (2019)** BiC; **Zhao et al.
  (2020)** Weight Aligning; **Rebuffi et al. (2017)** iCaRL. ⚠ — *The Class-IL bias-correction
  family; each targets pathology 1 (calibration) specifically. iCaRL is the origin of the NCM probe.*
- **[R22] Maltoni & Lomonaco (2019).** "Continuous learning in single-incremental-task scenarios."
  *Neural Networks* 116:56–73. ⚠ — *The MT/SIT/MIT stream-carving taxonomy, orthogonal to [R2]'s
  scenarios.*
- **[R23] Pinchetti et al. (2025).** "Benchmarking predictive coding networks — made simple." *ICLR.*
  ✅ https://arxiv.org/abs/2407.01163 · library https://github.com/liukidar/pcx — *PCX plus a
  standardised benchmark suite; PC matches backprop at VGG-7 scale but falls behind on 9-layer
  convnets and ResNets. The fair-comparison and scaling authority.*
- **[R24] Moreno-Torres et al. (2012).** "A unifying view on dataset shift in classification."
  *Pattern Recognition.* ⚠ — *The covariate / prior-probability / concept shift taxonomy.*
- **[R25] Lopez-Paz & Ranzato (2017).** "Gradient episodic memory for continual learning." *NeurIPS*
  30:6470–6479. ⚠ https://arxiv.org/abs/1706.08840 — *Gradient-constrained replay; the standard
  source of the formal ACC / backward-transfer / forward-transfer metrics, and of Rotated MNIST.*
- **[R26] Masse, Grant & Freedman (2018)** XdG; **Mallya & Lazebnik (2018)** PackNet; **Serra et al.
  (2018)** HAT. ⚠ — *Parameter-isolation and gating methods: effective in Task-IL with task ID
  available, and demonstrably capacity-limited as tasks accumulate.*
- **[R27] Mirzadeh et al. (2022).** "Wide neural networks forget less catastrophically." *ICML.* ⚠ —
  *Width reduces forgetting, largely via sparser and more orthogonal representations.*
- **[R28] Ramasesh, Dyer & Raghu (2021).** "Anatomy of catastrophic forgetting." *ICLR.* ⚠;
  **Yosinski et al. (2014)** ⚠ https://arxiv.org/abs/1411.1792 — *Forgetting concentrates in later
  layers; early features are general and transfer.*
- **[R29] Aljundi et al. (2019).** Maximally Interfered Retrieval ⚠ https://arxiv.org/abs/1908.04742;
  Task-free continual learning ⚠ https://arxiv.org/abs/1812.03596 — *Interference-based replay
  selection, and continual learning without training-time task boundaries — the target stream regime.*
- **[R30] Ororbia et al. (2018, 2019)** sequential neural coding networks ⚠; **Yoo & Wood (2022)**
  BayesPCN ✅ https://arxiv.org/abs/2205.09930; **Tang, Barron & Bogacz (2023)** temporal predictive
  coding ⚠ — *The closest direct prior work on predictive coding for continual learning; note the
  overlap between Ororbia's activation-sparsity mechanism and advisor point 3.*
- **[R-ADV] Advisor's four-point outline** (personal communication) — *The authoritative scope
  document; everything outside it "can/should wait".*

*A fuller literature review — biological plausibility by brain region, hardware requirements, and
which EBMs have demonstrated CF mitigation on which IL tasks — is `ref/ebm_literature_review.md`.*

---

## 13. Glossary

Terms whose meaning is load-bearing in this file. Anything added later is added here in the same
edit.

**Active set (𝒜)** — the output units the loss may touch on a given step. The scenario decides this,
and it decides which failures can occur. · **Anchor** — anything holding old knowledge in place:
replayed data, a penalty term, frozen weights. PC has none. · **Backward transfer (BWT)** — how much
earlier-task accuracy changed after later training; negative BWT is forgetting. · **Blackout
catastrophe** — what happens to Hopfield-type memories past capacity: nothing recalled, nothing new
stored. · **Calibration** — whether class scores are on a comparable scale, so comparing them is
meaningful. · **CKA** — centred kernel alignment; measures similarity between two sets of hidden
representations. · **Clamping** — holding a layer's values fixed. **Strong clamp** = output pinned
exactly to the target (PC); **weak clamp** = output nudged slightly toward it (EqProp). ·
**Collapse floor** — 100 ÷ n_classes, the score of a network answering with one class every time.
*Not* chance. · **Confound / control** — a second variable differing between conditions; a control
holds it equal. Ours are CTRL-1…CTRL-6 (§7). · **Crossover** — the accuracy at which old-task and
new-task curves cross after a switch. High = both held at once. · **Energy** — the scalar quantity
being minimised; for PC, total squared prediction error. · **Finite-difference estimator** —
approximating a derivative from two nearby states; what EqProp does, hence backprop's behaviour plus
extra noise. · **Fisher information** — how much the output changes when a weight changes; used as
"importance". · **Head / trunk** — the output layer / everything before it. · **Local rule** — a
weight update computable from quantities available at that connection alone. PC and EqProp are local;
backprop is not. · **Lyapunov function** — a quantity that only decreases as a system evolves, used
to prove it settles. The energy is one; the network never computes it. · **NCM** — nearest class
mean; classify by the closest class prototype in hidden space, ignoring the head. Our diagnostic
probe. · **Nudge (β)** — the size of EqProp's push toward the target. Not a learning rate; it is in
the *denominator*. · **PCN** — predictive coding network: value units plus explicit error units in a
hierarchy; energy = sum of squared prediction errors. · **Probe** — a simple classifier trained on a
frozen network's hidden layer, asking "is the information still in there?" · **Prospective
configuration** — the rule on a PCN: settle activities to a target-consistent state first, then one
local weight update. · **Settling / relaxation** — repeatedly nudging *unit values* (not weights)
downhill in energy until they stop moving. The inner loop. · **Task-free / task-agnostic** — no
boundaries during training / no context labels needed at any point. · **Target alignment** — [R1]'s
interference measure: how closely the direction the output actually moves matches the direction it
needed to move. · **Trajectory plot** — task-1 accuracy against task-2 accuracy over time; the
diagonal is an even trade, above it is better. Removes time from the picture.

**Abbreviations in the references:** BP backpropagation · PC predictive coding · BPTT
backpropagation through time · MT/SIT/MIT multi-task / single-incremental-task /
multiple-incremental-tasks · NI/NC/NIC new instances / new classes / both · ER, GEM, A-GEM replay
and gradient-projection methods · DGR, BI-R deep generative replay, brain-inspired replay ·
LwF, FROMP functional-regularisation methods · XdG context-dependent gating · iCaRL incremental
classifier and representation learning · MHN modern Hopfield network · VAE variational autoencoder ·
SI synaptic intelligence · SGD stochastic gradient descent.
