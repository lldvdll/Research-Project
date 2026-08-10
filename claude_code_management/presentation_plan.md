# Presentation Plan

Rebuilt 2026-08-10. Supersedes the plan in `archive/chat_logs/presentation_plan.md`.

Reference keys `[Rn]` resolve in `knowledge_base.md` §12.

---

# 0. Standing rules for this plan

**The presentation stands alone.** Every claim on a slide is supported by an experiment run *for
this presentation*, even where that means replicating work that was done before. No slide refers to
earlier experiments, earlier failures, or the reproduction attempt. If a comparison against earlier
work would genuinely help the story, it is built as a **new** experiment — not reported as old news.

**Prior work informs design only** (`knowledge_base.md` §6.6, and the evidence standard in
`CLAUDE.md`). It tells us what to ask and how to ask it. It supports nothing.

**The advisor's four points are guidance, not scope.** They arrived ad hoc, in response to results,
and have already shifted once. Taken seriously; not treated as a specification.

---

# 1. Target

> **Does an energy-based learning rule reduce catastrophic forgetting compared with
> backpropagation — and how much of the forgetting was available to be reduced in the first place?**

The second clause makes the first answerable. A learning rule decides how credit is shared among
hidden units; it does not decide whether output units compete. Forgetting therefore splits into a
part a rule could touch and a part it cannot, and the size of the first part — **the budget** — is
measurable. Measure it, then ask whether an energy-based rule claims any of it.

**Spine:** forgetting splits in two → a rule can only touch one part → make the rules comparable →
compare → measure the part that was available → explain the difference → test whether depth changes
the answer.

**Which EBM.** The model is **prospective configuration (PC)**. **EqProp is the control** that
isolates whether *full relaxation* is the active ingredient — strong clamp against weak clamp [R3].
If PC separates from backprop and EqProp does not, the effect is not energy-based-ness in general.
That is a pair by design, and it is the one-line answer when asked to pick one.

**A negative result is a result.** Readings are pre-committed in §3.1 so the interpretation is fixed
before the data arrives.

**Degree framing.** This is computational neuroscience and cognition, not only machine learning. Two
high-priority slides carry that: why energy-based learning is a hypothesis about cortex, and why
humans do not forget this way. The rest of the talk refers back to them.

---

# 2. Protocol — fixed, identical across every experiment

Defined **once**, in one object, imported by every script. A script varies one line of it and says
which line. This is the single discipline that makes results comparable across experiments.

| | Value |
|---|---|
| Dataset | MNIST, 14×14 (196 inputs), scaled [0,1] |
| Split | **2 tasks × 5 classes**, class assignment drawn per seed |
| Scenarios | **Class-IL** — 10 outputs, all active · **Domain-IL** — 5 outputs shared by both tasks. **Both are run and both are reported**; where the metrics differ between them is analysis material, not a problem |
| Architecture | 196 → H → out, tanh on the way out, biases on |
| Depth | 1 hidden layer for X1–X5. Depth is X6's variable: 1/2/3/4 layers, plus frozen-layer variants |
| Output structure | linear + squared error, one-hot 1/0, confirmed by X1 before anything depends on it |
| Optimiser | plain SGD, batch 32 |
| Learning rate | grid-searched **per rule** in X1, then fixed. Matched on **steps-to-threshold**, not on the LR value — the gradients are in different natural units |
| Task 1 | trained to a fixed accuracy threshold |
| Task 2 | **also trained to a fixed accuracy threshold**, not a fixed iteration budget — otherwise faster-learning rules get more opportunity to forget |
| Cap | max iterations per task; **how often the cap is hit is reported**, since it is itself a result |
| Seeds | ≥5. Report mean ± **standard error of the mean** (the 68% band), stated explicitly |
| Evaluation | held-out test split, always. Raw pre-argmax outputs logged alongside accuracy |
| Output | **every script writes `.npz` beside its figure.** Non-negotiable |

**Domain-IL pairing.** With 5 shared outputs, task-1 class *k* and task-2 class *k* share a unit. The
pairing is arbitrary, fixed at random per seed, and stated on the setup slide.

**Hidden width H** is set by X1 on capacity grounds (accuracy against width under joint training) and
by nothing else.

## 2.1 The metric grid

No single headline number. Each metric answers a different question and the set is reported for both
scenarios; where they disagree is the interesting part.

| Metric | What it answers | Note |
|---|---|---|
| Per-task accuracy vs step | the readable picture | the default figure |
| **Trajectory: ACC1 vs ACC2** | the shape of the trade-off, time removed | immune to a method simply being slower |
| Crossover height | were both tasks held at once | one number, high = held both |
| Task-1 accuracy when task 2 hits threshold | retention at a matched standard | pairs with accuracy-stopping |
| Final retention | the endpoint | **may be 0 in Class-IL for every rule** — that is itself a finding, and the Class-IL/Domain-IL contrast is where it pays off |
| **Target alignment** [R1 Fig 3b] | does learning move the output toward the target | the literature's own mechanism metric |
| **Weight-update path efficiency** | is the path direct or erratic | ‖net displacement‖ ÷ total path length; tests [R1] Supp. Fig 7's "less erratic updates" claim, and carries a metabolic-cost reading |

**Path efficiency, computed how.** Network-level ratio as the reportable number; per-weight path
length against net displacement as a **scatter**, not a ratio, because weights that barely move have
an unstable denominator and would dominate any mean; and binned by task-1 importance rank, which
merges it with "where does the weight movement go".

---

# 3. Experiments

Linear. Each depends on the one before it.

| ID | Question | Feeds slides |
|---|---|---|
| **X0** | *Not an experiment.* Correct and verify the code, especially deep architectures; simplify to reduce failure points without breaking backward compatibility; implement the metric grid once, in one module | all |
| **X1** | Can each rule learn under the shared output structure, at what learning rate, and how does accuracy vary with hidden width? | 9, 18 |
| **X2** | Do PC and EqProp forget less than backprop, in each scenario, under the fixed protocol? | 2, 10, 17 |
| **X3** | How much of the forgetting is in the output layer and how much in the hidden layer? | 5, 12 |
| **X4** | What is the target alignment of each rule in each scenario, and does it track retention? | 13 |
| **X5** | Is the weight-update path more direct for PC than for backprop, and where does the movement go? | 14, 22 |
| **X6** | Does depth change the answer — at 1/2/3/4 layers, and with layers frozen? | 15 |

**Order:** X0 → X1 → X2 → X3/X4/X5 → X6. Depth comes last deliberately: the tools and the
interpretation are established on the simple case first, then applied to [R1]'s depth claim with the
machinery already built and explained.

**Friday, realistically: X0, X1, X2.** That is the minimum standalone talk — the setup justified,
the rules shown to be comparable, and the comparison itself. It is ambitious in four days and the
code check is the risk.

## 3.1 Pre-committed readings

**X1** — if all rules learn to within a few points under linear + squared error with a 1/0 target,
that becomes the standard and any later difference is attributable to the rule. If one cannot learn
under it, the standard changes, and we find out before it contaminates anything.

**X2** — if PC separates from backprop on the trajectory and crossover while EqProp does not, the
active ingredient is full relaxation. If nothing separates in Class-IL but something does in
Domain-IL, the scenario determines whether the mechanism can express itself. If nothing separates
anywhere, X3 must show whether there was any room to separate — and if there was not, that is the
result.

**X3** — a large hidden-layer component means a rule had room and X2's outcome is about the rule. A
near-zero component means no rule could have helped, X2's null is explained rather than excused, and
this is what the literature's regime-dependence looks like from the inside.

**X6** — [R1] claims the advantage grows with depth. If it does here, depth 1 was an unfavourable
regime and our earlier result is a lower bound. If it does not, that is a substantive disagreement
with a published claim and needs stating carefully.

---

# 4. Slide list — three tiers

**High (10)** is the twenty minutes. **Medium (6)** is depth for questions and the report spine.
**Low (6)** is backup and permitted one-slide excursions.

Claim types: **L** literature, cited · **S** structural, a derived argument · **O** ours, needs an
experiment.

| # | Title | Tier | Claim | Needs |
|---|---|---|---|---|
| 1 | Title and question | H | — | — |
| 2 | Catastrophic forgetting: the problem | H | O + L | X2 |
| 3 | Why brains do not forget like this | H | L | — |
| 4 | Continual learning scenarios, and our choice | H | L + S | — |
| 5 | What a learning rule can and cannot change | H | S | — |
| 6 | Energy-based learning as a hypothesis about cortex | H | L | — |
| 7 | What actually changes: backprop, PC, EqProp | H | S + L | — |
| 8 | The interference claim: strong clamp vs weak clamp | H | L | — |
| 9 | Setup, protocol and controls | H | O | X1 |
| 10 | Results: four rules, two scenarios | H | O | X2 |
| 11 | Measuring forgetting: the metric grid | M | S | — |
| 12 | Where forgetting lives, and how much room there was | M | O | X3 |
| 13 | Target alignment | M | O + L | X4 |
| 14 | Weight-update path efficiency | M | O + L | X5 |
| 15 | Depth | M | O + L | X6 |
| 16 | Deviations from the literature's setup | M | S + L | — |
| 17 | Scenario comparison in detail | L | O | X2 |
| 18 | Output structure as a confound | L | O | X1 |
| 19 | Other energy-based models | L | L | — |
| 20 | Limitations and what is not shown | L | S | — |
| 21 | Where the weight movement goes, by importance | L | O | X5 |
| 22 | Next steps and remaining timeline | L | — | — |

**Flow.** 1–3 set the problem and the biological stake. 4 fixes the setting. 5 states the target's
structure. 6–8 introduce the method under test and the claim. 9 establishes the test is fair. 10 is
the result. Medium tier deepens the mechanism, then depth. Linear, with 16 and 19 as the permitted
excursions.

**Every O slide has an experiment.** No empirical claim appears without one.

---

# 5. Slide notes — high priority

**1 Title and question.** Title, name, supervisor, date. One line stating the two-clause question.
No agenda slide.

**2 Catastrophic forgetting.** Train A then B; accuracy on A collapses. Active overwriting, not
gradual decay — the collapse floor tell. *Needs X2's backprop panel.*

**3 Why brains do not forget like this.** [neuroscience, human side] Sequential learning without
collapse. Complementary learning systems: fast hippocampal, slow cortical, replay moving information
between them [R10]. Consolidation over multiple timescales inside a synapse [R14]; metaplasticity
[R13]; dendritic-spine persistence [R4]. Frames the whole talk: three mechanisms support memory in
brains, and the rest of the talk asks what a *learning rule alone* can do without the other two.

**4 Scenarios, and our choice.** Three scenarios [R2], distinguished by output layer and what is
known at test. Diagram: three output-layer schematics. Ours: one partition, 2 tasks × 5 classes,
both scenarios, so only the output layer differs. Note that Domain-IL at 2×5 with five shared
outputs is structurally [R1] Fig 4d.

**5 What a learning rule can and cannot change.** *The conceptual core.* A rule shares blame among
hidden units; it does not stop output units competing. Hence the split, hence the budget. Structural
argument, with slide 12 supplying the size. **Caveat to resolve first:** the standard suppression
derivation is for softmax cross-entropy and we use squared error, where the push has a fixed point
at zero. Restate for MSE before asserting it.

**6 Energy-based learning as a hypothesis about cortex.** [neuroscience, model side] Predictive
coding as a theory of cortical function — continual prediction of input, neurons signalling
prediction error, mapping onto canonical microcircuitry. Locality: each synapse updates from the two
neurons it connects, no separate backward pass, no weight transport. The honest counterweight [R3]:
PC still needs approximately symmetric weights and signed error signals, so it is not simply "more
plausible" — it is a different paradigm. Energy is never computed by the network; it is a Lyapunov
function used by us.

**7 What actually changes.** Backprop: hidden activities fixed by weights, error propagated back.
Energy-based: activities are free variables that settle first, weights move afterwards, locally. PC
clamps input and target and relaxes. EqProp settles free, settles nudged, updates from the
difference. Honest note: all three coincide in the infinitesimal inference limit [R7] — full
relaxation is what makes them different algorithms. Three-column diagram keyed to our variables.

**8 The interference claim.** [R1]: backprop updates each layer as though the others were fixed, so
updates interfere; settling first finds a mutually consistent configuration. Their evidence: target
alignment and its independence from learning rate, its degradation with depth for backprop but not
PC, the continual-learning result. Strong clamp (PC) vs weak clamp (EqProp) [R3] — and why that
makes EqProp the control rather than a second competitor. Costs: expensive relaxation, symmetric
weights, regime-dependent advantage.

**9 Setup, protocol and controls.** The protocol table. One thing varies at a time. Controls on
every forgetting run: backprop negative, replay positive — if replay recovers the task, the problem
is demonstrably solvable. *Needs X1* for per-rule learning rates, the width choice, and the evidence
that all rules learn under one output structure.

**10 Results: four rules, two scenarios.** The metric grid, both scenarios side by side. Final
accuracy on the new task reported alongside retention so "forgot less" cannot mean "learnt less".
Where Class-IL and Domain-IL disagree is discussed, not hidden. *Needs X2.*

---

# 6. Open items

1. **Synaptic caching paper** — the source for path efficiency is not in `ref/`. Needed before slide
   14 says anything about it.
2. **Slide 5's MSE derivation** — restate output suppression for squared error before asserting it.
3. **Report length** — 8,000 words as given; archived documents say ~10,000. Confirm.
4. **X0 is the critical risk.** Depth support lives in the least-verified part of the codebase.
5. **Advisor point 4** (gating plasticity by per-node prediction error) is not tested. It is the
   natural next step once X3 shows whether there is a budget to claim.
