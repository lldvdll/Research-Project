# Presentation Plan

Rebuilt 2026-08-10. Supersedes the plan in `archive/chat_logs/presentation_plan.md`.

Reference keys `[Rn]` resolve in `knowledge_base.md` §12. Papers still to collect are listed in
`reference_acquisition.md`.

---

# 0. Standing rules

**The presentation stands alone.** Every claim is supported by an experiment run *for this
presentation*, even where that repeats earlier work. No slide refers to earlier experiments, earlier
failures, or the reproduction attempt. If a comparison against earlier work would help the story, it
is built as a **new** experiment — not reported as old news.

**One question → one experiment → one script → one figure.** A figure may be a grid of panels
(accuracy-vs-step alongside the ACC1-vs-ACC2 trajectory, for instance) where the panels are framings
of the same question. Two questions means two scripts. **If a script's name is too long, the
experiment is too complicated.** Multiple analyses of the same run are separate scripts reading the
same saved arrays — which is why weights and arrays are always saved.

**Scripts are numbered sequentially from 40** for this phase. Name: `NN_question_key-detail.py`.

**Story order beats priority order.** The deck runs in the order the argument runs. A low-priority
slide sitting mid-deck is fine and is framed as an **open question**, so the trajectory is visible
and can be commented on. Slides without results yet are **placeholders**, and that is fine — we
report what is done and pick up the rest the following week.

**Prior work informs design only.** It says what to ask and how. It supports nothing.

**The advisor's four points are guidance, not scope.** Ad hoc, in response to results, already
shifted once. Taken seriously; not a specification.

---

# 1. Target

> **Does an energy-based learning rule reduce catastrophic forgetting compared with
> backpropagation — and how much of the forgetting was available to be reduced in the first place?**

The second clause makes the first answerable. A learning rule decides how credit is shared among
hidden units; it does not decide whether output units compete. Forgetting splits into a part a rule
could touch and a part it cannot, and the size of the first — **the budget** — is measurable.

**Which EBM.** The model is **prospective configuration (PC)**. **EqProp is the control** isolating
whether *full relaxation* is the active ingredient — strong clamp against weak clamp [R3]. That is a
pair by design, and it is the answer when asked to pick one.

**A negative result is a result.** Readings are pre-committed per script.

**Degree framing.** Computational neuroscience and cognition, not only ML. Two slides carry it —
energy-based learning as a hypothesis about cortex, and why humans do not forget this way — and the
mechanism work reports back to them through the metabolic reading of [R31].

---

# 2. Protocol

Defined **once**, in one object, imported by every script. A script varies one line and says which.

| | Value |
|---|---|
| Dataset | MNIST, 14×14 (196 inputs), scaled [0,1] |
| Split | **2 tasks × 5 classes**, class assignment drawn per seed |
| Scenarios | **Class-IL** — 10 outputs, all active · **Domain-IL** — 5 outputs shared. Both run, both reported; where they differ is analysis material |
| Architecture | 196 → H → out, tanh on the way out, biases on |
| Depth | **1 hidden layer throughout.** Depth is a separate later phase, not an axis of the main comparison |
| Output structure | linear + squared error, one-hot 1/0 — confirmed by script 41 before anything depends on it |
| Optimiser | plain SGD, batch 32 |
| Learning rate | grid-searched **per rule**, matched on **steps-to-threshold**, not on the LR value |
| Task 1 and Task 2 | **both trained to a fixed accuracy threshold**, not a fixed iteration budget, so faster rules do not get extra opportunity to forget |
| Cap | max iterations per task; **how often the cap is hit is reported** |
| Seeds | ≥5. Mean ± **standard error of the mean**, stated as such |
| Evaluation | held-out test split. Raw pre-argmax outputs logged alongside accuracy |
| Saving | **every script writes `.npz` beside its figure, including weight snapshots** at init, task-1 end and task-2 end |

**Domain-IL pairing** is arbitrary, fixed at random per seed, stated on the setup slide.
**Hidden width H** is set by script 40 on capacity grounds and by nothing else.

## 2.1 The metric grid

No single headline. Each answers a different question; reported for both scenarios.

| Metric | Answers |
|---|---|
| Per-task accuracy vs step | the readable picture |
| **Trajectory ACC1 vs ACC2** | shape of the trade-off, time removed — immune to a rule simply being slower |
| Crossover height | were both tasks held at once |
| Task-1 accuracy when task 2 hits threshold | retention at a matched standard |
| Final retention | the endpoint. **May be 0 for every rule in Class-IL** — itself a finding, and the Class-IL/Domain-IL contrast is where it pays |
| **Target alignment** [R1 Fig 3b] | does learning move the output toward the target |
| **Inefficiency** [R31] | is the learning path direct or wandering |

### Inefficiency — defined from [R31], not invented here

Li & van Rossum define metabolic cost as an **L1 path length**,
`M = Σ_i Σ_t |w_i(t) − w_i(t−1)|`, and **inefficiency as actual ÷ minimal**, where minimal is the
straight-line cost of the net displacement. Higher is worse.

Because numerator and denominator are both sums over synapses, **per-synapse inefficiency is
well-defined**: `Σ_t |Δw_i| ÷ |w_i(T) − w_i(0)|`. So the distribution comes for free, and that
distribution is the interesting object — see scripts 47–49.

Two readings the project gets from this. It is a **direct test of [R1] Supp. Fig 7's "less erratic
updates" claim**, using a measure they did not use. And it is a **metabolic** argument: a rule that
reaches the same place having spent less is a better hypothesis about a brain under evolutionary
pressure — which is [R31]'s own framing, and ties the mechanism work back to slide 6.

---

# 3. Slide list — story order

Priority is an attribute, not the running order. **H** must-have · **M** wanted · **O** open question,
framed to show trajectory and invite comment.

Claim type: **L** literature, cited · **S** structural, derived · **R** result, needs a figure.

| # | Slide | Pri | Claim | Figure from |
|---|---|---|---|---|
| 1 | Title and question | H | — | — |
| 2 | Catastrophic forgetting: the problem | H | R + L | **42** |
| 3 | Why brains do not forget like this | H | L | schematic |
| 4 | Continual learning scenarios, and our choice | H | L + S | schematic |
| 5 | What a learning rule can and cannot change | H | S | schematic |
| 6 | Energy-based learning as a hypothesis about cortex | H | L | schematic |
| 7 | What actually changes: backprop, PC, EqProp | H | S + L | diagram |
| 8 | The interference claim: strong clamp vs weak clamp | H | L | redrawn from [R1] |
| 9 | How big a network does this problem need? | M | R | **40** |
| 10 | Can every rule learn the same problem? | H | R | **41** |
| 11 | Setup, protocol and controls | H | S | table |
| 12 | Measuring forgetting | H | S | worked example |
| 13 | Do the energy-based rules forget less? | H | R | **43** |
| 14 | Class-IL against Domain-IL: where they differ | M | R | **44** |
| 15 | Where does forgetting live, and how much room was there? | H | R | **45** |
| 16 | Target alignment: testing their mechanism | M | R + L | **46** |
| 17 | Which weights move when a task is learned? | M | R | **47** |
| 18 | Which weights move during continual learning? | M | R | **48** |
| 19 | Is the learning path efficient? | M | R + L | **49** |
| 20 | Open question: does depth change the answer? | O | R + L | **50** |
| 21 | Open question: is the difference output structure, not the rule? | O | R | **51** |
| 22 | Open question: can prediction error gate plasticity? | O | S | — |
| 23 | Other energy-based models | O | L | — |
| 24 | Limitations and what is not shown | H | S | — |
| 25 | Next steps and timeline | H | — | — |

**Story check.** 1–2 pose the problem. 3 says brains solve it and names three mechanisms — local
plasticity, replay, consolidation — so the talk's question becomes *what can the first do alone*.
4–5 fix the setting and state the split that makes the target answerable. 6–8 introduce the
candidate and its claim. 9–12 establish that the test is fair and say how it is measured. 13–15 are
the result and its decomposition. 16–19 are mechanism, ending on the metabolic reading that returns
to slide 6. 20–23 are the open questions, in the order they arise. 24–25 close.

Slides 17–19 are the weight-displacement thread and are treated as a **line of research**, not three
one-off figures: which weights move when a task is learned, which move when a second task arrives,
and whether the path taken was efficient.

---

# 4. Scripts

One question, one script, one figure. Numbered from 40. Each writes `.npz` beside its figure,
including weight snapshots.

| # | Name | Question (one sentence) | Slide |
|---|---|---|---|
| 40 | `40_accuracy_vs_hidden_width_joint` | Under joint training, how does accuracy vary with hidden width, and what width does this problem actually need? | 9 |
| 41 | `41_can_each_rule_learn_shared_structure` | Under one shared output structure, can each rule learn the joint problem, and at what learning rate? | 10 |
| 42 | `42_does_backprop_forget_2x5` | Does backprop forget in each scenario, and does replay recover it? | 2 |
| 43 | `43_do_ebm_rules_forget_less` | Under the fixed protocol, do PC and EqProp forget less than backprop in each scenario? | 13 |
| 44 | `44_scenario_contrast` | Where do the metrics disagree between Class-IL and Domain-IL, and which rule does that favour? *(re-analysis of 43's arrays)* | 14 |
| 45 | `45_where_does_forgetting_live` | How much of the forgetting is output-layer and how much is hidden-layer? | 15 |
| 46 | `46_target_alignment_per_rule` | What is each rule's target alignment in each scenario, and does it track retention? | 16 |
| 47 | `47_which_weights_move_learning_a_task` | When a single task is learned, how is weight movement distributed across weights? | 17 |
| 48 | `48_which_weights_move_during_il` | During task 2, does movement concentrate on the weights task 1 depended on? | 18 |
| 49 | `49_is_the_learning_path_efficient` | Is each rule's weight path more direct than backprop's, per [R31]'s inefficiency measure? | 19 |
| 50 | `50_does_depth_change_the_answer` | As hidden layers are added, how do retention and the mechanism metrics trend? | 20 |
| 51 | `51_output_structure_confound` | How much of any rule difference is output structure rather than credit assignment? | 21 |

**Order of work:** 40 → 41 → 42 → 43 → 45 → 44 → 46 → 47 → 48 → 49 → 50 → 51.

**Depth (50) is a separate phase**, deliberately after the mechanism work. With the metrics
understood and the tools built, it runs as a **sweep over `n_layers` reporting trends**, not as an
extra axis on every earlier plot — adding depth as an axis would make every figure unreadable.
Frozen-layer variants are available there too, and that phase can be as rich as it earns.

## 4.1 Pre-committed readings

**41** — if all rules land within a few points under linear + squared error with a 1/0 target, that
becomes the standard and any later difference is attributable to the rule. If one cannot learn under
it, the standard changes before it contaminates anything downstream.

**43** — if PC separates from backprop on trajectory and crossover while EqProp does not, the active
ingredient is full relaxation. If nothing separates in Class-IL but something does in Domain-IL, the
scenario governs whether the mechanism can express itself. If nothing separates anywhere, 45 must
show whether there was room to.

**45** — a large hidden-layer component means a rule had room, and 43's outcome is about the rule. A
near-zero component means no rule could have helped, 43's null is explained rather than excused, and
that is what regime-dependence looks like from the inside.

**48/49** — if PC's movement concentrates away from task-1-important weights, or its path is more
direct at equal endpoint, the locality mechanism has support and it carries a metabolic reading. If
movement overlaps equally and paths are equally wandering, the mechanism is not what distinguishes
the rules here.

**50** — [R1] claims the advantage grows with depth. If it trends that way here, single-layer results
are a lower bound. If it does not, that is a substantive disagreement with a published claim.

---

# 5. Slide notes — the ones that need them now

**3 Why brains do not forget like this.** Sequential learning without collapse. **Three mechanisms
support memory in brains:** local error-driven plasticity; replay [R10]; consolidation across
timescales [R14], with metaplasticity [R13] and spine persistence [R4]. This talk asks **what the
first can do on its own** — which is exactly why replay is the positive control rather than a
competitor, and why a null result would be informative rather than embarrassing.

**5 What a learning rule can and cannot change.** *The conceptual core.* A rule shares blame among
hidden units; it does not stop output units competing. Hence the split, hence the budget, with slide
15 supplying the size. **Blocked:** the standard suppression derivation is for softmax cross-entropy
and we use squared error, where the push has a fixed point at zero. Must be restated for MSE before
this slide asserts it.

**6 Energy-based learning as a hypothesis about cortex.** Predictive coding as a theory of cortical
function — continual prediction, prediction-error signalling, mapping onto canonical microcircuitry.
Locality: each synapse updates from the two neurons it connects; no separate backward pass, no
weight transport. Honest counterweight [R3]: approximately symmetric weights and signed error
signals are still required, so this is a *different paradigm*, not simply a more plausible one. The
energy is never computed by the network — it is a Lyapunov function used by us.

**12 Measuring forgetting.** The metric grid, with one worked example. Traps stated: the flat line at
100/n_classes is the collapse floor, not chance; accuracy is a threshold readout, so raw outputs are
logged; final new-task accuracy is always reported beside retention.

---

# 6. Open items

1. **Slide 5's MSE derivation** — restate output suppression for squared error. Blocks the
   conceptual core.
2. **References** — `reference_acquisition.md`. Slides 3 and 6 are high priority and currently have
   **no primary sources in the project**.
3. **Code** — untrusted and undocumented (`knowledge_base.md` §6.7). Verify and refactor before
   script 40.
4. **Report length** — 8,000 words as given; archived documents say ~10,000. Confirm.
5. **`ref/cl_course/`** — eight lecture PDFs. Worth checking `02_forgetting` and `04_evaluation`
   against the metric grid before it is fixed.
