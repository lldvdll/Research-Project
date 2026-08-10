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

**One deviation at a time.** Every script states its **single** deviation from the protocol in §2, in
its docstring and on its slide — "the protocol, with the loss masked", "the protocol, with
`n_layers` swept". Nothing varies two things at once. This is what keeps each experiment's question
legible without re-explaining the whole setting, and it is what makes the audience able to follow a
sequence of figures. **The protocol exists to give a fair comparison of backprop, replay, EqProp and
PC. That is its job, and deviations are justified against it.**

**Scripts are numbered sequentially from 40** for this phase. Name: `NN_question_key-detail.py`.

**Script order tracks slide order.** If they diverge, the project is not linear and something is
wrong. One inversion is accepted and stated: **script 40 (width) runs first because it fixes H, but
is presented at slide 11**, since opening the talk with a capacity sweep would be a bad talk.
Everything else is monotonic.

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
| Output structure | linear + squared error, one-hot 1/0 — confirmed by script 43 before anything depends on it |
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
pressure — which is [R31]'s own framing, and ties the mechanism work back to slide 8.

---

# 3. Slide list — story order

Priority is an attribute, not the running order. **H** must-have · **M** wanted · **O** open question,
framed to show trajectory and invite comment.

Claim type: **L** literature, cited · **S** structural, derived · **R** result, needs a figure.

| # | Slide | Pri | Claim | Figure from |
|---|---|---|---|---|
| 1 | Title and question | H | — | — |
| 2 | Catastrophic forgetting: here it is | H | R | **41** |
| 3 | Why brains do not forget like this | H | L | schematic |
| 4 | Continual learning scenarios, and our choice | H | L + S | schematic |
| 5 | **The output layer: where the downward push comes from** | H | S | worked example |
| 6 | What a learning rule can and cannot change | H | S | schematic |
| 7 | How much room is there? | H | R | **42** |
| 8 | Energy-based learning as a hypothesis about cortex | H | L | schematic |
| 9 | What actually changes: backprop, PC, EqProp | H | S + L | diagram |
| 10 | The interference claim: strong clamp vs weak clamp | H | L | redrawn from [R1] |
| 11 | Setup, protocol and controls | H | S + R | table + **40** |
| 12 | Can every rule learn the same problem? | H | R | **43** |
| 13 | Measuring forgetting | H | S | worked example |
| 14 | Do the energy-based rules forget less? | H | R | **44** |
| 15 | Class-IL against Domain-IL: where they differ | M | R | **45** |
| 16 | Target alignment: testing their mechanism | M | R + L | **46** |
| 17 | Which weights move when a task is learned? | M | R | **47** |
| 18 | Which weights move during continual learning? | M | R | **48** |
| 19 | Is the learning path efficient? | M | R + L | **49** |
| 20 | Open question: does depth change the answer? | O | R + L | **50** |
| 21 | Open question: which layer does forgetting live in? | O | R + L | **51** |
| 22 | Open question: is it the output structure, not the rule? | O | R | **52** |
| 23 | Open question: can prediction error gate plasticity? | O | S | — |
| 24 | Other energy-based models | O | L | — |
| 25 | Limitations and what is not shown | H | S | — |
| 26 | Next steps and timeline | H | — | — |

**Story check.** 1–2 pose the problem with a plain example. 3 says brains solve it and names three
mechanisms — local plasticity, replay, consolidation — so the talk's question becomes *what can the
first do alone*, which is also why replay is the positive control rather than a competitor. 4 fixes
the setting. **5 explains the output mechanics, 6 draws the consequence — a rule cannot touch that
part — and 7 measures how much is left over.** Claim and evidence adjacent. 8–10 introduce the
candidate and its claim. 11–13 establish the test is fair and say how it is measured. 14–15 are the
result. 16–19 are mechanism, ending on the metabolic reading that returns to slide 8. 20–24 are open
questions in the order they arise. 25–26 close.

**Slide 21 sits after depth deliberately.** At one hidden layer, "which layer forgets" has only two
answers and is close to vacuous; it needs layers before it can say anything. Slide 7 supplies the
budget that slide 14 needs; slide 21 supplies the layer-resolved version once there are layers.

Slides 17–19 are the weight-displacement thread, treated as a **line of research** rather than three
one-off figures: which weights move when a task is learned, which move when a second arrives, and
whether the path taken was efficient.

---

# 4. Scripts

One question, one script, one figure. Numbered from 40. Each writes `.npz` beside its figure,
including weight snapshots.

Each row states its **single deviation** from the protocol in §2.

| # | Name | Question (one sentence) | Deviation | Slide |
|---|---|---|---|---|
| 40 | `40_accuracy_vs_hidden_width_joint` | How does accuracy vary with hidden width, and what width does this problem need? | joint training, no task sequence; width swept | 11 |
| 41 | `41_does_backprop_forget` | Does backprop forget task 1 when task 2 arrives? | backprop only, no controls — a motivating example, not a comparison | 2 |
| 42 | `42_how_much_room_is_there` | How much of the forgetting survives when the output push is removed, and when the hidden layer is frozen? | loss masked / W1 frozen, backprop only | 7 |
| 43 | `43_can_each_rule_learn` | Under one shared output structure, can each rule learn the joint problem, and at what learning rate? | joint training, no task sequence | 12 |
| 44 | `44_do_ebm_rules_forget_less` | Do PC and EqProp forget less than backprop in each scenario? | **none — this is the protocol** | 14 |
| 45 | `45_scenario_contrast` | Where do the metrics disagree between Class-IL and Domain-IL? | none; re-analysis of 44's saved arrays | 15 |
| 46 | `46_target_alignment_per_rule` | What is each rule's target alignment, and does it track retention? | none; extra measurement during the protocol run | 16 |
| 47 | `47_which_weights_move_learning_a_task` | When one task is learned, how is weight movement distributed across weights? | single task only | 17 |
| 48 | `48_which_weights_move_during_il` | During task 2, does movement concentrate on the weights task 1 depended on? | none; extra measurement | 18 |
| 49 | `49_is_the_learning_path_efficient` | Is each rule's weight path more direct, by [R31]'s inefficiency measure? | none; re-analysis of saved weight snapshots | 19 |
| 50 | `50_does_depth_change_the_answer` | As hidden layers are added, how do retention and the mechanism metrics trend? | `n_layers` swept | 20 |
| 51 | `51_which_layer_does_forgetting_live_in` | With several hidden layers, which one carries the forgetting? | `n_layers` fixed > 1; layers frozen one at a time | 21 |
| 52 | `52_output_structure_confound` | How much of any rule difference is output structure rather than credit assignment? | output structure swept | 22 |

**Execution order:** 40 → 41 → 42 → 43 → 44 → 45 → 46 → 47 → 48 → 49 → 50 → 51 → 52.

Slide order matches, with the single stated inversion: **40 runs first but is presented at slide 11**,
because it fixes H for everything and a talk cannot open on a capacity sweep.

**Depth (50, 51) is a separate phase**, deliberately after the mechanism work. With the metrics
understood and the tools built, it runs as a **sweep over `n_layers` reporting trends**, not as an
extra axis on every earlier plot — adding depth as an axis would make every figure unreadable.
Frozen-layer variants live here too, and 51 is where "which layer forgets" finally has enough layers
to be worth asking.

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

**5 The output layer: where the downward push comes from.** *Written because the output mechanics
are the hardest part of this project to hold in your head, and everything downstream leans on them.*

The label says two things at once. Training on a 2, the target is 1 in position 2 and **0 in the
other nine**. "Unit 2 should be high" is what we intend; "unit 0 should be low, unit 1 should be
low…" is what we do not think about. Every image of a 2 is therefore also a training example that
pushes unit 0 down. Do that a few hundred times and unit 0 has been trained into silence — so when a
real 0 arrives it no longer wins, **not because the hidden layer forgot what a 0 looks like, but
because the output unit was taught to keep quiet.**

The four output structures differ only in *how hard and how far* they push:

| structure | push on an absent class | where it stops |
|---|---|---|
| softmax + cross-entropy | driven down relative to the winner | **nowhere** — unbounded |
| linear + squared error, target 0 | driven toward 0 | at 0 — the weight ends orthogonal to current features |
| linear + squared error, target −1 | driven toward −1 | at −1, a whole unit further |
| hinge, target −1 | driven toward −1 at constant force | at −1, and the push does not weaken on the way |

**The push is not caused by softmax.** Softmax makes it unbounded; the *label* is the source. That
is why swapping the output structure changes the severity but never removes it — only masking the
absent classes does. **This slide earns the protocol's choice** (linear + squared error, 1/0): it is
the mildest structure all three rules can take unaltered, which slide 12 then verifies.

**Work needed:** the derivation for squared error, in full, before this slide is drawn. The
familiar `∂L/∂z_o = p_o − 1[o=t]` result is for softmax cross-entropy; under MSE it is
`∂L/∂z_o = z_o`, with a fixed point at zero. The table above depends on getting that right.

**6 What a learning rule can and cannot change.** *The conceptual core, and it follows directly from
5.* That downward push happens at the output layer no matter how blame was shared among hidden
units. A learning rule changes the sharing; it cannot change whether the units compete. So forgetting
splits into a part no rule can touch and a part it might — and slide 7 measures how big the second
part is.

**8 Energy-based learning as a hypothesis about cortex.** Predictive coding as a theory of cortical
function — continual prediction, prediction-error signalling, mapping onto canonical microcircuitry.
Locality: each synapse updates from the two neurons it connects; no separate backward pass, no
weight transport. Honest counterweight [R3]: approximately symmetric weights and signed error
signals are still required, so this is a *different paradigm*, not simply a more plausible one. The
energy is never computed by the network — it is a Lyapunov function used by us.

**13 Measuring forgetting.** The metric grid, with one worked example. Traps stated: the flat line at
100/n_classes is the collapse floor, not chance; accuracy is a threshold readout, so raw outputs are
logged; final new-task accuracy is always reported beside retention.

---

# 6. Open items

1. **Slide 5's MSE derivation** — write out output suppression for squared error in full. Blocks the
   output-mechanics slide, which slide 6 then depends on.
2. **References** — `reference_acquisition.md`. Slides 3 and 8 are high priority and currently have
   **no primary sources in the project**.
3. **Code** — untrusted and undocumented (`knowledge_base.md` §6.7). Verify and refactor before
   script 40.
4. **Report length** — 8,000 words as given; archived documents say ~10,000. Confirm.
5. **`ref/cl_course/`** — eight lecture PDFs. Worth checking `02_forgetting` and `04_evaluation`
   against the metric grid before it is fixed.
