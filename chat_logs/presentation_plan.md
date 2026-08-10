# Presentation Plan

20 minutes. 20 slides planned: 15 mandatory, 5 optional. Optional slides are the report skeleton and Q&A backup, not presented Friday.

Reference keys `[Rn]` resolve in `knowledge_base.md` §12.

## Fixed decisions

- **Split: 5×2 for both scenarios.** Same data partition, same schedule. Only the output layer differs between Class-IL and Domain-IL, so scenario is the single variable.
- **Both scenarios presented**, with identical slide structure, plus a comparison slide.
- **All plots re-run.** No reuse of earlier results.
- **Audience:** peers on the course, mixed computational neuroscience and ML. Technical, but assume no continual-learning background.
- No claim is made about *why* the scenarios differ until the results are in. Slides state the structural difference and report what is observed.

## Timing

Mandatory slides total ~22 min as listed. If over: demote slide 9 to optional (−2 min), then trim slides 3 and 11 by 30 s each.

---

# 1. Slide list

| # | Title | | min |
|---|---|---|---|
| 1 | Title and question | M | 0.3 |
| 2 | Catastrophic forgetting | M | 1.5 |
| 3 | Continual learning scenarios | M | 2.0 |
| 4 | Scenario choice | M | 1.3 |
| 5 | Setup and controls | M | 1.0 |
| 6 | Deviations from the literature | O | — |
| 7 | Measuring forgetting | M | 2.0 |
| 8 | Backprop forgets | M | 1.3 |
| 9 | Where forgetting happens: trunk or head | M | 2.0 |
| 10 | Energy-based learning: what changes | M | 2.0 |
| 11 | Prospective configuration and the interference claim | M | 2.0 |
| 12 | Results: Class-IL | M | 1.5 |
| 13 | Results: Domain-IL | M | 1.3 |
| 14 | Scenario comparison | M | 1.0 |
| 15 | Target alignment | M | 1.5 |
| 16 | Where the weight movement goes | O | — |
| 17 | Does the learning rule change where forgetting happens | O | — |
| 18 | Depth | O | — |
| 19 | Summary: shown and not shown | M | 1.0 |
| 20 | Next steps | M | 1.5 |

---

# 2. Slide detail

## Slide 1 — Title and question [M]

**Contents**
- Title, name, supervisor, date.
- One line: *Does energy-based learning reduce catastrophic forgetting compared with backpropagation, and if so, by what mechanism?*
- No agenda slide.

**Work** — none.

**Experiments** — none.

---

## Slide 2 — Catastrophic forgetting [M]

**Contents**
- Train on task A, then task B. Accuracy on A collapses. Not gradual decay — active overwriting.
- Contrast: humans and animals learn sequentially without this.
- Figure: backprop on 5×2 split MNIST, per-task accuracy over training.
- The question this raises: is the collapse a property of gradient descent in general, or of backpropagation specifically?

**Work**
- Read [R11] McCloskey & Cohen for the original demonstration and its framing.
- Read [R2] introduction for the biological contrast and current framing of the field.
- Optional: [R4] for dendritic-spine persistence as the biological grounding of consolidation.

**Experiments** — **E1** (backprop and replay, both scenarios). Class-IL panel used here.

---

## Slide 3 — Continual learning scenarios [M]

Audience has no CL background. This slide is pedagogical and cannot be rushed.

**Contents**
- Continual learning: data arrives as a sequence of contexts; earlier data is not revisited.
- Three scenarios [R2], distinguished by the output layer and by what is known at test time.
- **Task-IL** — task identity given at test, one head per task. Analogue: playing different instruments; you always know which. Little forgetting for any method, so not diagnostic.
- **Domain-IL** — same output units reused across contexts, no task identity. Analogue: recognising objects under different lighting; driving in different weather.
- **Class-IL** — one output unit per class, all active, no task identity. Analogue: learning to discriminate a growing set of categories.
- Difficulty ordering, with one line on why: methods that work in Task-IL degrade in Domain-IL and collapse in Class-IL [R2].
- Diagram: three output-layer schematics, active units highlighted.

**Work**
- Read [R2] §"Three continual learning scenarios", Table 1, Fig 2, and Tables 2–3 for the difficulty ordering.
- Build the three-scenario schematic. Highest-reuse figure in the project — build it properly.

**Experiments** — none.

---

## Slide 4 — Scenario choice [M]

**Contents**
- We use one data partition — 5 contexts × 2 digits — for both scenarios. Only the output layer changes: 10 units all active (Class-IL) or 2 shared units (Domain-IL). Scenario is therefore the only variable.
- **Class-IL is primary**, for four reasons:
  1. It has the largest dynamic range between methods — parameter-regularisation methods drop to the no-method baseline there, so the scenario discriminates between learning rules [R2 Tables 2–3].
  2. It is the standard benchmark in the ML continual-learning literature.
  3. It matches the common biological case: discriminating a growing set of categories with no context cue supplied.
  4. It exercises both the shared trunk and the output layer, so forgetting can be attributed between them.
- **Domain-IL is also run**, because it is the scenario in which the interference claim was demonstrated [R1 Fig 4d: five output neurons shared by two five-class tasks], and because it exercises the trunk without the output-layer effects.
- **Deviation to state:** [R1] Fig 4d uses 2 tasks × 5 classes, alternating. We use 5 tasks × 2 classes, sequential, in both scenarios. Reason: holding partition and schedule fixed makes the output layer the only difference between our two scenarios; their protocol varies task count and schedule alongside it.

**Work**
- Confirm the [R1] Methods wording for Fig 4d directly (five output neurons, alternating 4 iterations to 84 total) — quote it on the slide.
- Read [R2] Table 1 and Tables 2–3 for the discriminative-range argument; extract the specific numbers showing EWC/SI at baseline in Class-IL.

**Experiments** — none.

---

## Slide 5 — Setup and controls [M]

**Contents**
- Data: MNIST downsampled to 14×14 (196 inputs), scaled [0,1]. 5×2 split.
- Network: 196 → 64 → 10 (Class-IL) or 196 → 64 → 2 (Domain-IL). One hidden layer, tanh, squared error.
- Architecture, loss and activation identical across all four learning rules. Only the rule varies.
- Plain SGD, batch 32, learning rate grid-searched independently per rule [R1 does this], ≥5 seeds, 68% CI.
- Controls on every run: backprop as negative control (should forget), replay as positive control (should recover). If replay recovers the task, the problem is demonstrably solvable.
- One number justifying 14×14: joint-training accuracy vs 28×28.

**Work**
- Read [R1] Methods for the LR grid values, batch size, epoch counts, seed count and CI method.
- Read [R2] Methods for their Split MNIST reference architecture and training protocol.

**Experiments** — **E0** (resolution check: joint accuracy at 28×28 / 14×14 / 8×8).

---

## Slide 6 — Deviations from the literature [O]

**Contents**
- Table: axis / theirs / ours / reason. Rows: dataset, resolution, depth, activation, loss, protocol, optimiser, learning rate, batch size, seeds.
- *We reproduce the protocol, not the hyperparameters. The claim under test is qualitative — does PC forget less than backprop under matched conditions — so this is a conceptual replication.*
- *Depth 1 is the conservative choice: [R1] Fig 3e and 4h claim the advantage grows with depth, so any advantage found at depth 1 is a lower bound.*

**Work**
- Read [R1] Fig 3e and Fig 4h and record exactly what is claimed about depth, so the "lower bound" statement is accurate rather than convenient.
- Read [R3] for the regime caveats (batch size, depth) it raises against the interference claim.

**Experiments** — none.

---

## Slide 7 — Measuring forgetting [M]

**Contents**
- Per-task accuracy on fixed class sets, held-out test split.
- Final retention; forgetting = max accuracy − final accuracy.
- Trajectory plot: accuracy on task *i* vs accuracy on task *j*, time removed from both axes, so a method that is simply slower is not flattered.
- Area above the diagonal: the learning/forgetting trade-off as one number.
- Target alignment [R1 Fig 3b], introduced here and used on slide 15.
- Two measurement traps worth stating: the flat line at 100/n_classes is the collapse floor, not chance; accuracy is a threshold readout, so raw pre-argmax outputs are logged as well.
- Final retention and trade-off efficiency need not give the same ordering. Both are reported.

**Work**
- Read [R1] Fig 3b–d for the target-alignment definition and the claim that it is nearly independent of learning rate.
- Lock the four metric definitions in one module; every later script imports it rather than redefining.
- Decide and record how multi-task (5-context) trajectories are plotted, since the trajectory plot was designed for two tasks. Options: pairwise first-vs-last, or mean-of-seen vs current.

**Experiments** — **E-metrics** (definitions and plotting module; no new run).

---

## Slide 8 — Backprop forgets [M]

**Contents**
- Two panels: Class-IL and Domain-IL. Backprop and replay only.
- Class-IL: per-task accuracies collapse toward the floor; replay recovers.
- Domain-IL: smaller drop; replay recovers.
- Note whether the magnitude ordering matches [R2].

**Work**
- Read [R2] Table 2 to have the published Split MNIST numbers available for comparison.

**Experiments** — **E1**.

---

## Slide 9 — Where forgetting happens: trunk or head [M]

First demotion candidate if timing is tight.

**Contents**
- One hidden layer means exactly two weight matrices: W1 (trunk) and W2 (head).
- Observational test: discard the trained head and classify from the 64 hidden features by nearest class mean. If accuracy recovers, the features survived and the head was the failure.
- Interventional test: during later contexts, freeze W1 / freeze W2 / freeze neither.
- Read off which matrix is responsible, per scenario.

**Work**
- Read [R2] eq. (2) for the active-unit softmax formulation.
- Confirm the output-layer gradient argument in `knowledge_base.md` §4.1 against the loss actually implemented (squared error, not softmax cross-entropy — the argument needs restating for MSE, where the suppression has a fixed point at zero).

**Experiments** — **E2** (nearest-class-mean probe), **E3** (freeze W1 / W2 / neither).

---

## Slide 10 — Energy-based learning: what changes [M]

**Contents**
- Backprop: hidden activities are determined by the weights; error is propagated backward by the chain rule; one update per example.
- Energy-based: hidden activities are free variables that minimise an energy. They settle first; weights change afterwards, locally.
- PC: clamp input and target, relax to equilibrium, update ∝ presynaptic activity × postsynaptic error.
- EqProp: settle free, settle weakly nudged, update ∝ difference between the two equilibria.
- Both are local — no weight transport, no global backward pass.
- Honest note: all three coincide in the infinitesimal inference limit [R7]. Full relaxation is what makes PC and EqProp different algorithms.
- Diagram: three columns (backprop / PC / EqProp) × four rows (hidden activities, credit assignment, weight update, treatment of the target).

**Work**
- Read [R1] Fig 2 and Methods for the PC formulation.
- Read [R5] for the EqProp two-phase formulation.
- Read [R6] for the local Hebbian plasticity result.
- Read [R7] for the infinitesimal-limit unification.
- Read [R3] for strong-clamp (PC) vs weak-clamp (EqProp).
- Build the three-column diagram keyed to our own variable names. Cite [R1] Fig 2 rather than redrawing it.

**Experiments** — none.

---

## Slide 11 — Prospective configuration and the interference claim [M]

**Contents**
- The claim [R1]: backprop updates each layer as though the others were fixed, so layer updates interfere and the output does not move directly toward the target. Settling first finds a mutually consistent configuration of activities, so the subsequent weight change moves the output along the direction of the target.
- Target alignment: cosine between the direction to the target and the direction learning actually moves the output, the latter measured without the target provided [R1 Fig 3b].
- Their evidence: alignment nearly independent of learning rate [Fig 3d]; alignment degrades with depth for backprop but not PC [Fig 3e]; continual-learning advantage [Fig 4d–e].
- Costs, from the commentary [R3]: expensive relaxation phase; approximately symmetric weights required; the advantage is regime-dependent.
- Biological motivation: local plasticity rule; activity settles before weight change, which is observed in cortex rather than being a cost to suppress.
- Figure: target-alignment geometry, redrawn simply with attribution.

**Work**
- Read [R1] Fig 3a–e, Fig 4d–e and Discussion closely; note exactly which claims are supported by which figure so nothing is overstated.
- Read [R3] in full — it is short and it is the source of both the strong/weak clamp framing and the counterweights.
- Note the [R1] Discussion remark on EqProp, which is what justifies including EqProp as a contrast rather than as a second competitor.

**Experiments** — none.

---

## Slide 12 — Results: Class-IL [M]

**Contents**
- Four methods: backprop, replay, PC, EqProp.
- Per-task accuracy curves; trajectory plot; summary table (final retention, area above diagonal, final accuracy on the last context).
- Final accuracy on the newest context is reported alongside retention, so "forgot less" cannot be confounded with "learned less".

**Work**
- Decide the summary table's exact columns before running, so the plot code is written once.

**Experiments** — **E4** (four methods, Class-IL 5×2, per-rule LR grid, ≥5 seeds).

---

## Slide 13 — Results: Domain-IL [M]

**Contents**
- Identical layout to slide 12, so the two can be compared directly by flipping between them.

**Work** — none beyond slide 12.

**Experiments** — **E5** (four methods, Domain-IL 5×2, per-rule LR grid, ≥5 seeds).

---

## Slide 14 — Scenario comparison [M]

**Contents**
- Side-by-side summary table: retention and trade-off efficiency for each method in each scenario.
- What differs between the scenarios and what does not.
- What can and cannot be attributed from these results alone — stated explicitly rather than inferred.

**Work**
- Write the attribution statement only after E4/E5 are in. Do not pre-write a conclusion.

**Experiments** — derived from **E4** and **E5**.

---

## Slide 15 — Target alignment [M]

**Contents**
- [R1]'s own interference metric, measured in our networks, per method, per scenario, over training.
- Comparison with what [R1] Fig 3 reports.
- Whether alignment tracks retention, or diverges from it.

**Work**
- Implement the metric exactly as defined in [R1] Fig 3b: the post-learning output must be measured *without* the target provided.
- Check [R1] Fig 3d before running, so the learning-rate control is set up the same way.

**Experiments** — **E6** (target alignment, four methods, both scenarios).

---

## Slide 16 — Where the weight movement goes [O]

**Contents**
- After settling, ΔW1 ∝ x0ᵀe1, so PC changes a weight in proportion to the activity displacement its settling required.
- Hypothesis: PC interferes with earlier contexts only to the extent that satisfying the current target forces movement in the hidden units those contexts depend on.
- Measure: accumulated per-weight |Δw| during later contexts, split by earlier-context importance rank; overlap between the "important to context 1" and "heavily updated in context 5" weight sets.
- Pre-committed outcomes: concentration away from earlier-context weights supports a locality mechanism; equal overlap with less forgetting means the advantage is something else, e.g. lower update variance.
- Retired hypothesis, stated for honesty: "an already-correct output has zero error so its weights barely move" is a within-one-forward-pass statement and gives earlier contexts no protection.

**Work**
- Decide the importance measure (diagonal Fisher, or accumulated |∂L/∂w|) and record the choice.
- Read [R4] for the Fisher formulation if that is the route taken.

**Experiments** — **E7**.

---

## Slide 17 — Does the learning rule change where forgetting happens [O]

**Contents**
- Slide 9's probe and freeze analysis, extended from backprop to all four rules.
- Question: does PC shift forgetting between trunk and head, or reduce both?

**Work** — none beyond slide 9.

**Experiments** — **E2** and **E3**, extended to all four methods.

---

## Slide 18 — Depth [O]

**Contents**
- [R1] claims alignment degrades with depth for backprop but not PC [Fig 3e], and that the advantage grows with depth [Fig 4h].
- Framed as justification: depth 1 is the regime least favourable to PC, so results are a lower bound.
- If run: 1 vs 3 hidden layers, Class-IL only, retention and target alignment. Direction matters, not magnitude.

**Work**
- Read [R1] Fig 3e and 4h and the corresponding Methods paragraphs.

**Experiments** — **E8**.

---

## Slide 19 — Summary: shown and not shown [M]

**Contents**
- Two columns.
- **Shown:** the controls behave (backprop forgets, replay recovers) in both scenarios; forgetting can be separated into trunk and output-layer components; results for four rules in two scenarios; target alignment measured in our own networks.
- **Not shown:** effect sizes are from one dataset at one depth with a limited seed count; no exact numeric reproduction of [R1]; Fashion-MNIST, depth ≥ 2 and convolutional architectures untested.
- Fill in the specific claims after the results are in.

**Work** — write after E4/E5/E6.

**Experiments** — none.

---

## Slide 20 — Next steps [M]

**Contents**
- Three mechanisms support memory in the brain, and each maps onto a family of continual-learning methods:

| Mechanism | Method family | In our EBM? |
|---|---|---|
| Local error-driven synaptic plasticity | PC, EqProp | yes |
| Hippocampal replay of past experience | replay buffer, generative replay [R10] | no |
| Synaptic consolidation across timescales | EWC [R4], SI [R12], Benna–Fusi [R14], metaplasticity [R13] | no |

- PC implements one of the three. It has a mechanism that reduces interference per update and none that preserves anything across updates.
- Proposal, following the advisor's fourth point: energy-based models compute prediction error per node at settling time for free. Use it to gate plasticity — update only the nodes whose settled activity differs most from their feedforward prediction. This is consolidation driven by a signal the model already computes, rather than a separately estimated importance matrix. Nearest template: [R13].
- Second option, one line: the EBM as its own replay generator.
- One line on remaining timeline.

**Work**
- Read [R13] for the metaplasticity mechanism and how the consolidation variable is derived.
- Read [R14] for the multi-timescale synapse argument.
- Read [R10] for generative replay framed as hippocampal function.
- Skim [R12] for how SI accumulates importance online, since that is closer to the gating proposal than EWC's post-hoc Fisher.

**Experiments** — none.

---

# 3. Experiment index

| ID | Question | Slides | Depends on |
|---|---|---|---|
| E0 | Does 14×14 preserve the joint-training accuracy of 28×28? | 5, 6 | — |
| E-metrics | Metric definitions and plotting module | 7, all | — |
| E1 | How much does backprop forget in each scenario, and does replay recover it? | 2, 8 | E-metrics |
| E2 | Do the hidden features survive when the head does not? (nearest-class-mean probe) | 9, 17 | E1 |
| E3 | Is the forgetting in W1 or W2? (freeze one, then the other) | 9, 17 | E1 |
| E4 | How do the four rules compare in Class-IL 5×2? | 12, 14 | E-metrics |
| E5 | How do the four rules compare in Domain-IL 5×2? | 13, 14 | E4 |
| E6 | What is the target alignment of each rule in each scenario? | 15 | E4, E5 |
| E7 | Where does the weight movement go during later contexts? | 16 | E4 |
| E8 | Does depth change the conclusion? | 18 | E4 |

Order of work: E-metrics → E1 → E4 → E5 → E2/E3 → E6 → E0 → E7 → E8.

---

# 4. Code

Current state: all work committed; no earlier revision to return to. The depth refactor and the Song & Bogacz protocol changes stay in.

- Keep both scenarios and the depth parameter. Do not remove functionality.
- Move conditionals out of the core functions in `eqprop.py` and `predictive_coding.py`. Scenario and depth should be resolved at construction time, not inside the update step.
- Once a core function is readable and tested, freeze it. Change it only when an experiment cannot be expressed without the change.
- One script per experiment, one figure per script, named identically.
- Every change made and tested on its own, with the previous experiment re-run to confirm it still produces the same figure.
