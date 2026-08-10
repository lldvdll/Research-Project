# Timeline

Sequential log of work on this project. One entry per distinct objective; a session with several
objectives gets several entries.

**Rules for this file**
- **Entries are append-only. Once written, an entry is never edited or deleted.** Corrections are
  made in `knowledge_base.md`, not here — or by adding a later entry that supersedes an earlier one
  and names it by ID.
- Each entry: an objective (what it set out to do) and an outcome (what actually happened), one
  sentence each. The outcome must say what happened, **including when the answer was negative or the
  work was abandoned**. "Investigated X" is not an outcome; "found X does not hold because Y" is.
- Loaded when chronology matters. For "what do we currently believe", use `knowledge_base.md`.

## Provenance and numbering

Entries **001–041** were reconstructed on 2026-08-10 from fourteen documents now in
`archive/chat_logs/`. Entry **042** onward is recorded live.

Three earlier, mutually independent timelines exist in the archive, each with its own numbering.
This file replaces all three. The mapping below is provided so archived references remain traceable;
it is not a claim that the schemes agree.

| Archived file | Its scheme | Covers |
|---|---|---|
| `timeline.md` | `#001`–`#009` | Coarse reconstruction + the 2026-08-10 presentation-planning session |
| `timeline1.md` | `S0`–`S4` / `T001`–`T031` | The most granular reconstruction + an undated consolidation session |
| `timeline2.md` | `001`–`014` | A third reconstruction + the 2026-07-29 triage session |

> ⚠️ **Dating caveat.** Only two dates are recorded anywhere in the sources: 2026-07-29 and
> 2026-08-10. Everything before that is undated and its ordering is *inferred from internal
> evidence*, not observed. Entries 001–030 are therefore ordered **logically, not chronologically**.
> The relative order of the three consolidation sessions (035–041) is genuinely unresolved — see
> `knowledge_base.md` §9.3 D0.

---

## Phase A — Original plan (28×28, Adam, three datasets)

### 001 — (reconstructed, undated) — Draft a from-scratch research plan for catastrophic forgetting
**Objective:** Lay out a build guide covering how forgetting arises and how replay, EWC and
energy-based models compare.
**Outcome:** Produced a 3×3 grid design — three IL scenarios × three 28×28 datasets (MNIST,
Fashion-MNIST, KMNIST) with EWC, replay and three EBM families — plus a fourteen-module `src/`
specification, almost none of which survives into the current scope (see `knowledge_base.md` §9.3 D9).

### 002 — (reconstructed, undated) — Fix the experimental substrate for the baseline notebook
**Objective:** Decide network size, optimiser and dataset preprocessing before running anything.
**Outcome:** Amended the plan to plain SGD (exact interference identity, no momentum confound) and
MLPs only (no CNNs, no inductive priors), and settled one hidden layer at width 64–128 from an
accuracy-vs-width knee, with a joint ceiling of ≈97–98%.

### 003 — (reconstructed, undated) — Establish whether split MNIST forgets, and in which scenario
**Objective:** Demonstrate catastrophic forgetting and locate it in the scenario taxonomy.
**Outcome:** Task-IL showed no forgetting under backprop — an apparent early effect was traced to a
polarity artefact of a balanced 2-way head, fixed with `max(acc, 1−acc)` — whereas Class-IL
collapsed to ≈21.6% final mean with individual tasks driven to ≈0%, i.e. active overwriting rather
than decay to chance.

### 004 — (reconstructed, undated) — Measure gradient interference directly during Class-IL
**Objective:** Test whether Class-IL forgetting corresponds to measurable gradient conflict.
**Outcome:** Cosine similarity between current-task and prior-task gradients was predominantly
negative (≈−0.5–0), with occasional positive (cooperative) cosine coinciding with slower new-task
learning, and the sign↔Δloss identity confirmed exact under plain SGD.

### 005 — (reconstructed, undated) — Verify that Class-IL forgetting is solvable at all
**Objective:** Use replay as a positive control to rule out "forgetting is simply inevitable".
**Outcome:** Replay reached ≈78% against ≈21% for the naive baseline, establishing that the problem
is solvable while confirming that buffer-limited replay is not equivalent to joint training.

### 006 — (reconstructed, undated) — Test EWC in Class-IL
**Objective:** Determine whether parameter regularisation rescues Class-IL.
**Outcome:** EWC scored ≈20%, indistinguishable from the no-defence baseline and reproducing van de
Ven et al. Table 2, diagnosed as deadlock plus an inability to create discrimination between classes
never seen together — leaving the unresolved hypothesis that the failure is in the readout rather
than the features, whose linear-probe test was never run.

### 007 — (reconstructed, undated) — Check whether extra passes over the sequence change the picture
**Objective:** Establish what multi-pass training measures.
**Outcome:** More than one pass is cyclic revisiting rather than longer continual learning; the
baseline mean drifts from ≈20% to ≈30% and plateaus, reflecting residual output structure once every
class has been trained, not learning-to-not-forget.

### 008 — (reconstructed, undated) — Survey the EBM literature for a candidate model
**Objective:** Determine which energy-based models are appropriate, biologically plausible, and
demonstrated to overcome forgetting.
**Outcome:** Separated two things both called "EBM" — the contrastive EBM-as-classifier, which
*does* overcome Class-IL forgetting but trains by ordinary backprop, and energy-based networks with
local rules (EqProp, PC, Hopfield), which are biologically plausible but not shown to overcome
Class-IL forgetting alone — and recommended the contrastive classifier, a recommendation later
superseded by the advisor's scope.

---

## Phase B — Scope reset and the scripts era (14×14, SGD)

### 009 — (reconstructed, undated) — Resolve scope sprawl and fix an authoritative project scope
**Objective:** Cut a widening wishlist down to a deliverable dissertation.
**Outcome:** Adopted the advisor's four-point outline as authoritative — pick one EBM, compare its
forgetting to backprop, explain why they differ, try to reduce it — and formally deferred generative
replay, VAE example-ordering, the 3×3 grid, other EBM families and efficiency comparisons.

### 010 — (reconstructed, undated) — Fix an experimental substrate that isolates the learning rule
**Objective:** Choose a setup cheap enough for EqProp and controlled enough for a rule comparison.
**Outcome:** Settled on MNIST downsampled to 14×14 (196 inputs), a single-hidden-layer MLP
196→64→10, plain SGD, Class-IL with a single 10-way head, and three splits (10×1, 5×2, 2×2), with
CNNs rejected to avoid inductive priors.

### 011 — (reconstructed, undated) — Implement Equilibrium Propagation as the chosen EBM
**Objective:** Get EqProp training, as the formal answer to advisor point 1.
**Outcome:** EqProp reached ≈91% validation on joint 14×14 MNIST in ≈6 minutes for 3 epochs on CPU,
with four failure modes documented — tanh saturation severing the feedback path, the nudged phase
never reaching an absolute tolerance (hence patience-based settling), collapse at batch size 1, and
no ½x² self-term making it a weak generator.

### 012 — (reconstructed, undated) — Add Predictive Coding as a second EBM
**Objective:** Implement the rule that actually makes the "energy-based learning reduces
interference" claim, since that claim is PC's and not EqProp's.
**Outcome:** PC was implemented with all three gradients verified against finite differences to
≈1e-9 and a numpy mirror learning a toy 3-class problem to 100%, though the torch version had not
been run end-to-end at the time of handoff.

### 013 — (reconstructed, undated) — Compare four methods across three Class-IL split granularities
**Objective:** Characterise forgetting for backprop, replay, EqProp and PC at 10×1, 5×2 and 2×2.
**Outcome:** At 10×1 and 5×2 every method except replay sat at the collapse floor (≈10% / ≈20%
against replay's ≈64% / ≈60%), while the 2×2 run proved most informative — backprop held task 1
until the switch then fell off a cliff, EqProp forgot *before* it learned, PC decayed as a slope to
≈15%, and replay retained both tasks at ≈95%.

### 014 — (reconstructed, undated) — Explain why every learning rule forgets in Class-IL
**Objective:** Account for the fact that the EBMs forget too.
**Outcome:** Attributed forgetting to output competition in the shared head rather than to credit
assignment — a local rule changes *how* credit is assigned, not *that* the outputs compete — with
EqProp worst because its hinge target is −1 for every non-target class against PC's one-hot 0.

### 015 — (reconstructed, undated) — Identify what would invalidate the four-method comparison
**Objective:** Audit the comparison for confounds before trusting it.
**Outcome:** Found the learning rates differ across methods (BP 0.05, EqProp 0.005, PC 0.05), so
forgetting speed was being compared at different learning speeds — recorded as CTRL-1, requiring
`to_learn` to be matched before any `to_forget` difference is attributed to the rule.

### 016 — (reconstructed, undated) — Adopt working practices that survive periods of overwhelm
**Objective:** Stop repeated loss of momentum from over-engineering and rabbit-holing.
**Outcome:** Fixed on one question per experiment written as a single sentence beforehand, backprop
and replay controls on every forgetting run, doubts getting one scheduled test and then closing, one
script producing one identically-named figure, and the deliberate deletion of wandb, Optuna, class
hierarchies and a shared `harness.py`.

---

## Phase C — Energy-based memory and the mechanics of forgetting

### 017 — (reconstructed, undated) — Test whether one EBM can carry sharp and smooth components
**Objective:** Assess whether a single energy could hold a per-sample memory component for
generative replay alongside a smoothed component for discrimination.
**Outcome:** Judged plausible and partly precedented — the memorisation↔generalisation transition is
controlled by inverse temperature in dense associative memory, and a diffusion model is already one
energy carrying both regimes indexed by noise scale — with the novelty narrowed to a *single
decomposed energy* rather than the usual two-module instantiations, and the whole thread parked as
outside the advisor's scope.

### 018 — (reconstructed, undated) — Get precise about continual-learning taxonomy
**Objective:** Name the project's target regime unambiguously against competing taxonomies.
**Outcome:** Separated three orthogonal axes — data shift, IL scenario, stream carving — showed the
taxonomies genuinely disagree (permuted MNIST is multi-task for Maltoni & Lomonaco but explicitly
Domain-IL and *not* Task-IL for van de Ven), and fixed the target regime as Class-IL under a
task-free stream.

### 019 — (reconstructed, undated) — Work out where forgetting lives in the architecture
**Objective:** Derive, rather than assume, what is damaged when a new class arrives.
**Outcome:** Derived ∂L/∂z_o = p_o − 1[o=t], identified the active set 𝒜 as the knob that determines
which mechanisms can fire in which scenario, and decomposed Class-IL forgetting into two pathologies
— logit suppression, a calibration failure that is cheap to fix without replay, and the absent
inter-context discriminative signal, a representation failure that is irreducible without old-class
information.

### 020 — (reconstructed, undated) — Test whether dropping the softmax removes suppression
**Objective:** Evaluate the proposal that a linear or independently-activated output would remove
normalisation-driven forgetting.
**Outcome:** Refuted as stated — the source is the one-hot target supplying zero for every absent
class, so linear+MSE and sigmoid+BCE still push absent logits down — but a real partial gain was
identified in MSE's fixed point at zero, and the underlying tension was named: more output coupling
buys calibration and costs suppression, and no single knob wins both.

### 021 — (reconstructed, undated) — Interpret the four-method comparison run
**Objective:** Read the results of backprop, replay, EqProp and PC across ten random digit pairings.
**Outcome:** Issued a premise correction — EqProp is *not* forgetting less, it is simply slower
throughout, and the trajectory plot with time removed shows it as the worst panel — and found two
orderings that disagree, trade-off efficiency running pc > replay > backprop > eqprop while final
task-1 retention runs replay ≫ pc > eqprop > backprop, with that divergence identified as the whole
story.

### 022 — (reconstructed, undated) — Give each method's forgetting signature a mechanism
**Objective:** Explain the shape of each method's curve rather than describing it.
**Outcome:** Attributed backprop's cliff to output-layer suppression plus cross-layer interference,
replay's floor to replayed samples arriving as *positives* that cancel the suppression term, PC's
above-diagonal path with no floor to it attacking trunk interference while doing nothing about
one-hot zeros, and EqProp's position below backprop to it being a finite-difference estimator of
backprop's gradient — making that the expected result, not an anomaly.

---

## Phase D — Conceptual rebuild, code verification, corrections

### 023 — (reconstructed, undated) — Rebuild the FFNN/BP/EBM/PCN/PC/EqProp account from primary sources
**Objective:** Obtain a mechanical, non-metaphorical understanding grounded in the papers.
**Outcome:** Settled three load-bearing statements — an FFNN *is* a PCN with its internal state
clamped to the forward pass; every energy-based rule runs an EM-like two-step with a fast activity
loop inside a slow weight loop, which backprop lacks entirely; and the global energy is never
computed, stored or transmitted, being only a Lyapunov function used by the analyst.

### 024 — (reconstructed, undated) — Verify the implementation against the equations
**Objective:** Confirm the code is correct before trusting any result from it.
**Outcome:** `predictive_coding.py`, `eqprop.py`, `methods.py`, `data.py` and
`11_consolidate_pairs_4methods.py` were all confirmed correct — including the sign of ∂F/∂W, the
hinge nudge, and the warm-started nudged phase — meaning EqProp's noisiness is EqProp being EqProp
rather than a bug.

### 025 — (reconstructed, undated) — Re-examine two earlier explanations of PC's advantage
**Objective:** Check two claims that had been asserted without support.
**Outcome:** Both were corrected — "PCN error nodes are basically backprop's errors" holds only
under partial relaxation or an engineered equivalence that is not general, and "an already-correct
output has zero error so its weights don't move" is a within-a-single-example statement that gives
task 1 no protection whatsoever during task-2 training, superseding the mechanism asserted in 022.

### 026 — (reconstructed, undated) — Replace the refuted mechanism with something testable
**Objective:** Turn PC's claimed advantage into a measurable quantity.
**Outcome:** Because `x1` is initialised to `mu1`, the residual `e1` after settling is literally the
displacement of the hidden layer from its feedforward value and ΔW1 is proportional to it — which
reframes the question from "is the error zero?" (no) to "where does the weight movement go?"
(measurable), and gives hypothesis H1.

### 027 — (reconstructed, undated) — Audit the comparison for confounds beyond learning rate
**Objective:** Check what else differs between the four methods.
**Outcome:** Found that `make_backprop` and `make_replay` use ReLU + cross-entropy while `pc` and
`eqprop` use tanh + squared error, so the comparison varies algorithm, nonlinearity and loss
simultaneously — recorded as CTRL-2 with a matched BP control specified as `x1 = x0 @ W1;
out = tanh(x1) @ W2; loss = ½|target − out|²`, no biases.

### 028 — (reconstructed, undated) — Lay out a separated experiment sequence
**Objective:** Replace mega-experiments with one research question per script.
**Outcome:** Queued a Song & Bogacz Fig 4d–e reproduction followed by experiments on whether task 1
lives in a subset of hidden units, whether importance correlates with magnitude, where the weight
movement goes, a freezing control, whether EWC stacks on PC, and a metaplasticity augmentation —
with the "kinematic consolidation" idea quarantined until the weight-movement experiment reports.

### 029 — (reconstructed, undated) — Run the four-method comparison over ten digit pairings (experiment 11)
**Objective:** Establish what is known, with variance across pairings, for all four methods.
**Outcome:** Produced the figures underlying entries 021–022, but the run's own split is recorded
inconsistently across the sources — 2 tasks × 2 classes in one document and "5×2" in two others —
and its replay retention figure appears as both ≈37% and ≈68%, both unresolved
(`knowledge_base.md` §9.3 D1, D2).

### 030 — (reconstructed, undated) — Attempt a like-for-like reproduction of Song & Bogacz Fig 4d (experiment 12)
**Objective:** Reproduce the published continual-learning result before dissecting its mechanism.
**Outcome:** The reproduction was **not achieved** — roughly a week was spent on hyperparameter
matching, previously working experiment-12 code was broken, and the codebase became unreadable to
the author.

---

## Phase E — Three parallel consolidation attempts

> The three sessions below each independently consolidated the same handoff documents into a
> `timeline.md` + `knowledge_base.md` pair, each inventing its own numbering, and **none references
> the others**. Only the 2026-08-10 session is securely last. See `knowledge_base.md` §9.3 D0.

### 031 — (undated; archived as `timeline1.md` S4/T026) — Diagnose the sense of information overload
**Objective:** Work out why the project felt both overwhelming and internally contradictory.
**Outcome:** Traced the confusion to collapsing three independent axes the literature treats as one
— credit assignment, continual-learning mitigation, and IL scenario — compounded by a metric
mismatch (Song & Bogacz report mean test error *during* training, van de Ven reports final average
accuracy, so their numbers were never comparable) and by numeric disagreements between the handoff
documents.

### 032 — (undated; archived as `timeline1.md` T027) — Extend the reference base
**Objective:** Add work not yet in the project.
**Outcome:** Added Pinchetti et al. (ICLR 2025, PCX) as the fair-comparison and scaling authority —
PC rivals backprop at VGG-7 scale but degrades on 9-layer convnets and ResNets — plus Ororbia's
sequential neural coding networks and BayesPCN as the closest direct prior work, and Lopez-Paz &
Ranzato for formal backward-transfer metrics.

### 033 — (undated; archived as `timeline1.md` T029) — Determine whether the four methods use different output structures
**Objective:** Establish whether the comparison could be reduced to one architecture.
**Outcome:** Confirmed three simultaneous differences — activation, output stage and loss, and label
coding — and proposed a single shared specification (no biases, linear hidden pre-activation, tanh on
the way out, one-hot target, squared error, plain SGD), chosen because squared error is the only loss
all three rules accept unaltered.

### 034 — (undated; archived as `timeline1.md` T030) — Establish where forgetting lives, and whether it matters
**Objective:** Decide between the hidden layer and the output layer as the dominant damage site.
**Outcome:** Three independent observations already in our own data — Task-IL barely forgets,
Class-IL task-1 accuracy falls to 0% rather than 50%, and replay alone retains anything — point to
the output layer, which implies no learning rule can fix it and makes the nearest-class-mean probe a
decision point rather than an optional extra.

### 035 — (undated; archived as `timeline1.md` T028, T031) — Create the first timeline and knowledge base
**Objective:** Establish the two maintained project files, and make them readable.
**Outcome:** Wrote `timeline1.md` and `knowledge_base1.md` from the three handoff documents plus the
four in-project PDFs — preserving numeric disagreements as D1–D6 rather than reconciling them by
guesswork — then added a label key, two diagrams and a full plain-English glossary, and renamed the
confound labels C1–C6 to CTRL-1–CTRL-6 to remove a clash.

### 036 — 2026-07-29 (archived as `timeline2.md` 013) — Triage the accumulated quarantine list
**Objective:** Assess a backlog of seventeen research ideas, group them, and order the work.
**Outcome:** Sorted the seventeen items into five research programmes, judged seven already answered
by the literature or superseded by cheaper reproductions, dropped two as architecture-blocked or
advisor-deferred, and defined a five-tier ordering with the head-vs-trunk probe and the
target-structure control as the highest-value next experiments.

### 037 — 2026-07-29 (archived as `timeline2.md` 014) — Create a second timeline and knowledge base
**Objective:** Establish the project's persistent record.
**Outcome:** Wrote `timeline2.md` and `knowledge_base2.md`, seeded from the same three handoff
documents, apparently without knowledge of entry 035 — producing a second, differently numbered
record of the same history, and reusing the labels D1–D3 for *experiments* where the other file uses
them for *discrepancies*.

### 038 — 2026-08-10 (archived as `timeline.md` #008) — Reset the project and critically review the draft deck
**Objective:** Install a timeline and knowledge-base workflow, evaluate a draft 13-slide deck, and
produce a locked slide plan for a 20-minute presentation.
**Outcome:** Restructured the deck into a 20-slide plan around a new spine — that Song & Bogacz Fig
4d is Domain-IL, not Class-IL, making the scenario contrast the headline — with scoped experiments
mapped one-to-one onto slides and the experiment-12 codebase ordered frozen rather than repaired;
the slide and experiment counts recorded here were revised in 039.

### 039 — 2026-08-10 (archived as `timeline.md` #009) — Fix the open decisions and rewrite the presentation plan
**Objective:** Close the decisions left open by the plan review and put the plan in a working format.
**Outcome:** Locked the 5×2 split for both scenarios with the output layer as the only difference,
kept Class-IL and Domain-IL as parallel result slides plus a comparison slide, **retired the
"scenario boundary" framing as premature**, and rewrote the plan as 20 slides (15 mandatory, 5
optional) with per-slide contents, reading and experiments and a ten-item experiment index.

### 040 — 2026-08-10 (archived as `current.txt`) — Rule on the reproduction and the code policy
**Objective:** Decide what to do about the failed Fig 4d reproduction and the unreadable codebase.
**Outcome:** Stopped chasing a numeric match and reframed the work as a conceptual replication of
the protocol; kept all committed code with no revert, ruled that the depth parameter and the Song &
Bogacz protocol work stay in, and set the refactoring brief to move conditionals out of the core
functions so scenario and depth resolve at construction time.

---

## Phase F — Current

### 041 — 2026-08-10 — Port the project to Claude Code and consolidate the archive
**Objective:** Install a Claude Code configuration with an enforced change-control boundary, and
replace fourteen overlapping documents with one knowledge base, one timeline and one slide plan.
**Outcome:** Wrote `CLAUDE.md`, `.claude/settings.json` (plan mode by default, `ask` on `src/` and
`experiments/`, `deny` on `ref/` and the archive), two path-scoped rules files and three
user-invoked skills; consolidated the fourteen source documents into this file plus
`knowledge_base.md` and `presentation_plan.md`, **preserving twelve contradictions as open items in
`knowledge_base.md` §9.3 rather than resolving them**; and moved the raw logs to
`archive/chat_logs/` unchanged.
