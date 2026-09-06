# Report track — eleven tiers, one figure spec per question

The project as it would be run from a standing start, given what is now known. Strictly cumulative:
each tier is only askable once the one above it has an answer. Tiers 1–9 are shared; tiers 10 and 11
split the mechanism investigation by scenario, and tier 9 is where that split earns its place.

## How the tiers map onto the report's sections

**This document is the QUESTION order, not the section order.** The report is written as
Methods M1–M3 / Results R1–R5 / Discussion D1 — that structure is authoritative in
`script_plan_800_900.md` (the 900 table) and in `report/sections/*.tex`. A tier is a question;
a section is where its answer is told. They are not one-to-one, and neither supersedes the other.

| Section | Tiers it draws on | Figures |
|---|---|---|
| **M1** what forgetting looks like, and what is measured | 1 | 901 |
| **M2** network, data, and the choices behind them | 4, 5 | 902–906 |
| **M3** why the comparison stops at matched competence | 2 | 907, 908 |
| **R1** two scenarios, two forgetting phenotypes | 3, 9 | 911 |
| **R2** the size of the learning-rule effect | 6, 7, 8 | 912, 913, 914 |
| **R3** testing the mechanism PC is credited with | — (new: 803) | 915, 916 |
| **R4** what repeated alternation reveals | — (new: 804, 805) | 917–920 |
| **R5** evidence for an output-competition component | 10, 11 | 921–926 |
| **D1** discussion and limits | 10, 11 intervention rows | 931, 932 |

Two things the tier list does not contain, because they were added after it was written:
**R3** (target alignment and settling displacement — a direct test of Song & Bogacz's own
credited mechanism, restored from the "deliberately out of scope" note at the foot of this file)
and **R4** (repeated alternation and joint-then-sequential). Where this file and the section
structure disagree about scope, the section structure is newer and wins; where they disagree
about a NUMBER, neither wins — `progress.md` does, because it is re-derived from the arrays.

**Every number here is carried from `progress.md`, which re-derived it from the saved arrays.**
Where a tier needs something that does not exist, it says so in its Status.

**Standing rule the earlier plans lacked.** A control tier (4, 5, 6) is not a one-time gate. It is a
precondition attached to *every axis later swept*. dt = 0.4 was correctly settled at H = 32,
depth = 1 and was silently wrong at depth ≥ 2 and at Class-IL H = 4 (346, 347), which cost two full
sweep re-runs. Re-verify the control whenever the configuration moves.

**Three figures must draw their censoring, not average over it** — tier 2, tier 6, tier 10/11's
intervention panel. A metric that is undefined on some seeds and averaged anyway reports something
that did not happen.

---

## Tiers 1–9 — shared

### 1. Does forgetting happen, and what does it look like?

Definitional. Names the switch, the plateau and the crossing point so nothing later needs
re-explaining.

| | |
|---|---|
| **Data / tool** | MNIST 14×14, 2×5 split, backprop, 10 seeds, 90% matched competence |
| **Plot form** | x = training step relative to the switch · y = accuracy (%) · thin per-seed lines behind a bold mean · shaded by which task is training |
| **Primary → secondary** | Task identity (t1 / t2) → scenario (panel) |
| **Status** | **Have** — 102's grid. Use the backprop row only; PC does not appear until tier 7 |

### 2. How do we measure forgetting, and does the choice change the answer?

Promoted from a setup step to a result. It is the most defensible finding in the project and it
licenses every comparison downstream.

| | |
|---|---|
| **Data / tool** | Backprop, both scenarios, 10 seeds, stopping threshold swept 75 → 95%; five metrics from the same runs |
| **Plot form** | **A** x = stopping threshold · y = metric value. **B** x = stopping threshold · y = paired PC − BP ± SEM, zero line, one small panel per metric. **C** x = stopping threshold · y = fraction of seeds on which the metric is defined |
| **Primary → secondary** | Metric identity → stopping point |
| **Status** | **Have** (111, 112, 113). **Panel C does not exist and must be added** |

**Result.** Class-IL: only crossover (+1.56 to +2.54) and crossover-of-peak (+1.93 to +2.85) hold one
sign with error bars clear of zero at all five thresholds, at 2 SEM. Every endpoint metric flips —
final t1 +2.24 → −0.30, S&B error −1.56 → +0.17, area retained +5.47 → −0.66. Domain-IL: nothing
survives, crossover included, consistent with there being no effect there.

**State on the figure:** crossover's *raw value* is not stopping-point independent (110: Domain
36.2 → 75.3 across the budget sweep). The stable quantity is the *paired difference*.

### 3. Does backprop forget, and does the character differ by scenario?

| | |
|---|---|
| **Data / tool** | Backprop, seeds 10–19, both scenarios, 90% matched competence |
| **Plot form** | Two panels · x = task-1 accuracy · y = task-2 accuracy, time removed · 10 seed lines coloured by which task is training · x=y diagonal marked with each seed's crossing · marginal histograms of crossing height and final task-1 |
| **Primary → secondary** | Scenario (panel) → seed (spread, not averaged) |
| **Status** | **Have** (310) |

**Result.** Final task 1 **5.4% (Class-IL) vs 38.5% (Domain-IL)**; crossover 65.0 vs 75.8. Same rule,
same protocol, same seeds — only the output layer differs. This is the first evidence for tier 9's
split, and the phase-plot form is what makes the difference in *character*, not just magnitude,
visible.

### 4. Is PC set up to be measured fairly?

Not a control "for forgetting" — a control for PC being a faithful implementation of itself. If
settling oscillates, the thing being measured is not PC.

| | |
|---|---|
| **Data / tool** | Settle traces run to a fixed cap with no early exit; `classify_settle` applied post-hoc |
| **Plot form** | **A** x = settle step · y = mean abs displacement (log) · one line per dt · convergence step marked. **B** x = dt · y = mean settle steps (a V-curve) with a companion panel of outcome metrics, flat. **C** grid, x = dt · y = depth, cells coloured by classification |
| **Primary → secondary** | dt → depth, then width |
| **Status** | **Have** (334, 330, 346, 347). Four figures available; use three |

**Result.** dt ≤ 0.5 all reach the **identical plateau** (0.0850 Class-IL, 0.0787 Domain-IL) — dt
changes the route, not the fixed point. Settle steps fall 71.2 → 8.5 from dt 0.02 to 0.4 with every
outcome flat: 8× cheaper, same answer. Above 0.4 the cost turns back up because updates stop
converging (2309.6/2316 converge at dt 0.5; 0/2960 at 0.85). The stable range **shrinks with
depth** — the only unstable cell in 347's grid is dt = 0.4 at depth ≥ 2, and the oscillation tail
sits 3–7× above the true converged value, so it is a qualitatively wrong state rather than overshoot.

### 5. How do we make the backprop/PC comparison fair?

Three controls, one three-panel figure; each panel answers a different objection.

| | |
|---|---|
| **Data / tool** | (a) fixed budget vs matched competence · (b) paired vs unpaired seeds · (c) one shared output specification |
| **Plot form** | **a** x = step · y = accuracy · 90% threshold line · fixed-budget above, matched-competence below. **b** same point estimate, two error bars. **c** crossover under budget 840 vs 840 and threshold 90 vs 90 |
| **Primary → secondary** | Protocol → pairing |
| **Status** | **Partly** — (a) and (c) have figures (76, 72/73). **(b) has no figure; the SEM-shrink numbers exist in 69 but were never drawn** |

**Result.** Crossover is invariant to the budget *and* the threshold, and the reason is stronger than
the number: 72's first 147 evals are bit-identical to 52's, so the extra budget is appended entirely
after the crossing; 73 and 57 are bit-identical over all 101 shared steps. Pairing does not move the
point estimate — it shrinks the SEM **4.5× in Class-IL and 11.5× in Domain-IL**, because the class
split is shared between rules at a given seed.

### 6. Does the learning rate manufacture the answer?

Its own tier, because the result is counterintuitive and a reader will otherwise assume the
comparison was tuned.

| | |
|---|---|
| **Data / tool** | backprop + PC + replay on one shared absolute lr grid {0.005 … 0.16}, 10 seeds, both scenarios |
| **Plot form** | **A** x = lr (log) · y = absolute crossover · one line per rule. **B** x = lr · y = PC − BP ± SEM, zero line. **C** x = lr · y = % of seeds censored |
| **Primary → secondary** | Rule → lr, then scenario (columns) |
| **Status** | **Have** (340, 343) |

**Result.** **Both rules peak near lr 0.01–0.02 and decline past it.** The growing PC − BP gap at high
lr is **backprop degrading faster, not PC improving** — the established defaults sit close to jointly
optimal, not in a blind spot. Panel C is not optional: at lr 0.16 backprop's Class-IL crossover is
undefined on **10/10** seeds and PC's on 6/10. A measurement-resolution failure, not a null result.

### 7. Does PC forget less than backprop?

| | |
|---|---|
| **Data / tool** | 332 against 310 paired seed-for-seed; then 341 (width 4–64) and 342 (depth 1–4), both re-run at the corrected dt = 0.2, 10 seeds |
| **Plot form** | **A** the tier-3 phase plot with PC overlaid. **B** x = lr / width / depth, one panel each · y = paired Δcrossover ± SEM · zero line · scenario as colour |
| **Primary → secondary** | Rule → swept axis, then scenario |
| **Status** | **Have** — but **panel B has never been drawn as one consolidated figure**; 340/341/342 each plot their own axis separately |

**Result.** At the working point (H = 32, depth 1): Class-IL **+1.42 ± 0.39**, Domain-IL
**−0.72 ± 0.19**, crossover defined 10/10 in both. Small, real, and scenario-flipped. Task-1 phase
lengths are comparable (719/707 vs 755/787 steps), so this is not PC retaining by training less.

**Depth amplifies the split in both directions** — new, and it contradicts the pre-100 "depth doesn't
matter" line:

| PC − BP crossover | depth 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Class-IL | +1.42 ± 0.39 | +2.91 ± 0.78 | +3.33 ± 0.66 | **+3.44 ± 0.85** |
| Domain-IL | −0.72 ± 0.19 | −0.32 ± 0.31 | −0.89 ± 0.57 | **−3.49 ± 0.61** |

Width behaves the same way and adds one reversal: Class-IL **H = 4 is −4.14 ± 0.70** — PC is
strongly *worse* at the narrowest width, and this survives the dt correction (it was −6.19 under the
confounded dt = 0.4). Domain-IL H = 4 is +2.21 ± 0.46, the only width where PC wins there.

### 8. Is the problem solvable at all?

The tier both earlier plans left out. Without it every null in tiers 10–11 has no scale to be read
against, and the obvious reader question — "is this even fixable?" — goes unanswered.

| | |
|---|---|
| **Data / tool** | replay, already present in 340, 341 and 342's arrays, 10 seeds, both scenarios |
| **Plot form** | The tier-7 panel with replay as a third line, and a single headline bar: paired Δcrossover for PC and for replay, both scenarios |
| **Primary → secondary** | Rule (bp / PC / replay) → scenario |
| **Status** | **Data exists, no figure isolates it.** Highest-value missing figure in the project |

**Result.**

| paired Δcrossover vs backprop, H = 32 | PC | replay |
|---|---|---|
| Class-IL | +1.42 ± 0.39 | **+9.84 ± 0.77** |
| Domain-IL | −0.72 ± 0.19 | **+3.43 ± 0.50** |

Replay is ~7× PC's effect where PC wins and positive everywhere PC is negative. It is flat across the
whole lr grid and across width and depth, so it is not a tuning artefact. The problem is solvable;
changing the credit-assignment rule is not what solves it.

### 9. Does the scenario difference justify two separate investigations?

The hinge of the report. Everything above is shared; everything below splits.

| | |
|---|---|
| **Data / tool** | Consolidation of every measurement made in both scenarios — no new runs |
| **Plot form** | One figure, rows = measurement, columns = the two scenarios, cells = the effect with its SEM; or a paired dot-plot with a line joining each measurement's two scenario values, so sign flips read as crossings |
| **Primary → secondary** | Scenario → measurement |
| **Status** | **Need to build** — pure re-analysis of saved arrays, no retraining |

**The case the figure has to make, and it has both halves.**

*Structural.* Class-IL has 10 output units, five of which receive no positive target during task 2,
so output suppression is available as a mechanism. Domain-IL has 5 shared units and suppression
**cannot occur** — whatever forgetting happens there must be in the representation or the shared
readout. The two scenarios do not merely differ in degree; they differ in which mechanisms are
physically available.

*Empirical.* The measurements separate by sign, not just magnitude — PC − BP is +1.42 vs −0.72 and
the depth trend runs opposite ways; the digit-identity effect is Class-IL only, and the Domain-IL
pairing correlation did not replicate; masking is only definable in Class-IL. Against that, two
things are the *same* in both and must be shown, or the figure is one-sided: freezing W2 recovers
nothing in either (+0.36 ± 0.43 and +0.15 ± 0.24), and the argmax-minus-probe gap is large in both
(+81.6 Class-IL, +29.5 Domain-IL) even though suppression cannot occur in one of them.

**Honest reading:** the split is justified by mechanism availability and by sign reversal. It is
*not* justified by the readout gap, which behaves similarly in both. Say so.

---

## Tier 10 — Class-IL: what shape, why, where, and does controlling it help?

One figure grid. Four questions, because in Class-IL they have a shared answer to test: that
forgetting is a readout failure over a surviving representation.

| | |
|---|---|
| **Shape** | Per-seed post-switch task-1 trace classified into collapse-no-recovery / holds-then-destroyed / gradual-relearn / noise-dominated — a direct analogue of `classify_settle`. x = category (4 bars) · y = seed count · rule as colour. **Need to build; cheap, no retraining** |
| **Why** | Retention split on which digits land in task 1. Rows = seeds ordered by retention · columns = the ten digits coloured by task. **Have** (312, 313) |
| **Where** | (a) freeze W1 / W2 / both during task 2, paired against the unfrozen control — *causal*. (b) trained/refit linear probe vs argmax — *observational*. **(a) Have (130); (b) needs building — NCM is excluded, see below** |
| **Does controlling help** | Every intervention on one axis: replay, masking, freezing, EWC, SI, k-WTA · x = intervention · y = paired Δcrossover ± SEM · zero line · censoring annotated. **Have the data (340, 120, 130, 200, 210, 220); the consolidated figure needs building** |

**Results.** Shape: retention splits into two groups, near-zero and roughly 15–30%, over 70 seeds
(313: mean 6.58, range 0–38.1, 18/50 above 5%). Averaging reports a number no individual run
produces. Why: a per-digit effect — 8 (11.6 vs 3.0, p = 0.006), 9 (10.8 vs 3.0, p = 0.005),
5 (9.5 vs 3.5, p = 0.025) raise retention when present in task 1; 6 (3.0 vs 10.5, p = 0.007) lowers
it. At Bonferroni only 6, 8, 9 survive. Where: the code survives while argmax reads the floor —
task-1 argmax **5.4** vs NCM **66.5** — but **freezing the output layer recovers nothing**
(+0.36 ± 0.43). Does controlling help: replay ≫ masking (+7.6 to +17.5, spec-dependent) > SI (+3.15)
≈ EWC (+1.80) > freezing (nothing) > k-WTA (**−9.8 bp, −32.4 pc**).

**The tension is the finding, and this grid must stage it rather than resolve it.** The readout
survives, but protecting the readout layer does not help. Either the damage is done through the
readout by a route freezing does not block, or the surviving-code reading is wrong. The project has
not distinguished these — and that is what the trained probe in "Where (b)" is for.

**Why NCM is excluded.** It is a centroid-only, first-order probe with almost no dynamic range: 21's
own grid puts frozen NCM at 69.9% at H = 32, a raw-pixel NCM baseline on all ten digits is 81.2%,
and 101 shows the trained representation is worth ~12 points (77.2 frozen-random vs 89.7 trained) —
more than NCM can resolve. A flat probe cannot answer a question about change. The one control it
does provide is worth keeping: on **task 2 the sign reverses**, NCM 70.8 against argmax 91.3, which
rules out "the probe is just a better classifier".

---

## Tier 11 — Domain-IL: what shape, why, where, and does controlling it help?

Same four questions, and **the honest answer is that this half is much weaker.** Output suppression
is structurally unavailable, so the explanation has to be representation drift — and the project has
no direct measurement of drift under the current protocol.

| | |
|---|---|
| **Shape** | Same classifier as tier 10, applied to Domain-IL traces. **Need to build** |
| **Why** | Two candidate drivers, and both are currently negative results: class-pair similarity at the shared output unit (**did not replicate**), and hyperparameters (lr, width, depth — swept, and backprop's own retention is broadly flat until high lr). **Have the data, needs re-reading against the shape categories** |
| **Where** | Freezing (130) says the output layer is not it — which in Domain-IL is expected, not informative. **No positive localisation exists.** The trained probe applied here is the one tool that could produce one |
| **Does controlling help** | Same intervention axis as tier 10, Domain-IL arm. **Have the data; figure needs building.** Note masking is undefined here — there are no absent classes to mask |

**Results.** Shape: a broad unimodal spread (13.1–64.2 across seeds in 300), not bimodal — genuinely
different from Class-IL, which is itself a finding. Why: pairing similarity gave r = +0.686
(p = 0.029) on seeds 0–9 and **r = +0.183 (p = 0.61)** on seeds 10–19; pooled over 20 seeds r = +0.509
(p = 0.022). Real but far weaker than the r = +0.787 originally seen, and absent from an independent
block. Does controlling help: replay **+3.43 ± 0.50**, EWC +0.74 bp / +0.09 pc, SI +0.83 / +0.87 (on
n = 9), freezing +0.15 ± 0.24, k-WTA **−30.2 for PC**.

**Say plainly what this tier does not have.** No positive answer to "where". The remaining
candidates — hidden-layer representation drift, and interference in the shared readout's weights —
have not been separated. Path efficiency and layerwise cosine (65, 66) measure the right kind of
thing but were run pre-300 under a different protocol and against four rules including one since
dropped; 66 is explicitly flagged as needing re-verification before citation. This is the largest
genuine gap in the project, and it should be reported as one rather than filled with the Class-IL
answer by analogy.

---

## What has to be built

Nothing requires new training. Six figures, in value order:

1. **Replay as positive control** (tier 8) — arrays in 340/341/342, no script draws it. Highest value.
2. **Trained/refit linear probe vs argmax** (tiers 10, 11) — the one tool never built; tier 11's only route to a positive answer.
3. **Forgetting-shape classifier** (tiers 10, 11) — analogue of `classify_settle`, cheap, and it answers the four-category question as posed.
4. **Consolidated PC − BP across lr × width × depth** (tier 7) — three existing scripts' arrays, one figure.
5. **Scenario-split justification** (tier 9) — pure re-analysis; the hinge of the report structure.
6. **Intervention summary bars** (tiers 10, 11) — six experiments' arrays on one axis.

Plus two small additions to existing figures: a censoring panel on tier 2, and the pairing
SEM-shrink panel on tier 5.

**Deliberately out of scope.** EqProp (dropped). Target alignment and path efficiency as headline
results (64, 65, 66 — they answer "how do the rules differ" rather than any tier above; 66 needs
re-verification; keep as appendix or as the raw material for tier 11's probe work).
