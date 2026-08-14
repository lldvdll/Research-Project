# Now

Dashboard. One line per fact. Detail lives in `current_state.md` and in each script's
docstring — keep this file short enough to read in one sitting.

**Running:** 64 (target alignment + interference) and 65 (synaptic path efficiency), both
EqProp-bound. Started 2026-08-13.

**52's re-run resolved the ⚠ inconsistency.** The `replay_frac` None→0.5 fix changed every
metric by ≤0.4 points and flipped no verdict (crossover +2.67→+2.56, retention +26.2→+26.6).
The double-batch confound was real in principle and nil in magnitude. **52 is comparable with
53/56/57.** Replay's advantage was never the extra data.

## What each change means for existing runs
| change | affects | action |
|---|---|---|
| `replay_frac` None → 0.5 | 52 only | **done — no material change** |
| EqProp `settle_tol` | nothing in the 50s — landed before 52 | none |
| `plotting._mean_of_all` | figures with ragged runs | **43 done**; 53/57 pending |
| `crossover_after=` on learning curves | all comparison figures | 52/56/60 done; 53/57 pending |
| metric grid | every comparison | script **62** re-reports without retraining |
| `settle_patience` restored | 01–27 and **61** now run again | none |
| **A1 ce-mask gate** | legacy-spec runs only | **none — 61 re-run reproduces exactly** |
| **A5 censoring report** | every crossover / half-life line | re-report via 62; no number changes |

**53 and 57 have no `--replot` path** — they re-train from scratch, so their figures cannot be
regenerated cheaply and still lack the crossover annotation. Numbers are unaffected and script
62 re-reports their full metric grid without retraining. Add `--replot` to both when next
touched; do not re-run them just for the figure.

## THE CODE AUDIT — 2026-08-13, all 12 modules, five fixes
Committed as `9e485aa`. PC's protocol path is **bit-identical** afterwards (regression-checked
against 60's saved seed-0 curve); 61 re-run reproduces −4.52 (3.5 sem) exactly.

- **A1 — `ce` applied a hard task mask regardless of `obj.mask`.** `output_error`'s `ce` branch
  masked the softmax whenever `active_vec` was passed, and `run_classil` passes it on every
  call while `active_vector` never returns None. So under `LEGACY_SPEC` **backprop and replay
  trained with an ORACLE TASK MASK** — absent classes got exactly zero gradient — while PC
  (mse) saw the full error and EqProp (hinge, ±1) actively pushed them down. Measured: ‖e‖ on
  task-1 units during task 2 = 0.0 / 5.97 / 10.0. In Class-IL, where output suppression is the
  dominant mechanism, that is a rule-independent advantage handed to two of four rules.
  **A second reason exp 12 is not evidence**, independent of its inverted control. Now gated on
  `obj.mask`; `LEGACY_SPEC` states `mask=True` explicitly so old scripts reproduce byte-for-byte.
  The A series never touched it — mse, `mask=False` throughout.
- **A2** — `pc_update`'s optimiser path divided by `n` where the raw path divided by
  `batch_scale`. Identical under `mean`; 0.29 apart under `sum`, the reduction that exists to
  match [R1].
- **A3** — `pc_update` froze `b1` along with `W1`; the other three rules freeze per named
  tensor. One freeze set now means one thing under all four rules.
- **A4** — `make_eqprop` defaulted `max_steps=500` against `METHOD_DEFAULTS`' 800.
- **A5 — censoring, and it hits the primary metric.** `report_grid` printed a group mean over
  all finite runs beside a paired difference over pairs finite on *both* sides, with no n. See
  the censoring section below.
- **A6** — `crossover(return_reason=True)` separates "t1 stayed up" from "t2 started up".

## CENSORING — a null crossover is a RESULT, not missing data
If the curves never cross, task 1 stayed above task 2 for the whole window, which can only
happen if **forgetting ran slower than learning throughout**. That is the best outcome
available, and dropping it removes each rule's *best* seeds — more of them from the rule that
is winning. Replay is censored on **11/24** seeds in 60 and 3/5 in 52/53; PC and EqProp never.

`metrics.paired_sign` ranks censored runs at the top and uses every seed. Replay vs backprop on
crossover: **24W–0L, p<0.001** in 60, where the parametric number used 13 of 24. So crossover
stays quantitative *as a comparison* under censoring; what it loses is the height.
**Floor: at 5 seeds a clean sweep gives p=0.0625, so censorable metrics need ≥6 seeds.**

Class-IL is not censored — 56/57 are 5/5 finite for every rule, so PC's +1.29 and +3.00 stand
as computed. But 56 is 5W–0L across seeds and **57 is only 4W–1L (p=0.375)**, so the
matched-competence version is the weaker of the two.

## 60 — THE SEED SPREAD IS EXPLAINED, AND IT IS FORGETTING, NOT LEARNING
24 seeds, Domain-IL, 52's settings. Backprop retains **23.0–71.0%** — a 48-point range.

| backprop | r vs pairing similarity | p | range |
|---|---|---|---|
| **retaining** task 1 | **+0.787** | <0.001 | 23.0–71.0% |
| learning task 2 (final) | +0.164 | 0.45 | 82.2–95.4% |
| learning task 1 (peak) | +0.155 | 0.47 | 84.8–96.6% |
| *speed* to 80% on task 2 | −0.354 | 0.088 | reached **24/24** |

Which two digits **share an output unit** predicts retention: +0.787 (backprop), +0.779 (pc),
+0.630 (replay). The control — same digits, same split, pairing ignored — gives −0.25 to +0.06,
all p>0.23, and **+0.7 points/sd against the pairing's +9.7**, so it is null in effect size and
not merely range-restricted. ~62% of the seed variance. `[EMPIRICAL]`

**The pairing is a pure interference term, not a learnability term.** Nothing ever failed to
learn; the speed correlation has the opposite sign to "dissimilar pairs are harder" and is not
significant. So the network learns both tasks equally well whatever the pairing, and only their
*coexistence* differs — which is what keeps the effect unconfounded with capacity.

**PC matches backprop's dependence on the data, not just its mean**: slope +9.5 vs +9.7 pts/sd.
A much stronger null than "not separated at 5 seeds". Replay is half as sensitive (+4.0), as a
buffer should be.

## 63 — THE LAST DOCUMENTED DIVERGENCE FROM [R1] IS CLOSED
PC's inference step rule: ours (fixed step) vs theirs (`x_lr_discount=0.9`, shrink the step when
the energy fails to fall). Measured at three levels, because a null at the outcome level means
nothing if the intervention never did anything at the state level.

| | at init | after task 1 |
|---|---|---|
| same fixed point? relative gap | 9.4e-05 | 5.3e-06 |
| distance from settled at k=1 | 2.8e-01 | 4.7e-02 |
| **at k=50, the protocol's setting** | **3.2e-04** | **1.9e-05** |
| at k=200 | 0.0 | ~0 |

`cos(ΔW_fixed, ΔW_backtracking) = 1.000000` at **every** step count tried (1–200). Over the full
metric grid on 5 paired Domain-IL runs the largest disagreement is **1.39e-03 accuracy points**.

**Equivalent to within float noise, as it should be:** backtracking changes the route to the
fixed point, not the fixed point, and the weight update is computed from the settled state. This
also independently confirms script 50's calibration — 50 steps leaves PC 3e-04 from equilibrium
at its worst. Note the relaxation gets *easier* once trained (4.7e-02 vs 2.8e-01 at k=1), as 50
found. `[SETTLED]`

**What stays open is the other direction.** If [R1]'s own fixed step count leaves *their*
relaxation short of equilibrium, their operating point is not the equilibrium and "prospective
configuration" would be a partially-relaxed state rather than a settled one — a different
algorithm wearing the same name. 63's panel 1 is the instrument for that; we do not have their
configuration.

## Then
1. **Class-IL is where the gaps are.** Depth is tested in Domain-IL (59, 8/8 cells) and
   **untested in Class-IL**, which is the only place PC shows anything. Likewise 42/43's
   suppression-vs-drift decomposition ran **backprop only** — PC's Class-IL forgetting has
   never been decomposed the same way, and that is the measurement the [HYPOTHESIS] below needs.
3. **Two untested schedule axes, both cheap** (`protocol.run` takes `tasks=`): an
   alternating/repeated schedule ([R1] Fig 4d) and **concept drift** ([R1] Fig 4f–g, where
   their largest advantage is claimed — not cycle length).

## Next three
1. **Decide what the thesis argues.** The A series answers the original question negatively and
   consistently. See "Where this leaves the project" below.
2. **B series** — mostly already answered; it is a write-up, not four experiments. See below.
3. **64/65** — the two mechanism metrics, running.

## Where this leaves the project — THE FRAMING, agreed 2026-08-13
**The thesis is about the character of forgetting and how to measure it. The four learning
rules are the instrument, not the subject.** They earn their place because they produce
genuinely different credit assignment (54: cos 0.197 EqProp, 0.985 PC on W1) while producing
the *same* forgetting — which is exactly what makes "different rule" and "different forgetting"
separable at all. On that framing the PC null is a controlled demonstration, not a
disappointment.

**Say "under a controlled protocol PC shows no retention advantage", never "S&B do not
replicate".** We have not run their setup in this series — the reproduction is old 30–34, it
failed, and it is closed. Our protocol differs from theirs in schedule, resolution, activation,
depth, batch size and loss reduction. A failed replication invites "you did it wrong"; a
controlled comparison invites "then where does their advantage come from?", which is the
question we can actually answer.

The question the project set out to answer — *do energy-based rules forget less than backprop?*
— now has a clear answer across 5 experiments, 2 scenarios, 2 measurement points, 4 depths and
2 widths: **no.** PC is indistinguishable from backprop; EqProp is worse — and **worse on every
seed**, 0W–5L on crossover in 53, 56 and 57, which is a positive finding rather than a null.
Replay separates everywhere, so the problem is solvable and they fail at something achievable.

**[HYPOTHESIS] — the trade-off is set by the relative magnitudes of the two tasks' output
contributions.** Untested, and testable with machinery that exists: `probes.output_unit_stats`
already returns per-unit mean raw scores, and 64's interference probe measures per-update
movement on task-1 data during task 2. A readout question, not new machinery.

That is a result, not a dead end, and it is stronger than a vague null because:
- **The rules genuinely differ.** 54 measured EqProp's output-layer update as nearly orthogonal
  to backprop's (cos 0.197) and PC's as 0.985-aligned on W1. So credit assignment changes
  without the retention/acquisition trade-off changing — the interesting question is *why those
  come apart*, which is what E's alignment and weight-path probes were always for.
- **Every controlled comparison is gated.** Matched task-1 competence, matched task-2
  competence, a separating positive control, and paired statistics. Each gate exists because
  its absence produced a wrong answer earlier in the series.
- **The scenario, the depth, the width, the measurement point and the output standardisation
  have each been eliminated** as explanations for the contradicting prior result (exp 12).

## 59 — the final A-series run. Depth and width do NOT rescue PC.
Depth {1,2,3,5} × width {32,128}, learning rates calibrated **per cell**, both tasks stopped on
accuracy with the task-2 threshold measured per cell, paired against backprop. All 8 cells
passed the validity gate (task-2 spread ≤ 1.2 points) and the positive control separated in
**8/8**.

| PC − backprop | 1 layer | 2 | 3 | 5 |
|---|---|---|---|---|
| **32 units** | +1.6 (0.9σ) | +1.8 (1.2σ) | +0.6 (0.4σ) | **−4.6 (2.3σ)** |
| **128 units** | +0.7 (0.8σ) | +1.6 (1.5σ) | +2.5 (1.4σ) | +0.2 (0.3σ) |

Replay over the same cells: +9.5 to +16.9, separated everywhere.

**PC never beats backprop in any cell.** The only cell where it separates at all is 5×32, where
it is *worse* — the deepest and narrowest, i.e. the most capacity-constrained. **H=32 was not
masking an effect**: at 128 units, four times the capacity, PC is +0.7 to +2.5 and never
separates. `[EMPIRICAL]`

With 52, 53, 56 and 57 this closes the A series: **PC does not reduce forgetting in this task
family at any depth or width tried, under either scenario, under either measurement point.**

## The single most important methodological finding
**Compare paired, per seed.** Every rule sees the same class split and initialisation at a given
seed, so the between-seed variance is shared and comparing group means throws it away. On
script 53's runs, group means put **the positive control at 0.7σ** and it read as a failure;
paired, the same runs give 5.1σ. Five seeds were always enough — the statistic was wrong.
`metrics.paired_diff` now carries this, with the numbers, in its docstring.

## The B series is smaller than planned
- **B1 — do standard metrics work here?** Answered. Group-mean endpoint retention cannot detect
  the positive control (0.7σ) where the paired difference gives 5.1σ. And in Class-IL at a fixed
  budget everything lands at 0–7%, so endpoint retention cannot rank anything at all.
- **B3 — where do we compare?** Answered. Fixed budget vs matched competence *flips PC's sign*.
- **B2 — rate metrics.** half-life and crossover height are already computed in 52/53/56/57.

So B is a write-up of evidence in hand. The one genuinely new figure worth building is the
paired-vs-unpaired demonstration — the strongest methodological point the project has.

## Fixed 2026-08-12
- **53's non-smooth curves were a plotting bug, not learning.** NaN-padded ragged runs meant
  `nanmean` averaged over a *varying number of seeds*, so the mean jumped as each run entered
  or left the switch-relative axis. `plotting._mean_of_all` now draws the mean only where every
  run is present. The `Mean of empty slice` warning was the symptom and was ignored.
- **53's task-2 threshold is now derived, not assumed.** 90% was unreachable — replay missed it
  on 4/5 seeds — so most runs were never at matched competence and the run was void.
- **Replay was training on a double batch.** `replay_frac` defaulted to `None`, appending the
  replay batch so replay saw 64 examples per step against everyone else's 32 — a confound
  `make_replay`'s own docstring names. Now `0.5`, batch size held fixed.
- **Trajectory plots: one axes, not four panels**, averaged against task-2 accuracy so ragged
  runs are comparable. The equal-trade-off diagonal is gone — it belongs to Class-IL; in
  Domain-IL the tasks share output units so sitting above it is expected and says nothing.
- **`--smoke` writes to `_SMOKE` paths.** A smoke run of 52 had overwritten the real `.npz`
  that 53 reads its threshold from.

## Settings now fixed, and where they came from
| | value | from |
|---|---|---|
| scenario | Domain-IL primary, Class-IL where relevant | decision, 2026-08-11 |
| hidden width | H = 32 | 41 |
| joint ceiling | 93.6% Class-IL, 94.3% Domain-IL — retention reads against these, not 100% | 41 |
| learning rates | backprop 0.01, replay 0.01, pc 0.02, eqprop 0.01 | 51 |
| matched at | ~420 updates to 90% on task 1; residual spread 1.25× | 51 |
| PC settling | 50 steps fixed (needs ≤ 18) | 50 |
| EqProp settling | `settle_tol = 1e-4` (needs ≤ 529 at init) | 50 |
| chance | 20% Domain-IL, 10% Class-IL. Below chance = output-unit capture, not lost knowledge | 40 |

## Checklist
- [x] **40** does backprop forget → 93.8% → **0.4%**, below the 10% floor. Relearning
      accelerates 320 → 110 → 60 updates. Ran at provisional H=64; **re-run at H=32.**
- [x] **41** capacity → **H = 32**. Both scenarios flatten at the same width.
- [x] **42** masked × frozen, read at saturation → everything reads 0. **Misleading; kept as
      the counterexample.**
- [x] **43** the same, read at matched competence → the correction. This pair is the argument.
- [x] **50** settling → PC fully settled; EqProp's stopping rule was truncating at init, fixed.
- [x] **51** learning-rate calibration → matched to 1.25×.
- [x] **52** four-rule comparison, fixed budget → **nothing separates except replay.**
      Retention: backprop 51.0±6.3, replay 77.3±3.0, pc 48.5±6.6, eqprop 43.1±5.5. PC is 0.3 SE
      from backprop, EqProp 1.0 SE, replay 3.7 SE.
- [~] **53** four-rule comparison, accuracy stopping → **void, re-run needed.** Task-2 threshold
      set to 90% without checking it was reachable; replay missed it on 4/5 seeds and EqProp on
      3/5, so most runs were never read at matched competence. Set the threshold from what is
      achievable, as 43 did.
- [x] **54** are the EBMs really settling → **yes, both. Neither is backprop.** But PC's hidden
      update is 0.985 aligned with backprop's, at every settling amount ≥ 1 step. EqProp 0.316.
- [x] **55** first depth pass (superseded by 59) — does PC need depth → **no.** Retention curves for PC and backprop are superimposed
      at depths 1, 2 and 3. Divergence grows toward the output (W4 0.623) while W1 stays
      backprop-like (0.952) — and W1 is the layer whose drift damages task 1.
- [x] **56/57** Class-IL, both readings -> same verdict as Domain-IL
- [x] **58** legacy spec -> cannot answer exp 12; backprop never learns task 2 there
- [x] **59** depth x width -> **PC never beats backprop in 8/8 cells**
- [ ] **B1/B2/B3** metrics, largely a write-up of evidence already in hand
- [ ] **C2** six-cell factorial, **C1** NCM figure
- [ ] **D** controlled comparison, then **E** why, then **F** does it generalise

## THE EXPERIMENT-12 MYSTERY IS SOLVED — its positive control fails
**61 re-ran experiment 12 verbatim** (100 updates per task, its own learning rates, legacy
per-rule spec, 10 pairings). Crossover height, the metric 12 was read on:

| | crossover | vs backprop | final t1 | final t2 |
|---|---|---|---|---|
| backprop | **64.0%** | — | 54.4% | 73.6% |
| pc | 59.5% | −4.52 (3.5σ) | 9.6% | 84.8% |
| replay | 48.7% | **−14.14 (5.2σ)** | 32.9% | 79.3% |
| eqprop | 32.4% | −30.59 (27σ) | 2.2% | 68.0% |

**REPLAY — THE POSITIVE CONTROL — COMES OUT WORSE THAN BACKPROP.** At 100 updates per task
nothing has converged: backprop is simply the slowest to learn task 2 (73.6%) and so has
displaced the least of task 1. Every ordering at that budget is dominated by task-2 learning
speed, not by forgetting. **A setup where the positive control inverts cannot detect a
forgetting difference at all**, so exp 12's result is not evidence about PC — and no bug in
`src` is needed to explain it. With a proper budget the same code gives replay +7.32 crossover
(9.7σ) in script 56.

One thing 61 *does* reproduce is the tension remembered from 12's plots: **PC beats replay on
crossover (59.5 vs 48.7) while losing badly on final task-1 accuracy (9.6 vs 32.9)** — which is
exactly why one metric is never enough.

## PC's small advantage is CLASS-IL SPECIFIC, and that is mechanistically coherent
Paired crossover vs backprop:

| | Domain-IL | Class-IL |
|---|---|---|
| fixed budget | −0.01 (0.0σ) | **+1.29 (3.7σ)** |
| matched competence | +0.49 (2.0σ) | **+3.00 (2.2σ)** |
| **59, all 8 depth × width cells** | **−1.75 to +0.52, never separated** | not run |

Replay separates in all 8 of 59's cells (+2.24 to +4.42), so those cells are sound.

**The story that ties 54, 55, 56/57 and 59 together:** PC's update diverges from backprop most at
the OUTPUT layer (54: cos 0.197 for EqProp, 0.814 for PC on W2; 55: 0.623 at W4 by depth 3) and
barely at all on W1 (0.985→0.952). Output-layer *suppression* is the dominant forgetting
mechanism in Class-IL (42/43) and **cannot occur in Domain-IL**, where every unit is a target for
some class. So PC helps a little exactly where the output layer is the problem, and not at all
where it isn't. `[HYPOTHESIS]` — coherent with everything measured, not yet tested directly.

## THE RESULT — paired difference in task-1 retention vs backprop, 5 seeds
Every rule sees the same class split and initialisation at a given seed, so the comparison is
**per seed**. σ = standard errors of the paired difference.

| | Domain-IL fixed | Domain-IL matched | Class-IL fixed | Class-IL matched |
|---|---|---|---|---|
| **replay** | +26.2 (6.5σ) | +11.6 (5.1σ) | +57.4 (17.2σ) | +41.4 (8.1σ) |
| **pc** | −2.6 (2.7σ) | +2.5 (1.3σ) | −2.0 (4.0σ) | +1.6 (0.7σ) |
| **eqprop** | −8.0 (4.9σ) | −2.9 (1.2σ) | −6.6 (3.3σ) | −13.9 (3.4σ) |

**Neither energy-based rule reduces forgetting, in either scenario, under either measurement.**
PC is within ±2.6 points of backprop everywhere — never better. EqProp is consistently worse.
Replay separates hugely in all four, so the problem is solvable and they fail at something
achievable. `[EMPIRICAL]`

**The scenario is not what explains exp 12.** Class-IL gives the same PC verdict as Domain-IL.

PC's sign flips between fixed budget (−) and matched competence (+): at a fixed budget it
learns task 2 further so it forgets more. Small either way, but it is the 42/43 lesson
reappearing inside the rule comparison.

## What we know
- **PC trades task 1 for task 2 along the SAME CURVE as backprop, at depths 1, 2 and 3 (55).**
  The retention curves — task-1 accuracy against task-2 accuracy, which removes time — are
  superimposed. This is a much stronger null than 52's "not separated at 5 seeds": the curves
  coincide along their whole length rather than at one sampled point, and it is robust to the
  uncalibrated per-depth learning rates, because a different rate moves a rule *along* the
  curve without moving the curve. **Depth does not rescue PC.** `[EMPIRICAL]`
- **Beware 55's endpoint table, which says the opposite.** It shows PC keeping 54.9% against
  backprop's 40.8% at depth 3 — but PC took 800 updates to reach competence on task 1 where
  backprop took 370, and reached 79.9% on task 2 against 85.9%. It is slower at an uncalibrated
  learning rate, so it simply travelled less far along the shared curve. Textbook "forgot less
  = learned less". The endpoint would have sold this as a win.
- **PC's divergence from backprop grows with DISTANCE FROM THE INPUT, not with depth (55).**
  cos(ΔW) at depth 3: W1 0.952, W2 0.939, W3 0.837, W4 0.623. The input mapping stays
  backprop-like at every depth; the output-side layers pull away. Since 42/43 established that
  **drift in W1 is what damages task 1**, PC is reconfiguring the end of the network and leaving
  the part that matters alone — which predicts exactly the null the retention curves show.
- ~~At one hidden layer PC ≈ backprop because there is no intermediate layer to reconfigure~~ —
  **wrong**, and it was my inference from 54, not a measurement. It predicted cos(ΔW1) would
  fall with depth. It does not (0.987 → 0.952). See the two entries above for what replaced it.
- **EqProp genuinely differs (cos 0.316 on W1) and still gained no retention.** So different
  credit assignment is not by itself sufficient. Its settling sweep is **non-monotonic** — most
  backprop-like at ~20 steps, least at full relaxation — so it is the *full* relaxation that
  makes it different, as [R1] claims for prospective configuration.
- **Both rules do settle.** Relaxation moves the state 19% (PC) and 10% (EqProp) at the switch,
  against 0% if they were not settling. Instrument verified: PC at `steps=0` reproduces
  backprop's W2 update exactly (cos 1.0000) and leaves W1 untouched, as `pc_settle` predicts.
- **No energy-based rule beat backprop at fixed budget (52).** Neither PC nor EqProp is
  distinguishable from backprop on final retention; only replay is. Read this as *no detected
  effect*, not as *no effect* — see the variance point below. `[EMPIRICAL]`
- **Seed variance is the binding constraint, not the effect size.** ±6 points SEM on a ~50%
  retention mean, with all four rules matched to ±0.8% at the switch — so the spread arises
  after the switch, not from unequal starting competence. Detecting a 6-point difference needs
  roughly 4× the seeds.
- **Area retained is less noisy than endpoint retention.** EqProp vs backprop: 1.0 SE on final
  retention, **2.2 SE on area**. Integrating the curve beats sampling its last point — an
  argument for the metrics work that arrived from the data rather than from the literature.
- **Task 1 is still falling when the budget ends.** The endpoint is not a resting place, so
  under a fixed budget "how much was forgotten" is partly "how long did we run". Visible
  directly in 52's figure; it is why 53 exists.
- **The information survives.** NCM (classify from the hidden layer, output layer discarded)
  holds ~80% on task 1 while the network's own prediction reads 0.2%. With the hidden layer
  frozen, NCM is flat by construction and the prediction still collapses — so the collapse is
  entirely in W2.
- **Two mechanisms, different durations.** Output-layer suppression is large and permanent.
  Representation drift is real but transient — it changes the *rate*, not the asymptote.
  Freezing W1 doubles the half-life and is worth +31 points at 50 updates post-switch, and 0
  at 2000. **Which one looks dominant is decided by when you stop the clock.**
- **Masking ≈ freezing the task-1 columns of W2.** Under squared error the gradient at output
  unit *j* is (target_j − out_j)·h; masking zeroes it for absent *j*, as does freezing column
  *j*. But **masking needs task identity and W2 freezing does not**, so `masked + frozen` is an
  oracle ceiling, not a method.
- **Output suppression cannot occur in Domain-IL** — every unit is a target for some class.
  That is why Domain-IL is primary: it leaves representation drift, which is what a rule acts on.
- **Capacity is never a confound downstream.** H=32 is past the knee in both scenarios.

## THE METRIC GRID — every metric, what it targets, what it is blind to
Reported in FULL on every comparison (`metrics.metric_grid` + `metrics.report_grid`), so a claim
cannot be made by quoting whichever metric happens to favour it. Each is blind to something and
the blind spots do not overlap.

| metric | targets | blind to | impl | running |
|---|---|---|---|---|
| **crossover height** | the accuracy where the two task curves meet — the joint trade-off point | the asymptote; **undefined if they never cross — but see censoring, that is a result** | ✅ | ✅ **primary** |
| final task-1 acc | what survived at the end | *when* the end was — set by budget or threshold | ✅ | ✅ descriptive |
| final task-2 acc | the guard: is "forgot less" really "learned less"? | task 1 entirely | ✅ | ✅ gate |
| peak task-1 acc | competence entering task 2 | everything after the switch | ✅ | ✅ gate |
| half-life | *rate* of forgetting | the asymptote; undefined if it never halves (replay) | ✅ | ✅ |
| area retained | integrated retention — uses the whole curve, so less noisy | the integral runs to wherever training stopped | ✅ | ✅ |
| **mean test error during training** — [R1]'s headline | combined: learning fast *and* forgetting little | **cannot separate those two**; a fast learner wins it without forgetting less | ✅ new | ✅ |
| ACC (Lopez-Paz & Ranzato 2017) | mean final accuracy over tasks | endpoint → the budget | ✅ new | ✅ |
| BWT (Lopez-Paz & Ranzato 2017) | final task-1 minus task-1 at the switch | endpoint → the budget | ✅ new | ✅ |
| forgetting (Chaudhry 2018) | peak task-1 minus final task-1 | endpoint → the budget | ✅ new | ✅ |
| savings / relearning speed (Ebbinghaus; Hetherington) | residual knowledge accuracy cannot see | needs a **third block** to relearn in | ⚠️ ad hoc in 40 | ❌ |
| NCM — nearest class mean | is the class information still in the hidden layer? | output-layer calibration | ✅ `probes.live_ncm_fn` | ⚠️ 42/43 only |
| target alignment ([R1] Fig 3b) | per-update interference *direction* | magnitude | ✅ `probes.alignment_probe` | ❌ |
| inefficiency ([R31]) | weight path ÷ net displacement | **not a forgetting metric** — belongs to "why do rules differ" | ✅ `metrics.inefficiency` | ❌ |
| *paired difference* | *the comparison statistic, not a metric* | — | ✅ | ✅ everywhere |

**Song & Bogacz's metric is now measured.** [R1] report the *mean test error over training*, not a
final accuracy — so their claim is made in that number and it has to be tested in it. It is
included, but never alone: it rewards fast learning and low forgetting together and cannot tell
them apart, which is precisely the separation this project exists to make.

**A METRIC'S USABILITY IS SCENARIO-DEPENDENT, and it is the same structural fact both times.**
Re-reported across all runs on 2026-08-13 with censoring counts:

| | Domain-IL | Class-IL |
|---|---|---|
| crossover, replay | censored 3/5 (52), 3/5 (53), **11/24** (60) | **5/5 defined** (56, 57) |
| half-life, backprop | **2/5** (52), **1/5** (53) — unusable | 5/5 (56), 5/5 (57) |
| half-life, replay | 0/5 | 0/5 |

In Domain-IL forgetting is routinely *slower than the metric's own reference event* — task 1
often neither halves nor falls below task 2 — so both metrics censor. In Class-IL task 1
collapses and both are defined. **Half-life should not be quoted for Domain-IL at all**; it is
computed on 1–2 seeds there. This is not two problems, it is one: the reference event is chosen
from the Class-IL picture of forgetting and does not occur in Domain-IL.

Also visible now in 53's line: replay's crossover *group mean* (72.34) is below backprop's
(72.57) while the *paired* difference is +2.75 — because the group mean is over replay's 3
uncensored, i.e. worst, seeds. Exactly the defect A5 fixed.

**Open gaps:** savings needs a 3-block schedule; NCM should run on the rule comparisons, not just
the intervention ones; alignment and inefficiency are running now (64, 65).

## Metrics — what we report, and why
The problem has a name: **setup-induced forgetting** (Michel et al. 2023, arXiv:2309.00462).
Every standard metric is evaluated at the end of training, so its value is set by how long
task 2 ran. In 2×5 Class-IL every condition reaches 0, so endpoint metrics rank nothing.

| metric | what it is | status |
|---|---|---|
| ACC, BWT, forgetting | standard endpoint metrics | report them, **and show they degenerate**. Lopez-Paz & Ranzato 2017; Chaudhry et al. 2018 |
| task 1 at matched task-2 accuracy | the headline | separates "retained more" from "learned less" |
| crossover **height** | the *accuracy* at which the curves intersect | budget-independent. **Not** *when* they cross — that is confounded by learning speed. Undefined when they never cross, which is itself the strongest thing it can say |
| half-life | updates to lose half the pre-switch peak | covers the case crossover cannot. Kept after being wrongly proposed for removal |
| savings | how fast old material is relearned | **established** — Ebbinghaus 1885, Hetherington for networks. Detects residual knowledge accuracy cannot see. Script 40 already measured it |
| target alignment | cos(target − out_before, out_after − out_before), [R1] Fig 3b | on task-1 data *during* task 2 it gives per-update interference, budget-independent by construction |
| inefficiency [R31] | path length ÷ net displacement | **not a forgetting metric.** Belongs to "why do rules differ" |

Also Díaz-Rodríguez et al. 2018 (arXiv:1810.13166) for the wider set.

## Decisions
- **2026-08-11 — EqProp stays in through the four-rule comparison, then has to be re-earned.**
  Only called in once PC results are established *and* there is a specific reason. It costs
  ~350× backprop per update and that is intrinsic.
- **2026-08-11 — Domain-IL is primary.** It is what Song & Bogacz use, and it removes
  output suppression, which is rule-independent. Class-IL is run where relevant, and its result
  explains why it is a different question rather than being avoided.
- **2026-08-11 — learning rates matched on updates-to-90% on task 1**, never on the crossover:
  that is the dependent variable. Target 420 chosen after seeing the grid — safe only because
  it is common to all four rules and measured before task 2 exists.
- **2026-08-11 — accuracy-stopped plots keep a step axis**, switch at x=0, curves **not**
  stretched to fill the panel. Stretching destroys the rate, and rate is half of what we
  measure. Trajectory plots handle the ragged lengths instead.
- **2026-08-11 — trajectory plots draw the whole path from initialisation**, not from the
  switch. Verified separately that an untrained net sits at 16–31% on both tasks against 20%
  chance, so initialisation is sound.
- **2026-08-11 — `settle_tol = 1e-4`**, calibrated inside script 50. Not a universal constant:
  re-run 50 if `dt`, width or depth change. Calibrate on *distance from the settled state*,
  never on the step count picked.
- **2026-08-11 — exp 12's learning rates are void**, not approximate. Set under the legacy
  per-rule specification (ReLU + cross-entropy vs hinge on ±1 targets).
- **Standing — no prior result is evidence.** Everything before the 40s used inconsistent
  setups on untrusted code. Prior work informs direction and experiment design only.

## Corrections — recorded so they are not re-derived
- ~~Freezing the hidden layer recovers nothing~~ — an artefact of measuring at saturation. It
  recovers 17 points at matched competence.
- ~~No room for a learning rule to act~~ — a narrative built on that artefact. Withdrawn.
- ~~Every rule receives the same output error, so none can change the output-layer update~~ —
  false. `pc_update` computes the error from the **settled** hidden state
  (`predictive_coding.py:114-118`). Slides may only claim a rule cannot change the loss or the
  target coding.
- ~~Freezing W2 makes task 2 unlearnable~~ — false; networks train with fixed output layers.
- ~~A short-budget capacity sweep measures capacity~~ — it produces a fake plateau
  indistinguishable from a ceiling. Measure convergence first.
- ~~The settling requirement grows as weights grow~~ — it falls (PC 17→8, EqProp 298→57).
  The trend is tolerance-dependent and not citable; the decisions do not rest on it.
- ~~EqProp's patience rule was merely over-conservative~~ — it also **truncated**, to 0.34× of
  what was needed, at initialisation.

## Proposed 60 — why does the SEED matter so much?
The non-convergence that prompted this was a threshold artefact, not a seed property: at a 90%
task-2 threshold replay missed on 4/5 seeds, and once the threshold was measured under the
right cap (80%) **every rule reached it on 5/5**. Do not go looking for seeds that "fail".

**But the underlying observation is real and large.** Backprop retained 38.2% on seed 2 and
78.0% on seed 3 — a 40-point range, bigger than any effect we are trying to measure. It is
shared across rules, which is why pairing rescues the comparison, but nothing explains it.

The obvious candidate, and it is testable: **which classes get paired onto the same output
unit.** Under Domain-IL `label_map` sends the i-th class of each task to unit i, so each unit
carries one task-1 digit and one task-2 digit, paired by the per-seed permutation. If a unit
carries two similar digits, the task-2 mapping may partly reuse the task-1 feature; if it
carries two dissimilar ones, they compete.

Questions, in the order they should be answered:
1. Does pairwise digit similarity across the five units predict retention? Cheap — correlate
   retention against a similarity measure over the existing seeds, no new training.
2. If it does: is a bad pairing *slower* to resolve, or does it settle at a worse place? A rate
   question, so read it off the retention curve, not the endpoint.
3. Where does the damage land — hidden layer or output? The **NCM probe** answers this directly
   and already exists (`probes.live_ncm_fn`): NCM high with argmax low means the hidden code
   survived and the output layer is at fault.

Worth doing because a 40-point nuisance term is larger than the effect under study, and because
question 3 connects it to the C1/C2 mechanism work already planned.

## The gap 56/57 do not close
Exp 12 differed from script 52 in **three** ways, not two: the scenario, the standardised
output structure, and the matched learning rates. 56/57 change only the scenario. **If exp 12's
PC advantage came from the legacy specification** — where backprop ran ReLU + cross-entropy and
EqProp ran a hinge over ±1 targets, so every rule had a different nonlinearity *and* a different
loss — then no run under the unified protocol can reproduce it, and 56/57 can neither confirm
nor rule it out. The clean test is a deliberate legacy-spec run through `methods.legacy()`.
Cheap, and it would settle the discrepancy rather than leaving it open. Not yet run.

## Archive — planned, not built. Revive with a new 50-series number if needed.
- **Capacity-dependence of the mechanism split (C3).** Does W2-suppression vs W1-drift shift
  with width? Old exp 23 hints the freeze-W1 effect is large at H=2 and near zero at H=64.
  Deferred, real question, off the critical path.
- **Depth.** One hidden layer means only one place drift can occur. The depth run is the test.
- **Harder data.** Is MNIST complex enough? Pinchetti et al. 2025 report PC matching backprop
  at small scale and losing at large. Fashion-MNIST is already wired (`Protocol.fashion`).
- **Class-IL as an explicit axis.** Currently run only where it explains a difference.
- **PC/EqProp settling unified to one mechanism.** Both verified to reach equilibrium, so this
  is exposition only — one sentence on a slide instead of two.
- **β for EqProp.** The one lever with real leverage on its cost (bias ~42% at β=0.3 vs ~1.6%
  at β=0.005, and a smaller β also cheapens the nudged phase). Not opened.

## Closed, do not reopen
- **Song & Bogacz reproduction** (old 30–34). Cost hours, failed, cause never found.
  `experiments/archive/`. **Do not re-run.**
- **EBM as a replay generator** (old 04–06). `experiments/archive/ebm_replay_generation/`.
- **wandb, Optuna, class hierarchies, a shared `harness.py`.** All built, all deliberately
  deleted. Do not reintroduce.
