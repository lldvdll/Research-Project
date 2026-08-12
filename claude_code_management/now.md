# Now

Dashboard. One line per fact. Detail lives in `current_state.md` and in each script's
docstring — keep this file short enough to read in one sitting.

**Running:** 52 (fixed budget) and 53 (accuracy stopping), in parallel. EqProp-bound.
Started 2026-08-11.

## Next three
1. Read **59** when it lands — depth × width, the last A-series run and PC's best chance.
2. **B series is mostly already answered** — see below. It is a write-up, not four experiments.
3. **60** — why the seed matters so much (see below). A 40-point nuisance term.

## Running
| script | question | status |
|---|---|---|
| 59 | depth {1,2,3,5} × width {32,128}, per-cell LR calibration, paired | running, hours |

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
- [x] **55** does PC need depth → **no.** Retention curves for PC and backprop are superimposed
      at depths 1, 2 and 3. Divergence grows toward the output (W4 0.623) while W1 stays
      backprop-like (0.952) — and W1 is the layer whose drift damages task 1.
- [ ] **B1/B2/B3** metrics, from one set of backprop runs
- [ ] **C2** six-cell factorial, **C1** NCM figure
- [ ] **D** controlled comparison, then **E** why, then **F** does it generalise

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
