# Now

Dashboard. One line per fact. Detail lives in `current_state.md` and in each script's
docstring — keep this file short enough to read in one sitting.

**Running:** 52 (fixed budget) and 53 (accuracy stopping), in parallel. EqProp-bound.
Started 2026-08-11.

## Next three
1. Read 52 and 53. Controls first: did replay recover, did every rule reach the threshold.
2. **B1/B2/B3** — metrics, all three from one set of backprop runs.
3. Decide whether EqProp continues past 53.

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
- [ ] **52** four-rule comparison, fixed budget ← running
- [ ] **53** four-rule comparison, accuracy stopping ← running
- [ ] **B1/B2/B3** metrics, from one set of backprop runs
- [ ] **C2** six-cell factorial, **C1** NCM figure
- [ ] **D** controlled comparison, then **E** why, then **F** does it generalise

## What we know
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
