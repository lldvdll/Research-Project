# Current state — 2026-08-11

Written before a context compression. This is the working record: what is established, what was
got wrong, what is decided, and what happens next. Read this before `presentation_plan.md`, which
is now partly out of date (see §7).

---

## 1. Results established

All under: MNIST 14×14 (196 inputs), 2 tasks × 5 classes, one hidden layer, tanh, linear output,
squared error, one-hot targets, plain SGD, batch 32, 5 seeds, class split drawn per seed.

### 41 — capacity. **H = 32.** `[SETTLED]`
Joint training to convergence, width swept 2→256, both scenarios.

| | best | at H | 16→32 | 32→64 | 128→256 |
|---|---|---|---|---|---|
| Class-IL | 93.6% | 128 | +1.4 | +0.5 | −0.0 |
| Domain-IL | 94.3% | 128 | +1.0 | +0.4 | −0.1 |

Both scenarios flatten at the same width and agree on H = 32. **Capacity is never a confound in
any later experiment.** Joint ceiling is the number retention must be read against — not 100%.

Domain-IL beats Class-IL by 29 points at H=2 and the gap closes by H=16: a small-network artefact
of 5 output units vs 10, not a scenario effect. Their chance levels differ (20% vs 10%), so
absolute accuracies are **not comparable across scenarios**.

### 50 — do the settling loops converge? `[SETTLED]`
Domain-IL, H=32, 3 seeds, measured at 0 / 50 / 200 / 500 updates into task 1. "Settled" = the
hidden state is within 2% of where a deliberately over-long relaxation puts it. Validity gate
passed: movement over the final reference step was 4.9e-10 (PC) and 1.4e-8 (EqProp).

| | needs, at init | needs, trained | what the rule does |
|---|---|---|---|
| PC | 17 (worst 18) | 8 | 50 steps, fixed → **over-settled 3–6×** |
| EqProp | 298 (worst 529) | 56–74 | see below |

**PC is not a truncated relaxation**, so the worry that its fixed step count makes it a
backprop approximation (Millidge 2022) does not apply at this size. This closes the open
question in §5. It has not been checked at greater depth.

**EqProp's stopping rule was wrong in both directions and has been replaced.** It stopped once
per-step movement failed to improve for 30 steps, which (a) fired on a temporary plateau — at
initialisation, seed 2 needed 529 steps and it stopped at **178, i.e. 0.34× of what was
needed** — and (b) overshot by 3–4× once trained, because movement decays long after the state
arrives. Replaced by a relative-residual test, `move ≤ settle_tol·‖state‖`.

`settle_tol = 1e-4`, **calibrated inside script 50, not chosen by hand**: it is the largest
tolerance whose worst-case stopping point is within a 2× safety factor of "settled" (0.57%
against a 2% budget). The tolerance is **not dimensionless** — it scales with `dt` and with the
energy's curvature — so changing dt, width or depth requires re-running 50, which re-derives it.
Calibrate on *distance from the settled state*, never on the step count the criterion picks:
step counts are grid-discretised and the distance falls steeply near convergence, so a
step-count test is over-sensitive exactly at the decision boundary. It rejected 3e-4 as
truncating when 3e-4 in fact stops 1.5% out, inside the 2% budget.

**Cost: unchanged.** 324 ms/update against ~344 ms before — 1.06×. The nudged phase got *more*
expensive (191 steps vs 147) because at β=0.3 the nudge moves the fixed point far enough that
the warm start does not help much. This is a correctness fix with no speedup attached. β is a
candidate lever later: [R1]-style bias is ~42% at β=0.3 and ~1.6% at β=0.005, and a smaller β
would also cheapen the nudged phase.

EqProp remains **~350× backprop's cost per update** (324 ms vs 0.93 ms; PC is 8.5 ms). That
sets the budget for every experiment involving it.

### 40 — backprop forgets. `[SETTLED]`
Three alternating cycles, fixed budget per block. Run at provisional H=64, **needs re-running at
H=32.**

- Task 1: 93.8% → 0.4% when task 2 trains. Repeats identically three times.
- **Relearning accelerates: 320 → 110 → 60 updates** to reach 94%.
- Task 1 falls **below the collapse floor** (0.4% against 10%), i.e. task-2 output units win the
  argmax on task-1 images.
- Retention after each task-2 block rises across cycles: 0.4% → 1.8% → 3.1%.

### 42 / 43 — where the forgetting is, Class-IL. `[SETTLED]`
Factorial of loss masking × hidden-layer freezing, backprop only.

**42 measured at saturation (task 2 run 2000 updates):**

| condition | task 1 kept | NCM task 1 |
|---|---|---|
| control | 0.2% | 71.7% |
| masked | 62.2% | 78.7% |
| frozen hidden | 0.0% | 84.2% |
| masked + frozen | 81.8% | 84.7% |

**43, identical but task 2 stopped at 50% accuracy:**

| condition | task 1 kept | crossover | half-life | final t1 |
|---|---|---|---|---|
| control | 35.7% | 59.1% | 32 | 36.3% |
| masked | 73.9% | never | never | 67.6% |
| frozen hidden | 53.1% | 56.3% | 100 | 54.4% |
| masked + frozen | 74.8% | never | never | 80.4% |

**The two headline findings, both robust to the measurement point:**

1. **The information survives.** NCM — classify from the hidden layer with the output layer
   discarded — holds ~80% on task 1 while the network's own prediction reads 0.2%. In the
   frozen-hidden condition the NCM line is *exactly flat* by construction, and the prediction
   still collapses, so the collapse is entirely in W2.
2. **Two mechanisms with different durations.** Output-layer suppression is large and permanent.
   Representation drift is real but transient — it changes the *rate*, not the asymptote
   (freezing doubles half-life, +31 points at 50 updates post-switch, 0 at 2000).

---

## 2. Corrections — things got wrong, recorded so they are not re-derived

- **"Freezing the hidden layer recovers nothing"** — FALSE. An artefact of measuring at
  saturation, where every condition reads 0. It recovers 17 points at matched competence.
- **"No room for a learning rule; Domain-IL is where the action must be"** — a narrative built on
  that artefact. Withdrawn.
- **"Every rule receives the same output error, so no rule can change the output-layer update"** —
  FALSE. `pc_update` computes the output error from the **settled** hidden state
  (`predictive_coding.py:114-118`), so both the presynaptic activity and the error differ from
  backprop. Slides 5–6 may only claim *a rule cannot change the loss or the target coding*.
- **"Freezing W2 makes task 2 unlearnable"** — FALSE. Networks train fine with a fixed output
  layer; the hidden layer adapts to read out correctly through it (Hoffer et al. 2018).
- **Masking ≈ freezing the task-1 columns of W2.** Under squared error the gradient at output
  unit j is (target_j − out_j)·h; masking zeroes it for absent j, as does freezing column j.
  **Masking requires task identity; W2 freezing does not.** Masking is an oracle upper bound.
- **`masked + frozen` is not a method.** It trains a new head on a fixed feature extractor and
  needs task identity. It is a ceiling demonstration.
- **A capacity sweep at a short budget produces a fake plateau** indistinguishable from a real
  capacity ceiling. At 2500 updates all widths ≥8 sat at ~87% and the answer looked like H=16.
  Measure convergence first, then sweep.
- **"Settling requirement grows as the weights grow"** — FALSE, and it was the reading
  committed before script 50 ran. It *falls*: PC 17→8 steps and EqProp 298→57 across task-1
  training. Caveat on the trend itself: script 50's tolerance is relative
  (distance ÷ ‖settled state‖) and ‖settled state‖ grows with the weights, so part of the fall
  may be the normalisation. The *decisions* do not rest on it — PC has 3–6× margin either way,
  and EqProp's mechanism is chosen from the spread, not the trend — but the trend is not
  citable without an absolute-tolerance check.
- **"EqProp's patience rule is merely over-conservative"** — FALSE. It also *truncates*. See §1.

---

## 3. Metrics — the decision and its citations

**The problem, and it has a name.** Every standard metric (ACC, BWT, forgetting) is evaluated at
the end of training, so its value is set by how long task 2 was trained. Michel et al. 2023
(arXiv:2309.00462) call this **setup-induced forgetting** and propose normalising by task
difficulty. Our own case is extreme: in 2×5 Class-IL every condition reaches 0, so endpoint
metrics cannot rank anything.

**What we use, and why:**

| metric | what it is | status |
|---|---|---|
| ACC, BWT, forgetting | standard endpoint metrics | report them, and show they degenerate. Lopez-Paz & Ranzato 2017; Chaudhry et al. 2018 |
| task-1 accuracy at matched task-2 accuracy | read retention when every condition has learned the new task equally well | separates "retained more" from "learned less" |
| crossover height | **the accuracy at which the two curves intersect** — not when they intersect | budget-independent. *When* they cross is a different quantity and is confounded by learning speed; do not report it as "crossover". `metrics.crossover` returns `(step, height)` and scripts take `[1]`. **Undefined when they never cross** (masked conditions) |
| half-life | updates to lose half the pre-switch peak | budget-independent; covers the case crossover cannot. Forgetting-rate tradition, define explicitly, claim no novelty |
| **savings** | how fast old material is relearned | **established metric** — Ebbinghaus 1885, Hetherington for networks. Detects residual knowledge accuracy cannot see. **Script 40 already measured it** (320→110→60) and the literature's "interference diminishes monotonically with repetition" matches our 0.4→1.8→3.1% |
| target alignment | cos(target − out_before, out_after − out_before), [R1] Fig 3b | measured **on task-1 data while training task 2** it gives per-update interference, budget-independent by construction |
| inefficiency [R31] | path length ÷ net displacement | **not a forgetting metric.** Belongs to "why do rules differ", not "how much do they forget" |

Also: Díaz-Rodríguez et al. 2018 (arXiv:1810.13166) for the broader metric set.

---

## 4. Decisions taken this session

- **Domain-IL becomes the primary scenario** for the rule comparison. Justification: it is what
  Song & Bogacz [R1] use; it removes output suppression, which is rule-independent, leaving
  representation drift, which is what a rule acts on; and it halves the compute.
  **This contradicts `CLAUDE.md`, which must be updated.** Class-IL results are kept and explain
  why Class-IL is a different question rather than being avoided.
- **The output-layer factor becomes three levels** — free / task-1 columns held (masked) /
  fully held — crossed with W1 free/frozen.
- **NCM gets its own single figure, control condition only**, two lines. It is not put into
  every factorial cell; that makes a hard concept unreadable.
- **Capacity-dependence of the mechanism split (C3) is deferred.** Real open question, not on the
  critical path. Old exp 23 suggests the freeze-W1 effect is large at width 2 and near zero at 64.
- **The trajectory figure is one axes with one line per condition**, means only. The per-seed grid
  is analysis output.
- **Training and plotting are decoupled**: every script takes `--replot` and redraws from its
  saved `.npz`. Already done for 41 and 42.
- **Half-life is kept** (see §3), reversing an earlier decision to drop it.

---

## 5. Open questions, not blocking

- ~~**PC settles a fixed 50 steps; EqProp settles with patience.** If PC's relaxation is
  truncated we may be testing a backprop approximation.~~ **ANSWERED by script 50** — PC needs
  ≤18 steps and runs 50, so it is fully settled. Not checked at greater depth.
- **The two rules still stop settling by different mechanisms** — PC on a fixed count, EqProp
  on a tolerance. Script 50 verifies both reach equilibrium, so this does not bias the
  comparison; it is exposition, one sentence on a slide instead of two. Unifying means adding
  the same relative-residual test to `pc_settle` (it would pick ~10–18 instead of 50). Pending,
  optional, not on the critical path.
- **Is MNIST complex enough?** Pinchetti et al. 2025 report PC matching backprop at small scale
  and losing at large. Open question; reference already on the acquisition list.
- **Does the mechanism split depend on width?** (deferred C3, above)
- **One hidden layer means only one place drift can occur.** The depth experiment is the test.
- **`Protocol.stop_threshold` = 0.9 and `Protocol.lr` = {} are still placeholders.** Learning
  rates are legacy-era defaults, so no comparison between rules means anything until they are
  grid-searched per rule and matched on steps-to-threshold.
- **Lower the learning rate for legibility.** Transitions occupy ~200 of 4000 updates, which is
  why the curves look like step functions. Slowing it breaks nothing, because comparisons are
  made at matched accuracy rather than at fixed steps.

---

## 6. Code state

- `src/protocol.py` — `Protocol` frozen dataclass, varied with `replace`. `load` / `build` / `run`.
  `run` returns curves, switches, reached, tasks, snapshots. `hidden` has no default and raises.
- `src/data.py` — **MNIST is cached** to `data/preprocessed/mnist_14.pt`. 8.1 ms → 0.30 ms per
  batch, 27×. Built once in 12 s.
- `src/metrics.py` — `summarise`, `inefficiency`, `sem`, plus crossover / value_when / half_life /
  area_retained.
- `src/probes.py` — NCM readouts, `alignment_probe`, `weight_path_probe` (both wrap `train_step`).
- All review findings closed; see `code_review.md`. Map of the code: `code_map.md`.

---

## 7. The plan — question tree, replacing the slide-driven ordering

Slides are deferred. The presentation falls out of the questions, not the other way round.

```
Do energy-based rules forget less than backprop?
│
├── A. Can the network hold both tasks?        ANSWERED — script 41, H=32
├── B. What does "forgets less" MEAN?
│      B1 do standard metrics work here?
│      B2 what rate metrics work?
│      B3 at what point do we compare?
├── C. Where does the forgetting happen?
│      C1 is the information still there?      ANSWERED — NCM ~80% vs argmax 0%
│      C2 is it W1 or W2?                      ANSWERED (Class-IL) — W2
│      C3 does it depend on capacity?          DEFERRED
│      C4 does it depend on scenario?          axis inside C2 and D, not its own script
├── D. Do the rules differ?
├── E. Why?   E1 target alignment · E2 weight movement
└── F. Does it generalise?   depth · harder data
```

**Agreed running order — revised 2026-08-11.** The rule comparison runs **before** the metrics
work: it puts a real result on the table and makes the metric question motivated rather than
pre-emptive — *we can see a difference, now how do we measure it?* Discover, then measure, then
explain. But the settling control moves ahead of everything, because it decides whether the
rules as configured are the rules we mean, and it sets the compute budget for all of it.

1. **50 — settling.** Do the relaxations converge; what stopping rule and what tolerance.
   **DONE.** Also the technical-intro material on how the energy-based rules work.
2. **51 — learning-rate calibration.** Match all four rules on updates-to-90% on task 1, so a
   forgetting difference is not a learning-speed difference. Matched on task 1 *alone*,
   before task 2 exists, so the calibration cannot be tuned on the dependent variable.
3. **52 — the four-rule grid, fixed budget.** Continuity with exp 12's question under the
   current protocol and code. Both tasks a fixed budget, so all runs share one x-axis.
4. **53 — the same grid, both tasks stopped on accuracy.** Same parameters, principled
   measurement point. Both tasks on the threshold, never a mix of budget and threshold.
   Main figure on a switch-relative step axis; companion trajectory plot for the ragged
   lengths. Curves are *not* stretched to fill the axis — that destroys the rate, and rate is
   half of what is measured.
5. **B1, B2, B3** — one set of backprop runs, three scripts reading the same saved arrays.
6. **C2** — the six-cell factorial, ordinary accuracy. **C1** — NCM, its own single figure.
7. **D controlled**, then **E**, then **F**.

Exp 12's learning rates are **void**, not approximate: they were set under the legacy
specification (backprop on ReLU + cross-entropy, EqProp on hinge with ±1 targets, a different
nonlinearity per rule). All four rules now run tanh, linear output, squared error, one-hot.

If the calibrated curves are still too cramped to read a crossover height off, the fallback is
to divide *every* rule's learning rate by the same factor. Script 51 tests whether that is
safe: it holds only if steps-to-threshold responds to learning rate with the same exponent for
every rule, i.e. if the curves are parallel on log-log axes. 51 fits and reports those slopes.

Scripts 40–43 stay as the exploratory record. 41's arrays are reused, not re-run.

---

## 8. Working relationship notes

- Explanations must map to variables, order of operations and code lines. **No metaphors, no
  analogies, no informal shorthand.** Define every term on first use.
- Plots: simple axis labels ("accuracy (%)"), **no long titles**, consistent colours
  (task 1 orange, task 2 blue), one summarising phrase at most.
- Present a recommendation, not a menu. Too many options is the main source of overwhelm.
- Check whether a stated position rests on a measurement choice before building an argument on it.
- Prefer a cited metric or method over an invented one; if inventing, say so and justify it.
