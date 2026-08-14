# Findings

Verified claims only. Every number is a **paired difference against backprop at the same seed**
(rules share the class split and initialisation), reported in standard errors of that paired
difference (σ). Domain-IL is primary; Class-IL is run where it changes the answer.

**Setup.** MNIST 14×14 (196 inputs), 2 tasks × 5 classes, one hidden layer H=32, tanh, squared
error, plain SGD, batch 32, learning rates grid-searched per rule and matched on updates-to-90%
on task 1. Only the learning rule varies. Chance 20% (Domain-IL) / 10% (Class-IL); joint ceilings
94.3% / 93.6%.

---

## Q1 — How should forgetting be measured?

**CLAIM 1. Endpoint metrics cannot rank these methods, and the metric Song & Bogacz report
cannot separate learning from forgetting. Crossover height can.**

| metric | what it is | why it fails / holds |
|---|---|---|
| final accuracy, ACC, BWT, forgetting | read at the end of training | value is set by how long task 2 ran. In 2×5 Class-IL all conditions reach ~0 and rank nothing |
| **mean test error over training** ([R1]'s headline) | average error across the run | rewards fast learning **and** low forgetting, and cannot distinguish them. A faster learner wins it without forgetting less |
| **crossover height** | accuracy at which the task-1 and task-2 curves intersect | a geometric feature of the trade-off: no budget, no threshold, invariant to uniform rescaling of time. **Primary** |
| half-life | updates to lose half the pre-switch peak | usable in Class-IL; **not quotable in Domain-IL** — defined on 1–2 of 5 seeds |

- **Measurement point decides the answer.** The same intervention (freezing W1) reads **0 points
  at saturation** and **+17 points at matched competence**; it is worth **+31 points at 50
  updates after the switch and 0 at 2000**. → `42`, `43`.
- **A null crossover is a result, not missing data.** If the curves never meet, task 1 stayed
  above task 2 throughout, i.e. forgetting ran slower than learning. Such runs are ranked at the
  top by a paired sign test, not dropped. Dropping them biases against the better rule: replay
  is censored on **11/24** seeds in Domain-IL and **0/5** in Class-IL.
- **Comparisons must be paired per seed.** Group means put the positive control at 0.7σ; the
  same runs paired give 5.1σ. → `53`.

**Experiments:** `42`, `43` (measurement point), `62` (every saved run re-reported under the full
metric grid).
**Plots:** `43_how_much_room_at_matched_competence.png` — read the masked/frozen bars against the
control bar at a matched task-2 accuracy, not at the end. `62_metric_grid_over_all_runs.png` —
one panel per metric, experiments along x, bars above zero favour the rule; the sign is
pre-corrected so up is better everywhere.

---

## Q2 — Do energy-based rules forget less than backprop?

**CLAIM 2. Predictive coding does not reduce forgetting in Domain-IL, at any depth or width
tested. Equilibrium propagation is consistently worse.**

Crossover height, paired against backprop:

| | Domain-IL fixed budget (`52`) | Domain-IL matched competence (`53`) |
|---|---|---|
| replay | **+2.56** (5W–0L, censored 3/5) | **+2.75** (5W–0L, censored 3/5) |
| pc | −0.01 (0.0σ) | +0.49 (2.0σ) |
| eqprop | −1.22 (1.6σ) | **−1.98 (3.8σ)**, 0W–5L |

- **Depth and width do not rescue PC.** Depth {1,2,3,5} × width {32,128}, learning rates
  calibrated per cell: PC spans **−1.75 to +0.52 and never separates in 8/8 cells**, while replay
  separates in 8/8. → `59`.
- **The retention curves coincide.** At depths 1–3 PC's and backprop's task-1-against-task-2
  curves are superimposed along their whole length, not merely at one sampled point. → `55`.
- **The problem is solvable.** Replay separates in every scenario, budget and cell, so the
  energy-based rules fail at something achievable.

**Experiments:** `52`, `53`, `55`, `59`.
**Plots:** `52_..._fixed_budget.png` and `53_...png` — four panels, one per rule; the dotted
horizontal line is that panel's crossover height, so panels compare at a glance.
`52_..._trajectory.png` — task-1 against task-2 accuracy on one axes, time removed: a curve
staying high as it rises retained task 1. `59_..._retention_{depth}x{width}.png` — one panel per
cell; PC (red) sits on backprop (grey) in all eight.

---

## Q3 — Does the scenario change the answer?

**CLAIM 3. PC's only advantage is in Class-IL, where forgetting has a mechanism that Domain-IL
structurally cannot have.**

| crossover vs backprop | Class-IL fixed (`56`) | Class-IL matched (`57`) |
|---|---|---|
| replay | +7.32 (9.7σ) | +11.67 (5.9σ) |
| **pc** | **+1.29 (3.7σ), 5W–0L** | **+3.00 (2.2σ), 4W–1L** |
| eqprop | −8.79 (13.6σ), 0W–5L | −5.13 (3.6σ), 0W–5L |

- **Class-IL forgetting is output-layer suppression.** Nearest-class-mean accuracy on task 1
  holds ~80% while the network's own argmax reads 0.2% — the hidden code survives and the output
  layer is what fails. → `42`, `43`.
- **Suppression is impossible in Domain-IL**, where every output unit is a target for some class.
  What remains there is representation drift.
- PC's advantage appears exactly where the output layer is the problem and is absent where it is
  not. Its size (1–3 points against replay's 7–12) should be read against that ceiling.

**Experiments:** `42`, `43`, `56`, `57`. (`67`, decomposing Class-IL forgetting for all four
rules rather than backprop alone, is running.)
**Plots:** `56_..._class_il.png`, `57_...png` — as Q2. `43_...png` third panel: argmax against
NCM per condition; points above the diagonal mean the hidden code survived and the output layer
failed.

---

## Q4 — Do the rules actually compute something different?

**CLAIM 4. Yes, and the difference is real, measurable, and located at the output layer.**

- **Update direction differs.** cos(ΔW) against backprop's update: EqProp **0.197–0.316** on W1,
  PC **0.985** on W1 and **0.814** on W2. → `54`.
- **PC's divergence grows with distance from the input, not with depth.** At depth 3:
  W1 0.952, W2 0.939, W3 0.837, W4 0.623. → `55`.
- **PC's output-layer route is more economical than backprop's.** Median L1 path ÷ net
  displacement per synapse, task 2: **W2 1.703 vs 3.279 (−1.58, 6.6σ)** — a 31% shorter path
  displacing 22% further. On W1 it is marginally worse (2.781 vs 2.508, +0.27, 9.8σ). → `65`.
- **EqProp travels further to arrive shorter.** W1 path 182.1 against backprop's 123.2 (+48%)
  with net displacement 28.3 against 43.8 (−35%); inefficiency 4.56 vs 2.51 (13.0σ) on W1 and
  8.85 vs 3.28 (12.1σ) on W2. → `65`.

**Experiments:** `54`, `55`, `65`.
**Plots:** `65_..._mechanism.png` — **the figure that shows what is being measured.** Each panel
is one layer's route through weight space for one seed: the wiggly line is the route, the three
straight lines are net displacement init→switch, switch→end, and init→end. Shaded area is the
wandering the ratio charges for. Read the *purple* line: the whole-run displacement is shorter
than the two blocks summed, and the difference is retraced ground.
`65_synaptic_path_efficiency.png` — the aggregate; one point per seed, bar is the mean.

---

## Q5 — Then why does the difference not reduce forgetting?

**CLAIM 5. The differences land in a layer and in a quantity that do not govern forgetting.**

- **Drift in W1 is what damages task 1** (`42`, `43`), and W1 is the layer where PC is *most*
  backprop-like (cos 0.952–0.985) and marginally *less* efficient. PC reorganises the output end
  and leaves the layer that matters alone.
- **[R1]'s own mechanism measure does not track forgetting.** Target alignment during task 2:

| | alignment | vs backprop | interference on task 1 |
|---|---|---|---|
| backprop | +0.277 | — | −0.030 |
| replay | +0.181 | −0.096 | −0.000 |
| pc | +0.280 | +0.003 (1% relative) | −0.025 |
| eqprop | +0.250 | −0.027 | **+0.039** |

  - **PC is not more target-aligned than backprop** — +0.003 on a 0.277 baseline. It reaches
    2.6σ only because a per-update measure averaged over ~130 measurements has a very small
    standard error; the effect size is negligible.
  - **The measure inverts at the extreme.** EqProp's updates move task-1 outputs *toward* their
    targets on average (+0.039, the best of the four) and EqProp forgets the most. The retention
    ordering is not the interference ordering.
  - **Alignment is a direction measure and is blind to magnitude.** A rule can point helpfully
    and still do more damage per update by moving further. Consistent with the step sizes seen
    on task-1 data in `64_..._mechanism.png`: |d_learn|/|d_target| is 0.127 (backprop), 0.172
    (replay), 0.201 (pc) and **0.834 (eqprop)** — EqProp's update is 4–6× larger, and its angle
    is 90° (cos −0.001), so it does no *directional* harm while displacing task 1 the furthest.
    *One seed, one update: this explains the aggregate inversion, it does not measure it.*
    The magnitude-aware quantity — `(d_learn · d_target) / |d_target|²`, the fraction of the
    remaining gap closed, negative when the gap widens — is **not yet recorded per seed.**
- **Forgetting is visible directly as retraced weight distance.** Over a whole run backprop's W1
  blocks displace 47 and 43 while init→end is 52: **38 units are travelled and given back**, so
  the full-run ratio (2.86x) is far worse than either block (1.77x, 1.53x). Replay retraces least
  (27 of 82). PC's W2 runs almost straight through the switch (full-run 1.40x vs backprop's
  3.05x): at the output layer its second task continues where backprop's reverses. → `65`.

**Experiments:** `64`, `65`.
**Plots:** `64_..._mechanism.png` — **the figure that shows what the cosine is.** Two arrows per
panel: `d_target` (where the target is) and `d_learn` (where the update went). The plane is
spanned exactly by those two vectors, so **the drawn angle is the reported cosine** — no
projection loss. Top row is data being trained on; bottom row is task-1 data that was not.
An obtuse bottom-row angle is an update pushing task 1 away from its own targets; replay's is
acute, which is the instrument's sanity check.
`64_target_alignment_and_interference.png` — the same two quantities per update across the run,
switch marked.

---

## Q6 — What does determine forgetting here?

**CLAIM 6. Which two digits share an output unit explains ~62% of the seed-to-seed variance in
Domain-IL retention — more than any rule difference measured.**

| backprop, 24 seeds | r vs pairing similarity | p | observed range |
|---|---|---|---|
| **retaining** task 1 | **+0.787** | <0.001 | 23.0 – 71.0% |
| learning task 2 (final) | +0.164 | 0.45 | 82.2 – 95.4% |
| learning task 1 (peak) | +0.155 | 0.47 | 84.8 – 96.6% |
| speed to 80% on task 2 | −0.354 | 0.088 | reached on 24/24 |

- Same for PC (+0.779) and replay (+0.630), all p ≤ 0.001.
- **The control is null.** Same digits, same split, pairing ignored: r between −0.25 and +0.06,
  all p > 0.23; **+0.7 accuracy points per sd of the predictor against the pairing's +9.7**, so
  null in effect size, not merely range-restricted.
- **It is interference, not learnability.** Every seed learned task 2; the speed correlation has
  the opposite sign to "dissimilar pairs are harder to learn".
- **Replay is half as sensitive** (+4.0 points/sd) — a buffer partly insulates against a bad
  pairing.

**Experiments:** `60`.
**Plots:** `60_why_does_the_seed_matter_so_much.png` — left column is the hypothesis (similarity
of the digits sharing a unit), right column is the control (all cross-task pairs). **The result
is the contrast: left sloped, right flat.** Open triangles are runs that never crossed, drawn at
their lower bound. `60_..._digits.png` — one row per seed ordered by pairing similarity, the two
digits sharing each output unit drawn adjacent (orange = task 1, blue = task 2); row label gives
similarity and task-1 retention.

---

## Q7 — What happens when the tasks alternate repeatedly instead of switching once?

**CLAIM 7. Under repeated alternation all three rules converge on a joint solution, at the same
rate and to the same place. Under [R1]'s own schedule and their own headline metric they are
indistinguishable.**

20 alternating blocks × 150 updates, Domain-IL, 5 seeds. EqProp excluded on cost.

| | crossover, cycle 1 → cycle 10 | [R1] mean test error, whole run | successive same-task W1 distance, last ÷ first |
|---|---|---|---|
| backprop | 53.9% → **81.3%** | 23.7% | 0.23 |
| replay | 28.6% → 79.3% *(n=29/100)* | 23.7% | 0.28 |
| pc | 54.2% → **81.7%** | **23.6%** | **0.17** |

- **It converges; it does not ping-pong or wander.** The (task 1, task 2) trajectory spirals
  inward with monotonically tightening arcs from (25,25) to ≈(85,85), and the weight distance
  between successive same-task states falls to 0.17–0.28 of its first-cycle value. Both readouts
  agree, and they are independent — one is accuracy, one is weight space.
- **Crossover rises steeply then plateaus**: ~29% at block 0 to ~75% by block 3, flat at ~81%
  thereafter. Repeated exposure buys most of its benefit in the first three cycles.
- **No rule separates.** Backprop 81.3 vs PC 81.7 on crossover (n=85 and 87 of 100 cells);
  [R1]'s mean test error 23.7 / 23.7 / 23.6. This is the project's most direct test of their
  claim, in the metric their claim is made in, under the schedule they use.
- **PC settles tightest in weight space** (0.17 vs backprop 0.23), consistent with Claim 4's
  route-efficiency result, and with no accuracy consequence.
- **Half-life is not rescued by alternation.** Defined on ~1 of 100 block × seed cells: once the
  network nears the joint solution the unattended task barely decays inside a block, so it never
  halves. Claim 1's exclusion of half-life for Domain-IL stands.
- **Replay's crossover is censored on 71 of 100 cells** — it holds both tasks up, so the curves
  rarely meet. Its apparent +50.7 improvement is over its non-censored blocks only and is not
  comparable with the other two.

**Experiments:** `68`.
**Plots:** `68_..._spiral.png` — **the figure to show.** Task 1 against task 2 accuracy, colour =
time, one panel per rule, per-seed runs faint. Read the *arc width*: tightening = converging,
constant = ping-ponging. `68_....png` — crossover, half-life, [R1] mean error and the same-task
weight distance against block index; the half-life panel is empty by construction and says so.
`68_..._accuracy.png` — all 20 blocks shaded by which task is training.

---

# Supporting

**S1. PC settles, and its inference step rule is equivalent to Song & Bogacz's.**
PC reaches equilibrium in ≤18 steps and the protocol runs 50, leaving it 3.2e-04 from the settled
state at initialisation and 1.9e-05 once trained. Their backtracking step rule (`x_lr_discount`
0.9) and our fixed step reach the **same fixed point** (relative gap 5e-06 to 9e-05),
`cos(ΔW) = 1.000000` at every step count from 1 to 200, and the largest disagreement over the
whole metric grid is **1.4e-03 accuracy points**. → `50`, `63`. Removes the implementation
objection: backtracking changes the route to the fixed point, not the fixed point.
*Open, and it runs the other way:* if [R1]'s own fixed step count leaves their relaxation short
of equilibrium, their operating point is not the equilibrium.
**Plot:** `63_...png` panel 1 — distance from settled against inference steps, log-log; the
dotted vertical line is the protocol's setting.

**S2. Replay is a working positive control everywhere.**
+26.6 retention (6.6σ) in `52`; **24W–0L on crossover, p < 0.001** in `60`; separates in 8/8 cells
in `59`. Any setup in which replay does not separate cannot detect a forgetting difference.

**S3. Why the earlier experiment-12 result is no longer supported.**
Re-run verbatim (`61`), that setup's crossover heights are backprop 64.0, pc 59.5 (−4.52, 3.5σ),
**replay 48.7 (−14.14, 5.2σ)**, eqprop 32.4.
- **The positive control inverts** — replay comes out below backprop. At 100 updates per task
  nothing has converged, so the ordering is set by task-2 learning speed, not by forgetting.
- **The output structure differed by rule.** Under that specification backprop and replay used
  cross-entropy with a task-restricted softmax, giving absent classes exactly zero gradient — an
  oracle task mask that PC (squared error) and EqProp (hinge, ±1 targets) did not have. In
  Class-IL this removes the dominant forgetting mechanism for two of the four rules.
- Both defects are properties of that configuration, not of the learning rules. Under a
  controlled protocol the same code gives replay +7.32 crossover at 9.7σ (`56`).

**S4. Two forgetting mechanisms with different time constants.**
Output-layer suppression is large and persistent; representation drift changes the *rate* and not
the asymptote — freezing W1 is worth +31 points at 50 updates after the switch and 0 at 2000.
Which mechanism appears dominant is decided by when measurement stops. → `42`, `43`.

**S5. Capacity is not a confound.**
H=32 is past the knee in both scenarios (`41`), and at H=128 — four times the capacity — PC is
+0.7 to +2.5 and still never separates (`59`).
**Plot:** `41_capacity_vs_hidden_width.png` — accuracy against hidden width; H is chosen at the
knee, and a sweep run at too short a budget produces a flat region indistinguishable from a
ceiling, so convergence is checked first.

**S6. PC matches backprop's dependence on the data, not only its mean.**
Slope of retention against pairing similarity: PC +9.5 points/sd, backprop +9.7. The two rules
respond identically to the structure that dominates the variance — a stronger null than
"not separated at n seeds". → `60`.

---

## Not claimed

- Any result from a configuration whose positive control does not separate.
- Any ranking on half-life in Domain-IL (defined on 1–2 of 5 seeds).
- **Inefficiency predicts forgetting within a rule.** The within-rule correlation (r = +0.81 to
  +0.89) is confounded: pairing similarity correlates 0.90–0.97 with retention and 0.50–0.75 with
  inefficiency across all four rules, so it is a common cause of both. Needs 24 seeds with the
  pairing partialled out.
- Any comparison to Song & Bogacz's reported results as a replication. Their setup was not run
  here; the schedule, resolution, activation, depth, batch size and loss reduction all differ.
  The claim is about a controlled comparison between rules, not about reproducing their figures.

## Open

| | question | status |
|---|---|---|
| `66` | does the weight-space route differ between Class-IL and Domain-IL? | running |
| `67` | does the Class-IL decomposition hold for all four rules, not just backprop? | running |
| — | concept drift ([R1] Fig 4f–g), where their claimed advantage is largest | not run |
| — | depth in Class-IL, the one scenario where PC shows an effect | not run |
| — | alignment magnitude (`\|d_learn\|`), to separate direction from step size | not recorded |
