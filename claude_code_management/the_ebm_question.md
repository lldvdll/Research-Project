# Do energy-based rules reduce catastrophic forgetting?

One experiment, five questions. Everything below comes from the same four runs, so the metric,
the scenario and the data can each be varied with nothing else moving.

**The experiment.** MNIST 14×14, 2 tasks × 5 classes, 196–32–out, tanh, squared error, plain SGD,
batch 32. Learning rates grid-searched per rule and matched on updates-to-90% on task 1, so no
rule is compared at a different learning speed. Four rules: **backprop** (negative control),
**replay** (positive control), **pc**, **eqprop**. Two scenarios × two measurement points:

| | fixed budget | matched competence |
|---|---|---|
| **Domain-IL** — 5 shared output units | `52` | `53` |
| **Class-IL** — 10 separate output units | `56` | `57` |

5 seeds. Every number is a **paired difference against backprop at the same seed** in standard
errors of that difference (σ), because all rules see the same class split and initialisation at a
given seed — see Q5.

---

## Q(a) — What does Song & Bogacz's metric say, and does it verify their claim?

[R1] report the **mean test error over training**, not a final accuracy. Their claim is made in
that number, so it is the number to test it in.

PC minus backprop, mean test error (negative = PC better):

| | task 1 | task 2 |
|---|---|---|
| Domain-IL fixed | **+2.97** (2.7σ, worse) | **−2.05** (3.0σ, better) |
| Domain-IL matched | −1.60 (2.0σ) | +0.97 (1.2σ) |
| Class-IL fixed | **+3.24** (5.5σ, worse) | **−2.27** (5.9σ, better) |
| Class-IL matched | −5.48 (2.6σ, better) | +0.92 (0.9σ) |

- **The two halves move in opposite directions in all four cells.** Where PC is better on task 2
  it is worse on task 1, and vice versa. A single mean over both tasks averages that away.
- The reason is structural, not incidental: mean-test-error rewards **learning the new task fast**
  and **not destroying the old one** with one number, and cannot say which produced a given value.
  A rule that learns task 2 faster wins it without retaining anything extra.
- Under repeated alternation — [R1]'s own schedule, 20 blocks — the whole-run mean test error is
  **backprop 23.7%, replay 23.7%, pc 23.6%**: the rules are indistinguishable in their metric,
  under their schedule. → `68`.

**Answer: no.** The metric does not verify their claim here, and it could not cleanly verify it
either way, because it cannot separate the two effects it sums.

**Plots:** `62_metric_grid_over_all_runs.png` (the `mean_err_t1` / `mean_err_t2` panels — note they
disagree in sign), `68_....png` bottom-left (mean test error per block; the three rules
superimpose).

---

## Q(b) — What do endpoint metrics say, and why should they not be trusted?

Endpoint metrics — final task-1 accuracy, ACC, BWT, forgetting — are read when training stops.

PC minus backprop, **final task-1 accuracy**:

| | fixed budget | matched competence |
|---|---|---|
| Domain-IL | **−2.56** (2.7σ, worse) | **+2.48** (1.3σ, better) |
| Class-IL | **−2.04** (4.0σ, worse) | **+1.60** (0.7σ, better) |

- **The sign flips in both scenarios**, from the same runs, purely by changing when the reading is
  taken. ACC, BWT and "forgetting" flip with it — they are algebraic rearrangements of the same
  endpoint.
- **Why:** at a fixed budget PC learns task 2 further, so it displaces more of task 1. At matched
  competence both rules are held to the same task-2 standard and that advantage is removed. The
  endpoint therefore measures *how long task 2 ran*, not how much was forgotten. This is
  **setup-induced forgetting** (Michel et al. 2023, arXiv:2309.00462).
- **In Class-IL the scale is degenerate.** Backprop retains **6.72%** at a fixed budget, so a
  ±2 point difference is read off a floor. Every condition collapses toward zero and endpoint
  metrics rank nothing.
- Demonstrated directly: freezing W1 reads **0 points at saturation** and **+17 at matched
  competence**, and is worth **+31 at 50 updates after the switch and 0 at 2000**. → `42`, `43`.

**Answer: they say PC is worse, or better, depending on a choice that has nothing to do with the
learning rule.**

**Plots:** `43_how_much_room_at_matched_competence.png` — the same interventions as `42` read at
matched competence rather than at saturation; the pair is the argument.

---

## Q(c) — What do the better metrics say, and why are they better?

**Crossover height** — the accuracy at which the task-1 and task-2 curves intersect.

PC minus backprop, crossover:

| | fixed budget | matched competence |
|---|---|---|
| Domain-IL | −0.01 (0.0σ) | +0.49 (2.0σ) |
| Class-IL | **+1.29** (3.7σ, 5W–0L) | **+3.00** (2.2σ, 4W–1L) |

- **The sign does not flip.** Same runs, same two measurement points that flipped every endpoint
  metric. Domain-IL stays at ~0; Class-IL stays positive.
- **Why it is stable:** it is a geometric feature of the trade-off curve, not a point on a time
  axis. It has no budget and no threshold in its definition, and is invariant to uniformly
  rescaling time — so a rule that is simply faster cannot win it.
- **What it cannot do:** it is undefined when the curves never meet. That is not missing data —
  it means task 1 stayed above task 2 throughout, i.e. forgetting ran slower than learning, which
  is the *best* outcome. Those runs are ranked at the top by a paired sign test rather than
  dropped. Replay is censored on **11/24** seeds in Domain-IL and **0/5** in Class-IL.
- **Half-life is not a usable substitute in Domain-IL** — defined on 1–2 of 5 seeds under a single
  switch, and on ~1 of 100 block×seed cells under repeated alternation (`68`), because task 1
  often never loses half its value.

**Answer: PC is indistinguishable from backprop in Domain-IL and ahead by 1–3 points in Class-IL,
and that answer is stable across measurement points.** Replay separates everywhere (+2.6 to
+11.7), so the problem is solvable and the energy-based rules fail at something achievable.

**Plots:** `52_..._fixed_budget.png`, `53_...png`, `56_...png`, `57_...png` — four panels each,
one per rule; the dotted horizontal line is that panel's crossover height, so panels compare at a
glance. `52_..._trajectory.png` — task 1 against task 2 with time removed.

---

## Q(d) — Is the answer different in the two scenarios, and why?

| crossover vs backprop | Domain-IL | Class-IL |
|---|---|---|
| pc | −0.01 / +0.49 | **+1.29 / +3.00** |
| eqprop | −1.22 / −1.98 | **−8.79 / −5.13** |
| replay | +2.56 / +2.75 | +7.32 / +11.67 |

**Yes. The scenarios forget by different mechanisms, and only one of them is a mechanism a
learning rule can act on.**

- **Class-IL forgetting is output-layer suppression.** Nearest-class-mean accuracy on task 1 holds
  ~80% while the network's own argmax reads 0.2% — the hidden representation survives and the
  output layer is what fails. → `42`, `43`.
- **Suppression is structurally impossible in Domain-IL**, where every output unit is a target for
  some class. What remains there is representation drift.
- **PC's updates diverge from backprop's most at the output layer**: cos(ΔW) 0.985 on W1 against
  0.814 on W2 (`54`), and 0.952 → 0.623 from first to last layer at depth 3 (`55`). Its
  output-layer route is also 31% shorter while displacing 22% further (`65`).
- So PC differs where Class-IL does its damage and matches backprop where Domain-IL does. The
  scenario pattern follows from where the rule differs, not from the rule being better.
- **Depth does not change this**: {1,2,3,5} × {32,128}, PC spans −1.75 to +0.52 and never
  separates in 8/8 Domain-IL cells while replay separates in 8/8. → `59`.
- **Effect size in context:** PC's +1.29/+3.00 sits against replay's +7.32/+11.67 in the same
  cells.

**Plots:** `43_...png` third panel — argmax against NCM; points above the diagonal mean the hidden
code survived and the output layer failed. `65_..._mechanism.png` — the weight-space route per
layer, showing where each rule's difference actually lives.

---

## Q(e) — How does the data affect the result, and how is it controlled?

**The split matters more than the rule.** In Domain-IL, `Protocol.label_map` sends the i-th class
of each task to output unit i, so every unit carries one task-1 digit and one task-2 digit, paired
by the per-seed permutation.

| backprop, 24 seeds (`60`) | r vs pairing similarity | p | range |
|---|---|---|---|
| **retaining** task 1 | **+0.787** | <0.001 | **23.0 – 71.0%** |
| learning task 2 | +0.164 | 0.45 | 82.2 – 95.4% |
| learning task 1 | +0.155 | 0.47 | 84.8 – 96.6% |

- **~62% of the seed variance in retention** is explained by which two digits share a unit — a
  48-point range, larger than every rule difference in this document.
- **It is interference, not learnability.** Every seed learned task 2; the speed correlation has
  the *opposite* sign to "dissimilar pairs are harder".
- **The control is null:** same digits, same split, pairing ignored — r between −0.25 and +0.06,
  and +0.7 accuracy points per sd of the predictor against the pairing's +9.7. So it is
  specifically the pairing, not which digits are where.

**The confound is not removed — it is cancelled.** Every rule sees the same split at the same
seed, so a per-seed difference subtracts it out. Measured (`69`):

| | unpaired | paired | sem ratio |
|---|---|---|---|
| pc − backprop, retention | −1.48 ± 3.54 (**0.4σ**) | −1.48 ± 0.31 (**4.8σ**) | **11.5×** |
| pc − backprop, crossover | −0.28 ± 1.12 (0.3σ) | −0.28 ± 0.19 (1.5σ) | 6.0× |
| replay − backprop, crossover | +3.26 ± 1.10 (3.0σ) | +3.26 ± 0.39 (8.3σ) | 2.8× |
| replay − backprop, retention | +30.65 ± 2.82 (10.9σ) | +30.65 ± 2.22 (13.8σ) | 1.3× |

- **The point estimates are identical.** Only the uncertainty differs.
- **Pairing helps in proportion to how similarly the two rules respond to the split.** PC tracks
  backprop's dependence almost exactly (slope +9.5 vs +9.7 points/sd), so nearly all of it
  cancels — 11.5×. Replay is half as sensitive (+4.0), so less cancels — 1.3×.
- Consequence: an unpaired comparison at 5 seeds would put the **positive control** at 0.7σ and
  read as a failure. The sample size was never the problem.
- *Whether Class-IL carries an analogous confound is measured by `69` and pending.*

**Plots:** `60_....png` — left column is the pairing (sloped), right column the control (flat);
the contrast is the result. `60_..._digits.png` — one row per seed ordered by similarity, the two
digits sharing each output unit drawn adjacent. `69_....png` — the two scatters side by side plus
the sem-ratio bars.

---

## Summary

| | answer |
|---|---|
| Does PC reduce forgetting? | **No in Domain-IL**, at any depth or width. **+1–3 points in Class-IL**, against replay's +7–12 in the same cells. |
| Does EqProp? | **No — consistently worse**, 0W–5L across seeds in three of four cells. |
| Does [R1]'s metric verify their claim here? | **No**, and it cannot separate the two effects it sums. Under their schedule and their metric the rules are indistinguishable (23.6 / 23.7 / 23.7). |
| Do endpoint metrics? | They flip sign with the measurement point and are degenerate in Class-IL. |
| Does the scenario matter? | Yes — Class-IL forgets by output suppression, which is a mechanism a rule can act on; Domain-IL cannot suppress. |
| Does the data matter? | More than the rule. 62% of seed variance. Cancelled by paired comparison, not removed. |

**One caveat carried throughout.** This is a controlled comparison between rules under one
protocol, not a replication of [R1]. Their setup was not run: schedule, resolution, activation,
depth, batch size and loss reduction all differ. The claim is that under a protocol where the
learning rule is the only thing that varies, PC's advantage does not appear — not that their
figures are wrong.
