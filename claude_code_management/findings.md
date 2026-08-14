# Findings — do energy-based rules reduce forgetting, and how does the metric change the answer?

**The experiment.** MNIST 14×14, 2 tasks × 5 classes, 196–32–out, tanh, squared error, plain SGD,
batch 32. Learning rates grid-searched per rule, matched on updates-to-90% on task 1. Four rules:
**backprop** (negative control), **replay** (positive control), **pc**, **eqprop**. Two scenarios
(Domain-IL: 5 shared output units; Class-IL: 10 separate) × two measurement points (fixed budget;
matched task-2 competence) = `52`, `53`, `56`, `57`. 5 seeds.

All numbers are **paired differences against backprop at the same seed**, in standard errors of
that difference (σ). See C9 for why.

*Every figure below was re-derived from the saved `.npz` arrays, not carried forward from earlier
notes.*

---

## Claims

1. Under **crossover height**, PC is indistinguishable from backprop in Domain-IL and 1–3 points
   ahead in Class-IL, against replay's 7–12 in the same cells.
2. Under **endpoint metrics**, the same runs say PC is worse *or* better depending only on when
   the reading is taken — the sign flips in both scenarios.
3. Under **Song & Bogacz's mean test error**, PC is better on one task and worse on the other in
   every cell, so averaging the two removes the effect; on their own alternating schedule the
   three rules are indistinguishable.
4. **EqProp is worse than backprop** under every metric, in every cell, on every seed.
5. **Depth and width do not change the Domain-IL answer**: PC never separates in 8 of 8 cells,
   while replay separates in 8 of 8.
6. **The scenario changes the answer** because Class-IL forgetting is output-layer suppression — a
   mechanism a learning rule can act on — and Domain-IL structurally cannot suppress.
7. **The rules do compute measurably different updates**, and the difference is located at the
   output layer, which is not the layer whose drift damages task 1.
8. **[R1]'s own mechanism metric does not track forgetting**: the rule with the least damaging
   update *direction* forgets the most, because direction is blind to step size.
9. **Which two digits share an output unit explains ~62% of the seed variance** in Domain-IL
   retention — more than any rule difference measured — and is cancelled by paired comparison
   rather than removed.
10. **Under repeated alternation all rules converge on a joint solution** at the same rate and to
    the same place.

---

## 1. Crossover height: no Domain-IL effect, a small Class-IL effect

PC minus backprop, crossover height:

| | fixed budget | matched competence |
|---|---|---|
| Domain-IL | −0.01 (0.0σ) | +0.49 (2.0σ) |
| Class-IL | **+1.29** (3.7σ, 5W–0L) | **+3.00** (2.2σ, 4W–1L) |
| *replay, same cells* | +2.56 / +7.32 | +2.75 / +11.67 |

**Impact of the metric choice.** Crossover is the accuracy at which the two task curves intersect
— a geometric feature of the trade-off, with no budget or threshold in its definition and
invariant to uniformly rescaling time. Consequence for this result: **the sign does not change
between the two measurement points** that flip every endpoint metric (C2). A rule that is merely
faster cannot win it.

Where it is silent: undefined when the curves never meet, which happens when task 1 stays above
task 2 throughout. Replay is censored on 11/24 Domain-IL seeds and 0/5 Class-IL seeds; those runs
are ranked top by a paired sign test, not dropped.

**Plots.** `52_..._fixed_budget.png`, `53_...png`, `56_...png`, `57_...png` — four panels each,
one per rule, dotted horizontal line = that panel's crossover height. *Why evidence:* the four
panels are the four rules under identical everything else, so the horizontal lines are directly
comparable and PC's sits on backprop's in Domain-IL and above it in Class-IL.
`52_..._trajectory.png` — task 1 against task 2 with time removed; *why evidence:* PC's and
backprop's curves coincide along their whole length, not just at one sampled point.

## 2. Endpoint metrics: the sign flips with the reading point

PC minus backprop, **final task-1 accuracy**:

| | fixed budget | matched competence |
|---|---|---|
| Domain-IL | **−2.56** (2.7σ) | **+2.48** (1.3σ) |
| Class-IL | **−2.04** (4.0σ) | **+1.60** (0.7σ) |

**Impact of the metric choice.** Endpoint metrics are read when training stops, so their value is
set by how long task 2 ran. At a fixed budget PC learns task 2 further and therefore displaces
more of task 1; at matched competence that advantage is removed and the sign reverses. ACC, BWT
and "forgetting" flip with it — they are rearrangements of the same endpoint. This is
setup-induced forgetting (Michel et al. 2023, arXiv:2309.00462).

In Class-IL the scale is additionally degenerate: backprop retains **6.72%** at a fixed budget, so
a ±2 point difference is read off a floor.

**Plots.** `42_how_much_room_masked_frozen.png` and `43_how_much_room_at_matched_competence.png`
— *why evidence:* the same four interventions, read at saturation and at matched competence. The
reading point, not the intervention, decides the number:

| freezing W1, minus control | value |
|---|---|
| at saturation (`42`, end of a 2000-update block) | **−0.2** |
| at matched task-2 competence (`43`) | **+17.4** |
| 50 updates after the switch (`42`) | **+31.2** |
| 2000 updates after the switch (`42`) | −0.2 |

## 3. Song & Bogacz's metric: the two halves cancel

PC minus backprop, mean test error over training (negative = PC better):

| | task 1 | task 2 |
|---|---|---|
| Domain-IL fixed | +2.97 (2.7σ, worse) | −2.05 (3.0σ, better) |
| Domain-IL matched | −1.60 (2.0σ) | +0.97 (1.2σ) |
| Class-IL fixed | +3.24 (5.5σ, worse) | −2.27 (5.9σ, better) |
| Class-IL matched | −5.48 (2.6σ, better) | +0.92 (0.9σ) |

Whole-run mean test error under 20 alternating blocks — [R1]'s own schedule:
**backprop 23.7%, replay 23.7%, pc 23.6%.**

**Impact of the metric choice.** [R1] report mean test error over training, not a final accuracy,
so their claim has to be tested in it. The metric sums two things — learning the new task quickly
and not destroying the old one — and cannot say which produced a value. Consequence here: **PC's
task-1 and task-2 halves have opposite signs in all four cells**, so a single mean over both
removes the effect. On their schedule, in their metric, the rules are indistinguishable.

**Plots.** `62_metric_grid_over_all_runs.png` — *why evidence:* the `mean_err_t1` and
`mean_err_t2` panels disagree in sign for PC in every experiment on the x-axis.
`68_....png` bottom-left — *why evidence:* the three rules' mean-error curves superimpose across
all 20 blocks.

## 4. EqProp is worse, everywhere

| crossover vs backprop | Domain-IL | Class-IL |
|---|---|---|
| eqprop | −1.22 (1.6σ) / −1.98 (3.8σ) | −8.79 (13.6σ) / −5.13 (3.6σ) |

**0W–5L across seeds** in `53`, `56` and `57`. It is also worse on endpoint retention, on
forgetting, on BWT and on mean test error for task 1. **Impact of metric choice: none** — the
sign is the same under every metric, which is what makes this a finding rather than a null.

Mechanism: its updates are nearly orthogonal to backprop's (cos 0.197–0.316 on W1, `54`) and it
travels **48% further to arrive 35% short** (W1 path 182.1 vs 123.2, net displacement 28.3 vs
43.8; inefficiency 4.56 vs 2.51 at 13.0σ, `65`).

## 5. Depth and width do not rescue PC

Depth {1,2,3,5} × width {32,128}, learning rates calibrated per cell, Domain-IL:
**PC spans −1.75 to +0.52 and separates in 0 of 8 cells. Replay spans +2.24 to +4.42 and
separates in 8 of 8.** Per-cell PC: 1×32 +0.37, 1×128 −0.14, 2×32 +0.48, 2×128 +0.52, 3×32 −0.05,
3×128 +0.39, 5×32 −1.75, 5×128 −0.29 — no cell above 1.6σ.

At H=128 — four times the capacity — PC is +0.7 to +2.5 and still never separates, so the H=32
result is not a capacity ceiling. H=32 is past the knee in both scenarios (`41`).

**Plots.** `59_..._retention_{depth}x{width}.png` — *why evidence:* one panel per cell; PC (red)
lies on backprop (grey) in all eight while replay (brown) sits clearly above.

## 6. The scenario changes the answer, and why

- **Class-IL forgetting is output-layer suppression.** Task-1 accuracy read two ways, backprop:

  | | network's own argmax | nearest class mean |
  |---|---|---|
  | at saturation (`42`) | **0.2%** | **71.7%** |
  | at matched competence (`43`) | 35.7% | 79.3% |

  The hidden representation survives at both reading points; the output layer is what fails.
  Masking the loss — which removes suppression by fiat and needs task identity, so it is an
  oracle rather than a method — recovers argmax to 62.2% at saturation and 73.9% at matched
  competence.
- **Domain-IL cannot suppress** — every output unit is a target for some class. What remains is
  representation drift.
- **PC's updates diverge from backprop's most at the output layer** (C7), so it differs where
  Class-IL does its damage and matches backprop where Domain-IL does. The scenario pattern in C1
  follows from *where* the rule differs.

**Plots.** `43_...png` third panel — argmax against NCM per condition; *why evidence:* points far
above the diagonal mean the hidden code survived and the output layer alone failed, which is the
suppression signature.

*Pending:* `67` decomposes Class-IL forgetting for all four rules rather than backprop alone.

## 7. The rules differ, at the output layer

- cos(ΔW) against backprop: PC **0.985** on W1, **0.814** on W2 (`54`); 0.952 → 0.623 from first
  to last layer at depth 3 (`55`).
- PC's output-layer route is **31% shorter while displacing 22% further** (median L1 path ÷ net
  displacement: W2 1.703 vs 3.279, 6.6σ) and marginally *worse* on W1 (2.781 vs 2.508) → `65`.
- W1 is the layer whose drift damages task 1 (`42`, `43`). **PC improves the layer that does not
  govern forgetting and is fractionally clumsier in the one that does.**
- Over a whole run backprop's W1 displaces 47 then 43 while init→end is only 52: **38 units are
  travelled and given back.** PC's W2 runs almost straight through the switch (full-run 1.40x vs
  backprop's 3.05x) — its second task continues where backprop's reverses.

**Plots.** `65_..._mechanism.png` — *why evidence:* it draws the quantity rather than summarising
it. Each panel is one layer's route through weight space with three net displacements
(init→switch, switch→end, init→end). The purple chord being shorter than the two blocks summed
*is* the retracing, visible directly.

*Pending:* `66` tests whether the route differs between scenarios, which would confirm the
suppression/drift split from the weight side independently of the NCM probe.

## 8. Target alignment does not track forgetting

| | alignment, task 2 | interference on task 1 | \|d_learn\|/\|d_target\| |
|---|---|---|---|
| backprop | +0.277 | −0.030 | 0.127 |
| replay | +0.181 | −0.000 | 0.172 |
| pc | +0.280 (+0.003) | −0.025 | 0.201 |
| eqprop | +0.250 | **+0.039** | **0.834** |

- **PC is not more target-aligned than backprop** — +0.003 on a 0.277 baseline, i.e. 1% relative.
  It reaches 2.6σ only because a per-update measure averaged over ~130 measurements has a very
  small standard error. [R1]'s mechanism does not reproduce at this scale.
- **The measure inverts at the extreme.** EqProp's updates move task-1 outputs *toward* their
  targets on average (+0.039, best of the four) and EqProp forgets the most.
- **Because alignment is direction-only.** EqProp's step is **4–6× larger** than any other rule's
  and its angle is 90° — perpendicular, doing no directional harm, while displacing task 1 the
  furthest. Damage is direction × magnitude; the cosine alone cannot rank it.

**Plots.** `64_..._mechanism.png` — *why evidence:* the plane is spanned exactly by `d_target` and
`d_learn`, so **the drawn angle is the reported cosine** — no projection loss. The bottom row
shows the same update seen by task 1; replay's arrow is acute (it rehearses task 1) while every
other rule's is obtuse, which is the instrument's sanity check.

## 9. The data split matters more than the rule, and pairing cancels it

| backprop, 24 seeds (`60`) | r vs pairing similarity | p | range |
|---|---|---|---|
| **retaining** task 1 | **+0.787** | <0.001 | **23.0 – 71.0%** |
| learning task 2 | +0.164 | 0.45 | 82.2 – 95.4% |
| learning task 1 | +0.155 | 0.47 | 84.8 – 96.6% |

Same for PC (+0.779) and replay (+0.630). The control — same digits, same split, pairing ignored —
gives r between −0.25 and +0.06 and **+0.7 accuracy points per sd against the pairing's +9.7**, so
it is specifically the pairing, not which digits are where. It is **interference, not
learnability**: every seed learned task 2, and the speed correlation has the opposite sign.

**The confound is cancelled, not removed.** All rules see the same split at the same seed, so a
per-seed difference subtracts it (`69`, Domain-IL):

| | unpaired | paired | sem ratio |
|---|---|---|---|
| pc − backprop, retention | −1.48 ± 3.54 (**0.4σ**) | −1.48 ± 0.31 (**4.8σ**) | **11.5×** |
| replay − backprop, crossover | +3.26 ± 1.10 (3.0σ) | +3.26 ± 0.39 (8.3σ) | 2.8× |
| replay − backprop, retention | +30.65 ± 2.82 (10.9σ) | +30.65 ± 2.22 (13.8σ) | 1.3× |

Point estimates identical; only uncertainty differs. Pairing helps in proportion to how similarly
the rules respond to the split — PC tracks backprop at +9.5 vs +9.7 points/sd so nearly all
cancels (11.5×); replay is half as sensitive at +4.0 so little does (1.3×). Unpaired at 5 seeds,
the **positive control** reads 0.7σ and looks like a failure.

**Plots.** `60_....png` — *why evidence:* left column is the pairing (sloped), right column the
control (flat); the contrast is the result, not either panel alone.
`60_..._digits.png` — *why evidence:* one row per seed ordered by similarity, the two digits
sharing each output unit drawn adjacent, so the correlation can be checked against the images.

*Pending:* `69` measures whether Class-IL carries an analogous confound.

## 10. Repeated alternation converges, equally for all rules

20 alternating blocks × 150 updates, Domain-IL, EqProp excluded on cost:

| | crossover, cycle 1 → 10 | [R1] mean error | successive same-task W1 distance, last ÷ first |
|---|---|---|---|
| backprop | 53.9% → 81.3% | 23.7% | 0.23 |
| replay | 28.6% → 79.3% *(n=29/100)* | 23.7% | 0.28 |
| pc | 54.2% → 81.7% | 23.6% | **0.17** |

Converging, not ping-ponging: the (task 1, task 2) trajectory spirals inward with monotonically
tightening arcs from (25,25) to ≈(85,85), and the weight distance between successive same-task
states falls to 0.17–0.28 of its first-cycle value. Two independent readouts — one accuracy, one
weight space. Most of the gain arrives in the first three cycles. **No rule separates.**

**Plots.** `68_..._spiral.png` — *why evidence:* colour is time, so tightening arcs show
convergence directly; a closed loop would have shown ping-ponging and does not appear.

---

## Controls and objections closed

- **Replay separates everywhere** — +26.6 retention (6.6σ) in `52`, 24W–0L crossover (p<0.001) in
  `60`, 8/8 cells in `59`. The problem is solvable; the EBMs fail at something achievable.
- **PC settles, and its inference step rule is equivalent to [R1]'s.** ≤18 steps needed, 50 run,
  3.2e-04 from equilibrium. Their backtracking rule and our fixed step reach the same fixed point
  (gap 5e-06 to 9e-05), `cos(ΔW) = 1.000000` at every step count 1–200, largest metric-grid
  disagreement **1.4e-03 accuracy points**. → `50`, `63`.
- **Capacity is not a confound** — H=32 is past the knee (`41`) and H=128 changes nothing (`59`).
- **The earlier experiment-12 result is not supported.** Re-run verbatim (`61`): crossover
  backprop 64.0, pc 59.5, **replay 48.7 (−14.14, 5.2σ)** — the positive control inverts, because
  at 100 updates per task nothing has converged and the ordering is set by task-2 learning speed.
  Separately, that specification gave backprop and replay a task-restricted softmax — zero
  gradient on absent classes — that PC and EqProp did not have, which removes the dominant
  Class-IL forgetting mechanism for two of four rules. Both are properties of that configuration.

## Not claimed

- Any ranking on **half-life in Domain-IL**: defined on 1–2 of 5 seeds under a single switch and
  ~1 of 100 block×seed cells under alternation, because task 1 often never loses half its value.
- **Inefficiency predicts forgetting within a rule.** r = +0.81 to +0.89, but pairing similarity
  correlates 0.90–0.97 with retention and 0.50–0.75 with inefficiency across all four rules, so it
  is a common cause. Needs 24 seeds with the pairing partialled out.
- **A replication of [R1].** Their setup was not run: schedule, resolution, activation, depth,
  batch size and loss reduction all differ. The claim is about a controlled comparison in which
  the learning rule is the only thing that varies.

## Open

| | question |
|---|---|
| `66` | does the weight-space route differ between Class-IL and Domain-IL? *(running)* |
| `67` | does the Class-IL suppression decomposition hold for all four rules? *(queued)* |
| `69` | is Class-IL confounded by the split as Domain-IL is? *(running)* |
| — | concept drift ([R1] Fig 4f–g), where their claimed advantage is largest |
| — | depth in Class-IL, the one scenario where PC shows an effect |
| — | magnitude-aware interference, `(d_learn · d_target) / \|d_target\|²` |
