# Presentation plan 2 — built from `findings.md`

One slide per section of `findings.md`. **Budget per slide: 3 bullets, 3 data items.** Data items
are tables, plots or diagrams and may not all be the same type. Every data item on a slide must
evidence the claim, and the claim is the title. Where one item does the job, only one is listed;
anything marked *supplementary* is for the appendix or for answering a question, not for the slide.

Each slide carries a **script** — what is said while it is up. Keep to it; the slide is deliberately
under-filled so the script carries the detail.

## Two structural decisions, both as requested

**1. Failed metrics come first.** Slides 2 → 3 → 4 run [R1]'s metric, then endpoint metrics, then
crossover. The argument is *"here is what the existing measures say, here is why they cannot be
trusted, here is one that can"* — so the metric that survives arrives last and inherits the
motivation, rather than being asserted up front.

**2. Domain-IL vs Class-IL is confined to slide 5.** It cannot be removed entirely — the headline
answer is genuinely scenario-dependent, so hiding it would misreport the result. What is done
instead: the two scenarios are *named once* on slide 1 as part of the design, appear on slides 2–4
only as **columns of a table with no discussion**, and are **explained once** on slide 5, placed
immediately after the slide that first makes the answer depend on them. Slides 6–11 refer back
rather than re-arguing. That is the tightest confinement the result allows.

**One judgement call flagged rather than made silently.** Slide 8 (the data split) is methodology as
much as it is a finding — every σ in the deck depends on paired comparison, which slide 8 justifies.
It sits at position 8 so the results land before the caveats. If the audience is
methodology-first, move it to position 2; nothing else has to change.

---

## Slide 1 — Do energy-based learning rules forget less than backpropagation?

**Bullets**
- Four rules, one protocol, nothing else varying: **backprop** (negative control), **replay**
  (positive control), **predictive coding**, **equilibrium propagation**.
- MNIST 14×14, 2 tasks × 5 classes, 196–32–out, tanh, squared error, plain SGD, batch 32.
  Learning rate grid-searched **per rule** and matched on updates-to-90% on task 1.
- Every number in this deck is a **paired difference against backprop at the same seed**, in
  standard errors of that difference (σ). 5 seeds unless stated.

**Data**
1. *(diagram, primary)* The 2 × 2 design grid — rows: Domain-IL (5 shared output units) /
   Class-IL (10 separate); columns: fixed budget / matched task-2 competence. Cells labelled
   `52`, `53`, `56`, `57`. **Why evidence:** it shows that scenario and reading point are the only
   two things varied, so any difference later is attributable.

**Script.** The question is whether learning rules derived from energy minimisation forget less than
backpropagation. Prior work says yes. To test it, the learning rule has to be the only thing that
changes — so everything else is fixed: same data, same architecture, same optimiser, same batch
size. The one thing that cannot be held fixed is the learning rate, because the rules are not
comparable at the same value, so it is grid-searched per rule and then matched on how many updates
each rule needs to reach 90% on the first task. Backprop is the negative control, replay the
positive control. Two scenarios and two points at which the result is read give four experiments.
Everything reported is a per-seed paired difference, for a reason I will come to on slide 8.

---

## Slide 2 — Song & Bogacz's metric cannot verify their claim

**Bullets**
- [R1] report **mean test error over training**, not a final accuracy — their claim lives in that
  number, so it has to be tested in that number.
- The metric sums two different things: **learning the new task quickly** and **not destroying the
  old one**. It cannot say which produced a given value.
- PC's task-1 and task-2 halves have **opposite signs in all four cells**, so averaging them removes
  the effect.

**Data**
1. *(table, primary)* PC − backprop, mean test error, negative = PC better:

   | | task 1 | task 2 |
   |---|---|---|
   | Domain-IL, fixed | +2.97 (2.7σ, worse) | −2.05 (3.0σ, better) |
   | Domain-IL, matched | −1.60 (2.0σ) | +0.97 (1.2σ) |
   | Class-IL, fixed | +3.24 (5.5σ, worse) | −2.27 (5.9σ, better) |
   | Class-IL, matched | −5.48 (2.6σ, better) | +0.92 (0.9σ) |

   **Why evidence:** the two columns disagree in sign in every row. A single mean over both tasks
   is the average of those two columns.
2. *(plot, supplementary)* `68_what_happens_under_repeated_task_switching.png`, mean-error panel —
   under [R1]'s own 20-block alternating schedule the three rules' curves superimpose:
   **backprop 23.7%, replay 23.7%, pc 23.6%**. **Why evidence:** their metric, their schedule, no
   separation.

**Script.** Song and Bogacz's headline number is the mean test error across training, not the
accuracy at the end. So that is where their claim has to be tested. The problem is structural: that
one number rewards learning the new task fast and rewards not destroying the old one, and it cannot
distinguish them. A rule that simply learns task 2 faster wins on it without retaining anything
extra. In our runs that is exactly what happens — in all four cells, PC is better on one task and
worse on the other, so the average cancels. And when we run their alternating schedule and compute
their metric, the three rules come out at 23.6, 23.7 and 23.7. Under their own measure, on their own
schedule, there is nothing to separate.

---

## Slide 3 — Endpoint metrics flip sign with the reading point

**Bullets**
- Final accuracy, ACC, BWT and "forgetting" are all read **when training stops**, so their value is
  set by how long task 2 ran — they are rearrangements of the same endpoint.
- Same runs, same rules: PC is **worse at a fixed budget and better at matched competence**, in both
  scenarios. Nothing about the learning rule changed between the two readings.
- Known effect: **setup-induced forgetting** (Michel et al. 2023, arXiv:2309.00462).

**Data**
1. *(table, primary)* PC − backprop, **final task-1 accuracy**:

   | | fixed budget | matched competence |
   |---|---|---|
   | Domain-IL | **−2.56** (2.7σ) | **+2.48** (1.3σ) |
   | Class-IL | **−2.04** (4.0σ) | **+1.60** (0.7σ) |

   **Why evidence:** the sign reverses across the columns; the columns differ only in when the
   number was taken.
2. *(table, primary)* The same effect isolated on a single intervention — freezing the hidden layer,
   minus control (`42`, `43`):

   | reading point | value |
   |---|---|
   | 50 updates after the switch | **+31.2** |
   | 200 updates | +12.0 |
   | 2000 updates (saturation) | **−0.2** |
   | at matched task-2 competence | **+17.4** |

   **Why evidence:** one intervention, one set of runs, four readings spanning +31 to −0.2. The
   reading point alone produces the whole range.

**Script.** The metrics used almost everywhere in continual learning — final accuracy, ACC, backward
transfer, "forgetting" — are all read at the moment training stops. That makes them a function of
the training budget as much as of the rule. Here is what that does. At a fixed budget PC learns task
2 further, so it displaces more of task 1, and reads as worse. Hold both rules to the same task-2
standard instead, and the advantage disappears and the sign reverses. Same runs. The second table
isolates it without any rule comparison at all: freezing the hidden layer is worth 31 points shortly
after the switch and exactly nothing at saturation. The intervention did not change — only when we
looked. This is documented in the literature as setup-induced forgetting.

---

## Slide 4 — Under a metric with no budget in it, PC does not reduce forgetting

**Bullets**
- **Crossover height** = the accuracy at which the task-1 and task-2 curves intersect. No budget and
  no threshold in its definition, and invariant to rescaling time — so a merely *faster* rule cannot
  win it.
- **The sign does not change** between the two reading points that flipped every endpoint metric.
- PC is indistinguishable from backprop in Domain-IL and **+1.3 to +3.0 in Class-IL**, against
  replay's **+2.6 to +11.7** in the same cells.

**Data**
1. *(plot, primary)* `52_do_the_rules_differ_fixed_budget.png` — four panels, one per rule, dotted
   horizontal line = that panel's crossover height. **Why evidence:** it shows what the metric *is*,
   on real curves, and the four panels are the four rules under identical everything else, so the
   dotted lines are directly comparable. PC's sits on backprop's.
2. *(table, primary)* PC − backprop, crossover height:

   | | fixed budget | matched competence |
   |---|---|---|
   | Domain-IL | −0.01 (0.0σ) | +0.49 (2.0σ) |
   | Class-IL | **+1.29** (3.7σ, 5W–0L) | **+3.00** (2.2σ, 4W–1L) |
   | *replay, same cells* | +2.56 / +7.32 | +2.75 / +11.67 |

   **Why evidence:** compare against slide 3's table — same runs, same two reading points, and here
   the sign is stable. Class-IL replicates at **24 seeds: +1.65 ± 0.25 (6.6σ)**.

**Script.** If the endpoint depends on the budget, use something that does not have a budget in it.
Crossover height is where the two accuracy curves cross — the first task falling, the second
rising. It is a geometric feature of the trade-off between them, so there is no stopping time and no
threshold in the definition, and if you speed a rule up uniformly the crossing point does not move.
Read this way, the answer stops moving too: PC sits exactly on backprop in Domain-IL and about one
to three points above it in Class-IL, and that holds at both reading points. For scale, replay — the
positive control — is between two and twelve points in the same cells. So the problem is solvable
and PC recovers a small fraction of what is available. One caveat: crossover is undefined when the
curves never meet, which means task 1 stayed above task 2 throughout. That is the best possible
outcome, not missing data, so those runs are ranked top by a sign test rather than dropped.

---

## Slide 5 — The two scenarios forget by different mechanisms, and only one is a mechanism a rule can act on

**Bullets**
- **Class-IL forgetting is output-layer suppression**: the hidden representation survives — nearest
  class mean reads 71.7% — while the network's own argmax reads 0.2%.
- **Domain-IL cannot suppress.** Every output unit is a target for some class, so what remains is
  representation drift.
- PC's only gain is in the scenario that *has* a suppression mechanism to avoid. Whether that is the
  **cause** is tested by `67` and is **not yet established** (slide 9).

**Data**
1. *(plot, primary)* `43_how_much_room_at_matched_competence.png`, argmax-against-NCM panel.
   **Why evidence:** points far above the diagonal mean the hidden code survived and the output
   layer alone failed — that gap *is* the suppression signature, read directly rather than inferred.
2. *(table, supplementary)* Task-1 accuracy read two ways, backprop, Class-IL:

   | | network's argmax | nearest class mean |
   |---|---|---|
   | at saturation (`42`) | **0.2%** | **71.7%** |
   | at matched competence (`43`) | 35.7% | 79.3% |

**Script.** The answer on the last slide depended on the scenario, so this slide is why. In Class-IL
every class has its own output unit, and while task 2 trains, the task-1 units get pushed down until
they cannot win an argmax any more. You can show that the information is still there: classify by
nearest class mean in hidden space instead of using the network's own output layer, and task 1 reads
about 72% while the network itself reads 0.2%. The representation survived; the readout failed. In
Domain-IL that cannot happen, because the five output units are shared — every unit is the right
answer for some class in both tasks, so nothing can be suppressed. What is left there is the hidden
representation drifting. Those are different failures, and a learning rule has something to act on
in the first and much less in the second. That is the shape of our result — but I want to be careful:
this makes PC's Class-IL gain *consistent* with reduced suppression, not caused by it. Slide 9 is
where that gets tested, and it does not come out the way I expected.

---

## Slide 6 — EqProp is worse than backprop under every metric, in every cell, on every seed

**Bullets**
- Worse on crossover in all four cells; **0W–5L across seeds** in three of them. Also worse on
  endpoint retention, on forgetting, on BWT and on task-1 mean error.
- **Impact of metric choice: none.** The sign is identical under every metric — which is what makes
  this a finding rather than a null.
- Mechanism: its updates are nearly **orthogonal** to backprop's (cos 0.197–0.316 on W1) and it
  travels **48% further to arrive 35% short**.

**Data**
1. *(table, primary)* EqProp − backprop:

   | | Domain-IL | Class-IL |
   |---|---|---|
   | crossover, fixed / matched | −1.22 (1.6σ) / −1.98 (3.8σ) | **−8.79** (13.6σ) / −5.13 (3.6σ) |
   | W1 path length (`65`) | 182.1 vs backprop's 123.2 | |
   | W1 net displacement | 28.3 vs backprop's 43.8 | |
   | inefficiency = path ÷ net | **4.56 vs 2.51** (13.0σ) | |

   **Why evidence:** the top row is the outcome under the metric that survived slides 2–4; the lower
   rows are the weight-space reason, measured on the same runs.

**Script.** Equilibrium propagation is the clear negative. It is worse than backprop under every
metric we computed, in every cell, and on every individual seed in three of the four cells. Unlike
everything else in this deck, the choice of metric makes no difference at all here — which is
precisely what makes it a result rather than an absence of one. The reason is visible in weight
space: its updates are close to perpendicular to backprop's, and it travels about half again as far
through weight space to end up a third closer to where it started. It is not taking a different good
route; it is wandering.

---

## Slide 7 — Depth and width do not rescue PC

**Bullets**
- Depth {1, 2, 3, 5} × width {32, 128}, learning rates **recalibrated per cell**, Domain-IL.
- **PC separates in 0 of 8 cells** (spans −1.75 to +0.52, no cell above 1.6σ). **Replay separates in
  8 of 8** (spans +2.24 to +4.42).
- At H = 128 — four times the capacity — PC is still never significant, so the main result is **not
  a capacity ceiling**.

**Data**
1. *(plot, primary)* `59_does_pc_help_in_a_bigger_network_retention_*.png` — one panel per cell.
   **Why evidence:** PC (red) lies on backprop (grey) in all eight panels while replay (brown) sits
   clearly above in all eight. The comparison is visual and needs no threshold.
2. *(table, supplementary)* Per-cell PC crossover minus backprop: 1×32 +0.37, 1×128 −0.14,
   2×32 +0.48, 2×128 +0.52, 3×32 −0.05, 3×128 +0.39, 5×32 −1.75, 5×128 −0.29.

**Script.** The obvious objection to a null result is that the network was too small — that PC needs
depth for its layer-wise inference to do anything, or that a 32-unit hidden layer was saturated. So
we ran the grid: four depths, two widths, learning rate re-searched in every cell so no cell is
handicapped. PC separates in none of the eight. Replay separates in all eight, which tells us the
test had the power to detect an effect of the size we care about. And at 128 hidden units — four
times the capacity — PC still does nothing, so the main result is not an artefact of running out of
room.

---

## Slide 8 — Which two digits share an output unit matters more than which learning rule is used

**Bullets**
- In Domain-IL every output unit carries one task-1 digit and one task-2 digit. **Which two digits
  are paired explains ~62% of the seed variance** in retention: r = +0.787, range **23% – 71%** —
  larger than every rule difference in this deck.
- It is **interference, not learnability**: every seed learned task 2, and the correlation with
  learning speed is null. The control — same digits, pairing ignored — is flat.
- It is **cancelled, not removed**. All rules see the same split at the same seed, so a per-seed
  difference subtracts it. **Unpaired at 5 seeds, the positive control reads 0.7σ and looks like a
  failure.**

**Data**
1. *(plot, primary)* `60_why_does_the_seed_matter_so_much_digits.png` — one row per seed, ordered by
   pairing similarity, with the two digits that share each output unit drawn adjacent.
   **Why evidence:** the predictor is shown as the actual images, so the correlation can be checked
   by eye against the retention number on each row rather than taken on trust.
2. *(table, primary)* Effect of pairing the comparison (`69`, 24 seeds, Domain-IL):

   | | unpaired | paired | sem ratio |
   |---|---|---|---|
   | pc − backprop, retention | −1.48 ± 3.54 (**0.4σ**) | −1.48 ± 0.31 (**4.8σ**) | **11.5×** |
   | replay − backprop, retention | +30.65 ± 2.82 (10.9σ) | +30.65 ± 2.22 (13.8σ) | 1.3× |

   **Why evidence:** the point estimates are **identical** — only the uncertainty differs. That
   isolates the effect of the comparison design from any effect of the data.
3. *(plot, supplementary)* `60_why_does_the_seed_matter_so_much.png` — left column pairing (sloped),
   right column control (flat).

**Script.** This is the one that changed how we run everything else. In Domain-IL the five output
units are shared, so each one is the target for one digit in task 1 and a different digit in task 2.
It turns out that which two digits get paired onto a unit explains about 62% of the variance between
seeds — retention ranges from 23% to 71% depending only on the split. That is a bigger effect than
any difference between learning rules we have measured. It is specifically interference and not
difficulty: every seed learned the second task fine, and if you shuffle which digits are in which
task while ignoring the pairing, the effect vanishes. We do not remove this — we cancel it. Because
every rule sees the same split at the same seed, taking the difference within a seed subtracts it
out. The table shows what that is worth: the same estimate goes from 0.4 sigma to 4.8. Run unpaired
at five seeds, replay — the positive control, a rule we know works — reads 0.7 sigma and would have
been called a failure. Sample size was never the problem.

---

## Slide 9 — The rules do compute different updates, but at the layer that does not govern forgetting

**Bullets**
- PC's updates diverge from backprop's **most at the output layer** — cos 0.985 on W1 against 0.814
  on W2 — and its W2 route is **31% shorter while displacing 22% further**.
- **W1 is the layer whose drift damages task 1.** PC improves the layer that does not govern
  forgetting and is fractionally clumsier in the one that does.
- The W2 advantage is **twice as large in Domain-IL, where PC gains nothing**, as in Class-IL, where
  it gains — **the opposite of the pre-registered prediction**.

**Data**
1. *(plot, primary)* `66_does_the_weight_route_differ_between_scenarios_mechanism.png` — each panel
   is one layer's actual route through weight space for one rule, with three net displacements drawn
   (init→switch, switch→end, init→end). **Why evidence:** it draws the measured quantity instead of
   summarising it — the chord being shorter than the two legs summed *is* the retracing. Rows are
   scenario × layer, so PC's near-straight Domain-IL output route and its ordinary Class-IL one are
   side by side.
2. *(table, primary)* PC − backprop at W2 (`66`), against the outcome:

   | | Domain-IL | Class-IL | difference |
   |---|---|---|---|
   | full-run path ÷ net | **−3.509 ± 0.115** | −1.910 ± 0.122 | +1.599 (9.5σ) |
   | fraction retraced | **−0.353 ± 0.010** | −0.147 ± 0.015 | +0.206 (11.5σ) |
   | *crossover vs backprop* | *−0.01 (0.0σ)* | *+1.29 (3.7σ)* | |

   **Why evidence:** the mechanism is strongest in the column where the outcome is zero. Read down
   the columns, not across.

**Script.** So the rules do genuinely compute something different — this is not a case of four
implementations of the same update. PC's updates depart from backprop's mainly at the output layer,
and its route through output-layer weight space is markedly more efficient: it takes a path about a
third shorter and ends up further from where it started. The trouble is which layer that is. The
layer whose drift damages the first task is the hidden layer, and there PC is fractionally worse
than backprop, not better. And then this experiment, which I set up expecting to confirm the story
from slide 5. If PC's output-layer efficiency were what produces its Class-IL gain, the effect should
be largest in Class-IL, because that is where the output layer does the damage. It is half the size
there. The biggest version of the mechanism sits in Domain-IL, where PC gains exactly nothing. That
is not consistent with the mechanism causing the result, so I am not claiming it does.

---

## Slide 10 — Song & Bogacz's own mechanism measure does not track forgetting

**Bullets**
- **Target alignment** = cos(target − output before, output after − output before). PC is **not more
  aligned than backprop**: +0.003 on a 0.277 baseline — 1% relative. [R1]'s mechanism does not
  reproduce at this scale.
- **The measure inverts at the extreme.** EqProp's updates move task-1 outputs *toward* their targets
  on average (+0.039, best of the four) — and EqProp forgets the most.
- **Because alignment is direction-only.** EqProp's step is **4–6× larger** and its angle is ~90°:
  perpendicular, doing no directional harm, while displacing task 1 furthest. Damage is direction ×
  magnitude, and a cosine cannot see magnitude.

**Data**
1. *(plot, primary)* `64_target_alignment_and_interference_mechanism.png` — the plane is spanned
   exactly by `d_target` and `d_learn`. **Why evidence:** because that plane is the span, **the angle
   drawn on the slide is the number reported**, with no projection loss. The bottom row shows the
   same update as task 1 sees it — replay's arrow is acute because it rehearses task 1, every other
   rule's is obtuse, which is the instrument's own sanity check.
2. *(table, supplementary)* Per rule: alignment / interference on task 1 / step-size ratio —
   backprop +0.277 / −0.030 / 0.127 · replay +0.181 / −0.000 / 0.172 · pc +0.280 / −0.025 / 0.201 ·
   eqprop +0.250 / **+0.039** / **0.834**.

**Script.** Song and Bogacz explain their result with a mechanism: that predictive coding's updates
point closer to the target than backprop's do, so they interfere less. We measured that directly.
PC's alignment is 0.280 against backprop's 0.277 — three thousandths, about 1% relative. It reaches
statistical significance only because it is averaged over roughly 130 measurements per run, which is
worth saying out loud: significant and negligible at the same time. Worse, the measure inverts. On
this metric EqProp looks best of the four — its updates move the old task's outputs toward their
targets on average — and EqProp is the rule that forgets most. The reason is that a cosine only sees
direction. EqProp's step is four to six times larger than anybody else's, sitting at right angles: it
does no directional harm while dragging the first task further than any other rule. Damage is
direction times magnitude, and this measure only has the first factor.

---

## Slide 11 — Under repeated alternation, all rules converge to a joint solution equally

**Bullets**
- 20 alternating blocks × 150 updates, Domain-IL: crossover rises **53.9 → 81.3** (backprop) and
  **54.2 → 81.7** (PC) over cycles 1 → 10. **No rule separates.**
- **Converging, not ping-ponging** — two independent readouts: the trajectory spirals inward with
  monotonically tightening arcs, and the weight distance between successive same-task states falls
  to **0.17 – 0.28** of its first-cycle value.
- In [R1]'s metric on [R1]'s schedule: **23.7 / 23.7 / 23.6**. Most of the gain arrives in the first
  three cycles.

**Data**
1. *(plot, primary)* `68_what_happens_under_repeated_task_switching_spiral.png` — task-1 accuracy
   against task-2 accuracy, colour = time. **Why evidence:** the arcs tighten monotonically from
   about (25, 25) to about (85, 85). A rule that ping-ponged between two solutions would trace a
   closed loop; none does, and the three rules' spirals lie on each other.

**Script.** One more way the question gets asked in the literature: what if you alternate between
the tasks repeatedly rather than switching once? We ran twenty blocks. All three rules converge —
crossover climbs from about 54 to about 81 — and they do it at the same rate and to the same place.
The spiral plot is the clearest way to see it: each loop is one pair of blocks, and the arcs tighten
inward toward the joint solution in the top right. If the network were merely ping-ponging between a
task-1 solution and a task-2 solution, that would be a closed loop that never contracts, and it is
not. The weight-space measurement agrees independently: the distance between successive task-1
states drops to about a fifth of what it was. Under Song and Bogacz's own metric on this schedule,
the three rules read 23.6, 23.7 and 23.7.

---

## Slide 12 — Controls, and the objections that are closed

**Bullets**
- **Replay separates everywhere** — +26.6 retention (6.6σ), 24W–0L on crossover (p < 0.001), 8 of 8
  depth×width cells. The problem is **solvable**, so the energy-based rules fail at something
  achievable.
- **PC settles, and its inference rule is equivalent to [R1]'s.** cos(ΔW) = **1.000000** at every
  step count 1–200; largest metric-grid disagreement **1.4 × 10⁻³ accuracy points**.
- **Capacity is not a confound** (H = 32 is past the knee, H = 128 changes nothing), and **the
  pipeline is deterministic across scripts** — `66` reproduced `52` and `56` **bit-identically**,
  max difference 0 over all 8 rule × scenario cells.

**Data**
1. *(table, primary)* One row per objection: *"too few seeds"* → replay at 24W–0L, p < 0.001;
   *"PC not converged"* → ≤18 settling steps needed, 50 run, 3.2 × 10⁻⁴ from equilibrium;
   *"wrong PC variant"* → their backtracking rule and our fixed step reach the same fixed point,
   gap 5 × 10⁻⁶ to 9 × 10⁻⁵; *"network too small"* → 8 of 8 cells, H = 128 included;
   *"code not trusted"* → independent re-run reproduces bit-identically.
   **Why evidence:** each row is the specific measurement that closes that specific objection, not a
   general reassurance.

**Script.** Briefly, the things that could have made all of this an artefact, and why they do not.
The positive control works everywhere, which is the single most important line on this slide — it
means the experiment can detect the effect we are looking for, so a null is informative. PC really
does reach equilibrium: it needs at most eighteen settling steps and we run fifty. We also
implemented Song and Bogacz's exact inference step rule and compared: the two reach the same fixed
point, and the resulting weight updates have a cosine similarity of one to six decimal places. The
network is not too small. And since the codebase had a history of not being trusted, one experiment
independently re-ran two others from scratch and reproduced them bit for bit.

---

## Slide 13 — What is not claimed

**Bullets**
- **The earlier experiment-12 PC advantage is not supported.** Re-run verbatim, the **positive
  control inverts** — replay −14.14 (5.2σ) — because at 100 updates per task nothing has converged
  and the ordering is set by task-2 learning speed. Separately, that specification gave backprop and
  replay a task-restricted softmax that PC and EqProp did not have, removing the dominant Class-IL
  forgetting mechanism for two of the four rules.
- **No ranking on half-life in Domain-IL** (defined on only 1–2 of 5 seeds), and **inefficiency
  predicts forgetting** is confounded by the slide-8 pairing and needs 24 seeds with it partialled
  out.
- **This is not a replication of [R1].** Schedule, resolution, activation, depth, batch size and loss
  reduction all differ. The claim is about a controlled comparison in which the learning rule is the
  only thing that varies — not that their figures are wrong.

**Data**
1. *(table, primary)* `61`, the earlier specification re-run as written — crossover: backprop 64.0,
   pc 59.5, **replay 48.7 (−14.14, 5.2σ)**. **Why evidence:** the positive control coming out worst
   is a self-diagnosing failure. No knowledge of PC is needed to see the configuration is invalid.

**Script.** Three things I am deliberately not claiming. First, an earlier experiment in this project
appeared to show a PC advantage, and it does not survive. Re-running that exact configuration, replay
— the rule we know rehearses the old task — comes out worst of the three. When the positive control
inverts, the configuration is measuring something other than forgetting; here it is measuring how
fast each rule learns task 2, because at a hundred updates per task nothing has converged. There was
also a second, independent defect in that setup. Second, some measurements are not yet clean —
half-life is undefined on most Domain-IL seeds, and the correlation between weight-path efficiency
and forgetting is confounded by the digit pairing from slide 8. Third and most important: this is not
a replication. We did not run their setup. The claim is narrower and, I think, more useful — under a
protocol where the learning rule is the only thing that varies, the advantage does not appear.

---

## Slide 14 — Answers

**Bullets**
- Under a metric that cannot be won by simply training the second task further, **predictive coding
  does not reduce forgetting in Domain-IL** at any depth or width, and gains **1–3 points in
  Class-IL** against replay's 7–12 in the same cells. **Equilibrium propagation is consistently
  worse.**
- **The choice of metric decides the published answer**: [R1]'s mean error cancels the effect,
  endpoint metrics flip sign with the reading point, and only crossover gives the same answer at both.
- **The largest single effect measured in this project is not a learning rule.** It is which two
  digits share an output unit — 62% of seed variance.

**Data**
1. *(table, primary)*

   | question | answer |
   |---|---|
   | Does PC reduce forgetting? | **No in Domain-IL**, at any depth or width. **+1–3 points in Class-IL**, against replay's +7–12. |
   | Does EqProp? | **No — consistently worse.** 0W–5L across seeds in three of four cells. |
   | Does [R1]'s metric verify their claim here? | **No**, and it cannot — it sums two effects it cannot separate. On their schedule: 23.6 / 23.7 / 23.7. |
   | Do endpoint metrics? | They flip sign with the reading point and are degenerate in Class-IL. |
   | Does the scenario matter? | Yes — Class-IL forgets by output suppression, which a rule can act on; Domain-IL cannot suppress. |
   | Does the data matter? | **More than the rule.** 62% of seed variance; cancelled by pairing, not removed. |

**Script.** To close. Under the one metric that gives a stable answer, predictive coding does not
reduce forgetting in the scenario Song and Bogacz use, at any depth or width we tried, and produces a
small gain of one to three points in the other scenario — against a replay baseline of seven to
twelve in the same cells. Equilibrium propagation is worse than backprop under every metric. The
second point is the methodological one and probably travels further than the first: which measure you
choose decides the answer you publish. Their metric averages the effect away, endpoint metrics give
you either sign depending on when you stop, and only a budget-free geometric measure is stable. And
the third is the one I did not expect going in — the largest effect in this entire project is not any
learning rule. It is which two digits happen to share an output unit.

---

## Coverage check against `findings.md`

| findings.md | slide |
|---|---|
| header — the experiment | 1 |
| C3 · §3 Song & Bogacz's metric | 2 |
| C2 · §2 endpoint metrics | 3 |
| C1 · §1 crossover height | 4 |
| C6 · §6 the scenario | 5 |
| C4 · §4 EqProp | 6 |
| C5 · §5 depth and width | 7 |
| C9 · §9 the data split | 8 |
| C7 · §7 output-layer mechanism | 9 |
| C8 · §8 target alignment | 10 |
| C10 · §10 repeated alternation | 11 |
| Controls and objections closed | 12 |
| Not claimed | 13 |
| — (new) | 14 |

All ten claims are covered. **Open** is deliberately not a slide — `67` is running and would
date the deck; fold its result into slide 5 or 9 when it lands, or add it as a closing "next" slide
if the talk wants one.

## Figures that need to exist before this is built

All cited figures exist except one gap:

- **Slide 1's design diagram** has to be drawn — it is the only item in the deck that is not
  already generated by a script.
- Slide 9 cites `66_..._mechanism.png`, which exists but is **4 rows × 4 columns** — too dense to
  project. It needs a cut-down version: PC and backprop only, W2 only, both scenarios (2 × 2).
- Slide 5 cites the argmax-against-NCM panel of `43_...png`, which is one panel of a multi-panel
  figure and will need extracting.
