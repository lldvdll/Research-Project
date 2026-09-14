# Plot walkthrough — the report read through its own figures

Purpose: read the report's actual argument, figure by figure, in the order it will appear
(M1–M3, R1–R5, D1), so the STORY can be checked before the writing pass starts. Not a caption
draft — a narrative check. Status tags: **final** (numbers locked — as of 2026-09-11 every
figure is final, since nothing is still running), **decision needed** (a real interpretive call
not yet made, and the only kind of open item left in this file).

Started 2026-09-10 with two jobs still running; **completed 2026-09-11 for the 800/900 BUILD
pass — every plot in `script_plan_800_900.md` was produced.** The status tags that remain here
are **decision needed** ones — interpretive calls for the writing pass, not missing numbers.

⚠ **"Nothing is waiting on compute" describes the build pass, not today.** The report-revision
pass that followed has added new runs (810, 811) and new figures (910), and has re-run others.
For what is actually outstanding, read `now.md` and `report_progress.md`; this file is the
argument each figure makes, not a status board.

---

## Methods

**The methods section's job**: justify every setup choice with a measurement, not a convention.
Nothing here argues the thesis; everything here removes an objection before R1 can be read.

- **901 — What forgetting looks like.** One seed, ten class lines, Class-IL, run to saturation
  on purpose (task 1 → 0.0%, not the matched-competence 5.4% R1 reports — the caption must say
  this or the report looks self-contradictory). Purely illustrative: this is what the reader
  should picture every time "forgetting" is mentioned afterward. **Final.**
- **902 — Architecture justification.** Two panels: (a) frozen random trunk vs. fully trained,
  both scenarios — training the hidden layer is worth ~12–14 points, so it isn't a dead layer;
  (b) joint accuracy vs. width — H=32 sits just off the capacity bottleneck (89.7%/90.65% at H=32
  vs. 90.3–90.9% at H≥64, essentially flat above it). **Known gap, not fixed this session**: both
  panels are backprop-only — H=32 was never checked against PC's own capacity needs. Worth a
  sentence acknowledging this as a limitation rather than silently assuming it transfers. **Final
  numbers, flagged limitation.**
- **903 — Specification.** Activation and output-loss choices. This is where the tanh-vs-sigmoid
  question used to live as a pure methods justification — **decision needed**: given 802/808 now
  make the sigmoid result a real Results finding (R2/R3), does 903 still carry the raw activation
  sweep, or does it just state "tanh is the default, robustness checked in R2" and point forward?
  My recommendation: the latter — don't duplicate the sigmoid story in Methods once it has a real
  home in Results.
- **904 — Settling reaches the same fixed point across dt.** Copy of 334. dt changes the route,
  not the destination, up to dt≈0.5. Justifies dt=0.4 as the default. **Final.**
- **905 — The stable dt band shrinks with depth.** Copy of 347. The dt×depth stability grid —
  dt=0.4 is the only unstable cell, at depth≥2. This is the figure that licenses the dt=0.2
  correction used in 341/342/807/809. **Final.**
- **906 — What pairing on seeds buys.** Same-seed comparison variance vs. group-mean variance —
  the empirical case for every paired statistic used from here on. **Final.**
- **907 — Why competence-matched stopping.** Merge of 111+343: endpoint metrics invert sign
  across stopping points; crossover doesn't. The single most load-bearing methods figure — R2
  through D1 all depend on the reader accepting this panel. **Final.**
- **908 — Metric survival across stopping points.** Copy of 112. Only crossover holds its sign
  across every threshold tested, at 2 SEM. **Final.**
- **909 — Crossover vs. retention** (new this session, not in the original 18). Pooled across
  340/341/342/802 (~500 points/scenario): within-scenario r = 0.65 (Class-IL) / 0.51 (Domain-IL).
  Correlated, not redundant — the evidence for why the report needs both metrics rather than
  picking one. **Decision needed**: this reads as a methods figure (justifies dual-metric
  reporting), so it likely belongs near 907/908, not as a standalone — candidate for merging into
  that pair rather than keeping its own slot, given the 18-figure budget.

**Methods, read start to finish**: the architecture is adequate (902), the settle control is
justified and its failure mode is known (904/905), pairing is why any of this has power (906),
and the metric is chosen because the alternative lies (907/908). Nothing here has said anything
about PC yet — that's deliberate, R1 hasn't happened.

---

## Results

### R1 — Two scenarios, two forgetting phenotypes

- **911 — the hinge figure.** 2×2, scenario across, rule down. Class-IL final t1 5.4%, Domain-IL
  38.5%; crossover 65.0% vs. 75.8%. The STRUCTURAL argument (Class-IL has 5 untargeted output
  units, suppression is available; Domain-IL's 5 shared units make it impossible) matters more
  than the numbers and has to come first in the caption. **Final.** One honest qualifier already
  written into the section: the argmax-minus-probe gap is large in BOTH scenarios (+81.6 /
  +29.5), even though suppression can't happen in one of them — R5 has to return to this, it
  can't be quietly dropped.

R1's whole job is to earn the right to split everything downstream by scenario. It does.

### R2 — The size of the learning-rule effect

- **912 — the headline.** Four panels now, not three: lr, width, depth, **and activation**
  (802 folded in this session). Reading: Class-IL sits above zero everywhere; Domain-IL sits
  below zero everywhere **except** the narrowest width (both reverse) **and** under sigmoid
  (Domain-IL alone reverses, d=1.25 — the largest standardized effect in the project). Depth
  numbers now **final** post-342-fix: Class-IL +1.4/+2.9/+3.3/+3.7pp (depths 1–4, all p≤0.021,
  growing not plateauing); Domain-IL −0.7/−0.3/−0.9/−3.0 (bookends significant, middle flat, D4
  now trustworthy at 10% cap-hit vs. the old pass's 80%). Width: −4.0±0.7 at H=4 Class-IL
  (p=0.002 — confirmed significant after the `_vals` bug fix; an earlier mid-session report that
  this had weakened was wrong and has been corrected). **Final**, and this is the figure that
  earns its slot as R2's single most important one.
- **913 — replay sets the scale.** Same axis, replay's effect (+9.84/+3.43) drawn at the same
  scale as 912 — roughly 7× PC's effect where PC wins, positive where PC is negative. The
  sentence this buys: the problem is solvable, changing the credit-assignment rule is not what
  solves it. **Final.**
- **914 (appendix)** — the three original sweep panels, pre-consolidation. Demoted on purpose;
  912 replaces rather than joins them.

**R2's argument**: PC's effect is real, small, and its sign depends on scenario, width, and
activation — three different axes, three different reversals, which argues the effect is
systematic rather than fragile (912's own framing), but also means no single number summarizes
it. That tension is R2's honest headline, and R3 exists because of it.

### R3 — Testing the mechanism prospective configuration is credited with

- **915 — target alignment.** Three panels now (was two): trained-batch and task-1-reference
  alignment through training (legacy 5-seed finding — flat, doesn't track forgetting — does NOT
  replicate at 10 seeds: r = +0.96/+0.74 Class-IL/Domain-IL for both rules); alignment against
  retention, per seed; and the new sigmoid panel — under the ONE condition where PC's Domain-IL
  advantage is real and large, interference-alignment is +0.0050 (bp) vs +0.0032 (pc), essentially
  identical. **The credited mechanism fails to explain the advantage even where the advantage is
  real and large — a stronger negative than the tanh result alone.** **Final.**
- **916 — displacement → ΔW → retention.** Panel (a) now three subplots: PC-only (backprop's
  displacement is zero by construction), does more settling displacement predict less output-
  weight movement and more retention — yes in Class-IL (partial r +0.89), weakly in Domain-IL
  (+0.12) under tanh, **rising to +0.50 under sigmoid** — moving toward the Class-IL pattern
  without fully reaching it. Panel (b): 344 regenerated — damping is in W2 (output), not W1
  (hidden), the opposite of the pre-registered prediction; PC's output update uses the SETTLED
  hidden activity, not feedforward. **Final.**

**R3's argument — REWRITTEN 2026-09-13, and the honest version is weaker than what was here.**

Three things have to be said in order, and only the first two are established.

**(i) Target alignment — ⚠ CORRECTED 2026-09-13, having first been written up backwards.**

An earlier version of this section said the credited mechanism "fails, including where PC's
advantage is largest". **That was wrong**, and it came from reading only the interference
variant for the sigmoid condition and generalising from it. 930 draws all three conditions and
both measures side by side; the honest reading is below.

**Target alignment on the trained batch — the quantity S&B actually credit — tracks the sign of
the benefit in all three conditions** (930, paired, 10 seeds):

| | paired Δ crossover | paired Δ target alignment | paired Δ interference alignment |
|---|---|---|---|
| Class-IL · tanh | **+1.27** (1.6 sem) | **+0.0236** (4.6 sem) | **−0.017** (3.1 sem) |
| Domain-IL · tanh | **−0.53** (3.2 sem) | +0.0029 (1.3 sem, null) | −0.0032 (1.0 sem, null) |
| Domain-IL · sigmoid | **+0.74** (4.0 sem) | **+0.0048** (7.9 sem) | −0.0018 (2.9 sem) |

Three out of three on sign, with the null landing exactly on the column where PC hurts. **On
that evidence the mechanism survives**, and R3 must not be written as a refutation.

Two things still do not add up, and both belong in the text:
- **Magnitude does not scale.** Class-IL carries five times the alignment difference of the
  sigmoid column (+0.0236 vs +0.0048) for a comparable benefit (+1.27 vs +0.74). A cause would
  be expected to move with its effect.
- **The interference measure inverts where PC wins most.** In Class-IL PC's updates move task 1
  *further* from its targets than backprop's (−0.017, 3.1 sem) while PC forgets *less*. Better
  alignment on the batch being trained, worse alignment on the task being kept.

So: consistent with S&B on direction, unexplained on magnitude, and complicated by the
interference inversion. That is the claim R3 can defend.

**(ii) There is a real, replicated rule difference, and it is at the output layer.** PC's W2
update multiplies the error against the **settled** hidden activity rather than the feedforward
one, and the consequence is measurable: log-log slope of realised step against learning rate is
W1 bp 0.93 / pc 0.90 (identical) but **W2 bp 0.99 / pc 0.78** (916b, regenerating 344). It shows
up independently as cos(ΔW) diverging toward the output (54/55: W1 0.985, W2 0.814), as PC
concentrating 87–89% of its W2 trajectory variance on PC1 against backprop's 62–68% (929), and
as PC's block-to-block updates not reversing at the switch (cos ≈ 0 to +0.24) where backprop's
reverse hard (−0.84 to −0.96). **This much is established.**

**(iii) ⚠ WHAT IS NOT ESTABLISHED — the bridge from (ii) to the retention benefit.**

The chain this section has been asserting is *settling displaces more → the output weights move
less → task 1 survives*. **Tested directly on 2026-09-13, the middle link fails.**

| | r(D, totW1) | r(D, totW2) | r(D, ret) | partial r(D,ret \| totW2) |
|---|---|---|---|---|
| Class-IL | −0.80 | −0.80 | +0.80 | **+0.87** |
| Domain-IL | −0.68 | −0.62 | +0.29 | +0.13 |
| Domain-IL sigmoid | +0.09 | +0.02 | +0.46 | **+0.52** |

Three problems, any one of which is enough to stop the causal claim:
- **Controlling for output-weight movement does not weaken D→retention; it slightly strengthens
  it** (+0.80 → +0.87). A mediator that is removed without cost is not the route.
- **Settling reduces movement in BOTH layers equally** (−0.80 / −0.80 in Class-IL), so nothing
  here is specific to the readout.
- **Under sigmoid, settling barely relates to weight movement at all** (+0.02) yet still predicts
  retention (+0.50). The link that is supposed to carry the effect is absent exactly where the
  effect is largest.
- And independently, **130 finds freezing W2 outright recovers nothing** on crossover in either
  scenario (+0.36 / +0.43 Class-IL, +0.15 / −0.32 Domain-IL). If stopping W2 entirely buys zero,
  damping it cannot buy +1.43.

**So the current state is: a robust correlation with no demonstrated mechanism.** D→retention
survives controlling for task-2 length in Class-IL (+0.89) and under sigmoid (+0.50), and is
absent in tanh/Domain-IL (+0.12) — which does track where PC helps, across all three conditions.
That correspondence is worth reporting. But *why* is open, and the obvious candidate is ruled out.

**THE TEST WAS RUN, 2026-09-13, AND THE CORRELATION DOES NOT SURVIVE IT.**

Backprop and PC see the *same* split at a given seed, so backprop's retention on that seed is a
clean measure of how hard the split is, with no settling in it at all. Partial out that measure:

| | r(bp ret, pc ret) | r(D, bp ret) | r(D, pc ret) | **partial r(D, pc ret \| bp ret)** |
|---|---|---|---|---|
| Class-IL | **+0.97** | +0.78 | +0.80 | **+0.30** |
| Domain-IL | **+0.97** | +0.21 | +0.29 | +0.39 |
| Domain-IL sigmoid | **+0.99** | +0.48 | +0.46 | **−0.25** |

Two readings, both important:

1. **Settling displacement is largely a proxy for split difficulty.** In Class-IL, D correlates
   with *backprop's* retention at +0.78 — and backprop has no settling whatsoever. Controlling
   for split difficulty drops D→retention from +0.80 to **+0.30**; under sigmoid it **reverses**
   to −0.25. The one number R3 was leaning on is mostly seed variance.
2. **Retention is set by the seed, not the rule.** r(backprop retention, PC retention) is
   **+0.97 / +0.97 / +0.99**. The split decides almost the whole outcome; the rule moves it by a
   point or two on top of that.

**CONCLUSION, as it stands on 2026-09-13.** Two different mechanism claims are in play and they
have come out differently, so keep them apart:

- **(i) S&B's target alignment is CONSISTENT WITH the data, on sign.** It is positive and
  significant in both conditions where PC helps and null where PC hurts. It is *not* refuted.
  What it lacks is proportionality (five times the effect for the same benefit) and a clean
  reading of the interference measure, which inverts where PC wins most. Report it as supported
  in direction, unexplained in magnitude.
- **(ii) A real structural rule difference exists at the output layer** — settled rather than
  feedforward activity, W2 lr-slope 0.78 vs 0.99, no update reversal at the switch. Solid as a
  *description of the rule*, with no demonstrated link to retention.
- **(iii) THIS PROJECT'S OWN proposed mechanism — settling displacement → less weight movement →
  retention — does NOT survive.** The mediation fails (partialling out total W2 movement leaves
  D→retention *stronger*, +0.80 → +0.87), 130 finds freezing W2 outright recovers nothing, and
  partialling out split difficulty collapses it (+0.80 → +0.30, and −0.25 under sigmoid).

So R3's honest shape is: **the literature's mechanism is not refuted here and may well be right
in direction; the alternative this project floated is refuted by its own controls; and the
dominant term in the whole comparison is neither — it is the seed.** r(backprop retention, PC
retention) is +0.97/+0.97/+0.99. That is a defensible and interesting section. It is not the
"credited mechanism fails" story an earlier draft of this file asserted.

⚠ **916's figure and docstring still assert the causal chain** ("more displacement, less total
output-weight movement, more retention"). Link 1 is real; the retention half needs re-stating or
the figure needs a panel showing the partial. Flagged, not yet changed.

**The two figures that now carry this section, both built 2026-09-13/14 so R3 can argue in
pictures rather than in prose.**

- **930 — benefit against credited mechanism.** Three columns ordered by what PC does (Class-IL
  helps, Domain-IL/tanh hurts, Domain-IL/sigmoid helps most), three rows: paired Δcrossover,
  paired Δtarget alignment, paired Δinterference alignment. The reading is whether row 2 tracks
  row 1, and it does on sign, 3/3, with the null landing on the negative column. The figure is
  the reason this section is **not** written as a refutation. **Final.**
- **933 — what determines retention, if not the rule.** The prior question 930 cannot ask.
  Three panels, and every number re-derived from 803/808 when the script was written:
  (a) backprop retention against PC retention per seed, identity line drawn — **+0.974 / +0.974
  / +0.994**, the points sit on the diagonal; (b) D→PC-retention raw beside the same correlation
  with backprop's retention partialled out — **+0.80→+0.30**, **+0.29→+0.39**, **+0.46→−0.25**,
  so the one correlation this project leaned on is largely split difficulty; (c) the seed effect
  and the rule effect in the same unit, points of task-1 accuracy — sd of backprop retention
  across seeds against the mean paired PC−backprop difference, **7× / 19× / 17×**. **Final.**
  ⚠ **The ~30× in the earlier proposal for this panel was wrong.** It compared a retention
  *range* against a *crossover* effect — two metrics. Measured honestly, within retention, the
  ratio is 7–19× depending on condition. Cite 7–19×, never 30×.

**Together 930 and 933 are R3's argument**: the mechanism the literature credits moves the right
way but does not scale; the mechanism this project proposed is removed by its own controls; and
both are small next to the seed. That is three claims, and the two figures separate them.

### R4 — Why these are two investigations, not one

This section's job (corrected once already, per its own heading note) is to justify the scenario
split with evidence, not narrate three experiments. It currently has the most new material of any
section from this session.

- **917 — repeated alternation geometry.** The pre-registered prediction (Class-IL closed loop,
  Domain-IL spiral) is **wrong as stated** — Class-IL's block lengths collapse too, just faster.
  The honest finding: scenarios differ in HOW FAST they find a joint solution, not whether they
  can. **Final**, title already corrected once for this reason.
- **918 — weight-space PCA.** Now both rules (was backprop-only — fixed this session). Both
  rules show real contraction (ratio 0.05–0.29, none near 1.0 — not orbiting). PC contracts
  MORE than backprop in most cells except one: Domain-IL's readout layer (W2), where PC's ratio
  (0.29) is worse than backprop's (0.27) — the one cell that's already PC's known weak point,
  showing up independently here. **Final.**
- **918's follow-up investigation, and what it settled.** The Domain-IL/pc/W2 panel looked wrong:
  every other panel bounces corner-to-corner across the ten alternation blocks, that one sweeps
  out once and back without oscillating, on an experiment whose whole design *forces* oscillation.
  The suspicion was a bug or a sort-order artefact, and it was chased down rather than written
  around. **It is real.** Five independent lines, all agreeing:
  1. **Data ordering is sound** — `steps` and `trace_W2` are the same length, strictly
     increasing and correctly aligned; the sort-order hypothesis is dead.
  2. **The code path is symmetric** — full review of 804 → `protocol.py` → `methods.py` →
     `predictive_coding.py` → `model.py`. Every scenario-dependent function is shared identically
     by both rules; the one rule-specific mechanism (PC's W2 update multiplies error against the
     **settled** hidden activity, not the feedforward one) is already validated against S&B's
     reference implementation at cos(ΔW) = 1.000000 (script 63) and is the same mechanism 344
     and 916(b) independently found.
  3. **Direct quantitative test** — cosine similarity of block-to-block net displacement. Backprop
     reverses hard at every task switch (−0.84 to −0.96, all seeds); **PC in Domain-IL/W2 does
     not reverse at all** (0.00 to +0.24). The absent oscillation is the measurement, not a
     drawing artefact.
  4. **It replicates across every seed** — 804 was rerun with `TRACE_SEEDS` raised 3 → 10 for
     exactly this question, and `--per-seed` draws all ten. The shape holds.
  5. **It is not one exploding weight** — see the participation ratios below.
- **What the diagnostics found that is worth reporting in its own right.** Two numbers came out
  of this that are not about the anomaly at all:
  - **PC concentrates its output-layer trajectory far more than backprop does, in both
    scenarios.** Share of W2 trajectory variance on PC1 alone, across 10 seeds: backprop
    **67.5 ± 5.9%** (Class-IL) and **69.1 ± 5.6%** (Domain-IL); PC **87.0 ± 4.9%** and
    **82.7 ± 6.9%**. Non-overlapping, seed-general, and it is a *rule* difference rather than a
    scenario one — PC's readout moves along essentially one direction while backprop's needs two.
    This is a new, clean statement of the same "PC reconfigures the output layer differently"
    thread that 55, 65, 344 and 916(b) each reached by a different route.
  - **That direction is distributed, not a single weight.** Participation ratio of PC1's loading
    vector (1.0 = spread evenly over all N weights, ~1/N = one weight does everything): W1 sits
    at **0.108–0.140 across all four conditions** — tight, no rule or scenario difference; W2
    spreads to **0.249–0.320**, highest for Domain-IL/pc. Roughly 15–25 of W2's 160–320 entries
    carry half of PC1's energy. So the concentration in the bullet above is concentration *of
    direction*, not of synapses.
  **Decision needed**: none of this is in the report figure, which still shows one seed. The
  per-seed and consolidated grids are diagnostics, correctly kept out of the 18. But the
  PC1-dominance number (87/83 vs 68/69) is a genuine finding with 10-seed backing and no home —
  it belongs either as a sentence in R4's text citing 918, or as a panel added to 916, which
  already owns the per-layer PC-vs-backprop mechanism story. My recommendation: a sentence in R4,
  because 916 is already at three subplots and this is corroboration rather than new mechanism.
- **919 — joint pre-training then sequential.** Protects partially, not fully (Class-IL
  joint-start retains 6.9–25.6% against scratch's 1.5–9.1%, still collapsing from ~77%). ⚠ A real
  confound the figure itself must carry: the joint arm's task-1 phase is much shorter (already
  above threshold), so "joint pre-training helps" is entangled with "trained task 1 less, had
  less to lose." Lean on the Domain-IL backprop arm specifically (partial r with phase-length
  −0.09, clean) rather than Class-IL (−0.64, confounded). **Final, with the confound stated.**
- **920 — tie-out.** Re-analysis: which measurements actually separate by scenario. This is the
  figure that tempers R1's clean framing — not everything splits as neatly as the headline
  suggests (echoes 911's own honest qualifier about the probe-argmax gap). **Final.**
- **927 — task-1 overtraining** (new, not in the original 18). Does retention improve if task 1
  trains past matched competence? **Backprop benefits in BOTH scenarios** (Class-IL d=0.73,
  Domain-IL d=1.38, retention climbing the whole way 1×→4×: 37.6→38.5→40.3→42.2→43.3).
  **PC benefits in Class-IL** (d=0.65, 6.0→7.4→8.4→9.7→10.4) **but in Domain-IL it rises
  slightly then genuinely FLATLINES at exactly 37.0 from 3× onward** (37.0→37.9→38.2→37.0→37.0,
  d=0.00 for 4×-vs-1×) — resolved with the 3.0× point added; this is a real saturation, not the
  noisy dip the first pass suggested. Because backprop keeps climbing while PC flatlines, the
  paired PC−backprop gap actually WIDENS with more consolidation time in Domain-IL (−0.6pp at
  1× → −6.3pp at 4×) — more training time doesn't just fail to help PC there, it makes the
  relative deficit look larger. **Final.** The sharpest evidence yet that PC's Domain-IL deficit
  is structural, not a consolidation-time artifact.
- **928 — forgetting-shape classification** (new). Applied `classify_forgetting` to 803's full
  2×2: Class-IL splits into collapse/delayed/partial for both rules, **9/10 seeds get the
  identical shape under both rules** — a data property, not a rule property. **Domain-IL is
  uniformly "partial", all 10 seeds, both rules, no exceptions.** **Final.**
- **Decision needed**: 927 and 928 arrived after R4's shape was set. They're not a detour — read
  together, they say "backprop's retention is consolidation-sensitive everywhere, PC's only
  where PC helps at all, and the SHAPE of forgetting is scenario-determined, not rule-determined"
  — which is a real paragraph, not just two more figures. Recommend a short new R4 subsection
  built around this specific synthesis, rather than bolting them onto the existing three-figure
  flow as unrelated extras.

**R4's argument, once 927 finalizes**: the two scenarios don't just differ in how much forgetting
happens (R1) or in whether PC helps (R2) — they differ in the underlying DYNAMICS (how fast a
joint solution is found, whether more time helps, what shape the damage takes). That's a
qualitatively different kind of evidence for the split than R1's structural argument, and it's
new this session — worth being visible as new, not folded in as if it were always the plan.

### R5 — Evidence for an output-competition component

This section needs the most editorial work before writing captions. Its title is already
deliberately weak ("a component", not "the mechanism") — that weakness has grown since.

- **921 — retention distribution.** Class-IL retention at 50 seeds is bimodal (a near-zero group
  and a 15–30% group), not the smooth spread the mean suggests; Domain-IL is unimodal. Backprop-
  only by data availability (the 50-seed block doesn't exist for PC) — checked this session, not
  an oversight. Panel (b): 130's freeze factorial — freezing the output layer recovers nothing,
  freezing the hidden layer is strongly negative, freeze-both undefined on 0/10 seeds (a result,
  not a gap). **Final.**
- **922 — partial-column freeze.** The decisive experiment: freezing exactly the weights masking
  spares recovers +1.9 points; masking itself recovers ~+46. **Masking does not work by sparing
  the task-1 output weights.** Confirmed at 10 seeds, both rules (806, this session). **Final,
  but see below — 922 needs a companion finding, not a replacement.**
- **923 — trained probe vs. argmax.** The random-init floor is drawn deliberately (81.8%
  untrained vs. 86.2% after task 1 — ~4–5 points attributable to training, not ~60). Task-2 panel
  is the control (the gap reverses there, ruling out "the probe is just a better classifier").
  Backprop-only, checked and justified this session (PC's probe reading is nearly identical:
  81.9/80.6 vs. 82.6/81.8). **Final.**
- **924 — per-layer weight path and hidden-code drift.** Built and current (924 plus 924b, the
  hidden-code-drift half, which reads the `807_code_drift` arrays — note that is a *different*
  807 from the overtraining run behind 927; see the collision flag in
  `script_plan_800_900.md`). Not rebuilt this session because nothing it reads changed.
  **Decision needed**: check it against 916(b)'s W1/W2 damping finding before the caption is
  written — both now speak to per-layer weight movement and the overlap may be worth
  cross-referencing rather than duplicating.
- **925 — task structure predicts what survives.** Copy of 311 + new: pair-similarity correlation
  with retention, replicated in one seed block, not the next (r=+0.69 then +0.18–0.51 depending
  on pooling) — a real, if unstable, data-dependency finding.
- **926 (appendix, missing)** — Class-IL digit table, copy of 312. Not yet built; P3, low
  priority.
- **The two findings this section must absorb, both from this session:**
  1. **806's masking-impairs-task-2 finding.** Even at 4× budget, masked task-2 accuracy craters
     to a mean of 46.5% (backprop, 9/10 seeds capped) / 60.4% (pc, 5/10 capped). "Masking
     recovers 45 points" is not a free lunch — it trades away task-2 competence, more so for
     backprop than PC. **This doesn't show up in 922's crossover-based reading** (crossover is
     mostly censored under masking for an unrelated reason) — it needs its own sentence, probably
     right after 922's own finding, or R5 will report masking as an unambiguous win when the full
     picture is a real trade-off.
  2. **809 — LANDED, and it answers the question R5 was built around. Final.** The probe from 923
     fitted on 806's four conditions. Task-1 argmax accuracy swings **42 points** across those
     conditions (backprop: control 4.8 → mask 46.8); the probe reading over the same conditions
     moves **0.4 points** (82.3 → 82.7, against an untrained-probe floor of 80.2). Every cell,
     both rules, sits in a 2-point band:

     | | argmax t1 (bp / pc) | probe (bp / pc) | above floor |
     |---|---|---|---|
     | control | 4.8 / 6.0 | 82.3 / 81.5 | +2.1 / +1.3 |
     | freeze_w2 | 3.2 / 4.3 | 81.4 / 80.7 | +1.3 / +0.5 |
     | freeze_w2_t1 | 6.7 / 6.3 | 81.9 / 81.2 | +1.7 / +1.0 |
     | mask | **46.8 / 27.6** | 82.7 / 81.5 | +2.5 / +1.3 |

     **Masking does not change what the trunk represents. It changes what argmax can read out
     of it.** A 42-point readout swing sitting on a 0.4-point representational change is about as
     clean a dissociation as this project has produced, and it is the direct evidence R5's title
     ("evidence for an output-competition component") was asserting without.
     ⚠ **The honest qualifier, which must travel with it:** the probe has only ~2–4 points of
     usable dynamic range above its floor (923 makes the same point — 81.8% untrained vs 86.2%
     after task 1). "Flat" is therefore *consistent with* no representational change rather than
     proof of it; a measurement with more headroom could still find something. Say it as a
     dissociation between two readouts, not as a proof that the representation is untouched.

**R5's argument, now that 809 has landed**: masking works and freezing doesn't; the difference
isn't which weights are held still (922); the recovery masking buys is a **readout** effect, not
a representational one (809 — 42 points of argmax against 0.4 points of probe); and it comes at
a real cost to task-2 learning that the retention number alone hides (806). That is a coherent
account, and it is stronger than the one this section had a day ago, when "masking changes what
the trunk learns, for reasons not yet isolated" was the best available.

**Decision needed — and 809 changes which way I'd call it.** The open question was whether to
soften the title further, admitting the component hadn't been located. I'd now argue the
opposite: 809 *locates* it. The section can say plainly that the Class-IL recovery masking
produces is output competition being removed, not representation being restored, with 922 ruling
out the obvious alternative (it isn't the task-1 output weights) and the probe's flatness as the
positive evidence. The title can firm up rather than soften — provided the dynamic-range caveat
above rides along with it, and provided 806's task-2 cost is stated in the same breath so
"masking works" is never left standing unqualified.

---

## Discussion

- **931 — intervention ranking**, regrouped this session (was a category error — listed "PC" as
  one row among interventions, when PC is a base rule, not something applied on top of one). Now:
  two vertical panels (Class-IL/Domain-IL), every intervention measured against its OWN rule's
  control, backprop and PC shown as adjacent bars. Surfaced something the old version buried:
  **k-WTA is far more harmful for PC than backprop** (Class-IL −32.4 vs −9.8pp; Domain-IL −30.2
  vs −2.6pp). EWC's lower-bound caveat (λ grid never bracketed its optimum) carried over, hatched.
  Mask row uses 806's corrected data but reads near-zero on crossover for the reason above — this
  figure alone will NOT convey the masking-impairs-task-2 finding; that has to live in prose or
  R5. **Final**, pending only stylistic polish.
- **932 (appendix)** — SI and k-WTA sweeps, copy of 210+220. Not touched this session.

**D1's argument**: laid beside every alternative, measured to the same standard where possible
(and flagged where not, EWC), the credit-assignment rule is a small lever. Replay and masking are
larger; k-WTA is actively harmful and more so for PC. The report's title-level claim — this isn't
solved by a smarter rule — rests most on this section and on 913.

---

## Open decisions, collected in one place

Every one of these is an interpretive or editorial call. **None is waiting on a run.**

1. **Figure budget** — 909, 927 and 928 are built, putting the count at 21 against a frozen 18.
   909 reads as Methods and likely merges into 907/908; 927/928 are real R4 evidence and have a
   good claim on slots, which means two current entries move to appendix. Candidates, weakest
   case first: 920 (tie-out re-analysis), 925 (a non-replication), 914 (already appendix).
2. **R5's headline sentence** — 809 has landed and, as argued in R5 above, I'd now firm the
   title up rather than soften it. Still a call to make, but it is a call with evidence behind
   it now rather than a hedge.
3. **903's scope** — does Methods still carry the raw activation sweep now that 912/915/916 own
   that story, or does it just point forward? Recommendation unchanged: point forward.
4. **924 vs 916(b)** — both speak to per-layer weight movement; check for overlap before either
   caption is written.
5. **918's PC1-dominance number** (87/83% vs 68/69%, 10 seeds) has no home in the figure set —
   R4 sentence or a 916 panel. Recommendation: R4 sentence. See R4 above.
6. **807 is claimed by two different training runs** (`807_code_drift`, `807_overtrain_task1`).
   Nothing on disk collides and no figure is wrong, but the number is ambiguous in every
   cross-reference. Renaming touches a script, two `.npz` files and 924b's reader, so it needs
   approval — flagged in `script_plan_800_900.md`, deliberately not fixed.

## Everything else is written, not computed

- ~~`807`'s 3.0× rerun~~ — **done**, 927 final.
- ~~`809` (probe on 806's conditions)~~ — **done**, R5 rewritten around it above.
- ~~`804` traced only 3 of 10 seeds~~ — **done**, rerun at `TRACE_SEEDS = 10`; the 918
  seed-robustness check rests on all ten.
- `926` is the **one unbuilt figure** in the whole list (Class-IL digit table, copy of 312).
  Appendix, P3 — the only remaining thing that would need running, and it is optional.
- `report/sections/*.tex` still carries stale `\gap{...not written}` captions for figures that
  now exist. That is the writing pass, and it is the user's own.
