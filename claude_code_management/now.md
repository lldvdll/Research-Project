# Now

Dashboard. Detail lives in `current_state.md`, `knowledge_base.md`, and each script's docstring.
**Keep this short.** If a section grows past a screen, it belongs somewhere else.

**Running:** 66 (Class-IL vs Domain-IL weight routes), 67 (Class-IL decomposition, four rules).
EqProp-bound. **Next:** concept drift (68) — needs a per-task `label_map` in `run_classil`.

---

## What the thesis argues
**The character of forgetting and how to measure it. The four learning rules are the instrument,
not the subject.** They earn their place by producing genuinely different credit assignment
(cos 0.197 EqProp, 0.985 PC on W1) while producing the *same* forgetting — which is what makes
"different rule" and "different forgetting" separable at all.

**Say "under a controlled protocol PC shows no retention advantage", never "S&B do not
replicate".** We never ran their setup in this series; the reproduction is old 30–34, failed,
closed. A failed replication invites "you did it wrong"; a controlled comparison invites "then
where does their advantage come from?"

---

## FINDINGS

### The rules
- **PC does not reduce forgetting vs backprop in Domain-IL.** Crossover −0.01 (0.0σ, 52),
  +0.49 (2.0σ, 53). Depth {1,2,3,5} × width {32,128}: −1.75 to +0.52, **never separated in 8/8
  cells** (59), where replay separated in 8/8.
- **PC helps slightly in Class-IL only.** Crossover +1.29 (3.7σ, 5W–0L, 56), +3.00 (2.2σ,
  4W–1L, 57). The matched-competence version is the weaker of the two.
- **EqProp is worse, on every seed.** 0W–5L on crossover in 53, 56 and 57. A positive finding,
  not a null.
- **Replay separates everywhere.** +26.6 retention (6.6σ, 52); 24W–0L crossover, p<0.001 (60).
  So the problem is solvable and the EBMs fail at something achievable.

### The data structure dominates everything
- **Which two digits share an output unit explains ~62% of Domain-IL seed variance.**
  r = +0.787 backprop, +0.779 pc, +0.630 replay; all p ≤ 0.001, n = 24 (60).
- **Control is null.** Same digits, same split, pairing ignored: r −0.25 to +0.06, all p > 0.23;
  **+0.7 points/sd against the pairing's +9.7**, so null in effect size, not range restriction.
- **It is interference, not learnability.** r(pairing, task-2 final) = +0.164 (p=0.45); all 24
  seeds reached 80% on task 2; speed correlation is −0.354 (p=0.088), the *opposite* sign to
  "dissimilar pairs are harder".
- Backprop retention over 24 seeds: **23.0–71.0%** — larger than every rule effect measured.
- **PC tracks backprop's dependence on the data**, not just its mean: slope +9.5 vs +9.7 pts/sd.
  A stronger null than "not separated". Replay is half as sensitive (+4.0), as a buffer should be.

### Mechanism
- **Class-IL forgetting is output-layer suppression.** NCM holds ~80% on task 1 while argmax
  reads 0.2% (42/43). Structurally impossible in Domain-IL, where every unit is a target.
- **Two mechanisms, different time constants.** Freezing W1 is worth **+31 points at 50 updates
  post-switch and 0 at 2000**. Which one looks dominant is decided by when you stop the clock.
- **PC diverges from backprop at the OUTPUT layer, not the input.** cos(ΔW) at depth 3: W1 0.952,
  W2 0.939, W3 0.837, W4 0.623 (55). W1 is the layer whose drift damages task 1 — so PC
  reconfigures the end and leaves the part that matters alone. Predicts the Domain-IL null.
- **EqProp's update is nearly orthogonal to backprop's** (cos 0.197–0.316 on W1, 54) and still
  gained no retention. Different credit assignment is not by itself sufficient.
- **PC is more efficient than backprop at the OUTPUT layer and slightly worse at the input**
  (65, median L1 path ÷ net displacement per synapse, task 2): W2 **1.703 vs 3.279**
  (−1.58, 6.6σ) — 31% shorter path while displacing 22% further — and W1 2.781 vs 2.508
  (+0.27, 9.8σ). **It improves the layer that does not matter for forgetting and is fractionally
  clumsier in the one that does**, which is the same story 55 tells by a different route.
- **EqProp travels 48% further to arrive 35% short** (65): W1 path 182.1 vs backprop's 123.2,
  net displacement 28.3 vs 43.8; inefficiency 4.56 vs 2.51 (13.0σ) on W1 and 8.85 vs 3.28
  (12.1σ) on W2. First quantitative evidence that its credit assignment is **worse**, not just
  different.
- ~~Inefficiency predicts forgetting within a rule~~ — **confounded**. r looks like +0.81 to
  +0.89, but `r(pairing, retention)` is 0.90–0.97 and `r(pairing, inefficiency)` 0.50–0.75
  across all four rules, so the pairing is a common cause. Needs 60's 24-seed treatment with
  the pairing partialled out before anything can be said.
- **[R1]'S OWN MECHANISM METRIC DOES NOT TRACK FORGETTING (64).** Target alignment during task 2:
  backprop +0.277, **pc +0.280** (+0.003, i.e. 1% relative — significant at 2.6σ only because a
  per-update measure has a tiny SEM; the effect size is negligible), replay +0.181, eqprop +0.250.
  **PC is not more target-aligned than backprop here, so [R1]'s mechanism does not reproduce at
  this scale.**
- **And the metric inverts at the extreme.** Interference on task-1 data *during* task 2:
  backprop −0.030, replay −0.000, pc −0.025, **eqprop +0.039** — EqProp's updates move task 1
  *toward* its targets on average, and EqProp forgets the **most** (0W–5L). The retention
  ordering replay > backprop > eqprop is not the interference ordering. `[EMPIRICAL]`
- **Why, probably: alignment is direction-only and blind to magnitude.** A rule can point
  helpfully and still do more damage by moving further per update. 64's aggregate records only
  cosines, not `|d_learn|`. **Gap — record the magnitude.**
- **Both EBMs genuinely settle.** PC needs ≤18 steps, runs 50, sits 3e-04 from equilibrium
  (50, 63). EqProp `settle_tol=1e-4`.
- **Forgetting is visible as RETRACED WEIGHT DISTANCE (65's mechanism figure).** Over a whole
  run the two task blocks displace W1 by 47 and 43 while init→end is only 52 — **38 units are
  travelled and given back**. So the full-run ratio (backprop W1 2.86x) is far worse than either
  block alone (1.77x, 1.53x). Replay retraces least (27 of 82) — the positive control behaving
  correctly. **PC's W2 route runs almost straight from init through the switch to the end**
  (full-run 1.40x against backprop's 3.05x, 2 of 13 retraced against 4 of 9): at the output
  layer PC's second task continues in the same direction where backprop's reverses.

### Metrics — the methodological core
- **Endpoint metrics inherit the budget** (setup-induced forgetting, Michel et al. 2023). In 2×5
  Class-IL every condition reaches ~0, so ACC/BWT/forgetting rank nothing.
- **[R1]'s mean test error conflates learning speed with forgetting** and cannot separate them.
  Measured and reported, never alone.
- **Crossover height is primary** — budget- and threshold-independent, invariant to uniform time
  rescaling.
- **A null crossover is a RESULT, not missing data:** task 1 stayed above task 2 throughout, so
  forgetting ran slower than learning. `paired_sign` ranks censored runs top and uses every seed.
- **Censoring is asymmetric and scenario-structural.** Replay censored 11/24 (60), 3/5 (52, 53)
  in Domain-IL; **0/5 in Class-IL**. Half-life is *unusable* in Domain-IL — backprop finite on
  2/5 (52), 1/5 (53). Same cause: the reference event is borrowed from the Class-IL picture of
  forgetting and does not occur in Domain-IL.
- **Sign-test floor at 5 seeds is p=0.0625** → censorable metrics need ≥6 seeds.
- **Compare paired, per seed.** Group means put the positive control at 0.7σ; paired, 5.1σ (53).

### Why no prior result is evidence
- **Experiment 12's positive control INVERTS.** Replay −14.14 crossover (5.2σ) *below* backprop
  at 100 updates/task (61). A setup whose positive control inverts cannot detect forgetting.
- **The legacy spec gave backprop and replay an oracle task mask.** ‖e‖ on absent classes: 0.0
  (ce) vs 5.97 (mse, PC) vs 10.0 (hinge, EqProp). Audit A1 — so 12 was confounded twice.
- **PC's inference step rule is equivalent to [R1]'s.** cos(ΔW) = 1.000000 at every step count;
  largest metric-grid disagreement 1.4e-03 points (63). That caveat is closed.

---

## Checklist
- [x] **40** does backprop forget → 93.8% → **0.4%**. Relearning accelerates 320 → 110 → 60.
- [x] **41** capacity → **H = 32**. Joint ceilings 93.6% Class-IL, 94.3% Domain-IL.
- [x] **42** masked × frozen at saturation → everything reads 0. **Kept as the counterexample.**
- [x] **43** the same at matched competence → the correction. This pair is the argument.
- [x] **50** settling → PC fully settled; EqProp's stopping rule was truncating, fixed.
- [x] **51** learning-rate calibration → matched to 1.25×.
- [x] **52** four rules, fixed budget → nothing separates except replay.
- [x] **53** four rules, matched competence → same.
- [x] **54** are the EBMs settling → yes, both, and neither is backprop.
- [x] **55** first depth pass (superseded by 59) → PC's divergence grows toward the output.
- [x] **56/57** Class-IL, both readings → PC's only positive result.
- [x] **58** legacy spec → cannot answer exp 12; backprop never learns task 2 there.
- [x] **59** depth × width → PC never beats backprop in 8/8 cells.
- [x] **60** why the seed matters → the output-unit pairing, ~62% of variance.
- [x] **61** exp 12 verbatim → its positive control inverts. Mystery closed.
- [x] **62** metric grid over all saved runs (re-analysis, trains nothing).
- [x] **63** PC's inference step rule → equivalent to [R1]'s. Caveat closed.
- [x] **64** target alignment → PC not more aligned; the metric anti-tracks retention
- [x] **65** synaptic path efficiency → PC efficient at the OUTPUT layer, EqProp wanders
- [ ] **66** Class-IL vs Domain-IL weight routes — running
- [ ] **67** Class-IL decomposition, all four rules
- [ ] **B1/B2/B3** metrics — largely a write-up of evidence already in hand
- [ ] **C2** six-cell factorial, **C1** NCM figure
- [ ] **D** controlled comparison → **E** why → **F** does it generalise

## Next
1. **Class-IL is where the gaps are.** Depth is tested in Domain-IL (59) and **untested in
   Class-IL**, the only place PC shows anything. And 42/43's suppression-vs-drift decomposition
   ran **backprop only** — PC's Class-IL forgetting has never been decomposed the same way.
2. **Two untested schedule axes, both cheap** (`protocol.run` takes `tasks=`): an alternating
   schedule ([R1] Fig 4d) and **concept drift** ([R1] Fig 4f–g, where their largest advantage is
   claimed — not cycle length).
3. **[HYPOTHESIS]** the trade-off is set by the relative magnitudes of the two tasks' output
   contributions. `probes.output_unit_stats` already returns per-unit raw scores; 64's
   interference probe is adjacent. A readout question, not new machinery.

---

## Settings, and where they came from
| | value | from |
|---|---|---|
| scenario | Domain-IL primary, Class-IL where relevant | decision 2026-08-11 |
| hidden width | H = 32 | 41 |
| joint ceiling | 93.6% Class-IL, 94.3% Domain-IL — retention reads against these | 41 |
| learning rates | backprop 0.01, replay 0.01, pc 0.02, eqprop 0.01 | 51 |
| matched at | ~420 updates to 90% on task 1; residual spread 1.25× | 51 |
| PC settling | 50 fixed steps (needs ≤18) | 50, 63 |
| EqProp settling | `settle_tol = 1e-4` | 50 |
| chance | 20% Domain-IL, 10% Class-IL. Below chance = output capture | 40 |

## Decisions
- **2026-08-13 — every complicated measurement gets TWO figures:** one showing the quantity
  being measured, on real data, annotated; then the aggregated result. A number nobody can point
  at is not evidence.
- **2026-08-11 — Domain-IL is primary.** It is what [R1] use, and it removes output suppression,
  which is rule-independent.
- **2026-08-11 — learning rates matched on updates-to-90% on task 1**, never on the crossover:
  that is the dependent variable.
- **2026-08-11 — accuracy-stopped plots keep a step axis**, switch at x=0, curves not stretched.
  Stretching destroys the rate, and rate is half of what we measure.
- **2026-08-11 — EqProp stays in through the four-rule comparison, then must be re-earned.**
  ~350× backprop per update, intrinsic. That comparison is now done.
- **2026-08-11 — exp 12's learning rates are void**, not approximate.
- **Standing — no prior result is evidence.** Prior work informs direction and design only.

## Corrections — recorded so they are not re-derived
- ~~Freezing the hidden layer recovers nothing~~ — artefact of measuring at saturation. It
  recovers 17 points at matched competence.
- ~~Every rule receives the same output error~~ — false. `pc_update` computes it from the
  **settled** hidden state (`predictive_coding.py:114-118`).
- ~~Freezing W2 makes task 2 unlearnable~~ — false; networks train with fixed output layers.
- ~~A short-budget capacity sweep measures capacity~~ — it produces a fake plateau.
- ~~The settling requirement grows as weights grow~~ — it *falls* (confirmed again by 63).
- ~~At one hidden layer PC ≈ backprop because there is no layer to reconfigure~~ — wrong, and it
  was an inference from 54, not a measurement. cos(ΔW1) does not fall with depth.
- ~~PC's Domain-IL null might be an H=32 capacity ceiling~~ — at H=128, PC is +0.7 to +2.5 and
  still never separates (59).
- **A significance test with a zero-variance denominator fails in the wrong direction.**
  `paired_diff` returns `n_sem = inf` when every paired difference is identically zero, so
  `> 2 sem` reported two bit-identical runs as a significant effect (63). Judge effect size.

## Closed, do not reopen
- **Song & Bogacz reproduction** (old 30–34). Hours spent, failed, cause never found.
- **EBM as a replay generator** (old 04–06).
- **wandb, Optuna, class hierarchies, a shared `harness.py`.** All built, all deleted.
- **53/57 `--replot`** — needs their ragged per-run arrays saved, which changes their reading
  logic. Numbers are unaffected; 62 re-reports them without retraining. Not worth it.
