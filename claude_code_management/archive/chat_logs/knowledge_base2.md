# Knowledge Base — Continual Learning & Energy-Based Models

*Consolidated reference for the MSc dissertation. Loaded at the start of every chat.
Amend entries in place; do not append duplicates. Timeline of chats lives in `timeline.md`.*

**Last updated:** 2026-07-29 (chat 013–014)

---

## 1. Scope

### 1.1 Core question
Does an energy-based model with a biologically plausible local learning rule reduce
catastrophic forgetting compared to backprop — and if not, why not, and can it be made to?

### 1.2 The authoritative scope: the advisor's four points
1. **Pick one EBM.** Be clear which; acknowledge others exist.
2. **Compare catastrophic forgetting** in that model vs a backprop model.
3. **Understand why they differ.** Can trivial differences — coding sparsity, network size — explain it?
4. **Reduce forgetting in the EBM.** EBMs are predictive, so find the nodes whose prediction differs most and select only those for learning new stimuli. Does this work?

Everything not serving these four points waits. Explicitly deferred by the advisor:
generative/synthetic replay, VAE example-ordering, the 3×3 scenario×dataset grid,
other EBM families, efficiency comparisons.

### 1.3 Target regime
**Class-IL under a task-free stream.** This is the one cell where parameter regularisation
provably isn't enough, where the deficit is specifically inter-context discrimination,
and where information about old classes must come from somewhere.

---

## 2. Settled decisions

| Decision | Value | Reason |
|---|---|---|
| Dataset | MNIST at **14×14** (196 inputs), scaled [0,1] | EqProp settling is the bottleneck; 14×14 keeps ~97% BP ceiling at ~4× lower cost. 8×8 degrades classes. |
| Architecture | MLP **196 → 64 → 10**, one hidden layer | Identical across methods so comparisons isolate the learning rule. No CNNs. |
| Optimiser | plain **SGD** everywhere | Exact interference identity; no momentum confound; matches EqProp's own updates. |
| Scenario | **Class-IL**, single 10-way head, no task ID at test | Hardest scenario; where softmax models collapse. |
| Splits | 10×1, 5×2, 2×2 (`TASKS` list) | 10×1 cleanest for forgetting; 2×2 best for seeing one crossing in detail. |
| EBM #1 | **Equilibrium Propagation** (from supervisor's `gem_lazyep.py`, rewritten) | The "pick one EBM" answer. |
| EBM #2 | **Predictive Coding / prospective configuration** | The interference claim in the reading is *PC's*, not EqProp's. To test the literature's claim you need PC. |

### 2.1 Working practice (adopted after a period of overwhelm — these matter)
- One question per experiment, written as a single sentence before running.
- Controls on every forgetting experiment: backprop (negative control) and replay (positive control).
- A doubt gets one scheduled test, then it is closed. Mid-run doubts go to the open-questions list.
- One script → one figure, named identically (figure name derived from `__file__`).
- Constants in an obvious block at the top of each script. Reusable logic in `src/`, scripts stay thin.
- Keep code minimal. Metaphors are banned in explanations — map to variables and lines.
- Finish and interpret one experiment before starting the next.

### 2.2 Explicitly abandoned
- wandb / Optuna / persistent HPO infrastructure.
- `harness.py` shared run+plot module — run/plot lives inside each experiment script.
- Per-class output heads — with task ID it becomes Task-IL (which nothing forgets); without it, calibration fails equally for all methods. Not a route to an EBM advantage.
- Contrastive/conditional-energy EBM (Li et al. 2022) as a build target — cited as the acknowledged alternative for advisor point 1, not implemented.

---

## 3. The four methods

All three learning rules answer *"the output was wrong — which weights change?"*
The difficulty is that hidden units have no target.

**The sharpest distinction:** in backprop, hidden activities are **fixed by the weights**.
In PC and EqProp they are **variables optimised first**, and only then do weights change.
That is the energy-based family in one sentence.

| | backprop | predictive coding | EqProp |
|---|---|---|---|
| hidden activities | fixed by weights | variables — settle to a target | variables — settle twice |
| credit assignment | chain rule from above | inferred by relaxation | difference of two equilibria |
| weight update | global backward pass | local: pre-activity × post-error | local: free vs nudged difference |
| passes per update | 1 fwd + 1 bwd | 1 settling | 2 settlings |
| gradient | exact | exact at equilibrium | approximate (β-biased) |
| target at output | supervision signal | clamped | a perturbation, not a target |
| non-target classes | softmax suppression | one-hot → 0 | **hinge → −1 (strongest suppression)** |

**Unifying idea:** every energy-based method runs an EM-like two-step — E-step, activities
relax to low energy; M-step, weights move to make that state more probable. Backprop
collapses the two.

**Why the global energy doesn't break biological plausibility:** it is never computed,
stored or transmitted. It is a Lyapunov function used in the proof. Because the energy is
a sum of local terms, its gradient w.r.t. any local variable is a purely local expression.

### 3.1 Three regimes of the PC↔BP relationship (important, citable)
1. **Partial relaxation / infinitesimal nudging → approximates backprop.** Whittington & Bogacz (2017) is literally titled "an approximation of…". Updating activities for only the first few steps makes the PC weight update *equal* backprop's.
2. **Engineered exact equivalence** (Song et al. 2020) — but Dong & Wu state this "is not general": it needs specific initialisation, a precise layer-wise schedule and particular inference settings.
3. **Full relaxation to equilibrium → prospective configuration, genuinely ≠ backprop.** This is the actual contribution of Song et al. 2024.

So *"same result, different route"* holds only in the small-step limit. Turning settling up
is what makes it a different algorithm.

---

## 4. Results to date

### 4.1 Earlier notebook work (28×28, Adam) — still valid
- **Task-IL does not forget** for backprop (shared trunk + per-task heads: probe and head both ~99%). An earlier apparent forgetting was a **polarity artifact** of a balanced 2-way head — fold with `max(acc, 1−acc)`.
- **Class-IL forgets catastrophically:** ~21.6% final mean, tasks collapse to ~0% — active overwriting, not decay to chance.
- **Gradient interference:** cosine between current- and prior-task gradients predominantly **negative**. Occasional positive (cooperative) cosine coincides with slower new-task learning.
- **Replay works:** ~78% vs ~21%. Not equivalent to joint training (memory-limited).
- **EWC fails in Class-IL:** ~20% ≈ baseline, reproducing van de Ven Table 2. Reasons: (a) deadlock — protecting old weights blocks new classes, learning new ones suppresses old logits; (b) preserving each task's function cannot create discrimination between classes never seen together. **Unresolved:** EWC's Fisher-importance distribution resembles replay's yet accuracy matches baseline ⇒ the failure may be in the readout, not the features. Test = linear probe on the EWC trunk. *Never run.*
- **Multi-pass caution:** >1 pass is cyclic revisiting, not longer continual learning.

### 4.2 Script-based Class-IL runs (14×14, SGD)

| Split | backprop | eqprop | replay |
|---|---|---|---|
| 10×1 (100 iters/task) | ~10% (floor) | ~10% | ~64% |
| 5×2 (100 iters/task) | ~20% | ~20% | ~60% |

*Note: 5×2 ran 500 total updates vs 1000 for 10×1 — set `ITERS=200` for a matched budget.*

**2×2 ([0,1] then [2,3], 100 iters/task, batch 32)** — the most informative run:
- **backprop:** task 1 holds at 100% until ~step 110, then collapses to 0 as task 2 rises. Clean sequential trade, a cliff.
- **replay:** both tasks end ~95%. Retains.
- **eqprop:** *forgets before it learns* — task 1 collapses at the switch, task 2 rises only afterwards. Worst crossing.
- **pc:** *forgets more slowly* — task 1 decays over ~100 steps while task 2 rises, ending ~15%. Slope, not cliff.

### 4.3 Four methods × 10 random digit pairings (5×2 split, switch at step 100)

| Method | Crossover | Final task 1 | Final task 2 | Shape |
|---|---|---|---|---|
| backprop | ~65% | ~0% | ~97% | vertical cliff at the switch |
| replay | ~85% | **~68%** | ~96% | dips then recovers; only method ending up-right |
| eqprop | — | ~0% | ~95% | noisy throughout; forgets before it learns |
| pc | ~75% | ~8–10% | ~97% | slope not cliff; bows above the diagonal, still ends top-left |

- Trade-off efficiency (area above the diagonal): **pc > replay > backprop > eqprop**
- Final task-1 retention: **replay ≫ pc > eqprop > backprop**
- **These two orderings differ, and the difference is the whole story.**

> ⚠ Figures in 4.2/4.3 are read off plots and are approximate. Verify against logged
> metrics before citing. Two runs of the same family appear in the source documents with
> slightly different numbers (final task-1 retention ~37% vs ~68% for replay) — **resolve
> this discrepancy before any of it goes in the thesis.**

### 4.4 Premise correction on record
**EqProp is not forgetting less.** Its decay looks gentle only because everything is slower
(lower peak on task 1, slower and lower task 2). The trajectory plot removes time from the
axes and shows eqprop is the worst panel, below the diagonal throughout.

---

## 5. Mechanism analysis (the theory the project already owns)

### 5.1 One gradient explains the output layer
With logits `z_o` and softmax over active set 𝒜:

```
∂L/∂z_o = p_o − 1[o = t]
```

For the true class this is negative (descent raises `z_t`); for **every other active unit** it
is `p_o > 0` (descent lowers `z_o`). With `z_o = w_oᵀh + b_o`, an old class with no data
present gets `∂L/∂w_o = p_o·h` and `∂L/∂b_o = p_o` — so `b_o` drifts monotonically **down**
and `w_o` is pushed **anti-parallel to the current feature mean**. No old data is involved;
the trunk may be entirely intact.

### 5.2 The active set 𝒜 is the real knob
Scenario → 𝒜 → which gradient paths exist → which forgetting mechanisms can fire → which
method families can possibly work.

- **Task-IL / multi-head:** 𝒜 = current context only. Old heads are literally disconnected from the loss. No output-layer interference by construction — which is why EWC/SI work here.
- **Domain-IL / single fixed head:** head is shared and does receive gradients, but every class is positive in every context, so suppression is symmetric — no recency bias.
- **Class-IL / single global head:** only current classes ever appear as positives ⇒ asymmetric suppression at full strength.

### 5.3 Two distinct pathologies in Class-IL (the most useful decomposition so far)
1. **Logit suppression / task-recency bias** — a **calibration** failure. Features are fine; per-class scale and offset are wrong. **Cheap to fix, no replay needed.**
2. **Absent inter-context discriminative signal** — a **representation** failure. A boundary must be placed between classes never co-observed, with no gradient ever comparing them. **Irreducible**; needs old-class information from somewhere.

"Class-IL needs replay" is only true of pathology 2 — and even then "replay" can mean a
generative model or class prototypes rather than a buffer.

**Full mechanism map:**

| Mechanism | Lives in | Bites in | Replay-free fix |
|---|---|---|---|
| Trunk representation drift | shared layers | all three | EWC/SI, gating, freezing, prospective configuration |
| Head–feature mismatch | old head ∘ new features | Task-IL, Domain-IL | trunk stabilisation |
| **Logit suppression / recency bias** | output layer | **Class-IL only** | masking, cosine classifier, weight alignment, NCM |
| **Missing inter-context boundary** | decision function | **Class-IL only** | prototypes, generative classifier — or replay |

### 5.4 Dropping the softmax does not fix it — a correction on record
The suppression comes from the **one-hot target supplying zero for every absent class**, not
from the softmax. Softmax is the transport mechanism; the label is the source.
- Linear + MSE: `∂L/∂z_o = z_o − y_o = z_o` for absent classes → still pushes down.
- Sigmoid + BCE: `∂L/∂z_o = σ(z_o) > 0` → still pushes down.
- Units decouple in the forward pass; the **labels stay coupled**.

**Where the intuition is right:** MSE's suppression has a **fixed point at zero** — `w_o` is
driven until it is *orthogonal* to the current context's features, so if old-class features
occupy a different subspace, `w_o` can remain informative. Softmax has no fixed point:
`z_o` is driven toward −∞ relative to `z_t` and `w_o` accumulates an unbounded anti-`h`
component. Swapping an unbounded relative target for a bounded absolute one buys something
real — just not everything.

**The tension, stated plainly:** more coupling → better calibration, more suppression; less
coupling → less suppression, worse calibration. You cannot win both with the same knob.
A **generative classifier** resolves it: p(x|y) per class, no negatives at all, common scale
from every class model being a normalised density over the same input space rather than from
a shared denominator. *This is already the architecture that resolves the tension, and it is
what E_sharp + E_smooth is.*

### 5.5 Per-method mechanism summary

| Method | Clamping / update | Interference reduced? | Suppression cancelled? | Signature |
|---|---|---|---|---|
| backprop | forward pass, fixed | no | no | cliff to 0% |
| + replay | same | no | **yes** | floor, on-diagonal, high variance |
| eqprop | weak clamp, β→0 | no (≈ BP + noise) | no | slow, noisy, below diagonal |
| pc | strong clamp, full relaxation | **yes** | no | above diagonal, no floor |

- **Why EqProp is worst:** its hinge target is +1 for the true class and **−1 for all others**, so every example actively drives all nine other outputs down — more aggressive than softmax.
- **Why PC decays gently:** one-hot target (0, not −1) ⇒ weaker suppression; settle-then-update ⇒ smaller weight change needed.
- **Why EqProp ≈ noisy backprop:** Song & Bogacz state directly that previous work made EBNs approximate backprop by preventing activity changing substantially before weight modification, via an infinitesimal supervision signal — **naming equilibrium propagation as the example**. EqProp is a finite-difference estimator of backprop's gradient, inheriting the interference and adding estimator variance plus finite-β bias. Being below backprop is the **expected** result, not an anomaly.

**Headline synthesis:** PC and replay attack different rows of the mechanism table and are
empirically complementary. That PC still decays to ~10% is itself evidence that the dominant
failure in this setup is output-layer suppression rather than representation drift — fixing
credit assignment buys a better path but no floor; supplying old positives buys a floor.

### 5.6 The reframe that drives the interpretability work
If tasks used disjoint hidden units, PC's updates would concentrate on task-2 units and leave
task-1 units alone. With 64 hidden units, overlapping digit inputs and a single head,
representational overlap is high, so task-1 units get overwritten anyway.

**This changes the question from "is the error zero?" (no) to "where does the weight movement
go?" (measurable).** That is the core interpretability experiment.

---

## 6. Quarantine list — triage (chat 013)

The 17 items below are the accumulated idea backlog, numbered in the order they appear in the
source list. **Verdict codes:** `RUN` = schedule it · `MERGE` = already covered by a scheduled
experiment · `REFRAME` = do a cheaper or better-targeted version · `ANSWERED` = the literature
settles the direction; cite, or run only as a confirmation figure · `BLOCKED` = needs an
architecture or scope change first · `DEFER` = post-thesis.

### 6.1 Item-by-item

| # | Item (abridged) | Verdict | Where it goes / why |
|---|---|---|---|
| Q1 | Example ordering by distance from previous task mean; VAE; batch 1; forward/reverse/random | REFRAME + DEFER | **Does not need a VAE** — class-mean distance in pixel or feature space gives the same ordering for free. Task-similarity → forgetting is established (Nguyen 2019; Ramasesh 2021), and interference-based *replay* selection is done (MIR; energy-based selection). Advisor-deferred. |
| Q2 | Activations/receptive fields before vs after task 2; overwrite / amend / merge; parameter-space forgetting | **RUN** | Splits into **M1** (parameter space: where does |Δw| go) and **M2** (unit space: which units change role). M1 is already spec'd as experiment 16 and is the decisive mechanism test. |
| Q3 | Which part forgets — hidden states, output, or both? | **RUN — highest priority** | This is **D1**, the NCM / linear-probe head-vs-trunk decomposition. Cheap, decisive, and every other question's interpretation depends on it. Mechanism already derived in §5.3; the experiment confirms and quantifies it *per learning rule*. |
| Q4 | Freeze high-activation hidden units and quarantine them | ANSWERED (Task-IL form) + **RUN** (EBM form) | Unit gating/freezing is XdG (Masse 2018), PackNet, HAT: works well in Task-IL, needs task ID, and **saturates on capacity** — which is exactly Q17's intuition. Novel part is gating by the **EBM's own prediction error** rather than a task label — that is advisor point 4 and `eqprop_update_gated`. → **M3**. |
| Q5 | Softmax reset for absent classes; freeze it / damp negative updates | ANSWERED (mechanism) + **RUN** (measurement) | §5.4: the source is the **one-hot target**, not the softmax; MSE and sigmoid+BCE suppress too. So "freeze the softmax" is the wrong fix — masking the **active set** is the right knob. Measurement version → **D3**. |
| Q6 | BP/EWC + freezing for hidden layers, energy-based head on top | REFRAME | A recombination of parts whose individual behaviour is known: EWC fails in Class-IL (§4.1); the "energy head" is a generative classifier / Li et al. 2022 conditional-energy EBM. Keep as a discussion point; only build if **M3** succeeds. |
| Q7 | Does freezing early or late layers matter more? | BLOCKED + ANSWERED | **The current net has one hidden layer — this is untestable as configured.** Literature answer: later layers forget more, early features are general (Yosinski 2014; Ramasesh 2021). Comes free *if* depth is added for **C3**. |
| Q8 | Non-stationary / rotation MNIST via VAE feature sampling | REFRAME | Song & Bogacz Fig 4f–g **is** the concept-drift experiment, and it is where they claim PC's *largest* advantage. Reproducing that is far cheaper than building a VAE sampler and targets their strongest claim. Rotated MNIST is a standard benchmark if a drift dataset is needed. |
| Q9 | Batch size 1 so degradation is observable | **RUN** (as control) + ANSWERED (direction) | Song & Bogacz Fig 4a–c is explicitly the online/batch-1 setting; Dong & Wu note PC's advantage is largest at batch 1 and with depth. **At batch 32 you are measuring PC near its weakest point.** Also: batch 1 destroys EqProp (known failure mode) — report that as a result. → **C3**. |
| Q10 | Generation produces noise — has it not settled? | ANSWERED (own diagnosis) | The EqProp energy has **no ½x² self-term**, so during generation `x` has no restoring force and pins at the clamp bounds. Not a research question — one confirmation test, then close. → **W2**. |
| Q11 | EqProp: add a component of `free` for other classes to pin state | DEFER — superseded | The much cheaper and better-motivated first move is the **target-structure control** (**D2**): swap EqProp's ±1 hinge for a one-hot 0-target. That directly tests the project's own explanation for why EqProp is worst. Revisit Q11 only if D2 confirms the story. |
| Q12 | Per-node mean activation + head-normalisation metric, for BP and EBM; use to show the trade-off and to transfer mitigation | **RUN — the best-formed item on the list** | This is the umbrella measurement study. Splits into **D1** (head vs trunk), **D3** (head diagnostics over training), and **C2** (sparsity). Its second half — *does a mitigation identified in backprop transfer to the EBM?* — is **the strongest genuinely novel thread in the whole list**, because nobody has run the head/trunk decomposition on prospective configuration in Class-IL. |
| Q13 | Task size 1 vs 2 vs 5 classes per task | **MERGE** | 2×5 *is* the Bogacz reproduction split. Already scheduled as experiment 12. `TASKS` is already parameterised — no new experiment. |
| Q14 | Track fastest-varying parameters, stiffen on convergence; damping by historic acceleration | DEFER (already quarantined) | Closest relative is **Synaptic Intelligence** (online path integral), **not EWC** — but SI is first-order and does not decay. The "time component / delay mechanic" framing is Benna & Fusi's multi-timescale synapse. Genuinely not done in this exact form. **Gate on M1:** if PC already leaves task-1 weights alone, stiffness adds little. |
| Q15 | Track Fisher importance over training — when does it emerge, is it predictable? | RUN small, or DEFER | Cheap on an MLP and a nice supporting figure, but a side-quest w.r.t. the four points. Also has an unresolved hook from §4.1 (EWC's Fisher distribution resembles replay's yet accuracy doesn't). Do only if **M1** needs support. |
| Q16 | Sweep settling steps to show EqProp/PC → BP; implement PC settling-convergence test | **RUN** (methods figure) + ANSWERED (theory) | **Proven, not claimed** — Millidge et al. 2022 unify PC, EqProp and CHL as reducing to backprop in the infinitesimal inference limit. Low novelty, **high defensive value**: it validates the implementation and grounds the project's own reading of the EqProp result. → **W1**. |
| Q17 | Network size vs activation spikes vs entanglement vs freezing capacity | **RUN** (folded into width sweep) + ANSWERED (direction) | Wide networks forget less (Mirzadeh 2022); Hopfield saturation / blackout catastrophe is discussed in Kirkpatrick 2017 itself. Your sharper claim — *entanglement forces polysemantic units, so activation-gating must saturate* — is the superposition argument and is a good framing for why **M3** has a ceiling. → **C2**. |

### 6.2 Five research programmes

| P | Programme | Question it answers | Advisor point | Items |
|---|---|---|---|---|
| **P1** | **Localisation of forgetting** | Where does forgetting live — head, trunk, or both — and does that differ by learning rule? | 3 | Q3, Q5, Q12 |
| **P2** | **Trivial-explanation controls** | Do learning rate, width, sparsity, batch size or target structure explain the gap? | 3 | Q9, Q13, Q17, part of Q5 |
| **P3** | **Targeted plasticity in the EBM** | Can prediction-error-selected updating reduce forgetting? | 4 | Q2, Q4, Q6, Q14, Q15 |
| **P4** | **Regime dependence & methods validation** | Does the implementation behave as the theory says, and where is the EBM measured fairly? | 1, 2 | Q7, Q8, Q10, Q16 |
| **P5** | **Deferred: energy as memory** | Can one EBM carry a sharp memory component and a smooth task component? | — | Q1, Q11, E_sharp/E_smooth thread |

### 6.3 Already answered in the literature — cite, don't rediscover

| Question | Settled by | What it says |
|---|---|---|
| Do PC/EqProp reduce to backprop with few settling steps? | Millidge et al. 2022; Whittington & Bogacz 2017 | Yes, in the infinitesimal inference limit — proven for PC, EqProp and CHL alike. |
| Does forgetting live in the head or the trunk in Class-IL? | van de Ven 2022; iCaRL, LUCIR, BiC, WA | Both, but the output-layer bias is large and separately fixable — the entire Class-IL bias-correction literature exists because of it. |
| Does per-unit gating/freezing stop forgetting? | Masse 2018 (XdG); PackNet; HAT | Yes in Task-IL with task ID; capacity saturates as tasks accumulate. |
| Do early or late layers forget more? | Yosinski 2014; Ramasesh 2021 | Later layers; early features are general and transfer. |
| Does network width affect forgetting? | Mirzadeh 2022; Kirkpatrick 2017 (saturation) | Wider forgets less; past capacity, consolidation methods can do *worse* than plain SGD. |
| Is PC's advantage regime-dependent? | Song & Bogacz 2024 Fig 4a–c, f–g; Dong & Wu | Yes — largest at batch size 1, with depth, and under concept drift. |
| Does task ordering/similarity affect forgetting? | Nguyen 2019; Ramasesh 2021 | Yes; sequence and semantic similarity measurably change the forgetting profile. |
| Which samples benefit most from replay? | MIR (Aljundi 2019); energy-based selection (2026 MHN/diffusion result) | Maximally-interfered / high-energy outlier samples. |
| Does EWC work in Class-IL? | van de Ven 2022 Table 2; reproduced here | No — essentially at the no-defence baseline. |

### 6.4 Priority ordering — the sequential story

The thesis narrative is: *everything forgets → does the literature's claim reproduce → where
does the forgetting live → is the difference trivial → can we target it.* Nothing downstream
is interpretable until Tier 0 and Tier 1 are done.

**Tier 0 — unblock (already scheduled; not from the quarantine list)**
1. **Run script 11.** Establishes what is known with variance across pairings. Written, compile-checked, never run.
2. **Fix the nonlinearity/loss confound** (§7.1) before drawing any conclusion from the 4-way comparison.
3. **Experiment 12 — Bogacz reproduction.** Alternating 5+5 Class-IL, PC vs matched BP on tanh + squared error, per-rule LR grid, MNIST and Fashion-MNIST. *Absorbs Q13.* **Must come first: everything downstream assumes there is an effect to explain.**

**Tier 1 — the diagnostic core (highest value per unit of compute)**
4. **D1 — head-vs-trunk probe (Q3, Q12).** Freeze each trained net, discard the head, classify task 1 by nearest-class-mean and by a linear probe. If backprop jumps from ~0% toward the joint baseline, the trunk survived and the head was the whole problem. Prediction: PC's advantage *shrinks* under this readout relative to argmax.
5. **D2 — target-structure control (Q5, Q11).** Swap EqProp's ±1 hinge for a one-hot 0-target and PC's one-hot for ±1. One-line change. If EqProp improves and PC worsens, the suppression story in §5.5 is confirmed and it explains the entire method ordering. **Highest insight-to-effort ratio on the list.**
6. **D3 — head diagnostics over training (Q12).** Log ‖w_o‖ and b_o per class, per-node mean activation, and raw (pre-argmax) outputs, for all four methods. Produces the learning-vs-forgetting trade-off figure and the "which mitigation to try" evidence.

**Tier 2 — trivial-explanation controls (advisor point 3)**
7. **C1 — matched learning rate / matched task-2 learning speed.** Known confound (BP 0.05, EqProp 0.005, PC 0.05). Comparing forgetting speed at different learning speeds measures the learning rate, not the rule.
8. **C2 — width and sparsity sweep (Q17).** 16/32/64/128/256 hidden units; measure activation sparsity and per-class activation overlap per method. Also establishes the capacity ceiling that bounds M3.
9. **C3 — batch-size (and optionally depth) sweep (Q9, Q7).** 1/8/32/128. Adding a second hidden layer here also unblocks Q7 for free.

**Tier 3 — mechanism → mitigation (advisor point 4)**
10. **M1 — where does the weight movement go? (Q2)** Per-weight |Δw| during task 2, split by task-1 importance rank; overlap between "task-1-important" and "task-2-heavily-updated"; representational drift of task-1 inputs. **Confirm/refute:** PC concentrating updates away from task-1-important weights supports the locality mechanism; equal overlap despite less forgetting refutes it and points to "less erratic updates" instead. *Either outcome is a defensible result.*
11. **M2 — unit specialisation across tasks (Q2).** Receptive fields and activation profiles before/after task 2: overwrite vs amend vs merge. Prerequisite for M3 — gating protects nothing if the same units respond to every class.
12. **M3 — prediction-error gating (Q4, Q6).** `eqprop_update_gated` (written, untested) plus a PC version. PC's explicit per-node prediction error is the more natural "which nodes differ most" signal.
13. **M4 — freezing control (Q4).** Freeze the top-k task-1-important weights during task 2; measure how much forgetting disappears. Causal evidence, and a hard-consolidation mini-EWC.

**Tier 4 — write-up support**
14. **W1 — settling-steps sweep (Q16).** PC and EqProp → BP as steps shrink; implement the EqProp settling-convergence test for PC too. Methods-chapter figure; validates the implementation.
15. **W2 — generation diagnosis (Q10).** Confirm the missing ½x² self-term explains the noise. One test, then close.

**Tier 5 — defer**
16. Q14 (kinematic consolidation), Q15 (Fisher over time), Q1 (ordering), Q8 (drift via VAE), Q6 (hybrid), Q11 (EqProp anchor term), and the E_sharp/E_smooth thread → future-work chapter, gated on M1.

### 6.5 Corrections to the list, on record
- **Q5 is diagnostically wrong:** the softmax is the transport, the one-hot target is the source (§5.4). Freezing the softmax would not fix it.
- **Q14 is not EWC:** it is closest to Synaptic Intelligence, which is a first-order online path integral and does not decay.
- **Q16 is proven, not claimed.**
- **Q1 and Q8 do not need a VAE.** Both have far cheaper equivalents.
- **Q7 cannot be run on the current architecture.**

---

## 7. Constraints and known pitfalls

### 7.1 ⚠ The live confound
`make_backprop` / `make_replay` use **ReLU + CrossEntropyLoss**; `pc` and `eqprop` use
**tanh + squared error**. The current 4-way comparison therefore varies **three** things at
once: algorithm, nonlinearity and loss. The matched BP control is exactly PC's function class,
no biases:

```
x1   = x0 @ W1
out  = tanh(x1) @ W2
loss = ½ |target − out|²      # one-hot target, SGD
```

### 7.2 Hardware / runtime
- **CPU only.** A GPU helps less than expected: each settling step is a tiny matmul, so per-step overhead dominates. Parallel CPU processes beat one GPU for sweeps.
- Cost ordering: EqProp ≫ PC ≫ backprop. PC's *prediction* is a plain feedforward pass with no settling.
- Keep sweeps cheap: subset to 10k, 1 epoch, fewer settle steps; confirm only the winner on full data.

### 7.3 EqProp failure modes (hard-won)
- **Saturation is the killer.** As weights grow, `tanh` flattens, `tanh'(h) → 0`, and the feedback path carrying the nudge is severed. Track `% of |tanh(h)| > 0.95` as a first-class diagnostic. Low `lr` is the main control.
- **The nudged phase never reaches an absolute tolerance** — the hinge keeps pushing while the margin is unmet, so per-step movement plateaus at a non-zero floor. Settling therefore stops on **patience**, not on a fixed `tol`. Worth a sentence in the write-up.
- **Warm-start the nudged phase from the free equilibrium**, or it dominates runtime.
- **Batch size 1 destroys EqProp.** With ±1 targets and no batch to average over, every update reconfigures the network to the most recent image. Slow training with the learning rate; keep batch ≥16. *(Note the tension with C3 — this is itself a reportable finding.)*
- **No ½x² self-term in the energy**, so during generation `x` pins at the clamp bounds → weak generator.

### 7.4 Metric pitfalls (all previously hit)
- **Never report train-batch accuracy.** A previous bug did this and invalidated a day of sweeps. Always use the held-out test split.
- **`cur%` is degenerate at 1 class/task** (predicting one class always scores 100%). **`seen%` has a changing denominator.** Prefer **per-task accuracy with fixed class sets.**
- **Accuracy is a threshold readout.** After a switch nothing appears to happen for ~20 steps while logits climb, then it flips. **Log raw outputs** to see the continuous dynamics.
- The flat line at exactly 10% / 20% / 25% is not chance — it is the **collapse floor** `100/n_classes`.
- Don't fit sigmoids to accuracy curves. Use threshold crossings and the **ACC1-vs-ACC2 trajectory** — the most robust forgetting metric found so far, and it removes time from the picture.

### 7.5 Scientific constraints carried from the literature
- **Capacity:** the sharp store cannot grow unbounded — Hopfield systems hit blackout catastrophe at saturation. This is why consolidation exists.
- **Pure memorisation cannot generate novel samples.** For recombinant replay, target the *intermediate*-noise regime, not the sharpest energy.
- **Generative ≠ automatically better for discrimination.** van de Ven found a real gap between BI-R and the generative classifier despite both having latent generative models.
- **Parameter cost:** generative-model methods used up to ~3× the parameters of discriminative baselines.
- **PC requires symmetric forward/backward weights** — a plausibility issue PCNs *share* with backprop, as are signed real-valued error signals. PCNs differ from backprop only on feedback influencing activity during inference.

### 7.6 Personal working constraints
- Plain language, no flowery metaphors. Concept first (ELI5 + graduate), then decisions with trade-offs, then code.
- One stage at a time; small increments; minimal machinery. Over-complication has repeatedly caused loss of momentum.
- Motivation dips have occurred. The engaging threads are: *what happens inside the network when a new class arrives* (overwritten, reused, or newly allocated) and *the EBM as its own replay generator*. Keep those visible.

---

## 8. Code

```
project/
├── data/
├── src/
│   ├── data.py                 # load_mnist, class_indices, make_eval_set
│   ├── eqprop.py               # eqprop_init/energy/settle/update/predict
│   │                           #   + eqprop_update_gated, eqprop_generate
│   ├── predictive_coding.py    # pc_init/forward/settle/update/predict
│   ├── methods.py              # make_backprop, make_replay, make_eqprop, make_pc,
│   │                           #   make_eqprop_gated, make_eqprop_replay, make_eqprop_synthetic
│   └── plotting.py             # plot_learning_curves, plot_trajectory
└── experiments/
    ├── 09_eqprop_learning_vs_forgetting.py
    ├── 10_pc_learning_vs_forgetting.py
    └── 11_consolidate_pairs_4methods.py    # written, NOT YET RUN
```

**Interface contract:** every `make_*` returns `(train_step, predict)`. `train_step(x, y)` does
one update; `predict(x, raw=False)` returns class indices or raw pre-argmax outputs. Adding a
model = one new `make_*`; experiment scripts change only the `methods` dict.

**Current hyperparameters:**
```python
IMG_SIZE = 14 ; IN_DIM = 196 ; HIDDEN = 64 ; OUT = 10
BATCH = 32 ; ITERS = 100 per task ; EVAL_PER_CLASS = 100 ; N_RUNS = 10
BP_LR  = 0.05
RP_LR  = 0.05 ; RP_PER_CLASS = 20
EQP_LR = 0.005 ; EQP_BETA = 0.3 ; EQP_DT = 0.3 ; EQP_MAX_STEPS = 500 ; EQP_SETTLE_PAT = 30
PC_LR  = 0.05  ; PC_DT = 0.1 ; PC_STEPS = 50
```
*(Joint-training EqProp config reaching ~91%: lr 0.03–0.1, beta 0.3–0.5, dt 0.3–0.5, batch 32–64.)*

**PC core (finite-difference verified to ~1e-9):**
```
x0 (clamped) -> x1 (free) -> x2 (clamped to target while training)
mu1 = x0 @ W1 ;  e1 = x1 - mu1
mu2 = tanh(x1) @ W2 ;  e2 = x2 - mu2
F   = ½|e1|² + ½|e2|²
Inference: relax x1 to reduce F with the target clamped.
Learning : dW1 = x0ᵀ e1 ,  dW2 = tanh(x1)ᵀ e2
```

**Metrics implemented:** `crossover(steps, t1, t2, switch)` (accuracy at the post-switch
crossing — high = held both, low = pure trade); `first_cross(...)` (steps to learn / to forget);
ACC1-vs-ACC2 trajectory plot.

**Snippets for the next experiments** (NCM probe, masked loss, cosine classifier, target
alignment) are in `energy-based-memory-and-continual-learning.md` §5.5.

---

## 9. Open questions

**Live (affect current work):**
1. Does the head-vs-trunk decomposition (§5.3) hold *per learning rule*, or does PC's advantage live somewhere else entirely? → D1.
2. Does the ±1 vs one-hot target structure explain the whole method ordering? → D2.
3. Resolve the discrepancy between the two reported four-method result sets (§4.3 warning).
4. Confirm the evaluation protocol is global argmax over a shared head, as the 0%-not-50% signature implies.
5. EWC's Fisher-importance distribution resembles replay's, yet its accuracy matches baseline — is the failure in the readout rather than the features? Test = linear probe on the EWC trunk. *Never run.*

**Deferred (P5):**
6. Can E_sharp and E_smooth be trained under a single unified objective, or does the sharp component need a separate fast-weight rule?
7. What is the right readout for replay — sharpest energy (verbatim) or intermediate noise (recombinant)? Literature points to intermediate.
8. What is the consolidation pathway from sharp to smooth, and what triggers it in a task-free stream?
9. Does PC + replay compose additively, or does PC's reduced interference change which samples are worth replaying?
10. Does energy-based replay selection (high-energy first) survive in a decomposed energy?

---

## 10. References

**Link status:** ✅ = link taken from a project PDF or handoff doc, verified in-project.
⚠ = identifier from general knowledge, **confirm before citing in the thesis.**

### 10.1 In-project PDFs
| Ref | Link | One-line summary |
|---|---|---|
| **Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz (2024).** "Inferring neural activity before plasticity: a foundation for learning beyond backpropagation." *Nat Neurosci* 27:348–358. `song_bogacz_24.pdf` | Code ✅ https://github.com/YuhangSong/Prospective-Configuration | The source of the "energy-based learning reduces interference" claim; Fig 1 interference, Fig 3b–e target alignment, **Fig 4d–e continual learning (the reproduction target)**, Fig 4a–c batch size, Fig 4f–g concept drift. |
| **van de Ven, Tuytelaars & Tolias (2022).** "Three types of incremental learning." *Nat Mach Intell* 4:1185–1197. `s42256022005683.pdf` | ✅ https://doi.org/10.1038/s42256-022-00568-3 · code ✅ https://github.com/GMvandeVen/continual-learning | Defines Task-IL / Domain-IL / Class-IL and shows EWC and SI collapse to baseline in Class-IL while replay holds up across all three. |
| **Dong, Peng & Wu (2025).** Commentary on Song et al. *Intelligent Computing.* `dong_wu_rev_song_bogacz.pdf` | (in project) | Frames PC as strong-clamp and EqProp as weak-clamp within an EM view, and flags that PC's advantage is largest at batch size 1 and with depth. |
| **Kirkpatrick et al. (2017).** "Overcoming catastrophic forgetting in neural networks." *PNAS* 114:3521–3526. `kirkpatrick_17.pdf` | ⚠ https://doi.org/10.1073/pnas.1611835114 | Introduces EWC (Fisher-weighted quadratic penalty) and discusses Hopfield saturation and blackout catastrophe past capacity. |

### 10.2 Core external
| Ref | Link | One-line summary |
|---|---|---|
| **Millidge, Song, Salvatori, Lukasiewicz & Bogacz (2022).** "Backpropagation at the infinitesimal inference limit of energy-based models." | ✅ https://arxiv.org/abs/2206.02629 | The formal proof that PC, EqProp and contrastive Hebbian learning all reduce to backprop in the infinitesimal inference limit — settles Q16. |
| **Scellier & Bengio (2017).** "Equilibrium propagation." *Front Comput Neurosci* 11:24. | ⚠ https://doi.org/10.3389/fncom.2017.00024 | The chosen EBM: a two-phase contrastive learning rule using free and weakly-nudged equilibria of a Hopfield-style energy. |
| **Whittington & Bogacz (2017).** "An approximation of the error backpropagation algorithm in a predictive coding network…" *Neural Computation.* | ⚠ (search by title) | Shows PC with partial relaxation approximates backprop — the small-step regime of §3.1. |
| **Song, Lukasiewicz, Xu & Bogacz (2020).** "Can the brain do backpropagation? Exact implementation of backpropagation in predictive coding networks." *NeurIPS 33.* | ⚠ (NeurIPS proceedings) | Engineers exact BP equivalence in a PCN — but Dong & Wu note the equivalence is not general. |
| **Li, Du, van de Ven & Mordatch (2022).** "Energy-based models for continual learning." *CoLLAs.* | ✅ https://arxiv.org/abs/2011.12216 · code ✅ https://github.com/ShuangLI59/ebm-continual-learning | The conditional-energy EBM that **does** beat replay in Class-IL, by dropping softmax normalisation over all classes in favour of contrastive divergence. |
| **van de Ven, Siegelmann & Tolias (2020).** "Brain-inspired replay." *Nat Commun* 11:4069. | ⚠ (search by title) | Generative replay with the generator internal to the classifier; hippocampus modelled as a generative network rather than a buffer. |
| **McCloskey & Cohen (1989).** "Catastrophic interference in connectionist networks." | (book chapter) | The original demonstration that sequential training overwrites prior associations. |

### 10.3 Consolidation / metaplasticity
| Ref | Link | One-line summary |
|---|---|---|
| **Zenke, Poole & Ganguli (2017).** Synaptic Intelligence. *ICML.* | ⚠ https://arxiv.org/abs/1703.04200 | Accumulates per-synapse importance online as a path integral along the trajectory — the closest existing relative of Q14, but first-order and non-decaying. |
| **Aljundi et al. (2018).** Memory Aware Synapses. *ECCV.* | ⚠ https://arxiv.org/abs/1711.09601 | Unsupervised online importance from the gradient of output magnitude, needing no labels or task boundaries. |
| **Benna & Fusi (2016).** "Computational principles of synaptic memory consolidation." *Nat Neurosci.* | ⚠ https://arxiv.org/abs/1507.07580 | Multi-timescale complex-synapse model — the biological grounding for the "delay mechanic" intuition in Q14. |
| **Laborieux, Ernoult, Hirtzlin & Querlioz (2021).** "Synaptic metaplasticity in binarized neural networks." *Nat Commun.* | ⚠ (search by title) | Makes plasticity itself plastic, achieving continual learning without task boundaries or replay. |

### 10.4 Gating, freezing, capacity
| Ref | Link | One-line summary |
|---|---|---|
| **Masse, Grant & Freedman (2018).** Context-dependent gating (XdG). *PNAS.* | ⚠ (search by title) | Gating a random subset of units per task plus stabilisation largely removes forgetting — in Task-IL, with task ID available. |
| **Mallya & Lazebnik (2018).** PackNet. *CVPR.* | ⚠ https://arxiv.org/abs/1711.05769 | Iterative pruning frees capacity for new tasks and freezes what each task claimed — and demonstrably runs out. |
| **Serra et al. (2018).** Hard Attention to the Task (HAT). *ICML.* | ⚠ https://arxiv.org/abs/1801.01423 | Learns per-task attention masks over units, with an explicit capacity/compression trade-off. |
| **Mirzadeh et al. (2022).** "Wide neural networks forget less catastrophically." *ICML.* | ⚠ (search by title) | Width reduces forgetting, largely via sparser and more orthogonal representations — the answer direction for Q17. |
| **Elhage et al. (2022).** "Toy models of superposition." | ✅ https://transformer-circuits.pub/2022/toy_model/index.html | Networks represent more features than they have units by superposing them — the mechanism behind Q17's entanglement intuition. |
| **Yosinski, Clune, Bengio & Lipson (2014).** "How transferable are features in deep neural networks?" *NeurIPS.* | ⚠ https://arxiv.org/abs/1411.1792 | Early layers learn general features, later layers task-specific ones — the answer direction for Q7. |
| **Ramasesh, Dyer & Raghu (2021).** "Anatomy of catastrophic forgetting." *ICLR.* | ⚠ (search by title) | Forgetting concentrates in later layers, and its severity tracks semantic similarity between tasks. |

### 10.5 Class-IL bias correction (pathology 1 fixes)
| Ref | Link | One-line summary |
|---|---|---|
| **Rebuffi et al. (2017).** iCaRL. *CVPR.* | ⚠ https://arxiv.org/abs/1611.07725 | Sigmoid+BCE training with nearest-class-mean inference plus exemplars — the origin of the NCM probe used in D1. |
| **Hou et al. (2019).** LUCIR. *CVPR.* | ⚠ (search by title) | A cosine classifier removes the magnitude degree of freedom that logit suppression corrupts. |
| **Wu et al. (2019).** BiC. *CVPR.* | ⚠ (search by title) | Fits a two-parameter correction to the biased logits of new classes using a small validation set. |
| **Zhao et al. (2020).** Weight Aligning. *CVPR.* | ⚠ (search by title) | Rescales output-layer weight norms post hoc to undo task-recency bias without any stored data. |

### 10.6 Streams, ordering, benchmarks
| Ref | Link | One-line summary |
|---|---|---|
| **Aljundi et al. (2019).** Maximally Interfered Retrieval. *NeurIPS.* | ⚠ https://arxiv.org/abs/1908.04742 | Selects replay samples by predicted interference rather than at random — the established version of Q1's ordering intuition. |
| **Aljundi et al. (2019).** Task-free continual learning. *CVPR.* | ⚠ https://arxiv.org/abs/1812.03596 | Continual learning without task boundaries at training time — the target stream regime. |
| **Lopez-Paz & Ranzato (2017).** GEM. *NeurIPS.* | ⚠ https://arxiv.org/abs/1706.08840 | Gradient-constrained replay; also the standard source of the Rotated MNIST benchmark relevant to Q8. |
| **Nguyen et al. (2019).** "Toward understanding catastrophic forgetting in continual learning." | ⚠ (search by title) | Task sequence properties and inter-task similarity systematically predict how much is forgotten. |
| **Maltoni & Lomonaco (2019).** "Continuous learning in single-incremental-task scenarios." *Neural Networks* 116:56–73. | ⚠ (search by title) | The MT/SIT/MIT stream-carving taxonomy, orthogonal to van de Ven's scenarios. |
| **Moreno-Torres et al. (2012).** "A unifying view on dataset shift in classification." *Pattern Recognition.* | ⚠ (search by title) | Covariate / prior-probability / concept shift — the data-shift axis of the taxonomy. |

### 10.7 Hardware framing (EqProp's real value proposition)
| Ref | Link | One-line summary |
|---|---|---|
| **Kendall et al. (2020).** | ✅ https://arxiv.org/abs/2006.01981 | EqProp on analog hardware, where settling is physics rather than computation. |
| **Martin et al. (2021).** EqSpike. | ✅ https://arxiv.org/abs/2010.07859 | A spiking, neuromorphic implementation of EqProp — the natural framing for the EqProp chapter. |

### 10.8 Adjacent, from the Dong & Wu reference list
| Ref | Link | One-line summary |
|---|---|---|
| **Dong, Peng & Wu (2025).** "Predictive learning in energy-based models with attractor structures." | ✅ https://arxiv.org/abs/2501.13997 | Combines predictive learning with attractor dynamics in an EBM — adjacent to the E_sharp/E_smooth thread. |
| **Peng, Dong & Wu (2025).** "Vector quantization in the brain: grid-like codes in world models." | ✅ https://arxiv.org/abs/2510.16039 | Discretised codes as a brain-like route to pattern separation. |

*A fuller literature review exists as `ebm_literature_review.md` (biological plausibility by
brain region, hardware requirements, which EBMs have demonstrated CF mitigation and on which
IL tasks). Not currently in the project files — re-add if needed.*
