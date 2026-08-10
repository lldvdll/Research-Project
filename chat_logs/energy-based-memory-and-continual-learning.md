# Energy-Based Memory & Continual Learning — Context Transfer

> Context-handoff document summarising a research thread on energy-based models, memory/generalization decomposition, continual-learning taxonomy, and the mechanics of catastrophic forgetting. Written to be self-contained: a new workspace should be able to resume from this alone.

---

## 1. Main Goal / Task

**Core hypothesis under investigation.** Can a *single* energy-based model carry two components simultaneously?

- A **spiky / true-to-sample** component — sharp, per-example energy basins that behave like deterministic memory. Intended use: **generation, specifically generative replay for continual learning**.
- A **smoothed** component — an interpolated distribution over accumulated experience. Intended use: **discrimination on task**.

The motivation is neuroscientific: memory encoded in synaptic weights, hippocampal replay driving consolidation, and the observation that ML is largely interpolation of a data distribution from limited samples — so *less smoothing between samples ⇒ more the model behaves like deterministic memory*. The proposal is essentially a single-model reframing of Complementary Learning Systems (CLS) theory.

**Sub-goals that emerged during the thread:**

1. Establish plausibility and locate precedent in contemporary literature.
2. Get precise about continual-learning taxonomy, so the target regime can be named unambiguously.
3. Understand the actual *mechanics* of forgetting — specifically where it lives in the architecture, and what is/isn't fixable without replay.
4. Interpret empirical results from a four-method comparison on split MNIST.

---

## 2. Key Decisions & Conclusions Reached

### 2.1 On the core hypothesis — verdict: plausible, partially precedented, novel in one specific respect

- **The smoothing knob is real and has a name.** The memorization↔generalization transition in dense associative memory is controlled by inverse temperature β, which sets the steepness of energy basins around stored patterns. Sharp basins → each training point is its own attractor (deterministic recall); softened → basins merge into manifold-like attractors that generalize. There is a phase-transition picture: single data point → memorization; few points → memorization or spurious phase; many points → generalization phase. In kernel-density terms this is bandwidth; in statistics it is bias–variance made geometric.

- **A diffusion model is already the two-component object.** Low-noise diffusion energy is asymptotically the modern-Hopfield energy (sharp, per-sample basins); high noise gives the smooth generalizing landscape. A score/diffusion model is an EBM (score = −∇E), so it is a **single energy function carrying both regimes, indexed continuously by noise scale σ**. This is strong evidence the decomposition is realizable in one model rather than needing two networks.

- **The role assignment is CLS and has been instantiated.** VAE+MHN architectures exist where the MHN does pattern separation (distinct episodic storage) and the VAE does pattern completion (generalized representations from replayed memories) — hippocampus vs neocortex, bridged by generative replay. The "use the generative model to make replay samples" idea is the deep generative replay (DGR) lineage; the argument there is that the hippocampus parallels a generative model better than a replay buffer, since reactivation yields recombined rather than verbatim output.

- **Most on-target precedent (May 2026).** A paper doing continual learning in modern Hopfield networks via the diffusion correspondence proves a piece of the hypothesis directly:
  - High-energy, outlier-like samples undergo a **larger energy increase** than cluster-like samples ⇒ samples in sharp, isolated basins are **more forgettable**.
  - **Replay is particularly effective for high-energy samples**, enabling *energy-based selection* of replay samples.
  - I.e. the spiky, true-to-sample components are exactly the ones that (a) behave like memory, (b) forget first, (c) most benefit from replay — including a prescription for *which* samples to replay.

- **Where this project pushes past current work.** Almost all CLS-inspired ML uses **two separate modules** (MHN + VAE, generator + classifier). The proposal here — a **single EBM with a decomposed energy**, e.g.

  ```
  E(x) = E_sharp(x; fast, high-precision, pattern-separated)
       + E_smooth(x; slow, averaged)
  ```

  with the sharp readout seeding replay and the smooth readout serving the task, under a unified objective — is the less-explored and more elegant version. The diffusion result suggests feasibility (noise level is the readout knob).

- **Non-EBM analogues (existence proofs of the interpolation architecture):** semi-parametric retrieval (kNN-LM style `p = λ·p_memory + (1−λ)·p_smooth`, where λ is literally the memory/generalization dial); and fast-weight/plasticity methods splitting weights into slow-generalizing and fast-Hebbian-memory.

### 2.2 On taxonomy — three orthogonal axes, not one list

This was needed to name the target regime precisely. **Most confusion in the literature comes from treating these as one taxonomy.**

| Axis | Question it answers | Canonical source |
|---|---|---|
| **Data shift** | Which factor of p(x,y) is non-stationary? | Moreno-Torres et al. 2012 |
| **IL scenario** | What must the model output, and what is it told? | van de Ven et al. 2022 |
| **Stream carving** | How is the stream cut into increments? | Maltoni & Lomonaco 2019 |

**Data shifts** (factorization view: p(x,y) = p(y|x)·p(x) = p(x|y)·p(y)):

| Shift | Changes | Held fixed | Where it lands architecturally |
|---|---|---|---|
| Covariate | p(x) | p(y\|x) | features / BatchNorm running stats (AdaBN often fixes it free) |
| Prior probability (label/target shift) | p(y) | p(x\|y) | **output layer biases only** — logit adjustment fixes it |
| Concept | p(y\|x) | p(x) | output weights; unavoidable overwrite in a single shared head |

**IL scenarios** (test-time axis — distinguished by whether context identity is known at test and, if not, whether it must be *inferred*):

| | Task-IL | Domain-IL | Class-IL |
|---|---|---|---|
| Mapping | f: X × C → Y | f: X → Y | f: X → Y × C |
| Context ID at test | given | not given, not needed | not given, **must be inferred** |
| Output layer | multi-head | single, fixed | single, global/expanding |
| Cross-context discrimination | never required | not required | **required** |

**Training-time axis (separate, frequently conflated):** *task-free* = no boundaries during training; *task-agnostic* = the method requires no context labels at any point. Usage is inconsistent across papers — best practice is to state explicitly whether boundaries are given at training and whether IDs are given at test.

**Stream carving (Maltoni & Lomonaco):** MT (isolated tasks, multi-head) / SIT (one task refined over time, single head) / MIT (several tasks, each incremental). Orthogonal content types inside SIT: NI (new instances), NC (new classes), NIC (both). SIT+NC ≈ Class-IL.

> ⚠️ **The taxonomies genuinely disagree.** Permuted MNIST is classified as **MT** by Lomonaco (tasks are isolated) but as **Domain-IL** — explicitly *not* Task-IL — by van de Ven. Neither is wrong; they measure different things. Check which axis a paper means by "multi-task".

**➤ Target regime for this project: Class-IL under a task-free stream.** This is the one cell where regularization provably isn't enough, where the deficit is specifically *inter-context discrimination*, and where actual samples from old contexts are therefore required.

### 2.3 On the mechanics of forgetting — the load-bearing insight

**One gradient explains the output-layer behaviour.** With logits z_o and softmax over the **active set** 𝒜:

```
∂L/∂z_o = p_o − 1[o = t]
```

For the true class the gradient is negative (descent raises z_t); for **every other active unit** it is p_o > 0 (descent lowers z_o). With z_o = w_oᵀh + b_o, an old class with no data present gets `∂L/∂w_o = p_o·h` and `∂L/∂b_o = p_o` — so over thousands of iterations b_o drifts monotonically **down** and w_o is pushed **anti-parallel to the current feature mean**. No old data involved; the trunk may be intact.

**The active set 𝒜 is the real knob.** Scenario → 𝒜 → which gradient paths exist → which forgetting mechanisms can fire → which method families can possibly work.

- **Task-IL / multi-head:** 𝒜 = current context only. Old heads are *literally disconnected from the loss* (zero gradient). No output-layer interference by construction. Only trunk drift can forget — which is why EWC/SI work well here, and why Separate Networks has exactly zero forgetting.
- **Domain-IL / single fixed head:** head is shared and does receive gradients, but every class is positive in every context, so suppression is **symmetric** — no recency bias. Intermediate difficulty.
- **Class-IL / single global head:** only current-context classes ever appear as positives ⇒ asymmetric suppression at full strength.

**➤ Two distinct pathologies in Class-IL (the most useful decomposition in this thread):**

1. **Logit suppression / task-recency bias** — a **calibration** failure. Features are fine; per-class scale and offset are wrong. **Cheap to fix, no replay needed.**
2. **Absent inter-context discriminative signal** — a **representation** failure. Must place a boundary between classes never co-observed, with no gradient ever comparing them. **Irreducible**; requires information about old classes from somewhere.

"Class-IL needs replay" is only true of pathology 2 — and even then "replay" can mean a generative model or class prototypes rather than a buffer.

**Full forgetting-mechanism map:**

| Mechanism | Lives in | Bites in | Replay-free fix |
|---|---|---|---|
| Trunk representation drift | shared layers | all three | EWC/SI, gating, freezing, prospective configuration |
| Head–feature mismatch | old head ∘ new features | Task-IL, Domain-IL | trunk stabilization |
| **Logit suppression / recency bias** | output layer | **Class-IL only** | masking, cosine classifier, weight alignment, NCM |
| **Missing inter-context boundary** | decision function | **Class-IL only** | prototypes, generative classifier — or replay |

### 2.4 On dropping the softmax — an important correction

Question asked: would replacing softmax with a linear/activated output unit remove normalization-based forgetting?

**Answer: partly, but the diagnosis is off.** The suppression comes from the **one-hot target supplying zero for every absent class**, not from the softmax. Softmax is the transport mechanism; the label is the source.

- Linear + MSE: `∂L/∂z_o = z_o − y_o` = z_o for absent classes → still pushes down.
- Sigmoid + BCE: `∂L/∂z_o = σ(z_o) − y_o = σ(z_o) > 0` → still pushes down.
- Units get decoupled in the forward pass; the **labels stay coupled**.

**Where the intuition is right:** MSE's suppression has a **fixed point at zero** — w_o is driven until it is *orthogonal* to the current context's features, so if old-class features occupy a different subspace, w_o can remain informative. Softmax has no fixed point: it is a competition, z_o is driven toward −∞ relative to z_t, and w_o accumulates an unbounded anti-h component. So swapping an unbounded relative target for a bounded absolute one buys something real — just not everything.

**Three components that were being treated as one:**

| Component | Function | Cost of removing |
|---|---|---|
| Normalizer Σⱼe^{zⱼ} | couples classes; **provides a common scale** | loses calibration |
| Negative targets in the label | **the actual source of suppression** | requires masking |
| Active set 𝒜 | which units the loss touches | *this is the real knob* |

**The sting:** the shared denominator is exactly what makes argmax over the full label set meaningful — it is the pathology-2 machinery. Dropping softmax discards the calibration mechanism while retaining the suppression source. With a raw linear readout, nothing pins each unit's scale, so cross-head argmax is near-arbitrary.

**The tension, stated plainly:** more coupling → better calibration, more suppression; less coupling → less suppression, worse calibration. *You cannot win both with the same knob.* Hence solutions get calibration from somewhere other than loss structure.

**Resolution relevant to this project:** a **generative classifier** trains p(x|y) per class in isolation, with **no negatives at all** (zero suppression by construction), and **the common scale comes from every class model being a normalized density over the same input space x, rather than from a shared denominator**. That is cross-class comparability *without* cross-class coupling during training — bought with a shared domain instead of a shared normalizer. **This is already the architecture that resolves the tension, and it is what E_sharp + E_smooth is.**

### 2.5 On the empirical results (4 methods × 10 seeds, 5×2 split MNIST)

**Premise correction: eqprop is NOT forgetting less.** Its decay looks gentle only because everything is slower (lower peak on task 1, slower/lower task 2). The trajectory plot removes time from the axes and shows eqprop is the **worst** panel — mean trajectory well below the diagonal throughout.

- Trade-off efficiency (area above diagonal): **pc > replay > backprop > eqprop**
- Final task-1 retention: **replay (~37%) ≫ pc (~10%) > eqprop (~3%) > backprop (~0%)**
- These two orderings differ, and the difference is the whole story.

**The 0%-not-50% tell.** Backprop's task-1 accuracy converges to **zero**, not the ~50% expected from a randomized 2-class decision. Below-chance accuracy is only possible if argmax is systematically captured by task-2 units ⇒ evaluation is **global argmax over a shared head**, and the failure is the softmax/one-hot suppression above, *not* representation loss. *(Worth confirming against the eval code.)*

**Per-method mechanism:**

- **Backprop → cliff.** (i) Output layer: task-2 samples supply target 0 for task-1 units for 100 steps. (ii) Hidden layers: each layer's update assumes the others are fixed, so layers interfere. Song & Bogacz quantify (ii) as **target alignment** (cosine between direction-to-target and direction learning actually moves the output), which degrades with depth.
- **Replay → a floor, high variance.** Replayed old-class samples are **positives**, so `p_o − 1[o=t]` flips sign and *directly cancels* the suppression; it also restores minibatch co-occurrence, supplying the inter-class boundary signal. Note its trajectory sits **on** the diagonal — it isn't reducing interference, it's paying for retention with capacity. Thin-line spread (5%–55%) is buffer-composition variance.
- **PC → best path, no floor.** PCN clamps input *and* output to target, relaxes to convergence, then updates weights at that settled state. Hidden activity has already moved to a configuration consistent with the correct output *before* any weight moves. **But PC attacks cross-layer trunk interference and does nothing about one-hot targets still supplying zeros** — hence path above the diagonal but endpoint (10%) far below replay's (37%).
- **EqProp → backprop with a noisier estimator.** *This is the key finding.* Song & Bogacz state directly that previous work made EBNs approximate backprop by **preventing neural activity from changing substantially before weight modification**, via an infinitesimally small supervision signal — **naming equilibrium propagation as the example**. EqProp's weak clamp with β→0 suppresses the very activity shift that *is* prospective configuration. Millidge et al. (2022) prove PC/EqProp/CHL all reduce to backprop in the infinitesimal-inference limit. So EqProp is a **finite-difference estimator of backprop's gradient** — inheriting the interference, adding estimator variance (scattered thin lines) and finite-β bias (depressed peaks). Being below backprop is the *expected* result, not an anomaly.

| Method | Clamping / update | Interference reduced? | Suppression cancelled? | Signature |
|---|---|---|---|---|
| backprop | forward pass, fixed | no | no | cliff to 0% |
| + replay | same | no | **yes** | floor, on-diagonal, high variance |
| eqprop | weak clamp, β→0 | no (≈ backprop + noise) | no | slow, noisy, below diagonal |
| pc | strong clamp, full relaxation | **yes** | no | above diagonal, no floor |

**➤ Headline synthesis: PC and replay attack different rows of the mechanism table and are empirically complementary.** The fact that PC still decays to 10% is itself *evidence* that the dominant failure in this setup is output-layer/one-hot suppression rather than representation drift — fixing the credit-assignment rule buys a better path but no floor; supplying old positives buys a floor.

---

## 3. Current Status & Next Steps

**Status.** Conceptual groundwork complete. One empirical comparison run: 4 methods (FFNN+backprop, FFNN+backprop+replay, PCN+EqProp, PCN+prospective configuration) × 10 random digit pairings on 5×2 split MNIST, 200 steps with task switch at step 100. Two figures produced (learning/forgetting curves; trajectory through (task1, task2) accuracy space).

**Next steps, in priority order:**

1. **NCM probe — separates pathology 1 from pathology 2.** Freeze each trained network, discard the head, classify task 1 by nearest-class-mean on features. If backprop jumps from 0% toward joint baseline, the representation survived and the softmax layer was the entire problem. Also predicts PC's advantage should *shrink* under this readout relative to argmax.
2. **Interference-regime check.** Dong et al. note PC's advantage is largest at batch size 1 and in deep networks, and backprop stays competitive when interference is minimal (shallow, large-batch). If the current net is shallow / batch is large, PC is being measured near its weakest point; the gap should widen with depth.
3. **Combine PC + replay.** Directly implied by the mechanism analysis — they attack different rows and neither subsumes the other. This is the obvious missing cell in the current 2×2.
4. **Sketch the concrete E_sharp + E_smooth objective** and the replay loop off the sharp component. *Offered in-thread but not yet taken up.*
5. Consider energy-based replay *selection* (high-energy/outlier samples first), per the May 2026 modern-Hopfield/diffusion result.

---

## 4. Important Constraints

- **Capacity.** The sharp store cannot grow unbounded — Hopfield-type systems hit "blackout catastrophe" at saturation. This is *why* the brain consolidates (sharp→smooth transfer via replay) and arguably why it forgets. The spiky component needs a **consolidation pathway, not just storage**.
- **Pure memorization can't generate novel samples.** It can only replay stored items. For brain-like recombinant replay, target the **intermediate**-noise regime (or pattern-separated compressed indices + generative decoder, à la internal replay) — *not* the absolute sharpest energy.
- **Discrimination-from-generative is not automatically better.** van de Ven found a real gap between BI-R and the generative classifier despite both having latent generative models; *how* the smooth component is used for the decision matters.
- **Parameter cost.** Generative-model methods (DGR, BI-R, generative classifier) used up to ~3× the parameters of discriminative baselines in van de Ven's comparison.
- **PC requires symmetric forward/backward weights.** The Dong commentary flags this as an unresolved biological-plausibility issue that PCNs *share* with backprop (as are signed real-valued error signals). PCNs differ from backprop only on the third point — feedback influencing activity during inference. The commentary's position: PCNs are not "more biologically plausible" but a *fundamentally different learning paradigm*.
- **PC's advantage is regime-dependent** — deep and/or small-batch. Backprop remains competitive when interference is minimal.
- **Parameter-isolation methods (XdG, PackNet, HAT) require context ID at test** ⇒ usable in **Task-IL only**. van de Ven could only run XdG and Separate Networks in that scenario.
- **Never put ReLU on the output layer.** Zero gradient below zero means an old-class unit pushed negative is *permanently dead* — converts a recoverable calibration problem into an irreversible one.

---

## 5. Equations, Data & Reference Configurations

### 5.1 Core gradients

```
Softmax + CE:    ∂L/∂z_o = p_o − 1[o=t]        (no fixed point; unbounded competition)
                 ∂L/∂w_o = (p_o − 1[o=t])·h
                 ∂L/∂b_o =  p_o − 1[o=t]

Linear + MSE:    ∂L/∂z_o = z_o − y_o           (fixed point at z_o = 0 ⇒ w_o ⟂ current features)
Sigmoid + BCE:   ∂L/∂z_o = σ(z_o) − y_o        (decoupled forward, still-coupled labels)
```

### 5.2 PC vs backprop vs EqProp

```
MLP forward:        x_l = w_{l−1} f(x_{l−1})
PCN local energy:   E_l = ½ (x_l − w_{l−1} f(x_{l−1}))²

Backprop:           Δw_l = −α ∂L/∂w_l           (chain rule from output; layers assume each other fixed)

PC (strong clamp):  clamp x_1 = s_in AND x_{L+1} = s_target
                    relax:  Δx_l = −γ ∂(E_l + E_{l+1})/∂x_l   until convergence x*
                    then:   Δw   = −α ∂E/∂w |_{x = x*}

EqProp (weak clamp): free-phase equilibrium, then β-nudged phase; Δw ∝ (1/β)(∇_β − ∇_free)
                     → recovers the backprop gradient as β → 0
```

### 5.3 Figure data — read off plots, approximate (±3%)

| Method | Peak task-1 (step 100) | Final task-1 (step 200) | Final task-2 (step 200) | Sum | Position vs diagonal |
|---|---|---|---|---|---|
| backprop | ~85% | ~0% | ~78% | 78 | below |
| replay | ~84% | ~37% | ~75% | 112 | on / slightly above |
| eqprop | ~74% | ~3% | ~66% | 69 | **well below (worst)** |
| pc | ~86% | ~10% | ~82% | 92 | **above for most of path** |

Mid-transition comparison at task-1 = 40% (diagonal would be 60%): pc ≈ 65–70%, replay ≈ 58–60%, backprop ≈ 48%, eqprop ≈ 35%.

Qualitative: backprop = cliff (85→0 in ~35 steps). replay = drop then plateau, huge thin-line spread. eqprop = gradual monotone decay, noisy scattered runs. pc = gradual decay, tightly clustered runs.

### 5.4 Reference configurations (van de Ven et al. 2022, for reproducibility / comparability)

```
Split MNIST base net:     2 × 400 ReLU FC + softmax
                          5 contexts × 2 digits; 2000 iters/context; batch 128; Adam lr 1e-3

Split CIFAR-100 base net: 5 conv layers (16/32/64/128/256 ch, 3×3, pad 1, stride 1 then 2, BatchNorm,
                          no pooling), pretrained on CIFAR-10 and FROZEN
                          + 2 × 2000 ReLU FC + softmax
                          10 contexts × 10 classes; 5000 iters/context; batch 256; Adam lr 1e-4

Output layer by scenario: Task-IL   → multi-head, only current context's units active
                          Domain-IL → single head, 2 units (MNIST) / 10 units (CIFAR-100)
                          Class-IL  → single head, 10 units (MNIST) / 100 units (CIFAR-100), all active
                          (expanding head vs all-active made little difference empirically)

Generative classifier     per-class VAE; enc & dec 2 × 85 ReLU, 5-unit latent
(Split MNIST):            1000 iters/class; classify by argmax_o p(x|o)
                          likelihood via importance sampling, S = 10,000 samples
```

### 5.5 Pseudocode sketches for the next steps

```python
# --- Next step 1: NCM probe (separates pathology 1 from pathology 2) ---
model.eval()
feats = {}                                  # collect penultimate-layer features
for x, y in train_loader_all_tasks:
    h = model.trunk(x)                      # discard the head entirely
    feats.setdefault(y, []).append(h)
prototypes = {c: torch.stack(v).mean(0) for c, v in feats.items()}

def ncm_predict(x):
    h = model.trunk(x)
    return min(prototypes, key=lambda c: (h - prototypes[c]).norm())
# If backprop's task-1 accuracy jumps from ~0% -> substantial, the trunk survived
# and the softmax head was the entire failure.


# --- Replay-free fix A: masked loss (removes suppression, worsens calibration) ---
logits = model(x)
mask = torch.full_like(logits, float('-inf'))
mask[:, classes_present_in_current_batch] = 0
loss = F.cross_entropy(logits + mask, y)


# --- Replay-free fix B: cosine classifier (keeps calibration, removes the
#     magnitude DOF that suppression corrupts) ---
z = scale * F.normalize(h, dim=1) @ F.normalize(W, dim=0)


# --- Target alignment: the interference metric from Song & Bogacz Fig. 3b ---
# cosine between (target − output_before) and (output_after − output_before),
# with output_after measured WITHOUT the target provided.
d_target = target - out_before
d_learn  = out_after_no_target - out_before
alignment = F.cosine_similarity(d_target, d_learn, dim=-1)
```

---

## 6. References

**In-project (PDFs available in the workspace):**

- **Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz (2024).** "Inferring neural activity before plasticity as a foundation for learning beyond backpropagation." *Nature Neuroscience* 27:348–358. → `song_bogacz_24.pdf`
  Key: Fig. 1 (interference example), Fig. 2 (energy machine), Fig. 3b–e (target alignment), **Fig. 4d–e (two alternating 5-class tasks — near-identical protocol to ours)**, Fig. 4f–g (concept drift), Discussion (**the EqProp remark**).
- **van de Ven, Tuytelaars & Tolias (2022).** "Three types of incremental learning." *Nature Machine Intelligence* 4:1185–1197. → `s42256022005683.pdf`
  Key: Table 1 (scenario mappings), Fig. 2 (Split MNIST three ways), eq. (2) (**active-unit softmax**), Tables 2–3 (empirical comparison), Methods (architectures, generative classifier, iCaRL).
- **Dong, Peng & Wu (2025).** Commentary on Song et al. *Intelligent Computing.* → `dong_wu_rev_song_bogacz.pdf`
  Key: EBM lineage, **strong-clamp vs weak-clamp framing**, interference-regime caveat (batch size 1 / depth), biological-plausibility re-examination, PCN non-contrastive normalization argument.
- **Kirkpatrick et al. (2017).** "Overcoming catastrophic forgetting in neural networks." *PNAS* 114:3521–3526. → `kirkpatrick_17.pdf`
  Key: EWC; discussion of Hopfield saturation / blackout catastrophe.

**External (not in project):**

- **Millidge, Song, Salvatori, Lukasiewicz & Bogacz (2022).** "Backpropagation at the infinitesimal inference limit of energy-based models: Unifying predictive coding, equilibrium propagation, and contrastive Hebbian learning." arXiv:2206.02629. → *the formal proof that EqProp → backprop.*
- **Scellier & Bengio (2017).** "Equilibrium propagation." *Front. Comput. Neurosci.* 11:24.
- **Maltoni & Lomonaco (2019).** "Continuous learning in single-incremental-task scenarios." *Neural Networks* 116:56–73. → MT/SIT/MIT.
- **Lomonaco & Maltoni (2017).** "CORe50." *CoRL.* → NI/NC/NIC benchmarks.
- **Moreno-Torres et al. (2012).** "A unifying view on dataset shift in classification." *Pattern Recognition.*
- **McCloskey & Cohen (1989).** "Catastrophic interference in connectionist networks."
- Class-IL bias correction: **Hou et al. (2019)** LUCIR (cosine classifier); **Wu et al. (2019)** BiC; **Zhao et al. (2020)** WA (weight alignment); **Rebuffi et al. (2017)** iCaRL (sigmoid+BCE, NCM inference).
- Task-free CL: **Aljundi et al. (2019)**; **Lee et al. (2020)** Dirichlet process mixture.

---

## 7. Open Questions Carried Forward

1. Can E_sharp and E_smooth be trained under a **single unified objective**, or does the sharp component need a separate fast-weight update rule?
2. What is the right **readout for replay** — the sharpest energy (verbatim, limited) or intermediate noise (recombinant, brain-like)? The literature points to intermediate.
3. What is the **consolidation pathway** from sharp to smooth, and what triggers it in a task-free stream?
4. Does **PC + replay** compose additively, or does PC's reduced interference change which samples are worth replaying?
5. Does the **energy-based replay selection** result (high-energy samples first) hold in a decomposed E_sharp + E_smooth model, or only in a single modern-Hopfield energy?
6. Confirm the evaluation protocol in the current experiments (global argmax over shared head, as the 0%-not-50% signature implies).

---

*Note: figure-derived numbers in §5.3 are read off plots and approximate. Verify against logged metrics before citing.*
