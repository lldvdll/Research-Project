# Energy-Based Models for Continual Learning — Literature Review
*Prepared for the EqProp/continual-learning project. Companion to `research_plan.md` / `research_log.md`.*

---

## Bottom line (read this first)
Two different things get called "energy-based models," and they solve **different** problems:

1. **EBM-as-classifier** (contrastive-divergence objective) — **overcomes catastrophic forgetting** in Class-IL, proven on Split MNIST. Trained with ordinary backprop. **Not** biologically plausible in its learning rule.
2. **Energy-based networks with local learning** (Equilibrium Propagation, Predictive Coding, Hopfield) — **biologically plausible and hardware-friendly**, but **not shown to overcome Class-IL forgetting on their own**. This is consistent with your EqProp result: it forgets in Class-IL.

**Recommendation:** the model that satisfies "overcomes forgetting + is generative for replay + has working code on your exact benchmark" is the **contrastive EBM classifier (Li, Du, van de Ven, Mordatch, CoLLAs 2022)**. Your EqProp work belongs to the *biological-plausibility / neuromorphic-hardware* story — a strong discussion thread, but a different question. The two can be united through **generative replay** (the EBM generates its own past examples), which is both a proven CL method and the hippocampal analogue.

---

## Q1 — Which EBMs are appropriate to our case?
Our case: MLP, small images (MNIST 14×14), Class-IL, ideally generative (for synthetic replay).

- **Conditional EBM classifier — Li, Du, van de Ven, Mordatch (CoLLAs 2022, arXiv 2011.12216).** Defines an energy `E(x,y)`; classifies by `argmin_y E(x,y)`; trained with a **contrastive-divergence loss** that pushes down the true label's energy and up a *sampled negative label's* energy, instead of a softmax normalised over all classes. Because it never normalises over all classes, learning a new class does **not** suppress old ones. Demonstrated on Split MNIST, permuted MNIST, CIFAR-10/100. **Public PyTorch code** (github.com/ShuangLI59/ebm-continual-learning). → **Most appropriate.**
- **Latent-space EBM — LSEBMCL (Li et al., ICAIIC 2025, arXiv 2501.05495).** Uses an EBM as an *outer generator* supporting both classification and generation. More machinery; aimed at text/trajectory. Relevant to the generative angle, not the first choice.
- **Boltzmann / Hopfield family** (RBMs; modern/dense Hopfield, Krotov & Hopfield). Classic *generative* EBMs and associative memories. Conceptually relevant (memory capacity ↔ forgetting) but not the mainstream CL solution.
- **Energy-based networks with local rules** (EqProp, Predictive Coding). Energy-based and biologically plausible, but *as classifiers* they still couple classes through the shared output — so they don't inherently solve Class-IL (your finding).

## Q2 — Which are biologically plausible, where, and why?
- **Predictive coding (PC).** Strongest cortical grounding: the cortex is proposed to continually predict sensory input, with neurons signalling *prediction error* (Rao & Ballard 1999; Friston, free-energy principle). Candidate implementations map onto canonical **cortical microcircuitry** (Bastos et al. 2012; Shipp 2016) and reproduce phenomena like end-stopping and binocular rivalry. In representational-similarity comparisons, PC matches human brain representations *better than backprop*. **Locus: neocortex** (hierarchical sensory areas).
- **Equilibrium Propagation (EqProp).** Biologically attractive because weight updates are **fully local** — each synapse updates from the activity of the two neurons it connects — and the *same* network performs inference and learning (no separate backward pass). Applies to energy-based / Hopfield networks; related to contrastive Hebbian learning. Its main *implausible* feature is weight symmetry (forward = feedback). **Locus: recurrent cortical circuits with feedback.**
- **Hopfield / attractor networks.** Energy-minimising attractor dynamics as associative memory; modern (dense) Hopfield nets have very large capacity. **Locus: hippocampal CA3 recurrent collaterals.**
- **Replay.** Hippocampal replay of activity during rest/sleep consolidates memories to cortex (complementary learning systems; McClelland et al. 1995). Generative-replay models treat the **hippocampus as a generative network** and replay as a generative process (van de Ven et al. 2020). **Locus: hippocampus → neocortex.**

## Q3 — Limitations in silicon; what hardware makes them efficient?
The core limitation is **exactly what you've been fighting**: energy-based networks require **iterative relaxation to equilibrium (settling)**, which is slow and costly to *simulate* on digital von-Neumann hardware. EBM *generation* (Langevin/MCMC sampling) is likewise iterative. On a CPU/GPU each inference is hundreds of sequential settling steps — hence your multi-minute runs.

Efficient hardware lets **physics do the settling for free**:
- **Analog / neuromorphic circuits** (Kendall et al. 2020, arXiv 2006.01981): EqProp is a training framework for end-to-end analog nets; the circuit relaxes continuously and the local update needs no backward pass.
- **Memristor crossbar arrays** (Frontiers 2020; Micromachines 2023; arXiv 2512.12428): weights live in-memory; EqProp fits because backprop's separate forward/backward passes are hard on crossbars while EqProp reuses one circuit.
- **Spiking neuromorphic** — EqSpike (Martin et al. 2021, arXiv 2010.07859): local in space *and* time; no external memory for activations/gradients.
- **Oscillator Ising Machines** (arXiv 2505.02103): continuous-phase oscillators perform energy descent at GHz; EqProp nudges the steady state. Potentially orders-of-magnitude more efficient; current blockers are relaxation time and initialisation.

**Through-line:** EBMs/EqProp are a *poor* fit for digital hardware (slow settling) but a *natural* fit for analog/neuromorphic hardware, where relaxation is physical rather than a simulated loop. This is the honest reason your CPU runs are slow — and the reason the field pursues this hardware at all.

## Q4 — Which have been *demonstrated* to overcome catastrophic forgetting?
- **Contrastive EBM classifier (Li/Du/van de Ven 2022): YES.** Outperforms softmax baselines and standard CL methods by a large margin in Class-IL, because the contrastive objective doesn't suppress old classes. The softmax classifier predicts only current-task classes; the EBM predicts across all seen classes. **Main positive result.**
- **Generative replay (Shin et al. 2017; van de Ven et al. 2020, "Brain-Inspired Replay", Nature Communications 11:4069): YES.** For Class-IL, generative replay is essentially the *only* family that performs well **without storing data**. An EBM *is* a generative model → it can be the generator (your synthetic-replay idea). Caveat: generative replay degrades on complex natural images.
- **EqProp / PC as classifiers alone: NOT demonstrated** to overcome Class-IL forgetting. Consistent with your result — the local rule reduces interference in the *weights*, but Class-IL forgetting is driven by *output competition*, which the rule doesn't remove.

## Q5 — On which IL tasks, and an MNIST 14×14 implementation
**The three scenarios** (van de Ven, Tuytelaars & Tolias 2022, *Nature Machine Intelligence* 4:1185):
- **Task-IL** — task identity given at test; choose only among that task's classes. Easiest; per-task heads barely forget even for backprop (you saw this).
- **Domain-IL** — identity *not* given; same label set but the input distribution shifts (e.g. odd/even under permutations/rotations). Structure fixed, context changes.
- **Class-IL** — identity *not* given; choose among **all** classes seen so far. Hardest; where softmax and EqProp collapse and where the EBM result matters.

The contrastive EBM was demonstrated on **Class-IL** on **Split MNIST (5×2)**, permuted MNIST (10 tasks), CIFAR-10/100 — i.e. your exact setting is where it's proven.

**MNIST 14×14 implementation (Class-IL, 10×1 or 5×2):**
1. Keep the shared trunk (196→64). Replace the softmax output with a **conditional energy** `E(x,y) = -score(f(x), y)` — e.g. feed a one-hot label into a small head that scores compatibility with features `f(x)`; classify by the `y` with lowest energy.
2. Train with the **contrastive-divergence loss**: for each `(x, y_true)`, sample a negative `y⁻ ≠ y_true` from the classes present in the current batch; push `E(x, y_true)` down and `E(x, y⁻)` up. No softmax over all classes → no suppression of old/unseen classes.
3. Evaluate as now: after each task, per-class accuracy from `argmin_y E(x,y)` over all seen classes.
4. Baselines already built: backprop (forgets), replay (fixes). Add the contrastive EBM as the third line — expect it well above backprop in Class-IL.
5. **Generative/replay extension:** the same `E(x,y)` can generate `x` for a class via short **Langevin sampling** (Du & Mordatch 2019, arXiv 1903.08689) → EBM-generated replay = your "generate synthetic examples from the EBM" loop, and the hippocampal-replay analogue.
6. Reference code: `github.com/ShuangLI59/ebm-continual-learning`, adaptable to 14×14.

---

## References
- Li, Du, van de Ven, Mordatch (2022). *Energy-Based Models for Continual Learning.* CoLLAs, PMLR 199. arXiv 2011.12216. Code: github.com/ShuangLI59/ebm-continual-learning.
- van de Ven, Tuytelaars, Tolias (2022). *Three types of incremental learning.* Nature Machine Intelligence 4:1185.
- van de Ven, Siegelmann, Tolias (2020). *Brain-inspired replay for continual learning with artificial neural networks.* Nature Communications 11:4069.
- Shin, Lee, Kim, Kim (2017). *Continual Learning with Deep Generative Replay.* NeurIPS.
- Du & Mordatch (2019). *Implicit Generation and Generalization in Energy-Based Models.* arXiv 1903.08689.
- Scellier & Bengio (2017). *Equilibrium Propagation.* Frontiers in Computational Neuroscience.
- Laborieux et al. (2021). *Scaling Equilibrium Propagation to Deep ConvNets…* Frontiers in Neuroscience (PMC7930909).
- Kendall et al. (2020). *Training End-to-End Analog Neural Networks with Equilibrium Propagation.* arXiv 2006.01981.
- Martin et al. (2021). *EqSpike: Spike-driven Equilibrium Propagation.* arXiv 2010.07859.
- Oscillator Ising Machines + EP (2025). arXiv 2505.02103.
- Rao & Ballard (1999); Bastos et al. (2012); Shipp (2016) — predictive coding & cortical microcircuitry.
- Krotov & Hopfield (2016) — dense associative memory / modern Hopfield.
- Li et al. (2025). *LSEBMCL: A Latent Space Energy-Based Model for Continual Learning.* ICAIIC. arXiv 2501.05495.
