# Literature review — do energy-based learning rules mitigate catastrophic forgetting?

Conducted 2026-08-17, independent of this project's own results. Primary sources read directly
(`ref/song_bogacz_24.pdf`, its independent commentary) where possible; everything else is
contemporary web search/fetch, not recalled from training data.

## 1. The neuroscience premise: why doesn't the brain forget catastrophically?

Not one mechanism — three, and they're not mutually exclusive:

| Theory | Claim | Where |
|---|---|---|
| **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly 1995; updated by Kumaran et al. 2016) | Hippocampus = fast, sparse, pattern-separated storage for new episodes. Neocortex = slow, overlapping, statistically-structured storage. Sleep replay interleaves hippocampal traces into cortex slowly enough to avoid interference. Forgetting is avoided by **separating fast and slow learners**, not by a better single learning rule. | Theoretical ancestor of *replay* as a continual-learning method — i.e. of your positive control. |
| **Synaptic consolidation / cascade models** (Fusi, Drew & Abbott 2005) | Each synapse has internal metaplastic states of varying plasticity. Recently-changed synapses are volatile; synapses that have survived many updates become progressively harder to change. Produces power-law forgetting instead of catastrophic collapse. Direct inspiration for EWC (Kirkpatrick et al. 2017, in `ref/`). | A per-synapse **importance/protection** mechanism, orthogonal to the credit-assignment rule. |
| **Predictive coding as a cortical theory** (Rao & Ballard 1999; Friston's free-energy principle) | The cortex is a hierarchical generative model minimizing prediction error. A claim about *representation and inference*, not originally about *catastrophic forgetting* at all. | The forgetting-mitigation claim is a **recent and separate addition** (Song & Bogacz 2024) grafted onto an older representational theory. |

The field's actual account of brain-level robustness is CLS + consolidation — architectural/systems-level
solutions. Predictive coding's link to forgetting specifically is one paper, not a consensus
neuroscience position.

## 2. What Song & Bogacz (2024) actually claim — read from the source PDF

Their forgetting-specific result is **Fig. 4d–e**, one of five "biologically relevant scenarios"
(online learning, continual learning, concept drift, small-data, RL) — continual learning is
one-fifth of their evidence, not the paper's center of mass (that's target alignment, Fig. 3).

**Exact setup, from their Methods:**
- **5 shared output units** (their words: *"the whole network was shared by the two tasks, but the
  network only had five output neurons"*) — this is exactly this project's Domain-IL, confirming
  the 2026-08-11 scenario decision.
- **3 hidden layers × 32 units** ("four layers" = 4 weight matrices). This project uses **1 hidden
  layer**.
- **Sigmoid activation**, MSE loss, batch 32, Xavier normal init.
- Alternation in bursts of **4 iterations per task** (4 batches × 32 = 128 examples), repeated to
  84 total iterations (~21 switches).
- n = 10 seeds.
- Concept drift (Fig. 4f–g): 10 output units, a *random 5-of-10 relabeling* every 64 epochs, 3000
  epochs total (~47 reshuffles) — not a two-task split at all.

**The mechanism they credit**, read off Fig. 3: **target alignment** — cos(direction from current
output to target, direction of the learning step). Fig. 3f–h show this gap widening with **depth**
on a *single-task* test-error curve (not a forgetting metric). The continual-learning result
(Fig. 4d–e) is presented as a downstream consequence of this same alignment property, not
independently mechanistically explained there.

## 3. Independent literature — does the field confirm this, beyond the one paper?

No paper directly reruns Fig. 4d–e. What exists is adjacent work bearing on the *general* claim
"PC/EqProp alone reduce forgetting by nature of the mechanism":

- **Preventing Deterioration of Classification Accuracy in Predictive Coding Networks**
  (arXiv 2208.07114) — states directly that **vanilla PC does suffer catastrophic forgetting** in
  class-incremental settings, and requires *added* importance-weighted regularization (an
  EWC-style bolt-on) to control it.
- **Sequential Neural Coding Network** (Ororbia et al., NeurIPS 2022) and **BayesPCN** (2022) —
  both are PC-*family* architectures built specifically to fight forgetting, using **added**
  mechanisms (task-dependent sparsity; Bayesian weight uncertainty) on top of standard PC. Their
  existence is evidence the field does not consider vanilla PC sufficient.
- **Toward Lifelong Learning in Equilibrium Propagation** (arXiv 2508.14081, 2025) — tests EP on
  sequential MNIST: EP-trained recurrent networks collapse to ~25% accuracy from ~97% under naive
  sequential training, **just as badly as BPTT** (~21%). Their words: *"networks trained with EP
  remain vulnerable to catastrophic forgetting"* and *"lack mechanisms akin to memory consolidation
  during sleep."* Fixed with an added Sleep Replay Consolidation mechanism (baseline 47% →
  61% overall accuracy in the related SRC line of work). Independent, different architecture, same
  conclusion as this project's C4: **EqProp confers no inherent forgetting resistance.**
- **"Energy-Based Models for Continual Learning"** (arXiv 2011.12216) — a false friend: this is
  generative/contrastive-divergence EBMs (Hopfield-adjacent) used as an *auxiliary training
  objective bolted onto existing replay/regularization methods*, not PC or EqProp as alternative
  credit-assignment rules. "Energy-based" is an umbrella term covering mechanically unrelated
  proposals.
- **Dong, Peng & Wu (2025), independent commentary on Song & Bogacz in *Intelligent Computing***
  — does not dispute the empirical result, but narrows the claimed scope precisely: *"PCNs under PC
  outperform BP when interference is prominent — such as in deep or online learning settings —
  while BP remains competitive when interference is minimal (e.g., shallow or large-batch
  training)."* Also flags that the PC/BP *equivalence* proof (used to justify comparing them as
  the same architecture, different rule) holds only under strict initialization/scheduling
  constraints — not general.
- **Sleep-like unsupervised replay reduces catastrophic forgetting** (Tadros et al., *Nature
  Communications* 2022) — architecture-agnostic consolidation-via-replay result, not tied to any
  particular credit-assignment rule. Converges with the CLS/consolidation framing in §1: the fix is
  systems-level, not rule-level.

## 4. Under what conditions is the premise true or false, per the literature itself

| Factor | Literature's own account |
|---|---|
| **Depth** | Claimed advantage is explicitly depth-dependent and grows with depth (Fig. 3h) — but only measured on raw test error, not on a forgetting metric, and confounded with BP's own depth-dependent training difficulty with sigmoid. |
| **Batch/burst size** | Claimed advantage is largest at very small batches / short bursts ("interference prominent... online learning"). S&B's own continual-learning experiment used **4-iteration bursts**. |
| **Architecture depth** in the forgetting-specific result | **3 hidden layers**, not a shallow toy net. |
| **Whether the rule is used alone** | Every piece of contrary/independent evidence found (PC deterioration paper, sequential coding network, BayesPCN, EP lifelong-learning paper) shows the *rule alone* — no added regularization, sparsity, or replay — does **not** solve forgetting. Every successful demonstration adds a second, memory-specific mechanism. |
| **Task/output structure** | Their result is specifically the shared-output-unit case (Domain-IL). No published result found tests the separate-output-unit case (Class-IL) against their claim at all. |

## Bottom line

"EBMs are more brain-like, therefore less prone to forgetting" conflates two literatures the field
keeps separate: neuroscience's actual account of brain robustness (CLS + synaptic consolidation —
systems-level, not a credit-assignment rule) and a single 2024 paper's claim that one specific
credit-assignment rule (prospective configuration/PC), tested in one narrow regime (small bursts,
3-hidden-layer sigmoid net, 5 shared outputs), reduces destructive interference via target
alignment. Independent evidence on the EqProp side directly contradicts extending that claim to
EqProp. Independent evidence on the PC side (from groups building on PC, not attacking it) treats
vanilla PC as forgetting-prone by default and adds mechanisms to fix it.

**Two concrete methodological gaps between this project's protocol and theirs:**
1. **Depth.** They test forgetting-adjacent claims only in a 3-hidden-layer net.
2. **Burst length.** Their alternation is 4 iterations/task; this project's closest analogue (`68`)
   uses 150/block — a ~37× difference.

## Sources

- Song, Millidge, Salvatori, Lukasiewicz, Xu & Bogacz (2024), *Nature Neuroscience* — read directly
  from `ref/song_bogacz_24.pdf`.
- [Dong, Peng & Wu (2025), Commentary, *Intelligent Computing*](https://spj.science.org/doi/10.34133/icomputing.0244) — read directly from `ref/dong_wu_rev_song_bogacz.pdf`.
- [Preventing Deterioration of Classification Accuracy in Predictive Coding Networks](https://arxiv.org/pdf/2208.07114)
- [Toward Lifelong Learning in Equilibrium Propagation](https://arxiv.org/html/2508.14081)
- [Lifelong Neural Predictive Coding (Ororbia & Mali)](https://arxiv.org/pdf/1905.10696)
- [BayesPCN](https://arxiv.org/pdf/2205.09930)
- [Energy-Based Models for Continual Learning](https://arxiv.org/pdf/2011.12216)
- [Sleep-like unsupervised replay reduces catastrophic forgetting, Tadros et al., Nature Communications 2022](https://www.nature.com/articles/s41467-022-34938-7)
- Complementary Learning Systems, McClelland, McNaughton & O'Reilly 1995 — background via search.
- Cascade Models of Synaptically Stored Memories, Fusi, Drew & Abbott 2005 — background via search.
