# Consolidation — reference for the conversation, not a plan

This is a record of the reasoning from the last several exchanges: what was actually argued,
what got corrected, what's still an open thread. Not a schedule, not experiments to run, not
sections to write. Read it back before designing whatever comes next, to avoid re-deriving the
same ground.

## The ideas, as they actually developed

**1. Intervention vs diagnostic — freeze and NCM are not the same kind of tool.**
Freeze changes training and watches the accuracy consequence (causal). NCM changes nothing and
just asks a different question of the network exactly as it ended up (observational). They can
disagree usefully: NCM says the information is still there; freezing W2 says protecting the
readout by *freezing it entirely* doesn't recover it anyway. Together they say the fix isn't
"stop the readout from changing," it's "let it change, just not destructively." This is a
reusable distinction for choosing tools later, not specific to these two experiments.

**2. Freeze-W2 was never actually a clean test of anything.**
If output-layer suppression really drives Class-IL forgetting, freezing W2 should have helped —
it didn't, in either scenario. Not because suppression isn't real, but because freezing blocks
the *helpful* part of output-layer learning along with the harmful part, and the two cancel.
Masking is the properly-targeted version (blocks only the harmful negative-gradient part) and
it's why masking worked where freezing didn't. Corollary: freeze-W2 is really a Class-IL-motivated
question — in Domain-IL there's no suppression mechanism to protect in the first place, so it's
closer to a trivial control than a real test.

**3. Crossover and endpoint accuracy answer different questions — the earlier metric split
wasn't principled when it was made, but a principle exists in hindsight.**
Crossover = is learning outracing forgetting (a rate). Endpoint = how much survives once both
tasks are adequately learned (a total). 130 used crossover, 131 used endpoint — at the time, that
was because crossover isn't defined for NCM without also building task-2 prototypes (never
built, an effort gap not a decision). Retrospectively, the split holds up for a real reason:
crossover fits *interventions* that change the race itself (freeze, mask, EWC, SI); endpoint
fits a *diagnostic* on an already-fixed final state (NCM). Separately: which one actually matters
depends on the training regime — this project always trains to matched competence, which makes
endpoint well-defined and arguably the thing that matters; crossover matters more as a
robustness check, or in a regime that never reaches a clean endpoint at all.

**4. Calibration vs representation is not a clean binary once depth is added.**
Calibration, strictly, is a property of whatever produces the final scores — structurally the
*last* weight matrix, at any depth. But whether "representation" is uniform across every earlier
layer, or graded, is unresolved. Pre-100 (not re-verified) divergence data hints at a gradient:
cosine similarity between backprop's and PC's updates falls from ~0.95 near the input to ~0.62
near the output at depth 3 — early layers looking generic/shared, later ones more task-specific.
Generalizing the calibration/representation split to a deep network would need an NCM probe at
*every* layer, not one at the end.

**5. The readout-stage hypothesis (the main synthesis so far).**
Chain: S&B claim PC is more target-aligned → that should mean a more efficient update path → 
taken to its natural conclusion, that would predict PC arranges the *trunk* so well that future
learning is mostly an output-layer change. This does **not** survive the existing (pre-100,
not-yet-reverified) signals: PC's and backprop's trunk updates look nearly identical (cosine
~0.95 near the input) — PC isn't organizing the trunk differently; PC's output-layer path-
efficiency edge was *largest* in Domain-IL, where there's no retention advantage, and smaller in
Class-IL, where there is one — backwards from what the story needs; target alignment itself was
already found near-identical between the two rules and anti-tracking retention entirely.
Refined version: **if PC has an edge at all, it's a readout-stage phenomenon, not a trunk-
reorganization phenomenon.** This is the one thread that ties everything else together — it
connects directly back to idea 1/2 (the damage lives at the readout) and reframes what a fresh
path-efficiency or target-alignment rerun should actually be asking (specifically about the
readout layer, not pooled across the whole network). Left unfinished — see interconnections below.

**6. New idea: does representation *organization*, not just adequacy, matter for cheap
recalibration?**
Train jointly to the ceiling, then continue training on task 1 or task 2 alone, and compare how
much the other degrades against the ordinary sequential-from-scratch case. Not yet run, not yet
designed in detail — just posed.

**7. Repeated-alternation question, corrected.**
Originally mis-stated (conflated with freezing, which trivially can't shift). The actual
question: for the *unfrozen*, normally-trained network, does the trunk's movement per task-switch
shrink over many rounds of task1↔task2 alternation, as if it settles into a shared representation
on its own? Connects to path efficiency and to an old (pre-100) repeated-switching experiment
that found convergence with no rule separating.

**8. What "forgetting" even means in Domain-IL.**
Distinguished from concept drift: concept drift is when the true mapping changes and updating is
*correct*; here the old mapping never stops being true, it's just not reinforced during task 2.
The joint-training ceiling (real evidence: both mappings ARE simultaneously learnable) is what
makes the loss "catastrophic" rather than a reasonable overwrite — it's a training-order
artifact, not something the network was ever forced into. No rigorous way yet to formalize
"acceptable" vs "catastrophic" forgetting beyond that ceiling comparison.

**9. Masking, independently re-derived from a different angle.**
The idea "freeze the readout's calibration for classes we don't currently see" is exactly what
masking already does. Worth flagging honestly: masking needs oracle knowledge of which classes
are "old" at training time — it isn't something plain gradient descent produces on its own — and
no claim has been established connecting it to biological gradual decay. Don't reach for that
parallel without more care.

**10. A depth-dependent metaplasticity idea can't be expressed without actual depth.**
"Bias plasticity to favour deeper layers" has nothing to be deeper *than* with one hidden layer —
freeze already tests the only two positions that exist (trunk, output). This is a concrete reason
depth work has to happen before this specific idea is even testable, separate from the general
"does depth change the picture" question.

**11. EWC and SI, actual new evidence (not historical).**
Both flat / no effect within the range that trains reliably (EWC up to λ=10; SI up to λ=0.1). SI
above that mostly returns **no defined crossover at all** (8-10/10 seeds `nan`, corrected from an
earlier overstatement that it "collapsed to chance"). SI's importance accumulates over the whole
of task 1, so its scale isn't comparable to EWC's one-shot Fisher snapshot at the same nominal λ
— the two sweeps aren't probing equivalent penalty strengths. Neither currently shows a positive
effect worth explaining mechanistically.

**12. Code review finding.**
`backprop_ewc`/`backprop_si`'s penalty doesn't check the freeze set (order: freeze zeros grad,
then the penalty adds back onto it, unconditionally) — freeze and EWC/SI would silently conflict
if ever combined. `pc_ewc`/`pc_si` don't have this problem. Nothing run so far is affected.

## How these connect

- **1, 2, 5, and 9 all point at the same place**: the readout is where Class-IL's damage lives
  and where its fix has to be surgical, not blunt. This is the most load-bearing conclusion so far.
- **3 is a lens, not a conclusion** — applies to whatever gets designed next: decide up front
  whether it's an intervention (crossover) or a diagnostic (endpoint), rather than choosing by
  what's easy to compute, which is what actually happened between 130 and 131.
- **4, 10, and 6 all converge on needing depth**, but for three different reasons bundled under
  one label: whether the calibration/representation split is graded, whether a depth-biased
  plasticity idea is even expressible, and whether representation organization (not just
  adequacy) matters. "Does depth matter" is at least three separate questions wearing one name.
- **8 reframes what "success" would even mean** for idea 6 specifically in Domain-IL — a smaller
  gap between joint-then-refine and sequential-from-scratch wouldn't necessarily mean the
  representation is badly organized; it might mean the ceiling itself leaves little room to show
  a difference.
- **11 and 5 are not yet connected**, and that's itself informative: EWC/SI would only bridge
  into the readout-stage hypothesis if their importance turned out to concentrate at the output
  layer — but neither has shown an effect yet worth explaining that way. k-WTA (still running) is
  the one remaining chance for evidence 11 to connect back into the conceptual thread rather than
  sit alongside it.
