# Interpretability chapter — plan

Tracked document, checked off as we go. Every section follows the same 7 steps:
1. Research question(s) 2. Tools/config 3. Presentation discussion + plot plan 4. Run
5. Assess results 6. Refine presentation 7. Consolidate into report.

Findings stay as bullet points here for now — paragraph writeup comes later.

## Numbering

No existing file is renamed — 100-131 and 200-230 are already written/running and renaming risks
live background jobs for no real benefit. Going forward, interpretability work is organised as:

- **130s — where forgetting lives.** 130 freeze factorial, 131 NCM, 132 depth (new).
- **140s — how/why, and does the literature's own mechanism claim hold up.** 140 path
  efficiency, 141 target alignment, 142 data/pairing structure (all new, rerunning pre-100-series
  questions under this series' rigor).
- **200s — bio-inspired extensions** (unchanged, already established: 200 EWC, 210 SI, 220
  k-WTA, 230 headline comparison).
- A possible **RQ7** (below) connects the 130s/140s findings to the 200s results, but only if
  the 200s sweeps show something worth explaining mechanistically — conditional, not scheduled.

---

## RQ1 (130) — Where does forgetting live: trunk (W1) or output layer (W2)?

### 1. Research question
Does freezing the trunk or the output layer during task 2 recover task-1 retention, and does
that differ by rule (backprop/PC) or scenario (Class-IL/Domain-IL)?

### 2. Tools / config
Freeze-W1/W2/both factorial via `handle["freeze"]`, backprop + PC, both scenarios,
matched-competence 90%, 10 seeds, crossover metric.

- [x] Research question defined
- [x] Tools/config decided
- [x] Presentation discussed (bar chart, x=condition, grouped by rule, one panel per scenario;
      3-condition and +ceiling variants)
- [x] Experiment run
- [x] Results assessed: freeze_w2 does ~nothing in either scenario (real finding: freezing ≠
      masking); **freeze_w1 not trustworthy as reported — 4/4 cells had 9-10/10 seeds hit the
      training cap, never reaching matched competence. Still open.**
- [ ] Presentation refined
- [ ] Consolidated into report

**Open item carried forward: fix freeze_w1's non-convergence** (raise the cap, or accept a
longer budget) before this RQ's numbers are citable.

---

## RQ2 (131) — Does the hidden code survive even when the readout doesn't?

### 1. Research question
In the unfrozen (control) condition, does NCM (frozen prototypes) hold information argmax has
already lost?

### 2. Tools / config
`src/probes.py` frozen NCM vs argmax, control condition only, backprop + PC, both scenarios,
matched-competence 90%, 10 seeds.

- [x] Research question defined
- [x] Tools/config decided
- [x] Presentation discussed (2x2 grid, rows=scenario, cols=rule, argmax vs NCM over time)
- [x] Experiment run
- [x] Results assessed: Class-IL argmax 0.0%/0.0% vs NCM 86.3%/77.6% (bp/pc) — head is the
      whole problem. Domain-IL argmax 38.1%/39.1% vs NCM 73.5%/71.3% — smaller but real gap
      even without suppression.
- [ ] Presentation refined
- [ ] Consolidated into report

---

## RQ3 (132, new) — Does depth change the picture?

### 1. Research question
Is one hidden layer adequate, or does adding depth change (a) whether PC beats backprop, or
(b) where forgetting lives? Resolves the standing "is depth a control we can stop worrying
about" question.

Scoped in two tiers so the cheap check happens first:
- **Tier A (does this even matter?):** does the plain backprop-vs-PC crossover comparison
  change across a small depth grid, both scenarios? Cheap, answers the actual blocker.
- **Tier B (only if Tier A finds something):** extend RQ1's freeze/NCM factorial across the
  same depth grid — does *where* forgetting lives move with depth?

### 2. Tools / config
`n_layers`/`hidden` as a tuple (already supported, no `src/` change needed). Depth grid
`{1, 2, 3}` hidden layers, width held at 32 per layer unless that fails to train, backprop + PC,
both scenarios, matched-competence 90%, 10 seeds.

- [ ] Research question defined — draft above, confirm before scripting
- [ ] Tools/config decided — confirm depth grid and per-layer width
- [ ] Presentation discussed
- [ ] Experiment run
- [ ] Results assessed
- [ ] Presentation refined
- [ ] Consolidated into report

---

## RQ4 (140, new) — How efficiently does each rule move its weights, and does it track retention?

### 1. Research question
Is PC's per-weight update path straighter (less wasted motion) than backprop's, at which layer,
and does that efficiency actually correlate with retention or is it a red herring? (Pre-100-series
work found PC's output-layer efficiency edge was *largest exactly where it bought nothing* —
worth knowing before re-deriving this, not as a substitute for doing so.)

### 2. Tools / config
Path efficiency = accumulated `sum|dw|` over training vs net displacement `|w(T)-w(0)|`, per
weight (`src/probes.py: weight_path_probe`, already built). Backprop + PC, both scenarios,
matched-competence 90%, 10 seeds.

- [ ] Research question defined — draft above, confirm before scripting
- [ ] Tools/config decided
- [ ] Presentation discussed
- [ ] Experiment run
- [ ] Results assessed
- [ ] Presentation refined
- [ ] Consolidated into report

---

## RQ5 (141, new) — Does target alignment explain anything?

### 1. Research question
Song & Bogacz's own credited mechanism: is PC's update more aligned with the target direction
than backprop's, and does that alignment track retention? This is a direct test of the thesis's
central claim, not a side curiosity — highest-priority of the "why" questions.
(Pre-100-series work found PC was *not* more aligned than backprop, and that the metric
*anti-tracked* retention entirely — a strong prior to be aware of, not to skip re-deriving.)

### 2. Tools / config
`src/probes.py: alignment_probe` — `cos(target - out_before, out_after_no_target - out_before)`,
measured in output space. Backprop + PC, both scenarios, during task 2, matched-competence 90%,
10 seeds.

- [ ] Research question defined — draft above, confirm before scripting
- [ ] Tools/config decided
- [ ] Presentation discussed
- [ ] Experiment run
- [ ] Results assessed
- [ ] Presentation refined
- [ ] Consolidated into report

---

## RQ6 (142, new) — Does class pairing confound the Domain-IL story?

### 1. Research question
Domain-IL is this project's primary scenario. Pre-100-series work found which two digits share
an output unit explained ~62% of seed variance in retention — if that still holds, it bears on
how confidently *any* Domain-IL finding in this chapter (RQ1-RQ5) can be stated, so there's a
case for running this early rather than last despite the numbering.

### 2. Tools / config
Correlate retention (crossover) against a pairing-similarity measure across enough seeds to
resolve it (pre-100-series work used 24 for this specific question — plain 10 may be too few;
flagging now rather than discovering it after a run). Domain-IL only — the question is
structurally about Domain-IL's shared output units.

- [ ] Research question defined — draft above, confirm before scripting
- [ ] Tools/config decided — seed count needs its own decision, see above
- [ ] Presentation discussed
- [ ] Experiment run
- [ ] Results assessed
- [ ] Presentation refined
- [ ] Consolidated into report

---

## RQ7 (conditional) — Do the bio-inspired mechanisms act where RQ1/RQ2 say forgetting lives?

Only scheduled if 200/210/220 (EWC/SI/k-WTA sweeps) show a real effect worth explaining
mechanistically. E.g.: does EWC/SI's importance concentrate on the output layer (matching RQ1's
Class-IL finding) or the trunk? Does k-WTA change the NCM picture?

- [ ] Not yet scheduled — gate on 200-series results
