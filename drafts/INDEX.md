# Drafts — published HTML artifacts, sequentially versioned

Every HTML page published to claude.ai as an artifact, kept here so the versions survive the
conversation that made them. **Numbering is a flat sequence across all documents**, not per
document — a new version of an existing page takes the next number, and the "Supersedes" column
says what it replaced.

## Convention

- `NNN_name.html` — the rendered page, images embedded as base64 data URIs, self-contained and
  openable offline in any browser.
- `NNN_name.build.py` — the generator that produced it, where one exists. **The generator is the
  source of truth**: it re-reads the PNGs from `experiments/` at build time, so re-running it
  after a figure changes produces an updated page. Editing the HTML directly is a dead end.
- Republishing to the same artifact URL keeps that URL. A version bump here does not mint a new
  link unless one was deliberately created.

⚠ **These files are large** — the embedded PNGs make each full page 3–5 MB. Only keep a rendered
HTML for a version that was actually reviewed. For intermediate iterations, keep the generator
alone; it is ~50 KB and reproduces the page exactly.

## Log

| # | File | Date | Artifact | Supersedes | What it is |
|---|---|---|---|---|---|
| 001 | `001_experiment_inventory.html` | 2026-09-04 | [4624f01c](https://claude.ai/code/artifact/4624f01c-f703-4c8b-9bcd-73d3a7bba500) | — | Experiment provenance board, grouped by report section. Tallies 28 use-as-is / 2 re-run / 5 not-run / 3 dropped. No figures — text and status only. |
| 002 | `002_figure_flow.html` | 2026-09-04 | [e1bf1195](https://claude.ai/code/artifact/e1bf1195-1094-4ec1-8132-d85f9f128a25) | — | 19 figures embedded in report order across six sections, status-coded. Built to see how the existing figures fit together. Predates the track. |
| 003 | *not retained* | 2026-09-06 | [10978e55](https://claude.ai/code/artifact/10978e55-0449-4f55-beb7-b6464030ac07) | — | First report-track audit: **11 tiers**, 40 figure slots, `minmax` figure grid. Overwritten in place by 004 before it was copied here — recorded so the sequence does not silently skip. |
| 004 | `004_report_track_v2.html` | 2026-09-06 | [10978e55](https://claude.ai/code/artifact/10978e55-0449-4f55-beb7-b6464030ac07) | 003 | Report-track audit, **9 sections**, 39 slots, 11 to build. Column grid: story runs left→right, stacked cards are direct comparisons. |
| 005 | `005_report_track_v3.html` | 2026-09-06 | [10978e55](https://claude.ai/code/artifact/10978e55-0449-4f55-beb7-b6464030ac07) | 004 | **10 sections**, 43 slots, 13 to build, each build tagged P1/P2/P3. Adds §6 on prospective configuration, a band mapping the audit onto six Results chapters, and the partial-freeze diagnostic. |
| 006 | `006_report_track_v4.html` | 2026-09-06 | [10978e55](https://claude.ai/code/artifact/10978e55-0449-4f55-beb7-b6464030ac07) | 005 | **9 sections**, 44 slots, 15 to build. Target alignment restored, weight-space PCA and joint-pretraining added, §2 reordered generic→specific with the metric as its climax, settling cut from four figures to two. |
| 007 | `007_report_track_v5.html` | 2026-09-06 | [10978e55](https://claude.ai/code/artifact/10978e55-0449-4f55-beb7-b6464030ac07) | 006 | Restructured as **Methods / Results / Discussion** and costed against an 8,000-word budget. 15 main-text figures. Setup becomes a parameter table; the two scenario sections merge into one organised by tool; added-mechanism sweeps demoted to Discussion. |
| 008 | `008_report_track_v6.html` | 2026-09-06 | [10978e55](https://claude.ai/code/artifact/10978e55-0449-4f55-beb7-b6464030ac07) | 007 | **The evidence version.** Same nine sections; the built 900-series figures replace the sketches and every verdict is re-derived from the runs. 31 cards, 6 still to build, **0 sections unanswered** (007 had three). Three findings changed the story rather than filling it in — see below. |

## What changed at 004

The restructure that produced 004 came out of a review of 003, and three changes are worth
recording because they are methodological rather than presentational.

1. **The metric justification was circular in 003 and is not in 004.** Script 112 selects
   crossover by asking which metric keeps the sign of the PC − backprop difference across
   stopping points — choosing the instrument by the answer it gives on the comparison it will
   then be used for. `CLAUDE.md` already forbids the same move for learning rates. 004 justifies
   the metric from **single-rule failures only** (111: endpoint fails as the stopping point
   moves; 343: crossover fails as lr rises) and demotes 112 to a robustness check reported
   *after* the result.

2. **The scenario split moved after the comparison.** In 003 it opened a tier before the results
   it summarised had been shown. In 004 it closes §6 as a tie-out of §5.

3. **Section count 11 → 9**, with the Class-IL / Domain-IL split as §7 and §8: an output-and-
   calibration investigation and a representation investigation, not the same analysis twice.

## What changed at 005

From an external review of 004. Three substantive changes, one presentational.

1. **The mechanism at the centre of the thesis was never measured against the outcome.** 004 ran
   behaviour → forgetting → PC vs backprop → interventions, and nothing touched what makes PC
   *PC*: that the hidden state relaxes to a configuration the feedforward pass would not produce.
   New **§6** adds it. 344 is the one existing result inside the mechanism — PC's damping appears
   in W2, not W1, because the output update uses the *settled* hidden activity — and the missing
   piece is correlating the settling displacement with ΔW and with retention. That displacement
   is already published on every train step as `handle["diag"]["displacement"]` and thrown away
   as a convergence check.

2. **Two section titles were asserting conclusions the evidence does not reach.** "Class-IL — an
   output and calibration problem" → "evidence for an output-competition component"; "Domain-IL —
   a representation problem" → "evidence for a representation-dependent failure".

3. **The masking-versus-freezing contradiction narrows further than either party had it.** Masking
   is train-time only — `active_vector` zeroes the error inside `output_error`, while `predict`
   still argmaxes over all ten units. So masking spares the task-1 output weights *while letting
   task-2 units keep learning*; freezing all of W2 spares them but also blocks task 2's readout
   path, forcing it into W1. That single difference is testable by freezing **only the task-1
   columns of W2** — added as a P1 build. It needs approval: `_apply_freeze` is whole-matrix.

4. Presentational: every build tagged **P1/P2/P3**, and a band at the top mapping the ten audit
   sections onto six Results chapters, so the audit is not mistaken for the report skeleton.

Rejected from the review: nothing outright, but the suggestion to compress to six sections was
applied as a **mapping** rather than a restructure — ten questions is the right granularity for
deciding what to build, six is the right granularity for a dissertation, and the page is the
former.

## What changed at 006

1. **Target alignment was missing and should never have been.** It is Song & Bogacz's *own*
   credited mechanism, so a report comparing PC to backprop cannot omit a direct test of it —
   whichever way it falls. It was filed as an appendix negative on the grounds that it "doesn't
   track forgetting"; that is the result, not a reason to hide it. Now the opening card of §5,
   marked re-run (the existing measurement is 5 seeds across four rules, pre-300).

2. **Weight-space PCA added to §6.** Accuracy can look settled while parameters wander, so the
   PCA of the weight trajectory under repeated alternation is the stronger version of the
   spiral-versus-loop panel: are weight groupings converging, orbiting, or drifting? 68 already
   stores the full W1 trajectory (600 × 6272 = 196×32, seed 0), so the Domain-IL W1 panel is
   computable from saved arrays today.

3. **Joint-then-sequential added to §6.** Train on the joint distribution first, then run the
   sequential protocol from there. If a network that already solves both tasks still collapses on
   task 1, forgetting is not a failure to *find* a joint solution but a failure to *stay* in one.
   That reframes the whole localisation question as parameter organisation rather than layer
   location, and it pairs with the PCA panel — does joint initialisation turn the Class-IL loop
   into a spiral?

4. **§2 reordered** generic → forgetting-specific: architecture, specification, PC's own control,
   statistics, then the metric. The metric column is last and visually flagged, because it is the
   most forgetting-specific decision in the report and everything downstream rests on it. In 005
   it had drifted into the middle of the section.

5. **Settling cut from four figures to two** and demoted from its own section into one column of
   §2. The cost curve (330) is appendix tuning detail; the displacement magnitude (346) is
   mechanism and moved to §5. Section count 10 → 9.

## What changed at 007

The trigger was a word-budget check. Written out at 150–225 words per working figure, 006's layout
came to ~26 main-text figures against a Results allowance of ~3,000 words — an overrun of roughly
40%. Five structural changes bring it to 15.

1. **The setup becomes a grouped parameter table** with figures as appendix evidence rather than
   six main-text figures each costing ~150 words to explain. Largest single saving.

2. **The two scenario sections merge into one, organised by tool rather than by scenario** (R5).
   Two sections that both ended in "we don't know" made a weak back half; by tool, the scenario
   contrast becomes the point of each figure instead of the section boundary.

3. **Added mechanisms demoted to Discussion** as a preliminary evaluation — but split by evidence
   quality, not wholesale. Replay, masking and freezing are results-grade at 10 seeds under the
   current protocol and stay in R2 and R5. **EWC, SI and k-WTA** go to Discussion: EWC's λ grid
   never bracketed its optimum, and 230's head-to-head was only smoke-tested.

4. **Replay stays in Results.** Moving it to the intervention section, and that section to
   Discussion, would leave the reader finishing the headline without knowing whether +1.42 is
   large. The scale belongs where the claim is made; the full ranking is Discussion.

5. **112 goes to appendix, not Methods.** It is inherently a paired PC−backprop figure, so placing
   it in Methods would show a rule-comparison result before the rule comparison exists — exactly
   the circularity removed at 004.

Also: every section now carries an explicit word and figure cost, and the page is banded into
Methods / Results / Discussion.

**Open question that changes the arithmetic:** whether captions count against the 8,000. If they
do not, a caption can carry ~100 words of "what was done / what it shows" per figure, and several
of the appendix demotions above become unnecessary. Worth settling before the 800 list is frozen.

## Section map of 007

| Part | § | Question | Verdict |
|---|---|---|---|
| Methods | M1 | What does forgetting look like? | needs rebuilding |
| Methods | M2 | The setup — parameter table, figures as supporting evidence | partly |
| Methods | M3 | How forgetting is measured, and why that metric | answered |
| Results | R1 | Two forgetting phenotypes | answered |
| Results | R2 | The rule comparison — small, systematic, scenario-dependent | partly |
| Results | R3 | Does prospective configuration explain the difference? | not answered |
| Results | R4 | Why these are two investigations, not one | not answered |
| Results | R5 | Where does the damage live? | not answered |
| Discussion | D1 | Added mechanisms — a preliminary evaluation | partly |

Methods ≈ 1,050 words · Results ≈ 3,000 · Discussion contribution ≈ 300.

## Section map of 006

| § | Question | Verdict |
|---|---|---|
| 1 | What does forgetting look like? | needs rebuilding |
| 2 | The setup, and why each choice was made | partly |
| 3 | Does each rule forget, and does it look the same in each scenario? | answered |
| 4 | Which rule is better, and is that stable? | partly |
| 5 | Is the difference actually caused by prospective configuration? | not answered |
| 6 | Why are these two investigations, not one? | not answered |
| 7 | Class-IL — evidence for an output-competition component | not answered |
| 8 | Domain-IL — evidence for a representation-dependent failure | not answered |
| 9 | What actually helps? | partly |

## Section map of 005

| § | Question | Verdict |
|---|---|---|
| 1 | What does forgetting look like? | needs rebuilding |
| 2 | The setup, and why each choice was made | partly |
| 3 | Is PC set up so that what we measure is actually PC? | answered |
| 4 | Does each rule forget, and does it look the same in each scenario? | answered |
| 5 | Which rule is better, and is that stable? | partly |
| 6 | Is the difference actually caused by prospective configuration? | not answered |
| 7 | Why are these two investigations, not one? | not answered |
| 8 | Class-IL — evidence for an output-competition component | not answered |
| 9 | Domain-IL — evidence for a representation-dependent failure | not answered |
| 10 | What actually helps? | partly |

**P1 builds (7).** §5 consolidated PC−BP sweep · §6 displacement vs ΔW vs retention ·
§7 Class-IL repeated alternation · §8 partial-column freeze · §8 trained linear probe ·
§8 per-layer weight path · §9 hidden-code displacement. Five of the seven are instrumentation of
runs that already exist rather than new experiments.

Related: the track itself is `claude_code_management/report_track.md`; the verified experiment
index is `claude_code_management/progress.md`.

## What changed at 008 — three findings, not three figures

008 is the first version where the verdicts come from runs rather than predictions, and three of
them moved the argument rather than confirming it. All three are recorded here because a later
reader will otherwise assume the plan was borne out.

1. **R3's mechanism replicates, and the replication is what indicts it.** Song & Bogacz's target
   alignment is higher for PC on the batch being trained (paired +0.0245 ± 0.0046, 5.4 sem,
   Class-IL) — their claim holds on our networks. Measured on a fixed task-1 batch during task 2,
   PC is *more* negative than backprop (−0.0175 ± 0.0053, 3.3 sem): the prospective configuration
   that helps the current task pushes the other task's outputs further away. An earlier draft had
   excluded alignment as an appendix negative; a negative result on the mechanism a paper credits
   is a result about that paper.

2. **R4's pre-registered prediction is refuted.** The plan predicted Class-IL would trace a closed
   loop under repeated alternation while Domain-IL spiralled in. Neither loops — both converge,
   and differ in where they stop (Class-IL plateaus near 35%, Domain-IL still climbing at 63%).
   The same run produced the project's sharpest scenario separation: PC − backprop is −0.72 ± 0.19
   after one switch and **−16.71 ± 2.45 (6.8 sem)** after five, in Domain-IL only. A two-task
   protocol understates it by more than twenty times.

3. **R5 changed its mind, and the section is stronger for saying so.** It was built on a readout
   account — the code survives, argmax misreads it, protect the readout and retention returns.
   806 shows freezing exactly the weights masking spares recovers +1.87 ± 1.03 against masking's
   +46.11 ± 4.52, and the two interventions do not even act at the same point of the curve
   (freezing moves the crossing, masking moves the endpoint). Independently, 923's probe reads
   82.6% on task-1 classes where argmax reads 21.1 — but **80.2% on the untrained network**, so
   ~2.4 points are attributable to training. The observation survives; the explanation does not.

Two defects in the 800 arrays were found while building these and are recorded rather than worked
around: 803 takes 2–5 checkpoints per run instead of ~20 (its interval divides the budget, not the
run), so 923 understates the end-of-training gap; and 805's joint-arm accuracy *curves* were lost
to a key collision, leaving only its endpoint scalars. Neither invalidates a result; both make one
weaker than it needs to be.
