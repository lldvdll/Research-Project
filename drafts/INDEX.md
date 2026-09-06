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
