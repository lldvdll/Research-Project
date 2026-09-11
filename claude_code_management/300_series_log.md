# 300 series log

## HANDOVER — current state snapshot (2026-09-05, latest). Overwrite this section wholesale at
## the next context consolidation — do not append to it, replace it.

**Where this project is right now**: inside the §2 BP-vs-PC comparison (narrative_plan.md), running
the 340-series (lr/width/depth paired sweeps). lr and the underlying mechanism questions (343/344)
are done. Width (341) and depth (342) were both found to be running under a broken PC settle
control partway through, and are being rerun correctly as of this snapshot — **that rerun may
still be in progress; check before trusting any 341/342 numbers dated before this snapshot.**

### Controls currently in force (the load-bearing decisions — check these before running anything)

- **PC settle control, MAIN/default** (used wherever width=32, depth=1 — i.e. everywhere except
  341/342): `dt=0.4`, adaptive stop (`stop_delta=1e-4, stop_patience=3, steps=100` cap). Chosen by
  the 330-334 series. Unaffected by anything below — still correct.
- **PC settle control, WIDTH/DEPTH-VARYING experiments ONLY** (341, 342, and any future script
  that sweeps width or depth away from 32/1): `dt=0.2`, same stop_delta/patience/cap. **This is a
  correction, not a preference** — dt=0.4 was found (345/346/347) to genuinely **oscillate**
  (confirmed NOT a tight overshoot near the true fixed point — the oscillation's mean sits 3-7x
  higher than the real converged value at the same depth) at depth≥2 and at Class-IL's H=4
  specifically. dt=0.3 fixes depth but NOT Class-IL H=4; **dt=0.2 is the only value confirmed to
  fix both**, and is still fast (≤37 settle steps even at depth=4, well under the 100 cap). If you
  are about to write or run a script that varies width or depth for PC, use dt=0.2 and say so, or
  first re-check with `src/metrics.py`'s `classify_settle` + a settle-trace script (346/347 are the
  templates) whether the new configuration is actually stable at dt=0.4 before assuming it isn't.
- **lr, shared value**: `0.02` for backprop (changed from the historical 0.01), PC unchanged at
  its existing `0.02`. Confirmed safe: 314 (exact copy of 310 at backprop lr=0.02) shows no
  substantive difference from 310 (all diffs small, p>0.34, same seeds). **310 itself was NOT
  rebuilt** — it still reports backprop at the old lr=0.01, and that's fine to keep citing; 314 is
  the standing proof it doesn't matter. If a NEW script needs backprop's lr and reads
  `config_300.yaml`, that file still says `0.01` — it was NOT edited (a deliberate, conservative
  choice: only 340/341/342 explicitly needed the new value and set it themselves; changing the
  shared config would cascade to every pre-existing 300-series script and was judged too large a
  side-effect to make unilaterally).
- **Metrics used throughout**: retention (`t1[-1]`, task-1 accuracy at the seed's own matched-
  competence stop point), crossover (height + step, `src.metrics.crossover`), S&B's mean test
  error (`src.metrics.mean_test_error`, computed but rarely plotted). Statistics: `paired_diff`
  (effect size, needs both sides finite), `paired_sign` (direction, censoring-aware, uses every
  seed), `paired_wilcoxon` (magnitude-aware companion to paired_sign, added this session — use it
  once a sweep gives more than a handful of paired points). Control is always `backprop` unless
  stated otherwise.
- **src/ changes made and approved this session** (both in `src/metrics.py`, both tested before
  use): `paired_wilcoxon(treatment, control)`; `classify_settle(disp, tol, patience, tail_frac)`
  — classifies a full settle displacement trace as converged/slow/oscillating/diverging. No other
  src/ files touched this session.

### Scripts, in the order this session built them, each with its actual status

| # | What it does | Status |
|---|---|---|
| 340 | lr sweep, backprop/replay/pc, shared grid, 10 seeds, both scenarios | **Done.** Figures + diff plot regenerated with connecting trend-lines (see below). |
| 314 | Copy of 310, backprop lr=0.02, paired against 310 | **Done.** No rebuild of 310 needed. |
| 343 | Plots 340's OWN saved data: absolute per-rule curves + crossover-censoring-rate | **Done.** No retraining involved, reruns instantly if 340's data changes. |
| 344 | Mechanism test: weight-step magnitude vs lr, W1 vs W2 (`weight_path_probe`) | **Done.** One-off; not wired into any other script. |
| 341 | Width sweep, backprop/replay/pc swept together, widths 4-128 | **Rerun in progress** (dt corrected 0.4→0.2, full fresh run, 10 seeds). Figures have the median-connecting-line fix; will regenerate correctly when this run completes. Superseded: an earlier dt=0.4 pass exists in memory/output logs but its Class-IL H=4 cell is invalid (settling didn't converge) — don't cite it. |
| 345 | Diagnostic: settle steps vs depth/width, fixed 200-step joint training, no threshold | **Done.** Explained why 342 was slow (settle steps balloon at depth≥2, and at Class-IL H=4). |
| 346 | Diagnostic: full settle TRACE vs depth/width at fixed dt=0.4, `classify_settle` applied | **Done.** Confirmed depth≥2 and Class-IL-H=4 are "oscillating", not slow/diverging. Also checked width=128 (converges fine) before it was added to 341. |
| 347 | Diagnostic: dt × depth stability grid (334's own dt values, depths 1-4) | **Done.** Found dt=0.2 is the only value stable at every depth AND at Class-IL H=4 (dt=0.3 covers depth only). |
| 342 | Depth sweep, backprop/replay/pc swept together, depths 1-4 | **First attempt killed** (2h+ in, depth≥2 producing invalid/non-convergent PC data, per 345). **Rerun in progress** (dt=0.2, full fresh run, 5 seeds), chained after 341 in the same background job. Never had valid depth≥2 data before this rerun. |

**Background job right now**: `341_width_sweep.py` (fresh, dt=0.2) `&&` `342_depth_sweep.py`
(fresh, dt=0.2), chained sequentially in one shell to avoid CPU contention. Estimated ~2-3h for
341, ~1h for 342 (grounded in 346/347's measured settle-step counts, not a guess). Launched
2026-09-05 (see this file's own timestamp / git log for the precise time if it matters). **Check
this job's output before doing anything else** — if it's finished, read both scripts' printed
paired-comparison tables and regenerate/inspect the figures; if not, it's expected to still be
running and silent (both scripts only print once a whole scenario finishes).

### Plotting conventions settled this session (apply to any new sweep figure)

- Box-and-whisker alone doesn't read as a trend — connect each method's own **median** across the
  sweep with a line in that method's colour (not the mean; the median is what the box already
  shows). NaN gaps in the connecting line (e.g. a fully-censored point) break the line rather than
  interpolating over it. Already applied in 340/341/342's `_box_panel` functions.
- Shared y-axis range per metric, spanning its real data (not 0-100), applied identically across
  scenario columns so Domain-IL and Class-IL panels are directly comparable at a glance (340's
  `_shared_ylim`).
- Cap-hit percentage annotated directly on any box where a rule didn't reach the matched-
  competence threshold on every seed — don't silently average in a capped run as if it converged.
- Best-lr/width/depth (by mean) marked directly on the panel for each method.
- A separate DIFFERENCE figure (paired_diff mean±SEM vs the swept axis) is expected alongside the
  absolute-value figure — absolutes-only can't show "does PC do better"; a diff-only plot can't
  distinguish "PC improved" from "the control degraded" (this exact mistake was made and corrected
  once already this session, in reporting 340's initial results).

### Next steps, in order

1. **Check the 341/342 rerun.** If done: read the printed paired tables, especially Class-IL H=4
   in 341 (does the crossover-loss finding survive now that settling actually converges there?),
   and 342's depth≥2 results (the first ones that have ever been valid).
2. Decide whether §2 (does PC forget too, and does it differ from BP) is now answerable as a
   headline finding, or whether Class-IL H=4 / any other flagged cell still needs work.
3. Update narrative_plan.md's §2 entry and this file's own (non-handover) log section with the
   341/342 rerun's actual numbers once in.
4. Move toward §3 (why does PC differ / is there a useful difference anyway — narrative_plan.md)
   — nothing built yet for the trained linear probe, layerwise cosine, path efficiency, or target
   alignment toolkit described there. 344's W2-settled-activity finding (PC's output-layer update
   uses the SETTLED hidden activity as its presynaptic term, not the feedforward one — this is
   where its lr-damping actually showed up, not W1 as hypothesized) is a live, unresolved thread
   that could feed directly into §3's mechanism question if picked up.
5. The idea of using `classify_settle`'s logic as a LIVE stopping criterion (bail out early on
   oscillation/divergence in production training, not just diagnose it after the fact) was raised
   and deliberately deferred — a real `src/predictive_coding.py` behaviour change, not yet
   designed, would need its own approval/testing cycle.

## Core (in the report)
- 310 (seeds 10-19) is the canonical §1 figure, going forward standard seed set for the series.

## Non-core (exploratory, reference only)
- 300 (seeds 0-9): superseded by 310 — seed 9 was an outlier.
- 301: Domain-IL retention correlates with shared-unit digit-pair similarity (r=0.69); Class-IL does not.
- 302: seed 9's outlier traced to intrinsic digit-set difficulty, not init or run-order.
- 311: 301's Domain-IL crossover correlation replicated (r=0.71); finals correlation did not (r=0.18).
- 312: Class-IL retention splits into ~0%/~20% groups tied to which digits land in task-1.
- 313: confirmed at 70 seeds (r=0.66); tied to individual-digit difficulty (8, 9, 5), not a bug.

## 320 — NCM (superseded by restructure, see narrative_plan.md)
- Built init-frozen NCM for both tasks across full training (not just post-switch) per request; bug fixed (grid spacing must match log_every not eval_every).
- Real result: NCM ~flat/high (70-90%) throughout for both tasks — raw-pixel (no network) NCM baseline on all 10 digits = 81.2%, checked directly.
- Conclusion: NCM is centroid-only (first-order), low ceiling set by task geometry, little dynamic range — same failure mode as pre-100 exp 20 (knowledge_base.md §6.6.3, "exp 21 invalidated exp 20's probe"). Inadequate for information-vs-access; needs a trained/refit linear probe instead (not yet built).

## Narrative restructure (2026-09-03)
- Old order (info-vs-access §2 → "does PC differ" §3) had no hypothesis bridging them — PC's relevance was unmotivated. No theory says PC fixes a readout-recalibration problem specifically (pre-100: PC's target alignment ≈ BP's, path-efficiency edge backwards from needed — not evidence, just why no bridge exists).
- New order: §2 = does PC differ from BP at all (empirical, first). §3a/b = explain the difference, or explain its absence, using the info-vs-access + layerwise toolkit. §4 (freeze/mask) onward unchanged.
- Full updated structure in narrative_plan.md.

## §2 / 330 series — PC settle-dynamics controls (dt, steps) chosen; PC headline figure built
- **src/ change (approved)**: `pc_settle`/`pc_update`/`make_pc` gained `stop_delta`/`stop_patience` — settling now stops itself (criterion: `|Δdisplacement| < 1e-4` for 3 consecutive steps) instead of running a fixed step count. Backward-compatible (byte-identical when unset, verified directly); `steps=0` invariant preserved.
- **333**: validated the criterion against the historical known-good config (dt=0.1, steps=50; scripts 50/51/59's "worst case 18") — converges ~14-26 steps at every training stage, real matched-competence run, both scenarios. Criterion was fine; the *original* 330 design (one global fixed step budget set by the slowest dt) was the actual cost problem.
- **334**: full dt-range settle traces (single seed, both scenarios) — dt 0.01–0.5 all converge to the SAME displacement plateau (dt doesn't move the fixed point, only the speed), dt ≳0.7 oscillates at a distinct higher plateau and never converges.
- **330 (final)**: 2 conditions (joint, sequential), crossover added, 5 seeds, both scenarios. dt-vs-settle-steps is a clean V; minimum shifted twice as resolution improved (0.2 → 0.3 → 0.4, via `--add-dt` merges, no full reruns). **dt=0.4 chosen**: fewest settle steps, flat on every accuracy/forgetting metric across the whole stable range up to the 0.5→0.6 instability edge. **Bug found and fixed**: `run_classil` already remaps labels/active under `label_map=`; the sequential-condition wrapper was remapping AGAIN, silently colliding classes onto wrong Domain-IL units (`[0,1,2,3,4] → [0,2,2,0,4]`) — explains why every Domain-IL seed was hitting the iteration cap regardless of dt. Class-IL unaffected (`label_map=None` there).
- **331** (dt=0.4 fixed, steps ∈ [1,2,3,5,10,20,50], BP reference, 5 seeds, both scenarios): residual displacement rises sharply steps=1→~5-10 then plateaus (state genuinely under-settled at steps=1), but accuracy/retention/crossover are flat from ~steps=5-10 onward, only a small (not rigorously paired) gap at steps=1. Validates adaptive stopping — it already lands ~8-10 steps at dt=0.4 on its own.
- **332** (PC headline, dt=0.4 adaptive stop, 10 seeds, both scenarios, exact copy of 310's figure): Class-IL retention=6.0% crossover=66.4%; Domain-IL retention=37.0% crossover=75.1%. Full budget (max_iters=5000) — the Domain-IL cap-hits seen in 330/331's trimmed sweeps (1500) did not recur, confirming those were a budget artifact, not a real per-seed Domain-IL difficulty.
- **Open Q1 (provisional)**: vs 331's BP reference (different seed count/budget, NOT a clean paired comparison), PC shows no clear uplift over BP — Class-IL slightly higher, Domain-IL slightly lower, both within a few points. 340 must confirm with a proper same-seed, same-budget paired comparison.
- **Open Q2 (shelved for §3)**: Class-IL retention (332) looks like two groups — near-zero and ~10-30% — not a smooth spread. Matches 312/313's established finding (retention splits ~0%/~20%, tied to which digits land in task-1), now also seen under PC. Not a convergence-speed artifact — the stop criterion already reads retention at matched task-2 competence, so it's legitimately comparable across seeds regardless of learning speed. Not a confound for the rule comparison (paired-seed design); relevant to §3's localization/character question later.

## §2 / 340 series (2026-09-05) — paired BP-vs-PC comparison across lr/width/depth, and a settle-stability correction

**src/ changes (both approved):**
- `src/metrics.py` gained `paired_wilcoxon(treatment, control)` — signed-rank companion to
  `paired_sign`, magnitude-aware where the sign test's small-n ceiling (p=0.062 at n=5, the best
  obtainable when all seeds agree) isn't enough to separate a real effect from an unlucky-but-
  unanimous small sample.
- `src/metrics.py` gained `classify_settle(disp, tol, patience, tail_frac)` — classifies a full
  settle displacement trace as converged / slow / oscillating / diverging, applying the existing
  stop_delta/stop_patience criterion post-hoc to a full uncapped trace rather than live. Verified
  on synthetic traces of all four kinds before use.

**340 (lr sweep)** — shared absolute lr grid for every rule (backprop, replay, pc all swept
identically; replay was NOT swept in an earlier version, corrected after review since replay's
`train_step` literally is backprop's update rule, so its lr is in the same units, unlike PC's).
Final grid `[0.005, 0.01, 0.02, 0.04, 0.08, 0.16]`, 10 seeds. Headline, corrected after an initial
misreading: **both rules' absolute retention/crossover peak near lr=0.01-0.02 and DECLINE past it
in every scenario/metric** — the growing PC-minus-backprop gap at high lr is backprop degrading
*faster*, not PC improving. Established defaults (0.01 bp / 0.02 pc) sit close to jointly-optimal,
not in a blind spot. Class-IL: PC edge grows toward high lr (crossover +3.1pp at lr=0.04, p=0.006).
Domain-IL: PC significantly *behind* on retention at the established 0.01/0.02 (p=0.062 each,
5-seed ceiling) — resolved to non-significant once seeds reached 10, see below. Crossover becomes
fully **censored** (undefined, 0/10 pairs) at lr=0.16 in Class-IL — a measurement-resolution
failure (eval cadence can't resolve the crossing once learning is fast enough), not a null result.
314 (below) confirms adopting lr=0.02 for backprop doesn't substantively change 310.

**343** (reuses 340's saved data, no retraining) — plotted absolute per-rule curves + a
crossover-censoring-rate panel directly. Confirmed: Class-IL censoring hits backprop first and
harder (100% by lr=0.16) than PC (60%). Fitted degradation slopes (common window, both rules) are
mixed, not uniformly "PC gentler": true in Class-IL both metrics and Domain-IL crossover, but
Domain-IL retention degrades *faster* for PC (-1.87 vs backprop's -1.52) — a real cross-metric
disagreement, not resolved further here.

**344** — direct mechanism test of "does PC's realised weight-step magnitude scale sub-linearly
with lr" (fixed 300-step joint training, `src.probes.weight_path_probe`, W1 vs W2 as intended
control). **Hypothesis refuted as stated**: W1 (hidden, PC's settled error) shows NO differential
damping (backprop/pc slopes 0.93/0.90, both ≈linear). W2 (output, "same direct error for both
rules") is where the damping actually shows up (backprop 0.99, pc 0.78) — because PC's W2 update
multiplies the same output error against the *settled*, not feedforward, hidden activity, so
settling reaches W2 too, just via a different route than assumed. Not further isolated.

**341/342 (width/depth sweeps)** — design: `--add-seeds=LO-HI` / `--add-width=X` /
`--add-depth=X` merge flags (330's dt-merge pattern, generalised); replay swept on the same
grid as backprop/pc (unlike 340, since width/depth are shared architecture parameters, not a
rule-specific tuning axis). **341 first pass (dt=0.4, 10 seeds after `--add-seeds`, widths
4-64+128)**: Class-IL — PC beats backprop at every width except H=4 (loses badly, crossover
-6.2pp p=0.002); wins significantly at H=8/16/32/64 (all p≤0.021). Domain-IL — mirror image:
PC wins only at H=4 (+2.1pp p=0.021), loses at H=32/64 (both p≤0.021), flat in between.
**342 first attempt (dt=0.4) never completed** — killed after ~2h at depth≥2 once the cause was
found (below); no valid depth≥2 data exists from that attempt.

**345** (diagnostic, fixed 200-step joint training, no threshold-stopping, settle_steps read from
`handle["diag"]`) — explains 342's stall directly: mean settle steps balloon from ~9 (depth=1) to
86-231 (depth 2-4), with 48-78% of updates exceeding the *production* 100-step cap at depth≥2, in
both scenarios. Width is clean 8-128; **Class-IL specifically at H=4** also shows a blowup
(164.6 mean, 65% cap-hit) not present in Domain-IL (11.0 mean, 0% cap-hit) — scenario-specific,
qualifies 341's H=4 crossover number (measured under a non-converging settle process there).

**346** (334's method — full uncapped settle trace, `classify_settle` — generalised from dt to
depth/width at FIXED dt=0.4) — resolves what 345 could only infer from step counts: **depth 2, 3,
and 4 all classify "oscillating", not slow, not diverging**, in both scenarios. A real stability
transition between depth=1 and depth=2 (not a progressively worsening problem — depths 3/4 sit at
a similar oscillation plateau to depth=2, don't escalate further). Width clean 8-128 both
scenarios; **Class-IL H=4 also oscillates** (Domain-IL H=4 converges fine, step 9) — confirms
345's scenario-specific finding directly. **Empirical check on what "oscillating" actually means**
(comparing dt=0.4's oscillation-tail mean against dt=0.3's genuine converged value at the same
depth): the oscillation is centered 3-7x HIGHER than the true fixed point, growing with depth —
this is NOT "found the right point, overshooting around it", it's a qualitatively wrong state.
Accepting oscillation as a stopping criterion was considered and rejected on this evidence.

**347** (dt × depth stability grid, 334's own depth=1 dt values `[0.02,0.05,0.1,0.2,0.3,0.4]`,
extended to depth 1-4) — **the only unstable cell in the entire grid is dt=0.4 at depth≥2.**
Every other (dt, depth) combination converges cleanly, both scenarios. dt=0.3 is stable at every
depth (settle steps ≤30 even at depth=4) but does NOT fix Class-IL's H=4 (checked directly,
still oscillates at dt=0.3). **dt=0.2 is the only value confirmed to fix BOTH problems** (every
depth 1-4, and Class-IL H=4) — adopted as the corrected control for width/depth-varying
experiments specifically, not as a replacement for the main dt=0.4 control (which stays valid and
unaffected at the established H=32/depth=1 baseline, and at every OTHER width 8-128).

**314** — exact copy of 310, backprop lr=0.02 instead of 0.01 (the shared value 340 supports).
Paired against 310's own saved arrays, same seeds: all differences small and non-significant
(retention -0.6/-0.9pp, crossover +0.2/-0.2pp, both scenarios, all p>0.34). **310 does not need
rebuilding** — safe to treat lr=0.02 as the adopted shared value going forward.

**CONTROL CORRECTION, applies to any current or future width/depth-varying PC experiment**:
use **dt=0.2**, not the established dt=0.4, whenever width or depth is swept away from the
H=32/depth=1 baseline. dt=0.4 remains correct and unaffected for every script that holds both
fixed (330-334, 340, 332, 310, 314). **341 and 342 are the only 300-series scripts affected**
(the only two that vary width/depth) — both corrected in place (`OPTIMAL_DT = 0.2`, commented)
and rerun in full from scratch, 2026-09-05. Pre-300 scripts that varied depth/width (41, 55, 59,
74) are not in scope — already excluded by this project's evidence standard.

**Open, not yet resolved:**
- Whether 330's "dt is flat in accuracy/forgetting within the stable range" finding (established
  at depth=1) generalises to depth>1 was NOT directly tested (347 only checked settle-trace
  stability, not downstream accuracy) — not needed once a single dt (0.2) covering the whole
  range was found, but worth keeping in mind if a future script needs to justify a per-depth dt.

## Process-management incident, and 341/342 corrected results (2026-09-05, later same day)

**What happened.** The original dt=0.4 342 run (the one the CONTROL CORRECTION section above
describes as needing replacement) was never actually killed — the `Stop-Process` call that
should have ended it was rejected earlier in the session, and the rejection was never revisited.
It ran standalone, on the wrong control, for **~3 hours** (13:37→16:44) before being found still
alive and still burning heavy CPU. Separately, a corrected `341 && 342` chain (dt=0.2) had
already completed 341 successfully once (output at 16:18), but a **second, redundant invocation**
of the same chain was then launched (16:35), which would have spent another 2-3h re-doing 341
before ever reaching 342. Both were killed. Per instruction, 341 and 342 were then re-launched
**independently, in parallel, not chained** — both at the already-corrected `dt=0.2` (both
scripts' `OPTIMAL_DT` constant was already 0.2 from the correction earlier in the day, so no code
edit was needed, just a different launch pattern). Both completed within the hour. **Seed count:
341 defaults to 5 seeds (10-14) when run with no flags** — the "10 seeds" in this file's own
CONTROL CORRECTION note above referred to the *stale, dt=0.4* pass; the fresh dt=0.2 numbers
below are n=5, with a `--add-seeds=15-19` extension queued and running in the background
(job `bes1rek99`) to bring 341 back to 10. **342 was always 5 seeds by design** (its own
docstring), so its numbers below are final at that seed count.

**341 corrected (dt=0.2, n=5, seeds 10-14) — resolves the open H=4 question above.** Paired
pc-backprop crossover, Class-IL: H4 **-3.5 ± 0.8** (p=0.062, PC still loses) / H8 +2.3 ± 1.6 (n.s.)
/ H16 **+3.2 ± 1.5** (p=0.062) / H32 **+1.7 ± 0.5** (p=0.062) / H64 **+1.4 ± 0.3** (p=0.062).
Domain-IL: H4 **+2.7 ± 0.5** (p=0.062, PC's only Domain-IL win) / H8 +1.1 ± 0.9 (n.s.) / H16 -0.5
± 0.5 (n.s.) / H32 **-1.0 ± 0.3** (p=0.062) / H64 **-1.0 ± 0.3** (p=0.062). **Answer: the H=4 loss
survives the dt correction** — it is not purely a settle-instability artefact — but its magnitude
roughly halves from the stale pass (-6.2 → -3.5). Because n also dropped 10→5 in the same
comparison, that shrinkage is confounded with sample size and should not be read as "half the
effect was the confound" until the seed-15-19 extension lands. Every other width's direction and
rough magnitude matches the stale pass closely (e.g. H32 +1.43→+1.7, H64 +0.97→+1.4), which is
reassuring — the settle correction mainly mattered at H=4, as 345/346 predicted, not everywhere.

**342 corrected (dt=0.2, n=5, seeds 10-14) — first-ever valid depth≥2 result, resolves the open
depth question above.** Paired pc-backprop crossover, Class-IL: D1 **+1.7 ± 0.5** (p=0.062) / D2
**+4.0 ± 1.1** (p=0.062) / D3 **+3.4 ± 1.2** (p=0.062) / D4 **+5.0 ± 1.2** (p=0.062) — **PC wins
at every depth tested, and the margin grows with depth** rather than washing out. Domain-IL: D1
**-1.0 ± 0.3** (p=0.062, PC loses) / D2 -0.5 ± 0.6 (n.s.) / D3 **-1.7 ± 0.6** (p=0.062, PC loses)
/ D4 -3.3 ± 1.2 (n.s., 4/5 seeds still favour backprop but not unanimous). So Class-IL's PC
advantage and Domain-IL's PC disadvantage **both persist across depth**, rather than being an
H=32-specific artefact — this is new evidence directly relevant to narrative_plan.md's
conditional §6 ("does the mechanism generalise with depth"), even ahead of §3 explaining why.
⚠ **Retention at depth is increasingly unreliable for PC specifically**: cap-hit (fraction of
seeds that never reached matched competence) for PC climbs **20% (D1) → 20% (D2) → 40% (D3) →
80% (D4)**, against 0% for backprop at every depth. Domain-IL D4's retention diff (+5.5 ± 3.6,
the one number that looks like PC "wins" on retention) is almost certainly this censoring, not a
real effect — only 1/5 of PC's D4 seeds reached matched competence at all. Crossover is not
subject to this (it is defined from the trajectory, not the stopping point), which is the same
reason this project already prefers crossover over endpoint retention (112) — this is a second,
independent case of the same failure mode showing up.

**341 at n=10 (job `bes1rek99` completed) — final, supersedes the n=5 numbers above.** Paired
pc-backprop crossover, Class-IL: H4 **-4.1 ± 0.7** (p=0.002, wilcoxon p=0.002 — PC loses, and
harder than the n=5 pass suggested) / H8 **+2.9 ± 0.8** (p=0.021, wilcoxon p=0.010) / H16 **+3.0
± 0.8** (p=0.002, wilcoxon p=0.002) / H32 **+1.4 ± 0.4** (p=0.021, wilcoxon p=0.006) / H64 **+0.9
± 0.3** (p=0.021, wilcoxon p=0.014). **PC wins significantly at every width except the H=4
capacity floor, where it loses significantly** — a clean, well-powered result, no more n=5
sign-test ceilings. Domain-IL: H4 **+2.2 ± 0.5** (p=0.021, wilcoxon p=0.004, PC's only Domain-IL
win) / H8 +0.8 ± 0.5 (n.s.) / H16 +0.2 ± 0.5 (n.s. — sign flipped from the n=5 pass, confirms
this cell is genuinely near zero) / H32 **-0.7 ± 0.2** (p=0.002, wilcoxon p=0.002) / H64 **-0.8 ±
0.2** (p=0.021, wilcoxon p=0.004). **The mirror-image pattern (PC wins only at the capacity
floor, loses at mid-to-large width) is now confirmed at real significance, not just n=5's p=0.062
ceiling.** Retention is directionally consistent with crossover at every width but reaches
significance nowhere (largest: Domain-IL H32 −1.5 ± 0.8, p=0.109) — crossover remains the metric
that separates here, consistent with 112's general finding.

**342 at n=10 (job `byur6g5f8` completed) — final, supersedes the n=5 numbers above and revises
two things.** Paired pc-backprop crossover, Class-IL: D1 **+1.4 ± 0.4** (p=0.021, wilcoxon p=0.006)
/ D2 **+2.9 ± 0.8** (p=0.021, wilcoxon p=0.010) / D3 **+3.3 ± 0.7** (p=0.002, wilcoxon p=0.002) /
D4 **+3.4 ± 0.9** (p=0.021, wilcoxon p=0.004). **PC wins significantly at every depth**, and the
advantage grows from depth 1 to 3 then **plateaus** at depth 4 (3.3→3.4) rather than continuing to
climb — the n=5 pass's D4 of +5.0 was noise; **revise "grows with depth" to "grows then
plateaus."** Domain-IL: D1 **-0.7 ± 0.2** (p=0.002, wilcoxon p=0.002, PC loses) / D2 -0.3 ± 0.3
(n.s.) / D3 -0.9 ± 0.6 (n.s. — **lost the significance the n=5 pass showed**) / D4 **-3.5 ± 0.6**
(p=0.021, wilcoxon p=0.004, PC's largest loss in the whole grid). **Revise the Domain-IL claim**:
not "persists at every depth" — significant at the depth-1 and depth-4 bookends, flat/noisy at
2-3. ⚠ **D4's cap-hit for PC is 80%** (only 2/10 seeds reached matched competence) — crossover is
computed from the trajectory and is less exposed to this than retention, but a result resting on
2 fully-converged seeds plus 8 capped ones needs more scrutiny before it is cited as the grid's
largest effect. D4 retention (+2.2 ± 2.9, n.s., huge SEM) is unusable for exactly this reason,
consistent with 112's established retention-under-censoring failure mode.

**Still open:**
- D4 Domain-IL's large crossover loss (-3.5) rests on a mostly-capped PC arm (80% cap-hit) — worth
  a longer budget or a dedicated check before treating it as a real depth-4-specific effect rather
  than an artefact of undertrained PC seeds.
- Whether 342's Class-IL advantage (grows then plateaus) / Domain-IL disadvantage (bookends
  significant, middle flat) is itself the mechanism-generalisation answer narrative_plan.md §6 is
  waiting on, or whether it still needs §3's explanatory toolkit run at depth to be a real "why"
  answer rather than just a bigger "what."
