# Handover — 2026-09-10 (supersedes the 2026-09-06 version wholesale — do not append to this
# file at the next consolidation, replace it, matching the 300-series log's own convention)

Everything needed to pick this up cold. Read `now.md` first (one screen), then this.

---

## 1. Where the project is

**All 800-series training has run.** All 900-series figures that were listed as "not written" in
`report/sections/results.tex`/`methods.tex` now exist as real PNGs — checked directly against the
figure list in `script_plan_800_900.md`, not assumed. What's happening now is a **refinement pass**:
folding two new findings (the sigmoid activation flip, 802/808; task-1 overtraining, 807) into
figures that were built before those findings existed, fixing one real bug found along the way,
and extending a handful of cells that never converged within budget.

**`report/sections/*.tex` still has stale captions.** Most of results.tex's `\gap{...not written}`
placeholders are now wrong — the figure exists, the caption just hasn't been written. This is
listed here as a known gap, not fixed — it's the user's own report-writing pass
("piecing together"), not a plotting task.

## 2. What changed this session, in the order it happened

1. **802's activation sweep was found to have actually finished** (handover previously said
   "queued") — full 10-seed run already on disk. Result: PC−backprop crossover under sigmoid
   flips **positive** in Domain-IL (+0.74 ± 0.19, Cohen's d = 1.25 — the largest standardized
   effect anywhere in the project), where tanh and relu both give a real negative. Class-IL
   stays positive under all three activations (activation-insensitive there).
2. **808** (new): 803's full mechanism toolkit (target alignment, displacement) re-run on the
   ONE condition where the sigmoid effect lives — sigmoid + Domain-IL, both rules, 10 seeds.
   Target alignment does **not** explain the advantage even there (interference-alignment
   +0.0050 bp vs +0.0032 pc, post-switch only — effectively identical) — a stronger negative
   than the tanh result, which had no advantage to explain in the first place.
3. **`cohens_d_paired`** added to `src/metrics.py` — standardized effect size, since S&B report
   neither a p-value nor an effect size for their own continual-learning result (graphical 68%
   CI bands only), so this project's numbers need their own standardized reading.
4. **`classify_forgetting`** added to `src/metrics.py` — shape classifier for post-switch task-1
   traces (collapse / delayed / partial / rising / noisy), mirroring `classify_settle`'s design.
   Found and fixed a real ordering bug before trusting it (noise must be checked before the
   rising-trend check, or a noisy tail's slope looks like a spurious rise by chance). Applied to
   803's full 2×2 (`928`): **Class-IL splits into collapse/delayed/partial for both rules, and
   9/10 seeds get the identical shape under both rules** — a data property, not a rule property.
   **Domain-IL is uniformly "partial", all 10 seeds, both rules, no exceptions.**
5. **807** (new): does retention improve if task 1 keeps training past the matched-competence
   threshold? Train to threshold once (N steps), branch four independent task-2 phases from
   snapshots at 1×/1.5×/2×/4×N. Finding: backprop benefits in BOTH scenarios (Domain-IL most:
   d=1.43), PC benefits in Class-IL (d=0.65) but **gets nothing from it in Domain-IL** (d=0.03,
   essentially zero) — needed `src/runner.py`'s `tail_iters` to accept a per-task list (small,
   approved, verified bit-identical for the scalar path every existing caller still uses).
6. **User feedback, applied**: 918 needed both rules (was backprop-only on an assumption that
   turned out not to hold up); 912 needed activation as a fourth axis; 915/916 needed the sigmoid
   condition folded in; 923's backprop-only choice needed checking, not assuming (checked: holds
   up there, unlike 918); a crossover-vs-retention diagnostic was wanted (built as `909`, new);
   927 needed jittered per-seed lines instead of bare error bars, plus a 3.0× point.
7. **Bug found while fixing 912 (real, cost real time): `_vals` in 341/342 doesn't sort by
   seed before pairing.** Every other code path (full run, `--add-seeds`, `--add-width`)
   preserves seed-order alignment by construction; a new `--extend-cap` feature (added this
   session to fix non-converged cap-hit cells) reassembles rows in a way that breaks that
   alignment, silently pairing the wrong seeds against each other in the PRINTED stats only
   (the saved arrays and the 912/927-style figures, which sort by seed correctly, were never
   wrong). **This produced one incorrect report to the user mid-session** (H=4 Class-IL falsely
   reported as no longer significant) — caught, corrected, `_vals` fixed in both files to sort by
   seed. **If you see a width/depth paired number from before this fix that doesn't match a
   `--replot` rerun, trust the replot.**
8. **`--extend-cap=X` added to 341/342** — re-runs only the (method, width/depth, seed) cells
   that never reached threshold within budget, at a raised cap, replacing just those rows.
   341 extended to 20000 (from 5000): 0 (class-il) / 1 (domain-il) cells still capped after.
   **342's extension was still running when this handover was written — check its output before
   trusting any depth-sweep number.**
9. **A recurring pattern found via the cap-hit scan: seed 13 (almost always PC, almost always
   Class-IL) is anomalously slow across AT LEAST SEVEN independent scripts** — 802, 803, 804,
   805 (both its scratch AND joint arms), 806, 808, and 340's own lr sweep. This is no longer
   "a cell that happened to be slow" — it is a robust, cross-cutting, independently-replicated
   pattern, very likely the same phenomenon as the old "seed 9" outlier (302), a genuinely hard
   digit split rather than noise. **804 and 805's seed-13 cells are NOT YET extended** (only
   802/803/806/808 were, via the patch job; 341/342 via `--extend-cap`) — deliberately not
   launched this session to avoid a third/fourth concurrent heavy job on top of the two already
   running. Worth its own small write-up regardless of whether every last cell gets extended:
   the CONSISTENCY across seven independently-written scripts is itself the finding.
10. **806 (masking) has a real complication, not just slow convergence**: several backprop
    seeds' task-2 accuracy under masking sits far below 90% (22–58%) rather than just missing
    threshold narrowly. Masking may genuinely impair/slow task-2 acquisition, not just protect
    task-1 for free — complicates the "masking recovers 45 points, clean win" framing. Being
    extended for honest numbers (patch job below), but **this is a finding to decide what to do
    with, not a bug to silently patch over.**
11. **340's own cap-hit cells were deliberately left untouched** — its docstring treats hitting
    the cap at low lr as intentional signal ("how far did it get before running out of budget"
    IS part of what that sweep measures), so extending it would change what's being measured,
    not just clean up noise.
12. **Floor-normalized probe reading** (new analysis, no new training): of the TRUE available
    headroom (joint ceiling at H=32 minus the untrained probe floor), training captures only
    **~18–30%** of it — Class-IL bp 25.3%/pc 17.9%, Domain-IL bp 30.0%/pc 20.6%. Backprop
    captures more of the available headroom than PC in both scenarios. Concrete number for "how
    much information is preserved", replacing a purely qualitative read.

## 3. Both jobs from §3 (old) finished — final confirmed numbers

**342's `--extend-cap=20000` completed, then re-run with `--replot`** (the extend job's own
process started before the `_vals` fix landed, so its OWN printed output was wrong — same bug,
same fix, confirmed the fix matters here too: several cells' SEM dropped noticeably, e.g.
Class-IL D3 crossover SEM 2.0→0.7). **Final, correct 342 numbers:**

```
Class-IL crossover (pc-bp): D1 +1.4±0.4 (p=0.021)  D2 +2.9±0.8 (p=0.021)
                            D3 +3.3±0.7 (p=0.002)  D4 +3.7±0.9 (p=0.021)
Domain-IL crossover (pc-bp): D1 -0.7±0.2 (p=0.002)  D2 -0.3±0.3 (n.s.)
                             D3 -0.9±0.6 (n.s.)     D4 -3.0±0.7 (p=0.021)
```
PC wins significantly at EVERY depth in Class-IL now, and the earlier "grows then plateaus"
read should probably be revised to "keeps growing" — D4 (+3.7) is now slightly above D3 (+3.3),
though within each other's error bars, not a clean plateau. Domain-IL's D4 result is now much
more trustworthy: cap-hit dropped from 80% (the old n=5 pass) to 10%, so the earlier concern that
D4's Domain-IL loss might be an artefact of undertrained seeds is largely resolved — it's real.
Residual cap-hit remains at the extremes even at 20000 (Class-IL D3 10%/D4 30% for pc,
Domain-IL D3/D4 10% for pc) — some depth-4 PC seeds may simply need a still-larger budget; this
is now a small residual, not the dominant effect it was.

**The capped-row patch (802/803/806/808) completed — and 806 turned up something significant,
not just slow convergence.** Even at 20000 steps (4x the original budget), MASKING substantially
impairs task-2 acquisition for a majority of seeds, not just a few borderline ones:

```
              backprop            pc
control      t1=4.8  t2=91.5    t1=6.0  t2=91.4     (0% cap-hit, both)
freeze_w2    t1=3.2  t2=91.7    t1=4.3  t2=91.5     (0% cap-hit, both)
freeze_w2_t1 t1=6.7  t2=91.5    t1=6.3  t2=91.3     (0% cap-hit, both)
mask         t1=46.8 t2=46.5    t1=27.6 t2=60.4     (bp 9/10 cap-hit, pc 5/10 cap-hit)
```
Task-2 accuracy under masking craters to a MEAN of 46.5% (backprop) / 60.4% (pc), against ~91%
everywhere else — this is not "slightly slower", several seeds never get near 90% even at 4x
budget (backprop seed 11: 23.3%; pc seed 12: 5.4%). **This is a real cost, not a footnote.**
Masking's task-1 recovery (46.8/27.6) has to be read against this: it is not a free lunch, and
backprop recovers MORE task-1 retention under masking than PC does (46.8 vs 27.6) while ALSO
losing more task-2 capability (46.5 vs 60.4) — so even the RULE COMPARISON under masking flips
depending which side you look at. **R5's masking story needs to account for this before going
in the report** — "masking recovers 45 points, clean win" is no longer an accurate summary.
`931` has been re-run with this corrected data; its `mask` bars still read near-zero because
931 plots CROSSOVER, which is mostly censored under masking (task-1 rarely falls below task-2)
— a reader would NOT see this new finding from 931 alone, it needs its own callout in prose.

**807 is re-running now** (3.0x multiplier added, launched once the above two jobs cleared).

## 4. Queued, blocked on the above finishing (to avoid 3-way CPU contention)

1. Re-verify 342's numbers with `--replot` once its extension completes (see §3).
2. Re-run `807_overtrain_task1.py` in full with `MULTIPLIERS` now including 3.0 (already edited
   into the script) — queued so it doesn't contend with the two jobs above.
3. Run the trained probe (`linear_probe_fn`) against 806's masked/frozen conditions specifically
   — needs 806's patched (converged) data first. This is the concrete way the probe and
   freeze/mask toolkits interact: does masking raise the probe's ceiling (real representational
   change) or leave it flat (unlocks existing information without changing it)?
4. ~~Rebuild the intervention-ranking figure(s)~~ — **done**. `931` regrouped: two vertical
   panels (Class-IL/Domain-IL), each intervention now measured against ITS OWN rule's control
   for both backprop and pc (the original version wrongly listed "predictive coding" as one row
   among the interventions — a category error, since PC is a base rule, not something applied on
   top of one). Surfaced something new: **k-WTA is far more harmful for PC than backprop**
   (Class-IL −32.4 pc vs −9.8 bp; Domain-IL −30.2 pc vs −2.6 bp) — bigger than the original
   figure's own "PC worse under k-WTA" note suggested, because the old figure never plotted PC's
   OWN k-WTA-vs-PC-control delta, only PC-vs-backprop-bare as one of many rows. **Re-run once
   806's patch lands** — current numbers use 806's pre-patch data for the `mask` row, everything
   else (130, 200, 210, 220) is unaffected by the pending jobs. `921` checked and left alone —
   its backprop-only design is a genuine data-availability constraint (the 50-seed block only
   exists for backprop), not an unchecked assumption like 918's was.
5. Update results.tex's stale `\gap{}` captions — NOT this session's job, flagged for the user.

## 5. Traps found this session — check these before writing new code

- **`_vals`-style pairing helpers must sort by seed explicitly.** Never rely on row insertion
  order for a paired comparison, even if the current code path happens to preserve it — the next
  merge/patch feature added later probably won't. (§2.7 above.)
  ~~Insertion order happens to align two methods' arrays~~ — true only by accident of how rows
  were originally generated; broke the moment a partial-patch code path was added.
- **`run_classil`'s `curves` reports EVERY task in the `tasks` list at every logged step, even
  ones with `max_iters_per_task=0` for that phase.** This is what makes 807's branching design
  work (pass `max_iters_per_task=[0, K]` to train only task 2 while still logging task 1's
  concurrent accuracy) — no custom evaluation loop needed. Same trick works in reverse.
- **A capped run's variance is deflated, not just its mean potentially biased.** A truncated
  budget pins similar seeds near the same ceiling, which can make a real effect look MORE
  significant than a properly-converged version of the same cell — the opposite of the usual
  "underpowered small sample" intuition. Always re-check paired significance after extending
  capped cells, don't assume the direction of the correction.
- Everything in the 2026-09-06 handover's §6 (label-space traps, `make_pc`'s raw path, mutating
  `handle["freeze"]`, column-freeze index hashability, `savefig.bbox`, `mpl.colormaps`, `_acc`
  returning fractions, don't pipe long runs through `tail`, 801 running to saturation on
  purpose, matched-competence confounding per-update quantities, 805's unequal task-1 phase
  lengths) — all still true, not re-litigated here.

## 6. Next actions, in order

1. Check both background jobs (§3). Re-verify 342 with `--replot`.
2. Run the four queued items in §4, in order.
3. Confirm seed 13's slowness resolves cleanly post-extension across all five affected scripts,
   or flag it as something more than a slow seed if it doesn't.
4. Decide what to do about 806's masking-impairs-task-2 finding — does R5's write-up need to
   change, or is this an appendix caveat?
5. Full walkthrough of every figure against the report's story (the user's explicit next ask) —
   hold until 1–4 land so the walkthrough isn't immediately stale.
6. Then: results.tex's caption pass (user's own task, not this session's).
