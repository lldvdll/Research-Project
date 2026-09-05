# 300 series log

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
