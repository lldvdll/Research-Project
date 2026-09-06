# Handover — 2026-09-06

Everything needed to pick this up cold. Read `now.md` first (one screen), then this.

---

## 1. Where the project is

The report structure is **settled** and the figure list is **frozen**. Implementation has started.

- **The track**: `report_track.md` — the **question** order, as eleven cumulative tiers. It was
  never rewritten into section order, and an earlier version of this handover wrongly said it
  had been. It now opens with a tier → section mapping table instead; the **section** order
  (M1–M3 / R1–R5 / D1, 18 main-text figures, costed against 8,000 words) is authoritative in
  `script_plan_800_900.md` and in `report/sections/*.tex`.
- **The audit page**: <https://claude.ai/code/artifact/10978e55-0449-4f55-beb7-b6464030ac07>
  (versioned in `drafts/`, currently 007). Shows every figure slot, its status, and an honest
  verdict per section.
- **The script plan**: `script_plan_800_900.md` — the 800/900 split, the many-to-many mapping,
  and the thirteen 800 slots deliberately left unwritten.
- **The verified experiment index**: `progress.md` — every experiment 01–347, re-derived from
  saved arrays. **This is ground truth for any number.** Where a note and an array disagreed,
  the array won.

## 2. The 800/900 architecture

**800 trains and saves arrays. It never draws. 900 loads and draws. It never trains.**

- A 900 declares its source paths as constants at the top, each with a comment naming the
  protocol that produced it. Repointing a legacy array to a fresh 800 array is a one-line edit.
- **One result, one plot, one file** by default; the report composes with LaTeX `subfigure`.
  Exception: panels that must SHARE AN AXIS for the comparison to read (911, 912, 917, 921, 923)
  stay in one file, take one caption, and are lettered by `style.panel_label`.
- **No shared harness.** `CLAUDE.md` records one was deliberately deleted.
- All styling is in `src/style.py`. Nothing else sets a font, colour or figure size.

## 3. Status of every run

| | State | Notes |
|---|---|---|
| `config_800.yaml` | **done** | backprop lr 0.01→0.02 (314 licenses it); both settle controls recorded |
| **801** definitional | **done**, 5 s | task 1 → 0.0% on all five classes |
| **802** activation sweep | smoke-tested, **queued** | run it when the chain clears |
| **803** mechanism-logged | **done**, 451 s | feeds 915, 916, 923, 924 |
| **804** repeated alternation | **running** | see §5 — result contradicts the plan |
| **805** joint→sequential | queued in chain | |
| **806** partial-column freeze | queued in chain | smoke landed on branch 2 — see §5 |
| **900 series** | **901, 911, 912, 913 done** | commit `0784c49`. Next: 917 (needs 804), 922 (needs 806) |
| **LaTeX skeleton** | **done** | commit `731f90e`. Builds clean, 8 pages, live figures already in |

Background job `bdd3yf7y1` runs `804 && 805 && 806`. Output:
`…/tasks/bdd3yf7y1.output`. **Check it before doing anything else.**

## 4. `src/` changes made this session (all committed)

| commit | change |
|---|---|
| `0436120` | **`src/style.py`** — new. rcParams, colour groupings, panel sizes. Draws nothing. |
| `05abca1` | **column-wise freezing** across `model.py` / `methods.py` / `predictive_coding.py` |
| `daf9033` | **`linear_probe_fn`** in `probes.py` |

Verified: with no column entries, 15 steps of backprop and PC are **bit-identical** to the
pre-change code (via `git stash`, abs-sum of every tensor to 12 s.f.).

**`src/probes.py` already had far more than expected** — `alignment_probe` (S&B Fig 3b, with a
`ref=` mode measuring alignment on task-1 data *during* task 2), `weight_trace_probe` with
`keys=("W1","W2")`, `weight_path_probe`, `code_snapshot`/`code_drift`, `output_unit_stats`.
Two `src/` changes originally planned were unnecessary. **Read `probes.py` before writing
anything new.**

## 5. Results found this session that change plans

### 804 contradicts R4's central prediction
The plan predicted Class-IL traces a **closed loop** under repeated alternation while Domain-IL
**spirals in**. On block lengths and end-of-run accuracy, Class-IL is *also* converging:

```
backprop seed 18  blocks [530, 120, 70, 60, 50, 60, 50, 40, 70, 40]  end t1 46.3
pc       seed 10  blocks [390, 850, 200, 110, 90, 80, 60, 70, 60, 60] end t1 49.3
```

Against ~5% after a single switch. **But it is not uniform** — pc seed 13 hit the 5000 cap
twice (`[400, 5000, 250, 5000, …]`, reached 8/10) and ended at t1 0.2. Hold judgement until 917
draws the actual trajectory: block length and end-of-block accuracy are not trajectory
contraction. If it holds, the honest claim is that the scenarios differ in **how fast** they
find a joint solution, not whether they can — which is more interesting than the prediction.

### 806 landed on its second pre-committed branch — CONFIRMED AT TEN SEEDS
Final task-1 accuracy, Class-IL, 10 seeds (all cells but `mask`/pc complete):

| condition | backprop | pc |
|---|---|---|
| control | 4.8 | 6.0 |
| freeze all of W2 | 3.2 | 4.3 |
| **freeze task-1 columns of W2 + b2** | **6.7** | **6.3** |
| **mask** | **50.9** | *running* |

Masking recovers **+46 points**. Freezing exactly the weights masking spares recovers **+1.9**.
So **masking does not work by sparing the task-1 output weights** — the pre-committed second
branch, now on the full seed block rather than smoke. The remaining candidate is that masking
changes what **W1** learns, the trunk reshaped to serve the suppression objective, which is not
a readout account at all. **R5 must be rewritten**; `report/sections/results.tex` already carries
that instruction in its R5 comment block.

Note `mask`'s crossover is censored on 3/10 seeds — with masking, task 1 never falls below
task 2 there, which is the *best* outcome and must be ranked, not dropped.

**923 points the same way independently.** The linear probe reads 82.6% on task-1 classes where
argmax reads 21.1 — but it reads **80.2% on the untrained network**, so only ~2.4 points are
attributable to what the trunk learned. Two independent lines now argue against "forgetting is a
readout failure over an intact representation".

### 803 has a checkpoint-spacing defect — cheap to fix, worth fixing
Its interval is `(2 * max_iters_per_task) // CHECKPOINTS` = `10000 // 20` = 500 updates, which
divides the **budget**, not the run. Matched competence finishes runs in 700–2500 updates, so
each gets **2–5 checkpoints instead of ~20**, and the last sits a median ~175 updates before the
end. argmax on task 1 reads 21.1 at that last checkpoint against a true final of 4.8, so 923
**understates** the end-of-training gap. Re-running 803 with the interval derived from run length
costs ~8 min and also sharpens 915, 916 and 924. Not done — it is a `src/`-adjacent change to a
committed 800 script and needs approval.

### The linear probe has a high floor
At H=32, Class-IL, task-1 classes: probe reads **81.8% untrained**, 86.2% after task 1, against
argmax 12.2 → 91.8. NCM reads 69.0 → 80.4. So the probe beats NCM by 12.8 points at init — the
case for replacing it — but only ~4 points of its range are attributable to training the trunk.
**923 must draw the random-init floor** or it over-claims exactly as NCM did. 803 records it.

### 805: joint pre-training protects partially
Smoke: Class-IL joint-start retains 6.9 / 25.6 / 12.5 / 20.3 against scratch 1.5 / 6.2 / 3.1 /
9.1 — consistently better, still collapsing from ~77%. Forgetting is not simply a failure to
*find* a joint solution.

## 6. Traps found — check these before writing code

1. **Label space.** `run_joint`, `alignment_probe(ref=)` and `linear_probe_fn` all index output
   **units**; the data carries raw class labels 0–9. Class-IL hides it (`lmap` is None, class ==
   unit); **Domain-IL raises**. Smoke every new script on Domain-IL first. `803` and `805` each
   carry a `_to_units` helper.
2. **`make_pc` uses the RAW path.** `opt=None` whenever `optimizer="sgd"`, which is every run
   here. Anything touching PC's weight update must handle that path, not just the torch one.
3. **`handle["freeze"]` must be MUTATED, not reassigned** — `handle["freeze"].add(...)`. The
   closure holds the original set object; `handle["freeze"] = {...}` silently does nothing.
4. **Column freeze indices must be hashable** — `("W2", (0,1,2,3,4))`, not a list. `freeze` is a
   set.
5. **`savefig.bbox="tight"` breaks fixed sizing.** A 3.44 in request saves as 3.11 in and the
   crop varies with content. `style.py` uses constrained layout and `bbox="standard"`.
6. **`mpl.cm.get_cmap` was removed in matplotlib 3.9**; this repo runs 3.11. Use
   `mpl.colormaps[...]`.
7. **`_acc` returns fractions, not percent.** Everything 803 saves is already ×100.
8. **Don't pipe a long background run through `tail`** — it buffers until exit and you lose all
   progress visibility.
9. **801 runs to saturation on purpose** (task 1 → 0.0%) where R1 reports 5.4% at matched
   competence. The caption must say so or the report looks self-contradictory.
10. **Matched competence makes task-2 length a DEPENDENT VARIABLE, and it confounds any
    per-update quantity.** Task-2 length runs 119 → 4999 updates in 803. Long runs have a small
    *mean* per-update ‖ΔW‖ **and** forget more, so mean step size correlates with retention at
    **r = +0.93** — which reads as "bigger updates preserve task 1" and is an artefact. The
    **total path** summed over task 2 has no such problem and gives r = −0.56, the interpretable
    sign. 916 claims only on totals. Check this before correlating anything per-update.
11. **805's two arms do not train task 1 for the same number of updates.** The joint-initialised
    arm already knows task 1, so it clears the threshold far sooner — 240 vs 30 steps on
    Class-IL seed 10. Any "joint pre-training protects" claim in 919 must be read against that,
    or it is partly "the joint arm trained task 1 less and so had less to lose". The per-arm
    task-1 phase lengths are printed by 805 and are in its array.

## 7. Next actions, in order

Steps 1–3 below are **done** (commits `0784c49`, `731f90e`) and are kept so the order is legible.

1. ~~Write 911, 912, 913~~ — done. Legacy arrays only; every number re-derived matches
   `progress.md` (PC−BP +1.42 ± 0.39 Class-IL, −0.72 ± 0.19 Domain-IL; replay +9.84, +3.43).
2. ~~Write 901~~ — done, from `801_definitional_run.npz`.
3. ~~LaTeX skeleton~~ — done. `\graphicspath` points at `../experiments/` rather than copying
   PNGs into `report/figures/`, so a 900 re-run updates the report with no sync step. `\fig{}`
   draws a labelled placeholder for a figure that does not exist yet, so it builds now.
4. **Check `bdd3yf7y1`.** When it clears, run `802_activation_sweep.py` (~1 h). Do **not** run it
   alongside the chain — CPU contention would roughly double both.
5. **Write 917** from 804 and settle the spiral-versus-loop question (§5).
6. **Write 922** from 806 and settle the masking question (§5).
7. **Write 915, 916, 923, 924** — all four read `803`, which has already run.

### Figure conventions now fixed by the four that exist

- A Mode B script sizes itself from `style.WIDTH["page"]` divided by the number of columns, not
  from `half_page`. Verified saved sizes: 911 is 7.060 × 6.353 in, 912 is 7.058 × 4.048 in,
  913a is 3.180 × 1.970 in.
- `aspect="equal"` breaks the usual `figsize = w × ratio` arithmetic, because constrained layout
  shrinks the axes to whatever square fits and pads the rest. 911 needed 0.90, found by measuring.
- A figure saved at 7.06 in **cannot** go in a `figure` — it needs `figure*`. 913 is a `figure*`
  for exactly this reason.
- **Palette collision, not yet fixed:** `style.TASK[0]` and `style.RULE["pc"]` are the same hex
  (`#d1682a`). In 911 orange means *task 1*; in 913 orange means *PC*. Both are correct per
  `style.py`, but across the report it is ambiguous. Fixing it is a one-line `src/` change and
  needs approval — it was not made silently.

## 8. Open questions for the user

- **Do captions count against the 8,000 words?** If not, several appendix demotions in
  `report_track.md` become unnecessary. This changes the figure count.
- **802 is the only genuinely new scientific question** in the 800 list; everything else
  instruments or re-runs. If time is short it is the first cut, and tanh becomes a declared
  limitation.

## 9. Uncommitted work not from this session

`src/metrics.py` (`paired_wilcoxon`, `classify_settle`) and the whole **340-series** (314,
340–347) are uncommitted, from a parallel session. `progress.md`, `300_series_log.md` and
`narrative_plan.md` also have uncommitted edits. **Not mine to commit** — left alone deliberately.
