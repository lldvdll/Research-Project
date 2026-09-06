# Handover — 2026-09-06

Everything needed to pick this up cold. Read `now.md` first (one screen), then this.

---

## 1. Where the project is

The report structure is **settled** and the figure list is **frozen**. Implementation has started.

- **The track**: `report_track.md` — nine sections as Methods (M1–M3) / Results (R1–R5) /
  Discussion (D1), costed against 8,000 words. 18 main-text figures.
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
| **900 series** | **none written yet** | start with 911, 912, 913 — legacy arrays only |

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

### 806 landed on its second pre-committed branch
Smoke, 2 seeds: control 3.8, freeze-all-W2 6.4, **freeze task-1 columns 7.2**, **mask 65.0**.
Freezing the task-1 readout weights does **not** reproduce masking. Adding the task-1 *biases*
to the freeze moved it 6.5 → 7.2, so that was not the explanation either.

If this survives ten seeds, **masking does not work by sparing those weights**. The remaining
candidate is that masking changes what **W1** learns — the trunk reshaped to serve the
suppression objective — which is not a readout account at all and would reframe R5.

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

## 7. Next actions, in order

1. **Check `bdd3yf7y1`.** When it finishes, run `802_activation_sweep.py` (~1 h).
2. **Write 911, 912, 913** — the three headline Results figures. They read **legacy arrays only**
   (310/332 for 911; 340/341/342 for 912 and 913), so they need nothing that is still running.
3. **Write 901** from `801_definitional_run.npz`, which exists now.
4. **Write 917** from 804 and settle the spiral-versus-loop question (§5).
5. **Write 922** from 806 and settle the masking question (§5).
6. **LaTeX skeleton** — headings for M1–M3 / R1–R5 / D1, `figure`/`figure*` stubs pointing at
   `report/figures/9NN_*.png`, and the Methods parameter table. Add `\usepackage{subcaption}`
   and a `\panelwidth` length; `\singlefigure = 0.45\textwidth` already exists.

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
