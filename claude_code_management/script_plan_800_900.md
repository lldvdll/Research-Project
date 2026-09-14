# The 800/900 script plan

The final figure set for the report, as two decoupled series.

**800 = training runs.** Generate data, save `.npz`, never draw a figure.
**900 = plots.** Load `.npz`, draw one report figure, never train.

Once settled, **this list does not change**, except for results that depend on an experiment that
has not run yet.

---

## STATUS — 2026-09-11: every run and every plot in this list has been produced

All 800-series training in THIS LIST has run and all 900-series figures in it exist as PNGs on
disk, checked directly rather than assumed. ⚠ The report-revision pass has since added runs and
figures beyond this list — 810, 811 and 910 at least. `now.md` and `report_progress.md` are
authoritative for what is outstanding; this file is the inventory, not the status. What remains is **writing**, not computing:
`report/sections/*.tex` still carries stale `\gap{...not written}` captions for figures that now
exist, and that is the user's own pass. `plot_walkthrough.md` reads the whole set in report order
and is the place to start.

Three additions arrived after the list was frozen and are recorded below rather than silently
folded in: **809** (a training run, absent from the original 800 table), **909**, and the
already-noted **927/928**. One collision is open and needs a decision — see the 807 note.

---

## The contract

- A 900 script declares its source paths as **constants at the top**, each with a comment naming
  the protocol that produced it. Repointing from a legacy array to a fresh 800 array is then a
  one-line edit and a re-run that takes seconds.
- **One result, one plot, one file — by default.** A 900 script writes `9NN_name_a.png`,
  `9NN_name_b.png`, … and the report composes them with LaTeX `subfigure`, each with its own
  subcaption. This **overrides `CLAUDE.md`'s "one figure per script"** for the 900 series only;
  the 800s draw nothing at all.
- **Exception — shared axes, where the comparison depends on a common scale.** Then the panels
  stay in one file with `sharex`/`sharey`, take **one** LaTeX caption, and are lettered in-figure
  by `style.panel_label`. Hand-matching limits across separate files is how inconsistency gets in,
  so where comparability is the point, matplotlib should enforce it.

| Mode | Files | LaTeX | Labelling |
|---|---|---|---|
| **A — default** | one PNG per result | `subfigure`, one subcaption each | `\subcaption` |
| **B — shared axes** | one PNG, several panels | one `figure`, one caption | `style.panel_label` |

**Mode B figures, and why each earns it:** **911** (2×2 scenario × rule — the claim *is* that
columns differ and rows do not, so both axes must match); **912** (1×3 lr/width/depth — different
x-axes, one shared PC−BP y with a zero line); **917** (1×2 scenario — spiral versus loop is
unreadable unless the axes are identical); **921** (1×2 scenario — same retention-% x-axis, the two
distributions *are* the comparison); **923** (2×2 task × scenario — all accuracy %, and the
argmax−probe gap is read across them). Everything else is Mode A, including 918, whose PCA bases
are not comparable across scenarios anyway.
- **Styling is global, in `src/style.py`.** Font, sizes, colour groupings and figure dimensions are
  defined once, so a palette or font change is one edit that re-renders every figure. The module
  sets `rcParams` and names colours — it draws nothing, so it is not a harness.
- **Figures are saved at final physical size.** `style.size("half_page")` and friends return
  inches matched to the document's column widths. If LaTeX scales an image it scales the fonts
  with it, and an 8 pt label lands as 5 pt.
  ⚠ This is why `savefig.bbox` is **`"standard"`, not `"tight"`**, with constrained layout doing
  the fitting instead. Measured: a 3.44 in request saves as **3.11 in** under `bbox="tight"`,
  because tight crops to drawn content — and the crop varies with content, so a grid of panels
  would each be scaled by a different factor and end up with different font sizes. Constrained
  layout returns exactly 3.440 in whatever the labels are.
- **No shared harness.** `CLAUDE.md` records that one was deliberately deleted; each 900 stands
  alone and imports only from `src/`.

### Panel sizes

| Name | Inches | Use |
|---|---|---|
| `half_col` | 1.55 | two panels across inside one column — avoid, too small for axes |
| `col` | 3.18 | one panel filling a column (`= \singlefigure`, already in `main.tex`) |
| `half_page` | 3.44 | two panels across a `figure*` spanning both columns — **the default for grids** |
| `page` | 7.06 | one panel spanning both columns |

Any multi-panel figure (911's 2×2, 917's two scenarios) is a `figure*` at `half_page` per panel.
That needs `\usepackage{subcaption}` and a `\panelwidth` length added to `report/main.tex`.
- Every 900 reading legacy data carries a `PROVENANCE` comment stating the protocol, seed block
  and any known caveat. This is the hybrid's one real risk — the legacy arrays span several
  protocols, and `progress.md` §40–347 documents where they differ.
- 800 numbering is by **dataset**; 900 numbering is by **report figure position**. The mapping is
  many-to-many, so a strict 1:1 would force artificial splits.

---

## 800 — training runs that must actually run

Six new runs. **Four of the seven P1 figures come from 803 alone**, which is why it is the first
thing to write.

| # | Run | What it produces | Feeds | Cost |
|---|---|---|---|---|
| **800** | `config_800.yaml` | Control parameters for the whole series, as `config_300.yaml` does now. Adopts backprop lr = 0.02 (314 shows it is free), pc lr = 0.02, dt = 0.4 adaptive at H=32/depth 1, **dt = 0.2 whenever width or depth varies**. | all | — |
| **801** | Definitional single run | Class-IL, **one seed, fixed budget**, backprop, logging **per-class** accuracy (not per-task) so all ten classes can be drawn. | 901 | minutes |
| **802** | Activation sweep | tanh / sigmoid / ReLU × both scenarios × bp + pc, 10 seeds. Closes the one setup choice with no justification at all. | 903 | ~1 h |
| **803** | **Mechanism-logged sequential** | The standard 2-task run, bp + pc, both scenarios, 10 seeds — instrumented. Per update: settling displacement `D`, per-layer ‖ΔW‖. At ~20 checkpoints: weight snapshot (for target alignment, computed post-hoc against the final state) and a **linear probe fitted online** on the held-out set. | 915, 916, 923, 924 | ~3 h |
| **804** | Repeated alternation | 20 alternations, **both scenarios**, bp + pc, **matched competence per block** (68 ran fixed budget), 10 seeds. Stores per-layer weight trajectories — W1 **and W2**, 68 stored only W1. ⚠ **Rerun 2026-09-10 with `TRACE_SEEDS` raised 3 → 10**, so every seed now carries a full weight trajectory, not an illustrative three. Accuracy arrays are unchanged (same seeds, deterministic); only trace coverage grew. 44 min. | 917, 918, 924 | ~4 h |
| **805** | Joint → sequential | Train on the joint distribution to convergence, then run the sequential protocol from there, against sequential-from-scratch. Both scenarios, bp + pc, 10 seeds. | 919 | ~2 h |
| **806** | Partial-column freeze | Freeze **only the task-1 columns of W2** during task 2, against freeze-all-W2, masking, and the unfrozen control. Class-IL, bp + pc, 10 seeds. | 922 | ~1 h |
| **807** | Task-1 overtraining | Train task 1 to threshold (N steps), then continue on ONE trajectory to 1.5N/2N/**3N**/4N, branching an independent task-2 phase from each snapshot. Both scenarios, bp + pc, 10 seeds (+ 2-seed weight trace, both methods). Added 2026-09 — not in the original plan; tests whether retention depends on task-1 consolidation time, the outer-loop analogue of PC's own settle-step question (330-334). The 3.0× point was added after the first pass, which made a dip look real that was not. | 927, 928 | ~1.5 h |
| **809** | Probe on the interventions | 806's four conditions (control, freeze_w2, freeze_w2_t1, mask) × bp + pc, re-run with `linear_probe_fn` fitted at the end, against an untrained-probe floor. Added 2026-09, **absent from the original 800 table**. Answers the one question 922 and 923 leave open: does masking change what the trunk represents, or only what argmax can read out of it. 1 h 47 min. | R5 prose; no figure of its own yet | ~1.8 h |

**803's storage discipline matters.** Storing raw hidden codes at every checkpoint would run to
hundreds of MB. Fit the probe *online* at each checkpoint and store only its accuracy; subsample
weight snapshots to ~20 per run. Everything then stays in the tens of MB.

**804 note.** Matched competence per block makes block length a *dependent variable*. That is
protocol-consistent and it is itself a result — whether relearning accelerates across blocks is
the "gradual relearn" shape from the four-category question.

**⚠ 807 IS CLAIMED TWICE — OPEN, NOT FIXED.** Two different 800-series training runs both carry
the number, both with real `.npz` on disk and both read by a live 900 script:

| File | Question | Arrays | Read by |
|---|---|---|---|
| `807_code_drift.py` | Where does Domain-IL's damage sit — does the hidden code drift, and for which units | `807_code_drift_{scenario}.npz` | `924b_hidden_code_drift.py` |
| `807_overtrain_task1.py` | Does retention improve if task 1 trains past threshold | `807_overtrain_task1_{scenario}.npz` | `927`, `928` |

The filenames differ so nothing collides *on disk* and no figure is wrong — this is a bookkeeping
break, not a data break. But "807" now names two experiments in conversation and in every
docstring cross-reference, which is exactly the ambiguity the numbering exists to prevent.
Renaming touches a script, its two array files and 924b's reader, so it needs approval and is
**deliberately left alone here**. Suggested resolution when someone picks it up: the code-drift
run predates the overtraining one and already feeds a 924-series figure, so give the *overtraining*
run the new number and leave code drift at 807.

### 800 slots left unwritten — a 900 loads the legacy array instead

If time allows at the end, write these and repoint the named 900 scripts. Until then they are
placeholders, not files.

| # | Would re-run | Legacy source | Protocol caveat |
|---|---|---|---|
| 810 | Capacity vs width | `100_capacity_vs_width` | current protocol, 10 seeds — clean |
| 811 | Frozen-trunk ceiling | `101_problem_complexity` | current protocol — clean |
| 812 | Output spec × masking | `120_output_maths_and_masking` | backprop only, no PC arm |
| 813 | Metric vs stopping threshold | `111_metric_sensitivity_to_threshold` | backprop only — which is the point |
| 814 | Metric vs lr, with censoring | `343` / `340_lr_sweep` | current protocol, 10 seeds — clean |
| 815 | Metric survival across thresholds | `112_which_metric_survives` | paired PC−bp; appendix only |
| 816 | Core sequential, bp + pc | `310` + `332` | **backprop arm at lr 0.01**, PC at 0.02; 314 shows lr 0.02 changes nothing (all p > 0.34) |
| 817 | lr / width / depth sweeps | `340` / `341` / `342` | current protocol, dt = 0.2 for 341/342 — clean |
| 818 | Freeze factorial | `130_freeze_factorial` | current protocol, 10 seeds — clean |
| 819 | Class-IL at 70 seeds | `313_class_il_scale_check` | seeds 20–69, backprop only |
| 820 | Pair similarity | `301` + `311` | two independent seed blocks; the non-replication *is* the result |
| 821 | Settling dt and stability | `334` + `347` | single seed traces — sufficient for what they claim |
| 822 | Added mechanisms | `200` / `210` / `220` | ⚠ EWC's λ grid never bracketed its optimum |

---

## 900 — one script per report figure

`M` = Methods, `R` = Results, `D` = Discussion, `A` = appendix. **17 main-text figures.**

| # | Where | Figure | Source | Built | Pri |
|---|---|---|---|---|---|
| **901** | M1 | What forgetting looks like — one seed, ten class lines, crossover marked, forgetting/learning arrows | 801 | ✔ done | P2 |
| 902 | M2·A | Architecture justification — trunk power and width | 811, 810 | ✔ done (copy 101 + 100) | P3 |
| 903 | M2·A | Specification — activation and output maths | 802, 812 | ✔ done — **scope decision open**, see walkthrough | P2 |
| **904** | M2 | Settling reaches the same fixed point across dt | 821 | ✔ done (copy 334) | P3 |
| 905 | M2·A | The stable dt band shrinks with depth | 821 | ✔ done (copy 347) | P3 |
| 906 | M2·A | What pairing on seeds buys | legacy 69 | ✔ done | P3 |
| **907** | M3 | Why competence-matched stopping — endpoint fails one way, crossover the other | 813, 814 | ✔ done (merge 111 + 343) | P2 |
| 908 | M3·A | Metric survival across stopping points | 815 | ✔ done (copy 112) | P3 |
| 909 | M3·A | Crossover vs. retention, pooled — correlated but not redundant | 340/341/342, 802 | ✔ done, added 2026-09 — **slot decision open** | P2 |
| **911** | R1 | **Two forgetting phenotypes** — 2×2, scenario across, rule down | 816 | ✔ done | P1 |
| **912** | R2 | **PC − backprop across lr, width and depth**, zero line, both scenarios | 817, 802 | ✔ done — **4 panels**, activation added | P1 |
| **913** | R2 | Replay on the same axis — the scale of a real fix | 817 | ✔ done | P1 |
| 914 | R2·A | The three sweeps behind 912 | 817 | ✔ done (copy 340/341/342) | P3 |
| **915** | R3 | **Target alignment** — through training, and against retention | 803, 808 | ✔ done — **3 panels**, sigmoid added | P1 |
| **916** | R3 | **Settling displacement → ‖ΔW‖ → retention**, with 344 as a panel | 803, 808 | ✔ done — **3 subplots**, sigmoid added | P1 |
| **917** | R4 | **Repeated alternation geometry** — spiral vs loop, both scenarios | 804 | ✔ done | P1 |
| **918** | R4 | **Weight-space PCA** under alternation, W1 and W2 | 804 | ✔ done — **both rules**, + 15 diagnostic files behind flags | P1 |
| **919** | R4 | **Joint pre-training then sequential** | 805 | ✔ done | P1 |
| **920** | R4 | Which measurements separate by scenario — tie-out | many | ✔ done | P2 |
| **921** | R5 | **Retention distribution, and what moves it** — interventions overlaid | 819, 818, 812 | ✔ done — backprop-only by data availability | P2 |
| **922** | R5 | **Partial-column freeze** vs freeze-all vs masking | 806 | ✔ done | P1 |
| **923** | R5 | **Trained linear probe vs argmax**, both tasks, both scenarios | 803 | ✔ done — backprop-only, checked not assumed | P1 |
| **924** | R5 | Per-layer weight path and hidden-code drift | 803, 804, 807-code-drift | ✔ done (924 + 924b) | P1 |
| **925** | R5 | Does task structure predict what survives — both scenarios | 820, 819 | ✔ done | P2 |
| 926 | R5·A | Class-IL digit table | 819 | ✘ **not built** — the one gap, appendix only | P3 |
| 927 | R4 | Retention/crossover vs. task-1 overtraining, both scenarios × both rules | 807-overtrain | ✔ done, added 2026-09 | P2 |
| 928 | R4 | Forgetting-shape classification (collapse / delayed / partial / rising / noisy) across the 2×2 | 803 | ✔ done, added 2026-09 | P2 |
| 929 | R4 | First five PCs over training updates, task-shaded — does the oscillation damp or migrate? | 804 | ✔ done, added 2026-09-13 | P2 |
| **930** | R3 | **Benefit against the mechanism S&B credit** — paired Δcrossover over paired Δalignment, three conditions | 802, 803, 808 | ✔ done, added 2026-09-13 | P1 |
| **931** | D1 | Intervention ranking, both scenarios | 822 + others | ✔ done — regrouped 2026-09-10 | P2 |
| 932 | D1·A | SI and k-WTA sweeps | 822 | ✔ done (copy 210 + 220) | P3 |
| **933** | R3 / D | **What determines retention if not the rule** — (a) bp vs pc retention per seed on the identity line, (b) D→retention raw vs split-difficulty removed, (c) seed effect against rule effect in one unit | 803, 808 | ✔ done, added 2026-09-14 | P1 |

**Bold = main text.** Count: 901, 904, 907, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922,
923, 924, 925, 931 = **18 main-text figures**, of which 3 are Methods and 1 Discussion, leaving
**14 in Results**. At ~3,000 Results words that is ~215 words each — workable, and one figure
above the earlier estimate because R5 kept the task-structure figure rather than demoting it.

**⚠ The budget is now over-subscribed, and that is a decision, not an oversight.** Five figures
arrived after the count above was struck — **909**, **927**, **928**, **930**, **933** — and all
five are built. 930 and 933 are the two mechanism figures the user asked for on 2026-09-13 after
saying the mechanism story was not yet coherent; between them they replace prose that R3 could
not otherwise carry, so they arrive with a claim on R3's existing slots (915, 916) rather than as
additions to them. **The user's ruling on 2026-09-14: figures are chosen AFTER the section's
bullet points exist, because figures support claims and not the other way round. Do not settle
the budget before the text does.**
927 and 928 are real R4 evidence (consolidation-time sensitivity; forgetting-shape
classification) and have a good claim on main-text slots; 909 justifies dual-metric reporting and
reads as a Methods figure that could merge into 907/908 rather than stand alone. Taking 927 and
928 in at 18 means two current entries move to appendix. Candidates, weakest case first: **920**
(tie-out re-analysis), **925** (task structure — a non-replication), **914** (already appendix).
`plot_walkthrough.md` carries the argument for each; the call has not been made.

**918 writes more than one file, by design.** The report figure is `918_weight_space_pca.png` and
nothing about it changed. Behind two flags it also writes 15 diagnostic files that are **not**
report figures and are not counted above: `--per-seed` gives one full grid per traced seed
(`_seed10` … `_seed19`, all on one global axis range), and `--consolidated` gives four
whole-set grids (`_consolidated`, `_consolidated_flat`, `_consolidated_grid`,
`_consolidated_grid_flat`) that differ only in how ten seeds are drawn and coloured. They exist
because a single-seed panel could not settle whether Domain-IL/pc/W2's odd shape was real; see
the R4 entry in `plot_walkthrough.md` for what they established.

---

## src/ changes required — need approval before 800 starts

Four, each small, each proposed separately with a diff as `CLAUDE.md` requires.

0. **`src/style.py`** (new, for every 900). rcParams, colour groupings, panel sizes. Draws
   nothing. Blocks all plotting work, so it is proposed first.
1. **`_apply_freeze` column masking** (for 806). Currently whole-matrix. Needs to accept a column
   index set so only the task-1 output columns are frozen.
2. **Per-layer ‖ΔW‖ and weight snapshots in the diag channel** (for 803, 804). `displacement` is
   already published; this adds the same pattern for weight-step norms and an optional snapshot
   hook.
3. **Online linear probe** (for 803). Fit a logistic/least-squares readout on the held-out set at
   a checkpoint and return its accuracy. Small, self-contained, and it replaces NCM everywhere.

### One palette collision to be aware of

Script 112 draws Domain-IL in green; 340/341/342 draw **replay** in green. `style.py` moves
Domain-IL to teal so a figure carrying both rules and scenarios cannot be misread. Until 112 is
redrawn as 908, its existing PNG disagrees with the new palette.

---

## Implementation order

1. **Approve the three `src/` changes.** Everything else waits on 1 and 3.
2. **Write and launch 803**, then 804. Together ~7 hours and they feed six figures.
3. **802, 805, 806, 801** while those run.
4. **900 scripts**, starting with the ones that need no new data — 911, 912, 913 are P1 and read
   only legacy arrays, so the three headline Results figures can exist before any 800 finishes.
5. **LaTeX skeleton** updated in parallel: section headings matching M1–M3 / R1–R5 / D1, figure
   environments with placeholder `\includegraphics` paths under `report/figures/9NN_*.png`, and
   the Methods parameter table stubbed with its rows. Figures then drop in as they arrive.

---

## Still open — now that everything has run

- **Do captions count against the 8,000 words?** If not, a caption can carry ~100 words of "what
  was done / what it shows" per figure and several appendix demotions above become unnecessary.
  This is the cheapest way to relieve the over-subscribed budget and it is still unanswered.
- **The figure budget** — 909/927/928 are built and the count is at 21 against a frozen 18. Two
  demotions needed; candidates listed under the count above.
- **807's double booking** — see the collision note under the 800 table.
- **926 is the one unbuilt figure** in the list (Class-IL digit table, copy of 312, appendix, P3).
- **903's scope** — does Methods still carry the raw activation sweep now that 912/915/916 own
  the sigmoid story in Results, or does it point forward instead?
- **924 vs 916(b)** — both now speak to per-layer weight movement; check for overlap before the
  captions are written.

~~**802's activation sweep** is the first cut if time is short~~ — **resolved, and it went the
other way.** It ran, and it produced the largest standardized effect in the project (PC−backprop
crossover flips positive under sigmoid in Domain-IL, d = 1.25). Far from being the cut, it forced
912, 915 and 916 to each grow a panel, and it is now load-bearing for R2 and R3.
