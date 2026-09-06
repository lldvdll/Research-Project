# The 800/900 script plan

The final figure set for the report, as two decoupled series.

**800 = training runs.** Generate data, save `.npz`, never draw a figure.
**900 = plots.** Load `.npz`, draw one report figure, never train.

Once settled, **this list does not change**, except for results that depend on an experiment that
has not run yet.

---

## The contract

- A 900 script declares its source paths as **constants at the top**, each with a comment naming
  the protocol that produced it. Repointing from a legacy array to a fresh 800 array is then a
  one-line edit and a re-run that takes seconds.
- **No shared harness.** `CLAUDE.md` records that one was deliberately deleted; each 900 stands
  alone and imports only from `src/`.
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
| **804** | Repeated alternation | 20 alternations, **both scenarios**, bp + pc, **matched competence per block** (68 ran fixed budget), 10 seeds. Stores per-layer weight trajectories — W1 **and W2**, 68 stored only W1. | 917, 918, 924 | ~4 h |
| **805** | Joint → sequential | Train on the joint distribution to convergence, then run the sequential protocol from there, against sequential-from-scratch. Both scenarios, bp + pc, 10 seeds. | 919 | ~2 h |
| **806** | Partial-column freeze | Freeze **only the task-1 columns of W2** during task 2, against freeze-all-W2, masking, and the unfrozen control. Class-IL, bp + pc, 10 seeds. | 922 | ~1 h |

**803's storage discipline matters.** Storing raw hidden codes at every checkpoint would run to
hundreds of MB. Fit the probe *online* at each checkpoint and store only its accuracy; subsample
weight snapshots to ~20 per run. Everything then stays in the tens of MB.

**804 note.** Matched competence per block makes block length a *dependent variable*. That is
protocol-consistent and it is itself a result — whether relearning accelerates across blocks is
the "gradual relearn" shape from the four-category question.

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

| # | Where | Figure | Source | Status | Pri |
|---|---|---|---|---|---|
| **901** | M1 | What forgetting looks like — one seed, ten class lines, crossover marked, forgetting/learning arrows | 801 | new | P2 |
| 902 | M2·A | Architecture justification — trunk power and width | 811, 810 | copy 101 + 100 | P3 |
| 903 | M2·A | Specification — activation and output maths | 802, 812 | new + copy 120 | P2 |
| **904** | M2 | Settling reaches the same fixed point across dt | 821 | copy 334 | P3 |
| 905 | M2·A | The stable dt band shrinks with depth | 821 | copy 347 | P3 |
| 906 | M2·A | What pairing on seeds buys | legacy 69 | new | P3 |
| **907** | M3 | Why competence-matched stopping — endpoint fails one way, crossover the other | 813, 814 | merge 111 + 343 | P2 |
| 908 | M3·A | Metric survival across stopping points | 815 | copy 112 | P3 |
| **911** | R1 | **Two forgetting phenotypes** — 2×2, scenario across, rule down | 816 | assemble 310 + 332 | P1 |
| **912** | R2 | **PC − backprop across lr, width and depth**, zero line, both scenarios | 817 | new | P1 |
| **913** | R2 | Replay on the same axis — the scale of a real fix | 817 | new, replay own scale | P1 |
| 914 | R2·A | The three sweeps behind 912 | 817 | copy 340/341/342 | P3 |
| **915** | R3 | **Target alignment** — through training, and against retention | 803 | new (re-run of 64) | P1 |
| **916** | R3 | **Settling displacement → ‖ΔW‖ → retention**, with 344 as a panel | 803 | new | P1 |
| **917** | R4 | **Repeated alternation geometry** — spiral vs loop, both scenarios | 804 | new | P1 |
| **918** | R4 | **Weight-space PCA** under alternation, W1 and W2 | 804 | new | P1 |
| **919** | R4 | **Joint pre-training then sequential** | 805 | new | P1 |
| **920** | R4 | Which measurements separate by scenario — tie-out | many | re-analysis | P2 |
| **921** | R5 | **Retention distribution, and what moves it** — interventions overlaid | 819, 818, 812 | new | P2 |
| **922** | R5 | **Partial-column freeze** vs freeze-all vs masking | 806 | new | P1 |
| **923** | R5 | **Trained linear probe vs argmax**, both tasks, both scenarios | 803 | new | P1 |
| **924** | R5 | Per-layer weight path and hidden-code drift | 803, 804 | new | P1 |
| **925** | R5 | Does task structure predict what survives — both scenarios | 820, 819 | copy 311 + new | P2 |
| 926 | R5·A | Class-IL digit table | 819 | copy 312 | P3 |
| **931** | D1 | Intervention ranking, both scenarios | 822 + others | new | P2 |
| 932 | D1·A | SI and k-WTA sweeps | 822 | copy 210 + 220 | P3 |

**Bold = main text.** Count: 901, 904, 907, 911, 912, 913, 915, 916, 917, 918, 919, 920, 921, 922,
923, 924, 925, 931 = **18 main-text figures**, of which 3 are Methods and 1 Discussion, leaving
**14 in Results**. At ~3,000 Results words that is ~215 words each — workable, and one figure
above the earlier estimate because R5 kept the task-structure figure rather than demoting it.

---

## src/ changes required — need approval before 800 starts

Three, each small, each proposed separately with a diff as `CLAUDE.md` requires.

1. **`_apply_freeze` column masking** (for 806). Currently whole-matrix. Needs to accept a column
   index set so only the task-1 output columns are frozen.
2. **Per-layer ‖ΔW‖ and weight snapshots in the diag channel** (for 803, 804). `displacement` is
   already published; this adds the same pattern for weight-step norms and an optional snapshot
   hook.
3. **Online linear probe** (for 803). Fit a logistic/least-squares readout on the held-out set at
   a checkpoint and return its accuracy. Small, self-contained, and it replaces NCM everywhere.

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

## Open before the list freezes

- **Do captions count against the 8,000 words?** If not, a caption can carry ~100 words of "what
  was done / what it shows" per figure and several appendix demotions above become unnecessary.
- **802's activation sweep** is the only genuinely new *scientific* question in the 800 list;
  everything else instruments or re-runs. If time is short it is the first cut, and tanh becomes a
  declared limitation instead.
