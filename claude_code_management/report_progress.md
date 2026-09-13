# Report progress — figure-by-figure revision

Working file for the figure revision pass. Started 2026-09-11, after the first
figure-focused draft built (19 pages, 31 figures placed, 0 errors).

## How this file works

- The user reviews **one section at a time** and sends comments on its figures.
- Each message adds a **new checklist to that section** below, written from those comments.
- Items that are done move to **Completed**, at the bottom, with the date.
- Comments that call for work **out of sequence** (a different section, or something
  that has to wait for another change first) go to **Deferred**, not into the active
  checklist, so they are not lost and not done early.
- Where a figure needs a script change, the script is edited and **re-run**; the figure
  is never scaled or patched in LaTeX to hide a sizing problem.
- The user supplies the prose that references each figure. **No captions or body text
  are written here without being given them.**

## Standing conventions for this pass

Given by the user 2026-09-11, and they apply to every figure unless overridden:

- **Colour standard** (given 2026-09-11, applies to all future plots; retro-fitted only to
  figures being worked on now):
  - backprop **black**, PC **red**
  - when a plot carries rule *and* scenario: Domain-IL **solid**, Class-IL **dashed**
  - when a plot carries scenario only, no rule split: Class-IL **purple**, Domain-IL **green**

- **Wide beats tall.** Prefer a wide, short panel to a tall one.
- **Budget is 4–5 figures per section** if they are carefully styled — so styling
  effort buys figure count, and a bloated figure costs a slot.
- **Minimum text inside the figure**: axes, annotations, legend. Nothing that the
  caption or body text should be carrying.
- Panel letters (A, B, C) come from `subcaption` via `\panel`, never typed by hand.
- Figures live in `experiments/` and are read from there by `\graphicspath`; they are
  never copied into `report/figures/`.

---

## Remote working — the user reviews the PDF on GitHub

Set up 2026-09-11, because the user is reviewing from elsewhere and sending comments by chat.

- **`report/main.pdf` is committed on purpose.** Its ignore rule in `report/.gitignore` was
  removed. It is the only way the user can see the built report remotely, so it has to be
  current.
- **After any change that alters a figure or the report: rebuild, then commit and push.**
  A stale PDF means the user is reviewing something that no longer exists. The push is the
  last step of the job, not an optional extra.
- Commit the things the report is made of — `report/*.tex`, `report/main.pdf`,
  `experiments/*.py`, `experiments/*.png`, and these management files.

⚠ **Never `git add` these two arrays:**

```
experiments/804_repeated_alternation_class_il.npz    137 MB
experiments/804_repeated_alternation_domain_il.npz   223 MB
```

They went over GitHub's **100 MB hard per-file limit** when 804 was re-run with
`TRACE_SEEDS = 10`, and staging either one makes the push fail outright. They are tracked from
an earlier, smaller version and are deliberately left modified-but-unstaged. The unpushed
history itself is clean — 0.13 GB of blobs, none over the limit — so everything else pushes
normally.

**The tidy fix has not been done because it needs the user's say-so:** `git rm --cached` those
two (and probably the other large `.npz`), add them to `.gitignore`, and let the scripts
regenerate them. That stops `git status` being permanently noisy and removes the trap where a
careless `git add -A` breaks the push. Raise it; do not do it unasked.

---

## Environment traps on this machine — these all cost real time

- **The shell here collapses `\` to `\` inside heredocs.** A `sed` written with
  `s/.../\par\vspace{6pt}/` reached sed as `\parspace{...}`, which sed read as escape
  codes: it turned `\p` into `p`, `` into a vertical tab and `
` into a carriage return,
  silently corrupting 53 places across three `.tex` files and splitting one line in half.
  **Do not put backslashes in a heredoc-delivered `sed` or Python string.** Build them with
  `chr(92)`, or use the Write/Edit tools, which are unaffected.
- **`/tmp` is not visible to the Windows Python binary.** Git Bash resolves it; `python.exe`
  does not, so a file written by a shell heredoc into `/tmp` cannot be opened by Python and the
  failure looks like a missing file. Use the session scratchpad directory instead.
- **`grep` patterns containing backslashes behave unpredictably here** for the same reason.
  When checking for LaTeX commands, scan with Python rather than trusting a `grep -c`.
- **A `git status` count is not a safety check.** 31 of the entries are `.npz`, two of which
  are over GitHub's limit. There is now a **pre-commit hook** (`.git/hooks/pre-commit`) that
  refuses any staged file over 90 MB — tested against the 212 MB array, and it holds. Hooks
  are not versioned, so this protects *this* clone only.

---

## Active

**Methods — review 1, received 2026-09-11.** Order of work as instructed: scripts first and
running, then the report figure layout, then the text with cross-references.

---

## Checklists by section

### Methods

#### M1 — Catastrophic forgetting (901)
Review 1, 2026-09-11:
- [ ] Move 901 **inline** (single-column `figure`, not full-width `figure*`) so it sits
      alongside the text describing it.
- [ ] 901: remove the switch line.
- [ ] 901: colour the **background** to match the task (phase shading instead of the line).
- [ ] 901: make the same-colour per-class training lines visible against that shading.
- [ ] 901: move the **crossover annotation to the left** of the switch.
- [ ] 901: add a **retention annotation in green**, bottom right, above the task-1 curve.
- [ ] 901: **reduce the learning rate** so the curve is easier to read — needs 801 re-run.
- [ ] 901: adjust the learning/forgetting arrows to match the new curve.
- [ ] 901: remove the title; remove the bracketed text from the legend; remove the chance line.
- [ ] Text: briefly **introduce crossover and retention here**, since both are plotted in 901
      and are easier to explain against a picture than in the abstract. They are referenced
      again in setup/controls.

#### M2 — Experiment setup and controls (table + 902/903/904/906)
Review 1, 2026-09-11. Prose to be brief and **ordered to match the parameter table**.

- [ ] **902a**: add the distribution of trained-trunk + probe. Figure must compare **random
      trunk + probe** AND **trained trunk + probe** against the **jointly trained** network.
      ⚠ needs a new training run — 101 holds only the random-trunk arm.
- [ ] **902b (capacity)**: add a **PC arm**. ⚠ needs a new training run — 100 is backprop-only.
- [ ] Apply the new colour standard to every figure touched in this section.
- [ ] Text — MNIST, and why 14x14, with a supporting figure of digits **pre/post resampling**
      (new script). Open question logged below: does 902a also belong here, as the evidence
      that the trunk matters and so there is something in the hidden state to find?
- [ ] Text — why depth 1, with a supporting figure (which figure? logged below).
- [ ] Text — hidden width was swept; H=32 chosen to remove a capacity-limit confound.
- [ ] Text — SGD, to avoid a momentum optimiser confounding behaviour at the task switch.
- [ ] Text — tanh: legacy choice, taken because it also worked for EqProp. Note forward to the
      later finding that activation matters in Domain-IL.
- [ ] Text — learning rate 0.02 for both rules in all experiments. Decide whether the
      supporting data can be cited here or must forward-reference a Results figure.
- [ ] Text — batch 32. Reason not currently remembered; check the archive before writing.
- [ ] Text — accuracy stopping, because (a) endpoint metrics move with the number of updates
      and (b) accuracy at a fixed step count is high-variance, so absolute retention and
      crossover are unstable. Thresholding on "the task has been learned well enough" is a
      more robust base for "how much is lost afterwards" than "we trained for x steps".
- [ ] Text — the 90% threshold specifically. Is there a figure that earns it? Cite the joint
      ceiling? If it is flat either side, decide whether that needs explaining at all.
- [ ] Text — seed pairing, **with 906** included to show pairing cuts variance in the paired
      comparison.
- [ ] **904 (PC settling) rebuild** — consolidate the two scenario panels into **one plot**:
      - clip to **100 steps** so the oscillation is visible
      - thinner lines, **bold for the chosen dt**
      - plot the **settle point per dt for both scenarios on the same axes**
      - show the **range of values over steps 400-500** per dt, so a high settle point is
        visibly also an oscillating one
      - story: a shared fixed point at low dt; pick the highest dt that still collapses to it
        (runtime saving at no cost); avoid higher dt because it oscillates.

#### M3 — Measuring forgetting (907, 909)
_no comments yet in review 1_

### Results

#### R1 — Two scenarios, two forgetting phenotypes
_awaiting review_

#### R2 — The size of the learning-rule effect
_awaiting review_

#### R3 — Testing the mechanism prospective configuration is credited with
_awaiting review_

#### R4 — Why these are two investigations, not one
_awaiting review_

#### R5 — Evidence for an output-competition component
_awaiting review_

### Discussion

#### D1 — Intervention ranking
_awaiting review_

---

## ⚠ DECISION NEEDED BEFORE THE NEXT FIGURE EDIT — the colour standard collides with the code

The standard given on 2026-09-11 (backprop **black**, PC **red**; scenario-only figures
Class-IL **purple**, Domain-IL **green**) does not match `src/style.py`, which every 900 script
is supposed to draw from:

| | standard asks for | `src/style.py` sets | line |
|---|---|---|---|
| backprop | black | `#5c5c5c` grey | 48 |
| pc | red | `#d1682a` orange | 48 |
| Class-IL | purple | `#6a4c93` purple — **agrees** | 61 |
| Domain-IL | green | `#1b7f79` **teal** | 61 |
| replay | — | `#3f7d3a` green | 48 |

**Domain-IL green is the real problem**: `style.py` deliberately spends green on **replay** and
gives Domain-IL teal to avoid the clash, with that reasoning written into the file. Making
Domain-IL green puts it in direct collision with replay, which appears in 913 and in the
intervention figures.

Three ways out, none taken yet — **ask before choosing**:
1. Change `style.py` to the new standard and move replay to a fourth colour. Consistent
   everywhere, but re-renders every figure in the report.
2. Apply the standard only to figures being revised, and accept that older figures disagree
   until their section comes up. Cheap now, inconsistent in the middle.
3. Keep Domain-IL teal and take the rest of the standard. Closest to intent, no replay clash.

Figures already drawn under the new standard: **904** (Class-IL purple, Domain-IL green — no
replay arm in it, so nothing actually collides yet).

## ⚠ FINDING that changes R4's text — 929, added 2026-09-13

`929_pc_components_over_training` plots PC1-PC5 against training updates instead of PC1 against
PC2, and it **overturns how 918 has been read**.

918's phase plane can only show the first two components. In Domain-IL/pc/W2 those two are
smooth — PC1 (89% of variance) rises monotonically, PC2 (8%) rises then falls — which is why
918 renders that cell as a single sweep while every other cell bounces. But **PC3, at ~1% of
the variance, oscillates cleanly through all five task switches.**

So the oscillation was never absent. It was **demoted into a low-variance component**, where the
phase plane could not see it. Backprop keeps its ringing in PC1/PC2, which is why 918 draws it
as a loop.

Last-cycle over first-cycle peak-to-peak, Domain-IL/pc/W2: PC1 **x0.14**, PC2 x0.44,
**PC3 x0.54**, PC4 x1.08, PC5 x0.06. PC1 damps hard; PC3 does not.

**What this means for the write-up:** the R4 claim should become "PC's oscillation is pushed
into a low-variance direction while a large monotone drift dominates PC1", not "PC does not
oscillate in Domain-IL's readout". That is consistent with the earlier cosine test (PC's
block-to-block updates do not reverse at the switch, cos ~0 to +0.24, where backprop reverses
at -0.84 to -0.96) and it is a sharper claim than the one currently written.

⚠ `plot_walkthrough.md`'s R4 entry for 918 still carries the old reading and needs revising.
929 is a DIAGNOSTIC and is not in the report — the figure budget is already 21 against a frozen
18. Decide whether it earns a slot or whether its finding just rewrites 918's paragraph.

## Deferred / out-of-sequence

Carried in from the build pass on 2026-09-11. None of these are started; each is
listed with the point at which it should be picked up.

- [ ] **Three figures are too wide to survive the page and need a 2×2 relayout in
      their scripts**, not a LaTeX fix. Natural width against a 7.06 in page:
      `915_target_alignment` 16.4 in (2.32×), `912_pc_minus_backprop_sweeps` 15.4 in
      (2.18×), `916_displacement_to_retention_a` 14.9 in (2.15×). All three are 1×4
      horizontal strips. **Do these when their own section comes up** (R3, R2, R3)
      rather than as a batch, since the relayout may change what the user wants on
      each axis anyway.
- [ ] **`\gap{}` captions are stale** across `results.tex` and `methods.tex` — several
      still say a run is "queued" or "running" for work that finished. They are the
      user's own text, so they are not edited here; flagged so they are not mistaken
      for current status.
- [ ] **926 is the one unbuilt figure** in the whole plan (Class-IL digit table, copy of
      312, appendix, P3). Optional.
- [ ] **807 is claimed by two different training runs** — `807_code_drift.py` (feeds
      924b) and `807_overtrain_task1.py` (feeds 927/928). Nothing collides on disk and
      no figure is wrong, but the number is ambiguous in every cross-reference.
      Renaming touches a script, two `.npz` files and 924b's reader, so it needs
      approval. Full note in `script_plan_800_900.md`.
- [ ] **Figure budget is over-subscribed** — 909, 927 and 928 are built, putting the
      count at 21 against a frozen 18. Two demotions needed. Candidates, weakest first:
      920, 925, 914. Decide once the sections have been reviewed, since the review may
      settle it on its own.
- [ ] **`report/main.tex.pregrid`** is a pre-edit backup left in the repo. Delete when
      the new layout is settled — user's call, not to be removed unasked.

---

## Findings and answers to the open questions in review 1

- **Batch 32 — reason recovered.** Not arbitrary: batch size 1 destroys EqProp ("with plus/minus
  1 targets and no batch to average over, every update reconfigures the network to the most
  recent image"), and the archived guidance was to keep batch >= 16 and slow training with the
  learning rate instead. So batch 32 is the same class of legacy decision as tanh — taken so
  that EqProp remained trainable, and kept for continuity after EqProp was shelved.
  ⚠ **And it carries a caveat worth putting in the text.** The literature this project is
  testing reports PC's advantage as largest at batch size 1 and with depth. At batch 32 and
  depth 1, PC is being measured near its *weakest* regime. That is already logged as CTRL-6 in
  the knowledge base and it is an honest limitation, not a flaw — but the Methods should say it
  rather than leave a reviewer to notice it.
- **Learning rate 0.02** — the supporting data exists but is not a Methods figure: 340 (both
  rules peak near 0.01-0.02 and decline past it) and 314 (an exact copy of the core run at
  0.02, paired: all p > 0.34, so the change is free). 340 is drawn as part of the R2 sweep
  figure, so Methods should forward-reference rather than duplicate it.
- **The 90% threshold** — once 810 lands it gives the joint ceiling at H=32 for *both* rules
  under the current protocol, which is the number the threshold should be read against; 907
  already carries the sensitivity-to-threshold half. No new figure needed.
- **Depth 1** — user's call: leave it, with a note pointing at the Results section.

## Completed

**2026-09-13**
- [x] **810 and 811 finished; 902 rebuilt from them** and added to `methods.tex` along with
      910 and 906. Report now builds at 20 pages, 35 figures, 0 errors.
      - **811 (trunk power)** — the trunk is worth **+13.0 / +16.4** points to backprop
        (Class-IL / Domain-IL) and **+7.0 / +9.0** to PC. `trained_probe` lands within 1.4
        points of `joint` in all four cells, so the probe is not the weak link and the control
        is sound. **PC's trunk gain is about half backprop's in both scenarios** — a Methods
        control that turned up something the Results will have to own.
      - **810 (capacity)** — H=32 is clear of the bottleneck for both rules: headroom to H=128
        is +0.8 / +0.9 for backprop and +1.3 / +1.5 for PC. ⚠ **PC's joint ceiling sits 6-8
        points below backprop's at every width** (86.7 vs 92.8 Class-IL, 85.9 vs 93.9
        Domain-IL at H=32). Real and consistent; decide whether it is a finding or a tuning
        artefact before the text leans on it.
      - 810 was re-run after being caught using dt=0.4 while sweeping width; it now pins
        dt=0.2, matching 341/342.

**2026-09-11**
- [x] **Remote review set up and pushed.** `report/main.pdf` un-ignored and committed; 74
      commits pushed to `origin/main` (the repo had never been pushed from this machine).
      240 figures and the built PDF are now on GitHub. `now.md` and `CLAUDE.md` rewritten to
      give a fresh context the right reading order; this file is the second read.
- [x] **901** rebuilt: switch line removed, task-phase background shading, per-class lines
      darkened so they read over the shading, crossover annotation moved left of the switch,
      green retention annotation bottom-right, title removed, bracketed legend text removed,
      chance line removed, arrows repositioned.
- [x] **801 re-run at lr 0.005** (from 0.02) so 901's transition is legible — it stretches the
      crossover out about fourfold. Crossover 64%, retention 1.4%, task 2 87.3%.
- [x] **901 moved inline** — now a single-column `figure`, built at 3.4in, sitting beside its
      text. `panelbox` was changed to size itself from `\linewidth` rather than `	extwidth`
      so it works in both one-column and full-width floats.
- [x] **904 rebuilt as one consolidated plot**: settled displacement against dt, both scenarios
      on one axes (Class-IL purple, Domain-IL green per the new standard), min-max bar over
      steps 400-500 so an oscillating dt shows a bar and a converged one does not, chosen dt
      marked. Result is clean — dt 0.02 to 0.5 land on an *identical* fixed point with exactly
      zero tail range in both scenarios; 0.85 and 1.5 both jump and oscillate.
- [x] **334 re-run with dt = 0.4 added** and the grid trimmed to seven values. The old grid
      stepped 0.2 -> 0.5 straight past 0.4, so the figure had been asserting a chosen value it
      had never measured. Old ten-value arrays kept as `*.npz.bak10dt`.
- [x] **904 dt axis labelled with the actual swept values** rather than log decades.
- [x] **910_input_resampling.py** (new 900-series script): MNIST at 28x28 over 14x14, one
      column per class, for the "why 14x14" row of the table.
- [x] Repaired a broken cross-reference in `results.tex` (`Fig.~
ef{fig:919}` had been split
      across two lines by my own bad `sed` earlier in the session, which dropped the comment
      marker and leaked the tail of the line into the body text). Scanned all three section
      files for the same damage class — this was the only instance.
