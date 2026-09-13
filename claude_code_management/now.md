# Now

**Dashboard. One screen. If a section grows past a screen it belongs somewhere else.**

---

# ⇢ START HERE — read these, in this order, before doing anything

1. **this file** — what the project is doing right now, and the next actions.
2. **`report_progress.md`** — THE operational file for the current pass. Carries the user's
   standing conventions (colour standard, figure style), the per-section checklist built from
   their review comments, a Deferred log, and what is already Completed.
3. Then, only as the task needs them:
   - **`script_plan_800_900.md`** — the figure inventory: which script feeds which figure, and
     what has been built.
   - **`plot_walkthrough.md`** — every figure read in report order, as an argument. Use when
     the question is "what does this figure claim", not "how do I draw it".
   - **`progress.md`** — one line per experiment, for any pre-800 number.
   - **`knowledge_base.md`** — §9.3 first. Use before making a substantive claim.

`current_state.md` **exists but is dated 2026-08-11** and predates the 800/900 series — history,
not current state; this file and `progress.md` supersede it. `handover.md` is the 2026-09-10
build-pass record, also history, superseded by `report_progress.md` for anything current.

---

## WHERE THE PROJECT IS — 2026-09-11

**All experiments have run and every figure in the plan exists.** Nothing is blocked on
results. The project is in its **final phase: revising the report's figures, one section at a
time, and writing the text around them.**

The loop, which the user set explicitly:
> user reviews a section's figures → sends comments → Claude adds a checklist to
> `report_progress.md` → implements: **scripts first and re-run, then the report layout, then
> the text with cross-references** → completed items move to Completed.

The user supplies the prose. **Do not write captions or body text that was not given.**
Out-of-sequence requests go to the Deferred log rather than being done early.

## THE REPORT — how to build it

- MiKTeX + LaTeX Workshop are installed. Build with `latexmk -pdf -synctex=1
  -interaction=nonstopmode -file-line-error main.tex` from `report/`, or Ctrl+Alt+B in VS Code.
- `\graphicspath` includes `../experiments/`, so figures are read **straight from where the
  900 scripts write them**. Never copy a PNG into `report/figures/`.
- latexmk tracks each figure by MD5, so re-running a script that changes bytes triggers a
  rebuild, and one that does not, does not.
- Grid macros live in `main.tex`: `panelbox` (boxed, full-width or column), `\panel{width}{file}`
  (lettered A, B, C by subcaption), `\solo{file}` (unlettered). Panel letters are never typed.
- A figure that does not fit is fixed by changing `figsize` in its 900 script and re-running —
  not by scaling it in LaTeX.

## REMOTE WORKING — the user reviews the PDF on GitHub

`report/main.pdf` **is committed on purpose** (its ignore rule was removed 2026-09-11) so the
user can read the built report remotely and comment by chat. **After any change that alters a
figure or the report, rebuild and push**, or they are reviewing a stale document.

⚠ **Do not `git add` the two large arrays.** `804_repeated_alternation_{class,domain}_il.npz`
are 137 MB and 223 MB after the 10-seed trace re-run, over GitHub's 100 MB hard limit; staging
them makes the push fail. They are tracked-but-modified on purpose. Everything else commits
normally. Full note in `report_progress.md`.

## NEXT ACTIONS

1. **`810_capacity_vs_width.py` and `811_trunk_power.py` were left running** (new 800-series
   runs, both rules). **A new session cannot see their job output — check for the arrays on
   disk instead:**

   ```
   ls experiments/810_capacity_vs_width.npz experiments/811_trunk_power.npz
   ```

   - **Both present** → build **902** from them. 902a wants random-trunk vs
     trained-trunk-with-probe vs joint, as a distribution over seeds; 902b wants the capacity
     sweep with a PC arm. Then rebuild the report and **push**.
   - **Missing** → the run did not finish. Just re-run the script; both are idempotent and
     write only their own array. 810 is the long one (~2-4 h, 24 cells, prints per cell);
     811 is ~1-2 h. Run them with `run_in_background` and do other work meanwhile.
   - Partial results are not saved — each script writes once, at the end.
2. Then the **M2 setup-and-controls text**, ordered to match the parameter table. The content
   the user asked for, and the answers already found for its open questions, are both in
   `report_progress.md`.
3. Two diagram placeholders in `methods.tex` have no figure and no script:
   `incremental_learning_diagram.png` and `learning_rules_diagram.png`. **Asked the user
   whether to build them; no answer yet.** Do not invent them.

---

---

## Older material

Everything that predated the 800/900 series — the 100-series stipulations, the findings list,
the checklist, the settings table and the corrections — moved to **`now_archive_pre800.md`** on
2026-09-11, so this file stays one screen. Nothing was discarded. For a pre-800 number prefer
`progress.md`, which re-derives each one from the saved arrays.
