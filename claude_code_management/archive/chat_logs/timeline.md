# Timeline

Sequential log of chats. One entry per objective. **Entries are never edited once added.**
Format: `### [YYYY-MM-DD] #NNN — Objective` / `**Outcome:** one sentence.`

---

### [pre-log] #001 — Establish project scope and choose an EBM to compare against backprop on catastrophic forgetting
**Outcome:** Scope locked to the advisor's four-point outline (pick one EBM → compare CF vs backprop → explain why they differ → try to reduce CF in the EBM), with EqProp chosen as the named EBM and PC added later because the interference claim under test is PC's.

### [pre-log] #002 — Build and validate the four method implementations (backprop, replay, EqProp, PC) on 14×14 MNIST
**Outcome:** All four train; PC gradients verified against finite differences to ~1e-9; EqProp reaches ~91% joint validation; interface contract `make_* → (train_step, predict)` settled.

### [pre-log] #003 — Establish Class-IL forgetting baselines and controls
**Outcome:** Backprop collapses to the floor (~10% on 10×1), replay recovers to ~64%, EWC fails at ~20% ≈ baseline, confirming the problem is solvable and that the negative/positive control pair is diagnostic.

### [pre-log] #004 — Rebuild the conceptual understanding of FFNN/BP/EBM/PCN/PC/EqProp from primary sources
**Outcome:** Two earlier claims corrected — PC only approximates backprop in the small-step limit, and the "zero error → weights don't move" argument gives task 1 no protection — replacing the mechanism hypothesis with the testable "where does the weight movement go?" formulation.

### [pre-log] #005 — Analyse the mechanics of forgetting at the output layer
**Outcome:** Forgetting decomposed into four mechanisms across two pathologies (logit suppression = calibration failure, missing inter-context boundary = representation failure), with the active output set identified as the knob that determines which can fire in which scenario.

### [pre-log] #006 — Run the 4-method × 10-seed comparison on the 5×2 Class-IL split (exp 11)
**Outcome:** PC gives the best interference trade-off (area above the ACC1–ACC2 diagonal) but only ~10% final task-1 retention, EqProp is the worst panel overall, and only replay ends genuinely up-and-right — i.e. PC degrades gracefully rather than retaining.

### [pre-log] #007 — Attempt a like-for-like reproduction of Song & Bogacz Fig 4d (exp 12)
**Outcome:** The reproduction was not achieved, roughly a week was spent on hyperparameter matching, previously working exp-12 code was broken, and the codebase became unreadable to the author.

### [2026-08-10] #008 — Reset the project: install a timeline + knowledge-base workflow, critically evaluate the draft 13-slide deck, and produce a locked slide plan with matched experiments for a 20-minute Friday presentation
**Outcome:** Draft deck restructured into a 20-slide plan (12 mandatory / 8 optional) around a new spine — Song & Bogacz Fig 4d is Domain-IL, not Class-IL, so the scenario contrast becomes the headline result — with seven scoped experiments (P0–P7) mapped one-to-one onto slides and the exp-12 codebase ordered frozen rather than repaired.

### [2026-08-10] #009 — Fix the open decisions from the plan review and rewrite the presentation plan in a working format
**Outcome:** Locked the 5×2 split for both scenarios with the output layer as the only difference, kept both Class-IL and Domain-IL as parallel result slides with a comparison slide, retired the "scenario boundary" framing as premature, and rewrote the plan as a 20-slide list plus per-slide contents/reading/experiments with a 10-item experiment index.
