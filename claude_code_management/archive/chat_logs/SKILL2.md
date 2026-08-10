---
name: experiment
description: Scaffold a new experiment script following the project conventions. Refuses if the research question is not a single sentence or is not attached to a slide.
disable-model-invocation: true
argument-hint: "<id> <one-sentence research question>"
arguments:
  - id
  - question
---

# New experiment: $id

**Question:** $question

## Instructions

### 1. Refuse if the question is not sound

Stop and say so, without writing anything, if any of these hold:

- The question is not a **single sentence**. Two sentences means two experiments.
- The question is not falsifiable — if no result would change what we do next, do not run it.
- The experiment is not attached to a slide in `claude_code_management/presentation_plan.md`.
  Check the file. If it is genuinely new work, say so and ask whether the plan should change first.

Refusing here is doing your job. Do not soften it into a caveat and proceed anyway.

### 2. Pre-commit the readings

Before writing code, state what each plausible outcome would mean, in this shape:

- If <result A> → <interpretation>, and the next step is <X>.
- If <result B> → <interpretation>, and the next step is <Y>.

Show me this and wait for a yes before writing the script.

### 3. Scaffold

Create `experiments/$id_<short_slug>.py` following the conventions in
`.claude/rules/experiments.md`:

- Research question as a single-sentence comment at the top.
- Constants block immediately below it: `TASKS`, `ITERS`, `BATCH`, `SEEDS`, `LR_GRID`, `SCENARIO`.
- Import the methods from `src/methods.py` and the metrics from the shared metrics module.
  Do not redefine a metric.
- Controls included: `backprop` and `replay` alongside whatever is under test.
- Figure name derived from `__file__`.
- Evaluate on the held-out test split.

Keep the script thin. Anything reusable belongs in `src/`, and adding it there is a separate
approved change.

### 4. Do not run it yet

Show me the script. I will approve it before it runs.
