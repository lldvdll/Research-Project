---
paths:
  - "experiments/**"
  - "results/**"
---

# Rules for experiments

## One question per script

Every experiment script opens with its research question as a **single sentence** in a comment at
the top. If the question needs two sentences, it is two experiments.

No script is started without its question written down first, and no script is started that is not
attached to a slide in `claude_code_management/presentation_plan.md`.

## Structure

- Constants at the top of the script. Reusable logic lives in `src/`, not here.
- One script produces one figure, named from `__file__`.
- Metric definitions are imported from the shared metrics module. Never redefine a metric inline.
- Controls on every forgetting experiment: `backprop` (negative — should forget) and `replay`
  (positive — should recover). If replay works, the problem is provably solvable.

## Pre-commit the readings

Before running, state what each possible outcome would mean. If no outcome would change what we
do next, the experiment is not worth running.

## Measurement traps — these have all bitten before

- **Never evaluate on training data.** Use the held-out test split. A bug here once invalidated a
  day of sweeps.
- The flat line at exactly `100 / n_classes` is **not chance**. It is the collapse floor — the
  score of a model predicting one class for everything.
- **Accuracy is a threshold readout.** After a task switch nothing appears to happen for ~20 steps
  while logits climb, then it flips fast. Always log raw pre-argmax outputs alongside accuracy.
- **Below-chance task-1 accuracy** (0%, not 50%) means argmax is being captured by later-task
  units — that is output-layer suppression, not representation loss.
- **Do not fit sigmoids to accuracy curves.** They are step-like and noisy. Use threshold crossings
  and the trajectory plot.
- **`cur%` is degenerate at one class per task**; **`seen%` has a changing denominator**. Prefer
  per-task accuracy on fixed class sets.
- **Report final new-task accuracy alongside retention.** "Forgot less" is not a result if it is
  confounded with "learned less".
- More than one pass over the data is cyclic revisiting, not longer continual learning.

## Reproducibility

Learning rate grid-searched independently per rule. At least 5 seeds. Report 68% confidence
intervals, matching Song & Bogacz.
