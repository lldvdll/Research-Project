---
paths:
  - "src/**"
---

# Rules for changing `src/`

## Approval

Every change to this directory needs explicit approval before it is written. Propose the change,
show the exact diff, and wait. `ask` rules in `.claude/settings.json` enforce this; do not try to
route around them by writing through a shell command.

## One change at a time

A change is one edit to one concern. After it is written:

1. Re-run the most recent experiment that exercises the changed code.
2. Confirm its figure is unchanged.
3. Commit that one change on its own.

Only then propose the next change. Do not batch.

## Current refactoring brief

The depth refactor and the Song & Bogacz protocol work stay in. Nothing is being reverted and
nothing is being removed.

The goal is readability, not rewriting:

- Move conditionals **out of the core functions** in `eqprop.py` and `predictive_coding.py`.
  Scenario (Class-IL vs Domain-IL) and depth should be resolved at construction time, not
  branched on inside the update step.
- Once a core function is readable and tested, **freeze it**. It changes again only when an
  experiment cannot be expressed without the change — not because a tidier form exists.

## Interface contract — do not break

Every `make_*` returns `(train_step, predict)`.

- `train_step(x, y)` performs one update.
- `predict(x, raw=False)` returns class indices; `raw=True` returns pre-argmax outputs.

Adding a method means adding one `make_*`. Experiment scripts should only ever change their
`methods` dict.

## Do not reintroduce

wandb, Optuna, persistent HPO infrastructure, class hierarchies, a shared `harness.py`. All were
built, found to cost more momentum than they saved, and deliberately deleted.
