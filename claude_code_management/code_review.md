# Code review — `src/`

Review 2026-08-10/11. All modules read. **Every finding is closed** — see the Status table at the
end for what was fixed, what was left alone deliberately, and the two protocol numbers that are
still placeholders.

**Status key:** `TRACED` — followed in the source, not executed · `VERIFIED` — reproduced by
running · `FIXED` — corrected and re-verified · `DESIGN` — not a bug, a decision to make.

A researcher-facing map of the code — what each module does, how a run flows through it, and the
output-mechanics axes — is in [`code_map.md`](code_map.md).

---

## Overall assessment

Better than expected. `model.py` is a clean separation of *what the network is* (`Arch`) from
*what the loss says* (`Objective`), with `output_error()` giving all three rules an identical
output signal — which is precisely what makes a learning-rule comparison a learning-rule
comparison, and what the plan's protocol depends on. Multi-layer support is real, not bolted on.
`Params` aliases `.W1`/`.W2` into `.Ws[]` so older code keeps working.

I traced the multi-layer predictive-coding relaxation indices and the EqProp nudge sign. **Both are
correct.** The energy-based cores are not where the problems are.

The problems are in the **glue**: freezing, and the `active` set under Domain-IL. Both are needed by
scripts the new plan depends on.

---

## B1 — `_apply_freeze` was broken `FIXED`

`methods.py:53–66` calls `resolve_freeze(freeze, len(p.W))`, but `model.py:209` defines
`resolve_freeze(p, freeze)` returning a **list of tensors**. Three faults in one line:

- arguments are swapped
- `p.W` does not exist — `Params.__getattr__` matches `r"([Wb])(\d+)$"`, which needs a digit, so
  bare `W` raises `AttributeError`
- the return value is unpacked as `wi, bi` but is a flat list

`_apply_freeze` returns early when `freeze` is empty, so this only fires **once something is
actually frozen** — which is why it has gone unnoticed. It is called from `make_backprop` and
`make_replay` only; `pc_update` and `eqprop_update` implement freezing correctly themselves, by
name.

All three faults confirmed by running: `p.W` → `AttributeError: W`; the swapped call →
`AttributeError: 'list' object has no attribute 'named'`; `resolve_freeze(p, ["W1"])` returns a
1-element list of tensors.

**Impact:** freezing under backprop or replay raised immediately. Script 42 (*how much room is
there*) is a freeze experiment on backprop, so it could not have run.

**Fix.** Zero the gradient of exactly the tensors `resolve_freeze` names:

```python
for t in resolve_freeze(p, freeze):
    if t.grad is not None:
        t.grad.zero_()
```

**One decision inside this fix.** The broken code intended `W1` to drag `b1` with it
(`for i in (bi | wi)`). PC and EqProp do **not** do that — `predictive_coding.py:117` updates
`b1` unless `b1` is itself in the freeze set, and `eqprop.py:120` zeroes only named tensors. Since
the backprop path had never executed, the exact-name convention is the only one that has ever run,
and it is now the convention everywhere. **Freezing a whole layer means naming both `W1` and
`b1`.** If the rules disagreed here, script 42 would be comparing different interventions.

Verified: with `freeze={"W1"}` under backprop, `W1` unchanged after 10 updates, `b1` and `W2` both
moved.

## B2 — `active` was not remapped under `label_map` `FIXED`

`runner.py:135` passes `task` — **global class ids** — as `active`. Under Domain-IL the plan uses
5 shared output units, so a second task of `[5,6,7,8,9]` reaches
`active_vector()` (`model.py:229`), which does `m[[5,6,7,8,9]] = 1.0` on a length-5 tensor.

`label_map` correctly remaps the training labels `y` and the accuracy check, but **not `active`**.

**Impact:** Domain-IL at 2×5 with 5 shared outputs — the protocol in the plan — raised
`IndexError: index 5 is out of bounds for dimension 0 with size 5` on the second task, confirmed by
running. Affected every rule, since `active_vector` is called unconditionally in `pc_update`,
`eqprop_update`, `make_backprop` and `make_replay`.

**Fix.** Compute the active set once per task, in output-unit space:

```python
active = task if label_map is None else sorted({label_map[int(c)] for c in task})
```

Verified: 2×5 Domain-IL with `out_dim=5` now runs under all four rules.

## B2b — replay's buffer under `label_map` `FIXED`

`make_replay` keyed `seen` and `_store` by the label it was handed, which under Domain-IL is an
**output unit**, not a class. After task 1 `seen == {0,1,2,3,4}`, so **task 2 was never stored**,
and `_store` drew images by unit id from `class_idx` anyway.

**Fix — a per-label reservoir.** `per_class` slots per label, each holding a uniform sample of
every example of that label seen so far (Algorithm R, one reservoir per label). The buffer now
stores from the stream and never consults the dataset.

Two weaker fixes were considered and rejected. *Keying by class id* is impossible — `train_step`
is handed the remapped label and never sees the class. *Keeping the first `per_class` from the
stream* fails identically, because units 0–4 fill during task 1. A reservoir needs **no knowledge
of task boundaries**, which is essential: under Domain-IL both `y` and `active` are identical
across tasks, so `train_step` cannot detect a boundary even in principle.

Verified by classifying the stored **images** against true class means, since under Domain-IL the
labels cannot tell you which task an example came from:

| | task-1 images | task-2 images |
|---|---|---|
| after task 1 | 93 | 7 ← nearest-mean misclassification, not real content |
| after task 2 | 53 | 47 |

Before the fix the second row would have been 100 / 0. Class-IL is unaffected and stays exactly
balanced at 20 per label across all ten.

**This changes replay's behaviour**, which is why it was raised rather than folded into the first
bundle. Two effects, both arguably improvements to the control: the buffer now fills over the
first few batches instead of instantly, and it no longer draws `per_class` images of a class from
the full training pool the moment it sees **one** example — i.e. it no longer uses data the
network has not been shown.

`train_data`/`class_idx` are now accepted and ignored, so every existing call site still works.

## B3 — `_call` masked genuine errors `FIXED`

`runner.py:41–46` decided whether `train_step` accepts `active` with `try/except TypeError`. A
`TypeError` raised **inside** `train_step` for an unrelated reason was swallowed and the update
silently retried without `active` — so a masked-loss experiment could run unmasked and still
produce a plausible curve. `import inspect` sat at `runner.py:19` **unused**, which suggests
signature inspection was intended and never finished.

**Fix.** Finish it — `_accepts_active()` reads the signature instead. Verified: returns `True` for
every builder in `methods.py`, `False` for a hand-written 2-argument closure. No experiment in
`experiments/` defines its own `train_step`, so the fallback branch protects nothing that
currently exists; it is kept because it costs one line.

## B4 — `BOGACZ_ARCH` encodes the superseded reading `TRACED`

`model.py:94` is `hidden=(32, 32)`, `out_dim=10` — two hidden layers, ten outputs. Experiment 34
later established the configuration as **784-32-32-32-5**: three hidden layers and **five shared
outputs**, which is what makes it Domain-IL. The comment still cites exp 30, which exp 34
supersedes. Not on the critical path — the reproduction is closed — but it is a wrong constant
sitting in a shared module.

## B5 — backtracking energy assumes MSE `FIXED`

`predictive_coding.py:74` computes the energy for the step-size backtracking rule as
`0.5 * (e_next**2).sum() + Σ hidden errors`. `e_next` is `output_error(...)`, which equals the
residual **only for MSE**. Under `ce` or `hinge` the backtracking decision uses a quantity that is
not the energy. Backtracking defaults off (`x_lr_discount=1.0`), so this is latent.

## B6 — `loss_value` ignores `obj.reduction` `FIXED`

`model.py:260–272` always averages over the batch, while `batch_scale` honours
`reduction="sum"`. Reported loss and applied gradient therefore disagree under sum reduction.
Reporting only.

---

## Design questions, not bugs

**D1 — `LEGACY_SPEC` fallback.** `RESOLVED — the default is now the protocol.` `_spec()` used to
fall back to each method's pre-unification specification when `arch`/`obj` were omitted, so a
forgotten argument silently reinstated the confound the plan exists to remove.

`_spec` now takes **no method name at all** — there is no longer any path by which the
specification can depend on which rule is being built, which is the property that makes the
comparison controlled. The old specifications are reachable only by asking for them by name,
`**legacy("backprop")`.

**All eleven runnable legacy scripts (01, 02, 03, 07–14) were pinned with `legacy()`** so the
default change cannot alter what they mean; 15 was pinned too, though it is dead for an unrelated
reason. Audited mechanically by walking each call site's parentheses — the only unpinned
constructions left are in 04 and 06, which fail at import anyway.

**D2 — the protocol object.** `RESOLVED — src/protocol.py.` A frozen `Protocol` dataclass holding
every setting that must be identical across rules, varied with `dataclasses.replace` so a script
changes one line and a deviation cannot leak into the next script. `load()` / `build()` / `run()`
fill protocol values into the existing functions. Two protocol requirements are handled centrally
because every script would otherwise reimplement them: the two disjoint eval sets, and weight
snapshots at init and each task end.

`hidden` has **no default** and `proto.arch` raises with instructions if it is unset — inheriting a
width from an old experiment is the failure this project is recovering from, so the code refuses to
guess.

Deliberately not a harness: `run()` is one function, and a script needing something else calls
`run_classil` directly rather than growing an option.

**D3 — the metric grid.** `RESOLVED.` `metrics.summarise()` returns the scalar grid in one call;
`metrics.inefficiency()` implements [R31] per synapse, so the distribution comes with it;
`metrics.sem()` matches the protocol's reporting. The two per-update measurements went to
`probes.py` rather than `metrics.py`, which is pure numpy by contract: `alignment_probe` ([R1]
Fig 3b, definition taken from `knowledge_base.md` §11.4, not invented) and `weight_path_probe`.
Both wrap `train_step`, so `runner.py` needed no change.

**D4 — `UNIFIED_ARCH` has `bias=True`, `LEGACY_SPEC` PC/EqProp have `bias=False`.** Prior
experiments 20–21 ran biases off and 22 onward ran them on. The protocol says on; worth confirming
that is deliberate.

---

---

## Modules reviewed in pass 2 — no findings

`data.py`, `metrics.py`, `plotting.py`, `probes.py` all read in full. **Clean.** Each is a set of
small pure functions with the reasoning already in the docstrings — `data.label_remapper` explains
why it must not sort, `metrics.half_life` explains why it exists at all (final accuracy has no
dynamic range once everything collapses to zero), `probes` explains the live-vs-frozen prototype
distinction. Nothing to change.

`check_names.py` and `test_numpy_mirror.py` not reviewed — neither is imported by `src/` or by any
experiment.

---

---

## Scripts that no longer run at all `VERIFIED`

Found while auditing call sites, unrelated to any change made here. All were written against the
pre-refactor interface:

| Script | Why | Status |
|---|---|---|
| `04_eqprop_with_replay.py` | imports `make_eqprop_replay` — removed in the depth refactor | archived |
| `05_eqprop_generation_examples.py` | imports `eqprop_generate` — removed | archived |
| `06_eqprop_with_synthetic_replay.py` | imports `make_eqprop_synthetic`, `make_eqprop_replay` — removed | archived |
| `15_matched_accuracy_forgetting.py` | calls `run_classil` positionally against the old signature, and unpacks three values from what is now a dict | **still in `experiments/`** |

04–06 moved to `experiments/archive/ebm_replay_generation/` — they were one line of work, the EBM
as its own replay generator, not carried into the current plan. The 30-series reproduction moved
to `experiments/archive/bogacz_reproduction/` alongside it. See `experiments/archive/README.md`.

**15 was left in place** — it is not part of either archived line and its question (matched-accuracy
forgetting) is live in the current plan. It needs a runner call updated to the current signature,
not archiving. Not done: no experiment is being re-run, and script 44 will ask this
question fresh.

---

## Status

All findings closed.

| | |
|---|---|
| B1 freezing · B2 Domain-IL `active` · B3 swallowed `TypeError` | fixed, verified |
| B2b replay buffer → per-label reservoir | fixed, verified |
| B4 `BOGACZ_ARCH` | documented, value kept on purpose |
| B5 PC backtracking · B6 `loss_value` reduction | fixed |
| D1 protocol as default | done, 12 legacy scripts pinned |
| D2 protocol object · D3 metric grid | done, `src/protocol.py` + `metrics`/`probes` |
| D4 bias on/off | settled by the protocol: biases on |

**What is not settled, and is not a code problem.** `Protocol.stop_threshold = 0.9` is a
placeholder, and `Protocol.lr` is empty so each rule falls back to `METHOD_DEFAULTS` — values
inherited from the legacy era. The protocol requires a per-rule grid search matched on
steps-to-threshold. **Script 43 does that, and no comparative claim should be believed before it.**
