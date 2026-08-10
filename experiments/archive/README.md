# Archived experiments

Closed lines of work, moved out of `experiments/` so the active directory holds only what the
current plan uses. Nothing here is deleted and nothing is a dead end on principle — but none of it
is evidence, and none of it should be re-run without a reason.

`00_run_all.py` globs `experiments/*.py` non-recursively, so these no longer run in the sweep.

---

## `ebm_replay_generation/` — the EBM as its own replay generator

**The question:** can an energy-based model generate its own replay samples, so continual learning
needs no stored data? Combining the EBM and replay into one mechanism rather than bolting a buffer
onto a learning rule.

| Script | Question |
|---|---|
| `04_eqprop_with_replay` | Is EqProp better with replay? Is replay worse with EqProp? |
| `05_eqprop_generation_examples` | Can EqProp generate synthetic samples of previous tasks? |
| `06_eqprop_with_synthetic_replay` | Can those generated samples be used *as* the replay set? |

**Why archived.** The line was not carried forward into the current plan. All three are also
**dead against the current `src/`**: they import `make_eqprop_replay`, `make_eqprop_synthetic` and
`eqprop_generate`, all removed in the depth refactor. They fail at import, before running a line.

Reviving this means restoring those three functions. The idea itself is a genuine one — it is the
generative-replay branch of the literature (van de Ven 2020) — so this is archived as *not
currently in scope*, not as *refuted*.

---

## `bogacz_reproduction/` — the Song & Bogacz Fig 4d/4e reproduction

**The question:** reproduce Song & Bogacz (2024) Fig 4d/e directly, to anchor our results against
the paper's.

`30` → `31` (analysis) → `32` → `33` (fixed) → `34`, plus `standalone_fig4d`, which is the same
figure attempted outside the main harness.

**Why archived. Read this before reopening it.** The reproduction **failed** — results were
inconsistent with the paper — and **the cause was never found**. It cost hours of compute per
attempt and was the main thing that slowed the project. `CLAUDE.md` carries a standing instruction
not to re-run it.

The depth refactor of `src/` was done *for* this work, which is why the current code supports
arbitrary depth. That part was kept.

**What survives as useful direction, not as evidence:**

- Fig 4d is **Domain-IL** — two five-class tasks sharing five output units, not ten.
- Experiment 34 read the architecture as 784-32-32-32-5. `model.BOGACZ_ARCH` still encodes the
  earlier and superseded `(32,32)`/`out_dim=10` reading from experiment 30.
- Their loss is **summed** over batch and output units with no division, which is why their
  learning-rate grid runs 0.0001–0.005 where ours ran 0.005–0.5. `Objective.reduction="sum"`
  exists for this.

The productive route, if this question is worth reopening, is **not** another reproduction attempt.
It is to close the gap between a setup we control and theirs one axis at a time — see
`claude_code_management/knowledge_base.md` §6.6.4.
