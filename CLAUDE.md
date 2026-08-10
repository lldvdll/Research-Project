# Project instructions

MSc thesis. Comparing backpropagation, replay, Equilibrium Propagation and Predictive Coding
(prospective configuration) on catastrophic forgetting in continual learning.

**Read `claude_code_management/knowledge_base.md` before answering anything substantive.** It is
not auto-imported — it is ~950 lines, and importing it every session would compete with these
instructions for adherence. Read it on demand instead, and always before making a claim about what
this project has established.

Three tags govern how much weight a knowledge-base entry carries. `[SETTLED]` is verified against a
source paper or against the code. `[EMPIRICAL]` is from our own runs and is **provisional until
re-derived from logged arrays**. `[HYPOTHESIS]` is untested. `[REFUTED]` entries are kept on purpose
so they are not accidentally re-derived. **Numbers marked `≈` are not citable.**

**§9.3 lists the contradictions between the archived source documents.** Several are still open.
Do not resolve one by picking the more recent document — D1 is a case where the newest document was
the wrong one. Flag it and ask.

## Documents

| File | What it is | When to read it |
|---|---|---|
| `claude_code_management/knowledge_base.md` | Consolidated reference. Amendable. | Before any substantive claim. Start with §9.3. |
| `claude_code_management/timeline.md` | Append-only log of chats. Entries are never edited. | When asked about history. |
| `claude_code_management/presentation_plan.md` | The slide plan. Authoritative for what work happens next. | Start of any presentation or experiment work. |
| `claude_code_management/archive/` | Raw superseded chat logs. **Read-only, never edit.** | Only when adjudicating a contradiction. |
| `ref/` | The source PDFs. **Read-only.** | When a claim needs a citation. |

## Change control — the standing rule

**No code changes without explicit approval.** Propose the change and show the diff first; wait
for a yes. This applies to `src/` and `experiments/` and is enforced by `ask` rules in
`.claude/settings.json`, not by this file.

One change at a time. Each change tested on its own, with the previous experiment re-run to
confirm its figure is unchanged, before the next change starts.

## Project facts

- Data: MNIST downsampled to 14×14 (196 inputs), scaled [0,1]. **5×2 split for both scenarios.**
- Network: 196 → 64 → 10 (Class-IL) or 196 → 64 → 2 (Domain-IL). One hidden layer, tanh,
  squared error. Identical across all four learning rules — only the rule varies.
- Optimiser: plain SGD everywhere. Batch 32. Learning rate grid-searched **per rule**.
- Scenarios: Class-IL is primary, Domain-IL also run. Only the output layer differs between them.
- Controls on every forgetting run: backprop (negative) and replay (positive).
- Four methods: `backprop`, `replay`, `pc`, `eqprop`.

## Code conventions

- Interface contract: every `make_*` returns `(train_step, predict)`. `train_step(x, y)` does one
  update. `predict(x, raw=False)` returns class indices, or raw pre-argmax outputs when `raw=True`.
- Adding a model means one new `make_*` function. Experiment scripts change only the `methods` dict.
- One script per experiment. One figure per script, named from `__file__`.
- Metric definitions live in one module and are imported, never redefined per script.
- Keep it minimal. wandb, Optuna, class hierarchies and a shared `harness.py` were all tried and
  **deliberately deleted**. Do not reintroduce them or anything like them.

## How to work with me

- Plain language. No metaphors — explanations must map to variables, order of operations and
  code lines.
- Concept first, then the decision with its trade-offs, then the code.
- Separate claims from hypotheses. Anything called a claim needs a figure-level citation.
- Do not agree with me to be agreeable. If a plan is wrong, say so and say why.
- Do not start a second thread of work while the first is open. Rabbit-holing is the failure mode
  this project is recovering from.
- Ask one question at a time when something is ambiguous, and attempt the task first where possible.

## Commands

| Command | Use |
|---|---|
| `/log-session` | End of every chat. Updates the timeline and knowledge base. |
| `/experiment <id> <question>` | Scaffold a new experiment script. |
| `/slide <n>` | Work on slide n of the presentation plan. |

## Environment

Python 3.11.2 in `.venv/` at the repo root. Activate with `.venv\Scripts\Activate.ps1`
(PowerShell). Dependencies are pinned in `requirements.txt` — numpy, torch, matplotlib,
scikit-learn, pandas.

Run one experiment from inside `experiments/`, e.g. `python 04_eqprop_with_replay.py`. Each
script writes its figure next to itself, named from `__file__`. `python experiments/00_run_all.py`
runs every `.py` in that directory in sorted order.
