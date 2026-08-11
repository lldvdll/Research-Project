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
| `claude_code_management/now.md` | **The dashboard.** What is running, next three actions, checklist, recent decisions. Deliberately short — keep it that way. | **First, every session.** |
| `claude_code_management/current_state.md` | **Where the project actually is.** Results, corrections, decisions, running order. | Second. It supersedes the plan where they disagree. |
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

- Data: MNIST downsampled to 14×14 (196 inputs), scaled [0,1]. **2×5 split — 2 tasks, 5 classes
  each — for both scenarios.**
- Network: 196 → H → 10 (Class-IL) or 196 → H → 5 (Domain-IL). One hidden layer, tanh, squared
  error. Identical across all four learning rules — only the rule varies.
- H was settled by script 41 rather than inherited, which matters because prior work disagrees
  with itself about width (`knowledge_base.md` §6.6.3). Note that a capacity sweep run at too
  short a budget produces a flat region that is indistinguishable from a capacity ceiling —
  measure convergence first.
- Optimiser: plain SGD everywhere. Batch 32. Learning rate grid-searched **per rule**.
- Scenarios: **Domain-IL is primary** (changed 2026-08-11). It is what Song & Bogacz use, and
  output-layer suppression — which is rule-independent — cannot occur there, leaving
  representation drift, which is what a learning rule acts on. Class-IL is run where relevant and
  its result explains why it is a different question. Only the output layer differs between them.
- **Hidden width H = 32**, fixed by script 41 on capacity grounds. Joint ceiling 93.6% (Class-IL)
  and 94.3% (Domain-IL) — retention is read against those, not against 100%.
- Controls on every forgetting run: backprop (negative) and replay (positive).
- Four methods: `backprop`, `replay`, `pc`, `eqprop`.

## Evidence standard — the line under previous work

**No prior result is evidence.** Everything before this point used inconsistent setups, contains
internal contradictions, and ran on code that is not trusted (`knowledge_base.md` §6.6, §6.7).
Prior work informs *direction and experiment design only*.

Any claim that reaches a slide is backed by a **fresh experiment, designed for that slide, run under
the new protocol**. Do not carry a number forward from `knowledge_base.md` §6. Do not re-run an old
experiment to settle a question — redesign it for the slide that needs it.

**Do not re-run the Song & Bogacz reproduction** (§6.5). It costs hours, failed, and the cause was
never found.

## Code conventions

- **The current interface is not documented and the code is not trusted.** `src/` was refactored for
  arbitrary depth during the failed reproduction and now uses `Arch` / `Objective` dataclasses and a
  `run_classil` runner. The older `make_* → (train_step, predict)` contract describes the
  pre-refactor code only. Read the modules before relying on either. A check-and-simplify pass is
  pending, ahead of or during the first experiment of the new series.
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
