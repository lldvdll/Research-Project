# Porting the project to Claude Code (VS Code)

Written 2026-08-10 against the current Claude Code docs. Everything here is verified from
https://code.claude.com/docs/en/ — the relevant pages are `memory`, `permissions`,
`permission-modes`, `settings`, `skills`, `model-config`.

---

## 0. Assessment of your plan

| Your item | Verdict |
|---|---|
| `chat_logs/` folder with all md files | Right instinct, wrong destination. Consolidate **once** into `knowledge_base.md` + `timeline.md`, then move the raw logs to an archive that is **not** loaded into context. Nine overlapping handoff docs in context every session is exactly the drift you are trying to escape. |
| VS Code extension installed and signed in | Done. One extension setting to change (step 5). |
| No extensions "for efficiency" | Fine. In Claude Code, "extensions" means **plugins**. For a small single-language research repo they are mostly overhead. Step 8 names the one or two worth checking and says to skip the rest. |
| Not blocked API use | Nothing to do. |
| No system prompt | **Correction: Claude Code has no user-editable system prompt.** `CLAUDE.md` is the mechanism — it is delivered as a *user message* after the system prompt, loaded every session. `--append-system-prompt` exists but must be passed on every launch, so it suits scripts, not interactive work. Step 2. |
| No project planning files | Step 2 and step 6. |
| Empty `claude_code_management/` folder | Keep it for **documents**, but Claude Code will not auto-discover configuration there. Config must live in `.claude/` and `CLAUDE.md` at the repo root. Both are used below. |
| No workflow decided | Steps 2–4 and 7. |

**Two things your plan is missing.**

1. **`CLAUDE.md` is not enforcement.** The docs are explicit: *"Permission rules are enforced by Claude Code, not by the model. Instructions in your prompt or CLAUDE.md shape what Claude tries to do, but they don't change what Claude Code allows."* Writing "don't change code without approval" in `CLAUDE.md` is a suggestion. To make it a boundary you need `permissions.ask` rules plus plan mode. Step 3.

2. **Auto mode becomes the default permission mode on 14 August 2026** for Pro, Max and Team plans. That is four days away. A default you set yourself stays in place; if you set nothing, new sessions will start auto-approving. Setting `defaultMode` explicitly in step 3 is time-sensitive.

---

## 1. Layout

Create this at the repo root. Directories you already have are marked.

```
<repo root>/
├── CLAUDE.md                          # NEW — loaded every session, keep under ~150 lines
├── .claude/
│   ├── settings.json                  # NEW — committed. Permissions + default mode.
│   ├── settings.local.json            # NEW — gitignored. Personal overrides.
│   ├── rules/
│   │   ├── code-changes.md            # NEW — loads when Claude touches src/
│   │   └── experiments.md             # NEW — loads when Claude touches experiments/
│   └── skills/
│       ├── log-session/SKILL.md       # NEW — /log-session
│       ├── experiment/SKILL.md        # NEW — /experiment
│       └── slide/SKILL.md             # NEW — /slide
├── claude_code_management/            # EXISTS — living documents
│   ├── knowledge_base.md              # NEW (step 6)
│   ├── timeline.md                    # NEW (step 6)
│   ├── presentation_plan.md           # NEW (step 6)
│   └── archive/                       # NEW — raw chat_logs move here, never imported
├── refs/                              # the four PDFs
├── src/
├── experiments/
└── results/
```

```bash
mkdir -p .claude/rules .claude/skills claude_code_management/archive refs results
```

Move the PDFs into `refs/` if they are not already there, and move `chat_logs/` under
`claude_code_management/archive/` **after** step 6, not before.

---

## 2. `CLAUDE.md`

Copy `CLAUDE.md` from this bundle to the repo root. Edit the two lines marked `<<<`.

Why it is shaped this way:

- Under 150 lines. The docs state adherence drops on longer files.
- Facts and standing rules only. Anything that is a multi-step procedure went into a skill (step 7);
  anything that only matters in one directory went into `.claude/rules/` (step 4).
- One `@` import, of `knowledge_base.md`. Imports expand into context at launch, so this gives you
  the "knowledge base loaded at the start of every chat" behaviour you asked for.
  **If you later find Claude ignoring the rules, delete that import line** — a large import
  competes with the instructions for adherence, and Claude can read the file on demand instead.
- `timeline.md` and `presentation_plan.md` are *not* imported. They are referenced by path so Claude
  reads them when relevant. Importing all three would roughly triple the standing context cost for
  no benefit.

After copying, run `/context` in a session and confirm `CLAUDE.md` and `knowledge_base.md` both
appear under **Memory files**.

---

## 3. `.claude/settings.json` — the change-control boundary

Copy `.claude/settings.json` from this bundle. This is the part that actually enforces
"code only changes if I approve it".

Three layers, in the order Claude Code evaluates them (**deny → ask → allow**):

| Layer | What it does here |
|---|---|
| `"defaultMode": "plan"` | Every session starts in plan mode. Claude reads, explores and proposes; it does not edit until you approve a plan. Also immunises you against the 14 August auto-mode default. |
| `permissions.ask` | `Edit(/src/**)` and `Edit(/experiments/**)` force a prompt **even after you approve a plan and even in acceptEdits mode**. Ask rules beat allow rules. This is the hard boundary. |
| `permissions.deny` | `Edit(/refs/**)` and `Edit(/claude_code_management/archive/**)` — the papers and the frozen logs can never be modified. |
| `permissions.allow` | A short list of read-only-ish shell commands so you are not prompted for `python -c` and `pip list`. Deliberately short. |

Notes on the syntax, from the docs:

- A leading single `/` anchors at the **project root** when the rule is in `.claude/settings.json`.
  `Edit(/src/**)` means `<repo>/src/`, not the filesystem root.
- Use `Edit(...)`, never `Write(...)`. `Edit` rules cover every file-editing tool; a `Write(path)`
  rule is accepted but never consulted, and produces a startup warning.
- Allow rules in a project settings file only take effect after you accept the **workspace trust
  dialog**. You will see it the first time you open the folder. Deny and ask rules apply immediately.

**Toggle for when you want to move fast:** `Shift+Tab` cycles plan → acceptEdits. The `ask` rules on
`src/` and `experiments/` still fire, so you keep approval on code while docs and notes flow freely.
That is the intended daily rhythm.

Also copy `.claude/settings.local.json` and add these to `.gitignore`:

```
.claude/settings.local.json
CLAUDE.local.md
data/
__pycache__/
```

---

## 4. `.claude/rules/`

Copy both files. These use `paths:` frontmatter, so they load only when Claude reads or edits a
matching file. They cost nothing the rest of the time.

- `code-changes.md` (`paths: src/**`) — the one-change-at-a-time discipline, the modularisation
  brief, the freeze rule, the re-run-the-previous-experiment rule.
- `experiments.md` (`paths: experiments/**`) — one question per script, one figure per script,
  the metric definitions, the measurement traps.

Path-scoped rules are **not** re-injected after `/compact`. They reload the next time Claude touches
a matching file, which in practice is immediately. Worth knowing if behaviour seems to slip
mid-session.

---

## 5. VS Code extension settings

Open Settings (`Cmd/Ctrl+,`), search `claudeCode`.

| Setting | Value | Why |
|---|---|---|
| `claudeCode.initialPermissionMode` | `plan` | Matches `defaultMode`. Belt and braces. |
| **Allow dangerously skip permissions** | leave **off** | Keeps bypass mode out of the `Shift+Tab` cycle entirely. |

The mode indicator sits at the bottom of the prompt box; click it to switch mid-session. The labels
map: Manual = `default`, Edit automatically = `acceptEdits`, Plan = `plan`.

---

## 6. Consolidate `chat_logs/` — your first session

Do this as one dedicated session. It is a large-context reading task, so use the strongest setup.

```
/model opus
/effort xhigh
```

Then paste this prompt:

> Read every file in `chat_logs/`. They are overlapping handoff documents from a series of chats on
> the same MSc project; expect duplication and at least two places where a later document corrects
> an earlier one.
>
> Produce three files in `claude_code_management/`:
>
> 1. `knowledge_base.md` — one consolidated, de-duplicated, structured reference. Where two
>    documents conflict, keep the later claim and record the correction explicitly rather than
>    silently dropping the old one. Every factual claim that came from a paper gets a reference key;
>    put a reference list at the end with a one-sentence summary of each source. Target 400 lines.
> 2. `timeline.md` — one entry per distinct objective across the logs, in order, each with a
>    one-sentence outcome. Entries are append-only from now on.
> 3. `presentation_plan.md` — the current slide plan, carried over verbatim if one of the logs
>    already contains a finished version.
>
> Do not invent content to fill gaps. Where the logs are silent, say so. When you are done, list
> every claim you found that two documents disagreed on, so I can adjudicate.

Review the output, then:

```bash
git mv chat_logs claude_code_management/archive/chat_logs
git add -A && git commit -m "Consolidate chat logs into knowledge base and timeline"
```

The archive is now covered by the `deny` rule from step 3, so it can never be silently edited.

---

## 7. Skills

Copy the three `SKILL.md` files. All three set `disable-model-invocation: true`, meaning **only you**
can trigger them — Claude will not decide to run them on its own. That is deliberate: these have side
effects on your documents.

| Command | Purpose | Model / effort |
|---|---|---|
| `/log-session` | End-of-chat ritual. Appends a `timeline.md` entry, amends `knowledge_base.md`, proposes a commit. | inherits |
| `/experiment <id> <question>` | Scaffolds one experiment script from the conventions, refuses if the question is not a single sentence. | inherits |
| `/slide <n>` | Loads slide `n` from `presentation_plan.md`, lists its reading and experiments, and plans the work. | `opus` / `xhigh` |

`/slide` is where the model-and-effort answer to your "token efficient but maximum thinking" question
lives: **skills can override model and effort in frontmatter**. Design work runs on Opus at xhigh;
everything else stays on your session default.

Skills reload live — edit a `SKILL.md` and it takes effect without restarting.

---

## 8. Model, effort, and plugins

### Models

Set your session default once:

```
/model opusplan
```

`opusplan` uses Opus during plan mode and switches to Sonnet for execution. Given your workflow —
think hard about the design, then make a small controlled edit — this is close to ideal, and it is
the single highest-leverage token decision available.

Override per task:

| Situation | Command |
|---|---|
| Frontier research question, experiment design, reading a paper closely | `/model opus` + `/effort xhigh` |
| One-off deep think without changing the session | put the word `ultrathink` in your prompt |
| Routine edits, plotting, refactors | `/model sonnet` (or just let `opusplan` do it) |

`ultrathink` is a recognised keyword; "think hard" and "think more" are **not** — they are passed
through as ordinary text.

### Effort

Default is `high`. `/effort` opens a slider. `xhigh` for design sessions, `max` only when you are
stuck (the docs warn it is prone to overthinking). Changing effort **invalidates the prompt cache**,
so set it at the start of a session rather than mid-way.

### Plugins

Honest answer: for a single-language research repo with a handful of files, most plugins are
overhead you do not need, and you have already been burned once by machinery.

Add the official marketplace and look at exactly one category:

```
/plugin marketplace add anthropics/claude-plugins-official
/plugin
```

- **Code intelligence** — if there is a Python entry, install it. It gives Claude an LSP and reduces
  how many files it has to read, which is a real token saving on a codebase it does not know yet.
- **`skill-creator`** — only if you later want to A/B test your skills. Skip for now.
- Everything else: skip.

---

## 9. Verify

Start a session in the repo root and run these:

| Command | Confirms |
|---|---|
| `/context` | `CLAUDE.md` and `knowledge_base.md` appear under Memory files. If not, the import path is wrong. |
| `/permissions` | The ask and deny rules are listed, with `.claude/settings.json` as the source. |
| `/status` | Model is `opusplan`, and the account is right. |
| `/doctor` | Catches malformed settings, oversized `CLAUDE.md`, and skill-listing overflow. |
| `/memory` | Shows the memory files, and lets you toggle auto memory. |

Then test the boundary directly:

> Add a comment at the top of `src/predictive_coding.py` saying "test".

You should get a **permission prompt**, not a silent edit. If it edits silently, the `ask` rule is
not matching — check that you wrote `Edit(/src/**)` and not `Write(/src/**)`, and that `/permissions`
shows the rule.

---

## 10. Auto memory — know it exists

Claude Code writes its own notes to `~/.claude/projects/<project>/memory/MEMORY.md`, separate from
your `CLAUDE.md`. It is on by default, machine-local, and not in git.

Leave it on — it picks up build commands and debugging patterns you would otherwise re-explain. But
it is **not** the system of record. `knowledge_base.md` is. Audit it with `/memory` occasionally, and
if it starts contradicting the knowledge base, edit or delete the offending file; it is plain markdown.

To turn it off entirely, put `{"autoMemoryEnabled": false}` in `.claude/settings.json`.

---

## 11. First working session

```
/model opusplan
/effort high
```

> Read `claude_code_management/presentation_plan.md` and `knowledge_base.md`. We are picking up at
> the code cleanup and E-metrics. Stay in plan mode.
>
> Read every file in `src/` and `experiments/` and give me: (a) what each module does in one line,
> (b) where the conditionals inside the core functions of `eqprop.py` and `predictive_coding.py`
> could be resolved at construction time instead, (c) which experiment scripts still run and which
> are broken. Do not propose changes yet — I want the map first.

Then, one change at a time, per `.claude/rules/code-changes.md`.

### The daily rhythm

1. Start in plan mode. Ask for a map or a plan first.
2. Approve the plan. `Shift+Tab` to acceptEdits if you want docs to flow.
3. Every `src/` edit prompts. Read the diff. Approve or redirect.
4. Re-run the previous experiment to confirm the figure is unchanged.
5. Commit that one change.
6. `/log-session` before you close the tab.
