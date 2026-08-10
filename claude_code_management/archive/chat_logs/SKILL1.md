---
name: log-session
description: End-of-chat ritual. Appends an entry to timeline.md, amends knowledge_base.md with anything new, and proposes a commit. Run this before closing a session.
disable-model-invocation: true
argument-hint: "[optional note on what this session covered]"
allowed-tools: Bash(git status *) Bash(git diff *) Bash(git log *)
---

# Log this session

## Uncommitted state

!`git status --short`

## Instructions

Work through these in order. Show me each proposed edit before writing it.

### 1. Timeline entry

Append to `claude_code_management/timeline.md`. **Never edit an existing entry** — if this session
corrected something from an earlier entry, that goes in the knowledge base, not the timeline.

Format:

```
### [YYYY-MM-DD] #NNN — <objective, one line>
**Outcome:** <one sentence>
```

Use today's date and the next number in sequence. If this session had more than one distinct
objective, write one entry per objective.

The outcome must be a **single sentence** and must say what actually happened, including if the
answer was negative or the work was abandoned. "Investigated X" is not an outcome. "Found X does
not hold because Y" is.

### 2. Knowledge base amendments

Amend `claude_code_management/knowledge_base.md` with anything from this session that a future
session would need. That means:

- **New results** — add them, with the experiment ID and date.
- **Corrections** — where this session contradicted something already in the file, replace the old
  claim and record that it was corrected, rather than silently deleting it.
- **Retired framings** — if an idea was demoted or parked, mark it and say why, so it is not
  rediscovered as new.
- **New references** — add to the reference list at the end, with a reference key and a
  one-sentence summary of the source.
- **Consolidation** — if two sections now say the same thing, merge them.

Do not append a session log to the knowledge base. It is a structured reference, not a diary.
That is what the timeline is for.

### 3. Presentation plan

If this session changed the slide plan, an experiment definition, or the experiment order, update
`claude_code_management/presentation_plan.md` to match. Flag any slide whose experiments are now
complete.

### 4. Commit

Propose a commit message covering the document changes and any code changes from this session.
Show it to me; do not run `git commit` yourself.

### 5. Open threads

Finish by listing, in plain bullets:

- anything left unresolved that a future session needs to pick up
- any question this session raised that was deliberately **not** chased

Keep that list short. If it is longer than four items, the session sprawled and that is worth
saying out loud.
