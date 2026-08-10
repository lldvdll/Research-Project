---
name: slide
description: Work on one slide of the presentation plan. Loads the slide's contents, reading list and experiments, then plans the work for that slide only.
disable-model-invocation: true
argument-hint: "<slide number>"
arguments:
  - n
model: opus
effort: xhigh
---

# Slide $n

Deep-thinking mode. This is design work, so take the time.

## Instructions

### 1. Load the slide

Read `claude_code_management/presentation_plan.md` and extract the section for slide $n only.
Report back:

- whether it is mandatory or optional
- its bullet-point contents as currently written
- its reading list, with reference keys
- its experiments, and whether each has been run

### 2. Do the reading

For each reference the slide cites, read the relevant section of the PDF in `ref/`. Do not work
from the knowledge base summary alone for anything the slide states as a **claim** — go to the
source and confirm what the figure or passage actually says.

Report, for each claim on the slide:

- the exact figure or section that supports it
- whether the paper actually says what we think it says
- whether the claim is a claim, a hypothesis, or our own inference

Flag anything we have been asserting that the source does not support. That has happened before
in this project and catching it is the point of this step.

### 3. Plan the work

Produce, for this slide only:

- the final bullet points, in the words that will appear
- any figure or diagram that must be built, described precisely enough to build it
- any experiment that must be run or re-run, with its one-sentence question
- what is missing and cannot be resolved without a decision from me

### 4. Stop

Do not move to slide $((n+1)). Do not start refining slides already done. One slide at a time is
the whole discipline here.

Stay in plan mode. Nothing gets written to `src/` or `experiments/` in this session without
separate approval.
