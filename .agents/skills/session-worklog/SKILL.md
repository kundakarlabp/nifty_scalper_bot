---
name: session-worklog
description: Create a compact durable worklog after substantial medical, research, document, software, or operational work so future chats can recover goals, decisions, artifacts, validation, unresolved risks, and next actions without replaying the entire conversation.
---

# Session Worklog

## Purpose

Preserve high-value continuity while avoiding noisy transcripts. Use after multi-step work, before switching chats, or when a project will continue later.

## What to record

- task goal and scope
- current status
- key decisions and rationale
- authoritative sources or files used
- files, branches, PRs, documents, or datasets changed
- validation performed and exact result
- unresolved questions, risks, and blockers
- single next action

## What not to record

- hidden reasoning or verbose chronological narration
- temporary hypotheses that were disproved
- credentials, access tokens, private keys, or secrets
- patient names, identifiers, images, or other protected health information
- unsupported claims of completion
- duplicated material already present in authoritative project documents

## Workflow

1. Re-read the task and final state.
2. Distinguish durable decisions from transient discussion.
3. Link to the authoritative artifact rather than copying large content.
4. Record verification evidence exactly; use `not run` or `blocked` when applicable.
5. State unresolved risk plainly.
6. Keep the entry short enough to load in a future session.

## Output template

```markdown
# Worklog — <date> — <task>

## Goal
<one paragraph>

## Current state
<done / partial / blocked with evidence>

## Decisions
- <decision and rationale>

## Artifacts
- <file, PR, document, dataset, or link>

## Validation
- <command/check>: <result>

## Residual risk
- <risk or none identified>

## Next action
<one concrete action>
```

## Validation

Before saving, confirm that the entry contains no secrets or patient identifiers and that every completion statement is supported by recorded evidence.

## Failure and uncertainty handling

When the final state is unclear, label the worklog `partial` and record the exact uncertainty. Do not turn an intended action into a completed action.
