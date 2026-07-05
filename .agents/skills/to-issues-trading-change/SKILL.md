---
name: to-issues-trading-change
description: Break a resolved NIFTY scalper PRD or plan into small vertical-slice implementation issues before TDD work.
---

# To Issues: Trading Change

Convert a PRD into narrow, independently reviewable slices. Keep each issue small enough for one focused PR.

## Issue template

```markdown
## Objective
One observable behavior to add, fix, or protect.

## Runtime layer
data | hydration | timestamp | quote-quality | readiness | risk | execution | telegram | deployment | docs

## Files likely touched
- source file
- test file

## Files not to touch
- high-risk or unrelated files

## Acceptance criteria
- [ ] Observable behavior
- [ ] Safety invariant
- [ ] Failure behavior
- [ ] Operator diagnostic

## TDD slice
RED:
GREEN:
REFACTOR:

## Validation
Focused command:
Full command:

## Human checkpoint
AFK | HITL
Reason:
```

## Rules

- Prefer vertical behavior slices over broad refactors.
- Do not mix unrelated runtime layers in one issue.
- Put data correctness before readiness, readiness before signal changes, and signal changes before risk or execution changes.
- Mark a slice as `HITL` when it changes broker assumptions, risk limits, runtime order behavior, or architecture ownership.
- Return 1 to 5 issues. If more are needed, create an epic first.
