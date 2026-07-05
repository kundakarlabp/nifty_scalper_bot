---
name: to-prd-trading-change
description: Convert resolved NIFTY scalper context into a concise PRD before implementation. Use after grill/domain-modeling when a change needs durable requirements, tests, non-goals, rollback, and module impact mapping.
---

# To PRD: Trading Change

Use this after `grill-trading-plan` and `domain-modeling-trading` have resolved the plan.

Do not ask the user again for information the repository can answer. Inspect the relevant files and tests first.

## PRD structure

Create a PRD with this structure:

```markdown
# PRD: <change name>

## 1. Problem statement
What is broken, unsafe, unclear, or missing?

## 2. Current behavior
Observed behavior with file/function/runtime evidence.

## 3. Desired behavior
Specific observable behavior after the change.

## 4. Non-goals
What must not be changed.

## 5. Domain and ownership
Owner module:
Affected runtime path:
Public interface:
State transition:
Safety invariant:

## 6. Operator stories
1. As the operator, ...
2. As the strategy runner, ...
3. As the execution layer, ...

## 7. Functional requirements
FR-1:
FR-2:
FR-3:

## 8. Failure behavior
Bad input:
Missing token:
Stale quote:
Broker rejection:
Timeout:
Restart/reconnect:

## 9. Observability
Logs:
Metrics:
Telegram diagnostics:
Readiness/blocker names:

## 10. Testing plan
Focused tests:
Negative tests:
Fixture/replay tests:
Validation commands:

## 11. Out of scope
Explicit exclusions.

## 12. Implementation notes
Smallest safe module changes.
Deep-module or seam opportunity if any.

## 13. Rollback
How to revert safely.
```

## Impact classification

Every PRD must state whether the change can affect:

```text
runtime entries
runtime exits
paper mode
shadow mode
backtest only
Telegram only
deployment only
```

Changes affecting runtime entries or exits require stronger review and focused tests.

## Output rule

The PRD is not an implementation plan. It defines observable behavior, ownership, tests, and non-goals. It must not prescribe broad refactors.

## Quality bar

A good PRD lets an agent implement a narrow TDD slice without guessing:

- which file owns the behavior
- which public interface to test
- what failure looks like
- which safety invariant must hold
- which validation commands prove it
- what is explicitly out of scope
