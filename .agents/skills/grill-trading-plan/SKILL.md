---
name: grill-trading-plan
description: Stress-test a NIFTY scalper plan before implementation. Use when the request is fuzzy, broad, safety-critical, profit-oriented, or could affect market data, strategy, risk, execution, deployment, or live runtime behavior.
disable-model-invocation: true
---

# Grill Trading Plan

Use this skill before writing a PRD or code when the plan is not yet sharp enough.

This is adapted from the spirit of Matt Pocock's `grill-me` / `grill-with-docs` approach, but constrained for a live NIFTY options trading bot.

## Rule

Ask one high-value question at a time only when the answer is not already available from the repository.

Questions the codebase can answer must be answered by inspecting the codebase, not by asking the user.

## First inspect

Before asking the user, inspect:

```text
AGENTS.md
docs/AGENT_START_HERE.md
docs/REPO_MAP.md
docs/AI_OPTIMIZATION_WORKFLOW.md
relevant source files
relevant tests
recent PR/issue context if available
```

## Grill dimensions

Resolve these before implementation:

| Dimension | Required answer |
|---|---|
| Objective | What exact behavior should change? |
| Non-goals | What must not change? |
| Runtime layer | data, hydration, strategy, risk, execution, Telegram, deployment |
| Owner module | Which module owns the behavior? |
| Public interface | What should tests call? |
| Safety invariant | What must remain true to protect capital? |
| Failure mode | What happens when input/broker/runtime state is bad? |
| Observability | What exact logs/blockers should operator see? |
| Validation | Which focused tests prove the change? |
| Rollback | How to revert safely? |

## Trading-specific questions

Prefer questions like:

- Is this change allowed to affect live order placement, or only paper/shadow behavior?
- Should this blocker prevent new entries only, or also exits?
- Is the source of truth broker state, local state, or an already-defined reconciliation result?
- What is the accepted behavior when quote quality is LTP-only?
- Which module owns this state transition?
- Which exact Telegram diagnostic should expose the result?

Avoid vague questions like:

- Should I improve this?
- Do you want a refactor?
- Should I add more indicators?

## Output format

Return this before moving to PRD or implementation:

```markdown
## Clarified plan
Objective:
Non-goals:
Runtime layer:
Owner module:
Public interface:
Safety invariant:
Failure behavior:
Observability:
Validation:
Rollback:

## Open question
<only if still required>
```

## Stop conditions

Do not proceed to implementation when:

- live execution impact is unclear
- owner module is unclear
- validation command is unclear
- the request combines unrelated runtime layers
- the plan depends on changing risk/profit behavior without explicit user acceptance
