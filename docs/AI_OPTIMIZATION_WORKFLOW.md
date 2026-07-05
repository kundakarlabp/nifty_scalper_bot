# AI Optimization Workflow for NIFTY Scalper Bot

This document defines the safe, sequential workflow for ChatGPT, Codex, Copilot, Claude Code, and human reviewers working on this repository.

The goal is **slow, evidence-based optimization** of the NIFTY options scalper, not broad refactoring or speculative profit optimization.

## Core principle

Every change must improve one of these in order:

1. Repository guardrails and validation
2. Live option data correctness
3. Contract/token/subscription correctness
4. Option OHLC hydration
5. Timestamp and same-bar evaluation
6. Quote quality: bid/ask/spread/depth/source/freshness
7. Readiness and rejection diagnostics
8. Risk and execution safety
9. Telegram/operator diagnostics
10. Strategy scoring
11. Profit optimization

Do not jump to later layers before earlier layers are stable.

## Skill chain

Use this chain for non-trivial work:

```text
grill-trading-plan
→ domain-modeling-trading
→ to-prd-trading-change
→ to-issues-trading-change
→ diagnosing-trading-bugs when bug-driven
→ tdd-trading-changes
→ codebase-design when an ownership/interface decision is required
→ runtime-contract-validation
→ pre-merge-trading-review
→ session-worklog
```

## When to use each skill

| Skill | Use when | Output |
|---|---|---|
| `grill-trading-plan` | The request is fuzzy, high-risk, or could affect live behavior | Clarified objective, non-goals, constraints, accepted risks |
| `domain-modeling-trading` | Terms or ownership are unclear | Canonical glossary and domain invariants |
| `to-prd-trading-change` | A change needs a durable spec before implementation | PRD with problem, solution, user stories, tests, out-of-scope items |
| `to-issues-trading-change` | A PRD or plan must become small implementation tasks | Vertical-slice issue breakdown |
| `diagnosing-trading-bugs` | Runtime symptom, failed test, wrong signal, duplicate order, data issue | Reproduction, hypotheses, root cause |
| `tdd-trading-changes` | Any behavior change | RED/GREEN/REFACTOR slice |
| `codebase-design` | Ownership, interface, seam, or module design is unclear | Small design choice with rejected alternatives |
| `runtime-contract-validation` | External data or cross-module payloads are involved | Boundary contract and invalid-input behavior |
| `pre-merge-trading-review` | Before merge | Verdict with blockers, validation, and residual risk |
| `session-worklog` | End of session or handoff | Durable state of work and next action |

## Sequential optimization policy

Agents must prefer one small PR at a time.

Allowed PR scope examples:

```text
readiness-diagnostics only
quote-depth propagation only
same-bar duplicate prevention only
Telegram command diagnostics only
risk guard test coverage only
runtime-contract validation for one broker response only
```

Disallowed mixed PR examples:

```text
market data + strategy optimization + execution rewrite
Telegram cleanup + risk changes
profit optimization + config defaults
large refactor + bug fix
```

## Pre-edit contract

Before editing code, write this mini-contract in the task notes or PR body:

```text
Objective:
Non-goals:
Affected runtime layer:
Owner module:
Public interface under test:
Safety invariant:
Files likely touched:
Files explicitly not touched:
Focused validation:
Rollback:
```

## Runtime layer ownership

Preserve these boundaries:

```text
InstrumentManager      -> contract selection and token resolution
MarketDataManager      -> ticks, subscriptions, quote quality, OHLC history
DataHub                -> read-only facade over current market data
StrategyRunner         -> evaluation loop and signal handoff
RiskManager            -> risk limits and telemetry
OrderManager           -> canonical live placement and lifecycle
PositionManager        -> position and pending-order state
BracketManager         -> protective exits and trailing decisions
TelegramController     -> operator commands and diagnostics
```

Do not introduce duplicate selectors, duplicate history stores, duplicate execution paths, or hidden fallbacks.

## Safety invariants

Every change must preserve:

- NIFTY spot is context-only.
- NIFTY futures are context-only unless explicitly configured otherwise.
- Only resolved NIFTY option instruments are executable.
- Readiness blockers must be specific and observable.
- Risk, capital, cooldown, open-position, and max-loss guards remain active.
- Failed broker operations cannot be reported as success.
- Paper, shadow, and live modes remain separate.
- No change silently increases live trading risk.

## Runtime contract validation

Treat external data as unknown until validated:

```text
.env / settings
broker REST response
websocket tick
polling quote fallback
instrument dump row
active option basket
DataHub read result
strategy signal
risk decision
order request
broker acknowledgement
fill/order update
Telegram command
backtest/replay input
```

Invalid input must become a safe rejection, readiness blocker, risk blocker, operator diagnostic, or bounded retry. It must not be silently coerced into tradable data.

## Required validation

Minimum validation for code changes:

```bash
python -m compileall -q src dashboard
python -m pytest -q
```

Focused tests must also be run for affected areas where available.

For documentation-only changes, validate by reviewing changed paths and ensuring no production files are touched.

## Merge policy

A PR is mergeable only when:

```text
scope is narrow
changed files match the stated objective
no production trading behavior changes unless intentionally specified
tests/validation are run or exact blockers are documented
pre-merge-trading-review has no blocking findings
residual risk is explicit
```

Do not merge because the explanation sounds plausible. Merge only when the diff and validation support it.
