# AI Optimization Workflow for NIFTY Scalper Bot

This document defines the safe workflow for ChatGPT, Codex, Copilot, Claude Code, and human reviewers working on this repository.

The goal is evidence-based optimization of the NIFTY options scalper without broad refactoring, duplicated ownership, or speculative profit optimization.

## Core principle

Every change should improve the earliest unstable layer first:

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

Do not jump to later layers while an earlier layer is the verified bottleneck.

## Minimal-context workflow

For non-trivial work:

```text
rank context with scripts/agent_context.py
→ identify the owner and one primary skill
→ reproduce or define one observable behavior
→ make the smallest coherent change
→ run focused validation
→ run full validation/final-head CI before merge
```

Do **not** automatically load or execute the full skill catalog. Skills are selective procedures.

### Primary routing

| Task | Primary skill | Optional secondary skill |
|---|---|---|
| Runtime symptom, failed test, stale data, wrong signal, duplicate order, no-trade state | `diagnosing-trading-bugs` | `runtime-contract-validation` for payload/boundary issues; `codebase-design` for ownership defects |
| Well-scoped behavior change | `tdd-trading-changes` | `runtime-contract-validation` or `codebase-design` only when needed |
| Ownership, SSOT, interface, seam, or duplicate-path change | `codebase-design` | `tdd-trading-changes` for implementation |
| Broker/config/external/cross-module contract change | `runtime-contract-validation` | `tdd-trading-changes` for implementation |
| Fuzzy or safety-critical feature | `grill-trading-plan` | `domain-modeling-trading` only if terminology/state ownership is unclear |
| Durable feature specification or multi-PR decomposition | `to-prd-trading-change`, then `to-issues-trading-change` if needed | Use only when the task is large enough to benefit |
| PR/diff review | `pre-merge-trading-review` | none by default |
| Handoff or context reset | `session-worklog` | none |

## Sequential optimization policy

Prefer one small PR at a time.

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

For non-trivial edits, record the minimum contract required to constrain the change:

```text
Objective:
Owner module:
Observable behavior:
Safety invariant:
Non-goals:
Focused validation:
```

Add public-interface, rollback, state-transition, deployment, or compatibility details only when the change requires them.

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

## Validation

Generate the validation plan for changed files:

```bash
python scripts/agent_check.py --files path/to/changed.py
```

Fast iteration ring:

```bash
python scripts/agent_check.py --files path/to/changed.py --run focused
```

Pre-merge ring:

```bash
python scripts/agent_check.py --files path/to/changed.py --run full
```

`focused` runs compilation plus the selected affected-area tests. `full` also runs the complete repository suite. Final-head CI remains authoritative before merge.

For documentation-only changes, review the changed paths and ensure no production trading files are touched.

## Merge policy

A PR is mergeable only when:

```text
scope is narrow
changed files match the stated objective
no production trading behavior changes unless intentionally specified
focused validation is green
full validation/final-head CI is green or an exact external blocker is documented
pre-merge review has no blocking findings
residual risk is explicit
```

Do not merge because the explanation sounds plausible. Merge only when the diff and validation support it.
