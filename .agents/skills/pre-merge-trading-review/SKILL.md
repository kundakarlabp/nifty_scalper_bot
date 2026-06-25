---
name: pre-merge-trading-review
description: Review a NIFTY scalper change or pull request before merge. Use for code review, risk review, backtest review, deployment readiness, or any change that can affect signals, positions, orders, losses, or live runtime stability.
---

# Pre-Merge Trading Review

Review against `AGENTS.md`, repository ownership, the actual diff, and executed validation. Do not approve based on a plausible explanation alone.

## 1. Scope and root cause

Confirm:

- the PR has one narrow category
- the stated root cause is supported by code or runtime evidence
- changed files are necessary for that cause
- unrelated refactors, renames, formatting churn, or feature additions are absent
- high-risk files were changed only when required

## 2. Architecture and ownership

Block the merge when the change:

- bypasses the authoritative runtime path
- creates a duplicate contract selector, history store, execution route, or state owner
- lets strategies call broker APIs or select contracts
- lets DataHub own history
- weakens explicit readiness or blocker reporting
- introduces a hidden fallback that changes live behavior silently

## 3. Market-data correctness

Check:

- timestamps have explicit timezone and comparable units
- freshness/staleness decisions are deterministic
- same-bar logic cannot cause accidental duplicate evaluation
- selected CE/PE symbols and tokens are consistent with expiry and strike
- subscriptions cover the executable option basket
- FULL quote/depth metadata is preserved
- polling fallback is labelled and cannot masquerade as full WebSocket data
- missing ticks, OHLC, depth, or token resolution fail visibly

## 4. Strategy and backtest validity

For strategy or optimisation changes, examine:

- look-ahead bias
- future-bar leakage
- use of incomplete candles
- timestamp alignment between spot, futures, and options
- unrealistic fills at candle extremes or untradeable prices
- bid/ask spread, slippage, brokerage, taxes, and rejected fills
- option expiry and contract-roll handling
- overfitting to one period, regime, strike, or parameter set
- selection bias from discarded failed runs
- consistency between backtest and live feature computation

Do not accept improved headline P&L without drawdown, trade count, exposure, costs, stability, and out-of-sample evidence.

## 5. Risk and execution safety

Verify that the change does not weaken:

- capital and position sizing limits
- daily drawdown or shutdown rules
- open-position and pending-order checks
- cooldowns and repeated-loss controls
- instrument-type restrictions
- idempotency and duplicate-order prevention
- partial-fill and rejection handling
- broker timeout/retry reconciliation
- bracket/exit ownership
- restart and reconnect recovery
- paper/shadow/live separation

A failed broker operation must not be reported as success.

## 6. Configuration and secrets

Check:

- no credentials or tokens are added to source, tests, logs, fixtures, or PR text
- existing configuration keys keep their semantics unless explicitly migrated
- new defaults cannot silently enable live trading or increase risk
- environment-specific behavior is documented
- `.env.example`, when used, contains dummy values only

Respect the repository's explicit `.env` rules in `AGENTS.md`.

## 7. Observability

Require actionable diagnostics for changed runtime paths:

- module and stage
- symbol and token
- instrument type
- timestamp and freshness
- mode: live, paper, or shadow
- readiness/risk decision and exact reason
- order request and broker result when applicable

Reject vague logs such as `not ready`, `failed`, or `no signal` without stage and reason.

## 8. Validation evidence

Require the exact commands and results. At minimum, expect:

```bash
python -m compileall -q src
pytest -q
```

Also require focused suites for the affected area. When external dependencies prevent execution, the PR must state what ran, what failed, why, and whether the failure is related.

Do not treat tests as valid when mocks bypass the production seam or when a test could place a live order.

## Output format

Return:

```markdown
## Verdict
APPROVE | REQUEST CHANGES | NEEDS EVIDENCE

## Blocking findings
- Finding with file/function evidence and required correction

## High-risk non-blocking findings
- Risk, likely impact, and mitigation

## Validation reviewed
- Command: result

## Trading impact
- Market data:
- Strategy:
- Risk:
- Execution:
- Deployment/recovery:

## Residual risk
- Explicit remaining uncertainty
```

Never merge automatically. The user merges only after CI, tests, architecture review, and runtime evidence are acceptable.