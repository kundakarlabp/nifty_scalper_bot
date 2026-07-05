---
name: domain-modeling-trading
description: Capture and refine the NIFTY scalper's domain language, ownership boundaries, state model, and irreversible design decisions. Use before PRD/design work or when agents confuse symbols, instruments, states, readiness, orders, or broker concepts.
---

# Domain Modeling for the NIFTY Scalper

Use this skill to prevent agents from using loose language that creates unsafe code.

The output should be durable vocabulary and invariants, not implementation code.

## Canonical vocabulary

When a term appears in a task, classify it:

| Term group | Examples | Required clarity |
|---|---|---|
| Instruments | NIFTY spot, NIFTY future, NIFTY option, CE, PE, ATM, expiry | Context-only vs executable |
| Tokens/symbols | internal symbol, Zerodha quote symbol, instrument token, tradingsymbol | Which module resolves it |
| Market data | tick, quote, LTP, bid, ask, spread, depth, OI, IV, OHLC | Source, freshness, quality |
| State | readiness, arming, cooldown, open position, pending order, kill switch | Owner and transition |
| Signals | score, direction, entry, SL, target, blockers, reasons | Required output schema |
| Execution | order request, acknowledgement, fill, rejection, timeout, retry | Idempotency and reconciliation |
| Modes | LIVE, PAPER, SHADOW, backtest | Separation and allowed side effects |

## Ownership map

Always preserve:

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

## State model minimum

For any stateful change, define:

```text
state name
owner module
entry condition
exit condition
allowed transitions
forbidden transitions
persistence/restart behavior
operator diagnostic
failure behavior
```

Examples of states that must not be implicit booleans scattered across modules:

```text
market data degraded
option basket hydrated
ready to evaluate
signal rejected
risk blocked
order pending
position open
cooldown active
emergency stopped
```

## Irreversible or ADR-worthy decisions

Create or update an architecture decision note when changing:

- canonical symbol format
- contract selection ownership
- broker response interpretation
- paper/shadow/live semantics
- order idempotency model
- restart/recovery behavior
- risk sizing semantics
- strategy signal schema
- persisted state format

## Output format

```markdown
## Domain model notes
Canonical terms:
- term: meaning, owner, executable/context-only status if relevant

State model:
- state: owner, transitions, diagnostics

Invariants:
- invariant 1
- invariant 2

Architecture decision needed:
YES/NO
Reason:
```

## Reject

Reject or rewrite language like:

```text
trade NIFTY
use future as fallback option
ignore missing token
if no bid/ask use LTP as tradable quote
order probably succeeded
not ready
some issue in websocket
```

Replace with explicit instrument, source, state, blocker, and owner language.
