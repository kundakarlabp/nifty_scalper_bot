# NIFTY Scalper Bot — repository contract

## Mission

Maintain a deterministic, observable, capital-protective Python NIFTY options trading system.

Capital protection, data correctness, and broker-state truth take priority over trade frequency or apparent profitability.

## Product boundaries

- Trade **NIFTY options only**.
- NIFTY spot and futures are context only; never place spot/futures orders.
- Every executable signal must resolve to a broker-validated option symbol and token.
- Never weaken readiness, risk, broker, or execution safeguards to increase trade count.
- Strategy profitability claims require reproducible costs, slippage, chronological out-of-sample evidence, paper/shadow evidence, and live observation where appropriate.

## Canonical ownership

| Concern | Owner |
|---|---|
| Contract discovery, expiry/strike selection, symbol/token mapping | `core/instrument_manager.py` |
| Runtime basket commit and subsystem wiring | `core/app.py` |
| Subscriptions, quote/depth/OI, freshness and OHLC hydration | `data/market_data_manager.py` |
| Tick-to-OHLC construction | `data/candle_engine.py` |
| Strategy-facing market-data reads | `data/data_hub.py` |
| Strategy orchestration/evaluation | `core/strategy_manager.py`, `strategies/*` |
| Risk limits and sizing | `risk/*` |
| Live placement and broker lifecycle | `execution/order_manager.py` |
| Position/pending-order state | `execution/position_manager.py` |
| Protective exits/trailing/recovery | `execution/bracket_manager.py` |
| Operator commands/diagnostics | `notifications/*` |

Do not create competing selectors, instrument caches, history owners, readiness owners, position owners, or execution paths.

## Runtime path

```text
InstrumentManager
→ App commits active basket
→ MarketDataManager subscribes/hydrates
→ CandleEngine builds bars
→ DataHub exposes prepared context
→ StrategyManager / StrategyRunner evaluate
→ Risk validates/sizes
→ OrderManager submits/reconciles
→ PositionManager owns position state
→ BracketManager manages exits
```

## Hard invariants

### Market data

- WebSocket FULL data is the primary live source.
- Preserve symbol/token identity, timestamps, bid/ask/spread, depth, OI, source, freshness, stale state, and tradable-quote state where required.
- Normalize at the owning boundary; do not repeatedly reshape data downstream.
- Freshness is monotonic: older polling, cached, synthetic, or LTP-only data cannot overwrite fresher FULL data.
- A degraded quote remains explicitly degraded.
- Reconnect and basket rotation must restore the correct subscriptions.

### Strategy

- Strategies consume prepared context; they do not select contracts, fetch broker instruments/history, or place orders.
- Underlying direction comes from the canonical direction/context authority, not option-premium direction alone.
- No future bar, incomplete-bar leakage, test-period leakage, or look-ahead.
- Identical prepared input/state must produce deterministic decisions.
- Missing optional context may reduce confidence; it must not become a new hard blocker unless explicitly required by the active strategy.

### Risk and execution

For a BUY:

```text
stop_loss < confirmed_entry_or_fill < take_profit
```

Position size is bounded by both risk and available margin.

- Broker-confirmed fills/positions own executed quantity.
- Requested or acknowledged quantity is not proof of a fill.
- Duplicate entry intent must remain idempotent.
- Rejected/cancelled/timed-out/partial orders must remain explicit.
- Exit/bracket logic uses reconciled position state and cannot create an independent entry path.
- Failed broker operations must never be reported as success.

### Blockers and observability

Use the gate sequence:

```text
data → strategy → risk → execution
```

Every material block should identify stage, symbol, reason/code, expected value, observed value, recoverability, and owner where available. Fix the earliest incorrect layer rather than adding downstream compensating gates.

## Change discipline

Before a non-trivial edit:

1. Read `docs/AGENT_START_HERE.md` and `docs/REPO_MAP.md`.
2. Use `scripts/agent_context.py` to rank relevant files/tests when the location is not already obvious.
3. Load one primary skill; add another only when the task genuinely crosses concerns.
4. Read only matching entries from `docs/ENGINEERING_FAILURE_PATTERNS.md`.
5. Trace symptom → owner → downstream safety effect.
6. Define one observable invariant and the smallest coherent change.
7. Add or identify a regression that proves the defect when practical.
8. Do not mix unrelated cleanup, strategy tuning, architecture changes, and runtime fixes.

Prefer existing owners/public interfaces. Do not add helpers, wrappers, compatibility layers, dependencies, or managers unless the requested behavior cannot be expressed cleanly through the existing architecture.

## Validation

Use the repository tooling rather than hand-building validation commands:

```bash
python scripts/agent_check.py --files <changed files> --run focused
python scripts/agent_check.py --files <changed files> --run full
```

The focused ring is risk-aware and runs changed-file quality checks before affected tests. Final-head GitHub CI remains authoritative before merge.

Before merge, use the merge guard with the exact validated base/head SHAs:

```bash
python scripts/agent_merge_guard.py \
  --validated-base <base-sha> \
  --validated-head <head-sha>
```

Never claim a test, replay, backtest, deployment, production state, or merge is verified unless it was actually observed.

## Skills and deeper procedures

- Routing: `docs/AGENT_START_HERE.md`
- Architecture/navigation: `docs/REPO_MAP.md`
- ChatGPT/GitHub workflow: `docs/CHATGPT_CODE_WORKFLOW.md`
- Optimization workflow: `docs/AI_OPTIMIZATION_WORKFLOW.md`
- Recurring failures: `docs/ENGINEERING_FAILURE_PATTERNS.md`
- Tooling design: `docs/AGENT_TOOLING_DESIGN.md`
- Specialist skills: `.agents/skills/README.md`

Keep this file compact. Detailed procedures belong in those documents/skills and executable guards.

## Production and secrets

Production diagnosis is read-only by default. Do not restart, redeploy, enable live execution, change credentials, or weaken risk limits merely to inspect state.

Never commit or print secrets, tokens, broker session material, private account data, or operator-only credentials.

Cross-repository edits are required only when the requested behavior changes a real shared interface, schema, protocol, deployment contract, or user workflow.
