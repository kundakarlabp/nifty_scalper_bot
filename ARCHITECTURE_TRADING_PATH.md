# Nifty Scalper Bot — Canonical Trading Path

## Authoritative live path

1. **Composition and startup**
   - `src/nifty_scalper_bot/core/app.py`
   - Constructs the market-data, strategy, execution, risk and notification components.

2. **Market data and routing**
   - `data/market_data_manager.py`
   - `data/data_hub.py`

3. **Signal generation**
   - `strategies/runner.py`
   - `StrategyRunner` validates strategy state and creates an immutable `TradePlan`.
   - The runner submits only through `OrderManager.submit_trade_plan_result()`.
   - It must not call `place_order()`, `execute_market_order()` or dynamic-TP helpers directly.

4. **Entry execution authority**
   - Public facade: `execution/order_manager.py`
   - Runtime owner: `execution/runtime_order_manager.py::RuntimeOrderManager`
   - Responsibilities:
     - preflight, spread, margin and risk validation;
     - broker submission and acknowledgement reconciliation;
     - bounded broker-rejection recovery;
     - partial-entry remainder cancellation;
     - actual fill quantity/VWAP hand-off to the bracket authority;
     - native unresolved-exit entry gate.

5. **Bracket and exit authority**
   - Public facade: `execution/bracket_manager.py`
   - Composition owner: `execution/ownership.py::BoundBracketManager`
   - Responsibilities:
     - entry-fill re-anchoring;
     - TP1 and final-target evaluation;
     - monotonic SL and adaptive trailing;
     - partial-exit residual protection;
     - stale-order rescue;
     - broker-position-flat confirmation;
     - fill-ledger persistence and exact scaled P&L;
     - release of the runner only after durable closure.

6. **Adaptive trailing authority**
   - Public facade: `execution/adaptive_trailing.py`
   - Runtime owner: `execution/hardened_adaptive_trailing.py::HardenedAdaptiveTrailingController`
   - The controller can tighten protection only; it cannot weaken an established stop.

7. **Notifications and audit**
   - `notifications/telegram_controller.py`
   - journal, structured logs and the bracket fill ledger
   - Notifications consume persisted lifecycle transitions; they do not own state.

## Canonical BO state progression

```text
TradePlan
→ preflight validated
→ broker submission
→ acknowledgement reconciled
→ partial/complete entry fill confirmed
→ bracket armed for actual quantity and VWAP
→ TP1 fill persisted
→ residual quantity protected
→ final exit fill persisted
→ broker flat confirmed
→ exact P&L persisted
→ Telegram/audit notification
→ runner re-armed
```

At every ambiguous broker state, new entries remain blocked until orders, fills and positions are reconciled.

## Compatibility-only modules

These modules are retained to avoid breaking external imports, tests or older tooling. They are not independent production authorities:

- `execution/order_manager_legacy.py` — base implementation inherited by `RuntimeOrderManager`.
- `execution/legacy_bracket_manager.py` — base implementation inherited by the canonical bracket chain.
- `execution/adaptive_trailing_legacy.py` — base implementation inherited by the hardened trailing controller.
- `execution/lifecycle_manager.py` — experimental standalone lifecycle API; not imported by production source.
- `execution/safe_order_manager.py` — compatibility/testing wrapper; not constructed in production source.
- OrderManager dynamic-TP helpers — compatibility API only; the live StrategyRunner does not call them.

Compatibility modules may be deleted only after a release confirms that no external consumer imports them.

## Forbidden runtime layers

These removed layers must not appear in production imports:

- `order_execution_hub.py`
- `execution_router.py`
- `preflight_validator.py`

## Engineering invariants

- One runner-facing entry API.
- One entry state owner.
- One bracket/exit state owner.
- One adaptive trailing controller.
- No import-time replacement of runtime classes or methods.
- No runner re-arm before broker-flat confirmation and durable close accounting.
- Protective exits remain executable while new entries are blocked.
