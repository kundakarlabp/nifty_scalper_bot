# Nifty Scalper Bot — Canonical Trading Path

## Authoritative live path

1. **Release-verified startup**
   - ASGI entry: `src/nifty_scalper_bot/deployment_main.py`
   - Runtime composition: `src/nifty_scalper_bot/core/app.py`
   - The embedded image SHA must match Railway's deployment SHA before the service binds its port.
   - A low-frequency watchdog exits an instance after a confirmed newer GitHub `main` revision.

2. **Market data and routing**
   - `data/market_data_manager.py`
   - `data/data_hub.py`

3. **Signal generation**
   - `strategies/runner.py`
   - `StrategyRunner` validates strategy state and creates an immutable `TradePlan`.
   - The runner submits only through `OrderManager.submit_trade_plan_result()`.
   - It must not call `place_order()` or any retired executor/processor directly.

4. **Entry execution authority**
   - Public facade: `execution/order_manager.py`
   - Core engine: `execution/order_manager_core.py`
   - Runtime owner: `execution/runtime_order_manager.py::RuntimeOrderManager`
   - Internal policy helper: `execution/options_policy.py`
   - Responsibilities:
     - preflight, spread, margin and risk validation;
     - broker submission and acknowledgement reconciliation;
     - bounded broker-rejection recovery;
     - partial-entry remainder cancellation;
     - actual fill quantity/VWAP hand-off to the bracket authority;
     - native unresolved-exit entry gate.

5. **Bracket and exit authority**
   - Public facade: `execution/bracket_manager.py`
   - Core engine: `execution/bracket_core.py`
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
   - Core controller: `execution/adaptive_trailing_core.py`
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

## Startup compatibility adapters

Two historical constructor names remain because `core/app.py` and operator components still accept them:

- `execution/safe_order_manager.py` delegates every order operation directly to the canonical `OrderManager`; it has no retry, throttle, regime, monitoring or order state machine.
- `execution/lifecycle_manager.py` is a no-op shell; it does not subscribe to ticks, calculate targets, trail stops or submit exits.

They are not BO authorities and architecture tests reject any reintroduction of independent execution logic.

## Removed duplicate paths

The following modules are deleted and forbidden from returning:

- `order_manager_legacy.py`
- `legacy_bracket_manager.py`
- `adaptive_trailing_legacy.py`
- `dynamic_tp.py`
- `order_executor.py`
- `order_processor.py`
- `entry_price.py`
- `order_execution_hub.py`
- `execution_router.py`
- `preflight_validator.py`

## Deployment freshness invariants

- The Docker image embeds `RAILWAY_GIT_COMMIT_SHA` in `/app/.build_commit_sha`.
- Strict Railway startup fails when the embedded and runtime commit identities differ or are missing.
- `/releasez` reports the effective revision and watchdog status.
- Railway activates a deployment only after `/releasez` passes.
- `overlapSeconds = 0` prevents simultaneous old and new trading instances.
- A confirmed GitHub `main` mismatch exits with code 42; transient GitHub/network failures do not stop trading.
- Railway restarts failed stale instances under the configured bounded retry policy.

## Engineering invariants

- One runner-facing entry API.
- One entry state owner.
- One bracket/exit state owner.
- One adaptive trailing controller.
- No import-time replacement of runtime classes or methods.
- No runner re-arm before broker-flat confirmation and durable close accounting.
- Protective exits remain executable while new entries are blocked.
- A stale image cannot arm live trading.
