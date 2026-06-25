# Nifty Scalper Bot — Canonical Trading Path

## Authoritative production deployment

1. **AWS Lightsail host**
   - Production runs on one Ubuntu AWS Lightsail instance.
   - Service owner: `niftybot.service` under `systemd`.
   - Application directory: `/home/ubuntu/nifty_scalper_bot`.
   - ASGI entrypoint: `nifty_scalper_bot.main:app` through Uvicorn.
   - `deploy/lightsail_setup.sh` installs the service, HTTPS proxy, validated updater and automatic rollback.
   - Railway configuration remains in the repository only as a compatibility artefact; it is not the production authority.

2. **Validated release activation**
   - `origin/main` is checked by the Lightsail systemd timer.
   - A candidate revision is compiled and subjected to focused architecture and execution-path regression tests in an isolated git worktree.
   - The active checkout is changed only after validation succeeds.
   - The service must recover on `/livez`; otherwise the updater restores the previous commit and restarts it.
   - Credentials and operator settings remain in the host-local `.env` and are never overwritten during deployment.

3. **Market data and routing**
   - `data/market_data_manager.py`
   - `data/data_hub.py`

4. **Signal generation**
   - `strategies/runner.py`
   - `StrategyRunner` validates strategy state and creates an immutable `TradePlan`.
   - The runner submits only through `OrderManager.submit_trade_plan_result()`.
   - It must not call `place_order()` or any retired executor/processor directly.

5. **Entry execution authority**
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

6. **Bracket and exit authority**
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

7. **Adaptive trailing authority**
   - Public facade: `execution/adaptive_trailing.py`
   - Core controller: `execution/adaptive_trailing_core.py`
   - Runtime owner: `execution/hardened_adaptive_trailing.py::HardenedAdaptiveTrailingController`
   - The controller can tighten protection only; it cannot weaken an established stop.

8. **Notifications and audit**
   - `notifications/telegram_controller.py`
   - systemd journal, structured logs and the bracket fill ledger
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

Compatibility adapters may preserve historical constructor or import shapes, but they must delegate to the canonical runtime owners above and must never create a second order, bracket, trailing or lifecycle authority.
