# Order Execution Hub Migration Guide

## Overview
The execution stack has been consolidated into the **OrderExecutionHub**, which orchestrates state, validation, lifecycle, queueing, routing, and reconciliation. Previously, order entry, lifecycle decisions, and reconciliation were scattered across individual modules with ad-hoc integrations. After the migration, OrderExecutionHub becomes the single coordination layer, improving observability and safety.

## Breaking Changes
- `order_manager.place_order()` calls should now flow through `order_execution_hub.submit_order_request()`.
- Direct access to position snapshots must use `state_tracker.get_position_state()`.

## Migration Steps
1. Update environment configuration with the new execution variables (see template below).
2. Run the database migration:
   ```bash
   python -m nifty_scalper_bot.execution.state_tracker migrate
   ```
3. Refactor strategy code to construct and submit `OrderRequest` objects instead of raw payloads.
4. Test the system in shadow mode for at least one week, monitoring `/execqueue` and `/execlast` diagnostics.
5. Enable live trading once shadow mode is stable.

### Environment template
```
EXECUTION_MODE=SHADOW
EXECUTION_RETRY_ATTEMPTS=3
EXECUTION_RETRY_DELAY_MS=500
SHADOW_DRIFT_THRESHOLD_BPS=20
SHADOW_DRIFT_AUTO_PAUSE=false
LIFECYCLE_TP1_R=1.0
LIFECYCLE_TP1_PARTIAL=0.6
LIFECYCLE_TP2_R_TREND=1.8
LIFECYCLE_TP2_R_RANGE=1.4
LIFECYCLE_TRAIL_ATR_MULT=0.8
LIFECYCLE_TIME_STOP_MIN=12
RECONCILIATION_INTERVAL_SEC=30
RECONCILIATION_ALERT_ON_MISMATCH=true
RECONCILIATION_BROKER_IS_TRUTH=true
```

## Backward Compatibility
- The legacy `OrderManager` is retained but marked deprecated; it proxies requests to the hub.
- Strategies that have not migrated will receive warnings directing them to the new API.

## Rollback Plan
If you encounter critical issues during rollout:
1. Set `EXECUTION_HUB_ENABLED=false` in `.env`.
2. Restart the bot — it reverts to the previous execution path.
3. Review `/execlast` and `/execqueue` logs to identify the failure point before attempting the migration again.

## Strategy Integration Example
### Before
```python
order_manager.place_order(
    symbol=signal.symbol,
    side=signal.side,
    quantity=signal.quantity,
    order_type="MARKET",
)
```

### After
```python
from nifty_scalper_bot.execution.order_queue import OrderRequest

order_execution_hub.submit_order_request(
    OrderRequest(
        symbol=signal.symbol,
        side=signal.side,
        quantity=signal.quantity,
        intent="ENTRY",
        source=strategy.name,
    )
)
```
