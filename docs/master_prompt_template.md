# Hardened Runtime Modules

- `streaming/resilient_streamer.py` wraps the existing websocket manager,
  providing automatic reconnects, gap detection, deduplication and bounded
  REST backfill.  The class exposes `simulate_disconnect()`,
  `is_connected()` and `backlog_size()` helpers used by tests and the
  `/health` endpoint.
- `execution/safe_order_manager.py` sits in front of the legacy
  `OrderManager`, enforcing multi-window throttles (≤5/s, ≤150/min,
  ≤2000/day by default), MIS/LIMIT defaults, price offsets and rejection
  metrics.  Order rejections feed back into the new risk manager.
- `risk/risk_manager.py` provides `check_order(signal)` gating that
  enforces daily P&L caps, 0.5% per-trade risk, cooldowns and consecutive
  loss stops while exposing compatibility helpers for the legacy runner.
- `notifications/telegram_enhanced.py` delivers asynchronous Telegram
  notifications with whitelisted chat IDs and a token bucket rate limiter
  (~25 msgs/s by default).
- `shadow/shadow_paper.py` mirrors live trades in a paper ledger,
  calculates drift between live and paper equity, emits alerts, and can
  auto-disable live routing when drift exceeds configurable thresholds.
- `infra/structured_logger.py` configures asynchronous JSON logging using
  the schema defined in the production playbook (timestamp, level, module,
  event metadata).
- `infra/health.py` exposes a FastAPI app with `/health` and `/metrics`.
  `/health` reports websocket connectivity, backlog depth, throttle and
  rejection counters, and whether live trading is enabled. `/metrics`
  exports Prometheus counters and gauges for latency, throttle usage and
  fallback operations.

# Usage Notes

`core/app.py` wires the new modules together.  The resilient streamer feeds
the legacy `MarketDataManager`, the safe order manager is passed into the
strategy runner, and the shadow trader mirrors every submitted order.  The
`NiftyScalperApp` exposes `simulate_disconnect()`, `is_connected()`,
`backlog_size()` and `rejection_count()` helper methods for operational
tooling.  `NiftyScalperApp.health_app` returns the FastAPI instance that can
be mounted when deploying to Railway or running locally.

Runtime configuration is centralised in `config/settings.py` and is sourced
from environment variables.  `ENABLE_LIVE` remains `false` by default, so
live order routing must be explicitly enabled once paper trading behaviour
has been observed.
