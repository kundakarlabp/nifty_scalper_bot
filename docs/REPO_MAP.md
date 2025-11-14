# Repository Map

## Top-Level Layout
- `src/` – Production bot source code including execution, data pipelines, strategies, risk, and utilities.
- `tests/` – Unit and integration tests mirroring the `src/` structure.
- `deploy/`, `ops/`, `scripts/` – Operational tooling for deployments, monitoring, and local helpers.
- `docs/` – Reference documentation (this map).

## Key Modules

### Notifications
- [src/notifications/telegram_controller.py](../nifty_scalper_bot/src/notifications/telegram_controller.py#L527-L680): Production Telegram controller wiring status/diagnostic providers, admin controls, and throttled decision alerts for live operations.

### Execution
- [src/execution/order_executor.py](../nifty_scalper_bot/src/execution/order_executor.py#L1138-L1270): Kite order router managing placement, microstructure validation, retries, circuit breakers, and order state machines for live and paper modes.
- [src/execution/micro_filters.py](../nifty_scalper_bot/src/execution/micro_filters.py#L23-L200): Microstructure guardrails deriving spread/depth metrics from quotes and applying configurable entry waits and depth requirements.

### Data
- [src/data/source.py](../nifty_scalper_bot/src/data/source.py#L1232-L1320): `LiveKiteSource` fetches ticks/quotes via Kite APIs with TTL caching, subscription tracking, and resilience gates for websocket and REST data.

### Strategies
- [src/strategies/atr_gate.py](../nifty_scalper_bot/src/strategies/atr_gate.py#L1-L160): ATR-based gating utilities throttling noisy logs and enforcing configurable volatility floors before taking signals.
- [src/strategies/scoring.py](../nifty_scalper_bot/src/strategies/scoring.py#L1-L176): Feature scoring helpers computing adjusted totals, regime-aware thresholds, and diagnostics used by strategy runners.
- [src/strategies/scalper.py](../nifty_scalper_bot/src/strategies/scalper.py#L1-L200): Simplified scalper orchestrator coordinating broker margin probes, entry timing, and hedging controls around short straddles.

### Risk
- [src/risk/position_sizing.py](../nifty_scalper_bot/src/risk/position_sizing.py#L1-L200): Position sizing utilities deriving live equity, enforcing premium caps, and returning structured block reasons for trade gating.

### Costs
- [src/backtesting/sim_connector.py](../nifty_scalper_bot/src/backtesting/sim_connector.py#L21-L68): `CostModel` dataclass encapsulating brokerage, exchange, tax, and stamp duty parameters consumed by simulators (no standalone `costs/model.py` module).

### Backtesting
- [src/backtesting/backtest_engine.py](../nifty_scalper_bot/src/backtesting/backtest_engine.py#L1-L200): Event-driven engine streaming historical bars through strategies, applying simulated fills/costs, and persisting trade results.

### Utilities
- [src/utils/market_time.py](../nifty_scalper_bot/src/utils/market_time.py#L1-L110): Market session utilities providing IST-aware time helpers, trading window resolution, and session bounds.
- [src/utils/freshness.py](../nifty_scalper_bot/src/utils/freshness.py#L1-L73): Freshness computations converting timestamps and scoring tick/bar latency vs. configured thresholds.
- [src/utils/circuit_breaker.py](../nifty_scalper_bot/src/utils/circuit_breaker.py#L1-L170): Thread-safe circuit breaker implementation tracking failure counts, cooldowns, and health metadata.
- [src/utils/rate_limiter.py](../nifty_scalper_bot/src/utils/rate_limiter.py#L1-L155): Shared leaky-bucket rate limiter with environment-driven buckets for broker REST throttling.

