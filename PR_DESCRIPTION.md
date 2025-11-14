Title: Hardening: resilient streamer, safe orders, risk gates, health/metrics

Summary
- Resilient websocket layer with reconnect, re-subscription, gap detection, bounded backfill, and dedupe.
- SafeOrderManager: multi-window throttles (≤5/s, ≤150/min, ≤2000/day), MIS/LIMIT defaults, price offsets, retries, metrics.
- New RiskManager: daily drawdown cap, ≤0.5% per-trade risk, loss/reject cooldowns, N-loss stop; shadow drift auto-disable.
- Enhanced Telegram notifier with whitelist + token bucket; async JSON logging; FastAPI /health and Prometheus /metrics.
- Centralized env-driven settings; live disabled by default.

Changes
- `streaming/resilient_streamer.py` (new)
- `execution/safe_order_manager.py` (new)
- `risk/risk_manager.py` (new) + `risk/__init__.py`
- `notifications/telegram_enhanced.py` (new)
- `infra/structured_logger.py`, `infra/health.py` (new) + `infra/__init__.py`
- `core/app.py` wiring/lifecycle; `config/settings.py` (new)
- Tests: streamer, safe orders, risk, shadow, health
- Docs: `docs/master_prompt_template.md`

Testing
- `pytest -q tests/test_resilient_streamer.py tests/test_safe_order_manager.py tests/test_risk_manager_new.py tests/test_shadow_paper.py tests/test_health_endpoint.py`
- `uvicorn` smoke test of `/health` and `/metrics` locally.

Checklist
- [x] No new runtime deps beyond allowed list
- [x] `/health` & `/metrics` operational
- [x] Risk gates block live unless enabled and approved
- [x] Shadow mode active; drift alerts and optional live auto-disable
- [x] No secrets in source; all env-driven
- [x] Live trading OFF by default

Rollout
- Env: `ENABLE_LIVE`, `DAILY_PNL_CAP_PCT`, `RISK_PER_TRADE_PCT`, `TELEGRAM_TOKEN`, `ALLOWED_CHAT_IDS`, `ORDER_RATE_PER_*`, `SHADOW_DRIFT_THRESHOLD_PCT`
- Deploy paper-only first; enable live after monitoring latency/rejections/drift via `/metrics`.
