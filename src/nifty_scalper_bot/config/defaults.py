"""Default configuration values for the Nifty scalper bot."""

from __future__ import annotations

DEFAULT_BROKER_BASE_URL = "https://api.example.com"
DEFAULT_BROKER_WEBSOCKET_URL = "wss://ws.example.com/stream"

# Single GitHub-owned current LIVE risk envelope. Keep observed exposure
# unchanged in this infrastructure-only refactor; sizing review is separate.
DEFAULT_LIVE_PER_TRADE_RISK_PCT = "5.0"
DEFAULT_LIVE_DAILY_LOSS_PCT = "5.0"

DEFAULT_RISK_MAX_DAILY_TRADES = 20
DEFAULT_RISK_MAX_ORDER_NOTIONAL = 200_000.0
DEFAULT_RISK_ALLOW_SHORT = False  # Options buying bot — never write/short options (requires margin)
DEFAULT_RISK_MAX_DRAWDOWN_PCT = 10.0
DEFAULT_RISK_MAX_DAILY_LOSS_PCT = 5.0
DEFAULT_RISK_MAX_POSITION_SIZE_PCT = 20.0
DEFAULT_RISK_MAX_TOTAL_EXPOSURE_PCT = 80.0
DEFAULT_RISK_MAX_CONCURRENT_POSITIONS = 1
DEFAULT_RISK_MAX_POSITIONS_PER_SYMBOL = 1
DEFAULT_RISK_MIN_RISK_REWARD_RATIO = 1.5

# Canonical read-only off-hot-path telemetry destinations. The Supabase Edge
# receiver is source/IP allowlisted; these URLs contain no credentials.
SUPABASE_FUNCTIONS_BASE_URL = "https://dehdptgkqbrkyzyodicd.supabase.co/functions/v1"
DEFAULT_SUPABASE_TRADE_URL = SUPABASE_FUNCTIONS_BASE_URL + "/nifty-trade-ingest"
DEFAULT_SUPABASE_ARCHIVE_URL = SUPABASE_FUNCTIONS_BASE_URL + "/nifty-log-archive-ingest"
DEFAULT_SUPABASE_HEARTBEAT_URL = (
    SUPABASE_FUNCTIONS_BASE_URL + "/nifty-runtime-heartbeat-ingest"
)

DEFAULT_LOG_LEVEL = "INFO"

DEFAULT_ORDER_RATE_LIMIT_CAPACITY = 5
DEFAULT_ORDER_RATE_LIMIT_REFILL_PER_SEC = 5.0

DEFAULT_REST_RATE_LIMIT_CAPACITY = 10
DEFAULT_REST_RATE_LIMIT_REFILL_PER_SEC = 10.0

DEFAULT_HIST_RATE_LIMIT_CAPACITY = 2
DEFAULT_HIST_RATE_LIMIT_REFILL_PER_SEC = 1.0

# ── Trading-path threshold map (owners) ─────────────────────────────────────
# QUOTE_STALE_THRESHOLD_MS (5s)      → REST quote freshness, data_hub/policy.
# MDM has_fresh_ws_ltp max_age (5s)  → WS tick freshness, data/market_data_hardening.py.
#   (The two 5s values are coincidental, not shared — change independently.)
# DEFAULT_OPTION_EXEC_MIN_BARS (30)  → option entry readiness, execution/readiness.py.
# pipeline MIN_REQUIRED_CANDLES (50) → strategy candle gate, data/pipeline.py.
# STRATEGY_CONTEXT_HARD_VETO_MAX_AGE_SECONDS (120s env)
#                                    → underlying direction freshness, core/strategy_manager.py.
QUOTE_STALE_THRESHOLD_MS = 5_000
# Single source of truth for the option execution min-bars readiness gate.
# Env: READINESS_OPTION_EXEC_MIN_BARS (legacy alias OPTION_EXECUTION_MIN_BARS).
DEFAULT_OPTION_EXEC_MIN_BARS = 30

DEFAULT_TELEGRAM_WEBHOOK_PATH = "/telegram_webhook"
DEFAULT_TELEGRAM_WEBHOOK_MAX_FAILURES = 5
DEFAULT_TELEGRAM_POLLING_INTERVAL_SECONDS = 5.0
DEFAULT_TELEGRAM_WEBHOOK_LISTEN_HOST = "0.0.0.0"
DEFAULT_TELEGRAM_WEBHOOK_LISTEN_PORT = 8000
