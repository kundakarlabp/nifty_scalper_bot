"""Config with nested models (original structure) + flat aliases for compatibility.

- Preserves your original names/shape so other modules don't break.
- Adds a few extra validations and guardrails.
- Keeps Telegram mandatory; Zerodha creds are only required if live trading is enabled.
- Adds optional historical backfill knobs under DataSettings.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path

import pandas as pd
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationInfo, field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.market_time import is_market_open, prev_session_last_20m

# Optional external dependencies for validation
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
try:
    from src.data.source import LiveKiteSource  # type: ignore
except Exception:  # pragma: no cover
    LiveKiteSource = None  # type: ignore


def _seed_env_from_defaults() -> None:
    """Populate ``os.environ`` with values from ``config/defaults.yaml``.

    Existing environment variables take precedence and are not overridden.
    Nested keys in the YAML are flattened using ``__`` as the delimiter to
    mirror Pydantic's ``env_nested_delimiter`` behaviour.
    """

    path = Path("config/defaults.yaml")
    if not path.is_file():
        return

    try:
        data = yaml.safe_load(path.read_text("utf-8")) or {}
    except Exception:  # pragma: no cover - defensive
        logging.getLogger("config").exception("Failed loading %s", path)
        return

    def _flatten(prefix: str, obj: dict[str, object]) -> None:
        for k, v in obj.items():
            key = f"{prefix}{k}".upper()
            if isinstance(v, dict):
                _flatten(f"{key}__", v)
            else:
                os.environ.setdefault(key, str(v))

    _flatten("", data)


_seed_env_from_defaults()


# ================= Sub-models =================

class ZerodhaSettings(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""

    @classmethod
    def from_env(cls) -> "ZerodhaSettings":
        """Load legacy flat env vars if present."""
        return cls(
            api_key=os.getenv("KITE_API_KEY", ""),
            api_secret=os.getenv("KITE_API_SECRET", ""),
            access_token=os.getenv("KITE_ACCESS_TOKEN", ""),
        )


class TelegramSettings(BaseModel):
    # Telegram is COMPULSORY in your deployment
    enabled: bool = True
    bot_token: str = ""
    chat_id: int = 0  # store as int to match controller usage

    @classmethod
    def from_env(cls) -> "TelegramSettings":
        """Support legacy flat env vars such as TELEGRAM_CHAT_ID."""
        raw_chat = os.getenv("TELEGRAM_CHAT_ID")
        try:
            chat = int(raw_chat) if raw_chat is not None else 0
        except ValueError:
            chat = 0
        enabled_env = os.getenv("TELEGRAM_ENABLED")
        enabled = (
            str(enabled_env).lower() not in {"0", "false"}
            if enabled_env is not None
            else True
        )
        return cls.model_construct(
            enabled=enabled,
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=chat,
        )

    @field_validator("chat_id")
    @classmethod
    def _v_chat_id(cls, v: int, info: ValidationInfo) -> int:
        """Ensure chat_id is provided when Telegram is enabled."""
        if info.data.get("enabled", True) and v == 0:
            raise ValueError(
                "TELEGRAM__CHAT_ID must be a non-zero integer when TELEGRAM__ENABLED is true"
            )
        return v


class DataSettings(BaseModel):
    # Live loop consumption
    lookback_minutes: int = 20
    timeframe: str = "minute"  # 'minute' recommended
    time_filter_start: str = "09:20"
    time_filter_end: str = "15:25"

    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60

    # Historical backfill (optional; runner/feeds can ignore if unsupported)
    history_days: int = 3          # 0 = off; otherwise backfill N days before now
    history_max_candles: int = 0   # 0 = unlimited within broker constraints

    @field_validator("time_filter_start", "time_filter_end")
    @classmethod
    def _v_time(cls, v: str) -> str:
        from datetime import datetime
        datetime.strptime(v, "%H:%M")
        return v

    @field_validator("timeframe")
    @classmethod
    def _v_tf(cls, v: str) -> str:
        v = (v or "").lower()
        allowed = {"minute", "3minute", "5minute", "10minute", "15minute", "day"}
        # Keep loose—brokers vary; still nudge common values
        if v not in allowed:
            logging.getLogger("config").warning(
                "Unsupported timeframe '%s'; defaulting to 'minute'", v
            )
            return "minute"
        return v

    @field_validator("lookback_minutes", "cache_ttl_seconds", "history_days", "history_max_candles")
    @classmethod
    def _v_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("numeric fields must be >= 0")
        return v


class InstrumentsSettings(BaseModel):
    spot_symbol: str = "NSE:NIFTY 50"
    trade_symbol: str = "NIFTY"
    trade_exchange: str = "NFO"
    instrument_token: int = 256265        # primary token (spot preferred for OHLC)
    spot_token: int = 256265               # optional explicit spot token (helps with logs/diagnostics)
    nifty_lot_size: int = 75
    strike_range: int = 0
    min_lots: int = 1
    max_lots: int = 10

    @field_validator("instrument_token", mode="before")
    @classmethod
    def _v_token(cls, v: object) -> int:
        """Coerce instrument token strings or floats to int."""
        try:
            return int(float(str(v)))
        except (TypeError, ValueError) as e:
            raise ValueError("instrument_token must be numeric") from e

    @field_validator("min_lots", "max_lots", "nifty_lot_size")
    @classmethod
    def _v_lots_pos(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Lot sizes must be positive")
        return v

    @field_validator("max_lots")
    @classmethod
    def _v_lots_order(cls, v: int, info: ValidationInfo) -> int:
        if "min_lots" in info.data and v < info.data["min_lots"]:
            raise ValueError("max_lots must be >= min_lots")
        return v


class StrategySettings(BaseModel):
    min_signal_score: int = 4
    confidence_threshold: float = 55.0  # 0..100
    min_bars_for_signal: int = 20
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    atr_sl_multiplier: float = 1.3
    atr_tp_multiplier: float = 2.2
    rr_min: float = 1.30
    rr_threshold: float | None = 1.5

    @field_validator("confidence_threshold")
    @classmethod
    def _v_conf(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError("confidence_threshold must be 0..100")
        return v

    @field_validator("ema_slow")
    @classmethod
    def _v_emas(cls, v: int, info: ValidationInfo) -> int:
        if "ema_fast" in info.data and info.data["ema_fast"] >= v:
            raise ValueError("ema_fast must be < ema_slow")
        return v

    @field_validator("atr_tp_multiplier")
    @classmethod
    def _v_tp_gt_sl(cls, v: float, info: ValidationInfo) -> float:
        if "atr_sl_multiplier" in info.data and v <= info.data["atr_sl_multiplier"]:
            raise ValueError("ATR_TP_MULTIPLIER must be > ATR_SL_MULTIPLIER")
        return v

    @field_validator("rr_min")
    @classmethod
    def _v_rr_min(cls, v: float) -> float:
        if v <= 1.0:
            raise ValueError("rr_min must be > 1.0")
        return v


class RiskSettings(BaseModel):
    use_live_equity: bool = True
    default_equity: float = 30000.0
    min_equity_floor: float = 25000.0
    equity_refresh_seconds: int = 60

    risk_per_trade: float = 0.01
    max_trades_per_day: int = 12
    consecutive_loss_limit: int = 3
    max_daily_drawdown_pct: float = 0.04
    max_position_size_pct: float = 0.10

    @field_validator("risk_per_trade")
    @classmethod
    def _v_risk_pct(cls, v: float) -> float:
        if not 0.001 <= v <= 0.10:
            raise ValueError("risk_per_trade must be between 0.1% and 10%")
        return v

    @field_validator("max_daily_drawdown_pct")
    @classmethod
    def _v_dd_pct(cls, v: float) -> float:
        if not 0.01 <= v <= 0.20:
            raise ValueError("max_daily_drawdown_pct must be between 1% and 20%")
        return v

    @field_validator("min_equity_floor", "default_equity")
    @classmethod
    def _v_equity_pos(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("equity amounts must be > 0")
        return v


class ExecutorSettings(BaseModel):
    exchange: str = "NFO"
    order_product: str = "NRML"
    order_variety: str = "regular"   # regular | bo | amo | co (depending on broker support)
    entry_order_type: str = "LIMIT"  # LIMIT | MARKET | SL | SLM
    tick_size: float = 0.05
    exchange_freeze_qty: int = 1800
    preferred_exit_mode: str = "REGULAR"
    use_slm_exit: bool = True
    partial_tp_enable: bool = True
    tp1_qty_ratio: float = 0.5
    breakeven_ticks: int = 2
    enable_trailing: bool = True
    trailing_atr_multiplier: float = 1.4
    fee_per_lot: float = 20.0
    slippage_ticks: int = 1
    ack_timeout_ms: int = 1500
    fill_timeout_ms: int = 10000
    max_place_retries: int = 2
    # microstructure execution guards
    max_spread_pct: float = 0.0035  # 0.35%
    depth_multiplier: float = 5.0   # top-5 depth >= mult * order size
    micro_retry_limit: int = 3
    require_depth: bool = False
    default_spread_pct_est: float = 0.25

    @field_validator("tp1_qty_ratio")
    @classmethod
    def _v_ratio(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("tp1_qty_ratio must be in 0..1")
        return v

    @field_validator(
        "breakeven_ticks",
        "slippage_ticks",
        "ack_timeout_ms",
        "fill_timeout_ms",
        "max_place_retries",
    )
    @classmethod
    def _v_nonneg_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("tick counts must be >= 0")
        return v


class HealthSettings(BaseModel):
    enable_server: bool = True
    port: int = 8000


class SystemSettings(BaseModel):
    max_api_calls_per_second: float = 8.0
    websocket_reconnect_attempts: int = 5
    order_timeout_seconds: int = 30
    position_sync_interval: int = 60
    log_buffer_capacity: int = 4000


# ================= Root settings =================

class AppSettings(BaseSettings):
    # General optional settings
    app_env: str = "production"
    ai_provider: str = ""
    openai_api_key: str = ""

    # Live trading is enabled by default; set to False to run in paper mode.
    enable_live_trading: bool = True
    allow_offhours_testing: bool = False
    enable_time_windows: bool = True
    tz: str = "Asia/Kolkata"
    log_level: str = "INFO"

    cb_error_rate: float = 0.10
    cb_p95_ms: int = 1200
    cb_min_samples: int = 30
    cb_open_cooldown_sec: int = 30
    cb_half_open_probe: int = 3
    max_place_retries: int = 2
    max_modify_retries: int = 2
    retry_backoff_ms: int = 200

    # Data warmup
    warmup_bars: int = 30

    MAX_DAILY_DD_R: float = 2.5
    MAX_TRADES_PER_SESSION: int = 40
    MAX_LOTS_PER_SYMBOL: int = 5
    MAX_NOTIONAL_RUPEES: float = 1_500_000.0
    MAX_GAMMA_MODE_LOTS: int = 2
    MAX_PORTFOLIO_DELTA_UNITS: int = 100
    MAX_PORTFOLIO_DELTA_UNITS_GAMMA: int = 60
    RISK_FREE_RATE: float = 0.065
    ROLL10_PAUSE_R: float = -0.2
    ROLL10_PAUSE_MIN: int = 60
    COOLOFF_LOSS_STREAK: int = 3
    COOLOFF_MINUTES: int = 45
    JOURNAL_DB_PATH: str = "data/journal.sqlite"

    zerodha: ZerodhaSettings = Field(default_factory=ZerodhaSettings.from_env)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings.from_env)
    data: DataSettings = DataSettings()
    instruments: InstrumentsSettings = InstrumentsSettings()
    strategy: StrategySettings = StrategySettings()
    risk: RiskSettings = RiskSettings()
    executor: ExecutorSettings = ExecutorSettings()
    health: HealthSettings = HealthSettings()
    system: SystemSettings = SystemSettings()

    @field_validator("warmup_bars")
    @classmethod
    def _v_warmup(cls, v: int) -> int:
        if v < 0:
            raise ValueError("warmup_bars must be >= 0")
        return v

    @field_validator("data", mode="before")
    @classmethod
    def _v_data_aliases(cls, v: object) -> object:
        ht = os.getenv("HISTORICAL_TIMEFRAME")
        if ht:
            if isinstance(v, dict):
                v.setdefault("timeframe", ht)
            elif isinstance(v, DataSettings):
                v.timeframe = ht
        return v

    model_config = SettingsConfigDict(
        env_file=".env",             # used locally; Railway uses real env vars
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",  # e.g., TELEGRAM__BOT_TOKEN
        extra="ignore",
    )

    # -------- Flat alias properties (read-only) --------
    # Strategy (flat)
    @property
    def strategy_min_signal_score(self) -> int: return self.strategy.min_signal_score
    @property
    def strategy_confidence_threshold(self) -> float: return self.strategy.confidence_threshold
    @property
    def strategy_min_bars_for_signal(self) -> int: return self.strategy.min_bars_for_signal
    @property
    def strategy_ema_fast(self) -> int: return self.strategy.ema_fast
    @property
    def strategy_ema_slow(self) -> int: return self.strategy.ema_slow
    @property
    def strategy_rsi_period(self) -> int: return self.strategy.rsi_period
    @property
    def strategy_bb_period(self) -> int: return self.strategy.bb_period
    @property
    def strategy_bb_std(self) -> float: return self.strategy.bb_std
    @property
    def strategy_atr_period(self) -> int: return self.strategy.atr_period
    @property
    def strategy_atr_sl_multiplier(self) -> float: return self.strategy.atr_sl_multiplier
    @property
    def strategy_atr_tp_multiplier(self) -> float: return self.strategy.atr_tp_multiplier
    @property
    def strategy_rr_min(self) -> float: return self.strategy.rr_min
    @property
    def strategy_rr_threshold(self) -> float | None: return self.strategy.rr_threshold

    # Risk (flat)
    @property
    def risk_use_live_equity(self) -> bool: return self.risk.use_live_equity
    @property
    def risk_default_equity(self) -> float: return self.risk.default_equity
    @property
    def risk_min_equity_floor(self) -> float: return self.risk.min_equity_floor
    @property
    def risk_equity_refresh_seconds(self) -> int: return self.risk.equity_refresh_seconds
    @property
    def risk_risk_per_trade(self) -> float: return self.risk.risk_per_trade
    @property
    def risk_max_trades_per_day(self) -> int: return self.risk.max_trades_per_day
    @property
    def risk_consecutive_loss_limit(self) -> int: return self.risk.consecutive_loss_limit
    @property
    def risk_max_daily_drawdown_pct(self) -> float: return self.risk.max_daily_drawdown_pct
    @property
    def risk_max_position_size_pct(self) -> float: return self.risk.max_position_size_pct

    # Instruments (flat)
    @property
    def instruments_spot_symbol(self) -> str: return self.instruments.spot_symbol
    @property
    def instruments_trade_symbol(self) -> str: return self.instruments.trade_symbol
    @property
    def instruments_trade_exchange(self) -> str: return self.instruments.trade_exchange
    @property
    def instruments_instrument_token(self) -> int: return self.instruments.instrument_token
    @property
    def instruments_nifty_lot_size(self) -> int: return self.instruments.nifty_lot_size
    @property
    def instruments_strike_range(self) -> int: return self.instruments.strike_range
    @property
    def instruments_min_lots(self) -> int: return self.instruments.min_lots
    @property
    def instruments_max_lots(self) -> int: return self.instruments.max_lots

    # Data (flat)
    @property
    def data_lookback_minutes(self) -> int: return self.data.lookback_minutes
    @property
    def data_timeframe(self) -> str: return self.data.timeframe
    @property
    def data_time_filter_start(self) -> str: return self.data.time_filter_start
    @property
    def data_time_filter_end(self) -> str: return self.data.time_filter_end
    @property
    def data_cache_enabled(self) -> bool: return self.data.cache_enabled
    @property
    def data_cache_ttl_seconds(self) -> int: return self.data.cache_ttl_seconds

    # Executor (flat)
    @property
    def executor_exchange(self) -> str: return self.executor.exchange
    @property
    def executor_order_product(self) -> str: return self.executor.order_product
    @property
    def executor_order_variety(self) -> str: return self.executor.order_variety
    @property
    def executor_entry_order_type(self) -> str: return self.executor.entry_order_type
    @property
    def executor_tick_size(self) -> float: return self.executor.tick_size
    @property
    def executor_exchange_freeze_qty(self) -> int: return self.executor.exchange_freeze_qty
    @property
    def executor_preferred_exit_mode(self) -> str: return self.executor.preferred_exit_mode
    @property
    def executor_use_slm_exit(self) -> bool: return self.executor.use_slm_exit
    @property
    def executor_partial_tp_enable(self) -> bool: return self.executor.partial_tp_enable
    @property
    def executor_tp1_qty_ratio(self) -> float: return self.executor.tp1_qty_ratio
    @property
    def executor_breakeven_ticks(self) -> int: return self.executor.breakeven_ticks
    @property
    def executor_enable_trailing(self) -> bool: return self.executor.enable_trailing
    @property
    def executor_trailing_atr_multiplier(self) -> float: return self.executor.trailing_atr_multiplier
    @property
    def executor_fee_per_lot(self) -> float: return self.executor.fee_per_lot
    @property
    def executor_slippage_ticks(self) -> int: return self.executor.slippage_ticks
    @property
    def executor_ack_timeout_ms(self) -> int: return self.executor.ack_timeout_ms
    @property
    def executor_fill_timeout_ms(self) -> int: return self.executor.fill_timeout_ms
    @property
    def executor_max_place_retries(self) -> int: return self.executor.max_place_retries

    # Health/System (flat)
    @property
    def health_enable_server(self) -> bool: return self.health.enable_server
    @property
    def health_port(self) -> int: return self.health.port
    @property
    def system_max_api_calls_per_second(self) -> float: return self.system.max_api_calls_per_second
    @property
    def system_websocket_reconnect_attempts(self) -> int: return self.system.websocket_reconnect_attempts
    @property
    def system_order_timeout_seconds(self) -> int: return self.system.order_timeout_seconds
    @property
    def system_position_sync_interval(self) -> int: return self.system.position_sync_interval


# Instantiate settings
settings = AppSettings()

import os as _os  # noqa: E402
object.__setattr__(settings, "ENABLE_SIGNAL_DEBUG", str(_os.getenv("ENABLE_SIGNAL_DEBUG", "false")).lower() in ("1","true","yes"))
object.__setattr__(settings, "TELEGRAM__PRETRADE_ALERTS", str(_os.getenv("TELEGRAM__PRETRADE_ALERTS", "false")).lower() in ("1","true","yes"))
object.__setattr__(settings, "DIAG_INTERVAL_SECONDS", int(_os.getenv("DIAG_INTERVAL_SECONDS", "60")))
object.__setattr__(settings, "MIN_PREVIEW_SCORE", float(_os.getenv("MIN_PREVIEW_SCORE", "8")))
object.__setattr__(settings, "ACK_TIMEOUT_MS", int(_os.getenv("ACK_TIMEOUT_MS", "1500")))
object.__setattr__(settings, "FILL_TIMEOUT_MS", int(_os.getenv("FILL_TIMEOUT_MS", "10000")))
object.__setattr__(settings, "RETRY_BACKOFF_MS", int(_os.getenv("RETRY_BACKOFF_MS", "200")))
object.__setattr__(settings, "MAX_PLACE_RETRIES", int(_os.getenv("MAX_PLACE_RETRIES", "2")))
object.__setattr__(settings, "MAX_MODIFY_RETRIES", int(_os.getenv("MAX_MODIFY_RETRIES", "2")))
object.__setattr__(settings, "PLAN_STALE_SEC", int(_os.getenv("PLAN_STALE_SEC", "20")))



def validate_critical_settings() -> None:
    errors = []

    # Live trading requires broker creds
    if settings.enable_live_trading:
        if not settings.zerodha.api_key:
            errors.append("ZERODHA__API_KEY is required when ENABLE_LIVE_TRADING=true")
        if not settings.zerodha.api_secret:
            errors.append("ZERODHA__API_SECRET is required when ENABLE_LIVE_TRADING=true")
        if not settings.zerodha.access_token:
            errors.append("ZERODHA__ACCESS_TOKEN is required when ENABLE_LIVE_TRADING=true")

    # Telegram configuration (required only when enabled)
    if settings.telegram.enabled:
        if not settings.telegram.bot_token:
            errors.append(
                "TELEGRAM__BOT_TOKEN is required when TELEGRAM__ENABLED=true"
            )
        if not settings.telegram.chat_id:
            errors.append(
                "TELEGRAM__CHAT_ID is required when TELEGRAM__ENABLED=true"
            )

    # Ensure lookback window can satisfy minimum bars requirement
    if settings.data.lookback_minutes < settings.strategy.min_bars_for_signal:
        errors.append(
            "DATA__LOOKBACK_MINUTES must be >= STRATEGY__MIN_BARS_FOR_SIGNAL"
        )

    # Instrument token sanity check (only in live mode with deps available)
    if (
        settings.enable_live_trading
        and not settings.allow_offhours_testing
        and KiteConnect is not None
        and LiveKiteSource is not None
        and not errors  # only attempt if creds are present
    ):
        token = int(getattr(settings.instruments, "instrument_token", 0) or 0)
        if token <= 0:
            errors.append("INSTRUMENTS__INSTRUMENT_TOKEN must be a positive integer")
        else:
            src = None
            try:
                kite = KiteConnect(api_key=str(settings.zerodha.api_key))
                kite.set_access_token(str(settings.zerodha.access_token))
                src = LiveKiteSource(kite=kite)
                try:
                    src.connect()
                    now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30))).replace(
                        second=0, microsecond=0
                    )
                    if is_market_open(now_ist):
                        start = now_ist - timedelta(minutes=20)
                        end = now_ist
                    else:
                        start, end = prev_session_last_20m(now_ist)
                    df = src.fetch_ohlc(
                        token=token, start=start, end=end, timeframe="minute"
                    )
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        # Outside market hours the historical API may return no data.
                        # Fall back to a simple last-price lookup so that a valid token
                        # doesn't trigger a false validation error.
                        ltp_fn = getattr(src, "get_last_price", None)
                        ltp = ltp_fn(token) if callable(ltp_fn) else None
                        if not isinstance(ltp, (int, float)):
                            errors.append(
                                f"instrument_token {token} returned no data; configure a valid F&O token"
                            )
                finally:
                    disconnect = getattr(src, "disconnect", None)
                    if callable(disconnect):
                        try:
                            disconnect()
                        except Exception:
                            logging.getLogger("config").debug(
                                "LiveKiteSource disconnect failed", exc_info=True
                            )
            except Exception as e:
                logging.getLogger("config").warning(
                    "instrument_token validation skipped: %s", e
                )
            finally:
                if src is not None:
                    disconnect_fn = getattr(src, "disconnect", None)
                    if callable(disconnect_fn):
                        try:
                            disconnect_fn()
                        except Exception:
                            logging.getLogger("config").debug(
                                "instrument_token validation disconnect failed", exc_info=True
                            )

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))


# -------- Back-compat convenience aliases (read-only) --------
# (These mirror the flat properties as plain module-level names if other modules import them.)
risk_default_equity = settings.risk_default_equity
risk_risk_per_trade = settings.risk_risk_per_trade
instruments_nifty_lot_size = settings.instruments_nifty_lot_size
time_filter_start = settings.data_time_filter_start
time_filter_end = settings.data_time_filter_end
