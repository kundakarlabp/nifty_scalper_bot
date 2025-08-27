# Path: src/config.py
from __future__ import annotations

"""
Config with nested models (original structure) + flat aliases for compatibility.

- Preserves your original names/shape so other modules don't break.
- Adds a few extra validations and guardrails.
- Keeps Telegram mandatory; Zerodha creds are only required if live trading is enabled.
- Adds optional historical backfill knobs under DataSettings.
"""

from pydantic import BaseModel, validator
from pydantic_settings import BaseSettings


# ================= Sub-models =================

class ZerodhaSettings(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""


class TelegramSettings(BaseModel):
    # Telegram is COMPULSORY in your deployment
    enabled: bool = True
    bot_token: str = ""
    chat_id: int = 0  # store as int to match controller usage

    @validator("chat_id")
    def _v_chat_id(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("TELEGRAM__CHAT_ID must be a positive integer")
        return v


class DataSettings(BaseModel):
    # Live loop consumption
    lookback_minutes: int = 30
    timeframe: str = "minute"  # 'minute' recommended
    time_filter_start: str = "09:20"
    time_filter_end: str = "15:20"

    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60

    # Historical backfill (optional; runner/feeds can ignore if unsupported)
    history_days: int = 0          # 0 = off; otherwise backfill N days before now
    history_max_candles: int = 0   # 0 = unlimited within broker constraints

    @validator("time_filter_start", "time_filter_end")
    def _v_time(cls, v: str) -> str:
        from datetime import datetime
        datetime.strptime(v, "%H:%M")
        return v

    @validator("timeframe")
    def _v_tf(cls, v: str) -> str:
        v = (v or "").lower()
        allowed = {"minute", "3minute", "5minute", "10minute", "15minute", "day"}
        # Keep loose—brokers vary; still nudge common values
        if v not in allowed:
            # Accept anything but keep a sensible default rather than failing hard
            return v
        return v

    @validator("lookback_minutes", "cache_ttl_seconds", "history_days", "history_max_candles")
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

    @validator("min_lots", "max_lots", "nifty_lot_size")
    def _v_lots_pos(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Lot sizes must be positive")
        return v

    @validator("max_lots")
    def _v_lots_order(cls, v: int, values) -> int:
        if "min_lots" in values and v < values["min_lots"]:
            raise ValueError("max_lots must be >= min_lots")
        return v


class StrategySettings(BaseModel):
    min_signal_score: int = 3
    confidence_threshold: float = 55.0  # 0..100
    min_bars_for_signal: int = 30
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    atr_sl_multiplier: float = 1.3
    atr_tp_multiplier: float = 2.2
    rr_min: float = 1.30

    @validator("confidence_threshold")
    def _v_conf(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError("confidence_threshold must be 0..100")
        return v

    @validator("ema_slow")
    def _v_emas(cls, v: int, values) -> int:
        if "ema_fast" in values and values["ema_fast"] >= v:
            raise ValueError("ema_fast must be < ema_slow")
        return v

    @validator("atr_tp_multiplier")
    def _v_tp_gt_sl(cls, v: float, values) -> float:
        if "atr_sl_multiplier" in values and v <= values["atr_sl_multiplier"]:
            raise ValueError("ATR_TP_MULTIPLIER must be > ATR_SL_MULTIPLIER")
        return v

    @validator("rr_min")
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

    @validator("risk_per_trade")
    def _v_risk_pct(cls, v: float) -> float:
        if not 0.001 <= v <= 0.10:
            raise ValueError("risk_per_trade must be between 0.1% and 10%")
        return v

    @validator("max_daily_drawdown_pct")
    def _v_dd_pct(cls, v: float) -> float:
        if not 0.01 <= v <= 0.20:
            raise ValueError("max_daily_drawdown_pct must be between 1% and 20%")
        return v

    @validator("min_equity_floor", "default_equity")
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

    @validator("tp1_qty_ratio")
    def _v_ratio(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("tp1_qty_ratio must be in 0..1")
        return v

    @validator("breakeven_ticks", "slippage_ticks")
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
    # You asked to default to LIVE. Be aware this enforces Zerodha creds at import time.
    enable_live_trading: bool = True
    allow_offhours_testing: bool = False
    log_level: str = "INFO"

    zerodha: ZerodhaSettings = ZerodhaSettings()
    telegram: TelegramSettings = TelegramSettings()
    data: DataSettings = DataSettings()
    instruments: InstrumentsSettings = InstrumentsSettings()
    strategy: StrategySettings = StrategySettings()
    risk: RiskSettings = RiskSettings()
    executor: ExecutorSettings = ExecutorSettings()
    health: HealthSettings = HealthSettings()
    system: SystemSettings = SystemSettings()

    class Config:
        env_file = ".env"              # used locally; Railway uses real env vars
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"   # e.g., TELEGRAM__BOT_TOKEN

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

    # Telegram is MANDATORY in your deployment
    if not settings.telegram.bot_token:
        errors.append("TELEGRAM__BOT_TOKEN is required (Telegram is mandatory)")
    if not settings.telegram.chat_id:
        errors.append("TELEGRAM__CHAT_ID is required (Telegram is mandatory)")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))


# Validate on import
validate_critical_settings()

# -------- Back-compat convenience aliases (read-only) --------
# (These mirror the flat properties as plain module-level names if other modules import them.)
risk_default_equity = settings.risk_default_equity
risk_risk_per_trade = settings.risk_risk_per_trade
instruments_nifty_lot_size = settings.instruments_nifty_lot_size
time_filter_start = settings.data_time_filter_start
time_filter_end = settings.data_time_filter_end