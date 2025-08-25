# Path: src/config.py
from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, validator


# ===== Sub-models =====

class ZerodhaSettings(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""


class TelegramSettings(BaseModel):
    # Telegram is compulsory — keep the flag, but enforce True in validation
    enabled: bool = True
    bot_token: str = ""
    chat_id: str = ""


class DataSettings(BaseModel):
    lookback_minutes: int = 30
    timeframe: str = "minute"
    time_filter_start: str = "09:20"
    time_filter_end: str = "15:20"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60

    @validator("time_filter_start", "time_filter_end")
    def _v_time(cls, v: str) -> str:
        from datetime import datetime
        datetime.strptime(v, "%H:%M")
        return v


class InstrumentsSettings(BaseModel):
    spot_symbol: str = "NSE:NIFTY 50"
    trade_symbol: str = "NIFTY"
    trade_exchange: str = "NFO"
    instrument_token: int = 256265
    nifty_lot_size: int = 75
    strike_range: int = 0
    min_lots: int = 1
    max_lots: int = 10

    @validator("min_lots", "max_lots")
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
    confidence_threshold: float = 55.0   # 0..100
    min_bars_for_signal: int = 50
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

    @validator("min_equity_floor")
    def _v_equity_floor(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("min_equity_floor must be > 0")
        return v


class ExecutorSettings(BaseModel):
    exchange: str = "NFO"
    order_product: str = "NRML"
    order_variety: str = "regular"
    entry_order_type: str = "LIMIT"
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


class HealthSettings(BaseModel):
    enable_server: bool = True
    port: int = 8000


class SystemSettings(BaseModel):
    max_api_calls_per_second: float = 8.0
    websocket_reconnect_attempts: int = 5
    order_timeout_seconds: int = 30
    position_sync_interval: int = 60


# ===== Root settings =====

class AppSettings(BaseSettings):
    enable_live_trading: bool = False
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
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"


settings = AppSettings()


def validate_critical_settings() -> None:
    errors = []

    # Broker creds — only if live
    if settings.enable_live_trading:
        if not settings.zerodha.api_key:
            errors.append("ZERODHA__API_KEY is required when ENABLE_LIVE_TRADING=true")
        if not settings.zerodha.api_secret:
            errors.append("ZERODHA__API_SECRET is required when ENABLE_LIVE_TRADING=true")
        if not settings.zerodha.access_token:
            errors.append("ZERODHA__ACCESS_TOKEN is required when ENABLE_LIVE_TRADING=true")

    # Telegram is compulsory
    if not settings.telegram.enabled:
        errors.append("TELEGRAM__ENABLED must be true (Telegram is mandatory)")
    if not settings.telegram.bot_token:
        errors.append("TELEGRAM__BOT_TOKEN is required (Telegram is mandatory)")
    if not settings.telegram.chat_id:
        errors.append("TELEGRAM__CHAT_ID is required (Telegram is mandatory)")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))


# validate at import
validate_critical_settings()

# Convenience aliases (some legacy code expects these)
risk_default_equity = settings.risk.default_equity
risk_risk_per_trade = settings.risk.risk_per_trade
instruments_nifty_lot_size = settings.instruments.nifty_lot_size
time_filter_start = settings.data.time_filter_start
time_filter_end = settings.data.time_filter_end