"""
Centralized, validated configuration using Pydantic v2 / pydantic-settings v2.

Key points:
- Export a SINGLE source of truth: `settings` (AppSettings instance).
- No `Config` singleton anywhere in the codebase.
- Sensible defaults that match your NIFTY scalper workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, PositiveFloat, PositiveInt, NonNegativeFloat, NonNegativeInt
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Optional[Path]:
    path = Path.cwd()
    for _ in range(6):
        candidate = path / ".env"
        if candidate.exists():
            return candidate
        path = path.parent
    return None


class DataSettings(BaseSettings):
    warmup_bars: PositiveInt = Field(30, alias="WARMUP_BARS")
    lookback_minutes: PositiveInt = Field(60, alias="DATA_LOOKBACK_MINUTES")
    timeframe: Literal["minute", "5minute"] = Field("minute", alias="HISTORICAL_TIMEFRAME")
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")


class InstrumentSettings(BaseSettings):
    spot_symbol: str = Field("NSE:NIFTY 50", alias="SPOT_SYMBOL")
    trade_symbol: str = Field("NIFTY", alias="TRADE_SYMBOL")
    trade_exchange: str = Field("NFO", alias="TRADE_EXCHANGE")
    spot_token: PositiveInt = Field(256265, alias="INSTRUMENT_TOKEN")
    nifty_lot_size: PositiveInt = Field(50, alias="NIFTY_LOT_SIZE")
    min_lots: PositiveInt = Field(1, alias="MIN_LOTS")
    max_lots: PositiveInt = Field(10, alias="MAX_LOTS")
    # ✅ allow 0 (ATM-only); previously PositiveInt caused crash with default 0
    strike_range: NonNegativeInt = Field(0, alias="STRIKE_RANGE")
    strike_selection_range: PositiveInt = Field(3, alias="STRIKE_SELECTION_RANGE")
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")


class StrategySettings(BaseSettings):
    min_signal_score: int = Field(5, alias="MIN_SIGNAL_SCORE")
    confidence_threshold: PositiveFloat = Field(6.0, alias="CONFIDENCE_THRESHOLD")
    base_stop_loss_points: PositiveFloat = Field(20.0, alias="BASE_STOP_LOSS_POINTS")
    base_target_points: PositiveFloat = Field(40.0, alias="BASE_TARGET_POINTS")
    atr_period: PositiveInt = Field(14, alias="ATR_PERIOD")
    atr_sl_multiplier: PositiveFloat = Field(1.5, alias="ATR_SL_MULTIPLIER")
    atr_tp_multiplier: PositiveFloat = Field(3.0, alias="ATR_TP_MULTIPLIER")
    sl_confidence_adj: NonNegativeFloat = Field(0.2, alias="SL_CONFIDENCE_ADJ")
    tp_confidence_adj: NonNegativeFloat = Field(0.3, alias="TP_CONFIDENCE_ADJ")
    time_filter_start: str = Field("09:15", alias="TIME_FILTER_START")
    time_filter_end: str = Field("15:30", alias="TIME_FILTER_END")
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")


class RiskSettings(BaseSettings):
    risk_per_trade: PositiveFloat = Field(0.01, alias="RISK_PER_TRADE")
    daily_loss_limit_r: PositiveFloat = Field(3.0, alias="DAILY_LOSS_LIMIT_R")
    max_trades_per_day: PositiveInt = Field(10, alias="MAX_TRADES_PER_DAY")
    default_equity: PositiveFloat = Field(30000.0, alias="DEFAULT_EQUITY")
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")


class ZerodhaSettings(BaseSettings):
    api_key: Optional[str] = Field(None, alias="ZERODHA_API_KEY")
    api_secret: Optional[str] = Field(None, alias="ZERODHA_API_SECRET")
    access_token: Optional[str] = Field(None, alias="KITE_ACCESS_TOKEN")
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")


class TelegramSettings(BaseSettings):
    enabled: bool = Field(True, alias="ENABLE_TELEGRAM")
    bot_token: Optional[str] = Field(None, alias="TELEGRAM_BOT_TOKEN")
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")


class ServerSettings(BaseSettings):
    port: int = Field(8000, alias="HEALTH_PORT")
    host: str = Field("0.0.0.0", alias="HEALTH_HOST")
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")


class AppToggles(BaseSettings):
    enable_live_trading: bool = Field(False, alias="ENABLE_LIVE_TRADING")
    allow_offhours_testing: bool = Field(False, alias="ALLOW_OFFHOURS_TESTING")
    preferred_exit_mode: Literal["AUTO", "GTT", "REGULAR"] = Field("AUTO", alias="PREFERRED_EXIT_MODE")
    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")


class AppSettings(BaseSettings):
    data: DataSettings = DataSettings()
    instruments: InstrumentSettings = InstrumentSettings()
    strategy: StrategySettings = StrategySettings()
    risk: RiskSettings = RiskSettings()
    zerodha: ZerodhaSettings = ZerodhaSettings()
    telegram: TelegramSettings = TelegramSettings()
    server: ServerSettings = ServerSettings()
    toggles: AppToggles = AppToggles()

    BASE_STOP_LOSS_POINTS: float = Field(default_factory=lambda: StrategySettings().base_stop_loss_points)
    BASE_TARGET_POINTS: float = Field(default_factory=lambda: StrategySettings().base_target_points)
    CONFIDENCE_THRESHOLD: float = Field(default_factory=lambda: StrategySettings().confidence_threshold)
    MIN_SIGNAL_SCORE: int = Field(default_factory=lambda: StrategySettings().min_signal_score)
    ATR_PERIOD: int = Field(default_factory=lambda: StrategySettings().atr_period)
    ATR_SL_MULTIPLIER: float = Field(default_factory=lambda: StrategySettings().atr_sl_multiplier)
    ATR_TP_MULTIPLIER: float = Field(default_factory=lambda: StrategySettings().atr_tp_multiplier)

    SPOT_SYMBOL: str = Field(default_factory=lambda: InstrumentSettings().spot_symbol)
    TRADE_SYMBOL: str = Field(default_factory=lambda: InstrumentSettings().trade_symbol)
    TRADE_EXCHANGE: str = Field(default_factory=lambda: InstrumentSettings().trade_exchange)
    INSTRUMENT_TOKEN: int = Field(default_factory=lambda: InstrumentSettings().spot_token)
    NIFTY_LOT_SIZE: int = Field(default_factory=lambda: InstrumentSettings().nifty_lot_size)
    MIN_LOTS: int = Field(default_factory=lambda: InstrumentSettings().min_lots)
    MAX_LOTS: int = Field(default_factory=lambda: InstrumentSettings().max_lots)
    STRIKE_RANGE: int = Field(default_factory=lambda: InstrumentSettings().strike_range)

    TIME_FILTER_START: str = Field(default_factory=lambda: StrategySettings().time_filter_start)
    TIME_FILTER_END: str = Field(default_factory=lambda: StrategySettings().time_filter_end)

    HEALTH_PORT: int = Field(default_factory=lambda: ServerSettings().port)

    model_config = SettingsConfigDict(env_file=_find_env_file(), extra="ignore")


settings = AppSettings()