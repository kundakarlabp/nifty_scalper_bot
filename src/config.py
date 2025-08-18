# src/config.py
"""
Pydantic v2-compatible settings with .env support and legacy Config shim.

- Uses pydantic-settings.BaseSettings (not pydantic.BaseSettings)
- Reads your existing UPPERCASE env names (case-insensitive)
- Exposes nested objects: settings.api, settings.telegram, settings.executor,
  settings.strategy, settings.risk
- Provides a minimal legacy `Config` class for old callsites:
    - Config.SPOT_SYMBOL
    - Config.INSTRUMENT_TOKEN
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# -------------------- Nested config models --------------------

class APISettings(BaseModel):
    zerodha_api_key: str = ""
    zerodha_api_secret: str = ""
    zerodha_access_token: str = ""


class TelegramConfig(BaseModel):
    bot_token: str = ""
    chat_id: int = 0


class ExecutorConfig(BaseModel):
    trade_exchange: str = "NFO"          # e.g. "NFO"
    trade_symbol: str = "NIFTY"          # index base name
    market_open: str = "09:15"           # HH:MM (local)
    market_close: str = "15:30"          # HH:MM (local)
    data_lookback_minutes: int = 60
    nifty_lot_size: int = 50
    tick_size: float = 0.05
    nfo_freeze_qty: int = 1800


class StrategyConfig(BaseModel):
    min_signal_score: int = 1
    confidence_threshold: float = 4.0
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 3.0
    sl_confidence_adj: float = 0.2
    tp_confidence_adj: float = 0.3
    spot_symbol: str = "NSE:NIFTY 50"
    strike_selection_range: int = 3
    min_bars_for_signal: int = 30


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 0.01       # 1%
    max_trades_per_day: int = 15
    consecutive_loss_limit: int = 3
    max_daily_drawdown_pct: float = 0.03   # 3%
    min_lots: int = 1
    max_lots: int = 10


# -------------------- Top-level settings --------------------

class Settings(BaseSettings):
    # Global toggles / logging
    enable_live_trading: bool = Field(False, validation_alias="ENABLE_LIVE_TRADING")
    enable_telegram: bool = Field(True, validation_alias="ENABLE_TELEGRAM")
    allow_offhours_testing: bool = Field(True, validation_alias="ALLOW_OFFHOURS_TESTING")
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")

    # API (flat; used to build nested APISettings)
    _zerodha_api_key: str = Field("", validation_alias="ZERODHA_API_KEY")
    _zerodha_api_secret: str = Field("", validation_alias="ZERODHA_API_SECRET")
    _zerodha_access_token: str = Field("", validation_alias="ZERODHA_ACCESS_TOKEN")

    # Telegram (flat)
    _telegram_bot_token: str = Field("", validation_alias="TELEGRAM_BOT_TOKEN")
    _telegram_chat_id: int = Field(0, validation_alias="TELEGRAM_CHAT_ID")

    # Executor (flat)
    _time_filter_start: str = Field("09:15", validation_alias="TIME_FILTER_START")
    _time_filter_end: str = Field("15:30", validation_alias="TIME_FILTER_END")
    _data_lookback_minutes: int = Field(60, validation_alias="DATA_LOOKBACK_MINUTES")
    _nfo_freeze_qty: int = Field(1800, validation_alias="NFO_FREEZE_QTY")
    _tick_size: float = Field(0.05, validation_alias="TICK_SIZE")
    _nifty_lot_size: int = Field(50, validation_alias="NIFTY_LOT_SIZE")
    _trade_symbol: str = Field("NIFTY", validation_alias="TRADE_SYMBOL")
    _trade_exchange: str = Field("NFO", validation_alias="TRADE_EXCHANGE")

    # Strategy (flat)
    _min_signal_score: int = Field(1, validation_alias="MIN_SIGNAL_SCORE")
    _confidence_threshold: float = Field(4.0, validation_alias="CONFIDENCE_THRESHOLD")
    _atr_period: int = Field(14, validation_alias="ATR_PERIOD")
    _atr_sl_multiplier: float = Field(1.5, validation_alias="ATR_SL_MULTIPLIER")
    _atr_tp_multiplier: float = Field(3.0, validation_alias="ATR_TP_MULTIPLIER")
    _sl_confidence_adj: float = Field(0.2, validation_alias="SL_CONFIDENCE_ADJ")
    _tp_confidence_adj: float = Field(0.3, validation_alias="TP_CONFIDENCE_ADJ")
    _spot_symbol: str = Field("NSE:NIFTY 50", validation_alias="SPOT_SYMBOL")
    _strike_selection_range: int = Field(3, validation_alias="STRIKE_SELECTION_RANGE")
    _min_bars_for_signal: int = Field(30, validation_alias="MIN_BARS_FOR_SIGNAL")

    # Risk (flat)
    _risk_per_trade_pct: float = Field(0.01, validation_alias="RISK_PER_TRADE_PCT")
    _max_trades_per_day: int = Field(15, validation_alias="MAX_TRADES_PER_DAY")
    _consecutive_loss_limit: int = Field(3, validation_alias="CONSECUTIVE_LOSS_LIMIT")
    _max_daily_drawdown_pct: float = Field(0.03, validation_alias="MAX_DAILY_DRAWDOWN_PCT")
    _min_lots: int = Field(1, validation_alias="MIN_LOTS")
    _max_lots: int = Field(10, validation_alias="MAX_LOTS")

    # Pydantic-settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,          # allow UPPERCASE env names
    )

    # --------- nested builders (properties) ---------

    @property
    def api(self) -> APISettings:
        return APISettings(
            zerodha_api_key=self._zerodha_api_key,
            zerodha_api_secret=self._zerodha_api_secret,
            zerodha_access_token=self._zerodha_access_token,
        )

    @property
    def telegram(self) -> TelegramConfig:
        return TelegramConfig(
            bot_token=self._telegram_bot_token,
            chat_id=int(self._telegram_chat_id or 0),
        )

    @property
    def executor(self) -> ExecutorConfig:
        return ExecutorConfig(
            trade_exchange=self._trade_exchange,
            trade_symbol=self._trade_symbol,
            market_open=self._time_filter_start,
            market_close=self._time_filter_end,
            data_lookback_minutes=int(self._data_lookback_minutes),
            nifty_lot_size=int(self._nifty_lot_size),
            tick_size=float(self._tick_size),
            nfo_freeze_qty=int(self._nfo_freeze_qty),
        )

    @property
    def strategy(self) -> StrategyConfig:
        return StrategyConfig(
            min_signal_score=int(self._min_signal_score),
            confidence_threshold=float(self._confidence_threshold),
            atr_period=int(self._atr_period),
            atr_sl_multiplier=float(self._atr_sl_multiplier),
            atr_tp_multiplier=float(self._atr_tp_multiplier),
            sl_confidence_adj=float(self._sl_confidence_adj),
            tp_confidence_adj=float(self._tp_confidence_adj),
            spot_symbol=self._spot_symbol,
            strike_selection_range=int(self._strike_selection_range),
            min_bars_for_signal=int(self._min_bars_for_signal),
        )

    @property
    def risk(self) -> RiskConfig:
        return RiskConfig(
            risk_per_trade_pct=float(self._risk_per_trade_pct),
            max_trades_per_day=int(self._max_trades_per_day),
            consecutive_loss_limit=int(self._consecutive_loss_limit),
            max_daily_drawdown_pct=float(self._max_daily_drawdown_pct),
            min_lots=int(self._min_lots),
            max_lots=int(self._max_lots),
        )


# Singleton settings object
settings = Settings()


# -------------------- Legacy shim for older imports --------------------

class Config:
    """
    Minimal compatibility for older modules expecting `from src.config import Config`.
    Only attributes actually referenced in legacy code are provided.
    """
    # Commonly used by utils/strike_selector
    SPOT_SYMBOL: str = settings.strategy.spot_symbol

    # NIFTY index token (override via env if you need: INSTRUMENT_TOKEN)
    INSTRUMENT_TOKEN: int = int(os.getenv("INSTRUMENT_TOKEN", "256265"))
