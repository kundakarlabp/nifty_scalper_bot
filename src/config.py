# src/config.py
from __future__ import annotations

import os
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# -------- Nested models --------
class APISettings(BaseModel):
    zerodha_api_key: str = ""
    zerodha_api_secret: str = ""
    zerodha_access_token: str = ""


class TelegramConfig(BaseModel):
    bot_token: str = ""
    chat_id: int = 0


class ExecutorConfig(BaseModel):
    trade_exchange: str = "NFO"
    trade_symbol: str = "NIFTY"
    market_open: str = "09:15"
    market_close: str = "15:30"
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
    risk_per_trade_pct: float = 0.01
    max_trades_per_day: int = 15
    consecutive_loss_limit: int = 3
    max_daily_drawdown_pct: float = 0.03
    min_lots: int = 1
    max_lots: int = 10


# -------- Settings (no leading underscores!) --------
class Settings(BaseSettings):
    # Modes / logging
    enable_live_trading: bool = False
    enable_telegram: bool = True
    allow_offhours_testing: bool = True
    log_level: str = "INFO"

    # API
    zerodha_api_key: str = ""
    zerodha_api_secret: str = ""
    zerodha_access_token: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: int = 0

    # Executor
    time_filter_start: str = "09:15"
    time_filter_end: str = "15:30"
    data_lookback_minutes: int = 60
    nfo_freeze_qty: int = 1800
    tick_size: float = 0.05
    nifty_lot_size: int = 50
    trade_symbol: str = "NIFTY"
    trade_exchange: str = "NFO"

    # Strategy
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

    # Risk
    risk_per_trade_pct: float = 0.01
    max_trades_per_day: int = 15
    consecutive_loss_limit: int = 3
    max_daily_drawdown_pct: float = 0.03
    min_lots: int = 1
    max_lots: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,  # UPPERCASE envs map automatically
    )

    # ---- nested views ----
    @property
    def api(self) -> APISettings:
        return APISettings(
            zerodha_api_key=self.zerodha_api_key,
            zerodha_api_secret=self.zerodha_api_secret,
            zerodha_access_token=self.zerodha_access_token,
        )

    @property
    def telegram(self) -> TelegramConfig:
        return TelegramConfig(
            bot_token=self.telegram_bot_token,
            chat_id=int(self.telegram_chat_id or 0),
        )

    @property
    def executor(self) -> ExecutorConfig:
        return ExecutorConfig(
            trade_exchange=self.trade_exchange,
            trade_symbol=self.trade_symbol,
            market_open=self.time_filter_start,
            market_close=self.time_filter_end,
            data_lookback_minutes=int(self.data_lookback_minutes),
            nifty_lot_size=int(self.nifty_lot_size),
            tick_size=float(self.tick_size),
            nfo_freeze_qty=int(self.nfo_freeze_qty),
        )

    @property
    def strategy(self) -> StrategyConfig:
        return StrategyConfig(
            min_signal_score=int(self.min_signal_score),
            confidence_threshold=float(self.confidence_threshold),
            atr_period=int(self.atr_period),
            atr_sl_multiplier=float(self.atr_sl_multiplier),
            atr_tp_multiplier=float(self.atr_tp_multiplier),
            sl_confidence_adj=float(self.sl_confidence_adj),
            tp_confidence_adj=float(self.tp_confidence_adj),
            spot_symbol=self.spot_symbol,
            strike_selection_range=int(self.strike_selection_range),
            min_bars_for_signal=int(self.min_bars_for_signal),
        )

    @property
    def risk(self) -> RiskConfig:
        return RiskConfig(
            risk_per_trade_pct=float(self.risk_per_trade_pct),
            max_trades_per_day=int(self.max_trades_per_day),
            consecutive_loss_limit=int(self.consecutive_loss_limit),
            max_daily_drawdown_pct=float(self.max_daily_drawdown_pct),
            min_lots=int(self.min_lots),
            max_lots=int(self.max_lots),
        )


# singleton
settings = Settings()


# -------- Legacy shim for old imports --------
class Config:
    SPOT_SYMBOL: str = settings.strategy.spot_symbol
    INSTRUMENT_TOKEN: int = int(os.getenv("INSTRUMENT_TOKEN", "256265"))
