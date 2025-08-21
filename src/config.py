# src/config.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from pydantic.functional_validators import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Optional[Path]:
    p = Path.cwd()
    for _ in range(6):
        f = p / ".env"
        if f.exists():
            return f
        p = p.parent
    return None


class DataSettings(BaseModel):
    warmup_bars: PositiveInt = Field(30, alias="WARMUP_BARS")
    lookback_minutes: PositiveInt = Field(60, alias="DATA_LOOKBACK_MINUTES")
    timeframe: Literal["minute", "5minute"] = Field("minute", alias="HISTORICAL_TIMEFRAME")


class RiskConfig(BaseModel):
    # accept RISK_PER_TRADE_PCT (e.g. 0.5 => 0.5%)
    default_equity: PositiveFloat = Field(30000.0, alias="DEFAULT_EQUITY")
    risk_per_trade: NonNegativeFloat = 0.01

    @field_validator("risk_per_trade", mode="before")
    @classmethod
    def _from_pct(cls, v, values):
        # Prefer flat env RISK_PER_TRADE_PCT when present
        import os
        pct = os.environ.get("RISK_PER_TRADE_PCT")
        if pct is not None:
            try:
                return float(pct) / 100.0
            except Exception:
                pass
        return v

    max_trades_per_day: PositiveInt = Field(1, alias="MAX_TRADES_PER_DAY")
    consecutive_loss_limit: NonNegativeInt = Field(3, alias="CONSECUTIVE_LOSS_LIMIT")
    max_daily_drawdown_pct: NonNegativeFloat = Field(0.05, alias="MAX_DAILY_DRAWDOWN_PCT")


class ZerodhaConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, alias="ZERODHA_API_KEY")
    api_secret: Optional[str] = Field(default=None, alias="ZERODHA_API_SECRET")
    access_token: Optional[str] = Field(default=None, alias="ZERODHA_ACCESS_TOKEN")
    public_token: Optional[str] = Field(default=None, alias="ZERODHA_PUBLIC_TOKEN")
    enctoken: Optional[str] = Field(default=None, alias="ZERODHA_ENCTOKEN")


class TelegramConfig(BaseModel):
    enabled: bool = Field(True, alias="ENABLE_TELEGRAM")
    bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    chat_id: Optional[int] = Field(default=None, alias="TELEGRAM_CHAT_ID")
    extra_admin_ids: Optional[str] = Field(default=None, alias="TELEGRAM_EXTRA_ADMINS")  # "111,222"

    @field_validator("extra_admin_ids", mode="after")
    def _split_admins(cls, v):
        if not v:
            return []
        if isinstance(v, list):
            return v
        return [int(x.strip()) for x in str(v).split(",") if x.strip()]


class ServerConfig(BaseModel):
    host: str = Field("0.0.0.0", alias="HEALTH_HOST")
    port: int = Field(8000, alias="HEALTH_PORT")


class StrategyConfig(BaseModel):
    min_bars_for_signal: int = Field(30, alias="MIN_BARS_FOR_SIGNAL")
    min_signal_score: int = Field(5, alias="MIN_SIGNAL_SCORE")
    confidence_threshold: float = Field(6.0, alias="CONFIDENCE_THRESHOLD")
    atr_period: int = Field(14, alias="ATR_PERIOD")
    base_stop_loss_points: float = Field(20.0, alias="BASE_STOP_LOSS_POINTS")
    base_target_points: float = Field(40.0, alias="BASE_TARGET_POINTS")
    atr_sl_multiplier: float = Field(1.5, alias="ATR_SL_MULTIPLIER")
    atr_tp_multiplier: float = Field(3.0, alias="ATR_TP_MULTIPLIER")
    sl_confidence_adj: float = Field(0.2, alias="SL_CONFIDENCE_ADJ")
    tp_confidence_adj: float = Field(0.3, alias="TP_CONFIDENCE_ADJ")
    strike_selection_range: int = Field(3, alias="STRIKE_SELECTION_RANGE")
    di_diff_threshold: float = Field(10.0, alias="DI_DIFF_THRESHOLD")


class InstrumentsConfig(BaseModel):
    spot_symbol: str = Field("NSE:NIFTY 50", alias="SPOT_SYMBOL")
    trade_symbol: str = Field("NIFTY", alias="TRADE_SYMBOL")
    instrument_token: int = Field(256265, alias="INSTRUMENT_TOKEN")
    nifty_lot_size: int = Field(75, alias="NIFTY_LOT_SIZE")
    min_lots: int = Field(1, alias="MIN_LOTS")
    max_lots: int = Field(10, alias="MAX_LOTS")
    strike_range: int = Field(0, alias="STRIKE_RANGE")  # ATM by default

    @field_validator("strike_range", mode="before")
    @classmethod
    def _prefer_selection_range(cls, v):
        import os
        sel = os.environ.get("STRIKE_SELECTION_RANGE")
        if sel is not None:
            try:
                return int(sel)
            except Exception:
                pass
        return v


class ExecutorConfig(BaseModel):
    # New fields with common aliases to your env
    exchange: str = Field("NFO", alias="TRADE_EXCHANGE")
    order_product: str = Field("NRML", alias="ORDER_PRODUCT")
    order_variety: str = Field("regular", alias="ORDER_VARIETY")
    entry_order_type: str = Field("LIMIT", alias="ENTRY_ORDER_TYPE")
    tick_size: float = Field(0.05, alias="TICK_SIZE")
    exchange_freeze_qty: int = Field(900, alias="NFO_FREEZE_QTY")

    partial_tp_enable: bool = Field(False, alias="PARTIAL_TP_ENABLE")
    tp1_qty_ratio: float = Field(0.5, alias="TP1_QTY_RATIO")  # 0..1
    breakeven_ticks: int = Field(2, alias="BREAKEVEN_TICKS")

    enable_trailing: bool = Field(True, alias="ENABLE_TRAILING")
    trailing_atr_multiplier: float = Field(1.5, alias="TRAILING_ATR_MULTIPLIER")
    use_slm_exit: bool = Field(True, alias="USE_SLM_EXIT")

    preferred_exit_mode: Literal["REGULAR", "OCO", "AUTO"] = Field("REGULAR", alias="PREFERRED_EXIT_MODE")
    fee_per_lot: float = Field(20.0, alias="FEE_PER_LOT")


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_nested_delimiter="__",
        extra="ignore",
    )

    # top-level flags
    enable_live_trading: bool = Field(False, alias="ENABLE_LIVE_TRADING")
    allow_offhours_testing: bool = Field(False, alias="ALLOW_OFFHOURS_TESTING")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    data: DataSettings = DataSettings()
    risk: RiskConfig = RiskConfig()
    zerodha: ZerodhaConfig = ZerodhaConfig()
    telegram: TelegramConfig = TelegramConfig()
    server: ServerConfig = ServerConfig()
    strategy: StrategyConfig = StrategyConfig()
    instruments: InstrumentsConfig = InstrumentsConfig()
    executor: ExecutorConfig = ExecutorConfig()

    # mirrors for legacy reads
    @property
    def DEFAULT_EQUITY(self) -> float: return float(self.risk.default_equity)
    @property
    def RISK_PER_TRADE(self) -> float: return float(self.risk.risk_per_trade)
    @property
    def MAX_TRADES_PER_DAY(self) -> int: return int(self.risk.max_trades_per_day)
    @property
    def MAX_DAILY_DRAWDOWN_PCT(self) -> float: return float(self.risk.max_daily_drawdown_pct)
    @property
    def PREFERRED_EXIT_MODE(self) -> str: return str(self.executor.preferred_exit_mode)

    # Legacy .api proxy
    @property
    def api(self) -> SimpleNamespace:  # type: ignore[override]
        return SimpleNamespace(
            zerodha_api_key=self.zerodha.api_key,
            zerodha_api_secret=self.zerodha.api_secret,
            zerodha_access_token=self.zerodha.access_token,
            zerodha_public_token=self.zerodha.public_token,
            zerodha_enctoken=self.zerodha.enctoken,
            telegram_bot_token=self.telegram.bot_token,
            telegram_chat_id=self.telegram.chat_id,
        )


settings = AppSettings()