# src/config.py
"""
Centralized, validated configuration using Pydantic v2 / pydantic-settings v2.

Exports a single source of truth: `settings` (AppSettings instance).

Back-compat features:
- Read-only properties that mirror common nested values (e.g., settings.DEFAULT_EQUITY)
- A legacy `.api` proxy so old code like `settings.api.zerodha_api_key` keeps working
- No duplicate fields; no recursive default_factory; single env-loading surface
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, NonNegativeFloat, NonNegativeInt
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Optional[Path]:
    """
    Find a .env by walking up from CWD up to 6 levels; return None if not found.
    """
    path = Path.cwd()
    for _ in range(6):
        candidate = path / ".env"
        if candidate.exists():
            return candidate
        path = path.parent
    return None


# ---------------- Component settings (pure data models; env is handled by AppSettings) ----------------

class DataSettings(BaseModel):
    warmup_bars: PositiveInt = Field(30, alias="WARMUP_BARS")
    lookback_minutes: PositiveInt = Field(60, alias="DATA_LOOKBACK_MINUTES")
    timeframe: Literal["minute", "5minute"] = Field("minute", alias="HISTORICAL_TIMEFRAME")


class RiskConfig(BaseModel):
    default_equity: PositiveFloat = 30000.0
    risk_per_trade: NonNegativeFloat = 0.01
    max_trades_per_day: PositiveInt = 1
    consecutive_loss_limit: NonNegativeInt = 3
    max_daily_drawdown_pct: NonNegativeFloat = 0.05


class ZerodhaConfig(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None  # aka KITE_ACCESS_TOKEN
    public_token: Optional[str] = None
    enctoken: Optional[str] = None


class TelegramConfig(BaseModel):
    enabled: bool = True
    bot_token: Optional[str] = None
    chat_id: Optional[int] = None


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class StrategyConfig(BaseModel):
    min_bars_for_signal: int = 30
    min_signal_score: int = 5
    confidence_threshold: float = 6.0
    atr_period: int = 14
    base_stop_loss_points: float = 20.0
    base_target_points: float = 40.0
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 3.0
    sl_confidence_adj: float = 0.2
    tp_confidence_adj: float = 0.3
    strike_selection_range: int = 3
    di_diff_threshold: float = 10.0  # used by regime detector


class InstrumentsConfig(BaseModel):
    spot_symbol: str = "NSE:NIFTY 50"
    trade_symbol: str = "NIFTY"
    instrument_token: int = 256265
    nifty_lot_size: int = 75
    min_lots: int = 1
    max_lots: int = 10
    strike_range: int = 0  # ATM-only default


class ExecutorConfig(BaseModel):
    preferred_exit_mode: Literal["REGULAR", "OCO", "AUTO"] = "REGULAR"
    fee_per_lot: float = 20.0
    enable_trailing: bool = True
    trailing_atr_multiplier: float = 1.5


# ---------------- Master application settings (single env-loading surface) ----------------

class AppSettings(BaseSettings):
    """
    Main settings container. Only this class reads environment variables.

    Env mapping strategy:
    - Nested fields via ENV_NESTED_DELIMITER="__"
      e.g., RISK__DEFAULT_EQUITY, TELEGRAM__ENABLED
    - Flat legacy names supported via validation_alias on selected top-level shims.
    - A .env file is auto-discovered by _find_env_file().
    """
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        extra="ignore",
        env_nested_delimiter="__",
    )

    # App-level toggles (with legacy env names for back-compat)
    enable_live_trading: bool = Field(False, alias="ENABLE_LIVE_TRADING")
    allow_offhours_testing: bool = Field(False, alias="ALLOW_OFFHOURS_TESTING")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Component models
    data: DataSettings = DataSettings()
    risk: RiskConfig = RiskConfig()
    zerodha: ZerodhaConfig = ZerodhaConfig()
    telegram: TelegramConfig = TelegramConfig()
    server: ServerConfig = ServerConfig()
    strategy: StrategyConfig = StrategyConfig()
    instruments: InstrumentsConfig = InstrumentsConfig()
    executor: ExecutorConfig = ExecutorConfig()

    # ---- Read-only mirrors for common fields (no duplicates; no recursion) ----
    @property
    def DEFAULT_EQUITY(self) -> float:
        return float(self.risk.default_equity)

    @property
    def RISK_PER_TRADE(self) -> float:
        return float(self.risk.risk_per_trade)

    @property
    def MAX_TRADES_PER_DAY(self) -> int:
        return int(self.risk.max_trades_per_day)

    @property
    def MAX_DAILY_DRAWDOWN_PCT(self) -> float:
        return float(self.risk.max_daily_drawdown_pct)

    @property
    def PREFERRED_EXIT_MODE(self) -> str:
        # mirror executor preference (for legacy reads)
        return str(self.executor.preferred_exit_mode)

    # ---- Legacy `.api` proxy for old code paths ----
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
