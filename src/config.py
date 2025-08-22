from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional, List

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, NonNegativeFloat, NonNegativeInt
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------- helpers ----------

def _find_env_file() -> Optional[Path]:
    p = Path.cwd()
    for _ in range(6):
        f = p / ".env"
        if f.exists():
            return f
        p = p.parent
    return None


def _as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y"}


def _csv_ints(v: Optional[str]) -> List[int]:
    if not v:
        return []
    out: List[int] = []
    for part in v.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


# ---------- component models ----------

class DataSettings(BaseModel):
    warmup_bars: PositiveInt = 30
    lookback_minutes: PositiveInt = 60
    timeframe: Literal["minute", "5minute"] = "minute"


class RiskConfig(BaseModel):
    default_equity: PositiveFloat = 30000.0
    risk_per_trade: NonNegativeFloat = 0.01   # fraction (0.01 = 1%)
    max_trades_per_day: PositiveInt = 5
    consecutive_loss_limit: NonNegativeInt = 3
    max_daily_drawdown_pct: NonNegativeFloat = 0.05


class ZerodhaConfig(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    public_token: Optional[str] = None
    enctoken: Optional[str] = None


class TelegramConfig(BaseModel):
    enabled: bool = True
    bot_token: Optional[str] = None
    chat_id: Optional[int] = None
    extra_admin_ids: List[int] = Field(default_factory=list)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class StrategyConfig(BaseModel):
    # gates
    min_bars_for_signal: int = 10
    min_signal_score: int = 2
    confidence_threshold: float = 4.0
    # indicators
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    adx_period: int = 14
    adx_trend_strength: int = 20
    di_diff_threshold: float = 10.0
    # ATR/SL/TP
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 3.0
    sl_confidence_adj: float = 0.12
    tp_confidence_adj: float = 0.35
    # regime tweaks
    trend_tp_boost: float = 0.6
    trend_sl_relax: float = 0.2
    range_tp_tighten: float = -0.4
    range_sl_tighten: float = -0.2


class InstrumentsConfig(BaseModel):
    spot_symbol: str = "NSE:NIFTY 50"
    trade_symbol: str = "NIFTY"
    trade_exchange: str = "NFO"  # corrected: matches INSTRUMENTS__TRADE_EXCHANGE
    instrument_token: int = 256265
    nifty_lot_size: int = 75
    min_lots: int = 1
    max_lots: int = 10
    strike_range: int = 3


class ExecutorConfig(BaseModel):
    exchange: str = "NFO"
    order_product: str = "NRML"
    order_variety: str = "regular"
    entry_order_type: str = "LIMIT"
    tick_size: float = 0.05
    exchange_freeze_qty: int = 900
    preferred_exit_mode: Literal["REGULAR", "OCO", "AUTO"] = "REGULAR"
    use_slm_exit: bool = True
    partial_tp_enable: bool = False
    tp1_qty_ratio: float = 0.5
    breakeven_ticks: int = 2
    enable_trailing: bool = True
    trailing_atr_multiplier: float = 1.5
    fee_per_lot: float = 20.0


# ---------- master settings ----------

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # top-level
    enable_live_trading: bool = False
    allow_offhours_testing: bool = False
    log_level: str = "INFO"

    # components
    data: DataSettings = DataSettings()
    risk: RiskConfig = RiskConfig()
    zerodha: ZerodhaConfig = ZerodhaConfig()
    telegram: TelegramConfig = TelegramConfig()
    server: ServerConfig = ServerConfig()
    strategy: StrategyConfig = StrategyConfig()
    instruments: InstrumentsConfig = InstrumentsConfig()
    executor: ExecutorConfig = ExecutorConfig()

    # mirrors / helpers
    @property
    def DEFAULT_EQUITY(self) -> float:
        return float(self.risk.default_equity)

    @property
    def RISK_PER_TRADE(self) -> float:
        return float(self.risk.risk_per_trade)

    @property
    def api(self) -> SimpleNamespace:
        return SimpleNamespace(
            zerodha_api_key=self.zerodha.api_key,
            zerodha_api_secret=self.zerodha.api_secret,
            zerodha_access_token=self.zerodha.access_token,
            telegram_bot_token=self.telegram.bot_token,
            telegram_chat_id=self.telegram.chat_id,
        )

    @property
    def telegram_ready(self) -> bool:
        return bool(self.telegram.enabled and self.telegram.bot_token and self.telegram.chat_id)

    def debug_summary(self) -> dict:
        def mask(v: Optional[str]) -> str:
            if not v:
                return "absent"
            return f"present({len(v)} chars)"
        return {
            "telegram": {
                "enabled": bool(self.telegram.enabled),
                "bot_token": mask(self.telegram.bot_token),
                "chat_id": "present" if self.telegram.chat_id else "absent",
                "extra_admin_ids": self.telegram.extra_admin_ids,
            },
            "zerodha": {
                "api_key": mask(self.zerodha.api_key),
                "access_token": mask(self.zerodha.access_token),
            },
            "log_level": self.log_level,
            "env_file": str(_find_env_file() or "none"),
        }


settings = AppSettings()