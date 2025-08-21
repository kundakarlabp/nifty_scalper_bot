from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional, List

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, NonNegativeFloat, NonNegativeInt
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Optional[Path]:
    """Look up to 6 parents for a .env file."""
    p = Path.cwd()
    for _ in range(6):
        cand = p / ".env"
        if cand.exists():
            return cand
        p = p.parent
    return None


# ---------------- Component models ----------------

class DataSettings(BaseModel):
    warmup_bars: PositiveInt = Field(30, alias="WARMUP_BARS")
    lookback_minutes: PositiveInt = Field(60, alias="DATA_LOOKBACK_MINUTES")
    timeframe: Literal["minute", "5minute"] = Field("minute", alias="HISTORICAL_TIMEFRAME")


class RiskConfig(BaseModel):
    default_equity: PositiveFloat = 30000.0
    # will be overridden by RISK_PER_TRADE_PCT (see post-init)
    risk_per_trade: NonNegativeFloat = Field(0.01, alias="RISK_PER_TRADE")
    max_trades_per_day: PositiveInt = 5
    consecutive_loss_limit: NonNegativeInt = 3
    max_daily_drawdown_pct: NonNegativeFloat = 0.05


class ZerodhaConfig(BaseModel):
    api_key: Optional[str] = Field(default=None, alias="ZERODHA_API_KEY")
    api_secret: Optional[str] = Field(default=None, alias="ZERODHA_API_SECRET")
    access_token: Optional[str] = Field(default=None, alias="ZERODHA_ACCESS_TOKEN")
    public_token: Optional[str] = None
    enctoken: Optional[str] = None


class TelegramConfig(BaseModel):
    # Support both ENABLE_TELEGRAM and TELEGRAM_ENABLED
    enabled: bool = Field(default=True, alias="ENABLE_TELEGRAM")
    bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    chat_id: Optional[int] = Field(default=None, alias="TELEGRAM_CHAT_ID")
    # Accept CSV string from env: "111,222,333"
    extra_admin_ids: List[int] = Field(default_factory=list)

    @staticmethod
    def _parse_csv_ids(raw: Optional[str]) -> List[int]:
        if not raw:
            return []
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        out: List[int] = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                continue
        return out


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class StrategyConfig(BaseModel):
    # entry gates
    min_bars_for_signal: int = Field(10, alias="MIN_BARS_FOR_SIGNAL")
    min_signal_score: int = Field(2, alias="MIN_SIGNAL_SCORE")
    confidence_threshold: float = Field(4.0, alias="CONFIDENCE_THRESHOLD")

    # indicators
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    adx_period: int = 14
    adx_trend_strength: int = 20
    di_diff_threshold: float = 10.0

    # ATR/SL/TP
    atr_period: int = Field(14, alias="ATR_PERIOD")
    atr_sl_multiplier: float = Field(1.5, alias="ATR_SL_MULTIPLIER")
    atr_tp_multiplier: float = Field(3.0, alias="ATR_TP_MULTIPLIER")
    sl_confidence_adj: float = Field(0.2, alias="SL_CONFIDENCE_ADJ")
    tp_confidence_adj: float = Field(0.3, alias="TP_CONFIDENCE_ADJ")

    # regime tweaks
    trend_tp_boost: float = 0.6
    trend_sl_relax: float = 0.2
    range_tp_tighten: float = -0.4
    range_sl_tighten: float = -0.2


class InstrumentsConfig(BaseModel):
    spot_symbol: str = Field("NSE:NIFTY 50", alias="SPOT_SYMBOL")
    trade_symbol: str = Field("NIFTY", alias="TRADE_SYMBOL")
    exchange: str = Field("NFO", alias="TRADE_EXCHANGE")
    instrument_token: int = Field(256265, alias="INSTRUMENT_TOKEN")
    nifty_lot_size: int = Field(75, alias="NIFTY_LOT_SIZE")
    min_lots: int = Field(1, alias="MIN_LOTS")
    max_lots: int = Field(10, alias="MAX_LOTS")
    strike_range: int = Field(3, alias="STRIKE_SELECTION_RANGE")


class ExecutorConfig(BaseModel):
    exchange: str = Field("NFO", alias="TRADE_EXCHANGE")
    order_product: str = "NRML"
    order_variety: str = "regular"
    entry_order_type: str = "LIMIT"
    tick_size: float = Field(0.05, alias="TICK_SIZE")
    exchange_freeze_qty: int = Field(900, alias="NFO_FREEZE_QTY")

    preferred_exit_mode: Literal["REGULAR", "OCO", "AUTO"] = "REGULAR"
    use_slm_exit: bool = True
    partial_tp_enable: bool = False
    tp1_qty_ratio: float = 0.5
    breakeven_ticks: int = 2
    enable_trailing: bool = True
    trailing_atr_multiplier: float = 1.5

    fee_per_lot: float = 20.0


# ---------------- App settings ----------------

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        extra="ignore",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

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

    # Compatibility shims for alternate env names
    TELEGRAM_ENABLED: Optional[bool] = None
    TELEGRAM_EXTRA_ADMINS: Optional[str] = None  # CSV
    RISK_PER_TRADE_PCT: Optional[float] = None

    def model_post_init(self, __ctx) -> None:  # type: ignore[override]
        # Allow TELEGRAM_ENABLED as well
        if self.TELEGRAM_ENABLED is not None:
            self.telegram.enabled = bool(self.TELEGRAM_ENABLED)
        # Accept CSV for extra admins
        if self.TELEGRAM_EXTRA_ADMINS:
            self.telegram.extra_admin_ids = TelegramConfig._parse_csv_ids(self.TELEGRAM_EXTRA_ADMINS)
        # Percent convenience
        if self.RISK_PER_TRADE_PCT is not None:
            try:
                self.risk.risk_per_trade = float(self.RISK_PER_TRADE_PCT) / 100.0
            except Exception:
                pass

    # Convenience mirrors
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


settings = AppSettings()