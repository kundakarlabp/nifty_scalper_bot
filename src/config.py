from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional, List

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


# ---------------- Component settings (pure data models; env handled by AppSettings) ----------------

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
    extra_admin_ids: List[int] = []


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
    # ATR/SL/TP
    atr_period: int = 14
    base_stop_loss_points: float = 20.0
    base_target_points: float = 40.0
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 3.0
    sl_confidence_adj: float = 0.2
    tp_confidence_adj: float = 0.3
    # regime
    di_diff_threshold: float = 10.0
    # strike
    strike_selection_range: int = 3


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
    # optional dynamic/partials knobs
    partial_tp_enable: bool = False
    tp1_qty_ratio: float = 0.5
    breakeven_ticks: int = 2
    tick_size: float = 0.05
    exchange_freeze_qty: int = 900
    order_product: str = "NRML"
    order_variety: str = "regular"
    entry_order_type: str = "LIMIT"
    use_slm_exit: bool = True
    exchange: str = "NFO"


# ---------------- Master application settings (single env-loading surface) ----------------

class AppSettings(BaseSettings):
    """
    Main settings container. Only this class reads environment variables.

    Env mapping strategy:
    - Nested fields via ENV_NESTED_DELIMITER="__"
      e.g., RISK__DEFAULT_EQUITY, TELEGRAM__ENABLED
    - Flat legacy names supported via shim fields below (copied into nested in model_post_init).
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

    # -------- Legacy flat env shims (copied into nested on init) --------
    # Telegram
    _legacy_tg_enabled: Optional[bool] = Field(None, alias="ENABLE_TELEGRAM")
    _legacy_tg_enabled2: Optional[bool] = Field(None, alias="TELEGRAM_ENABLED")
    _legacy_tg_token: Optional[str] = Field(None, alias="TELEGRAM_BOT_TOKEN")
    _legacy_tg_chat: Optional[int] = Field(None, alias="TELEGRAM_CHAT_ID")
    _legacy_tg_admins: Optional[str] = Field(None, alias="TELEGRAM_EXTRA_ADMINS")
    # Zerodha
    _k_api: Optional[str] = Field(None, alias="ZERODHA_API_KEY")
    _k_secret: Optional[str] = Field(None, alias="ZERODHA_API_SECRET")
    _k_access: Optional[str] = Field(None, alias="ZERODHA_ACCESS_TOKEN")
    _k_public: Optional[str] = Field(None, alias="ZERODHA_PUBLIC_TOKEN")
    _k_enc: Optional[str] = Field(None, alias="ZERODHA_ENCTOKEN")
    # Instruments (common flats)
    _spot_symbol: Optional[str] = Field(None, alias="SPOT_SYMBOL")
    _trade_symbol: Optional[str] = Field(None, alias="TRADE_SYMBOL")
    _inst_token: Optional[int] = Field(None, alias="INSTRUMENT_TOKEN")
    _lot_size: Optional[int] = Field(None, alias="NIFTY_LOT_SIZE")
    _strike_sel: Optional[int] = Field(None, alias="STRIKE_SELECTION_RANGE")

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        # ---- Telegram shims ----
        if self._legacy_tg_enabled is not None:
            self.telegram.enabled = bool(self._legacy_tg_enabled)
        if self._legacy_tg_enabled2 is not None:
            self.telegram.enabled = bool(self._legacy_tg_enabled2)
        if self._legacy_tg_token:
            self.telegram.bot_token = self._legacy_tg_token
        if self._legacy_tg_chat is not None:
            try:
                self.telegram.chat_id = int(self._legacy_tg_chat)
            except Exception:
                pass
        if self._legacy_tg_admins:
            try:
                parts = [p.strip() for p in str(self._legacy_tg_admins).split(",") if p.strip()]
                self.telegram.extra_admin_ids = [int(p) for p in parts]
            except Exception:
                self.telegram.extra_admin_ids = []

        # ---- Zerodha shims ----
        z = self.zerodha
        if self._k_api: z.api_key = self._k_api
        if self._k_secret: z.api_secret = self._k_secret
        if self._k_access: z.access_token = self._k_access
        if self._k_public: z.public_token = self._k_public
        if self._k_enc: z.enctoken = self._k_enc

        # ---- Instruments shims ----
        inst = self.instruments
        if self._spot_symbol: inst.spot_symbol = self._spot_symbol
        if self._trade_symbol: inst.trade_symbol = self._trade_symbol
        if self._inst_token is not None:
            try: inst.instrument_token = int(self._inst_token)
            except Exception: pass
        if self._lot_size is not None:
            try: inst.nifty_lot_size = int(self._lot_size)
            except Exception: pass
        if self._strike_sel is not None:
            try:
                inst.strike_range = int(self._strike_sel)
            except Exception:
                pass

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