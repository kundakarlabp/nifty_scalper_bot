# src/config.py
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, NonNegativeFloat, NonNegativeInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> Optional[Path]:
    path = Path.cwd()
    for _ in range(6):
        cand = path / ".env"
        if cand.exists():
            return cand
        path = path.parent
    return None


# ---------- component models ----------
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
    access_token: Optional[str] = None
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
    # gates
    min_bars_for_signal: int = 30
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
    strike_range: int = 0  # ATM only by default


class ExecutorConfig(BaseModel):
    preferred_exit_mode: Literal["REGULAR", "OCO", "AUTO"] = "REGULAR"
    order_variety: Literal["regular", "amo"] = "regular"
    order_product: Literal["NRML", "MIS"] = "MIS"
    entry_order_type: Literal["LIMIT", "MARKET"] = "LIMIT"
    exchange: Literal["NFO", "NSE"] = "NFO"
    tick_size: float = 0.05
    exchange_freeze_qty: int = 1800
    fee_per_lot: float = 20.0
    partial_tp_enable: bool = True
    tp1_qty_ratio: float = 0.5
    breakeven_ticks: int = 2
    use_slm_exit: bool = True
    enable_trailing: bool = True
    trailing_atr_multiplier: float = 1.5


# ---------- master settings ----------
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        extra="ignore",
        env_nested_delimiter="__",
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

    # Overlay flat env names from .env/Railway
    @model_validator(mode="after")
    def _apply_flat_env_overrides(self) -> "AppSettings":
        # --- Zerodha (accept both ZERODHA_* and KITE_ACCESS_TOKEN) ---
        self.zerodha.api_key = os.getenv("ZERODHA_API_KEY", self.zerodha.api_key)
        self.zerodha.api_secret = os.getenv("ZERODHA_API_SECRET", self.zerodha.api_secret)
        self.zerodha.access_token = (
            os.getenv("ZERODHA_ACCESS_TOKEN", self.zerodha.access_token)
            or os.getenv("KITE_ACCESS_TOKEN", self.zerodha.access_token)
        )
        self.zerodha.public_token = os.getenv("KITE_PUBLIC_TOKEN", self.zerodha.public_token)
        self.zerodha.enctoken = os.getenv("ENCTOKEN", self.zerodha.enctoken)

        # --- Telegram ---
        self.telegram.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", self.telegram.bot_token)
        _chat = os.getenv("TELEGRAM_CHAT_ID")
        if _chat:
            try:
                self.telegram.chat_id = int(_chat)
            except Exception:
                pass
        _tg_enabled = os.getenv("ENABLE_TELEGRAM")
        if _tg_enabled is not None:
            self.telegram.enabled = str(_tg_enabled).strip().lower() in {"1", "true", "yes", "on"}

        # --- Data / scheduling ---
        self.data.lookback_minutes = int(os.getenv("DATA_LOOKBACK_MINUTES", self.data.lookback_minutes))
        # MIN_BARS_FOR_SIGNAL maps to strategy.min_bars_for_signal (keep data.warmup_bars separate)
        _mbfs = os.getenv("MIN_BARS_FOR_SIGNAL")
        if _mbfs:
            try:
                self.strategy.min_bars_for_signal = int(_mbfs)
            except Exception:
                pass

        # --- Instruments (flat) ---
        self.instruments.spot_symbol = os.getenv("SPOT_SYMBOL", self.instruments.spot_symbol)
        self.instruments.trade_symbol = os.getenv("TRADE_SYMBOL", self.instruments.trade_symbol)
        _it = os.getenv("INSTRUMENT_TOKEN")
        if _it:
            try:
                self.instruments.instrument_token = int(_it)
            except Exception:
                pass
        _lot = os.getenv("NIFTY_LOT_SIZE")
        if _lot:
            try:
                self.instruments.nifty_lot_size = int(_lot)
            except Exception:
                pass
        _min_lots = os.getenv("MIN_LOTS")
        if _min_lots:
            try:
                self.instruments.min_lots = int(_min_lots)
            except Exception:
                pass
        _max_lots = os.getenv("MAX_LOTS")
        if _max_lots:
            try:
                self.instruments.max_lots = int(_max_lots)
            except Exception:
                pass
        # strike range can arrive either here or as STRIKE_SELECTION_RANGE (strategy)
        _sr = os.getenv("STRIKE_RANGE") or os.getenv("STRIKE_SELECTION_RANGE")
        if _sr:
            try:
                # reflect to both to keep older helpers happy
                v = int(_sr)
                self.instruments.strike_range = v
                self.strategy.strike_selection_range = v
            except Exception:
                pass

        # --- Strategy (flat) ---
        _mss = os.getenv("MIN_SIGNAL_SCORE")
        if _mss:
            try:
                self.strategy.min_signal_score = int(_mss)
            except Exception:
                pass
        _conf = os.getenv("CONFIDENCE_THRESHOLD")
        if _conf:
            try:
                self.strategy.confidence_threshold = float(_conf)
            except Exception:
                pass
        _atrp = os.getenv("ATR_PERIOD")
        if _atrp:
            try:
                self.strategy.atr_period = int(_atrp)
            except Exception:
                pass
        _slm = os.getenv("ATR_SL_MULTIPLIER")
        if _slm:
            try:
                self.strategy.atr_sl_multiplier = float(_slm)
            except Exception:
                pass
        _tpm = os.getenv("ATR_TP_MULTIPLIER")
        if _tpm:
            try:
                self.strategy.atr_tp_multiplier = float(_tpm)
            except Exception:
                pass
        _slc = os.getenv("SL_CONFIDENCE_ADJ")
        if _slc:
            try:
                self.strategy.sl_confidence_adj = float(_slc)
            except Exception:
                pass
        _tpc = os.getenv("TP_CONFIDENCE_ADJ")
        if _tpc:
            try:
                self.strategy.tp_confidence_adj = float(_tpc)
            except Exception:
                pass

        # --- Risk (flat uses *_PCT for percentage) ---
        _rpt = os.getenv("RISK_PER_TRADE_PCT")
        if _rpt:
            try:
                self.risk.risk_per_trade = float(_rpt)
            except Exception:
                pass
        _mtd = os.getenv("MAX_TRADES_PER_DAY")
        if _mtd:
            try:
                self.risk.max_trades_per_day = int(_mtd)
            except Exception:
                pass
        _cl = os.getenv("CONSECUTIVE_LOSS_LIMIT")
        if _cl:
            try:
                self.risk.consecutive_loss_limit = int(_cl)
            except Exception:
                pass
        _dd = os.getenv("MAX_DAILY_DRAWDOWN_PCT")
        if _dd:
            try:
                self.risk.max_daily_drawdown_pct = float(_dd)
            except Exception:
                pass

        # --- Execution / exchange ---
        _ex = os.getenv("TRADE_EXCHANGE")
        if _ex:
            self.executor.exchange = _ex.upper()
        _tick = os.getenv("TICK_SIZE")
        if _tick:
            try:
                self.executor.tick_size = float(_tick)
            except Exception:
                pass
        _fz = os.getenv("NFO_FREEZE_QTY")
        if _fz:
            try:
                self.executor.exchange_freeze_qty = int(_fz)
            except Exception:
                pass

        return self

    # mirrors
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

    # legacy `.api`
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