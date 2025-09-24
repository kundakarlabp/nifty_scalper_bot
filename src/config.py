"""Config with nested models (original structure) + flat aliases for compatibility.

- Preserves your original names/shape so other modules don't break.
- Adds a few extra validations and guardrails.
- Keeps Telegram mandatory; Zerodha creds are only required if live trading is enabled.
- Adds optional historical backfill knobs under DataSettings.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    SupportsFloat,
    Tuple,
    cast,
)

from dotenv import load_dotenv
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_EXPOSURE_CAP_PCT = 40.0
def _coerce_pct_env(name: str, default_pct: float) -> float:
    """Return an environment percentage converted to a fraction."""

    raw = os.getenv(name)
    if raw is None:
        value = float(default_pct)
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logging.getLogger(__name__).warning(
                "Invalid percentage for %s: %s", name, raw
            )
            value = float(default_pct)
    if value > 1.0:
        value = value / 100.0
    return float(value)


# --- Environment toggles (flat reads for legacy modules) ---
ALLOW_NO_DEPTH: bool = os.getenv("ALLOW_NO_DEPTH", "false").lower() == "true"
TICK_STALE_SECONDS: float = float(os.getenv("TICK_STALE_SECONDS", 5.0))
DEPTH_MIN_QTY: int = int(os.getenv("DEPTH_MIN_QTY", 200))
SPREAD_MAX_PCT: float = float(os.getenv("SPREAD_MAX_PCT", 0.35))

# Quote priming controls
QUOTES__MODE: str = os.getenv("QUOTES__MODE", "FULL").upper()
QUOTES__PRIME_TIMEOUT_MS: int = int(os.getenv("QUOTES__PRIME_TIMEOUT_MS", "2000"))
QUOTES__RETRY_ATTEMPTS: int = int(os.getenv("QUOTES__RETRY_ATTEMPTS", "3"))
QUOTES__RETRY_JITTER_MS: int = int(os.getenv("QUOTES__RETRY_JITTER_MS", "150"))

# Microstructure requirements
MICRO__REQUIRE_DEPTH: bool = (
    os.getenv("MICRO__REQUIRE_DEPTH", "false").lower() == "true"
)
MICRO__DEPTH_MULTIPLIER: float = float(os.getenv("MICRO__DEPTH_MULTIPLIER", "5.0"))
MICRO__STALE_MS: int = int(os.getenv("MICRO__STALE_MS", "1500"))

# Single source for risk exposure cap (decimal fraction)
RISK__EXPOSURE_CAP_PCT: float = _coerce_pct_env(
    "RISK__EXPOSURE_CAP_PCT", DEFAULT_EXPOSURE_CAP_PCT
)

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "logfmt").lower()
STRUCTURED_LOGS: bool = os.getenv("STRUCTURED_LOGS", "true").lower() == "true"
LOG_SAMPLE_RATE: float = float(os.getenv("LOG_SAMPLE_RATE", 0.05))
LOG_MIN_INTERVAL_SEC: float = float(os.getenv("LOG_MIN_INTERVAL_SEC", 3.0))
LOG_DIAG_DEFAULT_SEC: int = int(os.getenv("LOG_DIAG_DEFAULT_SEC", 60))
LOG_TRACE_DEFAULT_SEC: int = int(os.getenv("LOG_TRACE_DEFAULT_SEC", 30))
LOG_MAX_LINES_REPLY: int = int(os.getenv("LOG_MAX_LINES_REPLY", 30))

EXPOSURE_BASIS: str = os.getenv("EXPOSURE_BASIS", "premium")
RISK_USE_LIVE_EQUITY: bool = os.getenv("RISK_USE_LIVE_EQUITY", "true").lower() == "true"
RISK_DEFAULT_EQUITY: int = int(os.getenv("RISK_DEFAULT_EQUITY", 40000))


def get_equity_for_cap(equity_live: float | None) -> float:
    """Return the equity reference used when applying exposure caps."""

    if RISK_USE_LIVE_EQUITY and equity_live and equity_live > 0:
        return equity_live
    return float(RISK_DEFAULT_EQUITY)


def env_any(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment variable from ``names``."""
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return default


if TYPE_CHECKING:
    from src.server.logging_utils import LogGate


def build_log_gate() -> "LogGate":
    """Return a ``LogGate`` configured with the global sampling knobs."""

    from src.server.logging_utils import LogGate

    return LogGate(LOG_MIN_INTERVAL_SEC, LOG_SAMPLE_RATE)
# ================= Sub-models =================


class ZerodhaSettings(BaseModel):
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""

    @classmethod
    def from_env(cls) -> "ZerodhaSettings":
        """Load legacy flat env vars if present."""
        return cls(
            api_key=(
                env_any(
                    "ZERODHA__API_KEY",
                    "ZERODHA_API_KEY",
                    "KITE_API_KEY",
                )
                or ""
            ),
            api_secret=(
                env_any(
                    "ZERODHA__API_SECRET",
                    "ZERODHA_API_SECRET",
                    "KITE_API_SECRET",
                )
                or ""
            ),
            access_token=(
                env_any(
                    "ZERODHA__ACCESS_TOKEN",
                    "ZERODHA_ACCESS_TOKEN",
                    "KITE_ACCESS_TOKEN",
                )
                or ""
            ),
        )


class TelegramSettings(BaseModel):
    # Telegram is COMPULSORY in your deployment
    enabled: bool = True
    bot_token: str = ""
    chat_id: int = 0  # store as int to match controller usage

    @classmethod
    def from_env(cls) -> "TelegramSettings":
        """Support legacy flat env vars such as ``TELEGRAM_CHAT_ID``.

        If no bot token or chat ID is provided, Telegram alerts are disabled
        unless ``TELEGRAM_ENABLED`` is explicitly set to ``true``.
        """
        raw_chat = env_any("TELEGRAM__CHAT_ID", "TELEGRAM_CHAT_ID")
        token = env_any("TELEGRAM__BOT_TOKEN", "TELEGRAM_BOT_TOKEN", default="")
        try:
            chat = int(raw_chat) if raw_chat is not None else 0
        except ValueError:
            chat = 0
        enabled_env = env_any("TELEGRAM__ENABLED", "TELEGRAM_ENABLED")
        enabled = (
            str(enabled_env).lower() not in {"0", "false"}
            if enabled_env is not None
            else bool(token and chat)
        )
        if not token or chat == 0:
            if enabled_env is None:
                enabled = False
        return cls(
            enabled=enabled,
            bot_token=str(token or ""),
            chat_id=chat,
        )

    @field_validator("chat_id")
    @classmethod
    def _v_chat_id(cls, v: int, info: ValidationInfo) -> int:
        """Ensure chat_id is provided when Telegram is enabled."""
        if info.data.get("enabled", True) and v == 0:
            raise ValueError(
                "TELEGRAM__CHAT_ID must be a non-zero integer when TELEGRAM__ENABLED is true"
            )
        return v


class DataSettings(BaseModel):
    # Live loop consumption
    lookback_minutes: int = 15
    lookback_padding_bars: int = 5
    timeframe: str = "minute"  # 'minute' recommended
    time_filter_start: str = "09:20"
    time_filter_end: str = "15:25"

    # Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 60

    # Historical backfill (optional; runner/feeds can ignore if unsupported)
    history_days: int = 3  # 0 = off; otherwise backfill N days before now
    history_max_candles: int = 0  # 0 = unlimited within broker constraints
    hist_warn_ratelimit_seconds: int = Field(
        300,
        description=(
            "Seconds to suppress repeated historical data warnings per token and"
            " globally after authentication failures."
        ),
        validation_alias=AliasChoices(
            "HIST_WARN_RATELIMIT_S",
            "DATA__HIST_WARN_RATELIMIT_S",
            "DATA__HIST_WARN_RATELIMIT_SECONDS",
        ),
    )

    @field_validator("time_filter_start", "time_filter_end")
    @classmethod
    def _v_time(cls, v: str) -> str:
        from datetime import datetime

        datetime.strptime(v, "%H:%M")
        return v

    @field_validator("timeframe")
    @classmethod
    def _v_tf(cls, v: str) -> str:
        v = (v or "").lower()
        allowed = {"minute", "3minute", "5minute", "10minute", "15minute", "day"}
        # Keep loose—brokers vary; still nudge common values
        if v not in allowed:
            logging.getLogger("config").warning(
                "Unsupported timeframe '%s'; defaulting to 'minute'", v
            )
            return "minute"
        return v

    @field_validator(
        "lookback_minutes",
        "lookback_padding_bars",
        "cache_ttl_seconds",
        "history_days",
        "history_max_candles",
        "hist_warn_ratelimit_seconds",
    )
    @classmethod
    def _v_nonneg(cls, v: int) -> int:
        if v < 0:
            raise ValueError("numeric fields must be >= 0")
        return v


class InstrumentConfig(BaseModel):
    """Configuration for a tradable underlying or derivative complex."""

    symbol: str | None = None
    spot_symbol: str = "NSE:NIFTY 50"
    trade_symbol: str = "NIFTY"
    trade_exchange: str = "NFO"
    instrument_token: int = 256265
    spot_token: int = 256265
    nifty_lot_size: int = 50
    strike_range: int = 0
    min_lots: int = 1
    max_lots: int = 10

    @field_validator("instrument_token", mode="before")
    @classmethod
    def _v_token(cls, v: object) -> int:
        """Coerce instrument token strings or floats to int and ensure it is a positive integer."""
        try:
            token = int(float(str(v)))
        except (TypeError, ValueError) as e:
            raise ValueError("instrument_token must be numeric") from e
        if token <= 0:
            raise ValueError("instrument_token must be a positive integer")
        return token

    @field_validator("min_lots", "max_lots", "nifty_lot_size")
    @classmethod
    def _v_lots_pos(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Lot sizes must be positive")
        return v

    @field_validator("max_lots")
    @classmethod
    def _v_lots_order(cls, v: int, info: ValidationInfo) -> int:
        if "min_lots" in info.data and v < info.data["min_lots"]:
            raise ValueError("max_lots must be >= min_lots")
        return v

    @property
    def lot_size(self) -> int:
        """Return the canonical lot size alias used across the codebase."""

        return self.nifty_lot_size

    def key(self) -> str:
        """Return a normalized key representing the instrument."""

        candidate = self.symbol or self.trade_symbol or self.spot_symbol
        return str(candidate).upper()


class InstrumentsSettings(BaseModel):
    spot_symbol: str = "NSE:NIFTY 50"
    trade_symbol: str = "NIFTY"
    trade_exchange: str = "NFO"
    instrument_token: int = 256265  # primary token (spot preferred for OHLC)
    spot_token: int = (
        256265  # optional explicit spot token (helps with logs/diagnostics)
    )
    nifty_lot_size: int = 50
    strike_range: int = 0
    min_lots: int = 1
    max_lots: int = 10
    additional: Dict[str, InstrumentConfig] = Field(
        default_factory=dict,
        description="Optional portfolio of additional tradable instruments keyed by alias.",
    )

    @field_validator("instrument_token", mode="before")
    @classmethod
    def _v_token(cls, v: object) -> int:
        try:
            token = int(float(str(v)))
        except (TypeError, ValueError) as e:
            raise ValueError("instrument_token must be numeric") from e
        if token <= 0:
            raise ValueError("instrument_token must be a positive integer")
        return token

    @field_validator("min_lots", "max_lots", "nifty_lot_size")
    @classmethod
    def _v_lots_pos(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Lot sizes must be positive")
        return v

    @field_validator("max_lots")
    @classmethod
    def _v_lots_order(cls, v: int, info: ValidationInfo) -> int:
        if "min_lots" in info.data and v < info.data["min_lots"]:
            raise ValueError("max_lots must be >= min_lots")
        return v

    def _primary(self) -> InstrumentConfig:
        """Return the primary instrument configuration."""

        return InstrumentConfig(
            symbol=self.trade_symbol,
            spot_symbol=self.spot_symbol,
            trade_symbol=self.trade_symbol,
            trade_exchange=self.trade_exchange,
            instrument_token=self.instrument_token,
            spot_token=self.spot_token,
            nifty_lot_size=self.nifty_lot_size,
            strike_range=self.strike_range,
            min_lots=self.min_lots,
            max_lots=self.max_lots,
        )

    def portfolio(self) -> Dict[str, InstrumentConfig]:
        """Return a dictionary of all configured instruments keyed by normalized symbol."""

        items: Dict[str, InstrumentConfig] = {}
        primary = self._primary()
        items[primary.key()] = primary
        for alias, inst in self.additional.items():
            key = inst.key() if isinstance(inst, InstrumentConfig) else str(alias).upper()
            if not isinstance(inst, InstrumentConfig):
                # Pydantic already validated, but guard for defensive callers.
                continue
            items[key] = inst
            # Ensure alternative aliases map to the same object for lookup convenience.
            alt_keys = {
                str(alias).upper(),
                str(inst.trade_symbol).upper(),
                str(inst.spot_symbol).upper(),
            }
            for alt in alt_keys:
                items.setdefault(alt, inst)
        return items

    def instrument(self, symbol: str | None = None) -> InstrumentConfig:
        """Return configuration for ``symbol`` or fall back to the primary instrument."""

        if symbol is None:
            return self._primary()
        lookup = symbol.upper()
        portfolio = self.portfolio()
        inst = portfolio.get(lookup)
        if inst is None:
            # Unknown symbols inherit the primary configuration to preserve
            # backwards compatibility with single-instrument deployments.
            return self._primary()
        return inst


class StrategySettings(BaseModel):
    min_signal_score: int = 4
    confidence_threshold: float = 55.0  # 0..100
    min_bars_for_signal: int = 15
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    atr_sl_multiplier: float = 1.3
    atr_tp_multiplier: float = 2.2
    rr_min: float = 1.30
    rr_threshold: float | None = 1.5
    # Pick a tradable contract by default (prevents no_option_token on non-expiry days)
    option_expiry_mode: Literal["today", "nearest", "next"] = "nearest"
    @field_validator("confidence_threshold")
    @classmethod
    def _v_conf(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError("confidence_threshold must be 0..100")
        return v

    @field_validator("ema_slow")
    @classmethod
    def _v_emas(cls, v: int, info: ValidationInfo) -> int:
        if "ema_fast" in info.data and info.data["ema_fast"] >= v:
            raise ValueError("ema_fast must be < ema_slow")
        return v

    @field_validator("atr_tp_multiplier")
    @classmethod
    def _v_tp_gt_sl(cls, v: float, info: ValidationInfo) -> float:
        if "atr_sl_multiplier" in info.data and v <= info.data["atr_sl_multiplier"]:
            raise ValueError("ATR_TP_MULTIPLIER must be > ATR_SL_MULTIPLIER")
        return v

    @field_validator("rr_min")
    @classmethod
    def _v_rr_min(cls, v: float) -> float:
        if v <= 1.0:
            raise ValueError("rr_min must be > 1.0")
        return v


class RegimeSettings(BaseModel):
    """Thresholds controlling market regime classification."""

    # TREND regime thresholds (greater-than-or-equal comparisons)
    adx_trend_threshold: float = Field(
        default=18.0,
        description="Minimum ADX value before treating conditions as trending.",
        validation_alias=AliasChoices(
            "REGIME__ADX_TREND_THRESHOLD",
            "REGIME__ADX_TREND",
            "ADX_TREND_THRESHOLD",
        ),
    )
    di_delta_trend_threshold: float = Field(
        default=8.0,
        description="Minimum |DI+ - DI-| delta required to classify a trend.",
        validation_alias=AliasChoices(
            "REGIME__DI_DELTA_TREND_THRESHOLD",
            "REGIME__DI_DELTA_TREND",
            "DI_DELTA_TREND_THRESHOLD",
        ),
    )
    bb_width_trend_threshold: float = Field(
        default=3.0,
        description="Lower bound on Bollinger Band width (percent) for trend trades.",
        validation_alias=AliasChoices(
            "REGIME__BB_WIDTH_TREND_THRESHOLD",
            "REGIME__BB_WIDTH_TREND",
            "BB_WIDTH_TREND_THRESHOLD",
        ),
    )

    # RANGE regime guardrails (upper bounds / minimum dispersion)
    adx_range_threshold: float = Field(
        default=18.0,
        description="Upper bound on ADX before range setups are rejected.",
        validation_alias=AliasChoices(
            "REGIME__ADX_RANGE_THRESHOLD",
            "REGIME__ADX_RANGE",
            "ADX_RANGE_THRESHOLD",
        ),
    )
    di_delta_range_threshold: float = Field(
        default=6.0,
        description="Maximum |DI+ - DI-| delta tolerated when range trading.",
        validation_alias=AliasChoices(
            "REGIME__DI_DELTA_RANGE_THRESHOLD",
            "REGIME__DI_DELTA_RANGE",
            "DI_DELTA_RANGE_THRESHOLD",
        ),
    )
    bb_width_range_threshold: float = Field(
        default=2.0,
        description="Minimum Bollinger Band width (percent) for range conditions.",
        validation_alias=AliasChoices(
            "REGIME__BB_WIDTH_RANGE_THRESHOLD",
            "REGIME__BB_WIDTH_RANGE",
            "BB_WIDTH_RANGE_THRESHOLD",
        ),
    )

    @field_validator(
        "adx_trend_threshold",
        "di_delta_trend_threshold",
        "bb_width_trend_threshold",
        "adx_range_threshold",
        "di_delta_range_threshold",
        "bb_width_range_threshold",
        mode="before",
    )
    @classmethod
    def _v_non_negative(cls, value: float) -> float:
        val = float(value)
        if val < 0:
            raise ValueError("regime thresholds must be non-negative")
        return val


class RiskSettings(BaseModel):
    use_live_equity: bool = Field(
        True,
        validation_alias=AliasChoices(
            "RISK__USE_LIVE_EQUITY",
            "RISK_USE_LIVE_EQUITY",
            "USE_LIVE_EQUITY",
        ),
    )
    default_equity: float = Field(
        40000.0,
        validation_alias=AliasChoices(
            "RISK__DEFAULT_EQUITY",
            "RISK_DEFAULT_EQUITY",
            "DEFAULT_EQUITY",
        ),
    )
    min_equity_floor: float = Field(
        25000.0,
        validation_alias=AliasChoices(
            "RISK__MIN_EQUITY_FLOOR",
            "RISK_MIN_EQUITY_FLOOR",
            "MIN_EQUITY_FLOOR",
        ),
    )
    equity_refresh_seconds: int = 60

    risk_per_trade: float = 0.01
    max_trades_per_day: int = 12
    consecutive_loss_limit: int = 3
    max_daily_drawdown_pct: float = 0.04
    max_position_size_pct: float = 0.40
    trading_window_start: str = "09:15"
    trading_window_end: str = "15:30"
    loss_cooldown_minutes: int = Field(
        45,
        description=(
            "Base cool-down duration (minutes) applied after a loss streak. "
            "Set to 0 to disable the adaptive cool-down logic."
        ),
        validation_alias=AliasChoices(
            "loss_cooldown_minutes",
            "RISK__LOSS_COOLDOWN_MINUTES",
            "LOSS_COOLDOWN_MINUTES",
        ),
    )
    loss_cooldown_backoff: float = Field(
        1.6,
        description="Multiplier applied every time the cool-down triggers again in the same session.",
        validation_alias=AliasChoices(
            "loss_cooldown_backoff",
            "RISK__LOSS_COOLDOWN_BACKOFF",
            "LOSS_COOLDOWN_BACKOFF",
        ),
    )
    loss_cooldown_relax_multiplier: float = Field(
        0.6,
        description="Factor used to relax the cool-down severity after profitable trades.",
        validation_alias=AliasChoices(
            "loss_cooldown_relax_multiplier",
            "RISK__LOSS_COOLDOWN_RELAX_MULTIPLIER",
            "LOSS_COOLDOWN_RELAX_MULTIPLIER",
        ),
    )
    loss_cooldown_max_minutes: int = Field(
        240,
        description="Upper bound for the adaptive cool-down window in minutes.",
        validation_alias=AliasChoices(
            "loss_cooldown_max_minutes",
            "RISK__LOSS_COOLDOWN_MAX_MINUTES",
            "LOSS_COOLDOWN_MAX_MINUTES",
        ),
    )
    loss_cooldown_trigger_after_losses: int | None = Field(
        None,
        description=(
            "Optional override for the loss streak length that triggers the adaptive cool-down. "
            "Defaults to the consecutive loss limit when unset."
        ),
        validation_alias=AliasChoices(
            "loss_cooldown_trigger_after_losses",
            "RISK__LOSS_COOLDOWN_TRIGGER_AFTER_LOSSES",
            "LOSS_COOLDOWN_TRIGGER_AFTER_LOSSES",
        ),
    )
    loss_cooldown_drawdown_pct: float = Field(
        0.5,
        description=(
            "Fraction of the daily loss cap after which the cool-down is immediately triggered, "
            "even if the streak threshold is not hit."
        ),
        validation_alias=AliasChoices(
            "loss_cooldown_drawdown_pct",
            "RISK__LOSS_COOLDOWN_DRAWDOWN_PCT",
            "LOSS_COOLDOWN_DRAWDOWN_PCT",
        ),
    )
    loss_cooldown_drawdown_scale: float = Field(
        1.5,
        description="How aggressively to extend the cool-down when the drawdown threshold is exceeded.",
        validation_alias=AliasChoices(
            "loss_cooldown_drawdown_scale",
            "RISK__LOSS_COOLDOWN_DRAWDOWN_SCALE",
            "LOSS_COOLDOWN_DRAWDOWN_SCALE",
        ),
    )
    # Session guard: disallow new entries at or after this HH:MM (IST).
    no_new_after_hhmm: str | None = Field(
        None,
        description=(
            "Latest IST timestamp (HH:MM) to initiate new positions before "
            "risk gates halt entries."
        ),
        validation_alias=AliasChoices(
            "RISK__NO_NEW_AFTER_HHMM",
            "RISK_NO_NEW_AFTER_HHMM",
            "NO_NEW_AFTER_HHMM",
        ),
    )
    # Hard EOD flatten: force-close any open positions at or after this HH:MM (IST).
    eod_flatten_hhmm: str = Field(
        "15:28",
        description=(
            "IST timestamp (HH:MM) after which all open positions are "
            "flattened and orders cancelled."
        ),
        validation_alias=AliasChoices(
            "RISK__EOD_FLATTEN_HHMM",
            "RISK_EOD_FLATTEN_HHMM",
            "EOD_FLATTEN_HHMM",
        ),
    )
    max_daily_loss_rupees: float | None = None
    max_lots_per_symbol: int = 5
    max_notional_rupees: float = Field(
        default_factory=lambda: float(
            env_any("EXPOSURE_CAP", "RISK__MAX_NOTIONAL_RUPEES") or 1_500_000.0
        )
    )
    exposure_basis: Literal["underlying", "premium"] = "premium"
    exposure_cap_source: Literal["equity", "absolute"] = Field(
        default_factory=lambda: str(
            os.getenv("RISK__EXPOSURE_CAP_SOURCE", "equity")
        ).lower(),
        validation_alias=AliasChoices("RISK__EXPOSURE_CAP_SOURCE"),
    )
    exposure_cap_pct_of_equity: float = Field(
        default_factory=lambda: _coerce_pct_env(
            "RISK__EXPOSURE_CAP_PCT", DEFAULT_EXPOSURE_CAP_PCT
        ),
        validation_alias=AliasChoices("RISK__EXPOSURE_CAP_PCT"),
    )
    exposure_cap_abs: float = Field(
        default_factory=lambda: float(os.getenv("RISK__EXPOSURE_CAP_ABS", "0")),
        validation_alias=AliasChoices("RISK__EXPOSURE_CAP_ABS"),
    )
    premium_cap_per_trade: float = 10000.0
    allow_min_one_lot: bool = Field(
        default_factory=lambda: str(
            env_any(
                "RISK__ALLOW_MIN_ONE_LOT",
                "RISK_ALLOW_MIN_ONE_LOT",
                "ALLOW_MIN_ONE_LOT",
                default="true",
            )
        ).lower()
        == "true",
        description=(
            "Permit one-lot trades when the equity-funded premium exceeds the "
            "exposure cap by itself. Enabled by default."
        ),
        validation_alias=AliasChoices(
            "RISK__ALLOW_MIN_ONE_LOT",
            "RISK_ALLOW_MIN_ONE_LOT",
            "ALLOW_MIN_ONE_LOT",
        ),
    )

    @field_validator("exposure_basis", mode="before")
    @classmethod
    def _v_exposure_basis(cls, v: object) -> str:
        val = str(v).lower()
        if val not in {"premium", "underlying"}:
            raise ValueError("EXPOSURE_BASIS must be 'premium' or 'underlying'")
        return val

    @field_validator("exposure_cap_source", mode="before")
    @classmethod
    def _v_exposure_cap_source(cls, v: object) -> str:
        val = str(v).lower()
        if val == "env":
            # Historical name mapped to the new 'absolute' option.
            return "absolute"
        if val not in {"equity", "absolute"}:
            raise ValueError("EXPOSURE_CAP_SOURCE must be 'equity' or 'absolute'")
        return val

    @field_validator("exposure_cap_pct_of_equity")
    @classmethod
    def _v_exposure_cap_pct(cls, v: float) -> float:
        pct = float(v)
        if pct > 1.0:
            pct = pct / 100.0
        if not 0.0 < pct <= 1.0:
            raise ValueError(
                "EXPOSURE_CAP_PCT_OF_EQUITY must be within (0, 1]"
            )
        return pct

    @field_validator("exposure_cap_abs")
    @classmethod
    def _v_cap_abs(cls, v: float) -> float:
        val = float(v)
        if val < 0:
            raise ValueError("EXPOSURE_CAP_ABS must be >= 0")
        return val

    @field_validator("risk_per_trade")
    @classmethod
    def _v_risk_pct(cls, v: float) -> float:
        if not 0.0 < v <= 0.50:
            raise ValueError("risk_per_trade must be within (0, 0.50]")
        return v

    @field_validator("max_daily_drawdown_pct")
    @classmethod
    def _v_dd_pct(cls, v: float) -> float:
        if not 0.01 <= v <= 0.20:
            raise ValueError("max_daily_drawdown_pct must be between 1% and 20%")
        return v

    @field_validator("max_position_size_pct")
    @classmethod
    def _v_pos_size_pct(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("max_position_size_pct must be within (0, 1]")
        return v

    @field_validator("loss_cooldown_minutes", "loss_cooldown_max_minutes")
    @classmethod
    def _v_cooldown_minutes(cls, v: int) -> int:
        minutes = int(v)
        if minutes < 0:
            raise ValueError("Loss cool-down minutes must be >= 0")
        return minutes

    @field_validator("loss_cooldown_backoff")
    @classmethod
    def _v_cooldown_backoff(cls, v: float) -> float:
        val = float(v)
        if val < 1.0:
            raise ValueError("loss_cooldown_backoff must be >= 1.0")
        return val

    @field_validator("loss_cooldown_relax_multiplier")
    @classmethod
    def _v_cooldown_relax(cls, v: float) -> float:
        val = float(v)
        if not 0.0 <= val <= 1.0:
            raise ValueError("loss_cooldown_relax_multiplier must be within [0, 1]")
        return val

    @field_validator("loss_cooldown_trigger_after_losses")
    @classmethod
    def _v_cooldown_trigger(cls, v: int | None) -> int | None:
        if v is None:
            return None
        val = int(v)
        if val < 1:
            raise ValueError("loss_cooldown_trigger_after_losses must be >= 1 or None")
        return val

    @field_validator("loss_cooldown_drawdown_pct")
    @classmethod
    def _v_cooldown_drawdown_pct(cls, v: float) -> float:
        pct = float(v)
        if pct < 0.0:
            raise ValueError("loss_cooldown_drawdown_pct must be >= 0")
        if pct > 1.0:
            pct = pct / 100.0
        return min(1.0, pct)

    @field_validator("loss_cooldown_max_minutes")
    @classmethod
    def _v_cooldown_bounds(cls, v: int, info: ValidationInfo) -> int:
        minutes = int(v)
        base = info.data.get("loss_cooldown_minutes", 0)
        if base and minutes < base:
            raise ValueError("loss_cooldown_max_minutes must be >= loss_cooldown_minutes")
        return minutes

    @field_validator(
        "trading_window_start",
        "trading_window_end",
        "no_new_after_hhmm",
        "eod_flatten_hhmm",
    )
    @classmethod
    def _v_time(cls, v: str) -> str:
        datetime.strptime(v, "%H:%M")
        return v

    @field_validator(
        "max_daily_loss_rupees", "max_notional_rupees", "premium_cap_per_trade",
        mode="before",
    )
    @classmethod
    def _v_positive_float(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if float(v) <= 0:
            raise ValueError("value must be > 0")
        return float(v)

    @field_validator(
        "max_lots_per_symbol",
        "max_trades_per_day",
        "consecutive_loss_limit",
        "equity_refresh_seconds",
    )
    @classmethod
    def _v_positive_int(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError("value must be > 0")
        return int(v)

    @field_validator("min_equity_floor", "default_equity")
    @classmethod
    def _v_equity_pos(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("equity amounts must be > 0")
        return v


class MicroSettings(BaseModel):
    """Microstructure execution guard configuration."""

    spread_cap_pct: float = Field(
        0.35,
        description="Maximum acceptable bid/ask spread percentage for entries.",
        validation_alias=AliasChoices("MICRO_SPREAD_CAP"),
    )
    entry_wait_seconds: float = Field(
        8.0,
        description="Seconds to wait for microstructure to improve before aborting.",
        validation_alias=AliasChoices("ENTRY_WAIT_S"),
    )

    @field_validator("spread_cap_pct", "entry_wait_seconds", mode="before")
    @classmethod
    def _v_float(cls, v: Any) -> float:
        return float(v)

    @field_validator("spread_cap_pct")
    @classmethod
    def _v_nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("micro spread cap must be >= 0")
        return v

    @field_validator("entry_wait_seconds")
    @classmethod
    def _v_wait(cls, v: float) -> float:
        if v < 0:
            raise ValueError("entry wait seconds must be >= 0")
        return v

    @property
    def spread_cap_ratio(self) -> float:
        """Return spread cap as ratio (0.0035 for 0.35%)."""

        return float(self.spread_cap_pct) / 100.0


class ExecutorSettings(BaseModel):
    exchange: str = "NFO"
    order_product: str = "NRML"
    order_variety: str = (
        "regular"  # regular | bo | amo | co (depending on broker support)
    )
    entry_order_type: str = "LIMIT"  # LIMIT | MARKET | SL | SLM
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
    entry_slippage_pct: float = Field(
        0.25, validation_alias=AliasChoices("ENTRY_SLIPPAGE_PCT", "EXECUTOR__ENTRY_SLIPPAGE_PCT")
    )
    exit_slippage_pct: float = Field(
        0.25, validation_alias=AliasChoices("EXIT_SLIPPAGE_PCT", "EXECUTOR__EXIT_SLIPPAGE_PCT")
    )
    ack_timeout_ms: int = 1500
    fill_timeout_ms: int = 10000
    max_place_retries: int = 2
    # microstructure execution guards
    max_spread_pct: float = 0.0035  # 0.35%
    depth_multiplier: float = Field(
        5.0,
        description="Depth multiplier applied when checking market liquidity.",
        validation_alias=AliasChoices("EXECUTOR_DEPTH_MULTIPLIER"),
    )
    depth_min_lots: float = Field(
        3.0,
        description="Minimum number of lots that must be available in depth when depth checks are enabled.",
        validation_alias=AliasChoices(
            "EXECUTOR__DEPTH_MIN_LOTS", "EXECUTOR_DEPTH_MIN_LOTS"
        ),
    )
    micro_retry_limit: int = Field(
        3,
        description="Maximum refresh attempts when waiting for microstructure.",
        validation_alias=AliasChoices("EXECUTOR_MICRO_RETRY_LIMIT"),
    )
    require_depth: bool = False
    default_spread_pct_est: float = 0.25

    @field_validator("tp1_qty_ratio")
    @classmethod
    def _v_ratio(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("tp1_qty_ratio must be in 0..1")
        return v

    @field_validator(
        "breakeven_ticks",
        "slippage_ticks",
        "ack_timeout_ms",
        "fill_timeout_ms",
        "max_place_retries",
        "micro_retry_limit",
    )
    @classmethod
    def _v_nonneg_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("values must be >= 0")
        return v


class OptionSelectorSettings(BaseModel):
    """Strike selection caches and liquidity thresholds."""

    instruments_cache_ttl_seconds: float = Field(
        60.0,
        description="Seconds to retain the NFO instrument dump before re-fetching.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__INSTRUMENTS_CACHE_TTL_SECONDS",
            "STRIKE_SELECTOR__INSTRUMENTS_CACHE_TTL_SECONDS",
        ),
    )
    instruments_refresh_minutes: int = Field(
        15,
        description="Minutes to wait before refreshing the broker-supplied NFO instrument list.",
        validation_alias=AliasChoices(
            "INSTRUMENTS_REFRESH_MINUTES",
            "OPTION_SELECTOR__INSTRUMENTS_REFRESH_MINUTES",
            "STRIKE_SELECTOR__INSTRUMENTS_REFRESH_MINUTES",
        ),
    )
    ltp_cache_ttl_seconds: float = Field(
        2.0,
        description="Seconds to reuse cached spot LTP responses for strike selection.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__LTP_CACHE_TTL_SECONDS",
            "STRIKE_SELECTOR__LTP_CACHE_TTL_SECONDS",
        ),
    )
    rate_limit_interval_seconds: float = Field(
        0.25,
        description="Minimum spacing between broker API calls for strike helpers.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__RATE_LIMIT_INTERVAL_SECONDS",
            "STRIKE_SELECTOR__RATE_LIMIT_INTERVAL_SECONDS",
        ),
    )
    rate_limit_sleep_seconds: float = Field(
        0.05,
        description="Sleep duration when throttled by the strike helper rate limiter.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__RATE_LIMIT_SLEEP_SECONDS",
            "STRIKE_SELECTOR__RATE_LIMIT_SLEEP_SECONDS",
        ),
    )
    fallback_strike_step: int = Field(
        50,
        description="Default strike step when instruments configuration omits one.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__FALLBACK_STRIKE_STEP",
            "STRIKE_SELECTOR__FALLBACK_STRIKE_STEP",
        ),
    )
    weekly_expiry_weekday: int = Field(
        2,
        description="ISO weekday (1=Monday..7=Sunday) used to pick the weekly option expiry.",
        validation_alias=AliasChoices(
            "WEEKLY_EXPIRY_WEEKDAY",
            "OPTION_SELECTOR__WEEKLY_EXPIRY_WEEKDAY",
            "STRIKE_SELECTOR__WEEKLY_EXPIRY_WEEKDAY",
        ),
    )
    prefer_monthly_expiry: bool = Field(
        False,
        description="Prefer the monthly expiry when it falls on the configured weekly expiry weekday.",
        validation_alias=AliasChoices(
            "PREFER_MONTHLY_EXPIRY",
            "OPTION_SELECTOR__PREFER_MONTHLY_EXPIRY",
            "STRIKE_SELECTOR__PREFER_MONTHLY_EXPIRY",
        ),
    )
    banknifty_strike_step: int = Field(
        100,
        description="Strike step applied when the trade symbol contains BANKNIFTY.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__BANKNIFTY_STRIKE_STEP",
            "STRIKE_SELECTOR__BANKNIFTY_STRIKE_STEP",
        ),
    )
    allow_pm1_score_threshold: int = Field(
        9,
        description="Minimum score required before probing strikes one step from ATM.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__ALLOW_PM1_SCORE_THRESHOLD",
            "STRIKE_SELECTOR__ALLOW_PM1_SCORE_THRESHOLD",
        ),
    )
    min_open_interest: int = Field(
        500_000,
        description="Lowest acceptable open interest when selecting strikes.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__MIN_OPEN_INTEREST",
            "STRIKE_SELECTOR__MIN_OPEN_INTEREST",
        ),
    )
    max_spread_pct: float = Field(
        0.35,
        description="Maximum bid/ask spread percentage tolerated for strikes.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__MAX_SPREAD_PCT",
            "STRIKE_SELECTOR__MAX_SPREAD_PCT",
        ),
    )
    delta_iv_guess: float = Field(
        0.20,
        description="Initial implied volatility guess for delta-based strike picking.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__DELTA_IV_GUESS",
            "STRIKE_SELECTOR__DELTA_IV_GUESS",
        ),
    )
    delta_min_option_price: float = Field(
        1.0,
        description="Floor option price used when estimating IV for thin markets.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__DELTA_MIN_OPTION_PRICE",
            "STRIKE_SELECTOR__DELTA_MIN_OPTION_PRICE",
        ),
    )
    delta_option_price_pct_of_spot: float = Field(
        0.005,
        description="Fraction of spot used to estimate option price for IV guesses.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__DELTA_OPTION_PRICE_PCT_OF_SPOT",
            "STRIKE_SELECTOR__DELTA_OPTION_PRICE_PCT_OF_SPOT",
        ),
    )
    delta_min_time_to_expiry_years: float = Field(
        1.0 / 365.0,
        description="Minimum time to expiry in years when computing delta heuristics.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__DELTA_MIN_TIME_TO_EXPIRY_YEARS",
            "STRIKE_SELECTOR__DELTA_MIN_TIME_TO_EXPIRY_YEARS",
            "OPTION_SELECTOR__DELTA_MIN_TIME_TO_EXPIRY_DAYS",
            "STRIKE_SELECTOR__DELTA_MIN_TIME_TO_EXPIRY_DAYS",
        ),
    )
    needs_reatm_pct: float = Field(
        0.35,
        description="Percent drift in spot that should trigger ATM reselection.",
        validation_alias=AliasChoices(
            "OPTION_SELECTOR__NEEDS_REATM_PCT",
            "STRIKE_SELECTOR__NEEDS_REATM_PCT",
        ),
    )

    @field_validator(
        "instruments_cache_ttl_seconds",
        "ltp_cache_ttl_seconds",
        "rate_limit_interval_seconds",
        mode="before",
    )
    @classmethod
    def _v_nonnegative_float(cls, v: float) -> float:
        val = float(v)
        if val < 0:
            raise ValueError("cache and interval values must be >= 0")
        return val

    @field_validator(
        "rate_limit_sleep_seconds",
        "max_spread_pct",
        "delta_iv_guess",
        "delta_min_option_price",
        "delta_option_price_pct_of_spot",
        "delta_min_time_to_expiry_years",
        "needs_reatm_pct",
        mode="before",
    )
    @classmethod
    def _v_positive_float(cls, v: float) -> float:
        val = float(v)
        if val <= 0:
            raise ValueError("values must be > 0")
        return val

    @field_validator(
        "instruments_refresh_minutes",
        "fallback_strike_step",
        "banknifty_strike_step",
        "allow_pm1_score_threshold",
        "min_open_interest",
    )
    @classmethod
    def _v_nonnegative_int(cls, v: int) -> int:
        val = int(v)
        if val < 0:
            raise ValueError("integer settings must be >= 0")
        return val

    @field_validator("weekly_expiry_weekday")
    @classmethod
    def _v_weekday(cls, v: int) -> int:
        day = int(v)
        if not 1 <= day <= 7:
            raise ValueError("weekly_expiry_weekday must be between 1 and 7 (ISO weekday)")
        return day

class HealthSettings(BaseModel):
    enable_server: bool = True
    host: str = Field(
        "0.0.0.0",
        validation_alias=AliasChoices("HEALTH_HOST"),
        description="Bind address for the optional health server.",
    )
    port: int = Field(
        8000,
        validation_alias=AliasChoices("HEALTH_PORT"),
        description="Port used by the optional health server.",
    )


class SystemSettings(BaseModel):
    max_api_calls_per_second: float = 8.0
    websocket_reconnect_attempts: int = 5
    order_timeout_seconds: int = 30
    position_sync_interval: int = 60
    log_buffer_capacity: int = 4000


# ================= Root settings =================


def _micro_settings_factory() -> MicroSettings:
    """Return default ``MicroSettings`` instance with validated defaults."""

    return MicroSettings.model_validate({})


def _risk_settings_factory() -> RiskSettings:
    return RiskSettings()  # type: ignore[call-arg]


class AppSettings(BaseSettings):
    # General optional settings
    app_env: str = "production"
    ai_provider: str = ""
    openai_api_key: str = ""
    portfolio_reads: bool = Field(
        True, validation_alias=AliasChoices("PORTFOLIO_READS")
    )

    # Live trading runs by default; set to ``false`` for paper trading.
    enable_live_trading: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_LIVE_TRADING", "ENABLE_TRADING"),
    )
    allow_offhours_testing: bool = False
    enable_time_windows: bool = True
    tz: str = Field(
        "Asia/Kolkata", validation_alias=AliasChoices("LOG_TZ", "TZ")
    )
    log_level: str = Field("INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    log_format: Literal["logfmt", "json"] = Field(
        "logfmt",
        validation_alias=AliasChoices("LOG_FORMAT"),
        description="Output format for structured logs.",
    )
    log_json: bool = Field(False, validation_alias=AliasChoices("LOG_JSON"))
    log_path: Path | None = Field(
        None,
        validation_alias=AliasChoices("LOG_PATH", "LOG_FILE"),
        description="Optional log file path for structured logging output.",
    )
    log_ring_enabled: bool = Field(
        True,
        validation_alias=AliasChoices("LOG_RING_ENABLED"),
        description="Toggle the in-memory log ring buffer used for /logs.",
    )
    log_suppress_window_sec: float = Field(
        300.0,
        validation_alias=AliasChoices("LOG_SUPPRESS_WINDOW_SEC"),
        description=(
            "Seconds to suppress repeated warning/error logs per (kind, group, ident)."
        ),
    )
    telegram_periodic_logs: bool = Field(
        False,
        validation_alias=AliasChoices("TELEGRAM__PERIODIC_LOGS"),
        description="Enable periodic Telegram debug log snapshots (every 5 minutes).",
    )
    diag_ring_size: int = Field(
        4000,
        validation_alias=AliasChoices("DIAG_RING_SIZE"),
        description="Capacity of the diagnostic log ring buffer.",
    )
    diag_trace_events: bool = Field(
        False,
        validation_alias=AliasChoices("DIAG_TRACE_EVENTS"),
        description="Enable verbose diagnostic event tracing.",
    )
    heartbeat_interval_sec: float = Field(
        30.0,
        validation_alias=AliasChoices("HEARTBEAT_INTERVAL_SEC"),
        description="Seconds between heartbeat log lines when unchanged.",
    )
    plan_log_interval_sec: float = Field(
        60.0,
        validation_alias=AliasChoices("PLAN_LOG_INTERVAL_SEC"),
        description="Seconds between plan debug log lines when unchanged.",
    )
    block_summary_interval_sec: float = Field(
        30.0,
        validation_alias=AliasChoices("BLOCK_SUMMARY_INTERVAL_SEC"),
        description="Seconds between repeated block reason summaries.",
    )
    decision_interval_sec: float = Field(
        10.0,
        validation_alias=AliasChoices("DECISION_INTERVAL_SEC", "LOG_DECISION_THROTTLE_S"),
        description="Seconds between repeated decision logs when unchanged.",
    )
    quote_warmup_tries: int = Field(
        1,
        validation_alias=AliasChoices("QUOTE_WARMUP_TRIES"),
        description="Number of attempts to warm broker quotes before blocking.",
    )
    quote_warmup_sleep_ms: int = Field(
        0,
        validation_alias=AliasChoices("QUOTE_WARMUP_SLEEP_MS"),
        description="Delay between quote warmup attempts in milliseconds.",
    )
    # Fallback notional equity (₹) when live equity fetch fails or is disabled.
    RISK_DEFAULT_EQUITY: int = Field(
        default_factory=lambda: int(os.getenv("RISK_DEFAULT_EQUITY", "40000")),
        validation_alias=AliasChoices(
            "RISK_DEFAULT_EQUITY",
            "RISK__DEFAULT_EQUITY",
            "DEFAULT_EQUITY",
        ),
    )
    # Whether to pull live equity from the broker for risk calculations.
    RISK_USE_LIVE_EQUITY: bool = Field(
        default_factory=lambda: str(os.getenv("RISK_USE_LIVE_EQUITY", "true")).lower()
        in {"true", "1", "yes", "on"},
        validation_alias=AliasChoices(
            "RISK_USE_LIVE_EQUITY",
            "RISK__USE_LIVE_EQUITY",
            "USE_LIVE_EQUITY",
        ),
    )
    # Exposure calculations based on option premium or underlying notional.
    EXPOSURE_BASIS: Literal["premium", "underlying"] = Field(
        default_factory=lambda: str(
            os.getenv("RISK__EXPOSURE_BASIS", os.getenv("EXPOSURE_BASIS", "premium"))
        ).lower(),
        validation_alias=AliasChoices("RISK__EXPOSURE_BASIS", "EXPOSURE_BASIS"),
    )
    tp_basis: Literal["premium", "spot"] = "premium"
    # Source used to cap exposure (percentage of equity or absolute rupees).
    # Legacy value 'env' is treated as 'absolute' downstream for compatibility.
    EXPOSURE_CAP_SOURCE: Literal["equity", "absolute", "env"] = Field(
        default_factory=lambda: str(os.getenv("RISK__EXPOSURE_CAP_SOURCE", "equity")).lower(),
        validation_alias=AliasChoices("RISK__EXPOSURE_CAP_SOURCE"),
    )
    # Percent of equity allowed per trade (converted to ratio for risk model).
    EXPOSURE_CAP_PCT: float = Field(
        default_factory=lambda: float(
            os.getenv("RISK__EXPOSURE_CAP_PCT", str(DEFAULT_EXPOSURE_CAP_PCT))
        ),
        validation_alias=AliasChoices("RISK__EXPOSURE_CAP_PCT"),
    )
    # Absolute rupee cap fallback when percentage-based cap is insufficient.
    EXPOSURE_CAP_ABS: float = Field(
        default_factory=lambda: float(os.getenv("RISK__EXPOSURE_CAP_ABS", "0")),
        validation_alias=AliasChoices("RISK__EXPOSURE_CAP_ABS"),
    )
    # Hard cap on collected premium per trade (₹).
    PREMIUM_CAP_PER_TRADE: float = 10000.0
    # Maximum tolerated lag for inbound ticks in seconds.
    TICK_MAX_LAG_S: float = Field(
        default_factory=lambda: float(os.getenv("TICK_MAX_LAG_S", "5")),
        validation_alias=AliasChoices("TICK_MAX_LAG_S", "DATA__TICK_MAX_LAG_S"),
    )
    # Maximum tolerated lag for completed bars/candles in seconds.
    BAR_MAX_LAG_S: float = Field(
        default_factory=lambda: float(os.getenv("BAR_MAX_LAG_S", "2")),
        validation_alias=AliasChoices("BAR_MAX_LAG_S", "DATA__BAR_MAX_LAG_S"),
    )
    # Runner cadence bounds (seconds) used to throttle evaluation frequency.
    cadence_min_interval_s: float = Field(
        0.3,
        validation_alias=AliasChoices(
            "CADENCE_MIN_INTERVAL_S",
            "RUNNER__CADENCE_MIN_INTERVAL_S",
        ),
    )
    # Longest pause between evaluations when the market is quiet or breaker is open.
    cadence_max_interval_s: float = Field(
        1.5,
        validation_alias=AliasChoices(
            "CADENCE_MAX_INTERVAL_S",
            "RUNNER__CADENCE_MAX_INTERVAL_S",
        ),
    )
    # Increment applied when ticks arrive slowly to gradually widen cadence.
    cadence_interval_step_s: float = Field(
        0.3,
        validation_alias=AliasChoices(
            "CADENCE_INTERVAL_STEP_S",
            "RUNNER__CADENCE_INTERVAL_STEP_S",
            "CADENCE_STEP_S",
        ),
    )

    @field_validator("RISK_DEFAULT_EQUITY", mode="before")
    @classmethod
    def _v_app_default_equity(cls, v: object) -> int:
        try:
            val = int(float(str(v)))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("RISK_DEFAULT_EQUITY must be numeric") from exc
        if val <= 0:
            raise ValueError("RISK_DEFAULT_EQUITY must be > 0")
        return val

    @field_validator("RISK_USE_LIVE_EQUITY", mode="before")
    @classmethod
    def _v_app_live_equity(cls, v: object) -> bool:
        if isinstance(v, bool):
            return v
        return str(v).lower() in {"1", "true", "yes", "on"}

    @field_validator("log_format", mode="before")
    @classmethod
    def _v_log_format(cls, v: object) -> str:
        if isinstance(v, str):
            val = v.strip().lower()
            if val in {"logfmt", "json"}:
                return val
        if isinstance(v, bool):
            return "json" if v else "logfmt"
        raise ValueError("LOG_FORMAT must be 'logfmt' or 'json'")

    @field_validator("quote_warmup_tries", mode="before")
    @classmethod
    def _v_quote_warmup_tries(cls, v: object) -> int:
        try:
            val = int(float(str(v)))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("QUOTE_WARMUP_TRIES must be >= 1") from exc
        if val < 1:
            raise ValueError("QUOTE_WARMUP_TRIES must be >= 1")
        return val

    @field_validator("quote_warmup_sleep_ms", mode="before")
    @classmethod
    def _v_quote_warmup_sleep_ms(cls, v: object) -> int:
        try:
            val = int(float(str(v)))
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("QUOTE_WARMUP_SLEEP_MS must be >= 0") from exc
        if val < 0:
            raise ValueError("QUOTE_WARMUP_SLEEP_MS must be >= 0")
        return val

    @field_validator("EXPOSURE_BASIS", mode="before")
    @classmethod
    def _v_app_exposure_basis(cls, v: object) -> str:
        val = str(v).lower()
        if val not in {"premium", "underlying"}:
            raise ValueError("EXPOSURE_BASIS must be 'premium' or 'underlying'")
        return val

    @field_validator("EXPOSURE_CAP_SOURCE", mode="before")
    @classmethod
    def _v_app_exposure_cap_source(cls, v: object) -> str:
        val = str(v).lower()
        if val not in {"equity", "absolute", "env"}:
            raise ValueError(
                "EXPOSURE_CAP_SOURCE must be 'equity', 'absolute', or legacy 'env'"
            )
        return val

    @field_validator("EXPOSURE_CAP_PCT")
    @classmethod
    def _v_app_exposure_cap_pct(cls, v: float) -> float:
        pct = float(v)
        if pct <= 0:
            raise ValueError("EXPOSURE_CAP_PCT must be > 0")
        if pct <= 1.0:
            pct *= 100.0
        if pct > 100.0:
            raise ValueError("EXPOSURE_CAP_PCT must be <= 100")
        return pct

    @field_validator("EXPOSURE_CAP_ABS")
    @classmethod
    def _v_app_cap_abs(cls, v: float) -> float:
        val = float(v)
        if val < 0:
            raise ValueError("EXPOSURE_CAP_ABS must be >= 0")
        return val

    @field_validator("TICK_MAX_LAG_S", "BAR_MAX_LAG_S")
    @classmethod
    def _v_lag_nonnegative(cls, v: float) -> float:
        val = float(v)
        if val < 0:
            raise ValueError("lag thresholds must be >= 0 seconds")
        return val

    @field_validator(
        "cadence_min_interval_s",
        "cadence_max_interval_s",
        "cadence_interval_step_s",
        mode="before",
    )
    @classmethod
    def _v_cadence_positive(cls, v: Any) -> float:
        try:
            val = float(v)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("cadence intervals must be numeric") from exc
        if val <= 0:
            raise ValueError("cadence intervals must be > 0")
        return val

    @field_validator(
        "heartbeat_interval_sec",
        "plan_log_interval_sec",
        "block_summary_interval_sec",
        "decision_interval_sec",
        "log_suppress_window_sec",
        mode="before",
    )
    @classmethod
    def _v_positive_interval(cls, v: object) -> float:
        try:
            if isinstance(v, (int, float, str)):
                val = float(v)
            elif hasattr(v, "__float__"):
                val = float(cast(SupportsFloat, v))
            else:
                raise TypeError("unsupported interval type")
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("interval values must be numeric seconds") from exc
        if val <= 0:
            raise ValueError("interval values must be > 0 seconds")
        return val

    @model_validator(mode="after")
    def _v_cadence_bounds(self) -> "AppSettings":
        if self.cadence_min_interval_s > self.cadence_max_interval_s:
            raise ValueError(
                "cadence_min_interval_s must be less than or equal to cadence_max_interval_s"
            )
        return self

    @model_validator(mode="after")
    def _sync_log_format(self) -> "AppSettings":
        fields_set: set[str] = getattr(self, "model_fields_set", set())
        fmt = str(getattr(self, "log_format", "logfmt"))
        fmt = fmt.strip().lower() if fmt else "logfmt"
        if fmt not in {"logfmt", "json"}:
            fmt = "logfmt"
        if "log_json" in fields_set and "log_format" not in fields_set:
            self.log_format = "json" if self.log_json else "logfmt"
        else:
            fmt_literal = cast(Literal["logfmt", "json"], fmt)
            self.log_format = fmt_literal
            self.log_json = fmt == "json"
        return self

    @property
    def EXPOSURE_CAP_PCT_OF_EQUITY(self) -> float:
        """Return the exposure cap as a ratio for legacy callers."""

        pct = float(self.EXPOSURE_CAP_PCT)
        return pct / 100.0

    @property
    def RISK__EXPOSURE_CAP_PCT(self) -> float:
        """Unified exposure cap ratio consumed by micro + risk gates."""

        return float(self.EXPOSURE_CAP_PCT_OF_EQUITY)

    instruments_csv: str = Field(
        default_factory=lambda: str(
            os.getenv("INSTRUMENTS__CSV") or os.getenv("INSTRUMENTS_CSV", "")
        ),
        validation_alias=AliasChoices("INSTRUMENTS_CSV", "INSTRUMENTS__CSV"),
    )
    cb_error_rate: float = 0.10
    cb_p95_ms: int = 1200
    cb_min_samples: int = 30
    cb_open_cooldown_sec: int = 30
    cb_half_open_probe: int = 3
    max_place_retries: int = 2
    max_modify_retries: int = 2
    retry_backoff_ms: int = 200
    diag_interval_seconds: int = 60
    min_preview_score: float = 8.0
    plan_stale_sec: int = 20

    @property
    def PORTFOLIO_READS(self) -> bool:  # pragma: no cover - simple alias
        return self.portfolio_reads

    @property
    def INSTRUMENTS_CSV(self) -> str:  # pragma: no cover - legacy alias
        return self.instruments_csv

    # Data warmup
    warmup_bars: int = 15

    MAX_DAILY_DD_R: float = 2.5
    MAX_TRADES_PER_SESSION: int = 40
    MAX_LOTS_PER_SYMBOL: int = 5
    MAX_NOTIONAL_RUPEES: float = 1_500_000.0
    MAX_GAMMA_MODE_LOTS: int = 2
    MAX_PORTFOLIO_DELTA_UNITS: int = 100
    MAX_PORTFOLIO_DELTA_UNITS_GAMMA: int = 60
    RISK_FREE_RATE: float = 0.065

    def model_post_init(self, __context: object) -> None:  # pragma: no cover - sync risk
        del __context
        try:
            self.system.log_buffer_capacity = int(self.diag_ring_size)
        except Exception:
            pass
        self.risk.use_live_equity = bool(self.RISK_USE_LIVE_EQUITY)
        self.risk.default_equity = float(self.RISK_DEFAULT_EQUITY)
        self.risk.exposure_basis = self.EXPOSURE_BASIS
        cap_source = self.EXPOSURE_CAP_SOURCE
        if cap_source == "env":
            cap_source = "absolute"
        self.risk.exposure_cap_source = cap_source
        self.risk.exposure_cap_pct_of_equity = self.EXPOSURE_CAP_PCT_OF_EQUITY
        self.risk.exposure_cap_abs = self.EXPOSURE_CAP_ABS
        self.risk.premium_cap_per_trade = self.PREMIUM_CAP_PER_TRADE
        min_floor_env = env_any("RISK_MIN_EQUITY_FLOOR", "MIN_EQUITY_FLOOR")
        if min_floor_env is not None:
            try:
                self.risk.min_equity_floor = float(min_floor_env)
            except ValueError:
                pass
    ROLL10_PAUSE_R: float = -0.2
    ROLL10_PAUSE_MIN: int = 60
    COOLOFF_LOSS_STREAK: int = 3
    COOLOFF_MINUTES: int = 45
    JOURNAL_DB_PATH: str = "data/journal.sqlite"

    zerodha: ZerodhaSettings = Field(default_factory=ZerodhaSettings.from_env)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings.from_env)
    data: DataSettings = DataSettings()  # type: ignore[call-arg]
    instruments: InstrumentsSettings = InstrumentsSettings()
    strategy: StrategySettings = StrategySettings()  # type: ignore[call-arg]
    regime: RegimeSettings = RegimeSettings()
    micro: MicroSettings = Field(default_factory=_micro_settings_factory)
    risk: RiskSettings = Field(default_factory=_risk_settings_factory)
    executor: ExecutorSettings = Field(
        default_factory=lambda: ExecutorSettings(
            entry_slippage_pct=0.25,
            exit_slippage_pct=0.25,
            depth_min_lots=5.0,
            depth_multiplier=5.0,
            micro_retry_limit=3,
        )
    )
    option_selector: OptionSelectorSettings = Field(
        default_factory=lambda: OptionSelectorSettings()  # type: ignore[call-arg]
    )
    health: HealthSettings = HealthSettings()  # type: ignore[call-arg]
    system: SystemSettings = SystemSettings()

    @field_validator("warmup_bars")
    @classmethod
    def _v_warmup(cls, v: int) -> int:
        if v < 0:
            raise ValueError("warmup_bars must be >= 0")
        return v

    @field_validator("log_level")
    @classmethod
    def _v_log_level(cls, v: str) -> str:
        return str(v or "INFO").upper()

    @field_validator("log_path", mode="before")
    @classmethod
    def _v_log_path(cls, v: object) -> Path | None:
        if v is None:
            return None
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            if not v.strip():
                return None
            return Path(v.strip())
        raise TypeError("log_path must be a string or Path")

    @field_validator("diag_ring_size")
    @classmethod
    def _v_diag_ring(cls, v: int) -> int:
        val = int(v)
        if val <= 0:
            raise ValueError("diag_ring_size must be > 0")
        return val

    @field_validator("data", mode="before")
    @classmethod
    def _v_data_aliases(cls, v: object) -> object:
        ht = os.getenv("HISTORICAL_TIMEFRAME")
        if ht:
            if isinstance(v, dict):
                v.setdefault("timeframe", ht)
            elif isinstance(v, DataSettings):
                v.timeframe = ht
        return v

    model_config = SettingsConfigDict(
        env_file=".env",  # used locally; Railway uses real env vars
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",  # e.g., TELEGRAM__BOT_TOKEN
        extra="ignore",
    )

    # -------- Flat alias properties (read-only) --------
    @property
    def TZ(self) -> str:  # pragma: no cover - simple alias
        return self.tz

    @property
    def LOG_PATH(self) -> Path | None:  # pragma: no cover - simple alias
        return self.log_path

    # Strategy (flat)
    @property
    def strategy_min_signal_score(self) -> int:
        return self.strategy.min_signal_score

    @property
    def strategy_confidence_threshold(self) -> float:
        return self.strategy.confidence_threshold

    @property
    def strategy_min_bars_for_signal(self) -> int:
        return self.strategy.min_bars_for_signal

    @property
    def strategy_ema_fast(self) -> int:
        return self.strategy.ema_fast

    @property
    def strategy_ema_slow(self) -> int:
        return self.strategy.ema_slow

    @property
    def strategy_rsi_period(self) -> int:
        return self.strategy.rsi_period

    @property
    def strategy_bb_period(self) -> int:
        return self.strategy.bb_period

    @property
    def strategy_bb_std(self) -> float:
        return self.strategy.bb_std

    @property
    def strategy_atr_period(self) -> int:
        return self.strategy.atr_period

    @property
    def strategy_atr_sl_multiplier(self) -> float:
        return self.strategy.atr_sl_multiplier

    @property
    def strategy_atr_tp_multiplier(self) -> float:
        return self.strategy.atr_tp_multiplier

    @property
    def strategy_rr_min(self) -> float:
        return self.strategy.rr_min

    @property
    def strategy_rr_threshold(self) -> float | None:
        return self.strategy.rr_threshold

    # Risk (flat)
    @property
    def risk_use_live_equity(self) -> bool:
        return self.risk.use_live_equity

    @property
    def risk_default_equity(self) -> float:
        return self.risk.default_equity

    @property
    def risk_min_equity_floor(self) -> float:
        return self.risk.min_equity_floor

    @property
    def risk_equity_refresh_seconds(self) -> int:
        return self.risk.equity_refresh_seconds

    @property
    def risk_risk_per_trade(self) -> float:
        return self.risk.risk_per_trade

    @property
    def risk_max_trades_per_day(self) -> int:
        return self.risk.max_trades_per_day

    @property
    def risk_consecutive_loss_limit(self) -> int:
        return self.risk.consecutive_loss_limit

    @property
    def risk_max_daily_drawdown_pct(self) -> float:
        return self.risk.max_daily_drawdown_pct

    @property
    def risk_max_position_size_pct(self) -> float:
        return self.risk.max_position_size_pct

    @property
    def risk_trading_window_start(self) -> str:
        return self.risk.trading_window_start

    @property
    def risk_trading_window_end(self) -> str:
        return self.risk.trading_window_end

    @property
    def risk_no_new_after_hhmm(self) -> str | None:
        return self.risk.no_new_after_hhmm

    @property
    def risk_eod_flatten_hhmm(self) -> str:
        return self.risk.eod_flatten_hhmm

    @property
    def risk_max_daily_loss_rupees(self) -> float | None:
        return self.risk.max_daily_loss_rupees

    @property
    def risk_max_lots_per_symbol(self) -> int:
        return self.risk.max_lots_per_symbol

    @property
    def risk_max_notional_rupees(self) -> float:
        return self.risk.max_notional_rupees

    @property
    def risk_exposure_basis(self) -> str:
        return self.risk.exposure_basis

    @property
    def risk_exposure_cap_source(self) -> str:
        return self.risk.exposure_cap_source

    @property
    def risk_exposure_cap_pct_of_equity(self) -> float:
        return self.risk.exposure_cap_pct_of_equity

    @property
    def risk_premium_cap_per_trade(self) -> float:
        return self.risk.premium_cap_per_trade

    @property
    def risk_allow_min_one_lot(self) -> bool:
        return self.risk.allow_min_one_lot

    # Instruments (flat)
    @property
    def instruments_spot_symbol(self) -> str:
        return self.instruments.spot_symbol

    @property
    def instruments_trade_symbol(self) -> str:
        return self.instruments.trade_symbol

    @property
    def instruments_trade_exchange(self) -> str:
        return self.instruments.trade_exchange

    @property
    def instruments_instrument_token(self) -> int:
        return self.instruments.instrument_token

    @property
    def instruments_nifty_lot_size(self) -> int:
        return self.instruments.nifty_lot_size

    @property
    def instruments_strike_range(self) -> int:
        return self.instruments.strike_range

    @property
    def instruments_min_lots(self) -> int:
        return self.instruments.min_lots

    @property
    def instruments_max_lots(self) -> int:
        return self.instruments.max_lots

    # Data (flat)
    @property
    def data_lookback_minutes(self) -> int:
        return self.data.lookback_minutes

    @property
    def data_timeframe(self) -> str:
        return self.data.timeframe

    @property
    def data_time_filter_start(self) -> str:
        return self.data.time_filter_start

    @property
    def data_time_filter_end(self) -> str:
        return self.data.time_filter_end

    @property
    def data_cache_enabled(self) -> bool:
        return self.data.cache_enabled

    @property
    def data_cache_ttl_seconds(self) -> int:
        return self.data.cache_ttl_seconds

    # Executor (flat)
    @property
    def executor_exchange(self) -> str:
        return self.executor.exchange

    @property
    def executor_order_product(self) -> str:
        return self.executor.order_product

    @property
    def executor_order_variety(self) -> str:
        return self.executor.order_variety

    @property
    def executor_entry_order_type(self) -> str:
        return self.executor.entry_order_type

    @property
    def executor_tick_size(self) -> float:
        return self.executor.tick_size

    @property
    def executor_exchange_freeze_qty(self) -> int:
        return self.executor.exchange_freeze_qty

    @property
    def executor_preferred_exit_mode(self) -> str:
        return self.executor.preferred_exit_mode

    @property
    def executor_use_slm_exit(self) -> bool:
        return self.executor.use_slm_exit

    @property
    def executor_partial_tp_enable(self) -> bool:
        return self.executor.partial_tp_enable

    @property
    def executor_tp1_qty_ratio(self) -> float:
        return self.executor.tp1_qty_ratio

    @property
    def executor_breakeven_ticks(self) -> int:
        return self.executor.breakeven_ticks

    @property
    def executor_enable_trailing(self) -> bool:
        return self.executor.enable_trailing

    @property
    def executor_trailing_atr_multiplier(self) -> float:
        return self.executor.trailing_atr_multiplier

    @property
    def executor_fee_per_lot(self) -> float:
        return self.executor.fee_per_lot

    @property
    def executor_slippage_ticks(self) -> int:
        return self.executor.slippage_ticks

    @property
    def executor_entry_slippage_pct(self) -> float:
        return self.executor.entry_slippage_pct

    @property
    def executor_exit_slippage_pct(self) -> float:
        return self.executor.exit_slippage_pct

    @property
    def executor_ack_timeout_ms(self) -> int:
        return self.executor.ack_timeout_ms

    @property
    def executor_fill_timeout_ms(self) -> int:
        return self.executor.fill_timeout_ms

    @property
    def executor_max_place_retries(self) -> int:
        return self.executor.max_place_retries

    # Health/System (flat)
    @property
    def health_enable_server(self) -> bool:
        return self.health.enable_server

    @property
    def health_host(self) -> str:
        return self.health.host

    @property
    def health_port(self) -> int:
        return self.health.port

    @property
    def system_max_api_calls_per_second(self) -> float:
        return self.system.max_api_calls_per_second

    @property
    def system_websocket_reconnect_attempts(self) -> int:
        return self.system.websocket_reconnect_attempts

    @property
    def system_order_timeout_seconds(self) -> int:
        return self.system.order_timeout_seconds

    @property
    def system_position_sync_interval(self) -> int:
        return self.system.position_sync_interval


def _apply_env_overrides(cfg: AppSettings) -> None:
    """Apply environment-based overrides to the loaded settings."""
    def _bool_env(name: str, default: bool) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return str(val).lower() in {"1", "true", "yes", "on"}

    def _int_env(name: str, default: int) -> int:
        val = os.getenv(name)
        return int(val) if val is not None else default

    def _float_env(name: str, default: float) -> float:
        val = os.getenv(name)
        return float(val) if val is not None else default

    cfg.data.timeframe = os.getenv("HISTORICAL_TIMEFRAME", cfg.data.timeframe)
    object.__setattr__(
        cfg,
        "ENABLE_SIGNAL_DEBUG",
        _bool_env(
            "ENABLE_SIGNAL_DEBUG",
            bool(getattr(cfg, "ENABLE_SIGNAL_DEBUG", False)),
        ),
    )
    object.__setattr__(
        cfg,
        "TELEGRAM__PRETRADE_ALERTS",
        _bool_env(
            "TELEGRAM__PRETRADE_ALERTS",
            bool(getattr(cfg, "TELEGRAM__PRETRADE_ALERTS", False)),
        ),
    )
    periodic_logs = _bool_env(
        "TELEGRAM__PERIODIC_LOGS",
        bool(getattr(cfg, "TELEGRAM__PERIODIC_LOGS", cfg.telegram_periodic_logs)),
    )
    object.__setattr__(cfg, "TELEGRAM__PERIODIC_LOGS", periodic_logs)
    object.__setattr__(cfg, "telegram_periodic_logs", periodic_logs)
    object.__setattr__(
        cfg,
        "DIAG_INTERVAL_SECONDS",
        _int_env("DIAG_INTERVAL_SECONDS", cfg.diag_interval_seconds),
    )
    object.__setattr__(
        cfg,
        "MIN_PREVIEW_SCORE",
        _float_env("MIN_PREVIEW_SCORE", cfg.min_preview_score),
    )
    object.__setattr__(
        cfg, "ACK_TIMEOUT_MS", _int_env("ACK_TIMEOUT_MS", cfg.executor.ack_timeout_ms)
    )
    object.__setattr__(
        cfg,
        "FILL_TIMEOUT_MS",
        _int_env("FILL_TIMEOUT_MS", cfg.executor.fill_timeout_ms),
    )
    object.__setattr__(
        cfg,
        "RETRY_BACKOFF_MS",
        _int_env("RETRY_BACKOFF_MS", cfg.retry_backoff_ms),
    )
    object.__setattr__(
        cfg,
        "MAX_PLACE_RETRIES",
        _int_env("MAX_PLACE_RETRIES", cfg.max_place_retries),
    )
    object.__setattr__(
        cfg,
        "MAX_MODIFY_RETRIES",
        _int_env("MAX_MODIFY_RETRIES", cfg.max_modify_retries),
    )
    object.__setattr__(
        cfg,
        "PLAN_STALE_SEC",
        _int_env("PLAN_STALE_SEC", cfg.plan_stale_sec),
    )


def load_settings() -> AppSettings:
    """Return application settings loaded from the environment.

    Ensures Pydantic's settings validation directory exists under
    ``~/.config/pydantic/settings/nifty_scalper_bot``.
    """
    validation_dir = (
        Path.home() / ".config" / "pydantic" / "settings" / "nifty_scalper_bot"
    )
    validation_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYDANTIC_SETTINGS_DIR", str(validation_dir))
    load_dotenv(override=False)
    cfg = AppSettings(
        zerodha=ZerodhaSettings.from_env(),
        telegram=TelegramSettings.from_env(),
    )  # type: ignore[call-arg]
    _apply_env_overrides(cfg)
    snap = cfg.model_dump()
    for k in ("api_key", "api_secret", "access_token"):
        if k in snap.get("zerodha", {}):
            snap["zerodha"][k] = "***"
    if "telegram" in snap and "bot_token" in snap["telegram"]:
        snap["telegram"]["bot_token"] = "***"
    logging.getLogger("config").info("settings snapshot: %s", snap)
    return cfg


class _SettingsProxy:
    """Lazily load settings on first attribute access."""

    _settings: AppSettings | None = None

    def _load(self) -> AppSettings:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def __getattr__(self, item: str):  # pragma: no cover - passthrough
        return getattr(self._load(), item)

    def build_log_gate(self, interval_s: float | None = None):
        """Return a ``LogGate`` instance honouring optional overrides."""

        from src.utils.log_gate import LogGate

        cfg = self._load()
        base = interval_s
        if base is None:
            raw = getattr(cfg, "log_gate_interval_s", None)
            if raw is None:
                env_val = os.getenv("LOG_GATE_INTERVAL_S")
                if env_val:
                    try:
                        raw = float(env_val)
                    except ValueError:  # pragma: no cover - defensive
                        raw = None
            base = raw if raw is not None else 1.0
        try:
            interval = float(base)
        except Exception:  # pragma: no cover - defensive
            interval = 1.0
        return LogGate(interval_s=interval)


# Public singleton used by the rest of the application
settings = _SettingsProxy()

__all__ = [
    "AppSettings",
    "RegimeSettings",
    "InstrumentConfig",
    "InstrumentsSettings",
    "MicroSettings",
    "load_settings",
    "settings",
]
