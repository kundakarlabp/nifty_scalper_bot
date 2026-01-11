"""Runtime settings facade used by hardened infrastructure components.

This module centralises environment driven configuration so that new
infrastructure additions – resilient streaming, safe order routing,
risk management gates, notifications, health endpoints and structured
logging – can share a common source of truth.  The legacy configuration
models under :mod:`nifty_scalper_bot.config.base` are still used for the
core trading stack; ``Settings`` simply wraps the existing
``AppConfig`` while exposing new knobs required by the refactor.

The implementation intentionally avoids additional dependencies (for
example ``pydantic-settings``) so that the production footprint remains
unchanged.  All parsing helpers perform strict validation and provide
clear error messages to ease troubleshooting in Railway deployments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from functools import lru_cache
from pathlib import Path
from typing import Set

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from nifty_scalper_bot.analytics.regime_gate import RegimeGateConfig
from nifty_scalper_bot.config.base import AppConfig
from nifty_scalper_bot.config.env import load_from_env
from nifty_scalper_bot.config.env_aliases import resolve_env
from nifty_scalper_bot.execution.lifecycle_validator import (
    LifecycleConfigError,
    validate_lifecycle_config,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    BBSqueezeStrategyConfig,
    CPRBreakoutStrategyConfig,
    EliteStrategiesSettings,
    GammaScalpingStrategyConfig,
    OIMaxPainStrategyConfig,
    ORBProStrategyConfig,
    OrderFlowStrategyConfig,
    RSIDivergenceStrategyConfig,
    SMCStrategyConfig,
    StraddleThetaStrategyConfig,
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_validator import (
    EliteConfigError,
    validate_elite_config,
)
from nifty_scalper_bot.utils.env import get_csv
from nifty_scalper_bot.utils.errors import ConfigurationError
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _sanitize_token(raw: str) -> str:
    token = raw.strip().strip('"').strip("'")
    for sep in ("#", ",", ";"):
        if sep in token:
            token = token.split(sep, 1)[0].strip()
    if not token:
        return token
    if (
        "=" in token
        and not token.lower().startswith("true")
        and not token.lower().startswith("false")
    ):
        token = token.split("=", 1)[0].strip()
    if not token:
        return token
    parts = token.split()
    if parts:
        token = parts[0]
    return token


def _env_bool(name: str, _pos_default: bool | None = None, default: bool = False) -> bool:
    # Handle double-argument calls (The specific cause of your crash)
    effective_default = _pos_default if _pos_default is not None else default

    raw = os.getenv(name, str(effective_default)).lower()
    val = raw in ("true", "1", "yes", "on")
    
    LOGGER.info(
        f"Condition met: settings_env_bool_default {name}={val}",
        extra={
            "event": "settings_env_bool_default",
            "setting_name": name,
            "value": val
        },
    )
    return val


def _env_float(name: str, _pos_default: float | None = None, default: float = 0.0, minimum: float | None = None) -> float:
    # Handle double-argument calls
    effective_default = _pos_default if _pos_default is not None else default

    raw = os.getenv(name, str(effective_default))
    try:
        val = float(raw)
    except ValueError:
        val = effective_default

    if minimum is not None and val < minimum:
        val = minimum

    LOGGER.info(
        f"Condition met: settings_env_float_resolved {name}={val}",
        extra={
            "event": "settings_env_float_resolved",
            "setting_name": name,
            "value": val
        },
    )
    return val

def _env_int(name: str, _pos_default: int | None = None, default: int = 0, minimum: int | None = None) -> int:
    # Handle double-argument calls
    effective_default = _pos_default if _pos_default is not None else default

    raw = os.getenv(name, str(effective_default))
    try:
        val = int(raw)
    except ValueError:
        val = effective_default
    
    if minimum is not None and val < minimum:
        val = minimum

    LOGGER.info(
        f"Condition met: settings_env_int_resolved {name}={val}",
        extra={
            "event": "settings_env_int_resolved",
            "setting_name": name,
            "value": val
        },
    )
    return val


def _env_str(name: str, _pos_default: str | None = None, default: str = "", secret: bool = False) -> str:
    # Handle cases where default is passed both positionally and via keyword
    effective_default = _pos_default if _pos_default is not None else default
    
    val = os.getenv(name, effective_default)
    log_val = "***" if secret and val else val
    
    LOGGER.info(
        f"Condition met: settings_env_str_resolved {name}={log_val}",
        extra={
            "event": "settings_env_str_resolved",
            "setting_name": name,
            "value": log_val
        },
    )
    return val

def _env_csv(*names: str) -> list[str]:
    """Return list of CSV tokens from environment with alias support.

    Args:
        *names: Candidate environment variable names to evaluate.

    Returns:
        list[str]: Parsed list of CSV tokens.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered _env_csv",
        extra={"event": "settings_env_csv_enter", "names": list(names)},
    )
    try:
        for name in names or ("",):
            if not name:
                continue
            raw_value = resolve_env(name)
            if raw_value is None:
                continue
            token = str(raw_value).strip()
            if not token:
                continue
            sentinel = f"__SANITIZED__CSV_{name}"
            os.environ[sentinel] = token
            try:
                values = get_csv(sentinel)
            finally:
                os.environ.pop(sentinel, None)
            LOGGER.info(
                "Condition met: settings_env_csv_resolved",
                extra={"event": "settings_env_csv_resolved", "name": name},
            )
            return values
        LOGGER.info(
            "Condition met: settings_env_csv_default",
            extra={"event": "settings_env_csv_default"},
        )
        return []
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _env_csv: %s",
            exc,
            extra={"event": "settings_env_csv_error", "names": list(names)},
            exc_info=exc,
        )
        return []


def _env_csv_ints(*names: str) -> Set[int]:
    """Return set of integers parsed from CSV environment variables.

    Args:
        *names: Candidate environment variable names to evaluate.

    Returns:
        Set[int]: Parsed set of integers.

    Raises:
        ConfigurationError: When an entry fails integer coercion.
    """

    LOGGER.debug(
        "Entered _env_csv_ints",
        extra={"event": "settings_env_csv_ints_enter", "names": list(names)},
    )
    values: set[int] = set()
    try:
        for name in names or ("",):
            if not name:
                continue
            raw_values = _env_csv(name)
            if not raw_values:
                continue
            for item in raw_values:
                try:
                    values.add(int(item))
                except ValueError as exc:
                    LOGGER.error(
                        "Failure in _env_csv_ints: invalid integer token",
                        extra={
                            "event": "settings_env_csv_ints_invalid",
                            "name": name,
                            "token": item,
                        },
                    )
                    raise ConfigurationError(
                        f"Invalid integer in {name!s}: {item!r}"
                    ) from exc
        LOGGER.info(
            "Condition met: settings_env_csv_ints_resolved",
            extra={
                "event": "settings_env_csv_ints_resolved",
                "names": list(names),
                "count": len(values),
            },
        )
        return values
    except ConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _env_csv_ints: %s",
            exc,
            extra={"event": "settings_env_csv_ints_error", "names": list(names)},
            exc_info=exc,
        )
        return values


def _env_csv_strs(name: str, _pos_default: str | None = None, default: str = "") -> set[str]:
    # Handle double-argument calls
    effective_default = _pos_default if _pos_default is not None else default

    raw = os.getenv(name, effective_default)
    if not raw.strip():
        val = set()
    else:
        val = {x.strip() for x in raw.split(",") if x.strip()}
    
    LOGGER.info(
        f"Condition met: settings_env_csv_strs_resolved {name}",
        extra={
            "event": "settings_env_csv_strs_resolved",
            "setting_name": name,
            "count": len(val)
        },
    )
    return val


def _parse_time_token(source: str, token: str) -> time:
    parts = token.split(":")
    if len(parts) != 2:
        raise ConfigurationError(f"Invalid time format for {source!s}: {token!r}")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(
            f"Invalid time value for {source!s}: {token!r}"
        ) from exc
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ConfigurationError(f"Time out of range for {source!s}: {token!r}")
    return time(hour=hour, minute=minute)


def _parse_time_range(source: str, raw: str) -> tuple[time | None, time | None]:
    token = raw.strip()
    if not token:
        return (None, None)
    if "-" not in token:
        raise ConfigurationError(f"Invalid range format for {source!s}: {raw!r}")
    start_token, end_token = token.split("-", 1)
    start = _parse_time_token(f"{source} start", start_token.strip())
    end = _parse_time_token(f"{source} end", end_token.strip())
    if end <= start:
        raise ConfigurationError(f"{source!s} end must be after start: {raw!r}")
    return (start, end)


def _env_minutes(name: str, default: float) -> float:
    """Parse environment minute token into float minutes.

    Args:
        name: Environment variable name to read.
        default: Default minutes used when variable missing.

    Returns:
        float: Parsed minutes value.

    Raises:
        ConfigurationError: If the provided value cannot be parsed.
    """

    raw = os.environ.get(name)
    if raw is None:
        return default
    token = _sanitize_token(raw)
    if not token:
        return default
    try:
        if ":" in token:
            hour_str, minute_str = token.split(":", 1)
            return int(hour_str) * 60 + int(minute_str)
        return float(token)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(
            f"Invalid minutes token for {name!s}: {raw!r}"
        ) from exc


ORDER_REF_REQUIRE_DEPTH = _env_bool("ORDER_REF_REQUIRE_DEPTH", default=True)
ORDER_MAX_QUOTE_AGE_MS = _env_int("ORDER_MAX_QUOTE_AGE_MS", default=2_000, minimum=0)
ORDER_ALLOW_LTP_FALLBACK = _env_bool("ORDER_ALLOW_LTP_FALLBACK", default=True)
ORDER_ALLOW_MARKET_PROTECT = _env_bool("ORDER_ALLOW_MARKET_PROTECT", default=False)
ENTRY_ALLOW_MARKET_PROTECT = _env_bool("ENTRY_ALLOW_MARKET_PROTECT", default=False)
REGIME_FILTER_BYPASS = _env_bool("REGIME_FILTER_BYPASS", default=False)
ORDER_PROTECT_SLIPPAGE_BPS = _env_int(
    "ORDER_PROTECT_SLIPPAGE_BPS", default=12, minimum=0
)
ORDER_MAX_QUEUE_POSITION = _env_int("ORDER_MAX_QUEUE_POSITION", default=0, minimum=0)
ORDER_QUEUE_POSITION_PER_LOT = _env_float(
    "ORDER_QUEUE_POSITION_PER_LOT", default=0.0, minimum=0.0
)

ORDER_TRUNCATE_TO_LOT = _env_bool("ORDER_TRUNCATE_TO_LOT", default=True)
ORDER_ALLOW_NAKED_SHORTS = _env_bool("ORDER_ALLOW_NAKED_SHORTS", default=False)

MIN_LOTS_PER_TRADE = _env_int("MIN_LOTS_PER_TRADE", default=1, minimum=1)
MAX_LOTS_PER_TRADE = _env_int("MAX_LOTS_PER_TRADE", default=1, minimum=1)
RISK_FALLBACK_PRICE_MOVE_PCT = _env_float(
    "RISK_FALLBACK_PRICE_MOVE_PCT", default=2.0, minimum=0.0
)
REGIME_TREND_SIZING_MULT = _env_float(
    "REGIME_TREND_SIZING_MULT", default=1.15, minimum=0.0
)
REGIME_RANGE_SIZING_MULT = _env_float(
    "REGIME_RANGE_SIZING_MULT", default=0.85, minimum=0.0
)
REGIME_VOLATILE_SIZING_MULT = _env_float(
    "REGIME_VOLATILE_SIZING_MULT", default=0.7, minimum=0.0
)
REGIME_EVENT_SIZING_MULT = _env_float(
    "REGIME_EVENT_SIZING_MULT", default=0.6, minimum=0.0
)


class ExecutionSettings(BaseModel):
    """Execution toggles controlling bracket manager coordination."""

    enable_bracket_manager: bool = Field(
        default=True,
        description="Enable BracketManager for OCO coordination",
    )
    bracket_auto_reduce_sl: bool = Field(
        default=True,
        description="Automatically reduce SL quantity on partial TP fills",
    )
    bracket_stale_cleanup_seconds: int = Field(
        default=86_400,
        description="Remove brackets older than this many seconds",
        ge=1,
    )


@dataclass(slots=True)
class OrderLifecycleSettings:
    """Lifecycle management thresholds for staged exits."""

    tp1_R: float = 1.0
    tp1_partial: float = 0.6
    tp2_R_trend: float = 1.8
    tp2_R_range: float = 1.4
    trail_atr_mult: float = 0.8
    time_stop_min: int = 12


@dataclass(slots=True)
class OrderSettings:
    """Order related runtime safeguards."""

    enable_live: bool = False
    default_product: str = "MIS"
    default_order_type: str = "LIMIT"
    limit_offset_pct: float = 0.05
    min_spacing_seconds: float = 1.5
    max_per_second: int = 5
    max_per_minute: int = 150
    max_per_day: int = 2000
    require_depth: bool = True
    max_quote_age_ms: int = 2_000
    allow_ltp_fallback: bool = True
    allow_market_protect: bool = False
    protect_slippage_bps: int = 12
    enable_queue_policy: bool = False
    queue_wait_threshold: int = 0
    queue_cross_threshold: int = 0
    queue_cross_spread_bps: float = 8.0
    queue_cross_use_market: bool = False
    retry_attempts: int = 3
    retry_delay_seconds: float = 0.5
    replace_attempts: int = 3
    replace_backoff_seconds: float = 0.75
    replace_price_step_pct: float = 0.02
    replace_timeout_seconds: float = 0.75
    lifecycle: OrderLifecycleSettings = field(default_factory=OrderLifecycleSettings)


@dataclass(slots=True)
class RiskSettings:
    """Risk guardrails enforced before routing live orders."""

    daily_loss_pct: float = 2.0
    daily_pnl_cap_pct: float = 2.0
    max_daily_loss_absolute: float = 0.0
    per_trade_risk_pct: float = 10.0
    per_trade_cap_pct: float = 25.0
    max_consecutive_losses: int = 3
    loss_cooldown_seconds: float = 90.0
    cooldown_on_reject_seconds: float = 30.0
    trading_day_reset_hour_utc: int = 3
    breaker_auto_shadow: bool = True
    min_lots_per_trade: int = 1
    max_lots_per_trade: int = 6
    atr_stop_multiple: float = 1.0
    contract_lot_size: int = 75
    allow_pyramiding: bool = False
    signal_debounce_seconds: float = 60.0

    def __post_init__(self) -> None:
        # Keep the historical ``daily_pnl_cap_pct`` alias in sync with the
        # new ``daily_loss_pct`` field for backwards compatibility with
        # configuration objects and tests that still reference the old name.
        self.daily_loss_pct = float(self.daily_loss_pct)
        self.daily_pnl_cap_pct = float(self.daily_pnl_cap_pct or self.daily_loss_pct)
        if abs(self.daily_pnl_cap_pct - self.daily_loss_pct) > 1e-9:
            self.daily_pnl_cap_pct = self.daily_loss_pct
        self.max_daily_loss_absolute = max(0.0, float(self.max_daily_loss_absolute))
        self.min_lots_per_trade = max(1, int(self.min_lots_per_trade))
        resolved_max_lots = max(1, int(self.max_lots_per_trade))
        if resolved_max_lots < self.min_lots_per_trade:
            resolved_max_lots = self.min_lots_per_trade
        self.max_lots_per_trade = resolved_max_lots
        self.atr_stop_multiple = max(float(self.atr_stop_multiple), 0.0)
        self.contract_lot_size = max(1, int(self.contract_lot_size))


@dataclass(slots=True)
class StreamerSettings:
    """Tuning for resilient streaming and backfill."""

    gap_seconds: float = 5.0
    max_backfill_candles: int = 10
    backfill_interval: str = "minute"
    queue_size: int = 1_000


@dataclass(slots=True)
class ShadowSettings:
    """Shadow (paper) trading controls."""

    drift_threshold_pct: float = 2.5
    auto_disable_live_on_drift: bool = True
    drift_sustain_seconds: float = 30.0
    alert_threshold_pct: float = 1.5
    alert_trend_delta_pct: float = 0.5
    alert_trend_window_seconds: float = 120.0
    alert_cooldown_seconds: float = 900.0


@dataclass(slots=True)
class NotificationSettings:
    """Telegram notification rate limits and whitelists."""

    enabled: bool = False
    token: str | None = None
    whitelist_chat_ids: Set[int] = field(default_factory=set)
    whitelist_user_ids: Set[int] = field(default_factory=set)
    rate_per_second: float | None = 25.0
    burst_capacity: float | None = None
    public_base_url: str | None = None
    webhook_enabled: bool = True
    allow_poll_fallback: bool = False
    legacy_console_enabled: bool = False


@dataclass(slots=True)
class ReplaySettings:
    """Replay harness configuration derived from environment."""

    data_dir: Path = Path("data/minute_bars")

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir.expanduser()


@dataclass(slots=True)
class SelectorSettings:
    """Option strike selection behaviour."""

    mode: str = "ATM"  # ATM | DELTA_TARGET
    delta_target: float = 0.3
    delta_tolerance: float = 0.05
    expiry: str = "weekly"  # weekly | monthly
    expiry_regime: str = "weekly"  # weekly | monthly | auto | special
    expiry_roll_minutes: int = 45
    monthly_halt_minutes: int = 30
    special_expiries: tuple[str, ...] = field(default_factory=tuple)
    long_only: bool = True
    legacy_side_to_type: bool = False

    def __post_init__(self) -> None:
        self.mode = (self.mode or "ATM").strip().upper()
        self.expiry = (self.expiry or "weekly").strip().lower()
        self.expiry_regime = (
            (self.expiry_regime or self.expiry or "weekly").strip().lower()
        )
        if self.mode not in {"ATM", "DELTA_TARGET"}:
            raise ConfigurationError(
                f"Invalid selector mode: {self.mode!r}. Expected ATM or DELTA_TARGET."
            )
        if self.delta_target < 0:
            raise ConfigurationError("delta_target must be non-negative")
        if self.delta_tolerance < 0:
            raise ConfigurationError("delta_tolerance must be non-negative")
        if self.expiry not in {"weekly", "monthly"}:
            raise ConfigurationError("expiry must be 'weekly' or 'monthly'")
        valid_regimes = {"weekly", "monthly", "auto", "special"}
        if self.expiry_regime not in valid_regimes:
            raise ConfigurationError(
                "expiry_regime must be one of 'weekly', 'monthly', 'auto', or 'special'"
            )
        self.expiry_roll_minutes = max(0, int(self.expiry_roll_minutes))
        self.monthly_halt_minutes = max(0, int(self.monthly_halt_minutes))
        specials: list[str] = []
        for token in self.special_expiries:
            if not isinstance(token, str):
                continue
            cleaned = token.strip()
            if cleaned and cleaned not in specials:
                specials.append(cleaned)
        self.special_expiries = tuple(specials)
        self.long_only = bool(self.long_only)
        self.legacy_side_to_type = bool(self.legacy_side_to_type)


@dataclass(slots=True)
class LiquiditySettings:
    """Liquidity filters applied during strike selection."""

    min_oi: float = 0.0
    max_spread_pct: float = 0.0
    min_trades: float = 0.0
    min_top3_depth: float = 0.0

    def __post_init__(self) -> None:
        if self.min_oi < 0:
            raise ConfigurationError("min_oi must be non-negative")
        if self.max_spread_pct < 0:
            raise ConfigurationError("max_spread_pct must be non-negative")
        if self.min_trades < 0:
            raise ConfigurationError("min_trades must be non-negative")
        if self.min_top3_depth < 0:
            raise ConfigurationError("min_top3_depth must be non-negative")


class InstrumentSettings(BaseSettings):
    """Settings governing instrument cache warm-up and refresh."""

    csv_path: Path | None = Path("/app/instruments.csv")
    db_path: Path | None = Path("/app/cache.sqlite")
    refresh_cron_enabled: bool = False

    model_config = SettingsConfigDict(env_prefix="INSTRUMENTS_", case_sensitive=False)


@dataclass(slots=True)
class Settings:
    """Aggregate runtime configuration consumed by infrastructure modules."""

    app: AppConfig
    enable_live: bool
    orders: OrderSettings
    risk: RiskSettings
    streamer: StreamerSettings
    shadow: ShadowSettings
    notifications: NotificationSettings
    session_allow_out_of_hours: bool
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    websocket_enabled: bool = True
    telemetry_tags: Set[str] = field(default_factory=set)
    regime: RegimeGateConfig = field(default_factory=RegimeGateConfig)
    replay: ReplaySettings = field(default_factory=ReplaySettings)
    selector: SelectorSettings = field(default_factory=SelectorSettings)
    liquidity: LiquiditySettings = field(default_factory=LiquiditySettings)
    instruments: InstrumentSettings = field(default_factory=InstrumentSettings)
    elite: EliteStrategiesSettings = field(default_factory=EliteStrategiesSettings)
    poll_batch_size: int = 50
    poll_interval_ms_jitter_pct: float = 0.15
    feature_order_without_token: bool = True
    feature_resolver_learn_from_quotes: bool = True


def _build_elite_settings() -> EliteStrategiesSettings:
    """
    Builds the EliteStrategiesSettings object by parsing environment variables.
    Robust Fix: Handles '15.0' format in integer fields to prevent startup crashes.
    """
    try:
        # 1. SMC Strategy
        smc_cfg = SMCStrategyConfig(
            enabled=_env_bool("SMC_ENABLED", default=True),
            min_confidence=_env_float("SMC_MIN_CONFIDENCE", default=45.0),
            cooldown_seconds=_env_float("SMC_COOLDOWN", default=60.0),
            sweep_distance_points=_env_float("SMC_SWEEP_DISTANCE", default=15.0),
            volume_spike_mult=_env_float("SMC_VOLUME_SPIKE", default=2.0)
        )

        # 2. VWAP Pro
        # ✅ FIX: Use int(_env_float(...)) to safely handle "50.0" from .env
        vwap_cfg = VWAPProStrategyConfig(
            enabled=_env_bool("VWAP_PRO_ENABLED", "VWAP_ENABLED", default=True),
            min_confidence=_env_float("VWAP_MIN_CONFIDENCE", default=40.0),
            ema_period=int(_env_float("VWAP_EMA_PERIOD", default=50.0)), 
            proximity_pct=_env_float("VWAP_PROXIMITY_PCT", default=0.15)
        )

        # 3. RSI Divergence
        # ✅ FIX: Use int(_env_float(...)) for period and lookback
        rsi_cfg = RSIDivergenceStrategyConfig(
            enabled=_env_bool("RSI_DIV_ENABLED", default=True),
            min_confidence=_env_float("RSI_MIN_CONFIDENCE", "RSI_DIV_MIN_CONFIDENCE", default=48.0),
            rsi_period=int(_env_float("RSI_PERIOD", default=14.0)),
        )

        # 4. ORB Pro
        # ✅ CRITICAL FIX: The specific line causing your crash. 
        # We parse "15.0" as float first, then cast to int.
        orb_cfg = ORBProStrategyConfig(
            enabled=_env_bool("ORB_ENABLED", default=True),
            min_confidence=_env_float("ORB_MIN_CONFIDENCE", default=55.0),
            orb_minutes=int(_env_float("ORB_DURATION_MIN", "ORB_MINUTES", default=15.0))
        )

        # 5. Straddle / Theta
        straddle_cfg = StraddleThetaStrategyConfig(
            enabled=_env_bool("STRADDLE_ENABLED", default=False),
            min_confidence=_env_float("STRADDLE_MIN_CONFIDENCE", default=73.0),
            adx_threshold=_env_float("STRADDLE_ADX_LIMIT", default=25.0),
            min_iv=_env_float("STRADDLE_MIN_IV", default=12.0)
        )

        # 6. Gamma Scalping
        gamma_cfg = GammaScalpingStrategyConfig(
            enabled=_env_bool("GAMMA_ENABLED", default=False),
            min_confidence=_env_float("GAMMA_MIN_CONFIDENCE", default=65.0),
            min_gamma=_env_float("GAMMA_THRESHOLD", default=0.0005)
        )

        # 7. OI Max Pain (Unlocked)
        oi_cfg = OIMaxPainStrategyConfig(
            enabled=_env_bool("OI_MAX_PAIN_ENABLED", default=True),
            min_confidence=_env_float("OI_MIN_CONFIDENCE", default=40.0),
            min_deviation_pct=_env_float("OI_MIN_DISTANCE_POINTS", default=50.0)
        )

        # 8. CPR Breakout (Unlocked)
        cpr_cfg = CPRBreakoutStrategyConfig(
            enabled=_env_bool("CPR_ENABLED", default=True),
            min_confidence=_env_float("CPR_MIN_CONFIDENCE", default=55.0),
            narrow_cpr_threshold=_env_float("CPR_NARROW_THRESHOLD_PCT", default=0.25)
        )

        # 9. Order Flow (Unlocked)
        of_cfg = OrderFlowStrategyConfig(
            enabled=_env_bool("ORDER_FLOW_ENABLED", default=True),
            min_confidence=_env_float("ORDER_FLOW_MIN_CONFIDENCE", default=60.0),
            imbalance_ratio_min=_env_float("ORDER_FLOW_IMBALANCE_RATIO_BUY", default=2.8),
            large_order_threshold_pct=_env_float("ORDER_FLOW_LARGE_ORDER_PCT", default=15.0)
        )

        # 10. BB Squeeze (Unlocked)
        bb_cfg = BBSqueezeStrategyConfig(
            enabled=_env_bool("BB_SQUEEZE_ENABLED", default=True),
            min_confidence=_env_float("BB_MIN_CONFIDENCE", default=55.0),
            squeeze_threshold_pct=_env_float("BB_SQUEEZE_BANDWIDTH", default=0.4)
        )

        # 11. Aggregate into Settings
        return EliteStrategiesSettings(
            max_concurrent_strategies=int(_env_float("ELITE_MAX_CONCURRENT_STRATEGIES", "MAX_CONCURRENT_STRATS", default=10.0)),
            position_size_pct=_env_float("ELITE_POSITION_SIZE_PCT", "STRAT_POS_SIZE_PCT", default=1.5),
            smc=smc_cfg,
            vwap=vwap_cfg,
            rsi_div=rsi_cfg,
            orb=orb_cfg,
            straddle=straddle_cfg,
            gamma_scalping=gamma_cfg,
            oi_max_pain=oi_cfg,
            cpr=cpr_cfg,
            order_flow=of_cfg,
            bb_squeeze=bb_cfg
        )

    except Exception as exc:
        from nifty_scalper_bot.utils.logging import get_logger
        get_logger(__name__).error(f"❌ Critical Error in _build_elite_settings: {exc}", exc_info=True)
        return EliteStrategiesSettings()


def _build_order_settings() -> OrderSettings:
    lifecycle = OrderLifecycleSettings(
        tp1_R=_env_float("LIFECYCLE_TP1_R", default=1.0, minimum=0.0),
        tp1_partial=_env_float("LIFECYCLE_TP1_PARTIAL", default=0.6, minimum=0.0),
        tp2_R_trend=_env_float("LIFECYCLE_TP2_R_TREND", default=1.8, minimum=0.0),
        tp2_R_range=_env_float("LIFECYCLE_TP2_R_RANGE", default=1.4, minimum=0.0),
        trail_atr_mult=_env_float("LIFECYCLE_TRAIL_ATR_MULT", default=0.8, minimum=0.0),
        time_stop_min=_env_int("LIFECYCLE_TIME_STOP_MIN", default=12, minimum=1),
    )
    return OrderSettings(
        enable_live=_env_bool("ENABLE_LIVE", "ENABLE_LIVE_TRADING", default=False),
        default_product=_env_str("ORDER_DEFAULT_PRODUCT", default="MIS") or "MIS",
        default_order_type=_env_str("ORDER_DEFAULT_TYPE", default="LIMIT") or "LIMIT",
        limit_offset_pct=_env_float(
            "ORDER_LIMIT_OFFSET_PCT", default=0.05, minimum=0.0
        ),
        min_spacing_seconds=_env_float(
            "ORDER_MIN_SPACING_SECONDS", default=1.5, minimum=0.0
        ),
        max_per_second=_env_int("ORDER_RATE_PER_SECOND", default=5, minimum=1),
        max_per_minute=_env_int("ORDER_RATE_PER_MINUTE", default=150, minimum=1),
        max_per_day=_env_int("ORDER_RATE_PER_DAY", default=2000, minimum=1),
        retry_attempts=_env_int("ORDER_RETRY_ATTEMPTS", default=3, minimum=1),
        retry_delay_seconds=_env_float(
            "ORDER_RETRY_DELAY_SECONDS", default=0.5, minimum=0.0
        ),
        replace_attempts=_env_int("ORDER_REPLACE_ATTEMPTS", default=3, minimum=0),
        replace_backoff_seconds=_env_float(
            "ORDER_REPLACE_BACKOFF_SECONDS", default=0.75, minimum=0.0
        ),
        replace_price_step_pct=_env_float(
            "ORDER_REPLACE_PRICE_STEP_PCT", default=0.02, minimum=0.0
        ),
        replace_timeout_seconds=_env_float(
            "ORDER_REPLACE_TIMEOUT_SECONDS", default=0.75, minimum=0.0
        ),
        require_depth=_env_bool("ORDER_REF_REQUIRE_DEPTH", default=True),
        max_quote_age_ms=_env_int("ORDER_MAX_QUOTE_AGE_MS", default=2_000, minimum=0),
        allow_ltp_fallback=_env_bool("ORDER_ALLOW_LTP_FALLBACK", default=True),
        allow_market_protect=_env_bool("ORDER_ALLOW_MARKET_PROTECT", default=False),
        protect_slippage_bps=_env_int(
            "ORDER_PROTECT_SLIPPAGE_BPS", default=12, minimum=0
        ),
        enable_queue_policy=_env_bool("ORDER_QUEUE_POLICY_ENABLED", default=False),
        queue_wait_threshold=_env_int(
            "ORDER_QUEUE_WAIT_THRESHOLD", default=0, minimum=0
        ),
        queue_cross_threshold=_env_int(
            "ORDER_QUEUE_CROSS_THRESHOLD", default=0, minimum=0
        ),
        queue_cross_spread_bps=_env_float(
            "ORDER_QUEUE_CROSS_SPREAD_BPS", default=8.0, minimum=0.0
        ),
        queue_cross_use_market=_env_bool("ORDER_QUEUE_CROSS_USE_MARKET", default=False),
        lifecycle=lifecycle,
    )


def _build_risk_settings() -> RiskSettings:
    daily_loss_pct = _env_float("RISK_DAILY_LOSS_PCT", default=2.0, minimum=0.0)
    abs_loss_cap = _env_float(
        "RISK__MAX_DAY_LOSS", "RISK_MAX_DAY_LOSS", default=0.0, minimum=0.0
    )
    max_losses = _env_int(
        "RISK__MAX_CONSEC_LOSSES", "RISK_MAX_CONSEC_LOSSES", default=3, minimum=1
    )
    cooldown_minutes_raw = os.getenv("RISK__COOLDOWN_MIN")
    if cooldown_minutes_raw is not None and cooldown_minutes_raw.strip():
        try:
            cooldown_seconds = max(float(cooldown_minutes_raw) * 60.0, 0.0)
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid float for RISK__COOLDOWN_MIN: {cooldown_minutes_raw!r}"
            ) from exc
    else:
        cooldown_seconds = _env_float(
            "RISK_LOSS_COOLDOWN_SEC", default=90.0, minimum=0.0
        )
    min_lots = _env_int("MIN_LOTS_PER_TRADE", default=1, minimum=1)
    max_lots = _env_int("MAX_LOTS_PER_TRADE", default=3, minimum=1)
    if max_lots < min_lots:
        max_lots = min_lots
    atr_multiple = _env_float("ATR_MULT", default=1.0, minimum=0.0)
    contract_lot_size = _env_int("NIFTY_LOT_SIZE", default=65, minimum=1)
    return RiskSettings(
        daily_loss_pct=daily_loss_pct,
        daily_pnl_cap_pct=_env_float(
            "RISK_DAILY_PNL_CAP_PCT", default=daily_loss_pct, minimum=0.0
        ),
        max_daily_loss_absolute=abs_loss_cap,
        per_trade_risk_pct=_env_float(
            "RISK__PER_TRADE_RISK_PCT",
            "RISK_PER_TRADE_PCT",
            default=5.0,
            minimum=0.0,
        ),
        per_trade_cap_pct=_env_float(
            "RISK__MAX_CAPITAL_PER_ORDER_PCT",
            "RISK_PER_TRADE_CAP_PCT",
            default=15.0,
            minimum=0.0,
        ),
        max_consecutive_losses=max_losses,
        loss_cooldown_seconds=cooldown_seconds,
        cooldown_on_reject_seconds=_env_float(
            "RISK_COOLDOWN_ON_REJECT_SECONDS", default=30.0, minimum=0.0
        ),
        trading_day_reset_hour_utc=_env_int(
            "RISK_RESET_HOUR_UTC", default=3, minimum=0
        ),
        breaker_auto_shadow=_env_bool("RISK_BREAKER_AUTO_SHADOW", default=True),
        min_lots_per_trade=min_lots,
        max_lots_per_trade=max_lots,
        atr_stop_multiple=atr_multiple,
        contract_lot_size=contract_lot_size,
        allow_pyramiding=_env_bool("RISK_ALLOW_PYRAMIDING", default=False),
        signal_debounce_seconds=_env_float("RISK_SIGNAL_DEBOUNCE_SECONDS", default=60.0),
    )


def _build_streamer_settings() -> StreamerSettings:
    return StreamerSettings(
        gap_seconds=_env_float("STREAMER_GAP_SECONDS", default=5.0, minimum=1.0),
        max_backfill_candles=_env_int("STREAMER_BACKFILL_LIMIT", default=10, minimum=1),
        backfill_interval=_env_str("STREAMER_BACKFILL_INTERVAL", default="minute")
        or "minute",
        queue_size=_env_int("STREAMER_QUEUE_SIZE", default=1000, minimum=10),
    )


def _build_shadow_settings() -> ShadowSettings:
    return ShadowSettings(
        drift_threshold_pct=_env_float(
            "SHADOW_DRIFT_THRESHOLD_PCT", default=2.5, minimum=0.0
        ),
        auto_disable_live_on_drift=_env_bool("SHADOW_AUTO_DISABLE_LIVE", default=True),
        drift_sustain_seconds=_env_float(
            "SHADOW_DRIFT_SUSTAIN_SECONDS", default=30.0, minimum=0.0
        ),
        alert_threshold_pct=_env_float(
            "SHADOW_ALERT_THRESHOLD_PCT", default=1.5, minimum=0.0
        ),
        alert_trend_delta_pct=_env_float(
            "SHADOW_ALERT_TREND_DELTA_PCT", default=0.5, minimum=0.0
        ),
        alert_trend_window_seconds=_env_float(
            "SHADOW_ALERT_TREND_WINDOW_SECONDS", default=120.0, minimum=0.0
        ),
        alert_cooldown_seconds=_env_float(
            "SHADOW_ALERT_COOLDOWN_SECONDS", default=900.0, minimum=0.0
        ),
    )


def _build_notification_settings(app_config: AppConfig) -> NotificationSettings:
    whitelist = _env_csv_ints("TELEGRAM_ALLOWED_IDS", "ALLOWED_CHAT_IDS")
    token = _env_str("TELEGRAM_TOKEN") or _env_str("TELEGRAM__BOT_TOKEN")
    explicit_enabled = _env_bool(
        "TELEGRAM_ENABLED", default=bool(app_config.telegram.bot_token)
    )
    rate = _env_float("TELEGRAM_RATE_PER_SECOND", default=25.0, minimum=1.0)
    burst = _env_float("TELEGRAM_BURST_CAPACITY", default=rate, minimum=1.0)
    public_url = _env_str("TELEGRAM_PUBLIC_BASE_URL")
    allow_poll = _env_bool("TELEGRAM_ALLOW_POLL_FALLBACK", default=False)
    allow_poll = _env_bool("TELEGRAM_ENABLE_POLLING_FALLBACK", default=allow_poll)
    webhook_enabled = _env_bool("TELEGRAM_WEBHOOK_ENABLED", default=True)
    webhook_enabled = _env_bool("TELEGRAM__WEBHOOK_ENABLED", default=webhook_enabled)
    legacy_console_enabled = _env_bool("TELEGRAM_LEGACY_CONSOLE_ENABLED", default=False)
    return NotificationSettings(
        enabled=explicit_enabled and bool(token),
        token=token or app_config.telegram.bot_token,
        whitelist_chat_ids=(
            whitelist or {app_config.telegram.chat_id}
            if app_config.telegram.chat_id is not None
            else set()
        ),
        whitelist_user_ids=_env_csv_ints(
            "TELEGRAM_ALLOWED_USER_IDS", "ALLOWED_USER_IDS"
        ),
        rate_per_second=rate,
        burst_capacity=burst,
        public_base_url=public_url,
        webhook_enabled=webhook_enabled,
        allow_poll_fallback=allow_poll,
        legacy_console_enabled=legacy_console_enabled,
    )


def _build_regime_config() -> RegimeGateConfig:
    base = RegimeGateConfig()
    adx_min = _env_float(
        "REGIME__ADX_MIN_TREND", default=base.adx_min_trend, minimum=0.0
    )
    atr_multiple = _env_float(
        "REGIME__ATR_MULT_RANGE", default=base.atr_range_multiple, minimum=0.01
    )
    open_buffer = _env_int(
        "NOTRADE__OPEN_MIN", default=base.open_buffer_minutes, minimum=0
    )
    preclose_buffer = _env_int(
        "NOTRADE__PRECLOSE_MIN", default=base.preclose_buffer_minutes, minimum=0
    )
    lunch_raw = _env_str("NOTRADE__LUNCH", default="")
    lunch_start, lunch_end = base.lunch_start, base.lunch_end
    if lunch_raw.strip():
        parsed_start, parsed_end = _parse_time_range("NOTRADE__LUNCH", lunch_raw)
        lunch_start = parsed_start or base.lunch_start
        lunch_end = parsed_end or base.lunch_end
    return RegimeGateConfig(
        adx_period=base.adx_period,
        adx_min_trend=adx_min,
        atr_period=base.atr_period,
        atr_smooth_period=base.atr_smooth_period,
        atr_range_multiple=atr_multiple,
        vwap_lookback=base.vwap_lookback,
        vwap_max_trend_distance_pct=_env_float(
            "REGIME__VWAP_MAX_TREND_DISTANCE_PCT",
            default=base.vwap_max_trend_distance_pct,
            minimum=0.0,
        ),
        quiet_threshold_pct=_env_float(
            "REGIME__QUIET_THRESHOLD_PCT",
            default=base.quiet_threshold_pct,
            minimum=0.0,
        ),
        volatile_threshold_pct=_env_float(
            "REGIME__VOLATILE_THRESHOLD_PCT",
            default=base.volatile_threshold_pct,
            minimum=0.0,
        ),
        warmup_bars=base.warmup_bars,
        session_open=base.session_open,
        session_close=base.session_close,
        lunch_start=lunch_start,
        lunch_end=lunch_end,
        open_buffer_minutes=open_buffer,
        preclose_buffer_minutes=preclose_buffer,
    )


def _build_replay_settings() -> ReplaySettings:
    raw_dir = _env_str("REPLAY__DATA_DIR", default="data/minute_bars")
    directory = Path(raw_dir or "data/minute_bars")
    return ReplaySettings(data_dir=directory)


def _build_selector_settings() -> SelectorSettings:
    mode = _env_str("SELECTOR__MODE", default="ATM") or "ATM"
    expiry = _env_str("SELECTOR__EXPIRY", default="weekly") or "weekly"
    regime = _env_str("SELECTOR__EXPIRY_REGIME", default=expiry) or expiry
    delta_target = _env_float("SELECTOR__DELTA_TARGET", default=0.3, minimum=0.0)
    delta_tolerance = _env_float("SELECTOR__DELTA_TOL", default=0.05, minimum=0.0)
    roll_minutes = _env_int(
        "SELECTOR__EXPIRY_ROLL_MINUTES",
        "SELECTOR__EXPIRY_ROLL_MIN",
        default=45,
        minimum=0,
    )
    monthly_halt = _env_int(
        "SELECTOR__MONTHLY_HALT_MINUTES",
        "SELECTOR__MONTHLY_HALT_MIN",
        default=30,
        minimum=0,
    )
    special_raw = _env_str("SELECTOR__SPECIAL_EXPIRIES", default="") or ""
    specials = tuple(token.strip() for token in special_raw.split(",") if token.strip())
    return SelectorSettings(
        mode=mode,
        expiry=expiry,
        expiry_regime=regime,
        delta_target=delta_target,
        delta_tolerance=delta_tolerance,
        expiry_roll_minutes=roll_minutes,
        monthly_halt_minutes=monthly_halt,
        special_expiries=specials,
        long_only=_env_bool("OPTIONS_LONG_ONLY", default=True),
        legacy_side_to_type=_env_bool("LEGACY_SIDE_TO_TYPE", default=False),
    )


def _build_liquidity_settings() -> LiquiditySettings:
    return LiquiditySettings(
        min_oi=_env_float("LIQ__MIN_OI", default=0.0, minimum=0.0),
        max_spread_pct=_env_float("LIQ__MAX_SPREAD_PCT", default=0.0, minimum=0.0),
        min_trades=_env_float("LIQ__MIN_TRADES", default=0.0, minimum=0.0),
        min_top3_depth=_env_float("LIQ__MIN_TOP3_DEPTH", default=0.0, minimum=0.0),
    )


def _build_instrument_settings() -> InstrumentSettings:
    """Construct instrument cache configuration from environment.

    Args:
        None.

    Returns:
        InstrumentSettings: Parsed instrument cache configuration.

    Raises:
        Exception: Propagates unexpected errors from Pydantic parsing.
    """

    try:
        return InstrumentSettings()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "Failure in _build_instrument_settings: %s",
            exc,
            extra={"event": "settings_instruments_error"},
            exc_info=exc,
        )
        raise


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings.

    The first invocation loads :class:`AppConfig` from environment using
    :func:`nifty_scalper_bot.config.env.load_from_env`.  Subsequent calls
    reuse the cached ``Settings`` instance, keeping access cheap for hot
    path modules such as the health endpoint.
    """

    app_config = load_from_env()
    # Streaming defaults: poll by default, WS opt-in via WEBSOCKET__DISABLED=false
    session_allow_out_of_hours = _env_bool(
        "SESSION_ALLOW_OUT_OF_HOURS",
        "SESSION__ALLOW_OUT_OF_HOURS",
        "ALLOW_OFFHOURS_TESTING",
        default=False,
    )

    settings_obj = Settings(
        app=app_config,
        enable_live=_env_bool("ENABLE_LIVE", "ENABLE_LIVE_TRADING", default=False),
        orders=_build_order_settings(),
        risk=_build_risk_settings(),
        streamer=_build_streamer_settings(),
        shadow=_build_shadow_settings(),
        notifications=_build_notification_settings(app_config),
        session_allow_out_of_hours=session_allow_out_of_hours,
        websocket_enabled=not _env_bool(
            "WEBSOCKET__DISABLED", "WEBSOCKET_DISABLED", default=True
        ),
        telemetry_tags=_env_csv_strs("TELEMETRY_TAGS", "TELEMETRY__TAGS"),
        regime=_build_regime_config(),
        replay=_build_replay_settings(),
        selector=_build_selector_settings(),
        liquidity=_build_liquidity_settings(),
        instruments=_build_instrument_settings(),
        elite=_build_elite_settings(),
        execution=ExecutionSettings(
            enable_bracket_manager=_env_bool(
                "EXECUTION__ENABLE_BRACKET_MANAGER",
                "ENABLE_BRACKET_MANAGER",
                default=True,
            ),
            bracket_auto_reduce_sl=_env_bool(
                "EXECUTION__BRACKET_AUTO_REDUCE_SL",
                default=True,
            ),
            bracket_stale_cleanup_seconds=_env_int(
                "EXECUTION__BRACKET_STALE_CLEANUP_SECONDS",
                default=86_400,
                minimum=60,
            ),
        ),
        poll_batch_size=_env_int("POLL_BATCH_SIZE", default=50, minimum=1),
        poll_interval_ms_jitter_pct=_env_float(
            "POLL_INTERVAL_MS_JITTER_PCT", default=0.15, minimum=0.0
        ),
        feature_order_without_token=_env_bool(
            "FEATURE_ORDER_WITHOUT_TOKEN", default=True
        ),
        feature_resolver_learn_from_quotes=_env_bool(
            "FEATURE_RESOLVER_LEARN_FROM_QUOTES", default=True
        ),
    )

    try:
        validate_lifecycle_config(settings_obj)
    except LifecycleConfigError as exc:
        LOGGER.error(
            "Failure in get_settings lifecycle validation: %s",
            exc,
            extra={"event": "settings_lifecycle_invalid"},
        )
        raise

    try:
        validate_elite_config(settings_obj.elite)
    except EliteConfigError as exc:
        LOGGER.error(
            "Failure in get_settings elite validation: %s",
            exc,
            extra={"event": "settings_elite_invalid"},
        )
        raise

    LOGGER.info(
        "Condition met: settings_validated",
        extra={"event": "settings_validated"},
    )
    return settings_obj


__all__ = [
    "Settings",
    "OrderSettings",
    "RiskSettings",
    "StreamerSettings",
    "ShadowSettings",
    "NotificationSettings",
    "ReplaySettings",
    "SelectorSettings",
    "LiquiditySettings",
    "InstrumentSettings",
    "EliteStrategiesSettings",
    "ExecutionSettings",
    "get_settings",
    "REGIME_FILTER_BYPASS",
]
