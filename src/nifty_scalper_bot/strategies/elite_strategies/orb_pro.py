from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from nifty_scalper_bot.config.regime_ontology import MarketRegime, normalize_regime
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
)
from nifty_scalper_bot.strategies.setup_lifecycle import SetupStage, transition_setup
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)
_INDIA_TZ = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN = time(hour=9, minute=15)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default)) or default))
    except (TypeError, ValueError):
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (int, float)):
        number = float(value)
        if number > 10_000_000_000:
            number /= 1000.0
        result = datetime.fromtimestamp(number, tz=timezone.utc)
    elif isinstance(value, str) and value.strip():
        try:
            result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc).replace(microsecond=0)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class ORBProStrategy(EliteStrategy):
    """Underlying-led NIFTY opening-range breakout trigger.

    Spot/futures remain context-only. The strategy evaluates the selected option
    symbol but derives opening-range structure from already-hydrated NIFTY
    futures history, with spot as a fail-safe context fallback. It never selects
    or executes an underlying instrument.
    """

    MIN_BARS_REQUIRED = 5
    ROLE = "trigger"

    def __init__(self, config: ORBProStrategyConfig, indicator_engine: Any) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._cfg = config
        self._events: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._last_bar_by_key: dict[tuple[str, str, str], datetime] = {}
        self._event_count_by_key: dict[tuple[str, str, str], int] = {}

    def get_required_indicators(self) -> set[str]:
        """Return strategy-facing option and underlying context requirements."""
        return {
            "close",
            "open",
            "high",
            "low",
            "atr",
            "direction_bias",
            "underlying_direction_bias",
            "regime",
            "spread_pct",
            "quote_depth_valid",
            "tradable_quote",
            "stale_data_used",
            "futures_symbol",
            "futures_price",
            "spot_symbol",
            "spot_price",
            "futures_vwap_slope",
            # Legacy option ORB is retained only for diagnostics/backward telemetry.
            "orb_high",
            "orb_low",
            "orb_ready",
        }

    def _read_completed_bars(self, symbol: str) -> list[dict[str, Any]]:
        engine = self._indicator_engine
        if engine is None or not symbol or not hasattr(engine, "get_history"):