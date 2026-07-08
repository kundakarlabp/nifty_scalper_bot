"""Runtime context contract for live directional and quote-age strategy gates."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from typing import Any, Mapping


_LIVE_DIRECTION_CONTEXT_KEYS = frozenset(
    {
        "direction_bias",
        "underlying_direction",
        "direction_context",
        "direction_source",
        "context_source",
        "context_age_seconds",
        "context_timestamp",
        "direction_context_timestamp",
        "direction_updated_at",
        "spot_fresh",
        "fut_fresh",
        "futures_fresh",
        "spot_age_seconds",
        "futures_age_seconds",
        "spot_tick_age_s",
        "futures_tick_age_s",
        "underlying_symbol",
        "tick_age_ms",
        "tick_age_s",
        "quote_age_ms",
        "quote_age_s",
        "last_tick_age_ms",
        "last_tick_age_s",
        "market_data_age_ms",
        "market_data_age_s",
        "quote_update_version",
        "update_version",
        "tick_version",
        "last_tick_ts_ms",
        "timestamp_ms",
        "last_tick_timestamp",
        "real_ticks_last_60s",
        "tick_count_60s",
        "recent_real_tick_count",
        "quote_depth_valid",
        "quote_readiness_allowed",
        "quote_readiness_reason",
    }
)

_DIRECTION_ALIAS_KEYS = (
    "direction_bias",
    "underlying_direction",
    "direction_context",
)
_TRUTHY = {"1", "true", "yes", "y", "on"}


def _runtime_context_max_age_seconds(default: float = 5.0) -> float:
    try:
        return max(float(os.getenv("ORDERFLOW_MAX_CONTEXT_AGE_SECONDS", str(default)) or default), 0.0)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in _TRUTHY:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _coerce_direction(value: Any) -> str | None:
    raw = str(value or "").strip().upper()
    if raw in {"CE", "CALL", "CALLS", "BULL", "BULLISH", "UP", "LONG_CE"}:
        return "CE"
    if raw in {"PE", "PUT", "PUTS", "BEAR", "BEARISH", "DOWN", "LONG_PE"}:
        return "PE"
    return raw if raw in {"CE", "PE"} else None


def _coerce_age_seconds(value: Any) -> float | None:
    try:
        age = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if age != age or age < 0:
        return None
    return age


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, (int, float)):
        try:
            raw = float(value)
            ts = datetime.fromtimestamp(raw / 1000.0 if raw > 1e12 else raw, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _derive_freshness_from_age(context: Mapping[str, Any], *keys: str, max_age_seconds: float) -> bool | None:
    for key in keys:
        age = _coerce_age_seconds(context.get(key))
        if age is not None:
            return age <= max_age_seconds
    return None


def live_direction_context_has_proof(context: Mapping[str, Any], *, max_age_seconds: float | None = None) -> bool:
    """Return True if live spot/futures context has explicit fresh proof.

    This does not invent a CE/PE direction. It only confirms that the missing
    direction is not due to absent/stale underlying context. Live trigger logic
    can then decide whether its own microstructure evidence is strong enough.
    """

    if not isinstance(context, Mapping):
        return False
    max_age = _runtime_context_max_age_seconds() if max_age_seconds is None else max(float(max_age_seconds), 0.0)
    spot_fresh = _coerce_bool(context.get("spot_fresh"))
    fut_fresh = _coerce_bool(context.get("fut_fresh"))
    futures_fresh = _coerce_bool(context.get("futures_fresh"))
    if fut_fresh is None:
        fut_fresh = futures_fresh
    derived_spot = _derive_freshness_from_age(context, "spot_age_seconds", "spot_tick_age_s", max_age_seconds=max_age)
    derived_fut = _derive_freshness_from_age(context, "futures_age_seconds", "futures_tick_age_s", max_age_seconds=max_age)
    if spot_fresh is None:
        spot_fresh = derived_spot
    if fut_fresh is None:
        fut_fresh = derived_fut
    context_age = _coerce_age_seconds(context.get("context_age_seconds"))
    context_age_ok = context_age is None or context_age <= max_age
    return bool(context_age_ok and (spot_fresh or fut_fresh))


def normalise_live_direction_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        return {}

    preserved = {key: context[key] for key in _LIVE_DIRECTION_CONTEXT_KEYS if key in context}
    max_age = _runtime_context_max_age_seconds()

    if "direction_bias" not in preserved:
        for key in _DIRECTION_ALIAS_KEYS:
            direction = _coerce_direction(context.get(key))
            if direction is not None:
                preserved["direction_bias"] = direction
                break
    else:
        direction = _coerce_direction(preserved.get("direction_bias"))
        if direction is not None:
            preserved["direction_bias"] = direction

    derived_spot = _derive_freshness_from_age(context, "spot_age_seconds", "spot_tick_age_s", max_age_seconds=max_age)
    derived_fut = _derive_freshness_from_age(context, "futures_age_seconds", "futures_tick_age_s", max_age_seconds=max_age)
    if "spot_fresh" not in preserved and derived_spot is not None:
        preserved["spot_fresh"] = derived_spot
    if "fut_fresh" not in preserved and derived_fut is not None:
        preserved["fut_fresh"] = derived_fut
    if "futures_fresh" in preserved and "fut_fresh" not in preserved:
        preserved["fut_fresh"] = bool(preserved["futures_fresh"])
    if "fut_fresh" in preserved and "futures_fresh" not in preserved:
        preserved["futures_fresh"] = bool(preserved["fut_fresh"])

    age = _coerce_age_seconds(preserved.get("context_age_seconds"))
    if age is not None:
        preserved["context_age_seconds"] = age
    else:
        for key in ("context_timestamp", "direction_context_timestamp", "direction_updated_at"):
            ts = _coerce_timestamp(context.get(key))
            if ts is not None:
                preserved["context_age_seconds"] = max(0.0, time.time() - ts.timestamp())
                break
    preserved["live_direction_context_proof"] = live_direction_context_has_proof(preserved, max_age_seconds=max_age)

    return preserved


def install_indicator_runtime_context_contract() -> bool:
    try:
        from nifty_scalper_bot.strategies.indicators import IndicatorEngine
    except Exception:
        return False

    current = getattr(IndicatorEngine, "set_runtime_context", None)
    if current is None or getattr(current, "_live_direction_contract_installed", False):
        return False

    original = current

    def set_runtime_context(self: Any, symbol: str, context: Mapping[str, Any], *, merge: bool = True) -> None:
        original(self, symbol, context, merge=merge)
        extras = normalise_live_direction_context(context or {})
        if not symbol or not extras:
            return
        try:
            with self._lock:
                existing = self._runtime_context.setdefault(symbol, {}) if merge else dict(self._runtime_context.get(symbol, {}) or {})
                existing.update(extras)
                self._runtime_context[symbol] = existing
                self._cache.pop(symbol, None)
        except Exception as exc:
            logger = getattr(self, "_logger", None)
            if logger is not None:
                logger.error("LIVE_DIRECTION_CONTEXT_PRESERVE_FAILED symbol=%s error=%s", symbol, exc, extra={"event": "LIVE_DIRECTION_CONTEXT_PRESERVE_FAILED", "symbol": symbol, "error": str(exc)})
            raise

    set_runtime_context.__name__ = getattr(original, "__name__", "set_runtime_context")
    set_runtime_context.__doc__ = getattr(original, "__doc__", None)
    setattr(set_runtime_context, "_live_direction_contract_installed", True)
    setattr(set_runtime_context, "_original", original)
    IndicatorEngine.set_runtime_context = set_runtime_context  # type: ignore[assignment]
    return True
