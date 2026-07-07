"""Runtime context contract for live directional and quote-age strategy gates."""

from __future__ import annotations

from datetime import datetime, timezone
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


def normalise_live_direction_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        return {}

    preserved = {key: context[key] for key in _LIVE_DIRECTION_CONTEXT_KEYS if key in context}

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
