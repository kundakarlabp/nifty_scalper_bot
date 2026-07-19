"""Compatibility installer for native CandleEngine state invariants."""

from __future__ import annotations

import logging
from typing import Any, Literal, Mapping

import pandas as pd

LOGGER = logging.getLogger(__name__)
_INSTALLED_ATTR = "_candle_state_hardening_installed"
_NATIVE_ATTR = "_candle_state_native_invariants_active"
_REQUIRED_NATIVE_APIS = (
    "reconcile_current_with_finalized",
    "is_state_consistent",
    "latest_finalized_minute",
)


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("Asia/Kolkata")
    return timestamp.tz_convert("Asia/Kolkata")


def _latest_completed_minute(engine: Any) -> pd.Timestamp:
    native = getattr(engine, "latest_finalized_minute", None)
    if callable(native):
        latest = native()
        return pd.NaT if latest is None else _coerce_timestamp(latest)
    completed = getattr(engine, "_completed_candles", None)
    if not completed:
        return pd.NaT
    try:
        return _coerce_timestamp(completed[-1].get("timestamp"))
    except Exception:
        return pd.NaT


def _current_minute(engine: Any) -> pd.Timestamp:
    current = getattr(engine, "current_candle", None)
    if not isinstance(current, Mapping):
        return pd.NaT
    return _coerce_timestamp(current.get("timestamp"))


def _lock_for(engine: Any) -> Any:
    """Compatibility alias returning the native per-engine lock when present."""
    return getattr(engine, "_lock", _NullContext())


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> Literal[False]:
        return False


def reconcile_stale_current(engine: Any, *, reason: str) -> bool:
    """Delegate stale-current reconciliation to native CandleEngine logic."""
    native = getattr(engine, "reconcile_current_with_finalized", None)
    if not callable(native):
        raise AttributeError("CandleEngine lacks native reconciliation support")
    mapped_reason = "flush" if reason == "clock_flush" else reason
    return bool(native(reason=mapped_reason))


def _state_consistent(engine: Any) -> bool:
    native = getattr(engine, "is_state_consistent", None)
    if callable(native):
        return bool(native())
    return False


def install_candle_state_hardening(engine_cls: type[Any]) -> None:
    """Mark native CandleEngine invariants active without monkey-patching methods."""
    if bool(getattr(engine_cls, _INSTALLED_ATTR, False)):
        return
    missing = [
        name
        for name in _REQUIRED_NATIVE_APIS
        if not callable(getattr(engine_cls, name, None))
    ]
    if missing:
        raise AttributeError(
            "CandleEngine lacks native state hardening APIs: " + ", ".join(missing)
        )
    setattr(engine_cls, _NATIVE_ATTR, True)
    setattr(engine_cls, _INSTALLED_ATTR, True)
    LOGGER.info(
        "candle_state_hardening_native_active",
        extra={"event": "candle_state_hardening_native_active"},
    )


__all__ = [
    "install_candle_state_hardening",
    "reconcile_stale_current",
    "_current_minute",
    "_latest_completed_minute",
    "_lock_for",
    "_state_consistent",
]
