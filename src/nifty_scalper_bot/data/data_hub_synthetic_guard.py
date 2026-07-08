"""Synthetic timestamp guard for DataHub quotes.

DataHub is a read facade. It may retain quotes with missing or malformed broker
timestamps for display and diagnostics, but those quotes must not become hard
live-readiness proof simply because the facade synthesized ``now()``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

_PATCH_ATTR = "_synthetic_timestamp_guard_installed"
_ORIGINALS_ATTR = "_synthetic_timestamp_guard_originals"
_UNUSABLE_TIMESTAMP_QUALITIES = {"synthetic", "unknown", "invalid"}
_WS_SOURCES = {"ws", "websocket", "stream"}
LOGGER = logging.getLogger(__name__)


def _valid_timestamp_ms(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, datetime):
            ts = pd.Timestamp(value)
        elif isinstance(value, (int, float)):
            raw = float(value)
            if raw <= 0:
                return None
            seconds = raw / 1000.0 if raw > 1e11 else raw
            ts = pd.Timestamp(datetime.fromtimestamp(seconds, tz=timezone.utc))
        else:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts) or getattr(ts, "year", 1970) < 2020:
            return None
        return float(pd.Timestamp(ts).timestamp() * 1000.0)
    except Exception:
        return None


def _timestamp_quality(payload: Mapping[str, Any]) -> str:
    explicit = str(payload.get("timestamp_quality") or "").strip().lower()
    if explicit:
        return explicit
    if _valid_timestamp_ms(payload.get("exchange_timestamp")) is not None:
        return "exchange"
    if _valid_timestamp_ms(payload.get("timestamp")) is not None:
        return "broker"
    if _valid_timestamp_ms(payload.get("received_at")) is not None:
        return "received_at"
    if any(key in payload for key in ("exchange_timestamp", "timestamp", "received_at")):
        return "invalid"
    return "synthetic"


def _quote_getter(payload: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return payload.get(key, default)
    except AttributeError:
        return default


def _quote_timestamp_ms(payload: Mapping[str, Any]) -> float | None:
    for key in ("timestamp_ms", "exchange_timestamp", "timestamp", "received_at"):
        value = _quote_getter(payload, key)
        if key == "timestamp_ms":
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and parsed > 0:
                return parsed if parsed > 1e11 else parsed * 1000.0
        parsed = _valid_timestamp_ms(value)
        if parsed is not None:
            return parsed
    return None


def _tick_price(tick: Mapping[str, Any]) -> float | None:
    for key in ("ltp", "last_price", "price", "close"):
        value = tick.get(key)
        try:
            if value is not None:
                parsed = float(value)
                if parsed > 0:
                    return parsed
        except (TypeError, ValueError):
            continue
    return None


def install_datahub_synthetic_timestamp_guard(datahub_cls: type[Any]) -> bool:
    """Install DataHub timestamp-quality wrappers once."""

    if bool(getattr(datahub_cls, _PATCH_ATTR, False)):
        return False
    originals = {
        "store_quote": getattr(datahub_cls, "store_quote"),
        "_canonicalize_tick_payload": getattr(datahub_cls, "_canonicalize_tick_payload"),
        "get_cached_ltp": getattr(datahub_cls, "get_cached_ltp"),
    }

    def store_quote(self: Any, symbol: str, quote_data: dict[str, Any], source: str = "ws", seed: bool = False) -> None:
        payload = dict(quote_data or {})
        if not any(payload.get(key) not in (None, "") for key in ("exchange_timestamp", "timestamp", "received_at", "timestamp_quality")):
            payload["timestamp_quality"] = "synthetic"
            payload["synthetic_timestamp"] = True
        return originals["store_quote"](self, symbol, payload, source=source, seed=seed)

    def _canonicalize_tick_payload(self: Any, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        input_payload = dict(payload or {})
        quality = _timestamp_quality(input_payload)
        tick = originals["_canonicalize_tick_payload"](self, input_payload)
        if tick is None:
            return None
        tick["timestamp_quality"] = quality
        if quality in _UNUSABLE_TIMESTAMP_QUALITIES:
            tick["synthetic_timestamp"] = True
            tick["hard_readiness_eligible"] = False
            tick["tradable_quote"] = False
        else:
            tick.setdefault("hard_readiness_eligible", True)
        return tick

    def get_cached_ltp(
        self: Any,
        symbol: str,
        *,
        max_age_seconds: float | None = None,
        require_ws: bool = False,
    ) -> float | None:
        mdm_cached = getattr(getattr(self, "_mdm", None), "get_cached_ltp", None)
        if callable(mdm_cached):
            return mdm_cached(symbol, max_age_seconds=max_age_seconds, require_ws=require_ws)
        quote = self.get_quote(symbol, allow_pull=False)
        if not quote:
            return None
        quality = str(quote.get("timestamp_quality") or "").strip().lower()
        guarded = bool(require_ws or max_age_seconds is not None)
        if guarded and quality in _UNUSABLE_TIMESTAMP_QUALITIES:
            return None
        if require_ws and str(quote.get("source") or "").strip().lower() not in _WS_SOURCES:
            return None
        if max_age_seconds is not None:
            ts_ms = _quote_timestamp_ms(quote)
            if ts_ms is None:
                return None
            age_s = max(0.0, time.time() - (ts_ms / 1000.0))
            if age_s > max(0.0, float(max_age_seconds)):
                return None
        return _tick_price(quote)

    setattr(datahub_cls, "store_quote", store_quote)
    setattr(datahub_cls, "_canonicalize_tick_payload", _canonicalize_tick_payload)
    setattr(datahub_cls, "get_cached_ltp", get_cached_ltp)
    setattr(datahub_cls, _ORIGINALS_ATTR, originals)
    setattr(datahub_cls, _PATCH_ATTR, True)
    LOGGER.info(
        "DATAHUB_SYNTHETIC_TIMESTAMP_GUARD_INSTALLED",
        extra={"event": "DATAHUB_SYNTHETIC_TIMESTAMP_GUARD_INSTALLED"},
    )
    return True


__all__ = ["install_datahub_synthetic_timestamp_guard"]
