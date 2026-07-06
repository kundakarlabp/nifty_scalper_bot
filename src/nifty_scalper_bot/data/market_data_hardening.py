"""Runtime hardening hooks for MarketDataManager freshness handling."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Mapping

import pandas as pd

_INSTALLED_ATTR = "_freshness_hardening_installed"
_ORIGINAL_NORMALIZE_ATTR = "_freshness_hardening_original_normalize_ws_tick"
_ORIGINAL_FAST_RECORD_ATTR = "_freshness_hardening_original_record_ws_arrival_fast"

_SYNTHETIC_QUALITIES = {"synthetic", "unknown", "invalid"}
_ALLOWED_WS_SOURCES = {"ws", "ws_full", "full"}


def install_market_data_manager_hardening(manager_cls: type[Any]) -> None:
    """Install idempotent MDM freshness hardening hooks."""
    if bool(getattr(manager_cls, _INSTALLED_ATTR, False)):
        return

    original_normalize = getattr(manager_cls, "_normalize_ws_tick", None)
    if callable(original_normalize):
        setattr(manager_cls, _ORIGINAL_NORMALIZE_ATTR, original_normalize)

        def _normalize_ws_tick_with_quality(self: Any, raw: dict[str, Any]) -> dict[str, Any] | None:
            quality = _timestamp_quality(raw)
            normalized = original_normalize(self, raw)
            if normalized is not None:
                normalized["timestamp_quality"] = quality
            return normalized

        setattr(manager_cls, "_normalize_ws_tick", _normalize_ws_tick_with_quality)

    original_fast_record = getattr(manager_cls, "_record_ws_arrival_fast", None)
    if callable(original_fast_record):
        setattr(manager_cls, _ORIGINAL_FAST_RECORD_ATTR, original_fast_record)

        def _record_ws_arrival_fast_with_quality(self: Any, *, symbol: str, token: int | None, ltp: Any, raw_tick: dict[str, Any]) -> None:
            quality = _timestamp_quality(raw_tick)
            original_fast_record(self, symbol=symbol, token=token, ltp=ltp, raw_tick=raw_tick)
            _tag_fast_cache_quality(self, symbol=symbol, token=token, quality=quality)

        setattr(manager_cls, "_record_ws_arrival_fast", _record_ws_arrival_fast_with_quality)

    setattr(manager_cls, "has_fresh_ws_ltp", _has_fresh_ws_ltp_strict)
    setattr(manager_cls, _INSTALLED_ATTR, True)


def _timestamp_quality(raw: Mapping[str, Any]) -> str:
    """Classify source timestamp quality before any synthetic fallback."""
    if _valid_timestamp(raw.get("exchange_timestamp")):
        return "exchange"
    if _valid_timestamp(raw.get("timestamp")):
        return "broker"
    if _valid_timestamp(raw.get("received_at")):
        return "received_at"
    return "synthetic"


def _valid_timestamp(value: Any) -> bool:
    if value is None or value == "":
        return False
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return False
    if pd.isna(ts):
        return False
    try:
        return int(ts.year) >= 2020
    except Exception:
        return False


def _tag_fast_cache_quality(self: Any, *, symbol: str, token: int | None, quality: str) -> None:
    try:
        canonical = self._canonical_symbol(symbol)
    except Exception:
        canonical = str(symbol or "")
    keys = {canonical}
    try:
        if self._is_nifty_spot_tick(canonical, token):
            keys.add("NSE:NIFTY")
    except Exception:
        pass
    lock = getattr(self, "_lock", None)
    if lock is None:
        return
    with lock:
        for key in keys:
            tick = self._latest_ticks.get(key)
            if isinstance(tick, dict):
                tick["timestamp_quality"] = quality
            cached = self._tick_cache.get(key)
            if isinstance(cached, dict):
                cached["timestamp_quality"] = quality


def _tick_age_seconds(tick_ts: Any, now: float) -> float | None:
    if isinstance(tick_ts, datetime):
        ts = tick_ts if tick_ts.tzinfo is not None else tick_ts.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - ts).total_seconds(), 0.0)
    if tick_ts is None:
        return None
    try:
        raw = float(tick_ts)
    except (TypeError, ValueError):
        try:
            parsed = pd.to_datetime(tick_ts, utc=True, errors="coerce")
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        return max(now - float(pd.Timestamp(parsed).timestamp()), 0.0)
    if raw > 1e12:
        raw /= 1000.0
    if raw < 946684800:
        return None
    return max(now - raw, 0.0)


def _has_fresh_ws_ltp_strict(self: Any, symbols: list[str] | tuple[str, ...] | None = None, max_age_seconds: float = 5.0) -> bool:
    """Return fresh WS LTP proof only when timestamp quality is acceptable."""
    now = time.time()
    max_age = max(float(max_age_seconds), 0.1)
    with self._lock:
        candidate_symbols = (
            [self._canonical_symbol(sym) for sym in symbols]
            if symbols is not None
            else list(self._active_subscribed_symbols)
        )
        if not candidate_symbols:
            candidate_symbols = list(self._latest_ticks.keys())
        for symbol in candidate_symbols:
            tick = self._latest_ticks.get(symbol)
            if not isinstance(tick, Mapping):
                continue
            source = str(tick.get("source") or self._last_tick_source.get(symbol) or "").lower()
            if source not in _ALLOWED_WS_SOURCES:
                continue
            quality = str(tick.get("timestamp_quality") or "").lower()
            if quality in _SYNTHETIC_QUALITIES:
                continue
            price = tick.get("ltp", tick.get("last_price", tick.get("price", 0)))
            try:
                ltp = float(price)
            except (TypeError, ValueError):
                continue
            if ltp <= 0:
                continue
            age = _tick_age_seconds(tick.get("exchange_timestamp") or tick.get("timestamp"), now)
            if age is not None and age <= max_age:
                return True
    return False
