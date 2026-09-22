"""Focused runtime hardening for reliability defects observed on 2026-08-07.

No trading threshold, order path, risk control, market-hours guard, or strategy
permission is changed here. The adapters correct overload age attribution,
bounded consensus-quality evidence, tick hot-path duplication, and runtime
diagnostics while preserving fail-closed behavior for entry-critical queues.
"""

from __future__ import annotations

import time
from datetime import datetime
from functools import wraps
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.symbols import normalize_symbol

_LOG = get_logger(__name__)
_PATCH_ATTR = "_runtime_reliability_hardening_installed"
_UNUSABLE_TIMESTAMP_QUALITIES = {"synthetic", "unknown", "invalid"}


def _critical_oldest_pending_age_ms_locked(mdm: Any) -> float:
    """Return oldest entry-critical queue age.

    Non-selected active-basket ``near_atm`` options remain in normal FIFO
    queues so every tick still reaches CandleEngine/OHLC processing. Their age
    is optional strategy context, however, and must not by itself disarm fresh
    selected CE/PE execution. Selected options, spot/future context, open
    positions, and any unclassified normal queue remain age-critical. Global
    pending-count overload remains unchanged and still covers every lane.
    """

    oldest_mono: float | None = None
    for queue in (getattr(mdm, "_pending_tick_queues", {}) or {}).values():
        if not queue:
            continue
        tick = queue[0]
        if not isinstance(tick, Mapping):
            return float(
                getattr(mdm, "_overload_enter_oldest_ms", 2000.0) or 2000.0
            )
        bucket = str(tick.get("_mdm_priority_bucket") or "").strip().lower()
        if bucket == "near_atm":
            continue
        timestamp = tick.get("_mdm_enqueued_mono")
        if not isinstance(timestamp, (int, float)):
            return float(
                getattr(mdm, "_overload_enter_oldest_ms", 2000.0) or 2000.0
            )
        candidate = float(timestamp)
        oldest_mono = (
            candidate if oldest_mono is None else min(oldest_mono, candidate)
        )
    if oldest_mono is None:
        return 0.0
    return max(0.0, (time.monotonic() - oldest_mono) * 1000.0)


def _install_mdm_overload_patch() -> bool:
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

    if bool(getattr(MarketDataManager, _PATCH_ATTR, False)):
        return True

    def _update_pipeline_overload_locked(self: Any) -> None:
        pending = self._pending_count_locked()
        total_oldest_ms = self._oldest_pending_age_ms_locked()
        critical_oldest_ms = _critical_oldest_pending_age_ms_locked(self)
        if not self._pipeline_overloaded:
            if (
                pending >= self._overload_enter_pending
                or critical_oldest_ms >= self._overload_enter_oldest_ms
            ):
                self._pipeline_overloaded = True
                self._overload_since_mono = time.monotonic()
                self._logger.warning(
                    "DATA_PIPELINE_OVERLOAD_ENTER pending_ticks=%d oldest_pending_age_ms=%.0f "
                    "critical_oldest_pending_age_ms=%.0f enter_pending=%d enter_oldest_ms=%.0f",
                    pending,
                    total_oldest_ms,
                    critical_oldest_ms,
                    self._overload_enter_pending,
                    self._overload_enter_oldest_ms,
                    extra={
                        "event": "DATA_PIPELINE_OVERLOAD_ENTER",
                        "pending_ticks": pending,
                        "oldest_pending_age_ms": total_oldest_ms,
                        "critical_oldest_pending_age_ms": critical_oldest_ms,
                        "enter_pending": self._overload_enter_pending,
                        "enter_oldest_ms": self._overload_enter_oldest_ms,
                    },
                )
        elif (
            pending <= self._overload_exit_pending
            and critical_oldest_ms <= self._overload_exit_oldest_ms
        ):
            duration = 0.0
            if self._overload_since_mono is not None:
                duration = max(0.0, time.monotonic() - self._overload_since_mono)
            self._pipeline_overloaded = False
            self._overload_since_mono = None
            self._logger.warning(
                "DATA_PIPELINE_OVERLOAD_RECOVERED pending_ticks=%d oldest_pending_age_ms=%.0f "
                "critical_oldest_pending_age_ms=%.0f overloaded_for_s=%.1f",
                pending,
                total_oldest_ms,
                critical_oldest_ms,
                duration,
                extra={
                    "event": "DATA_PIPELINE_OVERLOAD_RECOVERED",
                    "pending_ticks": pending,
                    "oldest_pending_age_ms": total_oldest_ms,
                    "critical_oldest_pending_age_ms": critical_oldest_ms,
                    "overloaded_for_s": duration,
                },
            )

    MarketDataManager._update_pipeline_overload_locked = (  # type: ignore[method-assign]
        _update_pipeline_overload_locked
    )
    setattr(MarketDataManager, _PATCH_ATTR, True)
    return True


def _runtime_tick_timestamp_ms(payload: Mapping[str, Any]) -> float | None:
    """Return MDM runtime timestamp milliseconds without pandas reparsing."""

    timestamp_ms = payload.get("timestamp_ms")
    if isinstance(timestamp_ms, (int, float)) and not isinstance(timestamp_ms, bool):
        value = float(timestamp_ms)
        return value if value > 0 else None
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
        value = float(timestamp.timestamp() * 1000.0)
        return value if value > 0 else None
    return None


def _is_canonical_runtime_tick(payload: Mapping[str, Any]) -> bool:
    """Return whether MDM already produced the complete live-tick contract."""

    symbol = str(payload.get("symbol") or "").strip()
    if not symbol or normalize_symbol(symbol) != symbol:
        return False
    token = payload.get("instrument_token") or payload.get("token")
    price = payload.get("ltp") or payload.get("last_price")
    timestamp_ms = _runtime_tick_timestamp_ms(payload)
    try:
        if int(token) <= 0 or float(price) <= 0 or timestamp_ms is None:
            return False
    except (TypeError, ValueError):
        return False
    source = str(payload.get("source") or "").strip().lower()
    if source not in {
        "ws",
        "ws_full",
        "websocket",
        "stream",
        "poll",
        "rest",
        "rest_poll",
        "fallback",
        "quote",
        "rest_quote",
    }:
        return False
    explicit_quality = str(payload.get("timestamp_quality") or "").strip().lower()
    if explicit_quality in _UNUSABLE_TIMESTAMP_QUALITIES:
        return False
    if payload.get("source_timestamp_valid") is not True:
        return False
    return all(
        key in payload
        for key in (
            "timestamp",
            "timestamp_source",
            "received_at",
            "depth_available",
            "tradable_quote",
        )
    )


def _install_datahub_tick_hotpath_patch() -> bool:
    """Avoid rebuilding MDM's already-canonical tick before Runner dispatch."""

    from nifty_scalper_bot.data.data_hub import DataHub

    attr = "_mdm_tick_hotpath_hardening_installed"
    if bool(getattr(DataHub, attr, False)):
        return True
    original = DataHub._canonicalize_tick_payload

    @wraps(original)
    def _canonicalize_tick_payload(
        self: Any, payload: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if _is_canonical_runtime_tick(payload):
            tick = dict(payload)
            symbol = str(tick["symbol"])
            timestamp_ms = _runtime_tick_timestamp_ms(tick)
            if timestamp_ms is None:
                return original(self, payload)
            timestamp = tick.get("timestamp")
            if isinstance(timestamp, datetime):
                tick["timestamp"] = timestamp.isoformat()
            tick["timestamp_ms"] = timestamp_ms
            timestamp_source = str(tick.get("timestamp_source") or "").lower()
            timestamp_quality = str(tick.get("timestamp_quality") or "").lower()
            if not timestamp_quality:
                timestamp_quality = (
                    "exchange" if "exchange" in timestamp_source else "broker"
                )
                tick["timestamp_quality"] = timestamp_quality
            tick.setdefault("hard_readiness_eligible", True)
            tick.setdefault("quote_source", tick.get("source"))
            tick.setdefault("exchange_symbol", symbol)
            tick.setdefault("quote_identity_timestamp_source", timestamp_quality)
            return tick
        return original(self, payload)

    DataHub._canonicalize_tick_payload = (  # type: ignore[method-assign]
        _canonicalize_tick_payload
    )
    setattr(DataHub, attr, True)
    return True


def apply_patches() -> dict[str, bool]:
    """Install focused reliability adapters idempotently."""

    state = {
        "mdm_overload": _install_mdm_overload_patch(),
        "datahub_tick_hotpath": _install_datahub_tick_hotpath_patch(),
    }
    if not all(state.values()):
        raise RuntimeError(f"runtime_reliability_hardening_incomplete state={state}")
    _LOG.info(
        "RUNTIME_RELIABILITY_HARDENING_INSTALLED state=%s",
        state,
        extra={"event": "RUNTIME_RELIABILITY_HARDENING_INSTALLED", **state},
    )
    return state


__all__ = [
    "apply_patches",
    "_critical_oldest_pending_age_ms_locked",
    "_is_canonical_runtime_tick",
    "_runtime_tick_timestamp_ms",
]
