"""Live-entry preflight guard for broker and market truth.

This module is deliberately side-effect free. Runtime patching lives in
``core.live_entry_preflight_safety`` so tests and tools can evaluate the guard
without importing the full application.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

_UNUSABLE_TIMESTAMP_QUALITIES = {"synthetic", "unknown", "invalid"}
_UNACCEPTABLE_TIMESTAMP_SOURCES = {
    "",
    "unknown",
    "synthetic",
    "invalid",
    "received_at",
    "received_at_fallback",
    "received_ts",
    "received_time",
}
_ACCEPTABLE_TIMESTAMP_SOURCES = {
    "exchange_timestamp",
    "last_trade_time",
    "last_traded_time",
    "last_trade_timestamp",
    "exchange_update_time",
    "last_price_time",
    "timestamp",
    "broker_timestamp",
    "ts",
}


@dataclass(frozen=True, slots=True)
class SelectedOptionProof:
    """Market-data proof required before a selected option can be traded live."""

    symbol: str
    quote_present: bool = False
    quote_tradable: bool = False
    timestamp_quality: str | None = None
    timestamp_source: str | None = None
    candle_count: int = 0
    last_candle_ts: Any = None
    last_candle_close: float | None = None
    max_candle_age_seconds: float = 180.0
    candle_recent: bool | None = None
    now: Any = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        value = payload.get("last_candle_ts")
        if isinstance(value, datetime):
            payload["last_candle_ts"] = value.astimezone(timezone.utc).isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class LiveEntryPreflightSnapshot:
    """Broker + market truth required before live entries are allowed."""

    broker_positions_fetched: bool = False
    broker_orders_reconciled: bool = False
    local_positions_match_broker: bool = False
    selected_options: tuple[SelectedOptionProof, ...] = field(default_factory=tuple)
    context: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "broker_positions_fetched": self.broker_positions_fetched,
            "broker_orders_reconciled": self.broker_orders_reconciled,
            "local_positions_match_broker": self.local_positions_match_broker,
            "selected_options": [item.to_dict() for item in self.selected_options],
            "context": dict(self.context),
        }


@dataclass(frozen=True, slots=True)
class LiveEntryPreflightDecision:
    ready: bool
    blockers: tuple[str, ...]
    primary_blocker: str | None
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "blockers": list(self.blockers),
            "primary_blocker": self.primary_blocker,
            "details": dict(self.details),
        }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "ok", "ready", "matched"}


def _get(payload: Mapping[str, Any] | object | None, key: str, default: Any = None) -> Any:
    if payload is None:
        return default
    if isinstance(payload, Mapping):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _candle_recent(proof: SelectedOptionProof) -> bool:
    if proof.candle_recent is not None:
        return bool(proof.candle_recent)
    ts = _parse_ts(proof.last_candle_ts)
    if ts is None:
        return False
    now = _parse_ts(proof.now) or pd.Timestamp.now(tz=timezone.utc)
    return bool((now - ts).total_seconds() <= max(float(proof.max_candle_age_seconds), 0.0))


def quote_timestamp_source_acceptable(source: Any, quality: Any = None) -> bool:
    quality_text = str(quality or "").strip().lower()
    if quality_text in _UNUSABLE_TIMESTAMP_QUALITIES:
        return False
    source_text = str(source or "").strip().lower()
    if source_text in _UNACCEPTABLE_TIMESTAMP_SOURCES:
        return False
    if source_text in _ACCEPTABLE_TIMESTAMP_SOURCES:
        return True
    return bool(source_text.startswith("exchange") or source_text.startswith("last_trade"))


def _coerce_option_proof(payload: SelectedOptionProof | Mapping[str, Any] | object) -> SelectedOptionProof:
    if isinstance(payload, SelectedOptionProof):
        return payload
    return SelectedOptionProof(
        symbol=str(_get(payload, "symbol", "") or ""),
        quote_present=_truthy(_get(payload, "quote_present", _get(payload, "quote_available", True))),
        quote_tradable=_truthy(_get(payload, "quote_tradable", _get(payload, "tradable_quote", False))),
        timestamp_quality=_get(payload, "timestamp_quality", None),
        timestamp_source=_get(payload, "timestamp_source", _get(payload, "source", None)),
        candle_count=int(float(_get(payload, "candle_count", _get(payload, "bars", 0)) or 0)),
        last_candle_ts=_get(payload, "last_candle_ts", _get(payload, "last_bar_ts", None)),
        last_candle_close=_get(payload, "last_candle_close", _get(payload, "close", None)),
        max_candle_age_seconds=float(_get(payload, "max_candle_age_seconds", 180.0) or 180.0),
        candle_recent=_get(payload, "candle_recent", None),
        now=_get(payload, "now", None),
    )


def _coerce_snapshot(payload: LiveEntryPreflightSnapshot | Mapping[str, Any]) -> LiveEntryPreflightSnapshot:
    if isinstance(payload, LiveEntryPreflightSnapshot):
        return payload
    selected_raw = payload.get("selected_options") or payload.get("options") or []
    selected_options = tuple(_coerce_option_proof(item) for item in selected_raw)
    return LiveEntryPreflightSnapshot(
        broker_positions_fetched=_truthy(payload.get("broker_positions_fetched")),
        broker_orders_reconciled=_truthy(payload.get("broker_orders_reconciled")),
        local_positions_match_broker=_truthy(payload.get("local_positions_match_broker")),
        selected_options=selected_options,
        context=dict(payload.get("context") or {}),
    )


def evaluate_live_entry_preflight(snapshot: LiveEntryPreflightSnapshot | Mapping[str, Any]) -> LiveEntryPreflightDecision:
    """Return exact blockers that must prevent live entries."""

    snap = _coerce_snapshot(snapshot)
    blockers: list[str] = []
    option_details: list[dict[str, object]] = []

    if not snap.broker_positions_fetched:
        blockers.append("broker_positions_not_fetched")
    if not snap.broker_orders_reconciled:
        blockers.append("broker_orders_not_reconciled")
    if not snap.local_positions_match_broker:
        blockers.append("broker_position_mismatch")

    if not snap.selected_options:
        blockers.append("selected_option_candle_unproven")
    for proof in snap.selected_options:
        details = proof.to_dict()
        quote_ok = bool(proof.quote_present and proof.quote_tradable)
        timestamp_ok = quote_timestamp_source_acceptable(proof.timestamp_source, proof.timestamp_quality)
        candle_ok = bool(proof.candle_count > 0 and proof.last_candle_close is not None and _candle_recent(proof))
        details.update({"quote_ok": quote_ok, "timestamp_ok": timestamp_ok, "candle_ok": candle_ok})
        option_details.append(details)
        if not quote_ok:
            blockers.append("selected_option_quote_missing")
        if not timestamp_ok:
            blockers.append("selected_option_timestamp_unusable")
        if not candle_ok:
            blockers.append("selected_option_candle_unproven")

    ordered = tuple(dict.fromkeys(blockers))
    return LiveEntryPreflightDecision(
        ready=not ordered,
        blockers=ordered,
        primary_blocker=ordered[0] if ordered else None,
        details={
            **snap.context,
            "broker_positions_fetched": snap.broker_positions_fetched,
            "broker_orders_reconciled": snap.broker_orders_reconciled,
            "local_positions_match_broker": snap.local_positions_match_broker,
            "selected_options": option_details,
        },
    )


__all__ = [
    "LiveEntryPreflightDecision",
    "LiveEntryPreflightSnapshot",
    "SelectedOptionProof",
    "evaluate_live_entry_preflight",
    "quote_timestamp_source_acceptable",
]
