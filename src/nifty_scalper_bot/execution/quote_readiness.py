"""Canonical option-quote readiness evaluation for live execution gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from nifty_scalper_bot.execution.readiness import (
    quote_timestamp_quality_allows_hard_readiness,
    resolve_quote_bid_ask_spread,
)


def _value(payload: Mapping[str, Any] | object | None, key: str) -> Any:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def _float(payload: Mapping[str, Any] | object | None, *keys: str) -> float | None:
    for key in keys:
        raw = _value(payload, key)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value == value and value not in {float("inf"), float("-inf")}:
            return value
    return None


def resolve_tick_age_ms(payload: Mapping[str, Any] | object | None) -> float | None:
    age_ms = _float(payload, "tick_age_ms", "quote_age_ms", "last_tick_age_ms", "market_data_age_ms")
    if age_ms is not None:
        return max(0.0, age_ms)
    age_s = _float(payload, "tick_age_s", "quote_age_s", "data_age_seconds", "age_s", "age_seconds", "last_tick_age_s", "market_data_age_s")
    if age_s is not None:
        return max(0.0, age_s * 1000.0)
    return None


def resolve_tick_age_seconds(payload: Mapping[str, Any] | object | None) -> float | None:
    age_ms = resolve_tick_age_ms(payload)
    return None if age_ms is None else age_ms / 1000.0


def resolve_quote_version(payload: Mapping[str, Any] | object | None) -> object | None:
    for key in ("quote_update_version", "update_version", "tick_version", "last_tick_ts_ms", "timestamp_ms", "last_tick_timestamp", "timestamp"):
        value = _value(payload, key)
        if value not in (None, "", 0, 0.0):
            return value
    return None


def resolve_real_tick_count(payload: Mapping[str, Any] | object | None, *, tick_age_ms: float | None, max_age_ms: float, has_bid_ask: bool) -> tuple[int, bool]:
    for key in ("real_ticks_last_60s", "tick_count_60s", "recent_real_tick_count"):
        raw = _value(payload, key)
        if raw is None:
            continue
        try:
            return max(0, int(raw)), False
        except (TypeError, ValueError):
            return 0, False
    explicit_ms = any(_value(payload, key) is not None for key in ("tick_age_ms", "quote_age_ms", "last_tick_age_ms", "market_data_age_ms"))
    if explicit_ms and tick_age_ms is not None and tick_age_ms <= max_age_ms and has_bid_ask:
        return 1, True
    return 0, False


@dataclass(frozen=True, slots=True)
class ExecutionQuoteReadiness:
    symbol: str
    allowed: bool
    reason: str
    bid: float | None
    ask: float | None
    spread_pct: float | None
    tick_age_ms: float | None
    quote_update_version: object | None
    real_ticks_last_60s: int
    real_tick_count_derived: bool
    depth_available: bool
    tradable_quote: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_execution_quote(symbol: str, payload: Mapping[str, Any] | object | None, *, live_mode: bool, max_tick_age_ms: float, max_spread_pct: float, require_depth: bool, min_real_ticks_last_60s: int = 0) -> ExecutionQuoteReadiness:
    timestamp_quality_ok = quote_timestamp_quality_allows_hard_readiness(payload)
    bid, ask, derived_spread_pct, _bid_ask_source = resolve_quote_bid_ask_spread(payload)
    has_bid_ask = bool(bid is not None and ask is not None and bid > 0 and ask > bid)
    spread_pct = _float(payload, "spread_pct")
    if spread_pct is None:
        spread_pct = derived_spread_pct
    tick_age_ms = resolve_tick_age_ms(payload)
    quote_version = resolve_quote_version(payload)
    depth = _value(payload, "depth")
    depth_available = bool(_value(payload, "depth_available") is True or _value(payload, "quote_depth_valid") is True or (isinstance(depth, Mapping) and bool(depth.get("buy")) and bool(depth.get("sell"))))
    explicit_tradable = _value(payload, "tradable_quote")
    tradable_quote = bool(has_bid_ask and explicit_tradable is not False)
    real_ticks, derived = resolve_real_tick_count(payload, tick_age_ms=tick_age_ms, max_age_ms=max_tick_age_ms, has_bid_ask=has_bid_ask)

    reason = "ready"
    if live_mode and not timestamp_quality_ok:
        reason = "timestamp_quality_unusable"
    elif not has_bid_ask:
        reason = "bid_ask_missing"
    elif live_mode and tick_age_ms is None:
        reason = "tick_age_missing"
    elif tick_age_ms is not None and tick_age_ms > max_tick_age_ms:
        reason = "quote_stale"
    elif spread_pct is None:
        reason = "spread_missing"
    elif spread_pct > max_spread_pct:
        reason = "spread_too_wide"
    elif require_depth and not depth_available:
        reason = "quote_depth_missing"
    elif not tradable_quote:
        reason = "quote_not_tradable"
    elif real_ticks < max(0, int(min_real_ticks_last_60s)):
        reason = "insufficient_real_ticks"

    return ExecutionQuoteReadiness(symbol=symbol, allowed=reason == "ready", reason=reason, bid=bid, ask=ask, spread_pct=spread_pct, tick_age_ms=tick_age_ms, quote_update_version=quote_version, real_ticks_last_60s=real_ticks, real_tick_count_derived=derived, depth_available=depth_available, tradable_quote=tradable_quote)


__all__ = ["ExecutionQuoteReadiness", "evaluate_execution_quote", "resolve_quote_version", "resolve_real_tick_count", "resolve_tick_age_ms", "resolve_tick_age_seconds"]
