"""Canonical strategy history-context builder.

This module owns the small, deterministic context-building contract consumed by
strategy evaluation. It does not select contracts, fetch broker instruments,
place orders, import execution, or change strategy scores.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import os
from typing import Any, Mapping

SPOT_SYMBOLS = {"NSE:NIFTY", "NIFTY", "NSE:NIFTY50", "NIFTY50"}
OPTION_SUFFIXES = ("CE", "PE")
_HISTORY_METHODS = ("get_ohlc_bars", "get_history", "get_bars")


@dataclass(frozen=True)
class StrategyHistoryContext:
    """Typed representation of domain-aware strategy history readiness."""

    history_count: int
    indicator_history_count: int
    option_history_count: int
    spot_history_count: int
    underlying_history_count: int
    history_symbol_key: str
    history_source: str
    history_domain_used: str
    history_resolved_count: int
    oldest_bar_ts: Any | None
    latest_bar_ts: Any | None
    history_quality: str
    history_required_min: int
    history_ready: bool
    smc_history_required_min: int
    history_ready_for_smc: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_count": self.history_count,
            "indicator_history_count": self.indicator_history_count,
            "option_history_count": self.option_history_count,
            "spot_history_count": self.spot_history_count,
            "underlying_history_count": self.underlying_history_count,
            "history_symbol_key": self.history_symbol_key,
            "history_source": self.history_source,
            "history_domain_used": self.history_domain_used,
            "history_resolved_count": self.history_resolved_count,
            "oldest_bar_ts": self.oldest_bar_ts,
            "latest_bar_ts": self.latest_bar_ts,
            "history_quality": self.history_quality,
            "history_required_min": self.history_required_min,
            "history_ready": self.history_ready,
            "smc_history_required_min": self.smc_history_required_min,
            "history_ready_for_smc": self.history_ready_for_smc,
        }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return max(default, minimum)
    return max(_safe_int(raw, default), minimum)


def _bar_ts(bar: Any) -> Any:
    if isinstance(bar, Mapping):
        return bar.get("timestamp") or bar.get("ts") or bar.get("datetime") or bar.get("time")
    return (
        getattr(bar, "timestamp", None)
        or getattr(bar, "ts", None)
        or getattr(bar, "datetime", None)
        or getattr(bar, "time", None)
    )


def classify_history_domain(symbol: str) -> str:
    symbol_upper = str(symbol or "").upper()
    if symbol_upper.endswith(OPTION_SUFFIXES):
        return "options"
    if symbol_upper in SPOT_SYMBOLS:
        return "spot"
    return "underlying"


def collect_history_bars(source: Any, symbol: str) -> list[Any]:
    if source is None:
        return []
    for method_name in _HISTORY_METHODS:
        method = getattr(source, method_name, None)
        if not callable(method):
            continue
        with suppress(Exception):
            bars = list(method(symbol) or [])
            if bars:
                return bars
    return []


def build_strategy_history_context(
    *,
    symbol: str,
    indicator_engine: Any,
    data_hub: Any | None,
    runner_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build domain-aware strategy history metadata without side effects."""

    data_hub_bars = collect_history_bars(data_hub, symbol)
    indicator_bars = collect_history_bars(indicator_engine, symbol)
    bars = data_hub_bars or indicator_bars
    history_source = (
        "data_hub" if data_hub_bars else "indicator_engine" if indicator_bars else "unavailable"
    )

    history_domain_used = classify_history_domain(symbol)
    raw_count = len(bars)
    option_count = raw_count if history_domain_used == "options" else 0
    spot_count = raw_count if history_domain_used == "spot" else 0
    underlying_count = raw_count if history_domain_used == "underlying" else 0

    if runner_context:
        option_count = max(option_count, _safe_int(runner_context.get("option_history_count"), 0))
        spot_count = max(spot_count, _safe_int(runner_context.get("spot_history_count"), 0))
        underlying_count = max(
            underlying_count,
            _safe_int(runner_context.get("underlying_history_count"), 0),
        )

    if history_domain_used == "options":
        resolved_history_count = option_count
    elif history_domain_used == "spot":
        resolved_history_count = spot_count
    else:
        resolved_history_count = underlying_count

    option_eval_min_bars = _env_int("OPTION_EVAL_MIN_BARS", 5)
    context_min_bars = _env_int("CONTEXT_MIN_BARS", 50)
    smc_min_bars = _env_int("SMC_MIN_BARS_REQUIRED", 30)
    domain_min_required = (
        option_eval_min_bars if history_domain_used == "options" else context_min_bars
    )
    oldest_bar_ts = _bar_ts(bars[0]) if bars else None
    latest_bar_ts = _bar_ts(bars[-1]) if bars else None

    return StrategyHistoryContext(
        history_count=resolved_history_count,
        indicator_history_count=raw_count,
        option_history_count=option_count,
        spot_history_count=spot_count,
        underlying_history_count=underlying_count,
        history_symbol_key=symbol,
        history_source=history_source,
        history_domain_used=history_domain_used,
        history_resolved_count=resolved_history_count,
        oldest_bar_ts=oldest_bar_ts,
        latest_bar_ts=latest_bar_ts,
        history_quality="warm" if resolved_history_count >= domain_min_required else "cold",
        history_required_min=domain_min_required,
        history_ready=resolved_history_count >= domain_min_required,
        smc_history_required_min=smc_min_bars,
        history_ready_for_smc=resolved_history_count >= smc_min_bars,
    ).to_dict()


__all__ = [
    "StrategyHistoryContext",
    "build_strategy_history_context",
    "classify_history_domain",
    "collect_history_bars",
]
