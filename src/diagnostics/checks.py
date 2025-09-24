"""Diagnostics helpers.

This module hosts structured logging utilities as well as lightweight
health checks consumed by the test-suite and the `/why` endpoint.  Only a
subset of the historical checks are required for the current task so the
implementation focuses on ``check_data_window`` and quote diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.diagnostics.registry import CheckResult, register
from src.logs import structured_log
from src.utils import ringlog


def emit_quote_diag(
    *,
    token: int,
    symbol: str | None = None,
    sub_mode: str | None = None,
    bid: float | None = None,
    ask: float | None = None,
    bid_qty: int | None = None,
    ask_qty: int | None = None,
    last_tick_age_ms: float | None = None,
    retries: int = 0,
    reason: str = "unknown",
    source: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured diagnostic record summarizing quote readiness."""

    payload: dict[str, Any] = {
        "token": int(token),
        "symbol": symbol,
        "sub_mode": sub_mode,
        "bid": bid,
        "ask": ask,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "last_tick_age_ms": last_tick_age_ms,
        "retries": retries,
        "reason": reason,
        "source": source,
    }
    if extra:
        payload.update(dict(extra))
    structured_log.event("quote_diag", **payload)


IST = ZoneInfo("Asia/Kolkata")

# Shared diagnostics ring buffer used by multiple components.
TRACE_RING = ringlog

# Map ``reason_block`` codes to human-readable descriptions used by diagnostics.
REASON_MAP: dict[str, str] = {
    "cap_lt_one_lot": "premium cap too small for 1 lot",
}


def _summary(**values: Any) -> str:
    """Render ``values`` as a compact ``key=value`` string."""

    parts: list[str] = []
    for key, value in values.items():
        if value in {None, ""}:
            val = "-"
        elif isinstance(value, float):
            val = f"{value:.4f}".rstrip("0").rstrip(".")
        else:
            val = str(value)
        parts.append(f"{key}={val}")
    return " ".join(parts)


def _as_aware_ist(ts: Any) -> pd.Timestamp:
    """Normalize ``ts`` into an aware timestamp in IST."""

    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None or stamp.tz is None:
        return stamp.tz_localize(IST)
    return stamp.tz_convert(IST)


def _ok(
    msg: str,
    *,
    name: str,
    summary: str | None = None,
    **details: Any,
) -> CheckResult:
    """Helper to build a successful :class:`CheckResult`."""

    payload: dict[str, Any] = dict(details)
    summary_text = msg if summary is None else summary
    payload.setdefault("summary", str(summary_text))
    return CheckResult(name=name, ok=True, msg=msg, details=payload)


def _bad(
    msg: str,
    *,
    name: str,
    fix: str,
    summary: str | None = None,
    **details: Any,
) -> CheckResult:
    """Helper to build a failed :class:`CheckResult`."""

    payload: dict[str, Any] = dict(details)
    summary_text = msg if summary is None else summary
    payload.setdefault("summary", str(summary_text))
    return CheckResult(name=name, ok=False, msg=msg, fix=fix, details=payload)


@register("data_window")
def check_data_window() -> CheckResult:
    """Verify the OHLC cache is fresh and sufficiently populated."""

    from src.strategies.runner import StrategyRunner

    runner = StrategyRunner.get_singleton()
    if runner is None:
        return _bad(
            "runner not ready",
            name="data_window",
            fix="start the bot",
            summary=_summary(runner="missing"),
        )
    frame = runner.ohlc_window()
    if frame is None or frame.empty:
        return _bad(
            "no bars",
            name="data_window",
            fix="enable backfill or broker history",
            summary=_summary(bars=0),
        )
    now_ist = _as_aware_ist(runner.now_ist)
    last_ts_ist = _as_aware_ist(frame.index[-1])
    lag_s = (now_ist - last_ts_ist).total_seconds()
    tf_s = 60  # timeframe is minute
    ok = lag_s <= 3 * tf_s
    msg = "fresh" if ok else "stale"
    fix = None if ok else "investigate broker clock/backfill"
    summary = _summary(bars=len(frame), lag_s=round(lag_s, 1), tf_s=tf_s)
    details = {
        "bars": len(frame),
        "last_bar_ts": last_ts_ist.isoformat(),
        "lag_s": lag_s,
        "tf_s": tf_s,
    }
    if ok:
        return _ok(msg, name="data_window", summary=summary, **details)
    assert fix is not None
    return _bad(msg, name="data_window", fix=fix, summary=summary, **details)


__all__ = [
    "emit_quote_diag",
    "check_data_window",
    "TRACE_RING",
    "REASON_MAP",
]
