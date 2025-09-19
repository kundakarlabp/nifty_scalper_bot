# src/diagnostics/healthkit.py
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

from src.config import settings


@dataclass
class HealthItem:
    name: str
    ok: bool
    hint: str | None = None
    detail: str | None = None
    severity: str = "info"  # info|warn|error


def to_dict(
    items: List[HealthItem],
    *,
    last_signal: dict | None = None,
    meta: dict | None = None,
) -> Dict[str, Any]:
    return {
        "ok": all(x.ok for x in items),
        "checks": [asdict(x) for x in items],
        "last_signal": bool(last_signal),
        "meta": meta or {},
    }


def render_compact(items: List[HealthItem]) -> str:
    bullets = []
    for x in items:
        dot = "🟢" if x.ok else "🔴"
        bullets.append(f"{dot} {x.name}")
    head = "✅ Flow looks good" if all(i.ok for i in items) else "❗ Flow has issues"
    return head + "\n" + " · ".join(bullets)


def render_detailed(items: List[HealthItem], *, last_signal_present: bool) -> str:
    lines = ["🔍 Full system check"]
    for x in items:
        dot = "🟢" if x.ok else "🔴"
        extra = x.hint or x.detail or ""
        lines.append(f"{dot} {x.name}" + (f" — {extra}" if extra else ""))
    lines.append(f"📈 last_signal: {'present' if last_signal_present else 'none'}")
    return "\n".join(lines)


def snapshot_pipeline() -> Dict[str, Any]:
    """Return a compact snapshot of the trading pipeline's health."""

    from src.strategies.runner import StrategyRunner

    base_snapshot: Dict[str, Any] = {
        "market_open": False,
        "equity": None,
        "risk": {
            "daily_dd": None,
            "cap_pct": float(getattr(settings, "EXPOSURE_CAP_PCT", 0.0)),
        },
        "signals": {"regime": None, "atr_pct": None, "score": None},
        "micro": {"ce": None, "pe": None},
        "open_orders": 0,
        "positions": 0,
        "latency": {"tick_age": None, "bar_lag": None},
    }

    runner = StrategyRunner.get_singleton()
    if runner is None:
        return base_snapshot

    snapshot = dict(base_snapshot)
    snapshot["market_open"] = bool(
        getattr(runner, "_within_trading_window", lambda *_: False)(None)
    )

    equity_cached = getattr(runner, "_equity_cached_value", None)
    if equity_cached is not None:
        try:
            snapshot["equity"] = round(float(equity_cached), 2)
        except Exception:
            snapshot["equity"] = float(equity_cached)

    risk_state = getattr(runner, "risk", None)
    daily_dd = getattr(risk_state, "day_realized_loss", None)
    if daily_dd is None:
        engine = getattr(runner, "risk_engine", None)
        daily_dd = getattr(getattr(engine, "state", None), "cum_loss_rupees", None)
    if daily_dd is not None:
        try:
            snapshot["risk"]["daily_dd"] = round(float(daily_dd), 2)
        except Exception:
            snapshot["risk"]["daily_dd"] = daily_dd

    plan = getattr(runner, "last_plan", None) or getattr(
        runner, "_last_signal_debug", {}
    )
    if isinstance(plan, dict):
        for key in ("regime", "atr_pct", "score"):
            snapshot["signals"][key] = plan.get(key)

    source = getattr(runner, "data_source", None)
    micro_states: Dict[str, Any] = {"ce": None, "pe": None}
    if source is not None:
        tokens = getattr(source, "atm_tokens", (None, None)) or (None, None)
        getter = getattr(source, "get_micro_state", None)
        for label, token in zip(("ce", "pe"), list(tokens)[:2]):
            if callable(getter) and token:
                try:
                    micro_states[label] = getter(token)
                except Exception:
                    micro_states[label] = {"error": "micro_state_failed"}
            else:
                micro_states[label] = None
    snapshot["micro"] = micro_states

    executor = getattr(runner, "executor", None)
    if executor is not None:
        orders_fn = getattr(executor, "get_active_orders", None)
        try:
            open_orders = orders_fn() if callable(orders_fn) else []
        except Exception:
            open_orders = []
        snapshot["open_orders"] = (
            len(open_orders)
            if isinstance(open_orders, (list, tuple, set))
            else int(getattr(executor, "open_count", 0))
        )

        pos_fn = getattr(executor, "get_positions_kite", None)
        try:
            positions = pos_fn() if callable(pos_fn) else {}
        except Exception:
            positions = {}
        if isinstance(positions, dict):
            snapshot["positions"] = len(positions)
        elif isinstance(positions, (list, tuple, set)):
            snapshot["positions"] = len(positions)
        elif positions:
            snapshot["positions"] = 1

    now_epoch = time.time()
    tick_age = None
    if source is not None:
        tick_ts = getattr(source, "last_tick_ts", None)
        if tick_ts:
            try:
                tick_age = max(now_epoch - float(tick_ts), 0.0)
            except Exception:
                tick_age = None
    snapshot["latency"]["tick_age"] = tick_age

    bar_lag = None
    window = getattr(runner, "_ohlc_cache", None)
    if window is not None:
        try:
            is_empty = bool(getattr(window, "empty", False))
        except Exception:
            is_empty = False
        if not is_empty:
            try:
                last_ts = window.index[-1]
            except Exception:
                last_ts = None
            if last_ts is not None:
                last_dt: datetime | None
                if hasattr(last_ts, "to_pydatetime"):
                    last_dt = last_ts.to_pydatetime()
                elif isinstance(last_ts, datetime):
                    last_dt = last_ts
                else:
                    try:
                        last_dt = datetime.fromisoformat(str(last_ts))
                    except Exception:
                        last_dt = None
                if last_dt is not None:
                    now_dt_fn = getattr(runner, "_now_ist", None)
                    if callable(now_dt_fn):
                        now_dt = now_dt_fn()
                    else:
                        now_dt = datetime.utcnow()
                    if last_dt.tzinfo is None and getattr(now_dt, "tzinfo", None) is not None:
                        last_dt = last_dt.replace(tzinfo=now_dt.tzinfo)
                    try:
                        bar_lag = max((now_dt - last_dt).total_seconds(), 0.0)
                    except Exception:
                        bar_lag = None
    snapshot["latency"]["bar_lag"] = bar_lag

    return snapshot
