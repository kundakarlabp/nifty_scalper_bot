"""Concrete diagnostic checks for bot components."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import settings
from src.diagnostics.registry import CheckResult, register
from src.execution.order_executor import OrderManager
from src.execution.order_state import OrderState
from src.features.indicators import atr_pct
from src.risk.position_sizing import PositionSizer
from src.signals.patches import resolve_atr_band
from src.utils.expiry import last_tuesday_of_month, next_tuesday_expiry

# Indian Standard Time is the canonical timezone for bot diagnostics.
IST = ZoneInfo("Asia/Kolkata")


def _summary(**values: Any) -> str:
    """Render ``values`` as a compact ``key=value`` string."""

    parts: List[str] = []
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
    """Normalize ``ts`` into an aware :class:`~pandas.Timestamp` in IST."""

    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None or stamp.tz is None:
        return stamp.tz_localize(IST)
    return stamp.tz_convert(IST)

# Map ``reason_block`` codes to human‑readable descriptions used by
# diagnostics endpoints like ``/why``.  Only codes that require custom
# wording need to be present; unknown keys fall back to the raw code.
REASON_MAP: Dict[str, str] = {
    "cap_lt_one_lot": "premium cap too small for 1 lot",
}


def _ok(
    msg: str,
    *,
    name: str,
    summary: str | None = None,
    **details: Any,
) -> CheckResult:
    """Helper to build a successful :class:`CheckResult`."""

    payload = dict(details)
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

    payload = dict(details)
    summary_text = msg if summary is None else summary
    payload.setdefault("summary", str(summary_text))
    return CheckResult(name=name, ok=False, msg=msg, fix=fix, details=payload)


@register("config")
def check_config() -> CheckResult:
    """Ensure critical configuration keys are present and sane."""

    miss = []
    if getattr(settings.strategy, "rr_threshold", None) is None:
        miss.append("strategy.rr_threshold")
    if getattr(settings.risk, "risk_per_trade", None) is None:
        miss.append("risk.risk_per_trade")
    lookback = int(getattr(settings.data, "lookback_minutes", 0))
    min_bars = int(getattr(settings.strategy, "min_bars_for_signal", 0))
    if lookback < min_bars:
        return _bad(
            "lookback < min_bars",
            name="config",
            fix="increase lookback_minutes or lower min_bars_for_signal",
            summary=_summary(lookback=lookback, min_bars=min_bars),
            lookback=lookback,
            min_bars=min_bars,
        )
    if miss:
        return _bad(
            "missing keys",
            name="config",
            fix="set defaults in config/defaults.yaml",
            summary=_summary(missing=",".join(miss)),
            missing=miss,
        )
    rr_val = getattr(settings.strategy, "rr_threshold", None)
    risk_pct = getattr(settings.risk, "risk_per_trade", None)
    return _ok(
        "loaded",
        name="config",
        summary=_summary(rr=rr_val, risk=risk_pct),
        rr=getattr(settings.strategy, "rr_threshold", None),
        risk_pct=getattr(settings.risk, "risk_per_trade", None),
    )


@register("data_window")
def check_data_window() -> CheckResult:
    """Verify the OHLC cache is fresh and sufficiently populated."""

    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad(
            "runner not ready",
            name="data_window",
            fix="start the bot",
            summary=_summary(runner="missing"),
        )
    df = r.ohlc_window()
    if df is None or df.empty:
        return _bad(
            "no bars",
            name="data_window",
            fix="enable backfill or broker history",
            summary=_summary(bars=0),
        )
    now_ist = _as_aware_ist(r.now_ist)
    last_ts_ist = _as_aware_ist(df.index[-1])
    lag_s = (now_ist - last_ts_ist).total_seconds()
    tf_s = 60  # timeframe is minute
    ok = lag_s <= 3 * tf_s
    msg = "fresh" if ok else "stale"
    fix = None if ok else "investigate broker clock/backfill"
    summary = _summary(bars=len(df), lag_s=round(lag_s, 1), tf_s=tf_s)
    details = {
        "bars": len(df),
        "last_bar_ts": last_ts_ist.isoformat(),
        "lag_s": lag_s,
        "tf_s": tf_s,
    }
    if ok:
        return _ok(msg, name="data_window", summary=summary, **details)
    assert fix is not None
    return _bad(msg, name="data_window", fix=fix, summary=summary, **details)


@register("atr")
def check_atr() -> CheckResult:
    """Ensure ATR percentage meets configured minimum."""

    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad(
            "runner not ready",
            name="atr",
            fix="start the bot",
            summary=_summary(runner="missing"),
        )
    df = r.ohlc_window()
    atr_period = int(getattr(settings.strategy, "atr_period", 14))
    if df is None or len(df) < max(10, atr_period + 1):
        return _bad(
            "insufficient bars for ATR",
            name="atr",
            fix="increase lookback/min_bars",
            summary=_summary(bars=len(df) if df is not None else 0),
            bars=len(df) if df is not None else 0,
        )
    atrp = atr_pct(df, period=atr_period) or 0.0
    minp, maxp = resolve_atr_band(r.strategy_cfg, r.under_symbol)
    ok = atrp >= float(minp)
    details = {
        "atr_pct": atrp,
        "min_atr_pct": float(minp),
        "atr_band": (float(minp), float(maxp)),
    }
    summary = _summary(atr_pct=round(atrp, 4), min=float(minp), max=float(maxp))
    if ok:
        return _ok(
            f"atr%={atrp:.4f} (min {minp})",
            name="atr",
            summary=summary,
            **details,
        )
    return _bad(
        f"atr%={atrp:.4f} (min {minp})",
        name="atr",
        fix="lower MIN_ATR_PCT temporarily or wait for volatility",
        summary=summary,
        **details,
    )


@register("regime")
def check_regime() -> CheckResult:
    """Check that the regime detector returns a valid state."""

    from src.signals.regime_detector import detect_market_regime
    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad(
            "runner not ready",
            name="regime",
            fix="start the bot",
            summary=_summary(runner="missing"),
        )
    df = r.ohlc_window()
    if df is None or df.empty:
        return _bad(
            "no bars",
            name="regime",
            fix="collect more bars",
            summary=_summary(bars=0),
        )
    res = detect_market_regime(df=df)
    ok = res.regime in {"TREND", "RANGE", "NO_TRADE"}
    return _ok(
        "ok" if ok else "bad",
        name="regime",
        summary=_summary(regime=res.regime),
        regime=res.regime,
    )


@register("strategy")
def check_strategy() -> CheckResult:
    """Ensure strategy produces a structured plan."""

    from src.strategies.runner import StrategyRunner
    from src.strategies.scalping_strategy import EnhancedScalpingStrategy

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad("runner not ready", name="strategy", fix="start the bot")
    df = r.ohlc_window()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return _bad("no data", name="strategy", fix="wait for bars")
    st = EnhancedScalpingStrategy()
    plan: Dict[str, Any] = (
        st.generate_signal(df, current_price=float(df["close"].iloc[-1])) or {}
    )
    need = {"action", "rr", "score", "reasons"}
    missing = [k for k in need if k not in plan]
    if missing:
        return _bad(
            "plan missing keys",
            name="strategy",
            fix="return required keys from build_plan",
            missing=missing,
        )
    return _ok(
        "plan ok",
        name="strategy",
        summary=_summary(
            action=plan.get("action"),
            score=plan.get("score"),
            rr=plan.get("rr"),
        ),
        action=plan["action"],
        score=plan["score"],
        rr=plan["rr"],
    )


@register("sizing")
def check_sizing() -> CheckResult:
    """Validate that position sizing uses percent risk."""

    ps = PositionSizer.from_settings(
        risk_per_trade=settings.risk.risk_per_trade,
        min_lots=settings.instruments.min_lots,
        max_lots=settings.instruments.max_lots,
        max_position_size_pct=settings.risk.max_position_size_pct,
        exposure_basis=settings.EXPOSURE_BASIS,
    )
    qty, lots, diag = ps.size_from_signal(
        entry_price=200,
        stop_loss=190,
        lot_size=50,
        equity=100_000,
        spot_sl_points=10,
        delta=0.5,
    )
    expected = 100_000 * float(settings.risk.risk_per_trade)
    unit_expected = 200 * 50
    if (
        settings.EXPOSURE_CAP_SOURCE == "equity"
        and float(settings.EXPOSURE_CAP_PCT_OF_EQUITY) > 0
    ):
        min_eq_expected = unit_expected / float(settings.EXPOSURE_CAP_PCT_OF_EQUITY)
    else:
        min_eq_expected = unit_expected
    ok = (
        diag["risk_rupees"] == expected
        and diag["unit_notional"] == unit_expected
        and diag["min_equity_needed"] == min_eq_expected
    )
    summary = _summary(
        lots=diag.get("lots"),
        risk_rupees=diag.get("risk_rupees"),
        unit_notional=diag.get("unit_notional"),
    )
    if ok:
        return _ok("percent-risk", name="sizing", summary=summary, **diag)
    return _bad(
        "percent-risk",
        name="sizing",
        fix="use equity*risk_pct and mid*lot for unit_notional/min_equity",
        summary=summary,
        **diag,
    )


@register("order_manager")
def check_order_manager() -> CheckResult:
    """Self-test partial fill handling and timeout cancellation."""

    calls: List[Dict[str, Any]] = []

    def _place(payload: Dict[str, Any]) -> str:
        calls.append(payload)
        return f"OID{len(calls)}"

    om = OrderManager(_place, tick_size=1.0, fill_timeout_ms=1)
    quote = {
        "bid": 99.0,
        "ask": 100.0,
        "depth": {
            "buy": [{"price": 99.0, "quantity": 10}],
            "sell": [
                {"price": 100.0, "quantity": 5},
                {"price": 101.0, "quantity": 5},
            ],
        },
        "tick": 1.0,
    }
    om.submit({"action": "BUY", "symbol": "TEST", "quantity": 8, "quote": quote})
    cid = next(iter(om.orders))
    om.handle_partial(cid, 3, 100.0, quote)
    partial_ok = om.orders[cid].state == OrderState.PARTIAL and len(calls) == 2
    om.orders[cid].expires_at = datetime.utcnow() - timedelta(milliseconds=10)
    om.check_timeouts()
    timeout_ok = om.orders[cid].state == OrderState.CANCELLED
    summary = _summary(partial=partial_ok, timeout=timeout_ok, calls=len(calls))
    if partial_ok and timeout_ok:
        return _ok("ok", name="order_manager", summary=summary)
    return _bad(
        "lifecycle",
        name="order_manager",
        fix="review order manager",
        summary=summary,
    )


@register("micro")
def check_micro() -> CheckResult:
    """Check option microstructure metrics (spread and depth)."""

    from src.execution.micro_filters import micro_from_l1
    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad(
            "runner not ready",
            name="micro",
            fix="start the bot",
            summary=_summary(runner="missing"),
        )
    q = r.get_current_l1()
    if not q:
        return _ok(
            "micro N/A (soft-pass)",
            name="micro",
            summary=_summary(status="no_quote"),
            reason="no_quote",
        )
    spread, depth_ok, _ = micro_from_l1(
        q, lot_size=r.lot_size, depth_min_lots=r.strategy_cfg.depth_min_lots
    )
    depth_status = "-" if depth_ok is None else ("ok" if depth_ok else "fail")
    summary = _summary(
        spread_pct=None if spread is None else round(spread, 3),
        depth=depth_status,
    )
    return _ok(
        "micro",
        name="micro",
        summary=summary,
        spread_pct=spread,
        depth_ok=depth_ok,
    )


@register("expiry")
def check_expiry() -> CheckResult:
    """Return next weekly and monthly expiries using Tuesday rules."""

    wk = next_tuesday_expiry()
    mo = last_tuesday_of_month()
    summary = _summary(weekly=wk, monthly=mo)
    return _ok(
        "tuesday rules",
        name="expiry",
        summary=summary,
        weekly=str(wk),
        monthly=str(mo),
    )


@register("broker")
def check_broker() -> CheckResult:
    """Ensure a broker session is present."""

    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None or r.kite is None:
        return _bad(
            "no session",
            name="broker",
            fix="re-login / check API key",
            summary=_summary(connected=False),
            connected=False,
        )
    return _ok(
        "connected",
        name="broker",
        summary=_summary(connected=True),
        connected=True,
    )


@register("risk_gates")
def check_risk_gates() -> CheckResult:
    """Verify that risk gates have been evaluated at least once."""

    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad(
            "runner not ready",
            name="risk_gates",
            fix="start the bot",
            summary=_summary(runner="missing"),
        )
    gates = r.get_last_flow_debug().get("risk_gates")
    if isinstance(gates, dict):
        if not gates:
            ok = False
            status = "no-eval"
        else:
            ok = all(bool(v) for v in gates.values())
            status = "ok" if ok else "blocked"
    else:
        ok = False
        status = "no-eval"
    return _ok(
        "evaluated" if ok else "no-eval",
        name="risk_gates",
        summary=_summary(status=status),
        gates=gates or {},
    )
