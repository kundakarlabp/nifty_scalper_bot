"""Concrete diagnostic checks for bot components."""

from __future__ import annotations

from typing import Any, Dict

from src.config import settings
from src.diagnostics.registry import CheckResult, register
from src.features.indicators import atr_pct
from src.risk.position_sizing import PositionSizer
from src.utils.expiry import last_tuesday_of_month, next_tuesday_expiry


def _ok(msg: str, *, name: str, **details: Any) -> CheckResult:
    """Helper to build a successful :class:`CheckResult`."""

    return CheckResult(name=name, ok=True, msg=msg, details=details)


def _bad(msg: str, *, name: str, fix: str, **details: Any) -> CheckResult:
    """Helper to build a failed :class:`CheckResult`."""

    return CheckResult(name=name, ok=False, msg=msg, fix=fix, details=details)


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
            lookback=lookback,
            min_bars=min_bars,
        )
    if miss:
        return _bad(
            "missing keys",
            name="config",
            fix="set defaults in config/defaults.yaml",
            missing=miss,
        )
    return _ok(
        "loaded",
        name="config",
        rr=getattr(settings.strategy, "rr_threshold", None),
        risk_pct=getattr(settings.risk, "risk_per_trade", None),
    )


@register("data_window")
def check_data_window() -> CheckResult:
    """Verify the OHLC cache is fresh and sufficiently populated."""

    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad("runner not ready", name="data_window", fix="start the bot")
    df = r.ohlc_window()
    if df is None or df.empty:
        return _bad("no bars", name="data_window", fix="enable backfill or broker history")
    last_ts = df.index[-1]
    lag_s = (r.now_ist - last_ts).total_seconds()
    tf_s = 60  # timeframe is minute
    ok = lag_s <= 3 * tf_s
    msg = "fresh" if ok else "stale"
    fix = None if ok else "investigate broker clock/backfill"
    return CheckResult(
        name="data_window",
        ok=ok,
        msg=msg,
        details={"bars": len(df), "last_bar_ts": str(last_ts), "lag_s": lag_s, "tf_s": tf_s},
        fix=fix,
    )


@register("atr")
def check_atr() -> CheckResult:
    """Ensure ATR percentage meets configured minimum."""

    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad("runner not ready", name="atr", fix="start the bot")
    df = r.ohlc_window()
    atr_period = int(getattr(settings.strategy, "atr_period", 14))
    if df is None or len(df) < max(10, atr_period):
        return _bad(
            "insufficient bars for ATR",
            name="atr",
            fix="increase lookback/min_bars",
            bars=0,
        )
    atrp = atr_pct(df, period=atr_period) or 0.0
    minp = (
        r.strategy_cfg.min_atr_pct_banknifty
        if r.under_symbol == "BANKNIFTY"
        else r.strategy_cfg.min_atr_pct_nifty
    )
    ok = atrp >= float(minp)
    return CheckResult(
        name="atr",
        ok=ok,
        msg=f"atr%={atrp:.4f} (min {minp})",
        details={"atr_pct": atrp, "min_atr_pct": float(minp)},
        fix=None if ok else "lower MIN_ATR_PCT temporarily or wait for volatility",
    )


@register("regime")
def check_regime() -> CheckResult:
    """Check that the regime detector returns a valid state."""

    from src.signals.regime_detector import detect_market_regime
    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad("runner not ready", name="regime", fix="start the bot")
    df = r.ohlc_window()
    if df is None or df.empty:
        return _bad("no bars", name="regime", fix="collect more bars")
    res = detect_market_regime(df=df)
    ok = res.regime in {"TREND", "RANGE", "NO_TRADE"}
    return _ok("ok" if ok else "bad", name="regime", regime=res.regime)


@register("strategy")
def check_strategy() -> CheckResult:
    """Ensure strategy produces a structured plan."""

    from src.strategies.scalping_strategy import EnhancedScalpingStrategy
    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad("runner not ready", name="strategy", fix="start the bot")
    df = r.ohlc_window()
    st = EnhancedScalpingStrategy()
    plan = st.generate_signal(df, current_price=float(df["close"].iloc[-1]))
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
    )
    qty, lots, diag = ps.size_from_signal(
        entry_price=200, stop_loss=190, lot_size=50, equity=100_000
    )
    expected = 100_000 * float(settings.risk.risk_per_trade)
    ok = diag["risk_rupees"] == expected
    return CheckResult(
        name="sizing",
        ok=ok,
        msg="percent-risk",
        details=diag,
        fix=None if ok else "use equity * risk_per_trade",
    )


@register("micro")
def check_micro() -> CheckResult:
    """Check option microstructure metrics (spread and depth)."""

    from src.strategies.runner import StrategyRunner
    from src.execution.micro_filters import micro_from_l1

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad("runner not ready", name="micro", fix="start the bot")
    q = r.get_current_l1()
    if not q:
        return _ok("micro N/A (soft-pass)", name="micro", reason="no_quote")
    spread, depth_ok, _ = micro_from_l1(
        q, lot_size=r.lot_size, depth_min_lots=r.strategy_cfg.depth_min_lots
    )
    return _ok("micro", name="micro", spread_pct=spread, depth_ok=depth_ok)


@register("expiry")
def check_expiry() -> CheckResult:
    """Return next weekly and monthly expiries using Tuesday rules."""

    wk = next_tuesday_expiry()
    mo = last_tuesday_of_month()
    return _ok("tuesday rules", name="expiry", weekly=str(wk), monthly=str(mo))


@register("broker")
def check_broker() -> CheckResult:
    """Ensure a broker session is present."""

    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None or r.kite is None:
        return _bad("no session", name="broker", fix="re-login / check API key", connected=False)
    return _ok("connected", name="broker", connected=True)


@register("risk_gates")
def check_risk_gates() -> CheckResult:
    """Verify that risk gates have been evaluated at least once."""

    from src.strategies.runner import StrategyRunner

    r = StrategyRunner.get_singleton()
    if r is None:
        return _bad("runner not ready", name="risk_gates", fix="start the bot")
    gates = r.get_last_flow_debug().get("risk_gates")
    ok = isinstance(gates, dict) and bool(gates)
    return _ok("evaluated" if ok else "no-eval", name="risk_gates", gates=gates or {})
