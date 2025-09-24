# src/server/health.py
# Lightweight health server for Railway/Render probes.
# Uses Waitress in production when available.

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from flask import Flask, Response, request

from src.diagnostics.metrics import metrics as core_metrics
from src.diagnostics.metrics import runtime_metrics
from src.strategies.runner import StrategyRunner
from src.utils import ringlog
from src.utils.freshness import compute as compute_freshness

app = Flask(__name__)
log = logging.getLogger(__name__)

# Filled by run(callback=...)
_status_callback: Callable[[], dict[str, Any]] | None = None
_start_ts = time.time()

_DEFAULT_DIAG_TAIL = 50


def _resolve_last_tick_dt(ds: Any) -> datetime | None:
    """Return the latest tick timestamp as ``datetime`` if available."""

    last_tick_dt: datetime | None = None
    tick_attr = getattr(ds, "last_tick_ts", None)
    if callable(tick_attr):
        try:
            last_tick_dt = tick_attr()
        except Exception:  # pragma: no cover - defensive diagnostic path
            last_tick_dt = None
    if last_tick_dt is None:
        tick_dt_accessor = getattr(ds, "last_tick_dt", None)
        if callable(tick_dt_accessor):
            try:
                last_tick_dt = tick_dt_accessor()
            except Exception:  # pragma: no cover - defensive diagnostic path
                last_tick_dt = None
    if last_tick_dt is None:
        last_tick_dt = getattr(ds, "_last_tick_ts", None)
    return last_tick_dt


@app.route("/live", methods=["GET"])
def live() -> tuple[dict[str, Any], int]:
    """Liveness probe: returns 200 as long as the process is up."""
    return {"status": "live", "uptime_sec": int(time.time() - _start_ts)}, 200


@app.route("/ready", methods=["GET"])
def ready() -> tuple[dict[str, Any], int]:
    """Readiness probe requiring broker session and fresh tick data."""

    runner = StrategyRunner.get_singleton()
    if runner is None:
        return {"status": "starting"}, 503

    kite = getattr(runner, "kite", None)
    is_conn = getattr(kite, "is_connected", None)
    broker_ok = bool(kite) and (is_conn() if callable(is_conn) else True)
    if not broker_ok:
        return {"status": "down", "reason": "broker"}, 503

    ds = getattr(runner, "data_source", None)
    if ds is None:
        return {"status": "down", "reason": "data"}, 503

    last_tick_dt = _resolve_last_tick_dt(ds)

    fresh = compute_freshness(
        now=datetime.utcnow(),
        last_tick_ts=last_tick_dt,
        last_bar_open_ts=ds.last_bar_open_ts(),
        tf_seconds=ds.timeframe_seconds,
        max_tick_lag_s=int(getattr(runner.strategy_cfg, "max_tick_lag_s", 8)),
        max_bar_lag_s=int(getattr(runner.strategy_cfg, "max_bar_lag_s", 75)),
    )
    watchdog = getattr(ds, "tick_watchdog", None)
    red_details: dict[str, Any] = {}
    red_flag = False
    if callable(watchdog):
        red_flag = bool(watchdog())
        if red_flag:
            details_fn = getattr(ds, "tick_watchdog_details", None)
            if callable(details_fn):
                red_details = details_fn()
    if not fresh.ok or red_flag:
        resp: dict[str, Any] = {
            "status": "down",
            "reason": "stale",
            "tick_lag_s": fresh.tick_lag_s,
        }
        if red_flag:
            resp["red_flag"] = red_details
        return resp, 503
    return {"status": "ready"}, 200


# Explicit HEAD route to avoid framework quirks on some platforms
@app.route("/health", methods=["HEAD"])
def health_head() -> tuple[Response, int]:
    """HEAD /health for ultra-cheap probe."""
    return Response(status=200), 200


def _metrics_snapshot() -> dict[str, Any]:
    """Return runtime metrics enriched with live trading data."""
    runner = StrategyRunner.get_singleton()
    if runner is not None:
        try:
            now = getattr(runner, "now_ist", datetime.utcnow())
            last = getattr(runner, "_last_trade_time", None)
            if last:
                mins = (now - last).total_seconds() / 60.0
                runtime_metrics.set_minutes_since_last_trade(round(mins, 1))
            plan = getattr(runner, "last_plan", {}) or {}
            runtime_metrics.set_delta(float(plan.get("delta") or 0.0))
            runtime_metrics.set_elasticity(float(plan.get("elasticity") or 0.0))
            basis = getattr(runner.settings, "EXPOSURE_BASIS", "premium")
            runtime_metrics.set_exposure_basis(basis)
            lot = int(getattr(runner.settings.instruments, "nifty_lot_size", 75))
            entry = float(plan.get("opt_entry") or plan.get("entry") or 0.0)
            spot_entry = float(plan.get("spot_entry") or entry)
            unit = (entry if basis == "premium" else spot_entry) * lot
            runtime_metrics.set_unit_notional(round(unit, 2))
        except Exception:
            pass
    return runtime_metrics.snapshot()


def _prometheus_metrics() -> str:
    """Return selected metrics in Prometheus text format."""
    snap = runtime_metrics.snapshot()
    now = time.time()
    evals_per_sec = core_metrics.signals / max(now - core_metrics._start_ts, 1e-6)

    runner = StrategyRunner.get_singleton()
    tick_age = 0.0
    breaker = 0
    warm_bars = 0
    open_risk = 0.0
    if runner is not None:
        ds = getattr(runner, "data_source", None)
        if ds is not None:
            last = _resolve_last_tick_dt(ds)
            if last:
                tick_age = (datetime.utcnow() - last).total_seconds()
            try:
                api_h: dict[str, Any] = getattr(ds, "api_health", lambda: {})()
                quote_h: dict[str, Any] = api_h.get("quote") or {}
                state = str(quote_h.get("state", "CLOSED"))
                breaker = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}.get(state, 0)
            except Exception:
                breaker = 0
        warm = getattr(runner, "_warm", None)
        warm_bars = int(getattr(warm, "have_bars", 0)) if warm else 0
        open_risk = float((getattr(runner, "last_plan", {}) or {}).get("risk_rupees") or 0.0)

    lines = [
        f"evals_per_sec {evals_per_sec}",
        f"tick_age {tick_age}",
        f"breaker_state {breaker}",
        f"warmup_bars {warm_bars}",
        f"open_risk_rupees {open_risk}",
        f"micro_wait_ratio {snap.get('micro_wait_ratio', 0.0)}",
        f"slippage_bps {snap.get('slippage_bps', 0.0)}",
    ]
    return "\n".join(lines) + "\n"


def _plan_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact plan summary guarding against missing keys."""

    opt_type = plan.get("option_type") or plan.get("ot") or "-"
    side = plan.get("side") or plan.get("action") or "-"
    strike = plan.get("strike") or plan.get("atm_strike") or plan.get("symbol") or "-"
    qty = plan.get("qty_lots") or plan.get("lots") or plan.get("qty") or None
    score = plan.get("score")
    rr = plan.get("rr")

    reason = plan.get("reason_block") or plan.get("reason") or "none"
    micro_raw = plan.get("micro")
    micro = micro_raw if isinstance(micro_raw, Mapping) else {}

    spread = plan.get("spread_pct")
    if spread is None:
        spread = micro.get("spread_pct")

    age = (
        plan.get("quote_age_s")
        or plan.get("age")
        or plan.get("bar_age_s")
        or plan.get("last_bar_lag_s")
    )
    if age is None:
        age = micro.get("age")

    depth = micro.get("depth_ok") if micro else None

    return {
        "side": side,
        "option_type": opt_type,
        "strike": strike,
        "qty_lots": qty,
        "score": score,
        "rr": rr,
        "reason": reason,
        "quote_age_s": age,
        "spread_pct": spread,
        "depth_ok": depth,
    }


def _parse_tail_limit(raw: str | None) -> int | None:
    """Return sanitized tail limit from query params."""

    if raw is None:
        return _DEFAULT_DIAG_TAIL
    raw = raw.strip().lower()
    if not raw:
        return _DEFAULT_DIAG_TAIL
    if raw in {"all", "*", "full"}:
        return None
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_DIAG_TAIL
    return max(value, 0)


@app.route("/health", methods=["GET"])
def health_get() -> tuple[dict[str, Any], int]:
    """Return lightweight health status."""
    try:
        diag = _status_callback() if _status_callback else {}
        ok = bool(diag.get("ok", True))
        resp = {
            "ok": ok,
            "uptime": int(time.time() - _start_ts),
            "window": diag.get("within_window"),
            "diag": diag,
            "metrics": _metrics_snapshot(),
        }
        return resp, 200 if ok else 503
    except Exception as e:
        log.exception("Health GET error: %s", e)
        return {
            "ok": False,
            "error": str(e),
            "metrics": _metrics_snapshot(),
        }, 500


@app.route("/diag/last", methods=["GET"])
def diag_last() -> tuple[dict[str, Any], int]:
    """Return the most recent diagnostics records from the structured log ring."""

    try:
        limit = _parse_tail_limit(request.args.get("limit"))
        records = ringlog.tail(None if limit is None else limit)
        resp: dict[str, Any] = {
            "ok": ringlog.enabled(),
            "limit": limit,
            "count": len(records),
            "capacity": ringlog.capacity(),
            "records": records,
        }
        return resp, 200
    except Exception as exc:  # pragma: no cover - defensive guard
        log.exception("Diag tail error: %s", exc)
        return {"ok": False, "error": str(exc)}, 500


@app.route("/status", methods=["GET"])
def status_get() -> tuple[dict[str, Any], int]:
    """Return extended status information."""
    try:
        diag = _status_callback() if _status_callback else {}
        resp = {
            "ok": bool(diag.get("ok", True)),
            "uptime": int(time.time() - _start_ts),
            "window": diag.get("within_window"),
            "diag": diag,
            "metrics": _metrics_snapshot(),
        }
        return resp, 200
    except Exception as e:
        log.exception("Status GET error: %s", e)
        return {
            "ok": False,
            "error": str(e),
            "metrics": _metrics_snapshot(),
        }, 500


@app.route("/metrics", methods=["GET"])
def metrics_get() -> tuple[Response, int]:
    """Return runtime execution metrics in Prometheus format."""
    try:
        text = _prometheus_metrics()
        return Response(text, mimetype="text/plain"), 200
    except Exception as e:
        log.exception("Metrics GET error: %s", e)
        return Response(f"# error {e}\n", mimetype="text/plain"), 500


@app.route("/plan", methods=["GET"])
def plan_get() -> tuple[dict[str, Any], int]:
    """Expose the latest plan summary for quick diagnostics."""

    runner = StrategyRunner.get_singleton()
    if runner is None:
        return {"ok": False, "error": "runner_unavailable"}, 503

    raw_plan = getattr(runner, "last_plan", {}) or {}
    plan = raw_plan if isinstance(raw_plan, Mapping) else {}
    summary = _plan_summary(plan)

    return {"ok": True, "plan": summary}, 200


def run(
    callback: Callable[[], dict[str, Any]] | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """
    Start the health server in the current thread.

    Uses Waitress when installed, falling back to Flask's development server.

    Typical usage is from a daemon thread:

        threading.Thread(
            target=health.run,
            kwargs={"callback": runner.health_check},
            daemon=True,
        ).start()

    Configuration order of precedence (highest first):
      1. ``host``/``port`` arguments passed to ``run``
      2. Environment variables ``HEALTH_HOST`` / ``HEALTH_PORT``
      3. Values from ``settings.health`` (defaults to 0.0.0.0:8000)
    """
    global _status_callback, _start_ts
    _status_callback = callback
    _start_ts = time.time()

    from src.config import settings

    default_host = getattr(settings.health, "host", "0.0.0.0")
    default_port = int(getattr(settings.health, "port", 8000))

    env_host = os.environ.get("HEALTH_HOST")
    env_port = os.environ.get("HEALTH_PORT")

    bind_host = host or env_host or default_host
    bind_port = int(port if port is not None else env_port or default_port)

    # Attempt to use a production-grade server if available
    try:
        from waitress import serve  # type: ignore[import-not-found,import-untyped]
    except ImportError:  # pragma: no cover - import guarded for optional dep
        serve = None

    if serve is not None:
        # Route waitress logs through our unified root logger
        wl = logging.getLogger("waitress")
        wl.handlers.clear()
        wl.propagate = True
        serve(app, host=bind_host, port=bind_port)
    else:
        # No reloader, multi-threaded to avoid blocking the trading loop
        app.run(
            host=bind_host,
            port=bind_port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
