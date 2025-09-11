# src/server/health.py
# Lightweight health server for Railway/Render probes.
# Uses Waitress in production when available.

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from flask import Flask, Response

from src.strategies.runner import StrategyRunner
from src.utils.freshness import compute as compute_freshness
from src.diagnostics.metrics import runtime_metrics

app = Flask(__name__)
log = logging.getLogger(__name__)

# Filled by run(callback=...)
_status_callback: Optional[Callable[[], Dict[str, Any]]] = None
_start_ts = time.time()


@app.route("/live", methods=["GET"])
def live() -> Tuple[Dict[str, Any], int]:
    """Liveness probe: returns 200 as long as the process is up."""
    return {"status": "live", "uptime_sec": int(time.time() - _start_ts)}, 200


@app.route("/ready", methods=["GET"])
def ready() -> Tuple[Dict[str, Any], int]:
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

    fresh = compute_freshness(
        now=datetime.utcnow(),
        last_tick_ts=ds.last_tick_ts(),
        last_bar_open_ts=ds.last_bar_open_ts(),
        tf_seconds=ds.timeframe_seconds,
        max_tick_lag_s=int(getattr(runner.strategy_cfg, "max_tick_lag_s", 8)),
        max_bar_lag_s=int(getattr(runner.strategy_cfg, "max_bar_lag_s", 75)),
    )
    watchdog = getattr(ds, "tick_watchdog", None)
    red_details: Dict[str, Any] = {}
    red_flag = False
    if callable(watchdog):
        red_flag = bool(watchdog())
        if red_flag:
            details_fn = getattr(ds, "tick_watchdog_details", None)
            if callable(details_fn):
                red_details = details_fn()
    if not fresh.ok or red_flag:
        resp: Dict[str, Any] = {
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
def health_head() -> Tuple[Response, int]:
    """HEAD /health for ultra-cheap probe."""
    return Response(status=200), 200


def _metrics_snapshot() -> Dict[str, Any]:
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
            basis = getattr(runner.settings, "exposure_basis", "premium")
            runtime_metrics.set_exposure_basis(basis)
            lot = int(getattr(runner.settings.instruments, "nifty_lot_size", 75))
            entry = float(plan.get("opt_entry") or plan.get("entry") or 0.0)
            spot_entry = float(plan.get("spot_entry") or entry)
            unit = (entry if basis == "premium" else spot_entry) * lot
            runtime_metrics.set_unit_notional(round(unit, 2))
        except Exception:
            pass
    return runtime_metrics.snapshot()


@app.route("/health", methods=["GET"])
def health_get() -> Tuple[Dict[str, Any], int]:
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


@app.route("/status", methods=["GET"])
def status_get() -> Tuple[Dict[str, Any], int]:
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
def metrics_get() -> Tuple[Dict[str, Any], int]:
    """Return runtime execution metrics."""
    try:
        return _metrics_snapshot(), 200
    except Exception as e:
        log.exception("Metrics GET error: %s", e)
        return {"error": str(e)}, 500


def run(
    callback: Optional[Callable[[], Dict[str, Any]]] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
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

    Env overrides:
      HEALTH_HOST (default "0.0.0.0")
      HEALTH_PORT (default "8000")
    """
    global _status_callback, _start_ts
    _status_callback = callback
    _start_ts = time.time()

    bind_host = host or os.environ.get("HEALTH_HOST", "0.0.0.0")
    bind_port = int(port or int(os.environ.get("HEALTH_PORT", "8000")))

    # Attempt to use a production-grade server if available
    try:
        from waitress import serve  # type: ignore[import-not-found,import-untyped]
    except ImportError:  # pragma: no cover - import guarded for optional dep
        serve = None

    if serve is not None:
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
