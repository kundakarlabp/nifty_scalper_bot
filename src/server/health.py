# src/server/health.py
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
status_callback: Optional[Callable[[], Dict[str, Any]]] = None
_start_ts = time.time()
log = logging.getLogger(__name__)


@app.route("/live", methods=["GET"])
def live() -> tuple[Response, int] | tuple[Dict[str, Any], int]:
    """Liveness probe: process is up."""
    return jsonify({"status": "alive", "uptime_sec": int(time.time() - _start_ts)}), 200


@app.route("/ready", methods=["GET"])
def ready() -> tuple[Response, int] | tuple[Dict[str, Any], int]:
    """
    Readiness probe: returns 200 with status if callback is set, else 503.
    (If you want to gate on 'is_trading' or other fields, add logic here.)
    """
    try:
        if not status_callback:
            return jsonify({"ready": False, "reason": "status callback not configured"}), 503
        status = status_callback() or {}
        status.update({"ready": True, "uptime_sec": int(time.time() - _start_ts)})
        return jsonify(status), 200
    except Exception as e:
        log.error("Readiness callback error: %s", e, exc_info=True)
        return jsonify({"ready": False, "error": str(e)}), 500


@app.route("/health", methods=["GET", "HEAD"])
def health() -> tuple[Response, int] | tuple[Dict[str, Any], int]:
    """General health/status endpoint. Supports GET and HEAD."""
    try:
        # For HEAD, return empty body but still 200 quickly.
        if request.method == "HEAD":
            return Response(status=200), 200

        if status_callback:
            status = status_callback() or {}
        else:
            status = {"status": "ok", "message": "Status callback not configured."}
        status.setdefault("uptime_sec", int(time.time() - _start_ts))
        return jsonify(status), 200
    except Exception as e:
        log.error("Health endpoint error: %s", e, exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


def run(
    callback: Optional[Callable[[], Dict[str, Any]]] = None,
    *,
    host: str = "0.0.0.0",
    port: Optional[int] = None,
) -> None:
    """
    Start the lightweight health server. Safe for background threads.
    - Disables reloader & debug to avoid 'signal only works in main thread' errors.
    - Threaded=True to handle concurrent probes.
    - Port can be overridden via HEALTH_PORT env or 'port' kwarg.
    """
    global status_callback
    status_callback = callback
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    port = int(port or os.getenv("HEALTH_PORT", "8000"))
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
