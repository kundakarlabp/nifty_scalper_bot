# src/server/health.py
# Lightweight health server for Railway/Render probes.

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

from flask import Flask, Response

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
    """Readiness probe: lightweight 200; customize if you need deeper checks."""
    return {"status": "ready"}, 200


# Explicit HEAD route to avoid framework quirks on some platforms
@app.route("/health", methods=["HEAD"])
def health_head() -> Tuple[Response, int]:
    """HEAD /health for ultra-cheap probe."""
    return Response(status=200), 200


@app.route("/health", methods=["GET"])
def health_get() -> Tuple[Dict[str, Any], int]:
    """
    GET /health returns extended status.
    If a status callback is provided, merge its dict with defaults.
    """
    try:
        status = _status_callback() if _status_callback else {}
        if not isinstance(status, dict):
            status = {"detail": str(status)}
        status.setdefault("status", "ok")
        status.setdefault("uptime_sec", int(time.time() - _start_ts))
        return status, 200
    except Exception as e:
        log.exception("Health GET error: %s", e)
        return {"status": "error", "error": str(e)}, 500


def run(
    callback: Optional[Callable[[], Dict[str, Any]]] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """
    Start the health server in the current thread.
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
    global _status_callback
    _status_callback = callback

    bind_host = host or os.environ.get("HEALTH_HOST", "0.0.0.0")
    bind_port = int(port or int(os.environ.get("HEALTH_PORT", "8000")))

    # No reloader, multi-threaded to avoid blocking the trading loop
    app.run(host=bind_host, port=bind_port, debug=False, use_reloader=False, threaded=True)
