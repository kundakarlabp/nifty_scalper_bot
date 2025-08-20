# Lightweight health server for Railway/Render probes.

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
log = logging.getLogger(__name__)
status_callback: Optional[Callable[[], Dict[str, Any]]] = None
_start_ts = time.time()


@app.route("/live", methods=["GET"])
def live() -> tuple[Dict[str, Any], int]:
    return {"status": "live", "uptime_sec": int(time.time() - _start_ts)}, 200


@app.route("/ready", methods=["GET"])
def ready() -> tuple[Dict[str, Any], int]:
    # You can wire any deeper checks here if needed
    return {"status": "ready"}, 200


# IMPORTANT: HEAD route explicitly present (fixes .head decorator error)
@app.route("/health", methods=["HEAD"])
def health_head() -> tuple[Response, int]:
    return Response(status=200), 200


@app.route("/health", methods=["GET"])
def health_get() -> tuple[Dict[str, Any], int]:
    try:
        status = status_callback() if status_callback else {}
        status.setdefault("status", "ok")
        status.setdefault("uptime_sec", int(time.time() - _start_ts))
        return status, 200
    except Exception as e:
        log.exception("Health GET error: %s", e)
        return {"status": "error", "error": str(e)}, 500


def run(callback: Optional[Callable[[], Dict[str, Any]]] = None, host: Optional[str] = None, port: Optional[int] = None) -> None:
    """
    Start the health server (thread-safe). Use only for probes; WSGI not required.
    """
    global status_callback
    status_callback = callback
    host = host or os.environ.get("HEALTH_HOST", "0.0.0.0")
    port = int(port or os.environ.get("HEALTH_PORT", "8000"))
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)