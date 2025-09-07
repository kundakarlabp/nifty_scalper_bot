"""Minimal HTTP health check server."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional


@dataclass
class HealthState:
    """Tracks runtime health information for readiness probes."""

    started_ts: float = field(default_factory=time.time)
    last_tick_ts: float = 0.0
    broker_connected: bool = False
    max_tick_age_s: float = 3.0


STATE: HealthState = HealthState()
_srv: Optional[HTTPServer] = None
_thr: Optional[threading.Thread] = None


class _Handler(BaseHTTPRequestHandler):
    """Simple handler serving `/live` and `/ready` endpoints."""

    def do_GET(self) -> None:  # pragma: no cover - I/O bound
        if self.path.startswith("/live"):
            self._json({"status": "live"}, 200)
            return
        if self.path.startswith("/ready"):
            age = time.time() - STATE.last_tick_ts if STATE.last_tick_ts else 9e9
            ready = STATE.broker_connected and age <= STATE.max_tick_age_s
            self._json(
                {"status": "ready" if ready else "not_ready", "tick_age_s": age},
                200 if ready else 503,
            )
            return
        self._json({"status": "ok"}, 200)

    def log_message(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover
        """Silence default request logging."""
        return

    def _json(self, payload: Dict[str, Any], code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_health_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the health server in a background thread."""
    global _srv, _thr
    if _srv:
        return
    _srv = HTTPServer((host, int(port)), _Handler)
    _thr = threading.Thread(target=_srv.serve_forever, daemon=True)
    _thr.start()


def stop_health_server() -> None:
    """Stop the health server if running."""
    global _srv, _thr
    if not _srv:
        return
    try:
        _srv.shutdown()
    finally:
        _srv.server_close()
    if _thr:
        _thr.join(timeout=1)
    _srv = None
    _thr = None
