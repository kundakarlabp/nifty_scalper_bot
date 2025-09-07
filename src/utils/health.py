"""Minimal HTTP health check server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Tuple


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # pragma: no cover - simple I/O
        if self.path == "/live":
            self._send(200, {"live": True})
        elif self.path == "/ready":
            ready = getattr(self.server, "ready", False)  # type: ignore[attr-defined]
            self._send(200 if ready else 503, {"ready": bool(ready)})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:  # pragma: no cover
        return

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class HealthServer:
    """Tiny HTTP server exposing ``/live`` and ``/ready`` endpoints."""

    def __init__(self, addr: Tuple[str, int] = ("0.0.0.0", 8000)) -> None:
        self._srv = ThreadingHTTPServer(addr, _Handler)
        self._srv.ready = False  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._srv.shutdown()

    def set_ready(self, ready: bool) -> None:
        self._srv.ready = bool(ready)  # type: ignore[attr-defined]

