"""Simple HTTP health endpoints for liveness and readiness."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional


class _Handler(BaseHTTPRequestHandler):
    """Serve ``/live`` and ``/ready`` endpoints."""

    ready_fn: Optional[Callable[[], bool]] = None

    def do_GET(self) -> None:  # noqa: N802 (framework method)
        if self.path == "/live":
            self._send(200, b"ok")
        elif self.path == "/ready":
            if self.ready_fn and self.ready_fn():
                self._send(200, b"ok")
            else:
                self._send(503, b"not ready")
        else:
            self._send(404, b"not found")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        """Suppress default stdout logging."""

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)


_server: HTTPServer | None = None


def start(port: int = 8000, ready: Callable[[], bool] | None = None) -> None:
    """Start a background HTTP server exposing health endpoints."""

    global _server
    if _server is not None:
        return

    _Handler.ready_fn = ready
    _server = HTTPServer(("0.0.0.0", port), _Handler)
    thread = threading.Thread(target=_server.serve_forever, daemon=True)
    thread.start()


def stop() -> None:
    """Shutdown the health server if running."""

    global _server
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _server = None

