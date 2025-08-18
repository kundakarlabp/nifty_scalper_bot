# src/server/health.py
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Dict, Any


def run(status_getter: Callable[[], Dict[str, Any]], host: str = "0.0.0.0", port: int | None = None) -> None:
    """
    Lightweight HTTP health server. Exposes:
      GET /healthz  -> {"status":"ok", **status_getter()}
      GET /readyz   -> 200 if status_getter() doesn't raise

    Designed to be started on a daemon thread.
    """
    _port = int(port or os.environ.get("PORT", "8080"))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:  # quieter logs
            return

        def _write_json(self, code: int, payload: Dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            try:
                if self.path.startswith("/healthz"):
                    status = status_getter() or {}
                    status = {"status": "ok", **status}
                    self._write_json(200, status)
                    return
                if self.path.startswith("/readyz"):
                    _ = status_getter()
                    self._write_json(200, {"ready": True})
                    return
                self._write_json(404, {"error": "not found"})
            except Exception as e:
                self._write_json(500, {"status": "error", "message": str(e)})

    server = HTTPServer((host, _port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
