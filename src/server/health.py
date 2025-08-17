# src/server/health.py
"""
Health check server
-------------------
Purpose:
- Small Flask app exposing /health for Railway/uptime pings.

Key points / changes:
- ✅ Simple JSON with status, uptime, mode (LIVE/PAPER/TEST).
- ✅ No debug banner in production unless FLASK_DEBUG=1.
- ✅ run(port=...) callable used by main.py; suitable under threads/WGSI.

Tip:
- In production, prefer a WSGI server (gunicorn/uvicorn) if you need more.
"""

from __future__ import annotations
import os
from datetime import datetime
from flask import Flask, jsonify

try:
    from src.config import Config
except Exception:
    Config = None  # type: ignore

app = Flask(__name__)
_start = datetime.utcnow()


@app.get("/health")
def health():
    uptime = (datetime.utcnow() - _start).total_seconds()
    resp = {"status": "ok", "uptime_sec": round(uptime, 2)}
    if Config:
        resp["mode"] = (
            "LIVE" if getattr(Config, "ENABLE_LIVE_TRADING", False)
            else "PAPER" if getattr(Config, "ALLOW_OFFHOURS_TESTING", False)
            else "TEST"
        )
    return jsonify(resp), 200


def run(port: int = 8000) -> None:
    # Respect FLASK_DEBUG if set, else keep quiet in prod.
    debug = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=int(port or 8000), debug=debug, use_reloader=False)