"""
Health check server for the trading bot.

Exposes /health endpoint with:
- status: always "ok" if server alive
- uptime: seconds since server start
- mode: LIVE / PAPER / TEST from Config
- last_trade: timestamp + status of last trade (if available)
- drawdown_active: True if circuit breaker triggered
"""

from __future__ import annotations

from flask import Flask, jsonify
from datetime import datetime
import os

# Import config and (optionally) the trader using correct package paths
try:
    from src.config import Config
except Exception:
    Config = None  # type: ignore

try:
    # This import can be heavy; it's optional and guarded.
    from src.data_streaming.realtime_trader import RealTimeTrader  # type: ignore
except Exception:
    RealTimeTrader = None  # type: ignore

app = Flask(__name__)
start_time = datetime.utcnow()

@app.get("/health")
def health():
    uptime = (datetime.utcnow() - start_time).total_seconds()

    resp = {
        "status": "ok",
        "uptime_sec": round(uptime, 2),
    }

    # Mode from config
    if Config is not None:
        try:
            resp["mode"] = (
                "LIVE" if getattr(Config, "ENABLE_LIVE_TRADING", False) else
                "PAPER" if getattr(Config, "ALLOW_OFFHOURS_TESTING", False) else
                "TEST"
            )
        except Exception:
            pass

    # Optional extras from RealTimeTrader (if you later expose them as class-level attrs)
    try:
        if RealTimeTrader is not None and hasattr(RealTimeTrader, "last_trade_info"):
            resp["last_trade"] = getattr(RealTimeTrader, "last_trade_info")
    except Exception:
        resp["last_trade"] = None

    try:
        if RealTimeTrader is not None and hasattr(RealTimeTrader, "drawdown_triggered"):
            resp["drawdown_active"] = bool(getattr(RealTimeTrader, "drawdown_triggered"))
        else:
            resp["drawdown_active"] = False
    except Exception:
        resp["drawdown_active"] = False

    return jsonify(resp), 200


def run():
    # Reads port from environment (set by main.py)
    port = int(os.getenv("HEALTH_PORT", "8000"))
    app.run(host="0.0.0.0", port=port)