# src/server/health.py
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

# Import config and realtime trader safely
try:
    from config import Config
    from realtime_trader import RealTimeTrader  # optional, may be heavy
except Exception:
    Config = None
    RealTimeTrader = None

app = Flask(__name__)
start_time = datetime.utcnow()

@app.get("/health")
def health():
    uptime = (datetime.utcnow() - start_time).total_seconds()

    # Base response
    resp = {
        "status": "ok",
        "uptime_sec": round(uptime, 2),
    }

    # Mode from config
    if Config:
        resp["mode"] = (
            "LIVE" if Config.ENABLE_LIVE_TRADING else
            "PAPER" if Config.ALLOW_OFFHOURS_TESTING else
            "TEST"
        )

    # Last trade info if RealTimeTrader exposes it
    try:
        if RealTimeTrader and hasattr(RealTimeTrader, "last_trade_info"):
            resp["last_trade"] = RealTimeTrader.last_trade_info
    except Exception:
        resp["last_trade"] = None

    # Circuit breaker / drawdown
    if RealTimeTrader and hasattr(RealTimeTrader, "drawdown_triggered"):
        resp["drawdown_active"] = bool(RealTimeTrader.drawdown_triggered)
    else:
        resp["drawdown_active"] = False

    return jsonify(resp), 200


def run():
    port = int(os.getenv("HEALTH_PORT", 8000))
    app.run(host="0.0.0.0", port=port)