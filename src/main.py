from __future__ import annotations

import logging
import os
import threading
import time

from flask import Flask, jsonify

from src.config import settings
from src.notifications.telegram_controller import TelegramController
from src.strategies.runner import StrategyRunner

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = Flask("src.server.health")

runner = StrategyRunner()


# ---------------- HTTP health ----------------

@app.get("/health")
def health():
    s = runner.status()
    return jsonify({"ok": True, **s})


def _heartbeat_loop():
    while True:
        try:
            s = runner.status()
            log.info("⏱ heartbeat | live=%d paused=%s active=%d",
                     1 if s.get("live_trading") else 0,
                     s.get("paused"),
                     s.get("active_orders", 0))
        except Exception as e:
            log.warning("heartbeat error: %s", e)
        time.sleep(600)  # every 10 min


def _runner_loop():
    while True:
        try:
            runner.run_once()
        except Exception as e:
            log.exception("runner top-level error: %s", e)
        # default cadence (adjust in data source if you want faster)
        time.sleep(60)


def _logs_provider(n: int) -> str:
    # very simple tail of container logs if available
    # On Railway, you usually can't read system logs from file.
    # You can log-buffer in memory; here we just return a stub.
    return "use platform logs; in-app buffer not configured"


def _wire_telegram() -> TelegramController | None:
    tg = getattr(settings, "telegram", None)
    token = getattr(tg, "bot_token", None)
    chat_id = getattr(tg, "chat_id", None)

    if not token or not chat_id or not getattr(tg, "enabled", True):
        log.info("Telegram not started (missing bot token, missing chat id, or disabled).")
        return None

    ctrl = TelegramController(
        status_provider=runner.status,
        diag_provider=runner.diag,
        logs_provider=_logs_provider,
        positions_provider=runner.executor.positions_summary,
        actives_provider=runner.executor.active_orders,
        tick_once=runner.run_once,
        runner_pause=runner.pause,
        runner_resume=runner.resume,
        cancel_all=runner.executor.cancel_all,
        set_risk_pct=lambda pct: setattr(settings.risk, "risk_per_trade", float(pct) / 100.0),
        toggle_trailing=lambda b: setattr(settings.executor, "enable_trailing", bool(b)),
        set_trailing_mult=lambda v: setattr(settings.executor, "trailing_atr_multiplier", float(v)),
        toggle_partial=lambda b: setattr(settings.executor, "partial_tp_enable", bool(b)),
        set_tp1_ratio=lambda pct: setattr(settings.executor, "tp1_qty_ratio", float(pct) / 100.0),
        set_breakeven_ticks=lambda t: setattr(settings.executor, "breakeven_ticks", int(t)),
        set_live_mode=lambda b: setattr(settings, "enable_live_trading", bool(b)),
        set_quality_mode=runner.set_quality_mode,
        set_regime_mode=runner.set_regime_mode,
    )
    ctrl.start_polling()
    ctrl.send_startup_alert()
    log.info("Telegram polling thread started.")
    return ctrl


def main():
    log.info("Starting Nifty Scalper Bot | live_trading=%s", settings.enable_live_trading)

    # start background loops
    threading.Thread(target=_heartbeat_loop, name="heartbeat", daemon=True).start()
    threading.Thread(target=_runner_loop, name="runner", daemon=True).start()

    _wire_telegram()

    # start health server
    app.run(host=settings.server.host, port=settings.server.port, debug=False)


if __name__ == "__main__":
    main()