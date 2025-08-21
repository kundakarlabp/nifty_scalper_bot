from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Optional

from flask import Flask, jsonify

from src.config import settings
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

log = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = Flask("src.server.health")

runner: Optional[StrategyRunner] = None
tg: Optional[TelegramController] = None
_stop = threading.Event()


@app.get("/health")
def health():
    status = runner.to_status_dict() if runner else {}
    return jsonify({"ok": True, "status": status})


def _heartbeat_loop() -> None:
    while not _stop.is_set():
        s = runner.to_status_dict() if runner else {}
        src = s.get("data_source")
        log.info("⏱ heartbeat | live=%s paused=%s active=%s src=%s",
                 int(bool(s.get("live_trading"))), s.get("paused"), s.get("active_orders"), src)
        time.sleep(60)


def _start_telegram() -> None:
    global tg
    if not settings.telegram_ready:
        reason = []
        if not settings.telegram.enabled:
            reason.append("disabled")
        if not settings.telegram.bot_token:
            reason.append("missing bot token")
        if not settings.telegram.chat_id:
            reason.append("missing chat id")
        log.info("Telegram not started (%s).", ", ".join(reason) if reason else "not ready")
        return

    try:
        tg = TelegramController(
            status_provider=(runner.to_status_dict if runner else (lambda: {})),
            positions_provider=(runner.executor.get_positions_kite if runner and runner.executor else None),
            actives_provider=(runner.executor.get_active_orders if runner and runner.executor else None),
            runner_pause=(runner.pause if runner else None),
            runner_resume=(runner.resume if runner else None),
            cancel_all=(runner.executor.cancel_all_orders if runner and runner.executor else None),
            # strategy/executor tunables exposed via /config already
        )
        tg.start_polling()
        log.info("Telegram polling started.")
    except Exception as e:
        log.exception("Telegram init failed: %s", e)


def _main() -> None:
    global runner
    log.info("Starting Nifty Scalper Bot | live_trading=%s", settings.enable_live_trading)

    runner = StrategyRunner()
    # start telegram after runner so providers are wired
    _start_telegram()

    # health HTTP server
    t_http = threading.Thread(target=lambda: app.run(host=settings.server.host, port=settings.server.port, debug=False), daemon=True)
    t_http.start()

    # heartbeat
    t_hb = threading.Thread(target=_heartbeat_loop, daemon=True)
    t_hb.start()

    # graceful shutdown
    def _sigterm(_signo, _frame):
        log.info("Received signal %s, shutting down…", _signo)
        _stop.set()

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    # main runner loop (service trades even off-hours; entries gated inside)
    while not _stop.is_set():
        try:
            runner.run_once(_stop)
        except Exception as e:
            log.exception("run_once crashed: %s", e)
        time.sleep(5)


if __name__ == "__main__":
    _main()