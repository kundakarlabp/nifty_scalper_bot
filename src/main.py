from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from collections import deque
from typing import Any, Optional, List

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController

try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks

# -------- logging setup with ring buffer for /logs --------
_log = logging.getLogger(__name__)
RING: deque[str] = deque(maxlen=800)  # ~last 800 lines

class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            RING.append(msg)
        except Exception:
            pass

def _setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # attach ring
    rh = _RingHandler()
    rh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(rh)


def _logs_provider(n: int) -> List[str]:
    return list(RING)[-n:]


class Application:
    def __init__(self) -> None:
        self.runner = StrategyRunner(event_sink=self._handle_runner_event)
        self._tg: Optional[TelegramController] = None
        self._stop = threading.Event()
        self._start_ts = time.time()

        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        _log.info("Signal %s; shutting down…", signum)
        self._stop.set()

    def _status(self) -> dict:
        s = self.runner.to_status_dict()
        s["uptime_sec"] = int(time.time() - self._start_ts)
        return s

    def _handle_runner_event(self, evt: dict) -> None:
        if not self._tg:
            return
        t = evt.get("type")
        if t == "ENTRY_PLACED":
            self._tg.notify_entry(
                symbol=evt.get("symbol", "?"),
                side=evt.get("side", "?"),
                qty=int(evt.get("qty", 0)),
                price=float(evt.get("price", 0.0)),
                record_id=str(evt.get("record_id", "")),
            )
        elif t == "FILLS":
            fills = evt.get("fills") or []
            self._tg.notify_fills(fills)

    def run(self) -> None:
        _setup_logging()
        _log.info("Starting Nifty Scalper Bot | live_trading=%s", settings.enable_live_trading)

        # Health server
        threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status, "host": settings.server.host, "port": settings.server.port},
            daemon=True,
        ).start()

        # Telegram
        try:
            if settings.telegram.enabled and settings.telegram.bot_token and settings.telegram.chat_id:
                self._tg = TelegramController(
                    status_provider=self._status,
                    positions_provider=(self.runner.executor.get_positions_kite if self.runner.executor else None),
                    actives_provider=(self.runner.executor.get_active_orders if self.runner.executor else None),
                    diag_provider=self.runner.diagnose,
                    runner_pause=self.runner.pause,
                    runner_resume=self.runner.resume,
                    set_live_mode=self.runner.set_live_mode,
                    # runtime knobs -> mutate settings through runner/executor or settings directly
                    set_risk_pct=lambda pct: setattr(settings.risk, "risk_per_trade", float(pct)/100.0),
                    toggle_trailing=lambda v: setattr(settings.executor, "enable_trailing", bool(v)),
                    set_trailing_mult=lambda v: setattr(settings.executor, "trailing_atr_multiplier", float(v)),
                    toggle_partial=lambda v: setattr(settings.executor, "partial_tp_enable", bool(v)),
                    set_tp1_ratio=lambda pct: setattr(settings.executor, "tp1_qty_ratio", float(pct)/100.0),
                    set_breakeven_ticks=lambda k: setattr(settings.executor, "breakeven_ticks", int(k)),
                    logs_provider=_logs_provider,
                )
                self._tg.start_polling()
                self._tg.send_startup_alert()
            else:
                _log.info("Telegram not started (disabled or credentials missing).")
        except Exception as e:
            _log.warning("Telegram init failed: %s", e)
            self._tg = None

        # main loop
        cadence = 1.0
        hb_next = time.time() + 60.0  # 1-min heartbeat
        while not self._stop.is_set():
            try:
                result = self.runner.run_once(stop_event=self._stop)
                if result and _log.isEnabledFor(logging.DEBUG):
                    _log.debug("Signal: %s", result)
            except (NetworkException, TokenException, InputException) as e:
                _log.error("Transient broker error: %s", e)
            except Exception as e:
                _log.exception("Main loop error: %s", e)

            # heartbeat (info every 1 min, can be fetched anytime via /status)
            now = time.time()
            if now >= hb_next:
                st = self.runner.to_status_dict()
                _log.info("⏱ heartbeat | live=%d paused=%s active=%d src=%s",
                          1 if st.get("live_trading") else 0,
                          st.get("paused"), st.get("active_orders"),
                          st.get("data_source"))
                hb_next = now + 60.0

            if self._stop.wait(timeout=cadence):
                break

        if self._tg:
            try: self._tg.stop_polling()
            except Exception: pass
        _log.info("Nifty Scalper Bot stopped.")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"
    if cmd == "start":
        Application().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())