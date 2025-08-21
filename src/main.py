from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from typing import Any, Optional

from src.config import settings
from src.server import health as health_server
from src.strategies.runner import StrategyRunner
from src.notifications.telegram_controller import TelegramController
from src.utils.logging_tools import log_buffer_handler, get_recent_logs

# Optional broker SDK (safe import)
try:
    from kiteconnect import KiteConnect  # type: ignore
    from kiteconnect.exceptions import NetworkException, TokenException, InputException  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore
    NetworkException = TokenException = InputException = Exception  # fallbacks


log = logging.getLogger(__name__)


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    # In-memory buffer (for /logs)
    if not any(h is log_buffer_handler for h in root.handlers):
        root.addHandler(log_buffer_handler)


class Application:
    def __init__(self) -> None:
        self.runner = StrategyRunner(event_sink=self._event_sink)
        self.tg: Optional[TelegramController] = None
        self._start_ts = time.time()
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._last_live_flag = bool(getattr(settings, "enable_live_trading", False))

    # ---- graceful shutdown
    def _handle_signal(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        log.info("Received signal %s, starting graceful shutdown…", signum)
        self._stop_event.set()

    # ---- Telegram event sink from runner
    def _event_sink(self, evt: dict) -> None:
        if not self.tg:
            return
        t = evt.get("type")
        if t == "ENTRY_PLACED":
            self.tg.notify_entry(
                symbol=evt.get("symbol", "?"),
                side=evt.get("side", "?"),
                qty=int(evt.get("qty", 0)),
                price=float(evt.get("price", 0.0)),
                record_id=str(evt.get("record_id", "")),
            )
        elif t == "FILLS":
            self.tg.notify_fills(evt.get("fills") or [])
        # We intentionally do not spam on SIGNAL; use /summary.

    # ---- Telegram providers
    def _status_payload(self) -> dict:
        return self.runner.to_status_dict()

    def _positions_provider(self) -> dict:
        if self.runner.executor:
            return self.runner.executor.get_positions_kite()
        return {}

    def _actives_provider(self):
        if self.runner.executor:
            return self.runner.executor.get_active_orders()
        return []

    def _summary_provider(self) -> str:
        s = self.runner.to_status_dict()
        sig = s.get("last_signal")
        if not sig:
            return "No recent signal."
        return (
            "📈 Last signal\n"
            f"{sig.get('t')}\n"
            f"{sig.get('side')} | conf={sig.get('confidence')} | score={sig.get('score')}\n"
            f"entry={sig.get('entry_price')}  sl={sig.get('stop_loss')}  tp={sig.get('target')}"
        )

    # ---- main
    def run(self) -> None:
        _setup_logging()
        log.info("Starting Nifty Scalper Bot | live_trading=%s", bool(getattr(settings, "enable_live_trading", False)))

        # Health server
        threading.Thread(
            target=health_server.run,
            kwargs={"callback": self._status_payload, "host": settings.server.host, "port": settings.server.port},
            daemon=True,
        ).start()

        # Telegram
        try:
            tg_enabled = bool(getattr(settings.telegram, "enabled", True))
            bot_token = getattr(settings.telegram, "bot_token", None)
            chat_id = getattr(settings.telegram, "chat_id", None)
            if tg_enabled and bot_token and chat_id:
                self.tg = TelegramController(
                    status_provider=self._status_payload,
                    positions_provider=self._positions_provider,
                    actives_provider=self._actives_provider,
                    logs_provider=lambda n=60: get_recent_logs(n=n),
                    summary_provider=self._summary_provider,
                    runner_pause=self.runner.pause,
                    runner_resume=self.runner.resume,
                    cancel_all=(self.runner.executor.cancel_all_orders if self.runner.executor else None),
                    set_live_mode=self._set_live_mode,
                )
                self.tg.start_polling()
                self.tg.send_startup_alert()
            else:
                log.info("Telegram not started (disabled or credentials missing).")
        except Exception as e:
            log.warning("Telegram controller not started: %s", e)
            self.tg = None

        # Main loop with heartbeat
        cadence = 0.5
        last_heartbeat = 0.0
        while not self._stop_event.is_set():
            try:
                live_now = bool(getattr(settings, "enable_live_trading", False))
                if live_now != self._last_live_flag:
                    self._last_live_flag = live_now
                    log.info("Live mode set to %s.", "True" if live_now else "False")

                self.runner.run_once(stop_event=self._stop_event)

                # heartbeat every 60s so Railway logs show activity
                now = time.time()
                if now - last_heartbeat >= 60:
                    st = self._status_payload()
                    log.info(
                        "⏱ heartbeat | live=%s paused=%s active=%s src=%s",
                        "1" if st.get("live_trading") else "0",
                        st.get("paused"),
                        st.get("active_orders"),
                        st.get("data_source"),
                    )
                    last_heartbeat = now

            except (NetworkException, TokenException, InputException) as e:
                log.error("Transient broker error: %s", e)
            except Exception as e:
                log.exception("Main loop error: %s", e)

            if self._stop_event.wait(timeout=cadence):
                break

        if self.tg:
            try:
                self.tg.stop_polling()
            except Exception:
                pass
        log.info("Bot stopped.")

    # Telegram hook
    def _set_live_mode(self, val: bool) -> None:
        setattr(settings, "enable_live_trading", bool(val))
        log.info("Live mode set to %s.", "True" if val else "False")


# ---- CLI ----
def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nifty_scalper_bot")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("start", help="Start trading loop (default)")
    sub.add_parser("backtest", help="Run backtest from CSV file")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cmd = args.cmd or "start"
    if cmd == "start":
        Application().run()
    elif cmd == "backtest":
        log.info("Backtest command not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())