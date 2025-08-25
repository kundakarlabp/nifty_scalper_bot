# Path: src/main.py
from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional, List

from src.config import settings
from src.strategies.runner import StrategyRunner

# -----------------------------
# Logging
# -----------------------------
def _setup_logging() -> None:
    level = getattr(logging, (getattr(settings, "log_level", "INFO") or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # quiet noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------
# Fallback, in case Telegram import fails
# -----------------------------
class _NoopTelegram:
    def __init__(self, *_, **__):
        pass

    def send_message(self, *_a, **_k) -> None:
        pass

    def send_startup_alert(self) -> None:
        pass

    def start_polling(self) -> None:
        pass

    def stop_polling(self) -> None:
        pass


# -----------------------------
# Helpers
# -----------------------------
def _tail_logs(n: int = 100, path: str = "trading_bot.log") -> List[str]:
    """
    Return last n lines from a log file. Safe if the file is missing.
    Used by /logs via telegram controller.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                read_size = min(block, size)
                size -= read_size
                f.seek(size)
                data = f.read(read_size) + data
        text = data.decode(errors="ignore")
        return text.splitlines()[-n:]
    except Exception:
        return []


def _import_kite():
    try:
        from kiteconnect import KiteConnect  # type: ignore
        return KiteConnect
    except Exception as e:
        logging.getLogger("main").info("kiteconnect import unavailable (%s) — staying in paper mode.", e)
        return None


def _build_kite() -> Optional["KiteConnect"]:
    KiteConnect = _import_kite()
    if not bool(getattr(settings, "enable_live_trading", False)):
        logging.getLogger("main").info("Live trading disabled → paper mode.")
        return None
    if KiteConnect is None:
        logging.getLogger("main").warning("ENABLE_LIVE_TRADING=true but kiteconnect is not installed.")
        return None

    api_key = getattr(settings.zerodha, "api_key", None)
    access_token = getattr(settings.zerodha, "access_token", None)
    if not api_key or not access_token:
        logging.getLogger("main").error("Missing Zerodha credentials while live mode is enabled.")
        return None

    kite = KiteConnect(api_key=api_key)
    try:
        kite.set_access_token(access_token)
    except Exception as e:
        logging.getLogger("main").error("Failed to set Kite access token: %s", e)
        return None
    return kite


def _import_telegram_class():
    """
    Try to import TelegramController and return the class or None.
    Logs the exception (so you see *why* it failed) instead of crashing.
    """
    try:
        from src.notifications.telegram_controller import TelegramController  # type: ignore
        return TelegramController
    except Exception as e:
        logging.getLogger("main").error(
            "TelegramController import failed — running with no-op Telegram. "
            "Cause: %s", e, exc_info=True
        )
        return None


def _wire_telegram(runner: StrategyRunner):
    """
    Build Telegram controller if available; otherwise attach a no-op controller.
    This function never raises — the bot must continue to run.
    """
    TelegramController = _import_telegram_class()
    tg = None
    if TelegramController is not None:
        try:
            tg = TelegramController(
                # providers from runner/executor
                status_provider=runner.get_status_snapshot,
                positions_provider=getattr(runner.executor, "get_positions_kite", None),
                actives_provider=getattr(runner.executor, "get_active_orders", None),
                diag_provider=runner.get_last_flow_debug,
                logs_provider=_tail_logs,  # enables /logs [n]
                last_signal_provider=runner.get_last_signal_debug,
                # controls
                runner_pause=runner.pause,
                runner_resume=runner.resume,
                runner_tick=runner.runner_tick,  # supports dry=bool
                cancel_all=getattr(runner.executor, "cancel_all_orders", None),
                # risk/strategy mutators (wire later if you expose these)
                set_live_mode=runner.set_live_mode,
            )
            tg.start_polling()
        except Exception as e:
            logging.getLogger("main").error("Creating/starting Telegram controller failed: %s", e, exc_info=True)
            tg = None

    if tg is None:
        tg = _NoopTelegram()

    # keep both names for backward compatibility
    runner.telegram_controller = tg
    runner.telegram = tg
    return tg


# -----------------------------
# App lifecycle
# -----------------------------
_stop_flag = False


def _install_signal_handlers(_runner: StrategyRunner) -> None:
    def _handler(signum, _frame):
        global _stop_flag
        logging.getLogger("main").info("Signal %s received — shutting down…", signum)
        _stop_flag = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def main() -> int:
    _setup_logging()
    log = logging.getLogger("main")

    log.info(
        "Boot | live=%s window=%s-%s rr_min=%.2f",
        bool(getattr(settings, "enable_live_trading", False)),
        getattr(settings.data, "time_filter_start", "09:20"),
        getattr(settings.data, "time_filter_end", "15:20"),
        float(getattr(getattr(settings, "strategy", object()), "rr_min", 0.0) or 0.0),
    )

    kite = _build_kite()

    # create runner with a temporary no-op telegram so ctor can't raise
    runner = StrategyRunner(kite=kite, telegram_controller=_NoopTelegram())
    _install_signal_handlers(runner)

    # now try to wire the real Telegram controller (or a no-op if unavailable)
    tg = _wire_telegram(runner)

    # announce (only does anything if real Telegram is active)
    try:
        tg.send_message("🚀 Bot starting (shadow mode by default).")
    except Exception:
        log.warning("Telegram startup message failed (continuing).")

    # start loop if runner has .start()
    try:
        if hasattr(runner, "start"):
            runner.start()
    except Exception as e:
        log.exception("Runner start failed: %s", e)
        return 1

    # health loop
    try:
        while not _stop_flag:
            try:
                if hasattr(runner, "health_check"):
                    runner.health_check()
            except Exception as e:
                log.warning("Health check warn: %s", e)
            time.sleep(5)
    finally:
        try:
            if hasattr(runner, "shutdown"):
                runner.shutdown()
        except Exception:
            pass
        try:
            tg.send_message("🛑 Bot stopped.")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.getLogger("main").exception("Fatal error in main: %s", e)
        sys.exit(1)