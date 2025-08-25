from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional, Callable, Dict, Any, List

from src.config import settings
from src.strategies.runner import StrategyRunner

# Optional broker SDK (live only)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = None  # type: ignore


# -----------------------------
# Logging
# -----------------------------
def _setup_logging() -> None:
    level_name = getattr(settings, "log_level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------
# Tiny no-op Telegram (fallback)
# -----------------------------
class _NoopTelegram:
    def __init__(self) -> None:
        self._started = False

    def send_message(self, *_a, **_k) -> None:
        pass

    def start_polling(self) -> None:
        self._started = True

    def stop_polling(self) -> None:
        self._started = False


# -----------------------------
# Helpers
# -----------------------------
def _build_kite() -> Optional["KiteConnect"]:
    log = logging.getLogger("main")
    if not getattr(settings, "enable_live_trading", False):
        log.info("Live trading disabled → paper mode.")
        return None
    if KiteConnect is None:
        raise RuntimeError("ENABLE_LIVE_TRADING=true but kiteconnect is not installed.")
    api_key = settings.zerodha.api_key
    access_token = settings.zerodha.access_token
    if not api_key or not access_token:
        raise RuntimeError("Missing Zerodha credentials while live mode is enabled.")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def _tail_logs(n: int = 120, path: str = "trading_bot.log") -> List[str]:
    """Return last n lines from log file; safe if file missing."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 4096
            data = b""
            while size > 0 and data.count(b"\n") <= n:
                read_size = block if size >= block else size
                size -= read_size
                f.seek(size)
                data = f.read(read_size) + data
        text = data.decode(errors="ignore")
        return text.splitlines()[-n:]
    except Exception:
        return []


def _import_telegram_class():
    """
    Import TelegramController lazily and safely. We *never* crash the bot
    just because Telegram failed — we fall back to a no-op transport.
    """
    try:
        from src.notifications.telegram_controller import TelegramController  # type: ignore
        return TelegramController
    except Exception as e:
        logging.getLogger("main").error(
            "TelegramController import failed — running with no‑op Telegram. "
            "Cause: %s (telegram_controller.py)", e
        )
        return None


def _wire_real_telegram(runner: StrategyRunner):
    """
    If TelegramController is available, wire providers from the runner.
    Otherwise keep the no-op instance the runner already has.
    """
    TelegramController = _import_telegram_class()
    if TelegramController is None:
        return  # keep no-op

    tg = TelegramController(
        # status / introspection
        status_provider=runner.get_status_snapshot,
        positions_provider=getattr(runner.executor, "get_positions_kite", None),
        actives_provider=getattr(runner.executor, "get_active_orders", None),
        diag_provider=runner.get_diag_snapshot,         # rich diag
        logs_provider=_tail_logs,                       # /logs [n]
        last_signal_provider=runner.get_last_signal_debug,
        # runner controls
        runner_pause=runner.pause,
        runner_resume=runner.resume,
        runner_tick=runner.runner_tick,                 # accepts dry=bool
        cancel_all=getattr(runner.executor, "cancel_all_orders", None),
        # tuning
        set_live_mode=runner.set_live_mode,
    )

    # Keep both attributes (compat with legacy code)
    runner.telegram_controller = tg
    runner.telegram = tg

    try:
        tg.start_polling()
    except Exception:
        logging.getLogger("main").warning("Telegram polling failed to start; continuing.")


# -----------------------------
# App lifecycle
# -----------------------------
_stop_flag = False


def _install_signal_handlers(runner: StrategyRunner) -> None:
    def _handler(signum, _frame):
        nonlocal runner  # clarity only
        log = logging.getLogger("main")
        log.info("Signal %s received — shutting down…", signum)
        global _stop_flag
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
        getattr(settings, "enable_live_trading", False),
        settings.data.time_filter_start,
        settings.data.time_filter_end,
        getattr(settings.strategy, "rr_min", 0.0),
    )

    kite = _build_kite()

    # 1) Create runner with a temporary no-op telegram so ctor won't raise
    runner = StrategyRunner(kite=kite, telegram_controller=_NoopTelegram())
    _install_signal_handlers(runner)

    # 2) Try wiring your real TelegramController (keeps no-op on failure)
    _wire_real_telegram(runner)

    # announce
    try:
        runner.telegram_controller.send_message("🚀 Bot starting (shadow mode by default).")
    except Exception:
        log.warning("Telegram startup message failed (continuing).")

    # Start runner if it has a start() method
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
            runner.telegram_controller.send_message("🛑 Bot stopped.")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.getLogger("main").exception("Fatal error in main: %s", e)
        sys.exit(1)