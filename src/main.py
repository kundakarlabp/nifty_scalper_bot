# Path: src/main.py
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import Optional

from src.config import settings
from src.strategies.runner import StrategyRunner

# Optional broker SDK (only built when live)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

# Your existing Telegram controller from the ZIP (unchanged interface expected)
try:
    from src.notifications.telegram_controller import TelegramController  # type: ignore
except Exception as e:  # pragma: no cover
    TelegramController = None  # type: ignore


# -----------------------------
# Logging
# -----------------------------
def _setup_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Squash very chatty libs if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------
# Builders
# -----------------------------
def _build_kite() -> Optional["KiteConnect"]:
    """Create KiteConnect only in live mode; otherwise return None (paper mode)."""
    if not settings.enable_live_trading:
        logging.getLogger("main").info("Live trading disabled → paper mode.")
        return None

    if KiteConnect is None:
        raise RuntimeError("kiteconnect library not available but ENABLE_LIVE_TRADING=true")

    api_key = settings.zerodha.api_key
    access_token = settings.zerodha.access_token
    if not api_key or not access_token:
        raise RuntimeError("Missing Zerodha credentials while live mode is enabled.")

    kite = KiteConnect(api_key=api_key)
    # In many setups access_token is set on the underlying session
    kite.set_access_token(access_token)
    logging.getLogger("main").info("KiteConnect initialized (live mode).")
    return kite


def _build_telegram() -> "TelegramController":
    """Telegram is mandatory. Try both common constructor styles to stay compatible
    with the ZIP’s original controller (positional vs keyword)."""
    if TelegramController is None:
        raise RuntimeError("TelegramController module not found (but it is required).")

    token = settings.telegram.bot_token
    chat_id = settings.telegram.chat_id
    if not token or not chat_id:
        raise RuntimeError("Telegram credentials missing: TELEGRAM__BOT_TOKEN / TELEGRAM__CHAT_ID.")

    # 1) Try (bot_token, chat_id) positional
    try:
        return TelegramController(token, chat_id)  # type: ignore[call-arg]
    except TypeError:
        pass

    # 2) Try keyword form
    try:
        return TelegramController(bot_token=token, chat_id=chat_id)  # type: ignore[call-arg]
    except TypeError:
        pass

    # 3) Fallback to most basic, then set attributes if controller exposes them
    tg = TelegramController()  # type: ignore[call-arg]
    # If the controller expects setters / attributes, set them defensively
    if hasattr(tg, "bot_token"):
        tg.bot_token = token  # type: ignore[attr-defined]
    if hasattr(tg, "chat_id"):
        tg.chat_id = chat_id  # type: ignore[attr-defined]
    return tg


# -----------------------------
# App lifecycle
# -----------------------------
_stop_flag = False


def _install_signal_handlers(runner: StrategyRunner) -> None:
    def _handler(signum, _frame):
        global _stop_flag
        logging.getLogger("main").info(f"Signal {signum} received — shutting down…")
        _stop_flag = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


def main() -> int:
    _setup_logging()
    log = logging.getLogger("main")

    # Config was validated on import; log key toggles for clarity
    log.info(
        "Boot | live=%s window=%s-%s rr_min=%.2f",
        settings.enable_live_trading,
        settings.data.time_filter_start,
        settings.data.time_filter_end,
        getattr(settings.strategy, "rr_min", 0.0),
    )

    # Build integrations
    telegram = _build_telegram()
    kite = _build_kite()

    # Strategy runner (expects Telegram mandatory, we pass it)
    runner = StrategyRunner(kite=kite, telegram_controller=telegram)
    _install_signal_handlers(runner)

    # Announce boot
    try:
        telegram.send_message("🚀 Bot starting (shadow mode by default).")
    except Exception:
        log.warning("Telegram startup message failed (continuing).")

    # Start the runner
    try:
        # Your runner exposes start()/health_check()/shutdown(), and receives ticks from your data stack.
        # If ticks are pushed via websockets elsewhere, we just keep a health loop here.
        if hasattr(runner, "start"):
            runner.start()
    except Exception as e:
        log.exception("Runner start failed: %s", e)
        return 1

    # Lightweight health loop
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
            telegram.send_message("🛑 Bot stopped.")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.getLogger("main").exception("Fatal error in main: %s", e)
        sys.exit(1)