# src/notifications/telegram_listener.py
"""
Legacy-compatible Telegram command listener.

This script exists for setups that previously launched a standalone
Telegram listener. It wires up the existing TelegramController with
dummy callbacks (log-only) and starts polling. No trading actions
are performed here.

Usage:
    python -m src.notifications.telegram_listener
    # or
    python src/notifications/telegram_listener.py
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Any, Dict

# --- optional .env auto-loader so this script is standalone-friendly ---
def _load_dotenv_if_present() -> None:
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        here = Path(__file__).resolve()
        for p in (here.parent / ".env", here.parent.parent / ".env", Path.cwd() / ".env"):
            if p.exists():
                load_dotenv(p)
                logging.getLogger(__name__).info("🔐 Loaded environment from %s", p)
                return
        load_dotenv()  # fallback: cwd default behavior if present
    except Exception:
        # It's fine if dotenv isn't installed — controller will use env as-is
        pass

_load_dotenv_if_present()
# --- end env loader ---

# Prefer your packaged controller; fall back to flat import for older trees
try:
    from src.notifications.telegram_controller import TelegramController
except (ImportError, ModuleNotFoundError):
    try:
        from telegram_controller import TelegramController  # legacy flat file
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import TelegramController. Ensure src/notifications/telegram_controller.py "
            "(or telegram_controller.py) is on PYTHONPATH."
        ) from exc


# ---- logging ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("telegram_listener")


# ---- dummy callbacks (log-only) -------------------------------------------
def _dummy_status() -> Dict[str, Any]:
    """
    Returned to /status in legacy mode. Does NOT query a live trader.
    """
    logger.info("⚠️ /status received (legacy listener; no live trader attached).")
    return {
        "is_trading": False,
        "live_mode": False,
        "trades_today": 0,
        "open_positions": 0,
        "daily_pnl": 0.0,
        "session_date": None,
    }


def _dummy_control(cmd: str, arg: str = "") -> bool:
    """
    Log-only handler for commands like /start, /stop, /mode, etc.
    """
    cmd = (cmd or "").strip()
    arg = (arg or "").strip()
    logger.info("📩 Command received (no action taken): /%s %s", cmd, arg)
    return True


def _dummy_summary() -> str:
    """
    Returned to /summary in legacy mode.
    """
    logger.info("⚠️ /summary requested (no trade history in legacy mode).")
    return "<b>No trading context available in legacy listener mode.</b>"


# ---- main -----------------------------------------------------------------
def main() -> None:
    try:
        controller = TelegramController(
            status_callback=_dummy_status,
            control_callback=_dummy_control,
            summary_callback=_dummy_summary,
        )
    except TypeError as e:
        logger.error("Failed to initialize TelegramController: %s", e)
        logger.info("💡 Ensure TelegramController accepts status/control/summary callbacks.")
        return
    except Exception as e:
        logger.error("TelegramController init error: %s", e, exc_info=True)
        return

    # graceful shutdown wiring
    stopping = {"flag": False}

    def _stop(_sig=None, _frm=None):
        if stopping["flag"]:
            return
        stopping["flag"] = True
        logger.info("🛑 Stopping Telegram listener...")
        try:
            controller.stop_polling()
        except Exception:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except Exception:
            pass

    logger.info("📡 Telegram command listener started (legacy compatibility mode).")
    controller.start_polling()

    # keep main thread alive
    try:
        while not stopping["flag"]:
            time.sleep(1.0)
    finally:
        _stop()
        logger.info("✅ Telegram listener stopped gracefully.")


if __name__ == "__main__":
    # allow `python -m src.notifications.telegram_listener`
    # and `python src/notifications/telegram_listener.py`
    try:
        main()
    except SystemExit:
        # let normal sys.exit() pass through
        raise
    except Exception as e:
        logger.error("Fatal error in telegram_listener: %s", e, exc_info=True)
        sys.exit(1)