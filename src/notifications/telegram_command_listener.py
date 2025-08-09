"""
Backward-compatibility wrapper for the Telegram controller.

In older setups this script was run standalone. It now just wires up
dummy callbacks (when no live trader is bound) and starts the Telegram
poller in a non-blocking daemon thread.

You can still run:
    python -m src.notifications.telegram_command_listener
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Any, Dict

# Try importing the real controller (project layout first, then flat file fallback)
try:
    from src.notifications.telegram_controller import TelegramController
except (ImportError, ModuleNotFoundError):
    try:
        from telegram_controller import TelegramController  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not find 'TelegramController'. Ensure it exists at "
            "'src/notifications/telegram_controller.py' or alongside this file."
        ) from e

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------- dummy callbacks -------------------------- #

def _dummy_status() -> Dict[str, Any]:
    logger.info("⚠️ /status received, but no live trader connected.")
    return {"is_trading": False, "live_mode": False, "trades_today": 0, "open_orders": 0, "daily_pnl": 0.0}

def _dummy_control(cmd: str, arg: str = "") -> bool:
    logger.info("📩 Command received (no action): /%s %s", cmd, arg)
    return True

def _dummy_summary() -> str:
    logger.info("⚠️ /summary requested, but no trade history available.")
    return "No trading context available in legacy listener mode."


# ------------------------------ main -------------------------------- #

def main() -> None:
    """Start polling for Telegram commands with dummy callbacks (non-blocking)."""
    try:
        controller = TelegramController(
            status_callback=_dummy_status,
            control_callback=_dummy_control,
            summary_callback=_dummy_summary,
        )
    except Exception as e:
        logger.error("Failed to initialize TelegramController: %s", e, exc_info=True)
        logger.info("💡 Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your environment/.env")
        sys.exit(1)

    # Start the poller (daemon thread; returns immediately)
    controller.start_polling()
    logger.info("📡 Telegram command listener started (legacy compatibility mode).")

    # Graceful shutdown on SIGINT/SIGTERM
    def _shutdown(signum, _frame):
        logger.info("🛑 Signal %s received. Stopping Telegram polling...", signum)
        controller.stop_polling()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Keep the process alive without burning CPU
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)


if __name__ == "__main__":
    main()
