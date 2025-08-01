"""
Backward-compatibility wrapper for the Telegram controller.

In the original codebase, the Telegram bot command listener was a
standalone script. This refactored version rolls command handling
directly into ``TelegramController``. The separate listener is
therefore no longer required, but this file remains to avoid import
errors for existing tooling.

Running this module directly will simply instantiate a ``TelegramController``
and begin polling for commands. The controller does not perform any
trading itself; instead, it logs incoming commands.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Callable

# Attempt to import the real controller
try:
    from src.notifications.telegram_controller import TelegramController
except (ImportError, ModuleNotFoundError):
    try:
        from telegram_controller import TelegramController
    except ImportError:
        raise ImportError(
            "Could not find 'TelegramController'. "
            "Make sure 'telegram_controller.py' is in the path."
        )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Dummy callbacks for legacy mode (logging only)
def _dummy_status() -> Dict[str, Any]:
    logger.info("⚠️ /status received, but no live trader connected.")
    return {"is_trading": False, "live_mode": False, "trades_today": 0, "open_orders": 0}


def _dummy_control(cmd: str, arg: str = "") -> bool:
    logger.info(f"📩 Command received (no action): /{cmd} {arg}".strip())
    return True


def _dummy_summary() -> str:
    logger.info("⚠️ /summary requested, but no trade history available.")
    return "No trading context available in legacy listener mode."


def main() -> None:
    """Start polling for Telegram commands with dummy callbacks."""
    try:
        controller = TelegramController(
            status_callback=_dummy_status,
            control_callback=_dummy_control,
            summary_callback=_dummy_summary,
        )
    except TypeError as e:
        logger.error(f"Failed to initialize TelegramController: {e}")
        logger.info("💡 Hint: Ensure your TelegramController accepts optional callbacks.")
        return

    logger.info("📡 Telegram command listener started (legacy compatibility mode).")
    controller.start_polling()

    try:
        while True:
            pass  # Keep main thread alive
    except KeyboardInterrupt:
        logger.info("🛑 Telegram listener stopped via keyboard interrupt.")
        controller.stop_polling()


if __name__ == "__main__":
    main()