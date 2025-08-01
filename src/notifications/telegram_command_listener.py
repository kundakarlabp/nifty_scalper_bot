"""
Backwardâ€‘compatibility wrapper for the Telegram controller.

In the original codebase the Telegram bot command listener was a
standalone script.  This refactored version rolls command handling
directly into ``TelegramController``.  The separate listener is
therefore no longer required, but this file remains to avoid import
errors for existing tooling.

Running this module directly will simply instantiate a ``TelegramController``
and begin polling for commands.  The controller does not perform any
trading itself; instead, it logs incoming commands.
"""

from __future__ import annotations

import logging
from telegram_controller import TelegramController

logger = logging.getLogger(__name__)


def main() -> None:
    """Start polling for Telegram commands and log them."""
    controller = TelegramController()
    logger.info("Telegram command listener started (legacy mode).")
    controller.start_polling()


if __name__ == "__main__":
    main()