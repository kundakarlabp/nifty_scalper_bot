from __future__ import annotations

import re

from telegram import Message, Update

from nifty_scalper_bot.utils.logging import get_logger

LOG = get_logger(__name__)
REDACT_KEYS: tuple[str, ...] = (
    "token",
    "secret",
    "apikey",
    "api_key",
    "access",
    "password",
)
_REDACT_PATTERN = re.compile(
    "|".join(re.escape(key) for key in REDACT_KEYS), re.IGNORECASE
)


def redact(text: str) -> str:
    """Redact likely secret tokens from *text* for safe logging.

    Args:
        text: Raw string that may include sensitive fragments.

    Returns:
        Sanitised string with known sensitive keywords replaced by asterisks.

    Raises:
        None.
    """

    LOG.debug("Entered redact length=%d", len(text))
    return _REDACT_PATTERN.sub("***", text)


async def reply_ok(update: Update, text: str) -> None:
    """Send a success response associated with *update*.

    Args:
        update: Telegram update carrying the origin chat/message.
        text: Message body to deliver.

    Returns:
        None.

    Raises:
        None.
    """

    LOG.debug("Entered reply_ok")
    message: Message | None = update.effective_message
    if message is None:
        LOG.warning("reply_ok missing effective message")
        return
    try:
        await message.reply_text(text)
        LOG.info("reply_ok delivered message length=%d", len(text))
    except Exception as exc:  # noqa: BLE001 - defensive
        LOG.error("Failure in reply_ok: %s", exc)


async def reply_err(update: Update, text: str) -> None:
    """Send an error response for *update* with a warning prefix.

    Args:
        update: Telegram update carrying the origin chat/message.
        text: Error text to return to the operator.

    Returns:
        None.

    Raises:
        None.
    """

    LOG.debug("Entered reply_err")
    await reply_ok(update, f"❗ {text}")
