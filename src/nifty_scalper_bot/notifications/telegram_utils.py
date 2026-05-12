from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nifty_scalper_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from telegram import Update as TelegramUpdate
else:
    TelegramUpdate = Any

LOG = get_logger(__name__)

try:
    from telegram.constants import ParseMode as _ParseMode
except Exception:
    _ParseMode = None


def _reply_kwargs() -> dict[str, Any]:
    """Build safe reply kwargs. Args: none. Returns: dict. Raises: none."""
    kwargs: dict[str, Any] = {"disable_web_page_preview": True}
    if _ParseMode is not None:
        kwargs["parse_mode"] = _ParseMode.HTML
    return kwargs

def redact(text: str) -> str:
    """Redact sensitive keys/tokens from error messages."""
    if not text: return ""
    for key in ("token", "secret", "password", "api_key", "access_token"):
        if key in text.lower():
             return "[REDACTED]"
    return text

async def reply_ok(update: TelegramUpdate, text: str) -> None:
    """Send a success message (Green check) to the chat."""
    message = getattr(update, "effective_message", None)
    if message is None:
        return
    try:
        await message.reply_text(f"✅ {text}", **_reply_kwargs())
    except Exception as exc:
        LOG.warning("telegram_reply_ok_failed error=%s", exc)

async def reply_err(update: TelegramUpdate, text: str) -> None:
    """Send an error message (Red cross) to the chat."""
    message = getattr(update, "effective_message", None)
    if message is None:
        return
    try:
        await message.reply_text(f"❌ {text}", **_reply_kwargs())
    except Exception as exc:
        LOG.warning("telegram_reply_err_failed error=%s", exc)
