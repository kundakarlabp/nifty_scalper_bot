from __future__ import annotations
from telegram import Update
from telegram.constants import ParseMode
from nifty_scalper_bot.utils.logging import get_logger

LOG = get_logger(__name__)

def redact(text: str) -> str:
    """Redact sensitive keys/tokens from error messages."""
    if not text: return ""
    # Basic redaction logic - expand based on your needs
    for key in ["token", "secret", "password", "api_key"]:
        if key in text.lower():
             return "[REDACTED]"
    return text

async def reply_ok(update: Update, text: str) -> None:
    """Send a success message (Green check) to the chat."""
    if not update.effective_message: return
    try:
        # Escape HTML special chars if needed, or assume text is safe
        await update.effective_message.reply_text(
            f"✅ {text}", 
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
    except Exception as exc:
        LOG.warning("Failed to reply_ok: %s", exc)

async def reply_err(update: Update, text: str) -> None:
    """Send an error message (Red cross) to the chat."""
    if not update.effective_message: return
    try:
        await update.effective_message.reply_text(
            f"❌ {text}", 
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )
    except Exception as exc:
        LOG.warning("Failed to reply_err: %s", exc)
