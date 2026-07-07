from __future__ import annotations

from typing import Any

from telegram.error import NetworkError, TimedOut

from nifty_scalper_bot.utils.logging import get_logger

LOG = get_logger(__name__)
_PATCHED = False
_ORIGINAL_START_POLLING: Any = None


def classify_updater_polling_error(exc: object) -> str:
    text = str(exc or "")
    low = text.lower()
    etype = type(exc).__name__ if exc else "None"
    if "conflict" in low or "terminated by other getupdates" in low:
        LOG.error("TELEGRAM_POLLING_CONFLICT type=%s err=%s", etype, text, extra={"event": "TELEGRAM_POLLING_CONFLICT", "error_type": etype})
        return "conflict"
    if isinstance(exc, (NetworkError, TimedOut)) or "timed out" in low or "timeout" in low:
        LOG.warning("telegram_polling_transient type=%s err=%s", etype, text, extra={"event": "telegram_polling_transient", "error_type": etype})
        return "transient"
    LOG.error("TELEGRAM_UPDATER_ERROR type=%s err=%s", etype, text, extra={"event": "TELEGRAM_UPDATER_ERROR", "error_type": etype})
    return "unknown"


def ensure_polling_error_callback(kwargs: dict[str, Any]) -> dict[str, Any]:
    if kwargs.get("error_callback") is None:
        kwargs["error_callback"] = classify_updater_polling_error
    return kwargs


def apply_patch() -> bool:
    global _PATCHED, _ORIGINAL_START_POLLING
    if _PATCHED:
        return False
    try:
        from telegram.ext import Updater
    except Exception:
        return False
    if getattr(Updater, "_nifty_error_callback_patch", False):
        _PATCHED = True
        return False
    _ORIGINAL_START_POLLING = Updater.start_polling

    async def start_polling(self: Any, *args: Any, **kwargs: Any) -> Any:
        ensure_polling_error_callback(kwargs)
        return await _ORIGINAL_START_POLLING(self, *args, **kwargs)

    Updater.start_polling = start_polling
    Updater._nifty_error_callback_patch = True
    _PATCHED = True
    return True


__all__ = ["apply_patch", "classify_updater_polling_error", "ensure_polling_error_callback"]
