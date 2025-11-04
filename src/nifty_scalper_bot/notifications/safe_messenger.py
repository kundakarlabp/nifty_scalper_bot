"""Telegram messenger helper that guarantees valid output delivery."""

from __future__ import annotations

import html
from io import BytesIO

from telegram.constants import ParseMode
from telegram.error import BadRequest

from nifty_scalper_bot.utils.metrics import Counter

MAX = 4000
SUFFIX = "\n\n...(truncated)"

_SEND_FAILURES = Counter(
    "telegram_send_failures_total",
    "Total Telegram message send failures",
)


class SafeMessenger:
    """Send Telegram messages with escaping, truncation, and safe fallbacks."""

    def __init__(self, bot) -> None:  # type: ignore[no-untyped-def]
        self.bot = bot

    def _truncate(self, text: str, max_len: int = MAX) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - len(SUFFIX)] + SUFFIX

    async def send_text(
        self,
        chat_id: int,
        text: str,
        *,
        html_ready: bool = False,
        parse_mode: ParseMode | None = ParseMode.HTML,
        **kwargs,
    ):  # type: ignore[no-untyped-def]
        """Send plain or HTML text with safe fallbacks."""

        raw_text = str(text)
        if parse_mode is None:
            body = self._truncate(raw_text)
        else:
            body = self._truncate(raw_text if html_ready else html.escape(raw_text))
        try:
            return await self.bot.send_message(
                chat_id=chat_id,
                text=body,
                parse_mode=parse_mode,
                **kwargs,
            )
        except BadRequest:
            try:
                _SEND_FAILURES.inc()
            except Exception:  # pragma: no cover - optional metrics
                pass
            if parse_mode is not None:
                try:
                    return await self.bot.send_message(
                        chat_id=chat_id,
                        text=self._truncate(raw_text),
                        parse_mode=None,
                        **kwargs,
                    )
                except BadRequest:
                    try:
                        _SEND_FAILURES.inc()
                    except Exception:  # pragma: no cover - optional metrics
                        pass
            payload = BytesIO(raw_text.encode("utf-8"))
            payload.name = "output.txt"
            return await self.bot.send_document(
                chat_id=chat_id,
                document=payload,
                caption="Output too long",
            )

    async def send_html(
        self,
        chat_id: int,
        html_text: str,
        *,
        parse_mode: ParseMode = ParseMode.HTML,
        **kwargs,
    ):  # type: ignore[no-untyped-def]
        return await self.send_text(
            chat_id, html_text, html_ready=True, parse_mode=parse_mode, **kwargs
        )


__all__ = ["MAX", "SUFFIX", "SafeMessenger"]
