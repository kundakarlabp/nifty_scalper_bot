"""HTML response builder helpers for Telegram diagnostics outputs."""

from __future__ import annotations

from html import escape as _escape
import os
from typing import Iterable, Optional, Tuple

EMOJI = {
    "ok": "✅",
    "fail": "❌",
    "warn": "⚠️",
    "info": "ℹ️",
    "live": "🟢",
    "shadow": "🟡",
    "off": "⚫",
    "ws": "🌐",
    "poll": "🔁",
    "session": "🕘",
    "risk": "🧯",
    "orders": "🧾",
    "pos": "📊",
}


class ResponseBuilder:
    """Utility helpers for consistent HTML-escaped Telegram responses."""

    def __init__(self) -> None:
        val = os.getenv("TELEGRAM_UI_COMPACT", "").strip().lower()
        self.ui_compact = val in {"1", "true", "yes", "on"}

    def br(self, n: int = 1) -> str:
        n = 1 if self.ui_compact else max(1, n)
        return "<br>" * n

    def sep(self) -> str:
        """Return a section separator respecting the UI density toggle."""

        return self.br(2 if not self.ui_compact else 1)

    def bullets(self, rows: list[str]) -> str:
        """Join ``rows`` with single ``<br>`` separators."""

        return "<br>".join(rows)

    def section_block(self, title: str, lines: list[str]) -> str:
        """Render a titled section followed by bullet-style ``lines``."""

        body = self.bullets(lines)
        return f"{self.section(title)}{self.br()}{body}"

    def esc(self, value: object) -> str:
        """Return the HTML-escaped string representation of ``value``."""

        return _escape(str(value), quote=False)

    def badge(self, ok: bool, label: str) -> str:
        """Return a status badge with an emoji and bold label."""

        emoji = EMOJI["ok" if ok else "fail"]
        return f"{emoji} <b>{self.esc(label)}</b>"

    def kv(self, key: str, val: str) -> str:
        """Render a bold key with a ``code`` value."""

        return f"<b>{self.esc(key)}</b>: <code>{self.esc(val)}</code>"

    def grid(self, rows: Iterable[Tuple[str, str]]) -> str:
        """Render multiple key/value rows separated by ``<br>``."""

        return "<br>".join(self.kv(key, val) for key, val in rows)

    def section(self, title: str) -> str:
        """Render a section title."""

        return f"<b>{self.esc(title)}</b>"

    def truncate_lines(self, text: str, max_lines: int) -> tuple[str, int, bool]:
        """Truncate ``text`` to ``max_lines`` lines if necessary."""

        lines = text.splitlines()
        if max_lines and len(lines) > max_lines:
            return "\n".join(lines[:max_lines]), max_lines, True
        return text, len(lines), False

    def mono(self, text: str, max_lines: Optional[int] = None) -> str:
        """Render ``text`` inside a ``<pre>`` block with optional truncation."""

        truncated = False
        if max_lines is not None:
            text, _shown, truncated = self.truncate_lines(text, max_lines)
        escaped = self.esc(text)
        if truncated:
            escaped += "\n…"
        return f"<pre>{escaped}</pre>"

    def footer(self, page: int, total_pages: Optional[int]) -> str:
        """Render a pagination footer if ``page`` and ``total_pages`` are provided."""

        if page <= 0 or not total_pages:
            return ""
        return f"<i>— page {page}/{total_pages} —</i>"


RB = ResponseBuilder()

__all__ = ["EMOJI", "ResponseBuilder", "RB"]
