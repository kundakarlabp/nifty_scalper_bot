from __future__ import annotations

"""Telegram command listener using HTTP polling."""

import json
import logging
import threading
import time
from pprint import pformat
from typing import Any, Callable, Optional

from src.config import settings as global_settings


def prettify(value: Any) -> str:
    """Return a JSON-ish representation that is friendly for Telegram."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    except TypeError:
        return pformat(value, compact=True)


def format_block(value: Any, *, max_lines: int | None = None) -> str:
    """Wrap ``value`` in a fenced block, truncating excessive lines."""

    text = prettify(value)
    lines = text.splitlines()
    limit = max_lines if max_lines is not None else len(lines)
    if limit and len(lines) > limit:
        remainder = len(lines) - limit
        lines = lines[:limit] + [f"… (+{remainder} lines)"]
    body = "\n".join(lines)
    return f"```\n{body}\n```"

logger = logging.getLogger(__name__)


class TelegramCommands:
    """Poll a Telegram bot for commands and invoke a callback."""

    def __init__(
        self,
        bot_token: Optional[str],
        chat_id: Optional[str],
        on_cmd: Optional[Callable[[str, str], None]] = None,
        backtest_runner: Optional[Callable[[Optional[str]], str]] = None,
        *,
        settings: Any | None = None,
        source: Any | None = None,
    ) -> None:
        self.token = bot_token or ""
        self.chat = str(chat_id or "")
        self.on_cmd = on_cmd
        self._backtest_runner = backtest_runner
        self._offset: Optional[int] = None
        self._running = False
        self._th: Optional[threading.Thread] = None
        # runtime state for UX commands
        self.basis = "premium"
        self.unit_notional = 0.0
        self.lots = 0
        self.caps = 0.0
        self.risk_pct = 0.0
        self.exposure_mode = "premium"
        self.paused_until = 0.0
        self.settings = settings or global_settings
        self.source = source
        self._diag_until_ts = 0.0
        self._trace_until_ts = 0.0

    def start(self) -> None:
        """Start polling for commands."""
        if not (self.token and self.chat):
            logger.warning("Telegram disabled (token/chat_id missing)")
            return
        if self._running:
            return
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        logger.info("TelegramCommands started")

    def stop(self) -> None:
        """Stop polling and wait for the worker thread to exit."""
        self._running = False
        if self._th:
            self._th.join(timeout=1)
            self._th = None

    def _loop(self) -> None:
        import requests

        base = f"https://api.telegram.org/bot{self.token}"
        while self._running:
            try:
                params = {"timeout": 30}
                if self._offset is not None:
                    params["offset"] = self._offset
                r = requests.get(f"{base}/getUpdates", params=params, timeout=35)
                if r.status_code != 200:
                    time.sleep(1)
                    continue
                for upd in r.json().get("result", []):
                    self._offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    text = (msg.get("text") or "").strip()
                    chat_id = str((msg.get("chat") or {}).get("id") or "")
                    if not text or chat_id != self.chat:
                        continue
                    cmd, *rest = text.split(maxsplit=1)
                    arg = rest[0] if rest else ""
                    if self._handle_cmd(cmd, arg):
                        continue
                    if self.on_cmd:
                        try:
                            self.on_cmd(cmd, arg)
                        except Exception:
                            logger.exception("cmd handler failed")
            except requests.exceptions.ReadTimeout:
                continue
            except Exception:
                logger.exception("telegram polling error")
                time.sleep(1)

    # ------------ command helpers ------------
    def _send(self, text: str) -> None:
        if not (self.token and self.chat):
            return
        try:
            import requests

            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat, "text": text},
                timeout=10,
            )
        except Exception:
            logger.exception("telegram send error")

    def _enable_window(self, kind: str, seconds: int) -> float:
        """Enable ``kind`` window for ``seconds`` and return expiry timestamp."""

        try:
            seconds_i = max(int(seconds), 0)
        except Exception:
            seconds_i = 0
        until = time.time() + seconds_i if seconds_i > 0 else 0.0
        if kind == "diag":
            self._diag_until_ts = until
        elif kind == "trace":
            self._trace_until_ts = until
        return until

    def _window_active(self, kind: str) -> bool:
        """Return ``True`` when ``kind`` window is currently active."""

        if kind == "diag":
            expiry = self._diag_until_ts
        elif kind == "trace":
            expiry = self._trace_until_ts
        else:
            expiry = 0.0
        return float(expiry) > time.time()

    def _current_tokens(self) -> tuple[int | None, int | None]:
        source = self.source
        if source is None:
            return (None, None)
        getter = getattr(source, "current_tokens", None)
        if callable(getter):
            try:
                tokens = getter()
            except Exception:
                return (None, None)
            if isinstance(tokens, (list, tuple)):
                ce = tokens[0] if len(tokens) > 0 else None
                pe = tokens[1] if len(tokens) > 1 else None
                return (ce, pe)
        tokens_attr = getattr(source, "atm_tokens", None)
        if isinstance(tokens_attr, (list, tuple)) and tokens_attr:
            ce = tokens_attr[0] if len(tokens_attr) > 0 else None
            pe = tokens_attr[1] if len(tokens_attr) > 1 else None
            return (ce, pe)
        return (None, None)

    def _get_micro_state(self, token: int | None) -> Any:
        if not token or self.source is None:
            return None
        getter = getattr(self.source, "get_micro_state", None)
        if not callable(getter):
            return None
        try:
            return getter(token)
        except Exception:
            return None

    def _quote_snapshot(self, token: int | None) -> Any:
        if not token or self.source is None:
            return None
        getter = getattr(self.source, "quote_snapshot", None)
        if not callable(getter):
            return None
        try:
            return getter(token)
        except Exception:
            return None

    def _handle_cmd(self, cmd: str, arg: str) -> bool:
        """Handle built-in commands when no external handler is wired."""

        if self.on_cmd:
            return False

        reply = self._send
        if cmd == "/logs":
            parts = [p for p in arg.split() if p]
            if not parts or parts[0].lower() == "off":
                self._diag_until_ts = 0.0
                self._trace_until_ts = 0.0
                reply("✅ Logging windows disabled (back to baseline).")
                return True
            default = int(getattr(self.settings, "LOG_DIAG_DEFAULT_SEC", 60))
            seconds = default
            head = parts[0].lower()
            if head == "on" and len(parts) > 1:
                try:
                    seconds = int(parts[1])
                except Exception:
                    seconds = default
            elif head not in {"on", "off"}:
                try:
                    seconds = int(head)
                except Exception:
                    seconds = default
            until = int(self._enable_window("diag", seconds))
            reply(
                f"🟢 Logging window enabled for {seconds}s (diag) until {until}."
            )
            return True
        if cmd == "/trace":
            default = int(getattr(self.settings, "LOG_TRACE_DEFAULT_SEC", 30))
            try:
                seconds = int(arg.split()[0]) if arg.split() else default
            except Exception:
                seconds = default
            self._enable_window("trace", seconds)
            reply(
                f"🟠 TRACE window enabled for {seconds}s. Expect more DEBUG logs."
            )
            return True
        if cmd == "/diag":
            try:
                from src.diagnostics.healthkit import snapshot_pipeline

                snap = snapshot_pipeline()
            except Exception as exc:
                reply(f"Diag snapshot failed: {exc}")
                return True
            text = format_block(
                snap,
                max_lines=int(getattr(self.settings, "LOG_MAX_LINES_REPLY", 30)),
            )
            reply(text)
            return True
        if cmd == "/micro":
            ce, pe = self._current_tokens()
            payload = {
                "CE": self._get_micro_state(int(ce)) if ce else None,
                "PE": self._get_micro_state(int(pe)) if pe else None,
            }
            text = format_block(
                payload,
                max_lines=int(getattr(self.settings, "LOG_MAX_LINES_REPLY", 30)),
            )
            reply(text)
            return True
        if cmd == "/quotes":
            ce, pe = self._current_tokens()
            payload = {
                "CE": self._quote_snapshot(int(ce)) if ce else None,
                "PE": self._quote_snapshot(int(pe)) if pe else None,
            }
            text = format_block(
                payload,
                max_lines=int(getattr(self.settings, "LOG_MAX_LINES_REPLY", 30)),
            )
            reply(text)
            return True
        if cmd == "/pause":
            mins = 0
            if arg.endswith("m"):
                try:
                    mins = int(arg[:-1])
                except ValueError:
                    mins = 0
            self.paused_until = time.time() + mins * 60
            self._send(f"Paused for {mins}m")
            return True
        if cmd == "/risk":
            try:
                self.risk_pct = float(arg.strip("%"))
            except ValueError:
                self.risk_pct = 0.0
            self._send(f"risk_pct set to {self.risk_pct}%")
            return True
        if cmd == "/exposure":
            if arg in ("premium", "underlying"):
                self.exposure_mode = arg
            self._send(f"exposure mode = {self.exposure_mode}")
            return True
        if cmd == "/flatten":
            self.lots = 0
            self.unit_notional = 0.0
            self._send("positions flattened")
            return True
        if cmd == "/status" and arg == "brief":
            msg = (
                f"basis={self.basis} unit_notional={self.unit_notional} "
                f"lots={self.lots} caps={self.caps}"
            )
            self._send(msg)
            return True
        if cmd == "/backtest":
            if not self._backtest_runner:
                self._send("Backtest not available.")
                return True
            path = arg.strip() or None
            try:
                result = self._backtest_runner(path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("backtest runner failed")
                result = f"Backtest error: {exc}"
            if result:
                self._send(result)
            return True
        return False
