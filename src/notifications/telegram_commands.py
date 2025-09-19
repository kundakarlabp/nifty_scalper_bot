"""Telegram command listener using HTTP polling."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Optional

from src.config import settings


def _json_default(value: Any) -> Any:
    """Return a JSON-serializable representation of ``value``."""

    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_default(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()  # type: ignore[attr-defined]
        except Exception:
            return str(value)
        return _json_default(dumped)
    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _json_default(v)
                for k, v in vars(value).items()
                if not k.startswith("_")
            }
        except Exception:
            return str(value)
    return str(value)


def prettify(value: Any) -> str:
    """Return a prettified JSON representation of ``value``."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)
    except Exception:
        return json.dumps(_json_default(value), ensure_ascii=False, indent=2)


def format_block(value: Any, *, max_lines: int = 60) -> str:
    """Format ``value`` as a Telegram-friendly code block."""

    text = prettify(value)
    lines = text.splitlines()
    if max_lines > 0 and len(lines) > max_lines:
        remaining = len(lines) - max_lines
        truncated = lines[:max_lines]
        truncated.append(f"… ({remaining} lines truncated)")
        lines = truncated
    body = "\n".join(lines)
    if body.startswith("```"):
        return body
    return f"```json\n{body}\n```"

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
        settings_obj: Any | None = None,
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
        self.settings = settings_obj or settings
        self.source = source
        self._diag_until_ts = 0.0
        self._trace_until_ts = 0.0

    # --- diagnostic windows -------------------------------------------------
    def _enable_window(self, kind: str, seconds: int) -> float:
        """Enable a temporary logging window for ``seconds`` seconds."""

        try:
            seconds_i = int(seconds)
        except Exception:
            seconds_i = 0
        seconds_i = max(seconds_i, 0)
        until = time.time() + seconds_i
        if kind == "diag":
            self._diag_until_ts = until
        elif kind == "trace":
            self._trace_until_ts = until
        return until

    def _window_active(self, kind: str) -> bool:
        """Return ``True`` if the requested window is still active."""

        now = time.time()
        if kind == "trace":
            return self._trace_until_ts > now
        return self._diag_until_ts > now

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

    def _handle_cmd(self, cmd: str, arg: str) -> bool:
        """Handle built-in commands when no external handler is wired."""

        if self.on_cmd:
            return False

        def reply(text: str) -> None:
            self._send(text)

        if cmd == "/logs":
            args = arg.split()
            if not args or args[0].lower() == "off":
                self._diag_until_ts = 0.0
                self._trace_until_ts = 0.0
                reply("✅ Logging windows disabled (back to baseline).")
                return True
            default_sec = int(getattr(self.settings, "LOG_DIAG_DEFAULT_SEC", 60))
            candidate = default_sec
            target = args[1] if len(args) > 1 and args[0].lower() == "on" else args[0]
            try:
                candidate = int(float(target))
            except Exception:
                candidate = default_sec
            candidate = max(candidate, 0)
            until = self._enable_window("diag", candidate)
            reply(
                f"🟢 Logging window enabled for {candidate}s (diag) until {int(until)}."
            )
            return True
        if cmd == "/trace":
            default_sec = int(getattr(self.settings, "LOG_TRACE_DEFAULT_SEC", 30))
            try:
                seconds = int(float(arg.strip())) if arg.strip() else default_sec
            except Exception:
                seconds = default_sec
            seconds = max(seconds, 0)
            self._enable_window("trace", seconds)
            reply(f"🟠 TRACE window enabled for {seconds}s. Expect more DEBUG logs.")
            return True
        if cmd == "/diag":
            try:
                from src.diagnostics.healthkit import snapshot_pipeline

                snap = snapshot_pipeline()
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("diagnostic snapshot failed")
                reply(f"⚠️ diagnostic snapshot failed: {exc}")
                return True
            max_lines = int(getattr(self.settings, "LOG_MAX_LINES_REPLY", 120))
            reply(format_block(snap, max_lines=max_lines))
            return True
        if cmd == "/micro":
            source = getattr(self, "source", None)
            if source is None:
                reply("⚠️ Data source unavailable.")
                return True
            tokens_fn = getattr(source, "current_tokens", None)
            try:
                ce_token, pe_token = tokens_fn() if callable(tokens_fn) else (None, None)
            except Exception:
                ce_token, pe_token = (None, None)
            micro = {}
            getter = getattr(source, "get_micro_state", None)
            if callable(getter):
                micro["CE"] = getter(ce_token) if ce_token else None
                micro["PE"] = getter(pe_token) if pe_token else None
            max_lines = int(getattr(self.settings, "LOG_MAX_LINES_REPLY", 120))
            reply(format_block(micro, max_lines=max_lines))
            return True
        if cmd == "/quotes":
            source = getattr(self, "source", None)
            if source is None:
                reply("⚠️ Data source unavailable.")
                return True
            tokens_fn = getattr(source, "current_tokens", None)
            try:
                ce_token, pe_token = tokens_fn() if callable(tokens_fn) else (None, None)
            except Exception:
                ce_token, pe_token = (None, None)
            snapshots: dict[str, Any] = {}
            snap_fn = getattr(source, "quote_snapshot", None)
            if callable(snap_fn):
                snapshots["CE"] = snap_fn(ce_token) if ce_token else None
                snapshots["PE"] = snap_fn(pe_token) if pe_token else None
            max_lines = int(getattr(self.settings, "LOG_MAX_LINES_REPLY", 120))
            reply(format_block(snapshots, max_lines=max_lines))
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
