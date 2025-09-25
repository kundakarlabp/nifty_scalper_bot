from __future__ import annotations

"""Telegram command listener using HTTP polling."""

import json
import logging
import threading
import time
from datetime import datetime
from pprint import pformat
from typing import Any, Callable, Optional, SupportsFloat, cast

from src.config import settings as global_settings
from src.diagnostics import trace_ctl


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
        strategy: Any | None = None,
        risk: Any | None = None,
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
        self.strategy = strategy
        self.risk = risk
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

    def _diag_snapshot(self) -> dict[str, Any]:
        """Build a compact diagnostic snapshot for the trading pipeline."""

        settings_obj = self.settings
        cap_val = getattr(settings_obj, "RISK__EXPOSURE_CAP_PCT", None)
        if cap_val is None:
            cap_val = getattr(settings_obj, "EXPOSURE_CAP_PCT", 0.0)
        cap_pct = 0.0
        try:
            if isinstance(cap_val, (int, float, str)):
                cap_pct = float(cap_val)
            elif hasattr(cap_val, "__float__"):
                cap_pct = float(cast(SupportsFloat, cap_val))
        except (TypeError, ValueError):
            cap_pct = 0.0
        if cap_pct <= 1.0:
            cap_pct *= 100.0

        snapshot: dict[str, Any] = {
            "market_open": False,
            "equity": None,
            "risk": {"daily_dd": None, "cap_pct": cap_pct},
            "signals": {"regime": None, "atr_pct": None, "score": None},
            "micro": {"ce": None, "pe": None},
            "open_orders": 0,
            "positions": 0,
            "latency": {"tick_age": None, "bar_lag": None},
        }

        source = self.source
        tokens = self._current_tokens()
        if tokens[0]:
            snapshot["micro"]["ce"] = self._get_micro_state(int(tokens[0]))
        if len(tokens) > 1 and tokens[1]:
            snapshot["micro"]["pe"] = self._get_micro_state(int(tokens[1]))

        if source is not None:
            tick_ts = getattr(source, "last_tick_ts", None)
            if tick_ts:
                try:
                    snapshot["latency"]["tick_age"] = max(
                        time.time() - float(tick_ts),
                        0.0,
                    )
                except Exception:
                    snapshot["latency"]["tick_age"] = None

        strategy = self.strategy
        risk_obj = self.risk or getattr(strategy, "risk", None)

        if risk_obj is not None:
            daily_dd = getattr(risk_obj, "day_realized_loss", None)
            if daily_dd is None:
                engine_state = getattr(getattr(risk_obj, "state", None), "cum_loss_rupees", None)
                daily_dd = engine_state
            if daily_dd is not None:
                try:
                    snapshot["risk"]["daily_dd"] = round(float(daily_dd), 2)
                except Exception:
                    snapshot["risk"]["daily_dd"] = daily_dd
        elif strategy is not None:
            engine = getattr(strategy, "risk_engine", None)
            state = getattr(engine, "state", None)
            daily_dd = getattr(state, "cum_loss_rupees", None)
            if daily_dd is not None:
                try:
                    snapshot["risk"]["daily_dd"] = round(float(daily_dd), 2)
                except Exception:
                    snapshot["risk"]["daily_dd"] = daily_dd

        if strategy is None:
            return snapshot

        window_fn = getattr(strategy, "_within_trading_window", None)
        if callable(window_fn):
            try:
                snapshot["market_open"] = bool(window_fn(None))
            except TypeError:
                try:
                    snapshot["market_open"] = bool(window_fn())
                except Exception:
                    pass
            except Exception:
                pass
        else:
            market_flag = getattr(strategy, "market_open", None)
            if market_flag is not None:
                snapshot["market_open"] = bool(market_flag)

        equity_cached = getattr(strategy, "_equity_cached_value", None)
        if equity_cached is None:
            equity_cached = getattr(strategy, "equity", None)
        if equity_cached is not None:
            try:
                snapshot["equity"] = round(float(equity_cached), 2)
            except Exception:
                snapshot["equity"] = equity_cached

        plan = getattr(strategy, "last_plan", None)
        if not isinstance(plan, dict):
            plan = getattr(strategy, "_last_signal_debug", None)
        if isinstance(plan, dict):
            for key in ("regime", "atr_pct", "score"):
                if key in plan:
                    snapshot["signals"][key] = plan.get(key)

        executor = getattr(strategy, "executor", None) or getattr(
            strategy, "order_executor", None
        )
        if executor is not None:
            orders_fn = getattr(executor, "get_active_orders", None)
            try:
                open_orders = orders_fn() if callable(orders_fn) else None
            except Exception:
                open_orders = None
            if isinstance(open_orders, (list, tuple, set)):
                snapshot["open_orders"] = len(open_orders)
            elif isinstance(open_orders, dict):
                snapshot["open_orders"] = len(open_orders)
            elif open_orders is not None:
                try:
                    snapshot["open_orders"] = int(open_orders)
                except Exception:
                    pass

            pos_fn = getattr(executor, "get_positions_kite", None)
            try:
                positions = pos_fn() if callable(pos_fn) else None
            except Exception:
                positions = None
            if isinstance(positions, dict):
                snapshot["positions"] = len(positions)
            elif isinstance(positions, (list, tuple, set)):
                snapshot["positions"] = len(positions)
            elif positions:
                snapshot["positions"] = 1

        window = getattr(strategy, "_ohlc_cache", None)
        if window is not None:
            try:
                is_empty = bool(getattr(window, "empty", False))
            except Exception:
                is_empty = False
            if not is_empty:
                try:
                    last_ts = window.index[-1]
                except Exception:
                    last_ts = None
                if last_ts is not None:
                    last_dt = None
                    if hasattr(last_ts, "to_pydatetime"):
                        try:
                            last_dt = last_ts.to_pydatetime()
                        except Exception:
                            last_dt = None
                    elif isinstance(last_ts, datetime):
                        last_dt = last_ts
                    else:
                        try:
                            last_dt = datetime.fromisoformat(str(last_ts))
                        except Exception:
                            last_dt = None
                    if last_dt is not None:
                        now_fn = getattr(strategy, "_now_ist", None)
                        if callable(now_fn):
                            try:
                                now_dt = now_fn()
                            except Exception:
                                now_dt = datetime.utcnow()
                        else:
                            now_dt = datetime.utcnow()
                        if last_dt.tzinfo is None and getattr(now_dt, "tzinfo", None) is not None:
                            last_dt = last_dt.replace(tzinfo=now_dt.tzinfo)
                        try:
                            snapshot["latency"]["bar_lag"] = max(
                                (now_dt - last_dt).total_seconds(),
                                0.0,
                            )
                        except Exception:
                            snapshot["latency"]["bar_lag"] = None

        return snapshot

    def _subscription_lines(self) -> list[str]:
        source = self.source
        if source is None:
            return ["subs=unavailable"]
        getter = getattr(source, "subscription_modes", None)
        if not callable(getter):
            getter = getattr(source, "diag_subscriptions", None)
        if not callable(getter):
            return ["subs=unavailable"]
        try:
            snapshot = list(getter())
        except Exception as exc:  # pragma: no cover - defensive
            return [f"subs_error={exc}"]
        lines = [f"subs_count={len(snapshot)}"]
        for token, mode in snapshot:
            lines.append(f"{token} {mode}")
        if len(lines) == 1:
            lines.append("none")
        return lines

    def _ws_diag_lines(self) -> list[str]:
        source = self.source
        if source is None:
            return ["diag=unavailable"]
        getter = getattr(source, "ws_diag_snapshot", None)
        if not callable(getter):
            return ["diag=unavailable"]
        try:
            snapshot = getter()
        except Exception as exc:  # pragma: no cover - defensive
            return [f"diag_error={exc}"]
        ws_connected = snapshot.get("ws_connected")
        subs_count = snapshot.get("subs_count")
        last_tick_age = snapshot.get("last_tick_age_ms")
        reconnect_left = snapshot.get("reconnect_debounce_left_ms")
        per_token = snapshot.get("per_token_age_ms") or {}
        stale_counts = snapshot.get("stale_counts") or {}

        def _fmt_bool(flag: Any) -> str:
            return "1" if bool(flag) else "0"

        def _fmt_int(val: Any) -> str:
            try:
                return str(int(val))
            except Exception:
                return "na"

        header = (
            "ws_connected="
            + _fmt_bool(ws_connected)
            + f" subs_count={_fmt_int(subs_count)}"
            + f" last_tick_age_ms={_fmt_int(last_tick_age)}"
            + f" reconnect_debounce_left_ms={_fmt_int(reconnect_left)}"
        )

        def _fmt_map(name: str, mapping: dict[Any, Any]) -> str:
            if not mapping:
                return f"{name} none"
            parts = []
            for key in sorted(mapping):
                parts.append(f"{key}={mapping[key]}")
            return f"{name} " + " ".join(str(p) for p in parts)

        lines = [header]
        lines.append(_fmt_map("per_token_age_ms", per_token))
        lines.append(_fmt_map("stale_counts", stale_counts))
        return lines

    def _trigger_fresh_reconnect(self) -> str:
        source = self.source
        if source is None:
            return "fresh reconnect unavailable"
        handler = getattr(source, "force_hard_reconnect", None)
        context = {"origin": "telegram"}
        try:
            if callable(handler):
                handler(reason="telegram_fresh", context=context)
            else:
                fallback = getattr(source, "reconnect_ws", None)
                if not callable(fallback):
                    return "fresh reconnect unavailable"
                fallback(reason="telegram_fresh", context=context)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("fresh reconnect failed")
            return f"fresh reconnect failed: {exc}"
        return "fresh reconnect requested"

    def _handle_cmd(self, cmd: str, arg: str) -> bool:
        """Handle built-in commands when no external handler is wired."""

        if self.on_cmd:
            return False

        reply = self._send
        if cmd == "/subs":
            for line in self._subscription_lines():
                reply(line)
            return True
        if cmd == "/diag":
            for line in self._ws_diag_lines():
                reply(line)
            return True
        if cmd == "/fresh":
            reply(self._trigger_fresh_reconnect())
            return True
        if cmd == "/logs":
            parts = [p for p in arg.split() if p]
            if not parts or parts[0].lower() == "off":
                self._diag_until_ts = 0.0
                self._trace_until_ts = 0.0
                trace_ctl.disable()
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
            trace_ctl.enable(seconds)
            reply(
                f"🟢 Trace enabled (auto-off by TTL) for {seconds}s."
            )
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
