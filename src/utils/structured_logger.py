"""Structured logging helpers with trace-aware context binding."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, MutableMapping

try:  # Python 3.11+ includes zoneinfo in the stdlib
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:  # pragma: no cover - fallback for narrow envs
    from backports.zoneinfo import ZoneInfo  # type: ignore

from src.config import settings
from src.utils import ringlog
from src.utils.trace import Trace

_STATE_ATTR = "_structured_logger_state"


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def _get_setting(name: str, default: Any = None) -> Any:
    upper = getattr(settings, name, None)
    if upper is not None:
        return upper
    lower = getattr(settings, name.lower(), None)
    if lower is not None:
        return lower
    return default


def _resolve_level(level: Any) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), None)
        if isinstance(lvl, int):
            return lvl
    return logging.INFO


def _resolve_tz() -> ZoneInfo:
    tz_name = _get_setting("LOG_TZ", None) or _get_setting("TZ", "UTC")
    try:
        return ZoneInfo(str(tz_name))
    except Exception:  # pragma: no cover - fallback when tz database missing
        return ZoneInfo("UTC")


@dataclass(slots=True)
class _LoggerState:
    level: int
    tz: ZoneInfo
    json_enabled: bool
    log_path: str | None

    def formatter(self) -> "_StructuredFormatter":
        return _StructuredFormatter(self)


class _StructuredFormatter(logging.Formatter):
    def __init__(self, state: _LoggerState) -> None:
        super().__init__()
        self._state = state

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "_structured_payload", None)
        if payload is None:
            ts = datetime.now(self._state.tz).isoformat()
            return f"{ts} {record.levelname} {record.getMessage()}"

        if self._state.json_enabled:
            text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            if ringlog.enabled():
                ringlog.append(payload)
            return text

        msg = payload.get("msg") or payload.get("event")
        extras = {k: v for k, v in payload.items() if k not in {"ts", "level", "comp", "event", "trace_id", "symbol", "side", "msg"}}
        extra_str = f" {extras}" if extras else ""
        return (
            f"{payload['ts']} {payload['level'].upper()} {payload['comp']} "
            f"- {payload['event']} | {msg}{extra_str}"
        )


class _BoundLogger:
    def __init__(
        self,
        logger: logging.Logger,
        state: _LoggerState,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self._logger = logger
        self._state = state
        self._context: Dict[str, Any] = dict(context or {})

    def __call__(self, level: int, event: str, **fields: Any) -> Dict[str, Any]:
        payload = self._payload(level, event, fields)
        self._emit(level, payload)
        return payload

    def log(self, level: int, event: str, **fields: Any) -> Dict[str, Any]:
        return self(level, event, **fields)

    def bind(self, **ctx: Any) -> "_BoundLogger":
        merged = dict(self._context)
        merged.update(_normalize_context(ctx, self._state))
        return _BoundLogger(self._logger, self._state, merged)

    def with_trace(self, trace: Trace) -> "_BoundLogger":
        return self.bind(trace=trace)

    def _payload(
        self, level: int, event: str, fields: Mapping[str, Any]
    ) -> Dict[str, Any]:
        now = datetime.now(self._state.tz).isoformat()
        level_name = logging.getLevelName(level)
        context = dict(self._context)
        context.update(_normalize_context(fields, self._state))
        trace = context.pop("trace", None)
        if isinstance(trace, Trace):
            context.setdefault("trace_id", trace.trace_id)
            if trace.signal_id is not None:
                context.setdefault("signal_id", trace.signal_id)
            if trace.order_client_id is not None:
                context.setdefault("order_client_id", trace.order_client_id)

        trace_id = context.pop("trace_id", None)
        payload: Dict[str, Any] = {
            "ts": now,
            "level": str(level_name).lower(),
            "comp": context.pop("comp", self._logger.name),
            "event": event,
            "trace_id": trace_id,
            "symbol": context.pop("symbol", None),
            "side": context.pop("side", None),
            "msg": context.pop("msg", context.pop("message", None)),
        }
        for key in ("ts", "level", "comp", "event", "trace_id", "symbol", "side", "msg"):
            context.pop(key, None)
        payload.update(context)
        return payload

    def _emit(self, level: int, payload: Mapping[str, Any]) -> None:
        extra = {"_structured_payload": dict(payload)}
        message = payload.get("msg") or payload.get("event") or ""
        self._logger.log(level, message, extra=extra)


def _normalize_context(
    ctx: Mapping[str, Any] | MutableMapping[str, Any] | None,
    state: _LoggerState,
) -> Dict[str, Any]:
    if ctx is None:
        return {}
    data = dict(ctx)
    level = data.get("level")
    if level is not None:
        data["level"] = logging.getLevelName(_resolve_level(level)).lower()
    if "tz" in data and not isinstance(data["tz"], ZoneInfo):
        try:
            data["tz"] = ZoneInfo(str(data["tz"]))
        except Exception:
            data["tz"] = state.tz
    return data


def _configure_handlers(logger: logging.Logger, state: _LoggerState) -> None:
    if getattr(logger, "_structured_configured", False):
        return

    logger.setLevel(logging.NOTSET)
    logger.propagate = False
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(state.level)
    stream_handler.setFormatter(state.formatter())
    logger.addHandler(stream_handler)

    if state.log_path:
        directory = os.path.dirname(state.log_path) or "."
        os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(state.log_path)
        file_handler.setLevel(state.level)
        file_handler.setFormatter(state.formatter())
        logger.addHandler(file_handler)

    logger._structured_configured = True  # type: ignore[attr-defined]


def _logger_state() -> _LoggerState:
    level = _resolve_level(_get_setting("LOG_LEVEL", "INFO"))
    tz = _resolve_tz()
    json_enabled = _as_bool(_get_setting("LOG_JSON", False))
    log_path = _get_setting("LOG_PATH", None)
    return _LoggerState(level=level, tz=tz, json_enabled=json_enabled, log_path=log_path)


def get_logger(name: str) -> logging.Logger:
    """Return a structured logger configured from application settings."""

    logger = logging.getLogger(name)
    state: _LoggerState = getattr(logger, _STATE_ATTR, _logger_state())
    setattr(logger, _STATE_ATTR, state)
    _configure_handlers(logger, state)

    def bind(**ctx: Any) -> _BoundLogger:
        base = _normalize_context(ctx, state)
        return _BoundLogger(logger, state, base)

    setattr(logger, "bind", bind)
    return logger


__all__ = ["get_logger"]

