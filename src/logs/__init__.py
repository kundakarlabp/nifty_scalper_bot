"""Lightweight structured logging helpers used across the bot."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

__all__ = ["StructuredLogger", "structured_log"]


def _normalize(value: Any) -> Any:
    """Best-effort conversion of values to JSON-serialisable primitives."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(v) for v in value]
    for attr in ("__float__", "__int__"):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)()
            except Exception:  # pragma: no cover - extremely defensive
                continue
    return str(value)


_TRACE_GATED_EVENTS = {"market_data_snapshot", "regime_eval", "micro_eval"}


@dataclass
class StructuredLogger:
    """Structured logger that serialises events as JSON strings."""

    name: str = "structured"
    defaults: MutableMapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.name)
        self._defaults: dict[str, Any] = {}
        if self.defaults:
            self._defaults.update({k: _normalize(v) for k, v in self.defaults.items()})

    def bind(self, **fields: Any) -> StructuredLogger:
        """Return a child logger with ``fields`` included on every event."""

        combined = dict(self._defaults)
        combined.update({k: _normalize(v) for k, v in fields.items()})
        return StructuredLogger(name=self.name, defaults=combined)

    def event(self, event: str, /, **fields: Any) -> None:
        """Emit ``event`` with ``fields`` merged with defaults."""

        payload = {"event": event, **self._defaults}
        for key, value in fields.items():
            payload[key] = _normalize(value)
        _ensure_sections(payload)
        try:
            message = json.dumps(payload, sort_keys=True)
        except (TypeError, ValueError):  # pragma: no cover - double safety
            safe_payload = {k: _normalize(v) for k, v in payload.items()}
            message = json.dumps(safe_payload, sort_keys=True)

        trace_active = True
        if event in _TRACE_GATED_EVENTS:
            try:
                hk = import_module("src.diagnostics.healthkit")
            except Exception:  # pragma: no cover - optional dependency
                hk = None
            if hk is not None:
                trace_fn = getattr(hk, "trace_active", None)
                try:
                    trace_active = bool(trace_fn()) if callable(trace_fn) else True
                except Exception:  # pragma: no cover - defensive
                    trace_active = True
        if event in _TRACE_GATED_EVENTS and not trace_active:
            self._logger.debug(message)
        else:
            self._logger.info(message)


structured_log = StructuredLogger()


def _ensure_sections(payload: MutableMapping[str, Any]) -> None:
    """Ensure deploy log payloads expose expected nested keys."""

    def _coerce_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if value is None:
            return {}
        return {"value": _normalize(value)}

    def _apply(name: str, keys: Sequence[str]) -> None:
        section = _coerce_mapping(payload.get(name))
        for key in keys:
            section.setdefault(key, None)
        payload[name] = section

    _apply(
        "data",
        (
            "subscribe",
            "tokens_resolved",
            "tick_full",
            "tick_ltp_only",
        ),
    )
    _apply("signal", ("block_micro", "block_size"))
    _apply("trade", ("ctx",))
