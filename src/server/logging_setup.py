from __future__ import annotations

"""Utilities for structured logging across services."""

import json
import logging
from datetime import datetime
from typing import Any
from collections.abc import Mapping, Sequence

_EVENT_LOGGER = logging.getLogger("events")


def _sanitize(value: Any, *, depth: int = 0) -> Any:
    """Return a JSON-serialisable representation of ``value``."""

    if depth >= 4:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= 20:
                break
            out[str(key)] = _sanitize(val, depth=depth + 1)
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items: list[Any] = []
        for idx, item in enumerate(value):
            if idx >= 20:
                break
            items.append(_sanitize(item, depth=depth + 1))
        return items
    return str(value)


def log_event(event: str, level: str = "info", /, **fields: Any) -> None:
    """Emit a structured log entry for ``event``.

    Parameters
    ----------
    event:
        Short event name, e.g. ``"order.fill"``.
    level:
        Logging level name. Defaults to ``"info"``.
    **fields:
        Additional JSON-serialisable context.
    """

    level_name = str(level).upper()
    level_value = getattr(logging, level_name, logging.INFO)
    logger = _EVENT_LOGGER
    if not logger.isEnabledFor(level_value):
        return

    payload: dict[str, Any] = {"event": event, "ts": datetime.utcnow().isoformat()}
    for key, value in fields.items():
        payload[str(key)] = _sanitize(value)

    try:
        message = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        message = f"{event} {payload}"
    logger.log(level_value, message)
