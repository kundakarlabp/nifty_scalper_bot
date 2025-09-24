from __future__ import annotations

"""Utilities for recording lightweight trade diagnostics."""

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

from src.logs import structured_log


def _sanitize(value: Any, *, depth: int = 0, max_depth: int = 4) -> Any:
    """Best-effort conversion of values to JSON-friendly primitives."""

    if depth >= max_depth:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        out: MutableMapping[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 20:
                break
            out[str(key)] = _sanitize(item, depth=depth + 1, max_depth=max_depth)
        return dict(out)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = []
        for idx, item in enumerate(value):
            if idx >= 20:
                break
            items.append(_sanitize(item, depth=depth + 1, max_depth=max_depth))
        return items
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - defensive fallback
        return str(value)


def log_trade_context(logger: Any, context: Mapping[str, Any]) -> None:
    """Emit the trade sizing context to structured logs and the main logger."""

    safe_ctx = _sanitize(dict(context))
    try:
        structured_log.event("trade_context", **safe_ctx)
    except Exception:  # pragma: no cover - logging should not fail the runner
        if logger:
            logger.debug("log_trade_context structured logging failed", exc_info=True)
    try:
        if logger:
            logger.debug("trade.context", extra={"trade_ctx": safe_ctx})
    except Exception:  # pragma: no cover - logging should not fail the runner
        if logger:
            logger.debug("log_trade_context logger emit failed", exc_info=True)
