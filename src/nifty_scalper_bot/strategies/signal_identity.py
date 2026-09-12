"""Stable signal identity and setup-context utilities.

This module owns pure identity/observability helpers only. It does not mutate
strategy classes at import time; native owners call these helpers explicitly.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from nifty_scalper_bot.strategies.quote_update_identity import (
    build_evaluation_snapshot_id,
    resolve_quote_update_identity,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

_OPTION_SUFFIX = re.compile(r"(CE|PE)$")
_FIRST_DIGIT = re.compile(r"\d")
_MISSING_ANCHOR = "MISSING_SETUP_ANCHOR"
_ANCHOR_KEYS = (
    "setup_id",
    "setup_structure_id",
    "structure_id",
    "setup_candle_timestamp",
    "bar_timestamp",
    "latest_bar_ts",
    "signal_timestamp",
    "timestamp",
)
_SETUP_TIMESTAMP_KEYS = (
    "setup_candle_timestamp",
    "bar_timestamp",
    "latest_bar_ts",
    "signal_timestamp",
    "timestamp",
)
_SETUP_METADATA_LIMIT = 2048
_SETUP_METADATA_LOCK = threading.Lock()
_SETUP_METADATA_BY_SIGNAL_ID: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CURRENT_ORDER_SETUP_METADATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_order_setup_metadata",
    default=None,
)


def option_thesis(symbol: object, metadata: Mapping[str, Any]) -> tuple[str, str]:
    """Return normalized underlying and option side for signal identity."""
    side = str(
        metadata.get("option_side") or metadata.get("contract_side") or ""
    ).upper()
    text = str(symbol or "").strip().upper().split(":")[-1]
    suffix = _OPTION_SUFFIX.search(text)
    if side not in {"CE", "PE"} and suffix is not None:
        side = suffix.group(1)
    underlying = str(
        metadata.get("underlying") or metadata.get("base_symbol") or ""
    ).upper()
    if not underlying:
        body = _OPTION_SUFFIX.sub("", text)
        digit = _FIRST_DIGIT.search(body)
        underlying = body[: digit.start()] if digit is not None else body
    return underlying or text, side


def has_setup_anchor(metadata: Mapping[str, Any] | None) -> bool:
    """Return whether metadata carries an explicit setup/bar identity."""
    payload = metadata or {}
    return any(payload.get(key) not in (None, "") for key in _ANCHOR_KEYS)


def anchor_value(metadata: Mapping[str, Any]) -> str | None:
    """Return the first stable setup anchor encoded in metadata."""
    for key in _ANCHOR_KEYS:
        value = metadata.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, datetime):
            dt = (
                value
                if value.tzinfo is not None
                else value.replace(tzinfo=timezone.utc)
            )
            return dt.isoformat()
        if isinstance(value, (int, float)):
            return str(int(float(value)))
        return str(value).strip()
    return None


def _remember_setup_metadata(signal_id: str, metadata: Mapping[str, Any]) -> None:
    if str(metadata.get("role") or "trigger").strip().lower() == "context":
        return
    timestamp = next(
        (
            metadata.get(key)
            for key in _SETUP_TIMESTAMP_KEYS
            if metadata.get(key) not in (None, "")
        ),
        None,
    )
    if timestamp is None:
        return
    payload: dict[str, Any] = {"setup_candle_timestamp": timestamp}
    setup_id = metadata.get("setup_id")
    if setup_id not in (None, ""):
        payload["setup_id"] = setup_id
    with _SETUP_METADATA_LOCK:
        _SETUP_METADATA_BY_SIGNAL_ID.pop(signal_id, None)
        _SETUP_METADATA_BY_SIGNAL_ID[signal_id] = payload
        while len(_SETUP_METADATA_BY_SIGNAL_ID) > _SETUP_METADATA_LIMIT:
            _SETUP_METADATA_BY_SIGNAL_ID.popitem(last=False)


def setup_metadata_for_signal_id(signal_id: object) -> dict[str, Any]:
    """Return setup metadata previously bound to this deterministic signal id."""
    key = str(signal_id or "").strip()
    if not key:
        return {}
    with _SETUP_METADATA_LOCK:
        payload = _SETUP_METADATA_BY_SIGNAL_ID.get(key)
        if payload is None:
            return {}
        _SETUP_METADATA_BY_SIGNAL_ID.move_to_end(key)
        return dict(payload)


def current_order_setup_metadata() -> dict[str, Any]:
    """Return setup metadata scoped to the currently evaluated order call."""
    return dict(_CURRENT_ORDER_SETUP_METADATA.get() or {})


@contextmanager
def order_setup_context(signal_id: object) -> Iterator[dict[str, Any]]:
    """Scope exact setup metadata to one synchronous order/risk evaluation."""
    payload = setup_metadata_for_signal_id(signal_id)
    token = _CURRENT_ORDER_SETUP_METADATA.set(payload or None)
    try:
        yield payload
    finally:
        _CURRENT_ORDER_SETUP_METADATA.reset(token)


def _anchor(metadata: Mapping[str, Any]) -> str:
    value = anchor_value(metadata)
    if value is not None:
        return value
    LOGGER.error(
        "SIGNAL_IDENTITY_ANCHOR_MISSING strategy=%s",
        metadata.get("strategy_name") or metadata.get("strategy") or "unknown",
        extra={
            "event": "SIGNAL_IDENTITY_ANCHOR_MISSING",
            "strategy": str(
                metadata.get("strategy_name") or metadata.get("strategy") or "unknown"
            ),
        },
    )
    return _MISSING_ANCHOR


def deterministic_signal_id(signal: Any) -> str:
    """Return a stable setup identity without using wall-clock fallback time."""
    metadata = dict(getattr(signal, "metadata", {}) or {})
    strategy = str(
        metadata.get("strategy_name") or metadata.get("strategy") or "manual"
    )
    underlying, option_side = option_thesis(getattr(signal, "symbol", ""), metadata)
    setup_anchor = _anchor(metadata)
    action = str(getattr(signal, "action", ""))
    raw = f"{strategy}:{underlying}:{option_side}:{action}:{setup_anchor}"
    signal_id = hashlib.md5(raw.encode()).hexdigest()[:16]
    _remember_setup_metadata(signal_id, metadata)
    return signal_id


def stamp_evaluation_identity(signal: Any, indicators: Mapping[str, Any]) -> Any:
    """Attach exact quote-snapshot identity while preserving setup identity."""
    metadata = dict(getattr(signal, "metadata", {}) or {})
    version, resolved_source = resolve_quote_update_identity(
        ("indicator_context", indicators),
        ("signal_metadata", metadata),
    )
    if version is None:
        return signal
    setup_signal_id = deterministic_signal_id(signal)
    evaluation_snapshot_id = build_evaluation_snapshot_id(setup_signal_id, version)
    if str(resolved_source or "").startswith("indicator_context:"):
        source = str(
            indicators.get("quote_update_version_source") or resolved_source or ""
        )
    else:
        source = str(
            metadata.get("quote_update_version_source") or resolved_source or ""
        )
    updates = {
        "quote_update_version": version,
        "quote_update_version_source": source or None,
        "setup_signal_id": setup_signal_id,
        "evaluation_snapshot_id": evaluation_snapshot_id,
    }
    with_metadata = getattr(signal, "with_metadata", None)
    if callable(with_metadata):
        return with_metadata(**updates)
    mutable = getattr(signal, "metadata", None)
    if isinstance(mutable, dict):
        mutable.update(updates)
    return signal


def finalize_signal_observability(
    signal: Any,
    indicators: Mapping[str, Any],
    *,
    strategy_name: str,
    symbol: str,
) -> Any:
    """Stamp evaluation identity and emit the canonical strategy vote event."""
    if signal is None:
        return None
    signal = stamp_evaluation_identity(signal, indicators)
    metadata = dict(getattr(signal, "metadata", {}) or {})
    strategy = str(
        metadata.get("strategy_name")
        or metadata.get("strategy")
        or strategy_name
        or "unknown"
    )
    _, side = option_thesis(getattr(signal, "symbol", symbol), metadata)
    setup_anchor = anchor_value(metadata)
    setup_id = metadata.get("setup_id") or metadata.get("setup_structure_id")
    raw_score = metadata.get("raw_setup_score")
    if raw_score is None:
        raw_score = metadata.get("strategy_score") or metadata.get("context_score")
    role = str(metadata.get("role") or "trigger").lower()
    vote_event = (
        "STRATEGY_CONTEXT_VOTE" if role == "context" else "STRATEGY_TRIGGER_VOTE"
    )
    LOGGER.log(
        logging.DEBUG if role == "context" else logging.INFO,
        (
            f"{vote_event} strategy=%s symbol=%s side=%s "
            "raw_setup_score=%s confidence=%s setup_id=%s setup_anchor=%s "
            "quote_update_version=%s evaluation_snapshot_id=%s"
        ),
        strategy,
        getattr(signal, "symbol", symbol),
        side or None,
        raw_score,
        getattr(signal, "confidence", None),
        setup_id,
        setup_anchor,
        metadata.get("quote_update_version"),
        metadata.get("evaluation_snapshot_id"),
        extra={
            "event": vote_event,
            "strategy": strategy,
            "symbol": getattr(signal, "symbol", symbol),
            "side": side or None,
            "raw_setup_score": raw_score,
            "confidence": getattr(signal, "confidence", None),
            "setup_id": setup_id,
            "setup_anchor": setup_anchor,
            "quote_update_version": metadata.get("quote_update_version"),
            "quote_update_version_source": metadata.get("quote_update_version_source"),
            "setup_signal_id": metadata.get("setup_signal_id"),
            "evaluation_snapshot_id": metadata.get("evaluation_snapshot_id"),
            "role": role,
        },
    )
    return signal


__all__ = [
    "anchor_value",
    "current_order_setup_metadata",
    "deterministic_signal_id",
    "finalize_signal_observability",
    "has_setup_anchor",
    "option_thesis",
    "order_setup_context",
    "setup_metadata_for_signal_id",
    "stamp_evaluation_identity",
]
