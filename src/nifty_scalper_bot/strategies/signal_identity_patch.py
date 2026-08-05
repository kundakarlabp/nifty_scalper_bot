"""Make live signal identity stable and observable across tick retries."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Mapping

from nifty_scalper_bot.strategies.quote_update_identity import (
    build_evaluation_snapshot_id,
    resolve_quote_update_identity,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

_PATCHED = False
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


def _option_thesis(symbol: object, metadata: Mapping[str, Any]) -> tuple[str, str]:
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


def _anchor_value(metadata: Mapping[str, Any]) -> str | None:
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


def _anchor(metadata: Mapping[str, Any]) -> str:
    value = _anchor_value(metadata)
    if value is not None:
        return value
    # Never mint a new identity from wall-clock time. A stable sentinel keeps
    # malformed retries idempotent; the real-live preparation gate rejects
    # anchorless strategy entries before they can reach the broker.
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


def _deterministic_id(signal: Any) -> str:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    strategy = str(
        metadata.get("strategy_name") or metadata.get("strategy") or "manual"
    )
    underlying, option_side = _option_thesis(getattr(signal, "symbol", ""), metadata)
    setup_anchor = _anchor(metadata)
    action = str(getattr(signal, "action", ""))
    raw = f"{strategy}:{underlying}:{option_side}:{action}:{setup_anchor}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _stamp_evaluation_identity(signal: Any, indicators: Mapping[str, Any]) -> Any:
    """Attach exact quote-snapshot identity while preserving setup identity."""
    metadata = dict(getattr(signal, "metadata", {}) or {})
    version, resolved_source = resolve_quote_update_identity(
        ("indicator_context", indicators),
        ("signal_metadata", metadata),
    )
    if version is None:
        return signal
    setup_signal_id = _deterministic_id(signal)
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


def _install_elite_signal_observability() -> None:
    from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy

    current = EliteStrategy.generate_signal
    if getattr(current, "_elite_signal_observability_patch", False):
        return

    def generate_signal(
        self: Any,
        symbol: str,
        indicators: Mapping[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> Any:
        signal = current(self, symbol, indicators, current_price, position)
        if signal is None:
            return None
        signal = _stamp_evaluation_identity(signal, indicators)
        metadata = dict(getattr(signal, "metadata", {}) or {})
        strategy = str(
            metadata.get("strategy_name")
            or metadata.get("strategy")
            or getattr(self, "name", "unknown")
        )
        _, side = _option_thesis(getattr(signal, "symbol", symbol), metadata)
        setup_anchor = _anchor_value(metadata)
        setup_id = metadata.get("setup_id") or metadata.get("setup_structure_id")
        raw_score = metadata.get("raw_setup_score")
        if raw_score is None:
            raw_score = metadata.get("strategy_score") or metadata.get("context_score")
        role = str(metadata.get("role") or "trigger").lower()
        LOGGER.log(
            logging.DEBUG if role == "context" else logging.INFO,
            (
                "ELITE_SIGNAL_GENERATED strategy=%s symbol=%s side=%s "
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
                "event": "ELITE_SIGNAL_GENERATED",
                "strategy": strategy,
                "symbol": getattr(signal, "symbol", symbol),
                "side": side or None,
                "raw_setup_score": raw_score,
                "confidence": getattr(signal, "confidence", None),
                "setup_id": setup_id,
                "setup_anchor": setup_anchor,
                "quote_update_version": metadata.get("quote_update_version"),
                "quote_update_version_source": metadata.get(
                    "quote_update_version_source"
                ),
                "setup_signal_id": metadata.get("setup_signal_id"),
                "evaluation_snapshot_id": metadata.get("evaluation_snapshot_id"),
                "role": role,
            },
        )
        return signal

    generate_signal.__name__ = getattr(current, "__name__", "generate_signal")
    generate_signal.__doc__ = getattr(current, "__doc__", None)
    setattr(generate_signal, "_elite_signal_observability_patch", True)
    setattr(generate_signal, "_original", current)
    EliteStrategy.generate_signal = generate_signal  # type: ignore[assignment]


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from nifty_scalper_bot.strategies.signal_generator import Signal

    if not getattr(Signal, "_stable_setup_identity_patch", False):
        Signal.deterministic_id = property(_deterministic_id)
        Signal._stable_setup_identity_patch = True
    _install_elite_signal_observability()
    _PATCHED = True


__all__ = [
    "apply_patches",
    "_deterministic_id",
    "_option_thesis",
    "_stamp_evaluation_identity",
    "has_setup_anchor",
]
