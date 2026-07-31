"""Make live signal identity stable across option strike rotation and tick retries."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import re
from typing import Any, Mapping

_PATCHED = False
_OPTION_SUFFIX = re.compile(r"(CE|PE)$")
_FIRST_DIGIT = re.compile(r"\d")
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
    side = str(metadata.get("option_side") or metadata.get("contract_side") or "").upper()
    text = str(symbol or "").strip().upper().split(":")[-1]
    suffix = _OPTION_SUFFIX.search(text)
    if side not in {"CE", "PE"} and suffix is not None:
        side = suffix.group(1)
    underlying = str(metadata.get("underlying") or metadata.get("base_symbol") or "").upper()
    if not underlying:
        body = _OPTION_SUFFIX.sub("", text)
        digit = _FIRST_DIGIT.search(body)
        underlying = body[: digit.start()] if digit is not None else body
    return underlying or text, side


def _anchor(metadata: Mapping[str, Any]) -> str:
    for key in _ANCHOR_KEYS:
        value = metadata.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        if isinstance(value, (int, float)):
            return str(int(float(value)))
        return str(value).strip()
    # Legacy fallback retained only for signals that provide no setup/bar identity.
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M")


def _deterministic_id(signal: Any) -> str:
    metadata = dict(getattr(signal, "metadata", {}) or {})
    strategy = str(metadata.get("strategy_name") or metadata.get("strategy") or "manual")
    underlying, option_side = _option_thesis(getattr(signal, "symbol", ""), metadata)
    setup_anchor = _anchor(metadata)
    action = str(getattr(signal, "action", ""))
    raw = f"{strategy}:{underlying}:{option_side}:{action}:{setup_anchor}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from nifty_scalper_bot.strategies.signal_generator import Signal

    if getattr(Signal, "_stable_setup_identity_patch", False):
        _PATCHED = True
        return
    Signal.deterministic_id = property(_deterministic_id)
    Signal._stable_setup_identity_patch = True
    _PATCHED = True


__all__ = ["apply_patches", "_deterministic_id", "_option_thesis"]
