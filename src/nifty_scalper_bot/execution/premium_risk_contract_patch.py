"""Preserve option-premium risk geometry through the live execution boundary.

Elite strategies may publish explicit option-premium invalidation metadata.  The
legacy runner normalisation must not widen those stops or replace their targets
with an ATR from an ambiguous price domain.  This module keeps the existing
StrategyManager output patch and installs small, idempotent runtime hooks after
the application module is fully imported.
"""

from __future__ import annotations

import dataclasses
from contextlib import suppress
from typing import Any, Mapping

_PATCH_APPLIED = False
_ORIGINAL_GENERATE_SIGNAL: Any = None
_RUNNER_PATCH_ATTR = "_premium_geometry_hardening_installed"
_BRACKET_PATCH_ATTR = "_premium_exit_provenance_hardening_installed"
_TICK_SIZE = 0.05


def _positive_float(value: Any) -> float | None:
    with suppress(TypeError, ValueError):
        parsed = float(value)
        if parsed > 0.0:
            return parsed
    return None


def _metadata(signal: Any) -> dict[str, Any]:
    value = getattr(signal, "metadata", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _premium_domain(metadata: Mapping[str, Any]) -> bool:
    domain = str(
        metadata.get("invalidation_level_domain")
        or metadata.get("risk_domain")
        or metadata.get("atr_domain")
        or ""
    ).strip().lower()
    if domain in {"option_premium", "premium", "options"}:
        return True
    return bool(
        metadata.get("computed_from_premium")
        or metadata.get("premium_risk_contract_applied")
        or _positive_float(metadata.get("premium_stop_distance"))
        or _positive_float(metadata.get("premium_stop_pct"))
        or _positive_float(metadata.get("setup_invalidation_premium"))
        or _positive_float(metadata.get("premium_stop_price"))
    )


def _spread_distance(metadata: Mapping[str, Any], entry_price: float) -> float:
    bid = _positive_float(metadata.get("bid") or metadata.get("best_bid"))
    ask = _positive_float(metadata.get("ask") or metadata.get("best_ask"))
    if bid is not None and ask is not None and ask >= bid:
        return ask - bid
    spread_pct = _positive_float(metadata.get("spread_pct"))
    if spread_pct is not None:
        return entry_price * spread_pct / 100.0
    return 0.0


def _side_geometry_valid(side: str, entry: float, stop: float, target: float) -> bool:
    if side == "BUY":
        return 0.0 < stop < entry < target
    return 0.0 < target < entry < stop


def _risk_distance(
    signal: Any,
    *,
    entry_price: float,
    entry_side: str,
    atr: float,
) -> tuple[float, float, str, bool]:
    metadata = _metadata(signal)
    spread = _spread_distance(metadata, entry_price)
    trusted = _premium_domain(metadata)
    stop = _positive_float(getattr(signal, "stop_loss", None))
    explicit_stop = _positive_float(
        metadata.get("setup_invalidation_premium")
        or metadata.get("premium_stop_price")
    )
    explicit_distance = _positive_float(metadata.get("premium_stop_distance"))
    stop_pct = _positive_float(metadata.get("premium_stop_pct"))
    premium_atr = _positive_float(
        metadata.get("premium_atr") or metadata.get("option_atr")
    )
    if premium_atr is None and trusted:
        premium_atr = _positive_float(atr)

    source = "premium_percent_fallback"
    distance: float | None = None
    if explicit_stop is not None and trusted:
        candidate = (
            entry_price - explicit_stop
            if entry_side == "BUY"
            else explicit_stop - entry_price
        )
        if candidate > 0.0:
            distance = candidate
            source = "explicit_premium_stop"
    if distance is None and explicit_distance is not None:
        distance = explicit_distance
        source = "premium_stop_distance"
        trusted = True
    if distance is None and stop_pct is not None:
        normalized_pct = stop_pct / 100.0 if stop_pct > 1.0 else stop_pct
        if 0.0 < normalized_pct < 1.0:
            distance = entry_price * normalized_pct
            source = "premium_stop_pct"
            trusted = True
    if distance is None and stop is not None:
        candidate = entry_price - stop if entry_side == "BUY" else stop - entry_price
        untrusted_cap = max(entry_price * 0.30, spread * 4.0, 1.0)
        if candidate > 0.0 and (trusted or candidate <= untrusted_cap):
            distance = candidate
            source = "existing_premium_geometry"
    if distance is None and premium_atr is not None:
        distance = max(premium_atr * 1.2, entry_price * 0.02, spread * 1.5, 1.0)
        source = "premium_atr"
        trusted = True
    if distance is None:
        distance = max(entry_price * 0.10, spread * 1.5, 1.0)

    max_distance = max(
        entry_price * (0.60 if trusted else 0.30),
        spread * 4.0,
        1.0,
    )
    distance = min(max(distance, _TICK_SIZE), max_distance)
    if entry_side == "BUY":
        distance = min(distance, max(entry_price - _TICK_SIZE, _TICK_SIZE))
    rr = _positive_float(metadata.get("premium_target_rr")) or 2.0
    return distance, rr, source, trusted


def apply_premium_risk_contract(signal: Any, premium: float) -> Any:
    """Fill missing option-premium SL/TP from explicit strategy distance."""
    if signal is None or premium <= 0.0:
        return signal
    action = str(getattr(signal, "action", "") or "").upper()
    symbol = str(getattr(signal, "symbol", "") or "").upper()
    if action not in {"BUY", "SELL"} or not symbol.endswith(("CE", "PE")):
        return signal

    metadata = getattr(signal, "metadata", {})
    if not isinstance(metadata, Mapping):
        return signal
    distance = _positive_float(metadata.get("premium_stop_distance"))
    if distance is None:
        return signal
    domain = str(metadata.get("invalidation_level_domain") or "option_premium").lower()
    if domain != "option_premium":
        return signal

    existing_sl = _positive_float(getattr(signal, "stop_loss", None))
    existing_tp = _positive_float(getattr(signal, "take_profit", None))
    rr = _positive_float(metadata.get("premium_target_rr")) or 2.0

    if action == "BUY":
        stop_loss = existing_sl or max(0.05, premium - distance)
        take_profit = existing_tp or premium + distance * rr
        valid = stop_loss < premium < take_profit
    else:
        stop_loss = existing_sl or premium + distance
        take_profit = existing_tp or max(0.05, premium - distance * rr)
        valid = take_profit < premium < stop_loss
    if not valid:
        return signal

    updated_metadata = dict(metadata)
    updated_metadata.setdefault("premium_risk_contract_applied", True)
    updated_metadata.setdefault("premium_risk_source", "premium_stop_distance")
    updated_metadata.setdefault("premium_risk_reference_price", float(premium))
    return dataclasses.replace(
        signal,
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        metadata=updated_metadata,
    )


def validate_option_premium_geometry(
    self: Any,
    signal: Any,
    entry_price: float,
    entry_side: str,
    atr: float,
) -> Any:
    """Return domain-safe option SL/TP without using underlying-scale distances."""
    del self
    entry = float(entry_price or 0.0)
    side = str(entry_side or "BUY").upper()
    if entry <= 0.0 or side not in {"BUY", "SELL"}:
        return signal

    metadata = _metadata(signal)
    distance, rr, source, trusted = _risk_distance(
        signal,
        entry_price=entry,
        entry_side=side,
        atr=float(atr or 0.0),
    )
    spread = _spread_distance(metadata, entry)
    existing_stop = _positive_float(getattr(signal, "stop_loss", None))
    existing_target = _positive_float(getattr(signal, "take_profit", None))
    explicit_target = _positive_float(
        metadata.get("setup_target_premium")
        or metadata.get("premium_target_price")
    )
    has_rr_contract = _positive_float(metadata.get("premium_target_rr")) is not None

    if side == "BUY":
        stop = entry - distance
        target = entry + distance * rr
    else:
        stop = entry + distance
        target = max(_TICK_SIZE, entry - distance * rr)

    existing_stop_distance = None
    if existing_stop is not None:
        existing_stop_distance = (
            entry - existing_stop if side == "BUY" else existing_stop - entry
        )
    max_stop_distance = max(
        entry * (0.60 if trusted else 0.30), spread * 4.0, 1.0
    )
    stop_was_usable = bool(
        existing_stop_distance is not None
        and 0.0 < existing_stop_distance <= max_stop_distance
    )
    if stop_was_usable and source == "existing_premium_geometry":
        stop = float(existing_stop)
        distance = float(existing_stop_distance)
        if side == "BUY":
            target = entry + distance * rr
        else:
            target = max(_TICK_SIZE, entry - distance * rr)

    max_target_distance = max(entry, distance * max(rr, 3.0), spread * 8.0)
    if explicit_target is not None:
        explicit_valid = (
            entry < explicit_target <= entry + max_target_distance
            if side == "BUY"
            else max(_TICK_SIZE, entry - max_target_distance)
            <= explicit_target
            < entry
        )
        if explicit_valid:
            target = explicit_target
    elif not has_rr_contract and existing_target is not None and stop_was_usable:
        target_distance = (
            existing_target - entry if side == "BUY" else entry - existing_target
        )
        if 0.0 < target_distance <= max_target_distance:
            target = existing_target

    if not _side_geometry_valid(side, entry, stop, target):
        if side == "BUY":
            stop = max(_TICK_SIZE, entry - distance)
            target = entry + distance * rr
        else:
            stop = entry + distance
            target = max(_TICK_SIZE, entry - distance * rr)

    updated = dict(metadata)
    updated.update(
        {
            "premium_risk_distance": float(distance),
            "premium_risk_domain": "option_premium",
            "premium_risk_source": source,
            "premium_target_rr": float(rr),
            "premium_geometry_validated": True,
        }
    )
    if source == "premium_atr":
        updated.setdefault("premium_atr", float(distance) / 1.2)

    return dataclasses.replace(
        signal,
        stop_loss=max(_TICK_SIZE, float(stop)),
        take_profit=max(_TICK_SIZE, float(target)),
        metadata=updated,
    )


def anchor_option_geometry_to_execution(
    self: Any,
    signal: Any,
    *,
    signal_price: float,
    execution_price: float,
    entry_side: str,
    atr: float,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
) -> Any:
    """Translate valid distance geometry to execution price; never widen it."""
    del sl_mult, tp_mult
    execution = float(execution_price or 0.0)
    reference = float(signal_price or 0.0)
    side = str(entry_side or "BUY").upper()
    if execution <= 0.0 or side not in {"BUY", "SELL"}:
        return signal

    metadata = _metadata(signal)
    mode = str(metadata.get("bracket_anchor_mode") or "distance").lower()
    stop = _positive_float(getattr(signal, "stop_loss", None))
    target = _positive_float(getattr(signal, "take_profit", None))

    # Absolute technical invalidations are not moved. If fill drift invalidates
    # them, the downstream order preflight remains the fail-closed authority.
    if mode == "absolute_level":
        return signal

    delta = execution - reference
    if abs(delta) > _TICK_SIZE:
        if stop is not None:
            stop += delta
        if target is not None:
            target += delta

    if stop is not None and target is not None and _side_geometry_valid(
        side, execution, stop, target
    ):
        return dataclasses.replace(
            signal,
            stop_loss=float(stop),
            take_profit=float(target),
        )

    candidate = dataclasses.replace(
        signal,
        stop_loss=float(stop) if stop is not None else None,
        take_profit=float(target) if target is not None else None,
    )
    return validate_option_premium_geometry(
        self,
        candidate,
        entry_price=execution,
        entry_side=side,
        atr=atr,
    )


def _enrich_virtual_bracket_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Restore optional TP1/trailing fields from durable trade provenance."""

    enriched = dict(kwargs)
    provenance = enriched.get("trade_provenance")
    if not isinstance(provenance, Mapping):
        return enriched
    exit_plan = provenance.get("exit_plan")
    source = dict(exit_plan) if isinstance(exit_plan, Mapping) else dict(provenance)

    if _positive_float(enriched.get("tp1_price")) is None:
        tp1_price = _positive_float(source.get("tp1_price"))
        if tp1_price is not None:
            enriched["tp1_price"] = tp1_price
    if _positive_float(enriched.get("tp1_qty")) is None:
        tp1_qty = _positive_float(source.get("tp1_qty"))
        if tp1_qty is not None:
            enriched["tp1_qty"] = int(tp1_qty)
    if _positive_float(enriched.get("trailing_atr_mult")) is None:
        trailing = _positive_float(source.get("trailing_atr_mult"))
        if trailing is not None:
            enriched["trailing_atr_mult"] = trailing
    if _positive_float(enriched.get("resolved_lot_size")) is None:
        lot_size = _positive_float(source.get("resolved_lot_size"))
        if lot_size is not None:
            enriched["resolved_lot_size"] = int(lot_size)
    return enriched


def install_bracket_exit_provenance_hardening(bracket_cls: type[Any]) -> None:
    """Wrap the fully composed bracket class without bypassing ownership guards."""
    if bool(getattr(bracket_cls, _BRACKET_PATCH_ATTR, False)):
        return
    original = bracket_cls.register_virtual_bracket

    def register_virtual_bracket(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original(self, *args, **_enrich_virtual_bracket_kwargs(kwargs))

    bracket_cls._premium_exit_provenance_original_register = original
    bracket_cls.register_virtual_bracket = register_virtual_bracket
    setattr(bracket_cls, _BRACKET_PATCH_ATTR, True)


def install_runner_geometry_hardening(runner_cls: type[Any]) -> None:
    """Install the domain-safe geometry methods on StrategyRunner once."""
    if bool(getattr(runner_cls, _RUNNER_PATCH_ATTR, False)):
        return
    runner_cls._premium_geometry_original_validate = getattr(
        runner_cls, "_validate_long_option_geometry", None
    )
    runner_cls._premium_geometry_original_anchor = getattr(
        runner_cls, "_anchor_sl_tp_to_execution", None
    )
    runner_cls._validate_long_option_geometry = validate_option_premium_geometry
    runner_cls._anchor_sl_tp_to_execution = anchor_option_geometry_to_execution
    setattr(runner_cls, _RUNNER_PATCH_ATTR, True)


def _patched_generate_signal(
    self: Any, symbol: str, current_price: float, *args: Any, **kwargs: Any
) -> Any:
    signal = _ORIGINAL_GENERATE_SIGNAL(self, symbol, current_price, *args, **kwargs)
    return apply_premium_risk_contract(signal, float(current_price or 0.0))


def apply_patches() -> None:
    global _PATCH_APPLIED, _ORIGINAL_GENERATE_SIGNAL
    if _PATCH_APPLIED:
        return
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    if getattr(StrategyManager, "_premium_risk_contract_patch", False):
        _PATCH_APPLIED = True
        return
    _ORIGINAL_GENERATE_SIGNAL = StrategyManager.generate_signal
    StrategyManager.generate_signal = _patched_generate_signal
    StrategyManager._premium_risk_contract_patch = True
    _PATCH_APPLIED = True


__all__ = [
    "_enrich_virtual_bracket_kwargs",
    "anchor_option_geometry_to_execution",
    "apply_patches",
    "apply_premium_risk_contract",
    "install_bracket_exit_provenance_hardening",
    "install_runner_geometry_hardening",
    "validate_option_premium_geometry",
]
