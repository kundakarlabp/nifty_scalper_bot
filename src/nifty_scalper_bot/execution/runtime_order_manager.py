"""File purpose:
    Implement the single production order manager used by the trading runtime.

Key responsibilities:
    - Apply unresolved-exit entry blocking before broker submission.
    - Run bounded entry recovery and finalize partial-entry reconciliation.
    - Delegate unchanged order operations to ``order_manager_core``.

Operational constraints:
    - Protective exits must bypass entry blocking.
    - Recovery must remain bounded and must not create duplicate broker orders.
"""

from __future__ import annotations

import os
from contextlib import suppress
from typing import Any, Callable, Mapping

from nifty_scalper_bot.execution import order_manager_core as _core
from nifty_scalper_bot.execution.entry_recovery import (
    _finalize_partial_entry,
    _recover_submit,
)
from nifty_scalper_bot.execution.entry_recovery import (
    current_entry_blocker as _current_entry_blocker,
)
from nifty_scalper_bot.execution.native_entry_gate import (
    NO_BLOCK,
    block_result,
    configure_provider,
)

_EXIT_IDENTITY_KWARGS = {"linked_entry_order_id", "trade_lifecycle_id", "bracket_id"}


def _strip_exit_identity_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove exit identity metadata unsupported by core.place_order.

    The live bracket safety layer attaches immutable exit metadata to protective
    order requests. The native entry gate needs to see the original kwargs to
    classify the request as protective, but the base OrderManager.place_order
    currently does not accept these metadata-only fields. Strip them only at the
    delegation boundary so exits cannot fail with a TypeError before reaching the
    broker.
    """

    cleaned = dict(kwargs)
    identity = {
        key: cleaned.pop(key) for key in list(_EXIT_IDENTITY_KWARGS) if key in cleaned
    }
    return cleaned, identity


def _positive_float(value: Any) -> float | None:
    with suppress(TypeError, ValueError):
        parsed = float(value)
        if parsed > 0.0:
            return parsed
    return None


def _positive_int(value: Any) -> int:
    with suppress(TypeError, ValueError):
        parsed = int(float(value))
        if parsed > 0:
            return parsed
    return 0


def _enrich_trade_plan_exit_provenance(plan: Any) -> Any:
    """Persist TP1/trailing inputs inside the existing TradePlan provenance.

    The fill path already carries ``trade_provenance`` into ``OrderDetails`` and
    then into virtual-bracket registration. Keeping the optional exit plan in
    that existing durable contract avoids adding a second order model or side
    registry. TP1 is only created when at least two complete lots are present.
    """

    if str(getattr(plan, "intent", "ENTRY") or "ENTRY").upper() not in {
        "ENTRY",
        "SCALE_IN",
        "REVERSAL",
    }:
        return plan

    provenance = getattr(plan, "trade_provenance", {})
    enriched = dict(provenance) if isinstance(provenance, Mapping) else {}
    quantity = _positive_int(getattr(plan, "quantity", 0))
    lot_size = _positive_int(getattr(plan, "resolved_lot_size", 0))
    entry = _positive_float(getattr(plan, "entry_price", None))
    stop = _positive_float(getattr(plan, "stop_loss", None))
    target = _positive_float(getattr(plan, "take_profit", None))
    side = str(getattr(plan, "side", "BUY") or "BUY").upper()

    if lot_size > 0:
        enriched.setdefault("resolved_lot_size", lot_size)
    if entry is not None and stop is not None and target is not None:
        risk = entry - stop if side == "BUY" else stop - entry
        reward = target - entry if side == "BUY" else entry - target
        if risk > 0.0 and reward > 0.0:
            enriched.setdefault("initial_risk_points", float(risk))
            enriched.setdefault("initial_reward_points", float(reward))
            enriched.setdefault("initial_reward_risk", float(reward / risk))

            tp1_enabled = str(
                os.getenv("ENABLE_TP1_SCALE_OUT", "true") or "true"
            ).strip().lower() in {"1", "true", "yes", "on"}
            total_lots = quantity // lot_size if lot_size > 0 else 0
            if tp1_enabled and total_lots >= 2:
                tp1_r = _positive_float(os.getenv("TP1_R_MULT", "1.0")) or 1.0
                tp1_lots = max(1, total_lots // 2)
                if tp1_lots < total_lots:
                    tp1_qty = tp1_lots * lot_size
                    tp1_price = (
                        entry + risk * tp1_r
                        if side == "BUY"
                        else entry - risk * tp1_r
                    )
                    # TP1 must be strictly before the final target. Otherwise a
                    # single tick could trigger both exits and provide no scale-out.
                    strictly_before_final = (
                        entry < tp1_price < target
                        if side == "BUY"
                        else target < tp1_price < entry
                    )
                    if strictly_before_final:
                        enriched.setdefault("tp1_price", float(tp1_price))
                        enriched.setdefault("tp1_qty", int(tp1_qty))

    trailing_mult = _positive_float(enriched.get("trailing_atr_mult"))
    if trailing_mult is None:
        trailing_mult = _positive_float(os.getenv("BRACKET_TRAILING_ATR_MULT", "0"))
    if trailing_mult is not None:
        enriched["trailing_atr_mult"] = float(trailing_mult)

    setattr(plan, "trade_provenance", enriched)
    return plan


def _submit_core_with_exit_provenance(manager: Any, plan: Any) -> Any:
    """Enrich every initial or rebuilt recovery plan before core submission."""
    _enrich_trade_plan_exit_provenance(plan)
    return _core.OrderManager.submit_trade_plan_result(manager, plan)


class RuntimeOrderManager(_core.OrderManager):
    """Production order manager with native recovery and entry gating."""

    def set_trade_plan_rebuilder(
        self,
        callback: Callable[..., Any] | None,
    ) -> None:
        self._trade_plan_rebuilder = callback

    def set_unresolved_exit_provider(self, provider: Any | None) -> None:
        configure_provider(self, provider)
        # The provider is the canonical runtime bracket owner. Reconciliation
        # already reads ``order_manager._bracket_manager``; keep both references
        # aligned so a broker-flat snapshot can clear a completed exit lifecycle.
        self._bracket_manager = provider

    def current_entry_blocker(self) -> Mapping[str, Any] | None:
        return _current_entry_blocker(self)

    def _blocked(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return block_result(
            self,
            _core,
            _core.OrderManager.place_order,
            method_name,
            args,
            kwargs,
        )

    def submit_trade_plan_result(self, plan: Any) -> Any:
        _enrich_trade_plan_exit_provenance(plan)
        blocked = self._blocked("submit_trade_plan_result", (plan,), {})
        if blocked is not NO_BLOCK:
            return blocked
        return _recover_submit(
            _submit_core_with_exit_provenance,
            self,
            plan,
        )

    def submit_trade_plan(self, *args: Any, **kwargs: Any) -> Any:
        blocked = self._blocked("submit_trade_plan", args, kwargs)
        if blocked is not NO_BLOCK:
            return blocked
        return super().submit_trade_plan(*args, **kwargs)

    def place_managed_order_result(self, *args: Any, **kwargs: Any) -> Any:
        blocked = self._blocked("place_managed_order_result", args, kwargs)
        if blocked is not NO_BLOCK:
            return blocked
        previous = getattr(self, "_managed_strategy_name", None)
        self._managed_strategy_name = str(kwargs.get("strategy_name") or "runner")
        try:
            return super().place_managed_order_result(*args, **kwargs)
        finally:
            if previous is None:
                self.__dict__.pop("_managed_strategy_name", None)
            else:
                self._managed_strategy_name = previous

    def place_managed_order(self, *args: Any, **kwargs: Any) -> Any:
        blocked = self._blocked("place_managed_order", args, kwargs)
        if blocked is not NO_BLOCK:
            return blocked
        return super().place_managed_order(*args, **kwargs)

    def place_order(self, *args: Any, **kwargs: Any) -> Any:
        effective_kwargs = dict(kwargs)
        managed_strategy = getattr(self, "_managed_strategy_name", None)
        current_strategy = str(effective_kwargs.get("strategy_name") or "").strip().lower()
        if managed_strategy and current_strategy in {"", "manual"}:
            effective_kwargs["strategy_name"] = managed_strategy
        blocked = self._blocked("place_order", args, effective_kwargs)
        if blocked is not NO_BLOCK:
            return blocked
        cleaned_kwargs, identity = _strip_exit_identity_kwargs(effective_kwargs)
        if identity:
            self._last_exit_identity_kwargs = dict(identity)
        return super().place_order(*args, **cleaned_kwargs)

    def _update_from_response(
        self,
        order: Any,
        payload: dict[str, Any],
    ) -> Any:
        updated = super()._update_from_response(order, payload)
        try:
            _finalize_partial_entry(self, updated, payload)
        except Exception as exc:
            logger = getattr(self, "_logger", None)
            log = getattr(logger, "error", None)
            if callable(log):
                log(
                    "ENTRY_PARTIAL_FILL_RECONCILE_FAILED order_id=%s error=%s",
                    getattr(order, "order_id", ""),
                    exc,
                    extra={
                        "event": "ENTRY_PARTIAL_FILL_RECONCILE_FAILED",
                        "order_id": getattr(order, "order_id", ""),
                        "error_type": type(exc).__name__,
                    },
                )
        return updated


__all__ = [
    "RuntimeOrderManager",
    "_enrich_trade_plan_exit_provenance",
    "_strip_exit_identity_kwargs",
    "_submit_core_with_exit_provenance",
]
