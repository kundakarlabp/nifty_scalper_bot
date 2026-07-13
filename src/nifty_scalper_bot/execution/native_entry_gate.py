# mypy: ignore-errors
"""Native entry gate shared by the explicit runtime order manager."""

from __future__ import annotations

import inspect
from contextlib import suppress
from typing import Any, Mapping

NO_BLOCK = object()
_PROTECTIVE_INTENTS = {"EXIT", "REDUCE"}
_LEGACY_PROTECTIVE_TAG_PREFIXES = (
    "EXIT_",
    "SL_",
    "TP_",
    "EOD_",
    "PANIC",
    "FLATTEN",
    "SQUAREOFF",
)
_PROVIDER_BLOCKER_METHODS = (
    "current_entry_blocker",
    "current_execution_blocker",
    "current_reconciliation_blocker",
)


def configure_provider(manager: Any, provider: Any | None) -> None:
    manager._unresolved_exit_provider = provider
    manager._unresolved_exit_guard_installed = provider is not None


def _bound_place_order_values(
    base_place_order: Any,
    manager: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        bound = inspect.signature(base_place_order).bind_partial(
            manager,
            *args,
            **dict(kwargs),
        )
        return {key: value for key, value in bound.arguments.items() if key != "self"}
    except Exception:
        return dict(kwargs)


def _protective_place_order(
    base_place_order: Any,
    manager: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> bool:
    values = _bound_place_order_values(base_place_order, manager, args, kwargs)
    intent = str(values.get("intent") or "").strip().upper()
    if intent in _PROTECTIVE_INTENTS:
        return True

    # Explicit legacy flags are retained for compatibility with older restored
    # runtime objects. Tag text is diagnostic only; it is no longer sufficient to
    # prove protective status in live mode.
    if bool(values.get("reduce_only")) or bool(values.get("is_exit")):
        return True

    live_checker = getattr(manager, "is_live_mode", None)
    live_mode = False
    if callable(live_checker):
        with suppress(Exception):
            live_mode = bool(live_checker())
    if live_mode:
        return False

    tag = str(values.get("tag") or "").strip().upper()
    return tag.startswith(_LEGACY_PROTECTIVE_TAG_PREFIXES)


def _normalise_provider_block(result: Any, *, source: str) -> dict[str, Any] | None:
    if not result:
        return None
    if isinstance(result, Mapping):
        reason = (
            result.get("block_reason")
            or result.get("reason")
            or result.get("blocker")
            or source
        )
        details = dict(result)
    else:
        reason = str(result)
        details = {"reason": reason}
    reason_token = str(reason or source).strip() or source
    details.update(
        {
            "block_reason": reason_token,
            "broker_attempted": False,
            "retryable": False,
            "provider_blocker": True,
            "provider_blocker_source": source,
        }
    )
    return details


def _provider_block_details(provider: Any, manager: Any) -> dict[str, Any] | None:
    for method_name in _PROVIDER_BLOCKER_METHODS:
        checker = getattr(provider, method_name, None)
        if not callable(checker):
            continue
        try:
            try:
                result = checker()
            except TypeError:
                result = checker(manager)
        except Exception as exc:
            return {
                "block_reason": "entry_blocker_provider_error",
                "provider_error": f"{type(exc).__name__}: {exc}",
                "provider_blocker": True,
                "provider_blocker_source": method_name,
                "broker_attempted": False,
                "retryable": False,
            }
        details = _normalise_provider_block(result, source=method_name)
        if details is not None:
            return details
    return None


def unresolved_details(manager: Any) -> dict[str, Any] | None:
    self_blocker = getattr(manager, "current_entry_blocker", None)
    if callable(self_blocker):
        try:
            self_block = self_blocker()
        except TypeError:
            self_block = self_blocker(manager)
        details = _normalise_provider_block(self_block, source="current_entry_blocker")
        if details is not None:
            _record_block(manager, details)
            return details
    provider = getattr(manager, "_unresolved_exit_provider", None)
    provider_block = _provider_block_details(provider, manager)
    if provider_block is not None:
        _record_block(manager, provider_block)
        return provider_block

    checker = getattr(provider, "has_unresolved_exit", None)
    if not callable(checker):
        return None
    try:
        unresolved = bool(checker())
        provider_error = None
    except Exception as exc:
        unresolved = True
        provider_error = f"{type(exc).__name__}: {exc}"
    if not unresolved:
        return None
    bracket_id = None
    getter = getattr(provider, "get_first_unresolved_exit_bracket_id", None)
    if callable(getter):
        with suppress(Exception):
            bracket_id = getter()
    details: dict[str, Any] = {
        "block_reason": "unresolved_exit_position",
        "bracket_id": bracket_id,
        "broker_attempted": False,
        "retryable": False,
    }
    if provider_error:
        details["provider_error"] = provider_error
    _record_block(manager, details)
    return details


def _record_block(manager: Any, details: Mapping[str, Any]) -> None:
    reason = str(details.get("block_reason") or "entry_blocked")
    manager._last_order_decision = dict(details)
    setter = getattr(manager, "set_last_skip_reason", None)
    if callable(setter):
        with suppress(Exception):
            setter(reason)
    logger = getattr(manager, "_logger", None)
    log = getattr(logger, "critical", None)
    if callable(log):
        log(
            "ENTRY_BLOCKED_NATIVE_GATE reason=%s bracket_id=%s",
            reason,
            details.get("bracket_id"),
            extra={
                "event": "ENTRY_BLOCKED_NATIVE_GATE",
                "block_reason": reason,
                "bracket_id": details.get("bracket_id"),
                "native_gate": True,
            },
        )


def block_result(
    manager: Any,
    base_module: Any,
    base_place_order: Any,
    method_name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    if method_name == "place_order" and _protective_place_order(
        base_place_order,
        manager,
        args,
        kwargs,
    ):
        return NO_BLOCK
    details = unresolved_details(manager)
    if details is None:
        return NO_BLOCK
    reason = str(details.get("block_reason") or "entry_blocked")
    if method_name == "submit_trade_plan_result":
        return base_module.TradePlanSubmitResult(
            accepted=False,
            order_id=None,
            reason=reason,
            details=details,
            broker_attempted=False,
        )
    if method_name == "place_managed_order_result":
        return base_module.ManagedOrderResult(
            accepted=False,
            order_id=None,
            reason=reason,
            details=details,
            broker_attempted=False,
        )
    return None


__all__ = ["NO_BLOCK", "block_result", "configure_provider", "unresolved_details"]
