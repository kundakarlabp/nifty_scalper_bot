"""Native entry gate shared by the explicit runtime order manager."""

from __future__ import annotations

from contextlib import suppress
import inspect
from typing import Any, Mapping


NO_BLOCK = object()


def configure_provider(manager: Any, provider: Any | None) -> None:
    manager._unresolved_exit_provider = provider
    manager._unresolved_exit_guard_installed = provider is not None


def _protective_place_order(
    base_place_order: Any,
    manager: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> bool:
    try:
        bound = inspect.signature(base_place_order).bind_partial(
            manager,
            *args,
            **dict(kwargs),
        )
        values = {
            key: value
            for key, value in bound.arguments.items()
            if key != "self"
        }
    except Exception:
        values = dict(kwargs)
    if bool(values.get("reduce_only")) or bool(values.get("is_exit")):
        return True
    tag = str(values.get("tag") or "").strip().upper()
    return tag.startswith(
        ("EXIT_", "SL_", "TP_", "EOD_", "PANIC", "FLATTEN", "SQUAREOFF")
    )


def unresolved_details(manager: Any) -> dict[str, Any] | None:
    provider = getattr(manager, "_unresolved_exit_provider", None)
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
    manager._last_order_decision = dict(details)
    setter = getattr(manager, "set_last_skip_reason", None)
    if callable(setter):
        with suppress(Exception):
            setter("unresolved_exit_position")
    logger = getattr(manager, "_logger", None)
    log = getattr(logger, "critical", None)
    if callable(log):
        log(
            "ENTRY_BLOCKED_UNRESOLVED_EXIT bracket_id=%s",
            bracket_id,
            extra={
                "event": "ENTRY_BLOCKED_UNRESOLVED_EXIT",
                "bracket_id": bracket_id,
                "native_gate": True,
            },
        )
    return details


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
    if method_name == "submit_trade_plan_result":
        return base_module.TradePlanSubmitResult(
            accepted=False,
            order_id=None,
            reason="unresolved_exit_position",
            details=details,
            broker_attempted=False,
        )
    if method_name == "place_managed_order_result":
        return base_module.ManagedOrderResult(
            accepted=False,
            order_id=None,
            reason="unresolved_exit_position",
            details=details,
            broker_attempted=False,
        )
    return None


__all__ = ["NO_BLOCK", "block_result", "configure_provider", "unresolved_details"]
