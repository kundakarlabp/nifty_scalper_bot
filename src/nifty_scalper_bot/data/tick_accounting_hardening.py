"""Compatibility verifier for native MarketDataManager tick accounting."""

from __future__ import annotations

from typing import Any

_INSTALLED_ATTR = "_tick_accounting_hardening_installed"


def install_tick_accounting_hardening(manager_cls: type[Any]) -> None:
    """Verify that exact tick accounting is owned natively by MarketDataManager."""
    required = (
        "_pop_pending_tick_batch",
        "_drain_latest_ticks",
        "get_tick_pressure_stats",
    )
    missing = [
        name for name in required if not callable(getattr(manager_cls, name, None))
    ]
    if missing:
        raise RuntimeError(
            "tick_accounting_native_methods_missing "
            f"class={getattr(manager_cls, '__name__', manager_cls)!s} missing={missing}"
        )
    setattr(manager_cls, _INSTALLED_ATTR, True)


__all__ = ["install_tick_accounting_hardening"]
