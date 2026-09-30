"""Compatibility verifier for native WebSocket market-data hardening.

WebSocketManager now owns its tick-ingress, NSE-session, and close-safety
invariants directly.  This function remains temporarily for callers/tests that
still invoke the historical installer; it must never mutate the class.
"""

from __future__ import annotations

from typing import Any

_INSTALLED_ATTR = "_market_data_hardening_installed"


def install_websocket_market_data_hardening(manager_cls: type[Any]) -> None:
    """Verify that the transport exposes the native hardening contract."""
    required = (
        "_build_ticker",
        "_make_ticker_close_safe",
        "_is_within_trading_window",
        "_on_ticks",
        "_on_close",
    )
    missing = [
        name for name in required if not callable(getattr(manager_cls, name, None))
    ]
    if missing or not bool(getattr(manager_cls, _INSTALLED_ATTR, False)):
        raise RuntimeError(
            "websocket_native_hardening_missing "
            f"class={getattr(manager_cls, '__name__', manager_cls)!s} missing={missing}"
        )


__all__ = ["install_websocket_market_data_hardening"]
