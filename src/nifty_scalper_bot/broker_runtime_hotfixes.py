"""Runtime safety fixes for broker error classification.

This module is intentionally small and loaded from the package initializer so the
fix applies to every production import path, including direct imports of
``data.rest.zerodha_client``.  It can be removed once the same changes are folded
into the large Zerodha adapter module.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Mapping, NoReturn


def apply_broker_runtime_hotfixes() -> None:
    """Install deterministic Zerodha authentication/error handling once."""

    from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
    from nifty_scalper_bot.utils.errors import (
        BrokerError,
        ConfigurationError,
        OrderPlacementError,
    )

    if getattr(ZerodhaKiteClient, "_runtime_auth_hotfix_applied", False):
        return

    @staticmethod
    def _is_authentication_failure(
        *,
        status_code: int | None,
        payload: Mapping[str, Any] | None,
        error_text: str,
    ) -> bool:
        """Recognize all common Kite token/session failure spellings."""

        if status_code in {401, 403}:
            return True

        fragments: list[str] = [error_text or ""]
        if isinstance(payload, Mapping):
            for key in ("message", "error_type", "status", "error"):
                value = payload.get(key)
                if value is not None:
                    fragments.append(str(value))

        text = " ".join(fragments).lower()
        compact = "".join(ch for ch in text if ch.isalnum())
        exact_fragments = (
            "incorrect api_key",
            "incorrect access_token",
            "invalid session",
            "token expired",
            "permission denied",
            "authentication failed",
            "unauthorized",
        )
        compact_tokens = (
            "tokenexception",
            "invalidtoken",
            "incorrectapikey",
            "incorrectaccesstoken",
            "invalidsession",
            "tokenexpired",
            "authenticationfailed",
        )
        return any(token in text for token in exact_fragments) or any(
            token in compact for token in compact_tokens
        )

    def _raise_for_status(
        self: Any,
        response: Any,
        expect_order_response: bool,
    ) -> NoReturn:
        """Convert every failed response into a defined typed exception.

        The previous implementation left ``error`` unassigned for HTTP 400 on
        quote/history/profile endpoints, masking the broker's real message with
        ``UnboundLocalError`` and leaving live readiness permanently cold.
        """

        message = self._safe_error_message(response)
        status = int(response.status_code)
        payload: Mapping[str, Any] | None = None
        with suppress(Exception):
            raw_payload = response.json()
            if isinstance(raw_payload, Mapping):
                payload = raw_payload

        if self._is_authentication_failure(
            status_code=status,
            payload=payload,
            error_text=message,
        ):
            self._mark_authentication_invalid(message)

        if status in {400, 404} and expect_order_response:
            error: Exception = OrderPlacementError(message)
        elif status == 401:
            error = ConfigurationError("Zerodha authentication failed")
        elif status == 403:
            error = ConfigurationError("Zerodha access denied")
        elif status == 429:
            error = BrokerError("Zerodha rate limit exceeded")
        else:
            error = BrokerError(message)

        # Use the adapter's module logger through the method global namespace.
        logger = self._raise_for_status.__globals__.get("LOGGER")
        if logger is not None:
            logger.error(
                "Zerodha API error (%s): %s",
                status,
                message,
                extra={
                    "event": "zerodha_api_error",
                    "status_code": status,
                    "expect_order_response": bool(expect_order_response),
                    "error_type": type(error).__name__,
                },
            )
        raise error

    ZerodhaKiteClient._is_authentication_failure = _is_authentication_failure
    ZerodhaKiteClient._raise_for_status = _raise_for_status
    ZerodhaKiteClient._runtime_auth_hotfix_applied = True


__all__ = ["apply_broker_runtime_hotfixes"]
