"""Diagnostic script to validate Zerodha margin parsing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient  # noqa: E402
from nifty_scalper_bot.utils.logging import get_logger  # noqa: E402

LOGGER = get_logger(__name__)


def _sanitize_token(value: str | None, visible: int = 4) -> str:
    """Return sanitized token for logging.

    Args:
        value: Raw credential string.
        visible: Number of characters to expose from the start.

    Returns:
        str: Sanitized representation safe for logs.

    Raises:
        None.
    """

    if not value:
        return "missing"
    token = value.strip()
    if not token:
        return "missing"
    if len(token) <= visible:
        return "*" * len(token)
    return f"{token[:visible]}***"


def _log_segment_summary(segment: str, payload: dict[str, float]) -> None:
    """Log normalized margin summary for a segment.

    Args:
        segment: Zerodha segment identifier.
        payload: Normalized margin snapshot.

    Returns:
        None.

    Raises:
        None.
    """

    LOGGER.info(
        "balance_test_segment",
        extra={
            "event": "balance_test_segment",
            "segment": segment,
            "available": payload.get("available"),
            "used": payload.get("used"),
            "net": payload.get("net"),
        },
    )


def _ensure_client() -> ZerodhaKiteClient:
    """Build Zerodha client from environment variables.

    Args:
        None.

    Returns:
        ZerodhaKiteClient: Authenticated client instance.

    Raises:
        ConfigurationError: If credentials are missing or invalid.
    """

    api_key = (
        os.getenv("BROKER_API_KEY")
        or os.getenv("ZERODHA_API_KEY")
        or os.getenv("KITE_API_KEY")
        or ""
    )
    api_secret = (
        os.getenv("BROKER_API_SECRET")
        or os.getenv("ZERODHA_API_SECRET")
        or os.getenv("KITE_API_SECRET")
        or ""
    )
    access_token = (
        os.getenv("BROKER_ACCESS_TOKEN")
        or os.getenv("ZERODHA_ACCESS_TOKEN")
        or os.getenv("KITE_ACCESS_TOKEN")
        or ""
    )

    LOGGER.info(
        "balance_test_credentials",
        extra={
            "event": "balance_test_credentials",
            "api_key": _sanitize_token(api_key),
            "access_token": _sanitize_token(access_token),
        },
    )

    return ZerodhaKiteClient(
        api_key=api_key,
        api_secret=api_secret,
        access_token=access_token,
    )


def main() -> int:
    """Execute margin diagnostic workflow.

    Args:
        None.

    Returns:
        int: ``0`` on success, non-zero otherwise.

    Raises:
        None.
    """

    LOGGER.debug(
        "Entered test_balance_fetch.main",
        extra={"event": "balance_test_enter"},
    )

    try:
        client = _ensure_client()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "balance_test_client_init_failed",
            extra={"event": "balance_test_client_init_failed", "error": str(exc)},
            exc_info=exc,
        )
        return 1

    exit_code = 0
    for segment in ("equity", "commodity"):
        try:
            summary = client.get_margin_summary(segment=segment)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "balance_test_segment_failed",
                extra={
                    "event": "balance_test_segment_failed",
                    "segment": segment,
                    "error": str(exc),
                },
                exc_info=exc,
            )
            exit_code = 1
            continue
        _log_segment_summary(segment, summary)

    try:
        available = client.get_available_balance()
    except Exception as exc:  # noqa: BLE001
        LOGGER.error(
            "balance_test_available_failed",
            extra={"event": "balance_test_available_failed", "error": str(exc)},
            exc_info=exc,
        )
        exit_code = 1
    else:
        LOGGER.info(
            "balance_test_available",
            extra={"event": "balance_test_available", "available": available},
        )

    LOGGER.info(
        "balance_test_complete",
        extra={
            "event": "balance_test_complete",
            "status": "ok" if exit_code == 0 else "error",
        },
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
