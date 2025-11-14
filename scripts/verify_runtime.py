#!/usr/bin/env python3
"""Ensure critical runtime imports succeed during image build."""

from __future__ import annotations

import logging
from typing import Iterable

LOGGER = logging.getLogger("verify_runtime")


def configure_logger() -> None:
    """Configure logging for build verification.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.debug("Entered configure_logger")


def ok(module: str) -> None:
    """Import the provided module and exit on failure.

    Args:
        module: Dotted module path to import.

    Returns:
        None

    Raises:
        SystemExit: Raised when the module cannot be imported.
    """

    LOGGER.debug("Entered ok")
    try:
        __import__(module)
        LOGGER.info("[OK] %s", module)
    except Exception as exc:  # noqa: BLE001 - surface precise failure reason
        LOGGER.error("[FAIL] %s: %s", module, exc)
        raise SystemExit(1) from exc


def verify_runtime(modules: Iterable[str]) -> None:
    """Verify every module import to catch missing dependencies.

    Args:
        modules: Iterable of dotted module paths to validate.

    Returns:
        None

    Raises:
        SystemExit: Propagates when any module import fails.
    """

    LOGGER.debug("Entered verify_runtime")
    for module in modules:
        ok(module)


if __name__ == "__main__":
    configure_logger()
    verify_runtime(
        [
            "pydantic",
            "pydantic_settings",
            "dotenv",
            "requests",
            "fastapi",
            "uvicorn",
            "prometheus_client",
            "pandas",
        ]
    )
    LOGGER.info("[OK] verify_runtime: all critical imports present")
