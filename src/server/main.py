"""Standalone entry point for the diagnostics health server."""

from __future__ import annotations

from src.server.health import run as run_health_server
from src.server.logging_setup import setup_root_logger


def main() -> None:
    """Configure logging and start the health server."""

    setup_root_logger()
    run_health_server()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
