"""Entry point helpers for standalone server processes."""

from __future__ import annotations

from src.server.logging_setup import setup_root_logger


def main() -> None:
    """Initialize logging for server-side entry points."""

    setup_root_logger()


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
