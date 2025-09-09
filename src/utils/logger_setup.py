"""Helpers to configure application logging."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    level: str = "INFO", log_file: Optional[str] = None, json: bool = False
) -> None:
    """Configure the root logger.

    Parameters
    ----------
    level:
        Logging level name, e.g. ``"INFO"``.
    log_file:
        Optional path of a file to write logs to with rotation.
    json:
        If ``True`` attempt to format logs as JSON.
    """

    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    if json:
        try:
            import json_log_formatter  # type: ignore

            formatter = json_log_formatter.JSONFormatter()
        except Exception:
            formatter = logging.Formatter(fmt)
    else:
        formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(lvl)

    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
        fh.setLevel(lvl)
        fh.setFormatter(formatter)
        root.addHandler(fh)
