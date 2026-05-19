"""Regression tests for log_throttled args usage."""

from __future__ import annotations

import logging
from pathlib import Path

from nifty_scalper_bot.utils.logging import log_throttled


def test_no_log_throttled_args_keyword_in_src_package() -> None:
    """Ensure no invalid args keyword is passed to log_throttled in source package."""
    src_root = Path(__file__).resolve().parents[1] / 'src' / 'nifty_scalper_bot'
    offenders: list[str] = []
    for py_path in src_root.rglob('*.py'):
        text = py_path.read_text(encoding='utf-8')
        if 'log_throttled(' in text and 'args=' in text:
            offenders.append(str(py_path.relative_to(src_root.parent.parent)))
    assert offenders == []


def test_log_throttled_accepts_positional_format_args() -> None:
    """Ensure positional format args work and do not raise TypeError."""
    logger = logging.getLogger('test_log_throttled_accepts_positional_format_args')
    log_throttled(logger, 'key', 'hello %s', 'world', interval_sec=0)
