from __future__ import annotations

import ast
import logging
from pathlib import Path

from nifty_scalper_bot.utils.logging import log_throttled


def test_log_throttled_accepts_format_args(caplog) -> None:
    logger = logging.getLogger('nifty_scalper_bot.tests.log_throttled')
    caplog.set_level(logging.INFO, logger=logger.name)
    log_throttled(logger, 'k', 'value=%s', 1, interval_sec=0.0)
    assert 'value=1' in caplog.text


def test_no_invalid_log_throttled_calls() -> None:
    root = Path('src/nifty_scalper_bot')
    violations: list[str] = []
    for path in root.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_target = isinstance(func, ast.Name) and func.id == 'log_throttled'
            is_target = is_target or (
                isinstance(func, ast.Attribute) and func.attr == 'log_throttled'
            )
            if is_target and len(node.args) > 4:
                violations.append(f'{path}:{node.lineno}')
    assert not violations, f'Invalid log_throttled calls: {violations}'


def test_log_throttled_invalid_logger_is_safe(caplog) -> None:
    caplog.set_level(logging.INFO)
    log_throttled(123, "bad_logger", "safe message", interval_sec=0.0)
    assert "safe message" in caplog.text
