from __future__ import annotations

import logging
from pathlib import Path

from nifty_scalper_bot.core.message_bus import MessageBus


def _preceding_call(path: str, marker: str, *, width: int = 240) -> str:
    source = Path(path).read_text(encoding="utf-8")
    marker_index = source.index(marker)
    return source[max(0, marker_index - width) : marker_index]


def _following_block(path: str, marker: str, *, width: int = 1200) -> str:
    source = Path(path).read_text(encoding="utf-8")
    marker_index = source.index(marker)
    return source[marker_index : marker_index + width]


def test_opposite_side_suppression_is_debug_diagnostic() -> None:
    call = _preceding_call(
        "src/nifty_scalper_bot/strategies/runner.py",
        '"TRIGGER_EVAL_SKIPPED symbol=%s ' "reason=opposite_side_trigger_suppressed",
    )

    assert "self._logger.debug(" in call
    assert "self._logger.warning(" not in call
    decision = _following_block(
        "src/nifty_scalper_bot/strategies/runner.py",
        '"TRIGGER_EVAL_SKIPPED symbol=%s ' "reason=opposite_side_trigger_suppressed",
    )
    assert "self._emit_runner_eval_decision(" in decision


def test_successful_strategy_subscription_summary_is_not_critical() -> None:
    call = _preceding_call(
        "src/nifty_scalper_bot/core/app.py",
        '"📊 STRATEGY SUBSCRIBED SYMBOLS: %s"',
    )

    assert "LOGGER.info(" in call
    assert "LOGGER.critical(" not in call
    empty_call = _preceding_call(
        "src/nifty_scalper_bot/core/app.py",
        '"⛔ STRATEGY SUBSCRIPTION LIST IS EMPTY"',
    )
    assert "LOGGER.critical(" in empty_call


def test_empty_message_bus_start_is_informational(caplog) -> None:
    bus = MessageBus()

    with caplog.at_level(logging.INFO):
        assert bus.start() is False

    records = [
        record
        for record in caplog.records
        if "MESSAGE_BUS_START_SKIPPED reason=no_subscribers" in record.getMessage()
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
