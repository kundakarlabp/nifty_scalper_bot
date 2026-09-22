from __future__ import annotations

from pathlib import Path

import pytest

from nifty_scalper_bot.core.strategy_manager import _signal_score_for_diagnostics
from nifty_scalper_bot.strategies.signal_generator import Signal


def _signal(**metadata) -> Signal:
    return Signal(
        action="BUY",
        symbol="NFO:NIFTY2662324050CE",
        quantity=3,
        confidence=0.88,
        reason="test",
        stop_loss=95.0,
        take_profit=110.0,
        metadata=dict(metadata),
    )


def test_native_signal_score_prefers_final_trade_score_over_quantity() -> None:
    signal = _signal(final_trade_score=8.75, setup_score=9.0, raw_setup_score=9.0)

    assert _signal_score_for_diagnostics(signal) == pytest.approx(8.75)
    assert signal.quantity == 3


def test_native_signal_score_uses_setup_score_when_final_missing() -> None:
    signal = _signal(setup_score=9.0, raw_setup_score=8.5)

    assert _signal_score_for_diagnostics(signal) == pytest.approx(9.0)


def test_native_signal_score_returns_none_without_score_metadata() -> None:
    signal = _signal()

    assert _signal_score_for_diagnostics(signal) is None


def test_strategy_manager_does_not_use_quantity_as_signal_score() -> None:
    source = Path("src/nifty_scalper_bot/core/strategy_manager.py").read_text(
        encoding="utf-8"
    )

    assert "signal_score = float(combined.quantity)" not in source
    assert "signal_score = _signal_score_for_diagnostics(combined)" in source


def test_core_does_not_install_strategy_exit_score_monkey_patch() -> None:
    source = Path("src/nifty_scalper_bot/core/__init__.py").read_text(encoding="utf-8")

    assert "strategy_exit_score_diagnostics" not in source
