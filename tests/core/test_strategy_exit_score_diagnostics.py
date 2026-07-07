from __future__ import annotations

import pytest

from nifty_scalper_bot.core.strategy_exit_score_diagnostics import _extract_signal_score
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


def test_extract_signal_score_prefers_final_trade_score_over_quantity() -> None:
    signal = _signal(final_trade_score=8.75, setup_score=9.0, raw_setup_score=9.0)

    assert _extract_signal_score(signal) == pytest.approx(8.75)
    assert signal.quantity == 3


def test_extract_signal_score_uses_setup_score_when_final_missing() -> None:
    signal = _signal(setup_score=9.0, raw_setup_score=8.5)

    assert _extract_signal_score(signal) == pytest.approx(9.0)


def test_extract_signal_score_returns_none_without_score_metadata() -> None:
    signal = _signal()

    assert _extract_signal_score(signal) is None
