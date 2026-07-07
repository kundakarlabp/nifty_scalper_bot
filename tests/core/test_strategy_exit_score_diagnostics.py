from __future__ import annotations

import logging

import pytest

from nifty_scalper_bot.core.strategy_exit_score_diagnostics import _extract_signal_score, apply_patches
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
    signal = _signal(final_trade_score=8.75, setup_score=9.0)

    assert _extract_signal_score(signal) == pytest.approx(8.75)
    assert signal.quantity == 3


def test_exit_score_patch_emits_score_and_quantity_separately(monkeypatch, caplog) -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    apply_patches()
    manager = StrategyManager([], None, None)
    monkeypatch.setattr(
        manager,
        "_strategy_exit_score_diagnostics_original_generate_signal",
        lambda _symbol, _price, trace_id=None: _signal(
            final_trade_score=8.75,
            setup_score=9.0,
            approval_path="aligned_two_trigger_consensus",
        ),
        raising=False,
    )

    # Simulate the wrapped call directly by invoking the patched class method while
    # forcing the original pointer used by the wrapper to return a controlled signal.
    original = StrategyManager._strategy_exit_score_diagnostics_original_generate_signal
    monkeypatch.setattr(
        StrategyManager,
        "_strategy_exit_score_diagnostics_original_generate_signal",
        lambda self, symbol, current_price, trace_id=None: _signal(
            final_trade_score=8.75,
            setup_score=9.0,
            approval_path="aligned_two_trigger_consensus",
        ),
        raising=False,
    )
    try:
        with caplog.at_level(logging.INFO, logger="nifty_scalper_bot.core.strategy_exit_score_diagnostics"):
            signal = manager.generate_signal("NFO:NIFTY2662324050CE", 100.0, trace_id="score-diag")
    finally:
        monkeypatch.setattr(
            StrategyManager,
            "_strategy_exit_score_diagnostics_original_generate_signal",
            original,
            raising=False,
        )

    assert signal is not None
    records = [record for record in caplog.records if getattr(record, "event", "") == "STRATEGY_MANAGER_EXIT_SCORE"]
    assert records
    assert records[-1].signal_score == pytest.approx(8.75)
    assert records[-1].signal_quantity == 3
