from __future__ import annotations

from unittest.mock import Mock

from nifty_scalper_bot.core.strategy_manager import StrategyManager


class _OrderFlowStub:
    name = 'OrderFlow'

    def __init__(self) -> None:
        self.last_no_vote_reason = 'ltp_only_no_depth'
        self.generate_signal = Mock(return_value=None)


def test_ltp_only_orderflow_reason_surfaces_without_crash() -> None:
    manager = StrategyManager([], Mock(), None)
    manager._indicator_engine.get_history.return_value = [1] * 40
    manager._indicator_engine.get_indicators.return_value = {
        'vwap': 100.0,
        'avg_volume': 5_000.0,
        'volume': 2_000.0,
        'quote_quality': 'ltp_only',
    }
    stub = _OrderFlowStub()
    manager._strategies = [stub]
    assert manager.generate_signal('NFO:NIFTY26MAY23900CE', 100.0) is None
    assert stub.last_no_vote_reason == 'ltp_only_no_depth'
