from unittest.mock import Mock

from nifty_scalper_bot.core.strategy_manager import StrategyManager


class _S:
    def __init__(self, name: str):
        self.name = name
        self.last_no_vote_reason = 'none'
        self.generate_signal = Mock(return_value=None)


def _manager() -> StrategyManager:
    m = StrategyManager(Mock(), Mock(), None)
    m._indicator_engine.get_history.return_value = [1] * 40
    m._indicator_engine.get_indicators.return_value = {
        'vwap': 1.0, 'avg_volume': 1.0, 'volume': 1.0, 'direction_bias': 'CE'
    }
    m._required_indicators = {'vwap', 'avg_volume'}
    m._strategies = [_S('VWAPPro'), _S('OrderFlow'), _S('BBSqueeze')]
    return m


def test_option_strategies_skipped_for_fut_symbol() -> None:
    m = _manager()
    m.generate_signal('NFO:NIFTY26MAYFUT', 100.0)
    assert m._strategies[0].generate_signal.call_count == 0
    assert m._strategies[1].generate_signal.call_count == 0


def test_bb_squeeze_skipped_for_option_symbol() -> None:
    m = _manager()
    m.generate_signal('NFO:NIFTY26MAY23900CE', 100.0)
    assert m._strategies[2].generate_signal.call_count == 0
