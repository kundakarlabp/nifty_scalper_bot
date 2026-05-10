from nifty_scalper_bot.core.strategy_manager import StrategyVote, signal_to_vote, StrategyManager
from nifty_scalper_bot.strategies.signal_generator import Signal


class _NoopStrategy:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_required_indicators(self) -> list[str]:
        return []

    def generate_signal(self, symbol: str, indicators: dict, current_price: float, position=None):
        return None


class _DummyIndicatorEngine:
    def get_indicators(self, symbol: str, names):
        return {str(n): 0.0 for n in names}


class _DummyPositionManager:
    def get_position(self, symbol: str):
        return None


def test_signal_to_vote_uses_confidence_when_strategy_score_missing() -> None:
    signal = Signal(action='BUY', symbol='NFO:NIFTY26FEB22500CE', quantity=1, confidence=0.72, metadata={})
    vote = signal_to_vote(signal, 'VWAPPro')
    assert vote.score == 7.2
    assert vote.metadata['vote_score_from_confidence'] == 7.2


def test_signal_to_vote_overrides_conflicting_metadata_side_with_symbol_side() -> None:
    signal = Signal(
        action='BUY',
        symbol='NFO:NIFTY26FEB22500CE',
        quantity=1,
        confidence=0.8,
        metadata={'trade_side': 'PE', 'strategy_score': 8.0},
    )
    vote = signal_to_vote(signal, 'VWAPPro')
    assert vote.side == 'CE'
    assert vote.metadata['side_conflict'] is True
    assert vote.metadata['side_from_metadata'] == 'PE'


def test_apply_regime_vote_weight_preserves_raw_strategy_score() -> None:
    manager = StrategyManager([_NoopStrategy('VWAPPro')], _DummyIndicatorEngine(), _DummyPositionManager())
    vote = StrategyVote(
        strategy='VWAPPro',
        side='CE',
        score=8.0,
        confidence=0.8,
        reasons=[],
        metadata={'strategy_score': 8.0},
    )
    weighted = manager._apply_regime_vote_weight(vote=vote, regime_name='BULL')
    assert weighted.metadata['strategy_score'] == 8.0
    assert 'regime_weighted_vote_score' in weighted.metadata
