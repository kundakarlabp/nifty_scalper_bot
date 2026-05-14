from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_combined_metadata_preserves_refactor_fields() -> None:
    manager = StrategyManager.__new__(StrategyManager)
    signal = Signal(action='BUY', symbol='NFO:NIFTY25000CE', quantity=1, confidence=0.65, reason='VWAPPro', stop_loss=1.0, take_profit=2.0, metadata={'is_selected_option': True})
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=6.2, confidence=0.65, reasons=[], metadata={'raw_setup_score': 6.2, 'regime_weight': 1.0})
    out = manager._combine_strategy_votes(symbol=signal.symbol, signals=[(signal, vote)], indicators={})
    assert out is not None
    for key in ('raw_setup_score', 'setup_score', 'regime_weighted_score', 'regime_weight', 'context_bonus', 'context_penalty', 'final_trade_score', 'consensus_score'):
        assert key in out.metadata
