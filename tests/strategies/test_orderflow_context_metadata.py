from nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy


class _Dummy:
    pass


def test_orderflow_context_metadata_contract_side() -> None:
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(), _Dummy())
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500CE', {'bid': 100.0, 'ask': 101.0, 'buy_qty': 100, 'sell_qty': 90, 'tick_direction': 'UP', 'spread_pct': 5.0, 'atr': 2.0, 'direction_bias': 'UNKNOWN'}, 100.5)
    assert signal is not None
    metadata = signal.metadata
    assert metadata['role'] == 'context'
    assert metadata['contract_side'] in {'CE', 'PE'}
    assert metadata['trade_side'] == metadata['contract_side']
    assert metadata['direction_bias'] is None
    assert 'context_bonus_score' in metadata
    assert 'context_veto_score' in metadata
