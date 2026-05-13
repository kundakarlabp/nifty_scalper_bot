from nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy


def test_orderflow_no_missing_depth_when_bid_ask_and_depth_exist() -> None:
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None)
    indicators = {
        'bid': 100.0,
        'ask': 101.0,
        'spread_pct': 0.99,
        'depth': {'buy': [{'quantity': 200}], 'sell': [{'quantity': 100}]},
        'tick_direction': 'UP',
        'direction_bias': 'CE',
        'atr': 2.0,
        'data_age_seconds': 0.2,
    }
    signal = strategy._evaluate_signal('NFO:NIFTY26MAY24000CE', indicators, current_price=100.5)
    assert signal is not None
    assert getattr(strategy, 'last_no_vote_reason', None) != 'missing_depth'

from nifty_scalper_bot.strategies.indicators import IndicatorEngine


def test_runtime_context_depth_keys_are_preserved() -> None:
    engine = IndicatorEngine()
    engine.set_runtime_context(
        'NFO:NIFTY26MAY24000CE',
        {'depth': {'buy': [{'price': 10.0}], 'sell': [{'price': 11.0}]}, 'depth_available': True, 'bid': 10.0, 'ask': 11.0},
    )
    ctx = engine.get_runtime_context('NFO:NIFTY26MAY24000CE')
    assert ctx['depth_available'] is True
    assert ctx['depth']['buy'][0]['price'] == 10.0
