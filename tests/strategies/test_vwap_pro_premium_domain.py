from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


class _Dummy:
    pass


def _base_ind():
    return {
        'vwap': 100.0, 'open': 99.0, 'high': 102.0, 'low': 98.0, 'close': 101.0,
        'atr': 2.0, 'volume': 1000, 'avg_volume': 1000, 'direction_bias': 'PE',
        'spread_pct': 1.0,
    }


def test_vwap_pro_never_rewards_pe_only_for_trading_below_vwap():
    st = VWAPProStrategy(VWAPProStrategyConfig(enabled=True), _Dummy())
    ind = _base_ind()
    sig = st._evaluate_signal('NIFTY25MAY25000PE', ind, 99.0)
    assert sig is None
