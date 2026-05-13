from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


class _DummyEngine:
    pass


def test_vwap_metadata_contract_side_and_setup_scores_present():
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500CE', {'vwap': 100.0, 'atr': 5.0, 'close': 103.0, 'open': 100.0, 'high': 104.0, 'low': 99.0, 'volume': 1000.0, 'avg_volume': 900.0, 'spread_pct': 2.0, 'direction_bias': 'UNKNOWN'}, 101.0)
    assert signal is not None
    metadata = signal.metadata
    assert metadata['contract_side'] == 'CE'
    assert metadata['direction_bias'] is None
    assert metadata['raw_setup_score'] is not None
    assert metadata['setup_score'] is not None
