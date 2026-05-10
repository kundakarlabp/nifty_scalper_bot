from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


class _DummyEngine:
    pass


def _base_indicators(direction_bias: str = 'CE') -> dict[str, float | str]:
    return {
        'vwap': 100.0,
        'atr': 5.0,
        'close': 103.0,
        'open': 100.0,
        'high': 104.0,
        'low': 99.0,
        'volume': 1000.0,
        'avg_volume': 900.0,
        'spread_pct': 2.0,
        'direction_bias': direction_bias,
    }


def test_vwap_pro_never_emits_pe_for_ce_symbol() -> None:
    strategy = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500CE', _base_indicators('CE'), 101.0)
    assert signal is not None
    assert signal.metadata['side'] == 'CE'


def test_vwap_pro_never_emits_ce_for_pe_symbol() -> None:
    strategy = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500PE', _base_indicators('PE'), 99.0)
    assert signal is not None
    assert signal.metadata['side'] == 'PE'
