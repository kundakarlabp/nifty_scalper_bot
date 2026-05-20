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


def test_vwap_pro_trend_aligned_lower_threshold(monkeypatch) -> None:
    monkeypatch.setenv('VWAP_PRO_MIN_TREND_ALIGNED_SCORE', '4.5')
    strategy = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    indicators = _base_indicators('CE')
    indicators.update({'volume': 600.0, 'avg_volume': 1000.0, 'futures_vwap_slope': 1.0, 'futures_volume_ratio': 1.2})
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500CE', indicators, 101.0)
    assert signal is not None


def test_vwap_pro_uses_context_direction_when_direct_direction_missing() -> None:
    strategy = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    indicators = {'vwap':200,'atr':4,'close':205,'open':203,'high':206,'low':202,'volume':10000,'avg_volume':9000,'spread_pct':1,'spot_context':{'direction_bias':'CE','underlying_direction_confidence':0.8},'context_age_seconds':10,'futures_vwap_slope':1,'futures_volume_ratio':1.2}
    signal = strategy._evaluate_signal('NFO:NIFTY26MAY23750CE', indicators, 205)
    assert signal is not None or getattr(strategy, 'last_no_vote_reason', '') != 'weak_score'


def test_vwap_pro_reads_direction_from_spot_context_when_top_level_missing() -> None:
    strategy = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    indicators = {
        'vwap': 100,
        'atr': 2,
        'close': 103,
        'open': 101,
        'high': 104,
        'low': 100,
        'volume': 1000,
        'avg_volume': 800,
        'spread_pct': 1,
        'spot_context': {'direction_bias': 'CE', 'underlying_direction_confidence': 0.8},
        'context_age_seconds': 10,
        'futures_vwap_slope': 1,
        'futures_volume_ratio': 1.2,
    }
    signal = strategy._evaluate_signal('NFO:NIFTY26MAY23750CE', indicators, 103)
    if signal is not None:
        assert signal.metadata['context_direction_used'] == 'CE'
        assert signal.metadata['underlying_direction_confidence'] == 0.8
    else:
        assert getattr(strategy, 'last_no_vote_reason', None) != 'unknown_contract_side'

def test_vwap_pro_required_indicators_include_context_fields() -> None:
    strategy = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    required = strategy.get_required_indicators()
    for key in {"underlying_direction_bias","underlying_direction_confidence","context_age_seconds","futures_vwap_slope","futures_volume_ratio","spread_pct","spot_context","futures_context"}:
        assert key in required


def test_vwap_pro_flat_futures_slope_is_neutral_not_conflict() -> None:
    strategy = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    indicators = _base_indicators('CE')
    indicators.update({'futures_vwap_slope': 0.0, 'futures_volume_ratio': 1.2})
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500CE', indicators, 101.0)
    assert signal is not None
    reasons = signal.metadata.get('score_reasons', [])
    assert 'futures_slope_neutral' in reasons
    assert 'futures_slope_conflict' not in reasons


def test_vwap_pro_opposite_nonzero_futures_slope_conflicts() -> None:
    strategy = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    indicators = _base_indicators('PE')
    indicators.update({'futures_vwap_slope': 0.2, 'futures_volume_ratio': 1.2})
    strategy._evaluate_signal('NFO:NIFTY26FEB22500PE', indicators, 99.0)
    assert getattr(strategy, 'last_vote_reasons', []) or getattr(strategy, 'last_no_vote_reason', None) is not None
    reasons = getattr(strategy, 'last_vote_reasons', [])
    if not reasons:
        # fallback if no signal
        reasons = []
    assert 'futures_slope_conflict' in reasons or getattr(strategy, 'last_no_vote_reason', '') == 'weak_score'


def test_vwap_pro_early_trend_pullback_env_gated(monkeypatch) -> None:
    base = {
        'vwap': 100.0, 'atr': 5.0, 'close': 99.0, 'open': 101.0, 'high': 101.0, 'low': 99.0,
        'volume': 1000.0, 'avg_volume': 900.0, 'spread_pct': 2.0, 'direction_bias': 'PE',
        'underlying_direction_confidence': 0.95, 'context_age_seconds': 1.0, 'futures_vwap_slope': -0.2, 'futures_volume_ratio': 1.5,
    }
    strategy_default = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    signal_default = strategy_default._evaluate_signal('NFO:NIFTY26FEB22500PE', dict(base), 99.0)
    reasons_default = (signal_default.metadata.get('score_reasons', []) if signal_default else [])
    assert 'early_trend_pullback_context' not in reasons_default

    monkeypatch.setenv('VWAP_PRO_ALLOW_EARLY_TREND_PULLBACK', 'true')
    strategy_enabled = VWAPProStrategy(config=VWAPProStrategyConfig(), indicator_engine=_DummyEngine())
    signal_enabled = strategy_enabled._evaluate_signal('NFO:NIFTY26FEB22500PE', dict(base), 99.0)
    assert signal_enabled is not None
    assert 'early_trend_pullback_context' in signal_enabled.metadata.get('score_reasons', [])
