from nifty_scalper_bot.core.strategy_manager import (
    StrategyManager,
    signal_to_vote,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


class _DummyEngine:
    pass


BAR_TS = 1_785_000_000.0


def _indicators(**updates):
    payload = {
        'vwap': 100.0,
        'atr': 5.0,
        'close': 103.0,
        'open': 100.0,
        'high': 104.0,
        'low': 99.0,
        'volume': 1000.0,
        'avg_volume': 900.0,
        'spread_pct': 0.5,
        'bid': 102.5,
        'ask': 103.0,
        'quote_depth_valid': True,
        'tradable_quote': True,
        'direction_bias': 'CE',
        'underlying_direction_bias': 'CE',
        'underlying_direction_confidence': 0.95,
        'context_age_seconds': 0.0,
        'context_fresh': True,
        'regime': 'TREND_UP',
        'stale_data_used': False,
        'latest_bar_ts': BAR_TS,
    }
    payload.update(updates)
    return payload


def test_vwap_metadata_contract_side_and_setup_scores_present():
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    signal = strategy._evaluate_signal(
        'NFO:NIFTY26FEB22500CE',
        _indicators(direction_bias='UNKNOWN', underlying_direction_bias='UNKNOWN'),
        101.0,
    )
    assert signal is not None
    metadata = signal.metadata
    assert metadata['contract_side'] == 'CE'
    assert metadata['direction_bias'] is None
    assert metadata['raw_setup_score'] is not None
    assert metadata['setup_score'] is not None
    assert metadata['setup_id'].startswith('vwap:CE:')


def test_live_vwap_rejects_threshold_pass_from_small_noise(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())

    signal = strategy._evaluate_signal(
        'NFO:NIFTY26FEB22500CE',
        _indicators(
            close=100.05,
            open=100.04,
            high=100.10,
            low=99.90,
            volume=0.0,
            avg_volume=0.0,
        ),
        100.05,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == 'vwap_event_unconfirmed'


def test_live_vwap_accepts_meaningful_atr_penetration(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())

    signal = strategy._evaluate_signal(
        'NFO:NIFTY26FEB22500CE',
        _indicators(
            close=100.80,
            open=99.80,
            high=100.85,
            low=99.50,
            volume=0.0,
            avg_volume=0.0,
        ),
        100.80,
    )

    assert signal is not None
    assert signal.metadata['penetration_confirmed'] is True
    assert signal.metadata['vwap_event_confirmed'] is True


def test_vwap_thesis_uses_stable_structural_id_until_closed_candle_reset(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    symbol = 'NFO:NIFTY26FEB22500CE'

    first = strategy._evaluate_signal(symbol, _indicators(), 103.0)
    assert first is not None
    strategy.notify_entry_accepted('CE')

    same_thesis = strategy._evaluate_signal(
        symbol,
        _indicators(
            latest_bar_ts=BAR_TS + 60.0,
            close=104.0,
            open=103.8,
            high=104.2,
            low=103.5,
        ),
        104.0,
    )
    assert same_thesis is not None
    assert same_thesis.metadata['setup_id'] == first.metadata['setup_id']

    assert strategy._evaluate_signal(
        symbol,
        _indicators(
            latest_bar_ts=BAR_TS + 120.0,
            close=99.5,
            open=100.0,
            high=100.2,
            low=99.2,
        ),
        99.5,
    ) is None
    assert strategy.last_no_vote_reason == 'vwap_thesis_reset'

    next_thesis = strategy._evaluate_signal(
        symbol,
        _indicators(
            latest_bar_ts=BAR_TS + 180.0,
            close=103.0,
            open=100.5,
            high=103.2,
            low=100.2,
        ),
        103.0,
    )
    assert next_thesis is not None
    assert next_thesis.metadata['setup_id'] != first.metadata['setup_id']


def test_real_vwap_vote_clears_default_live_quality_gate(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('ORDER_MAX_SPREAD_PCT', '1.0')
    strategy = VWAPProStrategy(
        VWAPProStrategyConfig(min_confidence=0.0),
        _DummyEngine(),
    )
    indicators = _indicators()

    signal = strategy.generate_signal(
        'NFO:NIFTY26FEB22500CE',
        indicators,
        103.0,
    )

    assert signal is not None
    vote = signal_to_vote(signal, 'VWAPPro')
    manager = object.__new__(StrategyManager)
    quality_score, quality_meta = manager._compute_trade_quality_score(
        vote,
        indicators,
        symbol=signal.symbol,
        selected_ok=True,
        near_atm_ok=True,
        context_votes=[],
    )

    assert quality_score >= 7.0
    assert quality_meta['quality_evidence_complete'] is True
    assert signal.metadata['direction_alignment_score'] == 2.0
    assert signal.metadata['liquidity_score'] == 2.0
    assert signal.metadata['regime_time_suitability_score'] == 1.0
