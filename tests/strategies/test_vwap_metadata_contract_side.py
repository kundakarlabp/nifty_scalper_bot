from nifty_scalper_bot.strategies.elite_strategies.config_models import VWAPProStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


class _DummyEngine:
    pass


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
        'direction_bias': 'CE',
        'underlying_direction_confidence': 0.95,
        'context_age_seconds': 0.0,
    }
    payload.update(updates)
    return payload


def test_vwap_metadata_contract_side_and_setup_scores_present():
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    signal = strategy._evaluate_signal(
        'NFO:NIFTY26FEB22500CE',
        _indicators(direction_bias='UNKNOWN'),
        101.0,
    )
    assert signal is not None
    metadata = signal.metadata
    assert metadata['contract_side'] == 'CE'
    assert metadata['direction_bias'] is None
    assert metadata['raw_setup_score'] is not None
    assert metadata['setup_score'] is not None


def test_live_vwap_rejects_threshold_pass_from_small_noise(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())

    signal = strategy._evaluate_signal(
        'NFO:NIFTY26FEB22500CE',
        _indicators(
            close=100.05,
            open=100.04,
            high=100.10,
            low=100.02,
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
            open=100.75,
            high=100.85,
            low=100.70,
            volume=0.0,
            avg_volume=0.0,
        ),
        100.80,
    )

    assert signal is not None
    assert signal.metadata['penetration_confirmed'] is True
    assert signal.metadata['vwap_event_confirmed'] is True


def test_accepted_vwap_thesis_requires_closed_candle_reset(monkeypatch):
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    strategy = VWAPProStrategy(VWAPProStrategyConfig(), _DummyEngine())
    symbol = 'NFO:NIFTY26FEB22500CE'

    assert strategy._evaluate_signal(symbol, _indicators(), 103.0) is not None
    strategy.notify_entry_accepted('CE')

    assert strategy._evaluate_signal(symbol, _indicators(close=104.0), 104.0) is None
    assert strategy.last_no_vote_reason == 'vwap_thesis_not_reset'

    assert strategy._evaluate_signal(
        symbol,
        _indicators(close=99.5, open=100.0, high=100.2, low=99.2),
        99.5,
    ) is None
    assert strategy.last_no_vote_reason == 'vwap_thesis_reset'

    assert strategy._evaluate_signal(symbol, _indicators(), 103.0) is not None
