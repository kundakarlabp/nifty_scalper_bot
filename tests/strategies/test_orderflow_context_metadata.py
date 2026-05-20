from nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy


class _Dummy:
    pass


def _indicators() -> dict:
    return {
        'bid': 100.0,
        'ask': 101.0,
        'buy_qty': 100,
        'sell_qty': 90,
        'tick_direction': 'UP',
        'spread_pct': 1.0,
        'atr': 2.0,
        'direction_bias': 'CE',
        'depth': {'buy': [{'quantity': 500}], 'sell': [{'quantity': 100}]},
    }


def test_orderflow_context_metadata_contract_side(monkeypatch) -> None:
    monkeypatch.setenv('ORDERFLOW_ALLOW_TRIGGER_ROLE', 'false')
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(), _Dummy())
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500CE', _indicators(), 100.5)
    assert signal is not None
    metadata = signal.metadata
    assert metadata['role'] == 'context'
    assert metadata['contract_side'] in {'CE', 'PE'}
    assert metadata['trade_side'] == metadata['contract_side']
    assert metadata['can_trigger'] is False


def test_orderflow_trigger_role(monkeypatch) -> None:
    monkeypatch.setenv('ORDERFLOW_ALLOW_TRIGGER_ROLE', 'true')
    monkeypatch.setenv('ORDERFLOW_TRIGGER_MIN_SCORE', '5.0')
    monkeypatch.setenv('ORDERFLOW_TRIGGER_MAX_SPREAD_PCT', '12.0')
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(), _Dummy())
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500CE', _indicators(), 100.5)
    assert signal is not None
    assert signal.metadata['role'] == 'trigger'
    assert signal.metadata['can_trigger'] is True


def test_orderflow_missing_tick_direction_reports_reason(monkeypatch) -> None:
    monkeypatch.setenv('ORDERFLOW_ALLOW_TRIGGER_ROLE', 'true')
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(), _Dummy())
    inds = _indicators()
    inds['tick_direction'] = ''
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500CE', inds, 100.5)
    assert signal is not None
    assert signal.metadata['role'] == 'context'
    assert signal.metadata['trigger_block_reason'] == 'tick_direction_missing_or_neutral'


def test_orderflow_live_requires_direction_context(monkeypatch) -> None:
    monkeypatch.setenv('ORDERFLOW_ALLOW_TRIGGER_ROLE', 'true')
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(), _Dummy())
    inds = _indicators()
    inds['direction_bias'] = None
    signal = strategy._evaluate_signal('NFO:NIFTY26FEB22500CE', inds, 100.5)
    assert signal is not None
    assert signal.metadata['role'] == 'context'
    assert signal.metadata['trigger_block_reason'] == 'direction_context_missing_live'
