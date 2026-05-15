from __future__ import annotations

from nifty_scalper_bot.core.strategy_manager import StrategyManager, StrategyVote
from nifty_scalper_bot.strategies.elite_strategies.config_models import OrderFlowStrategyConfig, SMCStrategyConfig, VWAPProStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_orderflow_trigger_and_context_mode(monkeypatch) -> None:
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(enabled=True, quantity=1), indicator_engine=None)
    indicators = {
        'bid': 100.0, 'ask': 101.0, 'spread_pct': 0.9,
        'depth': {'buy': [{'quantity': 200}], 'sell': [{'quantity': 100}]},
        'tick_direction': 'UP', 'direction_bias': 'CE', 'atr': 2.0,
    }
    monkeypatch.setenv('ORDERFLOW_ALLOW_TRIGGER_ROLE', 'true')
    signal = strategy._evaluate_signal('NFO:NIFTY26MAY24000CE', indicators, 100.5)
    assert signal is not None and signal.metadata['role'] == 'trigger'
    monkeypatch.setenv('ORDERFLOW_ALLOW_TRIGGER_ROLE', 'false')
    signal = strategy._evaluate_signal('NFO:NIFTY26MAY24000CE', indicators, 100.5)
    assert signal is not None and signal.metadata['role'] == 'context'


def test_vwap_conflict_penalty_and_live_alignment(monkeypatch) -> None:
    strategy = VWAPProStrategy(VWAPProStrategyConfig(enabled=True, quantity=1), indicator_engine=None)
    indicators = {
        'vwap': 100.0, 'atr': 2.0, 'close': 102.0, 'open': 101.0, 'high': 103.0, 'low': 99.0,
        'volume': 1000.0, 'avg_volume': 1000.0, 'direction_bias': 'PE', 'futures_vwap_slope': 1.0,
    }
    monkeypatch.setenv('EXECUTION_MODE', 'SHADOW')
    sig = strategy._evaluate_signal('NFO:NIFTY26MAY24000CE', indicators, 101.0)
    assert sig is not None
    assert sig.metadata['conflict_penalty_applied'] >= 0.0
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('VWAP_PRO_REQUIRE_UNDERLYING_ALIGNMENT_LIVE', 'true')
    assert strategy._evaluate_signal('NFO:NIFTY26MAY24000CE', indicators, 101.0) is None


def test_smc_structure_live_gate(monkeypatch) -> None:
    strategy = SMCStrategy(SMCStrategyConfig(enabled=True, quantity=1), indicator_engine=None)
    indicators = {
        'high': 110.0, 'low': 90.0, 'close': 105.0, 'open': 100.0, 'atr': 5.0,
        'direction_bias': 'CE', 'liquidity_sweep_confirmed': True, 'premium_reclaim': True,
    }
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('SMC_REQUIRE_STRUCTURE_CONFIRMATION_LIVE', 'true')
    assert strategy._evaluate_signal('NFO:NIFTY26MAY24000CE', indicators, 105.0) is None
    indicators['choch_confirmed'] = True
    assert strategy._evaluate_signal('NFO:NIFTY26MAY24000CE', indicators, 105.0) is not None


def test_strategy_manager_trade_quality_gate(monkeypatch) -> None:
    manager = StrategyManager()
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_SCALP', 'true')
    monkeypatch.setenv('STRATEGY_ALLOW_SINGLE_VOTE_BY_MODE', 'true')
    monkeypatch.setenv('STRATEGY_MIN_TRADE_QUALITY_SHADOW', '7.0')
    signal = Signal(action='BUY', symbol='NFO:NIFTY26MAY24000CE', quantity=1, confidence=0.9, metadata={'strategy': 'VWAPPro', 'role': 'trigger', 'side': 'CE', 'strategy_score': 5.0, 'spread_pct': 30.0})
    vote = StrategyVote(strategy='VWAPPro', side='CE', score=5.0, confidence=0.9, reasons=['test'], metadata={'role': 'trigger', 'strategy_score': 5.0, 'spread_pct': 30.0})
    assert manager._combine_votes([(signal, vote)], [], {'stale_data_used': False}) is None
