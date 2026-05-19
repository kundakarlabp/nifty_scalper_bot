from __future__ import annotations

import logging
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.bb_squeeze import BBSqueezeStrategy
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    BBSqueezeStrategyConfig,
    CPRBreakoutStrategyConfig,
    ORBProStrategyConfig,
    OrderFlowStrategyConfig,
    RSIDivergenceStrategyConfig,
    SMCStrategyConfig,
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.cpr_breakout import CPRBreakoutStrategy
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy
from nifty_scalper_bot.strategies.elite_strategies.order_flow import OrderFlowStrategy
from nifty_scalper_bot.strategies.elite_strategies.rsi_divergence import RSIDivergenceStrategy
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy
from nifty_scalper_bot.strategies.elite_strategies.vwap_pro import VWAPProStrategy


class _DummyIndicatorEngine:
    def get_indicators(self, symbol: str, names: list[str] | None = None) -> dict[str, Any]:
        del symbol, names
        return {}


def test_no_vote_reasons_not_none_for_all_strategy_return_none_paths() -> None:
    indicator_engine = _DummyIndicatorEngine()
    strategies = [
        SMCStrategy(SMCStrategyConfig()),
        VWAPProStrategy(VWAPProStrategyConfig(), indicator_engine),
        CPRBreakoutStrategy(CPRBreakoutStrategyConfig(), indicator_engine),
        OrderFlowStrategy(OrderFlowStrategyConfig(), indicator_engine),
        BBSqueezeStrategy(BBSqueezeStrategyConfig(), indicator_engine),
        RSIDivergenceStrategy(RSIDivergenceStrategyConfig(), indicator_engine),
        ORBProStrategy(ORBProStrategyConfig(), indicator_engine),
    ]

    for strategy in strategies:
        signal = strategy.generate_signal(
            symbol='NFO:NIFTY26MAY24350CE',
            indicators={},
            current_price=0.0,
            position=None,
        )
        assert signal is None
        reason = getattr(strategy, 'last_no_vote_reason', None)
        assert reason not in (None, '', 'none')



def test_orderflow_missing_depth_no_vote_by_default(monkeypatch) -> None:
    monkeypatch.delenv('ORDERFLOW_ALLOW_LTP_TICK_FALLBACK', raising=False)
    strategy = OrderFlowStrategy(OrderFlowStrategyConfig(), _DummyIndicatorEngine())
    signal = strategy.generate_signal(
        symbol='NFO:NIFTY26MAY24350CE',
        indicators={'bid': 100.0, 'ask': 101.0, 'depth': {'buy': [], 'sell': []}, 'atr': 2.0},
        current_price=100.5,
        position=None,
    )
    assert signal is None
    assert getattr(strategy, 'last_no_vote_reason', None) == 'missing_depth'


def test_smc_live_structure_reject_logs_specific_reason(monkeypatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('SMC_REQUIRE_STRUCTURE_CONFIRMATION_LIVE', 'true')
    strategy = SMCStrategy(SMCStrategyConfig())
    signal = strategy.generate_signal('NFO:NIFTY26MAY24350CE', {'high':10,'low':9,'close':9.5,'open':9.7,'atr':1,'liquidity_sweep_confirmed':True,'premium_reclaim':False,'bos_confirmed':False,'choch_confirmed':False,'retest_confirmed':False}, 9.5, None)
    assert signal is None
    assert getattr(strategy, 'last_no_vote_reason', None) == 'smc_structure_required_live'


def test_smc_premium_not_reversing_up_does_not_raise_name_error(monkeypatch, caplog) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('SMC_REQUIRE_STRUCTURE_CONFIRMATION_LIVE', 'true')
    strategy = SMCStrategy(SMCStrategyConfig())
    caplog.set_level(logging.INFO)
    signal = strategy.generate_signal(
        symbol='NFO:NIFTY26MAY23750CE',
        indicators={
            'high': 102,
            'low': 98,
            'open': 101,
            'close': 99,
            'atr': 2,
            'direction_bias': 'CE',
            'underlying_direction_bias': 'CE',
            'context_age_seconds': 10,
            'liquidity_sweep_confirmed': False,
            'liquidity_sweep_confirmed_bear': True,
            'premium_reclaim': False,
            'bullish_reversal': False,
            'choch_confirmed': False,
            'bos_confirmed': False,
            'data_age_seconds': 1,
        },
        current_price=100,
        position=None,
    )
    assert signal is None
    assert strategy.last_no_vote_reason in {'premium_not_reversing_up', 'smc_structure_required_live', 'smc_quality_gate_failed'}
    assert any('STRATEGY_NO_VOTE strategy=SMC' in rec.message for rec in caplog.records)


def test_smc_uses_underlying_direction_when_direction_bias_missing(monkeypatch) -> None:
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')
    monkeypatch.setenv('SMC_REQUIRE_STRUCTURE_CONFIRMATION_LIVE', 'false')
    strategy = SMCStrategy(SMCStrategyConfig())
    signal = strategy.generate_signal(
        symbol='NFO:NIFTY26MAY23750CE',
        indicators={
            'high': 102,
            'low': 98,
            'open': 99,
            'close': 101,
            'atr': 2,
            'underlying_direction_bias': 'CE',
            'context_age_seconds': 10,
            'liquidity_sweep_confirmed': True,
            'premium_reclaim': True,
            'choch_confirmed': True,
            'bos_confirmed': True,
            'data_age_seconds': 1,
        },
        current_price=100,
        position=None,
    )
    assert signal is not None
    assert signal.metadata['underlying_direction_bias'] == 'CE'
