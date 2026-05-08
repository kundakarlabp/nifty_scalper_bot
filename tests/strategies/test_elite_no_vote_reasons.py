from __future__ import annotations

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
