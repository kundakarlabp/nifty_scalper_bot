from __future__ import annotations

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_mdm_readiness_snapshot_alias() -> None:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._last_readiness_state = {'hard_ready': True, 'spot_ready': True, 'missing_hard': []}
    assert mdm.readiness_snapshot() == mdm.readiness_state_snapshot()
