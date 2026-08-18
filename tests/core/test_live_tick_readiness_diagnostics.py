from __future__ import annotations

import inspect

from nifty_scalper_bot.core import app as app_module
from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_option_readiness_keys_are_described_as_live_tick_gaps() -> None:
    assert app_module._readiness_missing_diagnostics(
        ["options_ce", "options_pe", "other_blocker"]
    ) == [
        "ce_live_tick_readiness_insufficient",
        "pe_live_tick_readiness_insufficient",
        "other_blocker",
    ]


def test_operator_logs_do_not_describe_live_tick_gaps_as_missing_contracts() -> None:
    source = inspect.getsource(app_module)
    assert (
        '"DATA_PIPELINE_NOT_READY hard_ready=%s spot_ready=%s missing_live_tick=%s"'
        in source
    )
    assert '"startup_pipeline_incomplete missing_live_tick=%s"' in source
    assert (
        '"LIVE_TRADING_BLOCKED reason=startup_pipeline_incomplete missing_live_tick=%s"'
        in source
    )


def test_internal_readiness_keys_remain_unchanged_for_safety_gating() -> None:
    source = inspect.getsource(MarketDataManager._readiness_state)
    assert 'missing_hard.append("options_ce")' in source
    assert 'missing_hard.append("options_pe")' in source
