from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.history_hydration_orchestrator import (
    build_active_basket_hydration_plan,
    execute_hydration_plan,
)
from nifty_scalper_bot.data.market_data_manager import HydrationResult


class MDM:
    def __init__(self):
        self.calls = []

    async def ensure_history(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        return HydrationResult(
            symbol=symbol,
            interval="minute",
            role=kwargs.get("role"),
            phase=kwargs.get("phase"),
            reason=kwargs.get("reason"),
            required_bars=kwargs.get("required_bars"),
            target_bars=kwargs.get("target_bars"),
            cached_before=0,
            cached_after=kwargs.get("required_bars"),
            broker_fetch_started=True,
            joined_inflight=False,
            broker_fetch_observed=True,
            fetched_rows=kwargs.get("target_bars"),
            accepted_rows=kwargs.get("required_bars"),
            minimum_ready=True,
            target_ready=True,
            failure_reason=None,
        )


class Runner:
    def __init__(self):
        self.syncs = []

    def sync_history_from_mdm(self, symbol, **kwargs):
        self.syncs.append((symbol, kwargs))
        return SimpleNamespace(success=True)


@pytest.mark.asyncio
async def test_plan_deduplicates_and_executes_one_ensure_per_symbol() -> None:
    mdm = MDM()
    runner = Runner()
    ctx = SimpleNamespace(
        market_data_manager=mdm,
        strategy_runner=runner,
        active_contract_basket={
            "version": "v1",
            "spot_symbol": "nse:nifty",
            "selected_ce": "nfo:abcce",
            "selected_pe": "nfo:abcpe",
            "option_symbols": ["NFO:ABCCE", "NFO:ABCPE", "NFO:CTXCE"],
        },
    )
    plan = build_active_basket_hydration_plan(
        ctx, required_option_bars=5, required_context_bars=20
    )
    assert [r.symbol for r in plan.requirements].count("NFO:ABCCE") == 1
    result = await execute_hydration_plan(ctx, plan, phase="startup", reason="test")
    assert len(mdm.calls) == len(plan.requirements)
    assert result.started_fetches == len(plan.requirements)
    assert {s for s, _ in runner.syncs} == {r.symbol for r in plan.requirements}
