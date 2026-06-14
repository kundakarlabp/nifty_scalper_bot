from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


@dataclass
class Hydration:
    symbol: str
    minimum_ready: bool = True
    target_ready: bool = True
    fetched_rows: int = 0
    accepted_rows: int = 0
    failure_reason: str | None = None


@dataclass
class Sync:
    runner_bars: int
    indicator_bars: int
    success: bool = True
    failure_reason: str | None = None


class MDM:
    def __init__(self, bars: int = 0):
        self.bars = bars; self.calls = []
    async def ensure_history(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        if kwargs.get("target_bars", 0) > self.bars:
            self.bars = kwargs["target_bars"]
        return Hydration(symbol, fetched_rows=self.bars, accepted_rows=self.bars)
    def get_ohlc_bars(self, *_a, **_k):
        return [object()] * self.bars


class Runner:
    def __init__(self, bars: int = 0):
        self.bars = bars; self.calls = []
        self._context_required_bars = 20
        self._option_required_bars = 30
    def sync_history_from_mdm(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        self.bars = max(self.bars, kwargs["required_bars"])
        return Sync(self.bars, self.bars)


def ctx(mdm=None, runner=None):
    return SimpleNamespace(market_data_manager=mdm, strategy_runner=runner, settings=SimpleNamespace())


@pytest.mark.asyncio
async def test_spot_startup_policy_and_orchestration() -> None:
    mdm = MDM(); runner = Runner()
    result = await app.ensure_symbol_runtime_history(ctx(mdm, runner), "NSE:NIFTY", role="spot_context", phase="startup", reason="t")
    assert result.minimum_ready and result.sync_success
    assert mdm.calls and runner.calls


@pytest.mark.asyncio
async def test_selected_option_priority_and_target() -> None:
    mdm = MDM(); runner = Runner()
    result = await app.ensure_symbol_runtime_history(ctx(mdm, runner), "NFO:CE", role="selected_option", phase="startup", reason="t")
    assert result.required_bars >= 30
    assert result.target_bars >= result.required_bars


@pytest.mark.asyncio
async def test_context_option_closed_market_suppresses_broker(monkeypatch) -> None:
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "CLOSED")
    mdm = MDM(); runner = Runner()
    result = await app.ensure_symbol_runtime_history(ctx(mdm, runner), "NFO:CTXCE", role="option_context", phase="dynamic_update", reason="t")
    assert result.failure_reason == "broker_fetch_not_allowed"
    assert mdm.calls == []


@pytest.mark.asyncio
async def test_warm_runner_sync_only_no_broker_refetch(monkeypatch) -> None:
    mdm = MDM(bars=300); runner = Runner()
    result = await app.ensure_symbol_runtime_history(ctx(mdm, runner), "NFO:FUT", role="futures_context", phase="dynamic_update", reason="futures_context_refresh")
    assert result.minimum_ready
    assert len(mdm.calls) == 1  # ensure_history performs target-sufficient skip inside MDM


@pytest.mark.asyncio
async def test_missing_interfaces_return_controlled_failures() -> None:
    assert (await app.ensure_symbol_runtime_history(ctx(SimpleNamespace(), Runner()), "NSE:NIFTY", role="spot_context", phase="startup", reason="t")).failure_reason == "mdm_ensure_history_missing"
    assert (await app.ensure_symbol_runtime_history(ctx(MDM(30), SimpleNamespace()), "NSE:NIFTY", role="spot_context", phase="startup", reason="t")).failure_reason == "runner_sync_history_missing"
