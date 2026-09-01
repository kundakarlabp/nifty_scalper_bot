from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app, history_readiness


class _MDM:
    def __init__(self, *, spot: float = 24102.0, bars: int = 0) -> None:
        self.spot = spot
        self.bars = bars
        self.ensure_calls: list[str] = []

    def history_capacity_for(self, *_args, **_kwargs) -> int:
        return 1000

    async def ensure_history(self, symbol: str, **kwargs):
        self.ensure_calls.append(symbol)
        await asyncio.sleep(0)
        self.bars = max(self.bars, int(kwargs.get("target_bars", self.bars)))
        return SimpleNamespace(failure_reason=None)

    def get_ohlc_bars(self, *_args, **_kwargs):
        return [object()] * self.bars

    def get_symbol_snapshot(self, symbol: str):
        if symbol == "NSE:NIFTY":
            return SimpleNamespace(ltp=self.spot)
        return SimpleNamespace(ltp=1.0)


class _Runner:
    _option_required_bars = 30
    _context_required_bars = 20
    _required_candles = 30

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._symbol_history: dict[str, list[object]] = {}
        self._indicator_engine = SimpleNamespace(get_history=lambda _symbol: [])

    def sync_history_from_mdm(self, symbol: str, **kwargs):
        self.calls.append(symbol)
        bars = int(kwargs["required_bars"])
        self._symbol_history[symbol] = [object()] * bars
        return SimpleNamespace(
            runner_bars=bars,
            indicator_bars=bars,
            success=True,
            failure_reason=None,
        )


def _context(mdm: _MDM, runner: _Runner):
    return SimpleNamespace(
        market_data_manager=mdm,
        strategy_runner=runner,
        settings=SimpleNamespace(
            option_universe=SimpleNamespace(strike_step=50),
        ),
    )


def _force_open_market(monkeypatch) -> None:
    """Keep both runtime-history policy owners deterministic in CI."""
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "OPEN")
    monkeypatch.setattr(history_readiness, "get_runtime_market_mode", lambda: "OPEN")


@pytest.mark.asyncio
async def test_cold_far_context_hydration_is_not_on_atm_commit_path(monkeypatch) -> None:
    """A newly-added far strike must return before its non-gating history reseed."""
    _force_open_market(monkeypatch)
    mdm = _MDM(spot=24102.0, bars=0)
    runner = _Runner()
    context = _context(mdm, runner)
    symbol = "NFO:NIFTY2690124250CE"  # 150 points from 24100 ATM

    result = await app.ensure_symbol_runtime_history(
        context,
        symbol,
        role="option_context",
        phase="dynamic_update",
        reason="dynamic_option_universe",
    )

    # The dynamic basket can continue to selected-pair commit immediately.
    assert runner.calls == []
    assert result.minimum_ready is False
    assert result.failure_reason == "dynamic_context_hydration_deferred"

    # The canonical orchestrator still runs; no history work is discarded.
    await asyncio.sleep(0.05)
    assert mdm.ensure_calls == [symbol]
    assert runner.calls == [symbol]


@pytest.mark.asyncio
async def test_atm_context_candidate_stays_synchronous_fail_closed(monkeypatch) -> None:
    """A genuinely new ATM contract must be hydrated before the call returns."""
    _force_open_market(monkeypatch)
    mdm = _MDM(spot=24102.0, bars=0)
    runner = _Runner()
    context = _context(mdm, runner)
    symbol = "NFO:NIFTY2690124100CE"

    result = await app.ensure_symbol_runtime_history(
        context,
        symbol,
        role="option_context",
        phase="dynamic_update",
        reason="dynamic_option_universe",
    )

    assert mdm.ensure_calls == [symbol]
    assert runner.calls == [symbol]
    assert result.minimum_ready is True
    assert result.sync_success is True


@pytest.mark.asyncio
async def test_selected_option_role_is_never_deferred(monkeypatch) -> None:
    _force_open_market(monkeypatch)
    mdm = _MDM(spot=24102.0, bars=0)
    runner = _Runner()
    context = _context(mdm, runner)
    symbol = "NFO:NIFTY2690124250CE"

    result = await app.ensure_symbol_runtime_history(
        context,
        symbol,
        role="selected_option",
        phase="dynamic_update",
        reason="dynamic_option_universe",
    )

    assert mdm.ensure_calls == [symbol]
    assert runner.calls == [symbol]
    assert result.minimum_ready is True
    assert result.sync_success is True