from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


class _MDM:
    def __init__(self, bars: int = 30) -> None:
        self.bars = bars

    def history_capacity_for(self, *_args, **_kwargs) -> int:
        return 1000

    async def ensure_history(self, _symbol: str, **kwargs):
        self.bars = max(self.bars, int(kwargs.get("target_bars", self.bars)))
        return SimpleNamespace(failure_reason=None)

    def get_ohlc_bars(self, *_args, **_kwargs):
        return [object()] * self.bars


class _SlowRunner:
    _option_required_bars = 30
    _context_required_bars = 20
    _required_candles = 30

    def __init__(self) -> None:
        self.sync_thread_id: int | None = None
        self.sync_completed = False

    def sync_history_from_mdm(self, _symbol: str, **kwargs):
        self.sync_thread_id = threading.get_ident()
        # Model the real Runner/Indicator reseed cost seen during ATM rotations.
        time.sleep(0.05)
        bars = int(kwargs["required_bars"])
        self.sync_completed = True
        return SimpleNamespace(
            runner_bars=bars,
            indicator_bars=bars,
            success=True,
            failure_reason=None,
        )


@pytest.mark.asyncio
async def test_runtime_history_runner_sync_is_off_asyncio_owner(monkeypatch) -> None:
    """A slow canonical Runner reseed must not execute on the event-loop thread."""
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "OPEN")
    runner = _SlowRunner()
    context = SimpleNamespace(
        market_data_manager=_MDM(),
        strategy_runner=runner,
        settings=SimpleNamespace(),
    )
    asyncio_owner_thread = threading.get_ident()

    result = await app.ensure_symbol_runtime_history(
        context,
        "NFO:NIFTY2690124100CE",
        role="selected_option",
        phase="dynamic_update",
        reason="dynamic_option_universe",
    )

    assert runner.sync_completed is True
    assert result.minimum_ready is True
    assert result.sync_success is True
    assert runner.sync_thread_id is not None
    assert runner.sync_thread_id != asyncio_owner_thread
