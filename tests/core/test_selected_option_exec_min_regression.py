from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


@pytest.mark.asyncio
async def test_selected_option_low_request_uses_execution_min(monkeypatch):
    captured = {}
    class MDM:
        def history_capacity_for(self, *a, **k): return 100
        async def ensure_history(self, symbol, **kw):
            captured.update(kw)
            return SimpleNamespace(failure_reason=None)
        def get_ohlc_bars(self, symbol): return [{}] * 30
    class Runner:
        _option_required_bars = 30
        def sync_history_from_mdm(self, symbol, **kw):
            return SimpleNamespace(
                success=True, runner_bars=30, indicator_bars=30, failure_reason=None
            )
    ctx = SimpleNamespace(market_data_manager=MDM(), strategy_runner=Runner())
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "OPEN")
    result = await app.ensure_symbol_runtime_history(
        ctx,
        "NFO:NIFTY26JUN24000CE",
        role="selected_option",
        phase="startup",
        reason="test",
        required_bars=5,
    )
    assert captured["required_bars"] == 30
    assert result.required_bars == 30
    assert result.minimum_ready is True
