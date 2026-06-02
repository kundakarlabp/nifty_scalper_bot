from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


@pytest.mark.asyncio
async def test_deferred_basket_retry_without_ctx_data_ready_logs_data_not_ready(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression: slotted/legacy contexts without data_ready must not crash retries."""

    async def _spot_ready(*args, **kwargs):
        return 23701.0

    async def _build_basket(*args, **kwargs):
        return {
            "selected_ce": "NFO:NIFTY26JUN23700CE",
            "selected_pe": "NFO:NIFTY26JUN23700PE",
            "option_symbols": ["NFO:NIFTY26JUN23700CE", "NFO:NIFTY26JUN23700PE"],
            "symbols": ["NSE:NIFTY", "NFO:NIFTY26JUN23700CE", "NFO:NIFTY26JUN23700PE"],
            "atm_strike": 23700,
        }

    async def _noop_async(*args, **kwargs):
        return None

    monkeypatch.setattr(app, "_wait_for_live_spot_or_raise", _spot_ready)
    monkeypatch.setattr(app, "_build_and_hydrate_live_basket_from_spot", _build_basket)
    monkeypatch.setattr(app, "_ensure_strategy_runner_started", _noop_async)
    monkeypatch.setattr(app, "_recompute_and_push_runtime_readiness", _noop_async)

    ctx = SimpleNamespace(
        trading_ready=False,
        live_orders_armed=False,
        selected_ce="NFO:NIFTY26JUN23700CE",
    )

    caplog.set_level("INFO", logger=app.LOGGER.name)

    await app._deferred_basket_hydration_retry(
        ctx,
        configured_mode="LIVE",
        max_attempts=1,
        delay_seconds=0,
    )

    assert "reason=data_not_ready" in caplog.text
    assert "AttributeError" not in caplog.text
    assert not hasattr(ctx, "data_ready")
