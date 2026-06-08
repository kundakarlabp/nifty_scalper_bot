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

@pytest.mark.asyncio
async def test_deferred_retry_valid_hydrator_recomputes_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a valid build/hydration path calls recompute instead of crashing on ctx slots."""

    calls: list[str] = []

    async def _spot_ready(*args, **kwargs):
        return 23701.0

    async def _build_basket(*args, **kwargs):
        calls.append("build")
        return {
            "selected_ce": "NFO:NIFTY26JUN23700CE",
            "selected_pe": "NFO:NIFTY26JUN23700PE",
            "option_symbols": ["NFO:NIFTY26JUN23700CE", "NFO:NIFTY26JUN23700PE"],
            "symbols": ["NSE:NIFTY", "NFO:NIFTY26JUN23700CE", "NFO:NIFTY26JUN23700PE"],
            "atm_strike": 23700,
        }

    async def _start(*args, **kwargs):
        calls.append("start")

    async def _recompute(ctx, *args, **kwargs):
        calls.append("recompute")
        ctx.data_ready = True
        ctx.evaluation_ready = True

    monkeypatch.setattr(app, "_wait_for_live_spot_or_raise", _spot_ready)
    monkeypatch.setattr(app, "_build_and_hydrate_live_basket_from_spot", _build_basket)
    monkeypatch.setattr(app, "_ensure_strategy_runner_started", _start)
    monkeypatch.setattr(app, "_recompute_and_push_runtime_readiness", _recompute)

    ctx = SimpleNamespace(
        trading_ready=False,
        live_orders_armed=False,
        selected_ce="NFO:NIFTY26JUN23700CE",
    )

    await app._deferred_basket_hydration_retry(ctx, configured_mode="LIVE", max_attempts=1, delay_seconds=0)

    assert calls == ["build", "start", "recompute"]
    assert ctx.data_ready is True


def test_botcontext_hydration_status_fields_exist_after_construction() -> None:
    from nifty_scalper_bot.config.settings import Settings
    from nifty_scalper_bot.core.app import AppConfig, BotContext, RateLimiter

    ctx = BotContext(
        settings=Settings(),
        config=AppConfig(),
        rate_limiter=RateLimiter(),
        broker_client=None,
        websocket_client=None,
        websocket_manager=None,
        streamer=None,
        stream_supervisor=None,
        polling_fallback_streamer=None,
        message_bus=None,
    )

    assert ctx.hydration_status_by_symbol == {}
    assert ctx.last_hydration_status_at is None


def test_hydration_status_map_populates_botcontext_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nifty_scalper_bot.config.settings import Settings
    from nifty_scalper_bot.core.app import AppConfig, BotContext, RateLimiter
    from nifty_scalper_bot.execution.readiness import HydrationStatus

    def _ready_status(ctx, symbol: str, role: str, required: int) -> HydrationStatus:
        return HydrationStatus(
            symbol=symbol,
            role=role,
            required_bars=required,
            mdm_bars=required,
            datahub_bars=required,
            runner_bars=required,
            indicator_bars=required,
            tradable_quote=role in {"selected_ce", "selected_pe"},
            depth_available=role in {"selected_ce", "selected_pe"},
            ready_for_evaluation=True,
            ready_for_execution=True,
        )

    monkeypatch.setattr(app, "build_symbol_hydration_status", _ready_status)
    ctx = BotContext(
        settings=Settings(),
        config=AppConfig(),
        rate_limiter=RateLimiter(),
        broker_client=None,
        websocket_client=None,
        websocket_manager=None,
        streamer=None,
        stream_supervisor=None,
        polling_fallback_streamer=None,
        message_bus=None,
        active_trading_universe={
            "spot_symbol": "NSE:NIFTY",
            "selected_ce": "NFO:NIFTY26JUN23700CE",
            "selected_pe": "NFO:NIFTY26JUN23700PE",
            "option_symbols": ["NFO:NIFTY26JUN23700CE", "NFO:NIFTY26JUN23700PE"],
        },
    )

    statuses = app._hydration_status_map(
        ctx, required_option_bars=30, required_context_bars=20
    )

    assert set(statuses) == {
        "NSE:NIFTY",
        "NFO:NIFTY26JUN23700CE",
        "NFO:NIFTY26JUN23700PE",
    }
    assert (
        ctx.hydration_status_by_symbol["NFO:NIFTY26JUN23700CE"][
            "ready_for_execution"
        ]
        is True
    )
    assert ctx.last_hydration_status_at is not None
