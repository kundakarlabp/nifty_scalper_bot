from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.config.settings import Settings
from nifty_scalper_bot.core import app
from nifty_scalper_bot.core.active_basket import ActiveContractSelection
from nifty_scalper_bot.core.app import AppConfig, BOT_CONTEXT_RUNTIME_FIELD_DEFAULTS, BotContext, RateLimiter


def _selection(ce: str = "NFO:NIFTY2661623350CE", pe: str = "NFO:NIFTY2661623350PE", version: str = "v1") -> ActiveContractSelection:
    return ActiveContractSelection(
        selected_ce=ce,
        selected_pe=pe,
        basket_version=version,
        selected_at="2026-06-10T04:25:00Z",
        source="unit_test",
    )


def test_botcontext_declares_active_selection_runtime_fields() -> None:
    names = {field.name for field in __import__("dataclasses").fields(BotContext)}
    assert "_active_selection_sync_log_key" in names
    assert "_active_selection_drift_log_key" in names
    assert "last_active_selection_synced_at" in names
    assert "last_active_selection_drift_at" in names


def test_sync_active_selection_fresh_legacy_context_does_not_crash() -> None:
    ctx = SimpleNamespace()

    app._sync_active_selection_from_basket(ctx, _selection())

    assert ctx.selected_ce == "NFO:NIFTY2661623350CE"
    assert ctx.selected_pe == "NFO:NIFTY2661623350PE"
    assert ctx._active_selection_sync_log_key is not None
    assert ctx.last_active_selection_synced_at is not None
    assert isinstance(ctx.hydration_status_by_symbol, dict)


def test_active_selection_synced_logs_only_on_state_change(caplog) -> None:
    ctx = SimpleNamespace()
    caplog.set_level(logging.INFO, logger="nifty_scalper_bot.core.app")

    app._sync_active_selection_from_basket(ctx, _selection(version="v1"))
    app._sync_active_selection_from_basket(ctx, _selection(version="v1"))
    app._sync_active_selection_from_basket(ctx, _selection(version="v2"))

    events = [r for r in caplog.records if getattr(r, "event", "") == "ACTIVE_SELECTION_SYNCED"]
    assert len(events) == 2


def test_active_selection_drift_logs_only_on_state_change(caplog) -> None:
    ctx = SimpleNamespace(selected_ce="NFO:OLDCE", selected_pe="NFO:OLDPE")
    caplog.set_level(logging.WARNING, logger="nifty_scalper_bot.core.app")

    app._sync_active_selection_from_basket(ctx, _selection(version="v1"))
    ctx.selected_ce = "NFO:OLDCE"
    ctx.selected_pe = "NFO:OLDPE"
    app._sync_active_selection_from_basket(ctx, _selection(version="v1"))
    ctx.selected_ce = "NFO:OLDERCE"
    ctx.selected_pe = "NFO:OLDERPE"
    app._sync_active_selection_from_basket(ctx, _selection(version="v1"))

    events = [r for r in caplog.records if getattr(r, "event", "") == "ACTIVE_SELECTION_DRIFT_CORRECTED"]
    assert len(events) == 2


def test_hydration_status_map_repairs_missing_runtime_fields(monkeypatch) -> None:
    ctx = SimpleNamespace(
        active_trading_universe={"spot_symbol": "NSE:NIFTY", "selected_ce": "NFO:NIFTY2661623350CE", "selected_pe": "NFO:NIFTY2661623350PE"},
        strategy_runner=None,
    )
    monkeypatch.setattr(app, "build_symbol_hydration_status", lambda _ctx, sym, role, required: SimpleNamespace(to_dict=lambda: {"symbol": sym, "role": role}))

    statuses = app._hydration_status_map(ctx, required_option_bars=3, required_context_bars=1)

    assert set(statuses) == {"NSE:NIFTY", "NFO:NIFTY2661623350CE", "NFO:NIFTY2661623350PE"}
    assert ctx.last_hydration_status_at is not None


def _minimal_bot_context() -> BotContext:
    return BotContext(
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


def test_botcontext_runtime_fields_declared_or_ensured() -> None:
    app_text = Path(app.__file__).read_text(encoding="utf-8")
    patterns = [
        r"ctx\.(_[A-Za-z0-9_]*log_key)",
        r"ctx\.(last_[A-Za-z0-9_]*_at)",
        r"getattr\(ctx, ['\"](_[A-Za-z0-9_]*log_key)['\"]",
        r"getattr\(ctx, ['\"](last_[A-Za-z0-9_]*_at)['\"]",
        r"setattr\(ctx, ['\"](_[A-Za-z0-9_]*log_key)['\"]",
        r"setattr\(ctx, ['\"](last_[A-Za-z0-9_]*_at)['\"]",
    ]
    extracted = set()
    for pattern in patterns:
        extracted.update(re.findall(pattern, app_text))

    declared = {field.name for field in dataclasses.fields(BotContext)}
    ensured = set(BOT_CONTEXT_RUNTIME_FIELD_DEFAULTS)
    missing = sorted(extracted - declared - ensured)

    assert not missing, (
        "BotContext runtime fields must be declared on BotContext or listed in "
        f"BOT_CONTEXT_RUNTIME_FIELD_DEFAULTS: {missing}"
    )


def test_botcontext_has_post_market_basket_skip_fields() -> None:
    ctx = _minimal_bot_context()

    assert hasattr(ctx, "_post_market_basket_skip_log_key")
    assert ctx._post_market_basket_skip_log_key is None
    assert hasattr(ctx, "last_post_market_basket_skip_log_at")
    assert ctx.last_post_market_basket_skip_log_at is None


def test_ensure_bot_context_runtime_fields_adds_post_market_key_to_legacy_context() -> None:
    ctx = SimpleNamespace()

    app.ensure_bot_context_runtime_fields(ctx)

    assert hasattr(ctx, "_post_market_basket_skip_log_key")
    assert ctx._post_market_basket_skip_log_key is None
    assert hasattr(ctx, "last_post_market_basket_skip_log_at")
    assert ctx.last_post_market_basket_skip_log_at is None


@pytest.mark.asyncio
async def test_deferred_basket_hydration_no_attribute_error_for_post_market_skip_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = SimpleNamespace(
        trading_ready=False,
        live_orders_armed=False,
        selected_ce=None,
        active_contract_basket={"selected_ce": "NFO:NIFTY2661623350CE"},
        active_trading_universe={"selected_ce": "NFO:NIFTY2661623350CE"},
        basket_build_last_completed_mono=app.time_module.monotonic(),
        basket_build_lock=None,
        basket_build_in_progress=False,
        basket_build_last_started_mono=0.0,
        basket_build_last_error=None,
    )

    async def _spot_ready(*args, **kwargs) -> float:
        return 23500.0

    monkeypatch.setattr(app, "post_market_quiet_mode_enabled", lambda: True)
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "POST_MARKET")
    monkeypatch.setattr(app, "post_market_basket_refresh_seconds", lambda: 3600.0)
    monkeypatch.setattr(app, "_wait_for_live_spot_or_raise", _spot_ready)

    await app._deferred_basket_hydration_retry(
        ctx,
        configured_mode="LIVE",
        max_attempts=1,
        delay_seconds=0,
    )

    assert hasattr(ctx, "_post_market_basket_skip_log_key")
    assert ctx._post_market_basket_skip_log_key is not None


def test_post_market_basket_skip_path_no_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = SimpleNamespace(
        active_contract_basket={"selected_ce": "NFO:NIFTY2661623350CE"},
        basket_build_last_completed_mono=app.time_module.monotonic(),
    )
    monkeypatch.setattr(app, "post_market_quiet_mode_enabled", lambda: True)
    monkeypatch.setattr(app, "get_runtime_market_mode", lambda: "POST_MARKET")
    monkeypatch.setattr(app, "post_market_basket_refresh_seconds", lambda: 3600.0)

    skip, remaining = app._should_skip_post_market_basket_refresh(ctx)

    assert skip is True
    assert remaining > 0
    assert ctx._post_market_basket_skip_log_key is not None
    assert ctx.last_post_market_basket_skip_log_at is not None
