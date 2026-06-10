from __future__ import annotations

import logging
from types import SimpleNamespace

from nifty_scalper_bot.core import app
from nifty_scalper_bot.core.active_basket import ActiveContractSelection
from nifty_scalper_bot.core.app import BotContext


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
