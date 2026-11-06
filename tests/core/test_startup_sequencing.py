"""Regression tests for startup-sequencing fixes (basket commit classification,
token pending vs missing)."""

from __future__ import annotations

import logging
from pathlib import Path

from nifty_scalper_bot.core.app import classify_basket_commit, resolve_active_basket_tokens


def test_partial_basket_is_context_only_never_tradable_commit() -> None:
    event, context_only = classify_basket_commit(
        selected_ce=None, selected_pe=None, ce_token=None, pe_token=None, option_count=0
    )
    assert event == "CONTEXT_ONLY_BASKET_COMMITTED" and context_only is True

    event, context_only = classify_basket_commit(
        selected_ce="NFO:NIFTY2661623300CE", selected_pe=None,
        ce_token=111, pe_token=None, option_count=1,
    )
    assert event == "CONTEXT_ONLY_BASKET_COMMITTED" and context_only is True

    event, context_only = classify_basket_commit(
        selected_ce="NFO:NIFTY2661623300CE", selected_pe="NFO:NIFTY2661623300PE",
        ce_token=111, pe_token=None, option_count=2,
    )
    assert event == "CONTEXT_ONLY_BASKET_COMMITTED" and context_only is True


def test_complete_basket_commits_as_tradable() -> None:
    event, context_only = classify_basket_commit(
        selected_ce="NFO:NIFTY2661623300CE", selected_pe="NFO:NIFTY2661623300PE",
        ce_token=111, pe_token=222, option_count=8,
    )
    assert event == "ACTIVE_CONTRACT_BASKET_COMMITTED" and context_only is False


class _Ctx:
    def __init__(self, im: object | None) -> None:
        self.instrument_manager = im
        self.broker_client = None


class _IMNotLoaded:
    def is_loaded(self) -> bool:
        return False

    def get_token(self, symbol: str) -> int:
        raise KeyError(symbol)


class _IMLoadedButMissing:
    def is_loaded(self) -> bool:
        return True

    def get_token(self, symbol: str) -> int:
        raise KeyError(symbol)


def test_token_missing_before_im_ready_logs_pending_not_error(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nifty_scalper_bot.core.app"):
        resolve_active_basket_tokens(_Ctx(_IMNotLoaded()), ["NFO:NIFTY26JUNFUT"], None, None)
    assert any("ACTIVE_BASKET_TOKEN_PENDING" in rec.message for rec in caplog.records)
    assert not any(
        "ACTIVE_BASKET_TOKEN_MISSING" in rec.message and rec.levelno >= logging.ERROR
        for rec in caplog.records
    )


def test_token_missing_after_im_ready_is_error(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="nifty_scalper_bot.core.app"):
        resolve_active_basket_tokens(_Ctx(_IMLoadedButMissing()), ["NFO:NIFTY26JUNFUT"], None, None)
    assert any(
        "ACTIVE_BASKET_TOKEN_MISSING" in rec.message and rec.levelno >= logging.ERROR
        for rec in caplog.records
    )


def test_message_bus_is_wired_before_websocket_start() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src/nifty_scalper_bot/core/app.py"
    ).read_text(encoding="utf-8")
    assert source.index("_wire_and_start_message_bus(ctx)") < source.index(
        "start_websocket()"
    )


def test_startup_logs_mandatory_token_integrity_failures() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "src/nifty_scalper_bot/core/app.py"
    ).read_text(encoding="utf-8")
    assert "LIVE_TOKEN_INTEGRITY_BLOCKED" in source
    assert "mandatory_symbols" in source
    assert "mandatory_tokens_missing" in source
