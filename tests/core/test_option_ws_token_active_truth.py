"""A fresh, broker-accepted option tick proves a WS token is active."""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace

from nifty_scalper_bot.core.app import _emit_option_symbol_pipeline_status


class _Hub:
    _mdm_subscribed_symbols = {"NFO:X"}

    def _canonical_quote_symbol(self, symbol: str) -> str:
        return symbol

    def debug_subscription_status(self, symbol: str, token: int | None = None):
        return {
            "symbol": symbol,
            "canonical": symbol,
            "token": token,
            "symbol_callbacks": 1,
            "token_callbacks": 1,
            "quote_present": True,
            "token_quote_present": True,
            "aliases": [],
        }


def _context(*, confirmed: bool, age: float = 0.1):
    return SimpleNamespace(
        data_hub=_Hub(),
        strategy_runner=None,
        datahub_runner_subscriptions={"TOKEN:123"},
        market_data_manager=SimpleNamespace(
            _tracked_symbols={"NFO:X"},
            _subscribers={"NFO:X": {object()}},
            _active_subscribed_symbols={"NFO:X"},
            _last_tick_time={"NFO:X": time.time() - age},
            _confirmed_subscriptions={123} if confirmed else set(),
            _desired_tokens={123},
            _requested_tokens={123},
            _subscribed_tokens=set(),
            _active_tokens=set(),
            time_since_last_live_ws_tick=lambda _symbol: age if confirmed else None,
        ),
    )


def test_broker_ws_active_uses_confirmed_current_tick_not_empty_internal_sets(
    caplog,
) -> None:
    caplog.set_level(logging.INFO)
    _emit_option_symbol_pipeline_status(
        _context(confirmed=True),
        symbol="NFO:X",
        token=123,
        selected=True,
        hydrated_bars=50,
        runner_added=False,
        source="test",
        reason="test",
    )
    record = next(
        r
        for r in caplog.records
        if r.message.startswith("OPTION_SYMBOL_PIPELINE_STATUS")
    )
    assert record.broker_ws_token_active is True


def test_requested_token_alone_does_not_claim_confirmed_ws_active(caplog) -> None:
    caplog.set_level(logging.INFO)
    _emit_option_symbol_pipeline_status(
        _context(confirmed=False),
        symbol="NFO:X",
        token=123,
        selected=True,
        hydrated_bars=50,
        runner_added=False,
        source="test",
        reason="test",
    )
    record = next(
        r
        for r in caplog.records
        if r.message.startswith("OPTION_SYMBOL_PIPELINE_STATUS")
    )
    assert record.broker_ws_token_active is False
