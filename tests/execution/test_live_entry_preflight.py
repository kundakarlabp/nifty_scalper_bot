from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core.live_entry_preflight_safety import build_context_live_entry_preflight
from nifty_scalper_bot.execution.live_entry_preflight import (
    SelectedOptionProof,
    evaluate_live_entry_preflight,
    quote_timestamp_source_acceptable,
)


NOW = datetime.now(timezone.utc)


def _proof(symbol: str = "NFO:TESTCE") -> SelectedOptionProof:
    return SelectedOptionProof(
        symbol=symbol,
        quote_present=True,
        quote_tradable=True,
        timestamp_quality="exchange",
        timestamp_source="exchange_timestamp",
        candle_count=5,
        last_candle_ts=NOW - timedelta(seconds=30),
        last_candle_close=100.0,
        max_candle_age_seconds=180.0,
        now=NOW,
    )


def test_live_entry_preflight_allows_only_when_all_truths_hold() -> None:
    decision = evaluate_live_entry_preflight(
        {
            "broker_positions_fetched": True,
            "broker_orders_reconciled": True,
            "local_positions_match_broker": True,
            "selected_options": [_proof("NFO:TESTCE"), _proof("NFO:TESTPE")],
        }
    )

    assert decision.ready is True
    assert decision.blockers == ()


def test_live_entry_preflight_blocks_missing_selected_side_proof() -> None:
    decision = evaluate_live_entry_preflight(
        {
            "broker_positions_fetched": True,
            "broker_orders_reconciled": True,
            "local_positions_match_broker": True,
            "context": {"selected_ce": "NFO:TESTCE", "selected_pe": "NFO:TESTPE"},
            "selected_options": [_proof("NFO:TESTCE")],
        }
    )

    assert decision.ready is False
    assert "selected_option_candle_unproven" in decision.blockers


@pytest.mark.parametrize(
    ("field", "blocker"),
    [
        ("broker_positions_fetched", "broker_positions_not_fetched"),
        ("broker_orders_reconciled", "broker_orders_not_reconciled"),
        ("local_positions_match_broker", "broker_position_mismatch"),
    ],
)
def test_live_entry_preflight_blocks_each_broker_truth(field: str, blocker: str) -> None:
    payload = {
        "broker_positions_fetched": True,
        "broker_orders_reconciled": True,
        "local_positions_match_broker": True,
        "selected_options": [_proof("NFO:TESTCE"), _proof("NFO:TESTPE")],
    }
    payload[field] = False

    decision = evaluate_live_entry_preflight(payload)

    assert decision.ready is False
    assert decision.primary_blocker == blocker
    assert blocker in decision.blockers


def test_live_entry_preflight_rejects_received_at_timestamp_source() -> None:
    bad = SelectedOptionProof(
        symbol="NFO:TESTCE",
        quote_present=True,
        quote_tradable=True,
        timestamp_quality="received_at",
        timestamp_source="received_at",
        candle_count=5,
        last_candle_ts=NOW - timedelta(seconds=30),
        last_candle_close=100.0,
        now=NOW,
    )

    decision = evaluate_live_entry_preflight(
        {
            "broker_positions_fetched": True,
            "broker_orders_reconciled": True,
            "local_positions_match_broker": True,
            "selected_options": [bad, _proof("NFO:TESTPE")],
        }
    )

    assert decision.ready is False
    assert "selected_option_timestamp_unusable" in decision.blockers
    assert quote_timestamp_source_acceptable("exchange_timestamp", "exchange") is True
    assert quote_timestamp_source_acceptable("received_at", "received_at") is False


def test_live_entry_preflight_requires_recent_valid_candle() -> None:
    stale = SelectedOptionProof(
        symbol="NFO:TESTCE",
        quote_present=True,
        quote_tradable=True,
        timestamp_quality="exchange",
        timestamp_source="exchange_timestamp",
        candle_count=5,
        last_candle_ts=NOW - timedelta(minutes=30),
        last_candle_close=100.0,
        max_candle_age_seconds=180.0,
        now=NOW,
    )

    decision = evaluate_live_entry_preflight(
        {
            "broker_positions_fetched": True,
            "broker_orders_reconciled": True,
            "local_positions_match_broker": True,
            "selected_options": [stale, _proof("NFO:TESTPE")],
        }
    )

    assert decision.ready is False
    assert "selected_option_candle_unproven" in decision.blockers


class _MDM:
    def __init__(self) -> None:
        self._snapshots = {
            "NFO:TESTCE": SimpleNamespace(
                ltp=100.0,
                bid=99.0,
                ask=101.0,
                tradable_quote=True,
                timestamp_quality="exchange",
                timestamp_source="exchange_timestamp",
            ),
            "NFO:TESTPE": SimpleNamespace(
                ltp=100.0,
                bid=99.0,
                ask=101.0,
                tradable_quote=True,
                timestamp_quality="exchange",
                timestamp_source="exchange_timestamp",
            ),
        }
        self._bars = {
            "NFO:TESTCE": [{"timestamp": NOW - timedelta(seconds=30), "close": 100.0}],
            "NFO:TESTPE": [{"timestamp": NOW - timedelta(seconds=30), "close": 100.0}],
        }

    def get_symbol_snapshot(self, symbol: str):
        return self._snapshots[symbol]

    def get_ohlc_bars(self, symbol: str):
        return self._bars[symbol]


def test_context_preflight_snapshot_maps_runtime_state() -> None:
    ctx = SimpleNamespace(
        selected_ce="NFO:TESTCE",
        selected_pe="NFO:TESTPE",
        market_data_manager=_MDM(),
        position_reconciliation_completed=True,
        position_reconciliation_failed=False,
        broker_orders_reconciled=True,
    )

    snapshot = build_context_live_entry_preflight(ctx)
    decision = evaluate_live_entry_preflight(snapshot)

    assert snapshot["broker_positions_fetched"] is True
    assert snapshot["broker_orders_reconciled"] is True
    assert snapshot["local_positions_match_broker"] is True
    assert decision.ready is True
