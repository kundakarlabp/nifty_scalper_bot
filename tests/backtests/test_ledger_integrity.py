"""Regression checks for historical ledger evidence; no broker access."""

from __future__ import annotations

from nifty_scalper_bot.backtesting.ledger_integrity import (
    summarize_ledger_integrity,
)


def test_integrity_separates_historical_nonclosed_rows_from_live_positions() -> None:
    report = summarize_ledger_integrity(
        [
            {"trade_id": "old1", "state": "ENTRY_SUBMITTED"},
            {"trade_id": "old2", "state": "ENTRY_FILLED"},
            {"trade_id": "old3", "state": "REJECTED"},
            {"trade_id": "old4", "state": "CLOSED", "ledger_complete": False},
        ]
    )
    assert report["total_ledger_rows"] == 4
    assert report["nonclosed_historical_rows"] == 3
    assert report["closed_incomplete_rows"] == 1
    assert report["completed_closed_rows"] == 0
    assert report["live_open_positions_inferred"] is False
    assert report["broker_reconciled"] is False
    assert report["nonclosed_sample_ids"] == ["old1", "old2", "old3"]


def test_effective_broker_cost_is_not_confused_with_model_estimate() -> None:
    report = summarize_ledger_integrity(
        [
            {
                "trade_id": "new_trade",
                "state": "CLOSED",
                "ledger_complete": True,
                "build_sha": "a" * 40,
                "entry_order_id": "entry-1",
                "exit_order_id": "exit-1",
                "entry_filled_at": 100,
                "exit_filled_at": 115,
                "closed_at": 116,
                "entry_price": 172.6,
                "exit_price": 176,
                "gross_pnl": 221.0,
                "estimated_costs": 74.223184,
                "net_pnl": 147.11,
                "outcome": {
                    "cost_source": "broker_virtual_contract_note",
                    "effective_costs": {"total": 73.886614},
                },
            },
            {
                "trade_id": "legacy",
                "state": "CLOSED",
                "ledger_complete": True,
                "build_sha": None,
                "entry_order_id": "entry-old",
                "exit_order_id": "exit-old",
                "entry_filled_at": 10,
                "exit_filled_at": None,
                "closed_at": 20,
                "entry_price": 100,
                "exit_price": 98,
                "gross_pnl": -130,
                "estimated_costs": 60,
                "net_pnl": -190,
                "outcome": {},
            },
        ]
    )
    assert report["completed_closed_rows"] == 2
    assert report["completed_missing"]["valid_build_sha"] == 1
    assert report["completed_missing"]["exit_filled_at"] == 1
    assert report["completed_cost_source_counts"] == {
        "broker_virtual_contract_note": 1,
        "unattributed": 1,
    }
    assert report["completed_cost_arithmetic_mismatch"] == 0
    assert report["completed_cost_arithmetic_unverifiable"] == 0
    assert report["pnl_attributed_to_deployment"] is False


def test_missing_or_invalid_cost_data_is_explicitly_unverifiable() -> None:
    rows = [
        {
            "trade_id": "a",
            "state": "CLOSED",
            "ledger_complete": True,
            "gross_pnl": 100,
            "net_pnl": 80,
            "estimated_costs": None,
            "outcome": {},
        },
        {
            "trade_id": "b",
            "state": "CLOSED",
            "ledger_complete": True,
            "gross_pnl": 100,
            "net_pnl": 70,
            "estimated_costs": 20,
            "outcome": {},
        },
    ]
    report = summarize_ledger_integrity(rows)
    assert report["completed_cost_arithmetic_unverifiable"] == 1
    assert report["completed_cost_arithmetic_mismatch"] == 1


def test_empty_historical_ledger_is_an_explicit_unverified_result() -> None:
    report = summarize_ledger_integrity([])
    assert report["total_ledger_rows"] == 0
    assert report["completed_closed_rows"] == 0
    assert report["completed_cost_source_counts"] == {}
    assert report["broker_reconciled"] is False
