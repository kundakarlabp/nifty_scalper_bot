"""Read-only broker tradebook fill-verification regression tests."""

from __future__ import annotations

import csv

import pytest

from nifty_scalper_bot.backtesting.broker_tradebook_reconciliation import (
    read_broker_tradebook_csv,
    summarize_broker_tradebook_reconciliation,
)


def _ledger(**overrides):
    row = {
        "trade_id": "TRD-1",
        "state": "CLOSED",
        "ledger_complete": True,
        "entry_order_id": "entry-1",
        "exit_order_id": "exit-1",
        "quantity": 65,
        "entry_price": 100.0,
        "exit_price": 104.0,
        "side": "BUY",
        "symbol": "NFO:NIFTYCE",
    }
    row.update(overrides)
    return row


def _fill(tid, oid, side, quantity, price, symbol="NIFTYCE"):
    return {
        "trade_id": tid,
        "order_id": oid,
        "side": side,
        "quantity": quantity,
        "price": price,
        "symbol": symbol,
    }


def test_partial_entry_fills_aggregate_exact_quantity_and_vwap():
    broker = [
        _fill("broker1", "entry-1", "BUY", 40, 99.0),
        _fill("broker2", "entry-1", "BUY", 25, 101.6),
        _fill("broker3", "exit-1", "SELL", 65, 104.0),
    ]
    report = summarize_broker_tradebook_reconciliation([_ledger()], broker)
    assert report["fills_verified_trade_count"] == 1
    assert report["statuses"] == {"fills_matched": 1}
    assert report["actual_contract_note_charges_verified"] is False
    assert report["final_net_pnl_broker_reconciled"] is False


@pytest.mark.parametrize(
    ("broker", "expected"),
    [
        (
            [
                _fill("b1", "entry-1", "BUY", 65, 100.0),
                _fill("b2", "exit-1", "SELL", 65, 104.3),
            ],
            "entry:matched;exit:fill_price_mismatch",
        ),
        (
            [
                _fill("b1", "entry-1", "BUY", 65, 100.0),
                _fill("b2", "exit-1", "BUY", 65, 104.0),
            ],
            "entry:matched;exit:symbol_or_side_mismatch",
        ),
        (
            [
                _fill("b1", "entry-1", "BUY", 65, 100.0),
                _fill("b2", "exit-1", "SELL", 25, 104.0),
                _fill("b3", "exit-partial-2", "SELL", 40, 105.0),
            ],
            "entry:matched;exit:partial_order_fills",
        ),
        (
            [
                _fill("b1", "entry-1", "BUY", 65, 100.0),
            ],
            "entry:matched;exit:missing_order_fills",
        ),
    ],
)
def test_mismatched_and_split_exit_orders_are_never_declared_reconciled(
    broker, expected
):
    report = summarize_broker_tradebook_reconciliation([_ledger()], broker)
    assert report["fills_verified_trade_count"] == 0
    assert report["trades"][0]["fill_verification"] == expected
    assert report["live_positions_inferred"] is False


def test_broker_order_reconciliation_excludes_nonterminal_legacy_entries():
    report = summarize_broker_tradebook_reconciliation(
        [
            _ledger(),
            _ledger(trade_id="old", state="ENTRY_SUBMITTED"),
            _ledger(trade_id="incomplete", ledger_complete=False),
        ],
        [
            _fill("b1", "entry-1", "BUY", 65, 100),
            _fill("b2", "exit-1", "SELL", 65, 104),
        ],
    )
    assert report["completed_ledger_rows"] == 1
    assert report["fills_verified_trade_count"] == 1


def test_missing_order_identity_does_not_trigger_symbol_heuristic_matching():
    report = summarize_broker_tradebook_reconciliation(
        [_ledger(exit_order_id=None)],
        [
            _fill("b1", "entry-1", "BUY", 65, 100),
            _fill("b2", "other-sell", "SELL", 65, 104),
        ],
    )
    assert report["statuses"] == {"insufficient_ledger_identity": 1}
    assert report["fills_verified_trade_count"] == 0


def test_duplicate_broker_trade_id_fails_closed():
    with pytest.raises(ValueError, match="duplicate"):
        summarize_broker_tradebook_reconciliation(
            [_ledger()],
            [
                _fill("same", "entry-1", "BUY", 65, 100),
                _fill("same", "exit-1", "SELL", 65, 104),
            ],
        )


def test_csv_aliases_and_duplicate_broker_trade_ids_are_checked(tmp_path):
    path = tmp_path / "kite_trades.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trade_id",
                "order_id",
                "tradingsymbol",
                "transaction_type",
                "quantity",
                "average_price",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trade_id": "k1",
                "order_id": "entry-1",
                "tradingsymbol": "NIFTYCE",
                "transaction_type": "BUY",
                "quantity": 65,
                "average_price": 100.0,
            }
        )
    rows = read_broker_tradebook_csv(path)
    assert rows[0] == _fill("k1", "entry-1", "BUY", 65, 100.0)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("k1,entry-1,NIFTYCE,BUY,65,100\n")
    with pytest.raises(ValueError, match="duplicate_trade_id"):
        read_broker_tradebook_csv(path)


@pytest.mark.parametrize("quantity,price", [("0", "100"), ("65", "nan"), ("65.5", "100")])
def test_csv_invalid_numeric_inputs_fail_closed(tmp_path, quantity, price):
    path = tmp_path / "broken.csv"
    path.write_text(
        "trade_id,order_id,symbol,side,quantity,price\n"
        f"fill-1,entry-1,NIFTYCE,BUY,{quantity},{price}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid_row"):
        read_broker_tradebook_csv(path)
