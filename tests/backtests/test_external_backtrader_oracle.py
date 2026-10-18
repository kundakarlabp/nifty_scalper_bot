"""Independent Backtrader oracle must agree on resolved long-trade gross P&L."""

import json
import sys

import pytest
from scripts.external_backtrader_oracle import backtrader_gross_pnl


def test_backtrader_oracle_replays_long_round_trip_gross_pnl():
    trades = [
        {
            "entry_price": 100.0,
            "exit_price": 110.0,
            "quantity": 75,
            "gross_pnl": 750.0,
        },
        {
            "entry_price": 120.0,
            "exit_price": 115.0,
            "quantity": 50,
            "gross_pnl": -250.0,
        },
    ]
    assert backtrader_gross_pnl(trades) == pytest.approx(500.0)


def test_backtrader_oracle_rejects_invalid_trade():
    with pytest.raises(ValueError, match="external_oracle_trade_invalid"):
        backtrader_gross_pnl([{"entry_price": 0, "exit_price": 10, "quantity": 1}])


def test_empty_oracle_ledger_is_not_reported_as_trade_validation(
    tmp_path, monkeypatch
):
    from scripts.external_backtrader_oracle import main

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "development_results.json").write_text(
        json.dumps(
            [{"candidate": "raw_reference", "slippage_bps_per_side": 10, "trades": []}]
        )
    )
    output = tmp_path / "oracle.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "external_backtrader_oracle.py",
            "--study-dir",
            str(study_dir),
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text())
    assert report["all_rows_match"] is True
    assert report["trade_count_verified"] == 0
    assert report["trade_evidence_status"] == "NO_TRADES_VALIDATED"

