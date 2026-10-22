from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

from nifty_scalper_bot.journal.trade_ledger import (
    ensure_trade_ledger_schema,
    materialize_trade_events,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "reporting"
    / "analyze_completed_trades.py"
)
spec = importlib.util.spec_from_file_location("analyze_completed_trades", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_build_analysis_emits_research_cohorts_without_fabricated_facts() -> None:
    report = module.build_analysis([], block_size=20, components=())
    assert report["realized_post_cost_evidence"] == {
        "by_strategy": [],
        "by_setup": [],
        "by_regime": [],
        "by_confirmation_type": [],
        "by_entry_hour_ist": [],
        "by_days_to_expiry": [],
        "by_target_adjustment": [],
        "by_exit_reason": [],
    }
    assert report["parameter_change"]["allowed"] is False


def test_build_analysis_emits_post_cost_attribution_groups() -> None:
    outcome = {
        "cost_source": "broker_virtual_contract_note",
        "effective_costs": {"total": 20.0},
        "strategy_key": "vwap_pro",
        "strategy_role": "trigger",
        "signal_family": "directional_trigger",
        "setup_name": "continuation_pullback",
        "regime": "TREND",
        "approval_path": "single_trigger_context_confirmed",
        "direction_contract": {"passed": True},
        "setup_contract": {"passed": True},
        "confirmation_contract": {"passed": True},
        "confirming_trigger_strategies": ["VWAPPro"],
        "context_confirmation_strategies": ["OrderFlow"],
    }
    row = {
        "trade_id": "t1",
        "closed_at": 1.0,
        "strategy": "VWAPPro",
        "gross_pnl": 100.0,
        "estimated_costs": 20.0,
        "net_pnl": 80.0,
        "ledger_complete": True,
        "state": "CLOSED",
        "exit_reason": "TARGET",
        "outcome": outcome,
    }

    report = module.build_analysis([row], block_size=20, components=("VWAPPro",))

    assert report["attribution"]["ready"] is True
    assert report["attribution"]["groups"] == [
        {
            "regime": "TREND",
            "setup_name": "continuation_pullback",
            "confirmation_type": "single_trigger_context_confirmed",
            "summary": {
                "trade_count": 1,
                "gross_pnl": 100.0,
                "estimated_costs": 20.0,
                "effective_costs": 20.0,
                "broker_cost_trade_count": 1,
                "net_pnl": 80.0,
                "expectancy": 80.0,
                "win_rate": 1.0,
                "average_win": 80.0,
                "average_loss": 0.0,
                "profit_factor": None,
                "max_drawdown": 0.0,
            },
        }
    ]


def test_build_analysis_emits_repeatable_realized_post_cost_evidence() -> None:
    outcome = {
        "cost_source": "broker_virtual_contract_note",
        "effective_costs": {"total": 20.0},
        "setup_name": "breakout_retest",
        "r_multiple": 0.8,
        "mfe_r": 1.2,
        "mae_r": 0.25,
    }
    row = {
        "trade_id": "e1",
        "closed_at": 1.0,
        "strategy": "ORBPro",
        "gross_pnl": 100.0,
        "estimated_costs": 20.0,
        "net_pnl": 80.0,
        "ledger_complete": True,
        "state": "CLOSED",
        "exit_reason": "TARGET",
        "outcome": outcome,
    }

    report = module.build_analysis([row], block_size=20, components=("ORBPro",))

    assert report["realized_post_cost_evidence"]["by_strategy"][0] == {
        "dimension": "strategy",
        "value": "ORBPro",
        "trade_count": 1,
        "net_expectancy": 80.0,
        "net_expectancy_ci_lower": 80.0,
        "net_expectancy_ci_upper": 80.0,
        "mean_r_multiple": 0.8,
        "mean_mfe_r": 1.2,
        "mean_mae_r": 0.25,
        "mean_mfe_capture_ratio": 0.666667,
    }
    assert (
        report["realized_post_cost_evidence"]["by_setup"][0]["value"]
        == "breakout_retest"
    )
    exit_reason = report["realized_post_cost_evidence"]["by_exit_reason"][0]
    assert exit_reason["value"] == "TARGET"


def test_cli_reports_all_historical_states_without_claiming_open_broker_orders(
    tmp_path, capsys
) -> None:
    database = tmp_path / "journal.db"
    with sqlite3.connect(database) as conn:
        ensure_trade_ledger_schema(conn)
        materialize_trade_events(
            conn,
            [
                {
                    "trade_id": "old-submission",
                    "event_name": "order.acknowledged",
                    "timestamp": 1.0,
                    "build_sha": "a" * 40,
                },
                {
                    "trade_id": "old-fill",
                    "event_name": "entry.filled",
                    "timestamp": 2.0,
                    "build_sha": "a" * 40,
                },
            ],
        )

    assert module.main(["--trades-db", str(database)]) == 0
    result = json.loads(capsys.readouterr().out)
    integrity = result["ledger_integrity"]
    assert integrity["total_ledger_rows"] == 2
    assert integrity["completed_closed_rows"] == 0
    assert integrity["nonclosed_historical_rows"] == 2
    assert integrity["state_counts"] == {"ENTRY_FILLED": 1, "ENTRY_SUBMITTED": 1}
    assert integrity["live_open_positions_inferred"] is False
    assert integrity["broker_reconciled"] is False
    assert result["overall_post_cost"]["trade_count"] == 0


def test_cli_broker_tradebook_audit_requires_explicit_csv(tmp_path, capsys) -> None:
    database = tmp_path / "journal.db"
    with sqlite3.connect(database) as conn:
        ensure_trade_ledger_schema(conn)
    assert module.main(["--trades-db", str(database)]) == 0
    absent = json.loads(capsys.readouterr().out)
    assert absent["broker_tradebook_reconciliation"]["ready"] is False

    csv_path = tmp_path / "broker_trades.csv"
    csv_path.write_text(
        "trade_id,order_id,tradingsymbol,transaction_type,quantity,average_price\n"
        "k1,broker-entry,NIFTYCE,BUY,65,100\n",
        encoding="utf-8",
    )
    assert module.main(
        ["--trades-db", str(database), "--broker-trades-csv", str(csv_path)]
    ) == 0
    present = json.loads(capsys.readouterr().out)
    evidence = present["broker_tradebook_reconciliation"]
    assert evidence["broker_tradebook_fill_rows"] == 1
    assert evidence["completed_ledger_rows"] == 0
    assert evidence["unreferenced_broker_order_count"] == 1
    assert evidence["final_net_pnl_broker_reconciled"] is False
