from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "research"
        / "analyze_completed_trades.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_completed_trades", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trade(
    trade_id: str,
    *,
    closed_at: float,
    strategy: str = "VWAPPro",
    gross_pnl: float = 100.0,
    estimated_costs: float = 10.0,
    net_pnl: float = 90.0,
    with_quality: bool = False,
    exit_reason: str = "HARD_SL_BREACH src=ltp sl=99.0",
) -> dict[str, object]:
    outcome: dict[str, object] = {
        "strategy_profile_version": "production-v1-test",
    }
    if with_quality:
        outcome["signal_quality"] = {
            "alpha_score": 8.2,
            "direction_score": 7.4,
            "strategy_score": 8.0,
        }
        outcome["final_score"] = 8.1
        outcome["context_confirmation_evidence"] = [{"strategy": "OrderFlow"}]
    return {
        "trade_id": trade_id,
        "strategy": strategy,
        "symbol": "NFO:NIFTYTEST",
        "side": "BUY",
        "gross_pnl": gross_pnl,
        "estimated_costs": estimated_costs,
        "net_pnl": net_pnl,
        "entry_filled_at": closed_at - 10.0,
        "closed_at": closed_at,
        "exit_reason": exit_reason,
        "build_sha": "abc123",
        "outcome": outcome,
    }


def test_report_blocks_parameter_changes_without_counterfactual_walk_forward() -> None:
    module = _load_module()
    trades = [
        _trade("TRD_1", closed_at=1.0, strategy="ORBPro", with_quality=True),
        _trade("TRD_2", closed_at=2.0, strategy="SMC", with_quality=True),
        _trade("TRD_3", closed_at=3.0, strategy="VWAPPro", with_quality=True),
    ]

    report = module.build_evidence_report(trades, block_size=3)

    assert report["gates"]["canonical_dataset"] == "PASS"
    assert report["gates"]["chronological_post_cost"] == "PASS"
    assert report["gates"]["component_attribution"] == "DESCRIPTIVE_READY"
    assert report["gates"]["counterfactual_walk_forward"] == "REQUIRED"
    assert report["gates"]["parameter_changes"] == "BLOCKED"


def test_missing_decision_evidence_blocks_component_attribution() -> None:
    module = _load_module()
    trades = [
        _trade("TRD_1", closed_at=1.0, strategy="SMC"),
        _trade("TRD_2", closed_at=2.0, strategy="VWAPPro"),
    ]

    report = module.build_evidence_report(trades, block_size=2)

    coverage = report["dataset"]["coverage"]
    assert coverage["signal_quality_coverage"] == 0.0
    assert coverage["target_strategy_counts"]["ORBPro"] == 0
    assert report["component_score_attribution"] == []
    assert (
        report["gates"]["component_attribution"]
        == "BLOCKED_INCOMPLETE_DECISION_EVIDENCE"
    )


def test_chronological_blocks_use_post_cost_net_pnl() -> None:
    module = _load_module()
    trades = [
        _trade(
            "TRD_1",
            closed_at=1.0,
            gross_pnl=100.0,
            estimated_costs=25.0,
            net_pnl=75.0,
        ),
        _trade(
            "TRD_2",
            closed_at=2.0,
            gross_pnl=-20.0,
            estimated_costs=5.0,
            net_pnl=-25.0,
        ),
        _trade(
            "TRD_3",
            closed_at=3.0,
            gross_pnl=10.0,
            estimated_costs=10.0,
            net_pnl=0.0,
        ),
    ]

    blocks = module.chronological_blocks(trades, block_size=2)

    assert blocks[0]["gross_pnl"] == 80.0
    assert blocks[0]["estimated_costs"] == 30.0
    assert blocks[0]["net_pnl"] == 50.0
    assert blocks[0]["expectancy"] == 25.0
    assert blocks[0]["complete_block"] is True
    assert blocks[1]["net_pnl"] == 0.0
    assert blocks[1]["complete_block"] is False


def test_known_stale_quote_exit_is_flagged_not_silently_excluded() -> None:
    module = _load_module()
    trades = [
        _trade(
            "TRD_1",
            closed_at=1.0,
            exit_reason="HARD_SL_BREACH src=ltp_stale_quote sl=95.0",
        ),
        _trade("TRD_2", closed_at=2.0),
    ]

    report = module.build_evidence_report(trades, block_size=2)
    quality = report["dataset"]["execution_data_quality"]

    assert quality["known_stale_quote_exit_trades"] == 1
    assert quality["known_stale_quote_exit_fraction"] == 0.5
    assert quality["known_stale_quote_exit_performance"]["trade_count"] == 1
    assert quality["not_explicitly_stale_exit_performance"]["trade_count"] == 1
    assert any(
        "ltp_stale_quote" in reason
        for reason in report["gates"]["parameter_change_reasons"]
    )


@pytest.mark.parametrize(
    ("trades", "message"),
    [
        (
            [
                _trade(
                    "TRD_1",
                    closed_at=1.0,
                    gross_pnl=100.0,
                    estimated_costs=10.0,
                    net_pnl=95.0,
                )
            ],
            "net pnl identity failed",
        ),
        (
            [
                _trade("TRD_2", closed_at=2.0),
                _trade("TRD_1", closed_at=1.0),
            ],
            "chronologically ordered",
        ),
        (
            [
                _trade("TRD_1", closed_at=1.0),
                _trade("TRD_1", closed_at=2.0),
            ],
            "duplicate canonical trade_id",
        ),
    ],
)
def test_canonical_validation_fails_closed(
    trades: list[dict[str, object]],
    message: str,
) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match=message):
        module.validate_canonical_completed_trades(trades)


def test_loader_selects_only_closed_ledger_complete_rows(tmp_path: Path) -> None:
    module = _load_module()
    db_path = tmp_path / "trades.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE trade_ledger (
                trade_id TEXT PRIMARY KEY,
                strategy TEXT,
                symbol TEXT,
                side TEXT,
                gross_pnl REAL,
                estimated_costs REAL,
                net_pnl REAL,
                entry_filled_at REAL,
                closed_at REAL,
                exit_reason TEXT,
                build_sha TEXT,
                outcome_json TEXT,
                state TEXT,
                ledger_complete INTEGER
            )
            """
        )
        rows = [
            (
                "TRD_2",
                "SMC",
                "NFO:NIFTYTEST",
                "BUY",
                50.0,
                10.0,
                40.0,
                19.0,
                20.0,
                "HARD_SL_BREACH src=ltp sl=90.0",
                "sha2",
                json.dumps({"strategy_profile_version": "v2"}),
                "CLOSED",
                1,
            ),
            (
                "TRD_1",
                "VWAPPro",
                "NFO:NIFTYTEST",
                "BUY",
                100.0,
                10.0,
                90.0,
                9.0,
                10.0,
                "HARD_SL_BREACH src=ltp sl=90.0",
                "sha1",
                json.dumps({"strategy_profile_version": "v1"}),
                "CLOSED",
                1,
            ),
            (
                "TRD_OPEN",
                "VWAPPro",
                "NFO:NIFTYTEST",
                "BUY",
                100.0,
                10.0,
                90.0,
                29.0,
                30.0,
                "OPEN",
                "sha3",
                "{}",
                "OPEN",
                0,
            ),
        ]
        connection.executemany(
            """
            INSERT INTO trade_ledger
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    trades = module.load_canonical_completed_trades(db_path)

    assert [trade["trade_id"] for trade in trades] == ["TRD_1", "TRD_2"]
    assert trades[0]["outcome"]["strategy_profile_version"] == "v1"
