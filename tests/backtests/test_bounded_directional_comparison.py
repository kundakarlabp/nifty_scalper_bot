"""Broker-free bounded directional pilot: cohort and fail-closed tests."""

import json

import pytest
from scripts import run_bounded_directional_comparison as pilot


def _manifest():
    return {
        "sessions": [
            {
                "day": f"2019-01-{i:02d}",
                "bars": {"EQ": 120, "FUT": 120, "CE": 120, "PE": 120},
            }
            for i in range(1, 9)
        ]
        + [{"day": "2019-01-09", "bars": {"EQ": 120, "FUT": 120, "CE": 0, "PE": 120}}]
    }


def test_calendar_preselection_does_not_use_trade_results():
    a = pilot.select_sessions(_manifest(), per_quarter=2)
    assert a == ["2019-01-03", "2019-01-07"]
    b = _manifest()
    for session in b["sessions"]:
        session["simulated_profit"] = 1000000 if session["day"] == "2019-01-01" else -2
    assert pilot.select_sessions(b, per_quarter=2) == a


def test_cohort_fails_closed_on_duplicate_date_and_invalid_cap():
    m = _manifest()
    m["sessions"].append(dict(m["sessions"][0]))
    with pytest.raises(ValueError, match="duplicate_session"):
        pilot.select_sessions(m)
    with pytest.raises(ValueError, match="per_quarter_invalid"):
        pilot.select_sessions(_manifest(), per_quarter=0)


def test_summary_zero_fill_does_not_claim_alpha():
    rows = [
        {
            "study": "same_event_reversal",
            "phase": "validation",
            "candidate": "SMC",
            "day": "2019-01-01",
            "slippage_bps": 25.0,
            "resolved": 0,
            "unresolved": 1,
            "net_pnl": 0.0,
            "fees": 0.0,
            "matched_option_event_count": 4,
        }
    ]
    summary = pilot.summarize_pilot(rows)
    assert summary[0]["mean_net_per_resolved_fill"] is None
    assert summary[0]["unresolved"] == 1
    assert summary[0]["matched_option_event_count"] == 4
    assert summary[0]["promotion_eligible"] is False
    assert summary[0]["attribution_valid"] is False


def test_shadow_only_and_all_sessions_unusable(tmp_path, monkeypatch):
    root = tmp_path / "research"
    root.mkdir()
    (root / "data_manifest.json").write_text(json.dumps(_manifest()))
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="requires_shadow"):
        pilot.run(root)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    with pytest.raises(ValueError, match="all_sessions_unusable"):
        pilot.run(root)


def test_result_compaction_keeps_actual_resolved_and_missing():
    scenario = {
        "slippage_bps_per_side": 10.0,
        "trades": [
            {
                "entry_time": "2019-01-01T09:40:00+05:30",
                "exit_time": "2019-01-01T09:45:00+05:30",
                "entry_price": 100.0,
                "exit_price": 101.0,
                "gross_pnl": 65,
                "net_pnl": 30.0,
                "fees": 35.0,
                "quantity": 65,
                "duration_minutes": 5,
            }
        ],
        "unresolved_trades": [{"unresolved_reason": "missing_exit"}],
    }
    compact = pilot._compact_candidates(
        {"candidates": [{"name": "baseline", "scenarios": [scenario]}]},
        "vwap",
        "validation",
        "2019-01-01",
    )
    assert compact[0]["resolved"] == 1
    assert compact[0]["unresolved"] == 1
    assert compact[0]["net_pnl"] == 30.0

    rev = pilot._compact_reversal(
        {
            "scenarios": [
                {
                    "slippage_bps_per_side": 10.0,
                    "events": [
                        {
                            "outcomes": {
                                "SMC": {
                                    "resolved_fill": False,
                                    "unresolved_exit": False,
                                    "net_pnl": None,
                                    "costs": None,
                                },
                                "FailedBreakoutResearch": {
                                    "resolved_fill": True,
                                    "unresolved_exit": False,
                                    "net_pnl": 55.0,
                                    "costs": 34.0,
                                },
                            }
                        }
                    ],
                    "both_resolved_same_entry_count": 0,
                    "reversal_only_positive_resolved_fill_count": 1,
                }
            ]
        },
        "validation",
        "2019-01-01",
    )
    smc, failed = rev
    assert smc["net_pnl"] == 0
    assert smc["resolved"] == 0
    assert failed["net_pnl"] == 55.0
    assert failed["resolved"] == 1
    assert failed["reversal_only_positive_count"] == 1
