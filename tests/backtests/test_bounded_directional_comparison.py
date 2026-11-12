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


def test_zero_fill_summary_preserves_rejections_and_entry_funnel():
    rows = []
    for day, evaluations, reason_count in [
        ("2019-01-01", 12, 9),
        ("2019-01-02", 7, 5),
    ]:
        rows.append(
            {
                "study": "vwap",
                "phase": "validation",
                "candidate": "current_reference",
                "day": day,
                "slippage_bps": 25.0,
                "resolved": 0,
                "unresolved": 0,
                "net_pnl": 0.0,
                "fees": 0.0,
                "entry_funnel": {
                    "evaluations": evaluations,
                    "signals_emitted": 0,
                    "accepted_fills": 0,
                },
                "no_vote_reasons": {"distance_outside_band": reason_count},
            }
        )
    summary = pilot.summarize_pilot(rows)
    assert len(summary) == 1
    result = summary[0]
    assert result["session_count"] == 2
    assert result["entry_funnel"]["evaluations"] == 19
    assert result["no_vote_reasons"] == {"distance_outside_band": 14}
    assert result["zero_fill_status"] == "EVALUATIONS_WITHOUT_SIGNALS"
    assert result["promotion_eligible"] is False


def test_zero_fill_diagnostics_distinguish_no_evaluation_and_no_fill():
    base = {
        "study": "orb",
        "phase": "validation",
        "candidate": "cost_gated_reference",
        "day": "2019-01-01",
        "slippage_bps": 25.0,
        "resolved": 0,
        "unresolved": 0,
        "net_pnl": 0.0,
        "fees": 0.0,
    }
    assert pilot.summarize_pilot([base])[0]["zero_fill_status"] == (
        "NO_RECORDED_EVALUATIONS"
    )
    signalled = {
        **base,
        "entry_funnel": {"evaluations": 50, "signals_emitted": 1},
    }
    assert pilot.summarize_pilot([signalled])[0]["zero_fill_status"] == (
        "SIGNALS_WITHOUT_RESOLVED_FILL"
    )
    unresolved = {**signalled, "unresolved": 1}
    assert pilot.summarize_pilot([unresolved])[0]["zero_fill_status"] == (
        "UNRESOLVED_EXIT_PRESENT"
    )


@pytest.mark.parametrize(
    "diagnostics",
    [{"evaluations": -1}, {"evaluations": True}, {"evaluations": "5"}],
)
def test_pilot_rejects_invalid_diagnostic_counters(diagnostics):
    with pytest.raises(ValueError, match="diagnostics_invalid"):
        pilot.summarize_pilot(
            [
                {
                    "study": "vwap",
                    "phase": "validation",
                    "candidate": "current_reference",
                    "day": "2019-01-01",
                    "slippage_bps": 25.0,
                    "resolved": 0,
                    "unresolved": 0,
                    "net_pnl": 0.0,
                    "fees": 0.0,
                    "entry_funnel": diagnostics,
                }
            ]
        )
