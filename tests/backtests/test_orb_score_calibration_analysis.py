from scripts.analyze_orb_score_calibration import analyze


def test_score_calibration_reports_gross_and_net_expectancy_by_bucket():
    trades = [
        {
            "exit_time": "2026-01-05T10:00:00+05:30",
            "entry_time": "2026-01-05T09:45:00+05:30",
            "entry_price": 100.0,
            "exit_price": 102.0,
            "quantity": 1,
            "duration_minutes": 15.0,
            "gross_pnl": 2.0,
            "fees": 0.5,
            "net_pnl": 1.5,
            "raw_setup_score": 7.0,
            "entry_branch": "retest",
            "score_reasons": ["underlying_volume_confirmation"],
        },
        {
            "exit_time": "2026-01-05T11:00:00+05:30",
            "entry_time": "2026-01-05T10:45:00+05:30",
            "entry_price": 100.0,
            "exit_price": 99.0,
            "quantity": 1,
            "duration_minutes": 15.0,
            "gross_pnl": -1.0,
            "fees": 0.5,
            "net_pnl": -1.5,
            "raw_setup_score": 6.0,
            "entry_branch": "momentum",
            "score_reasons": [],
        },
    ]
    result = analyze(
        [
            {
                "slippage_bps_per_side": 10.0,
                "trades": trades,
            }
        ]
    )
    ten = result["10.0"]
    assert ten["overall"]["gross_pnl"] == 1.0
    assert ten["overall"]["net_pnl"] == 0.0
    assert ten["score_buckets"]["7.0-7.9"]["expectancy"] == 1.5
    assert ten["entry_branches"]["retest"]["gross_expectancy"] == 2.0
