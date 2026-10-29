"""Read-only reporting evidence regressions."""

from scripts.reporting.generate_daily_report import _execution_summary, _loss_tail_summary


def test_unknown_exit_price_provenance_is_not_zero_slippage() -> None:
    measured = [
        {"execution_quality": {"exit_order_type": "SL", "exit_quote_bid": None}},
        {
            "execution_quality": {
                "exit_order_type": "LIMIT",
                "exit_quote_bid": 10.0,
                "exit_quote_ask": 10.5,
                "exit_slippage_bps": 5.0,
            }
        },
    ]
    result = _execution_summary(measured)
    assert result["exit_quote_complete_trades"] == 1
    assert result["exit_quote_missing_by_order_type"] == {"SL": 1}
    assert result["exit_slippage_unmeasured_trades"] == 1


def test_loss_tail_summary_is_descriptive() -> None:
    result = _loss_tail_summary(
        [
            {"net_pnl": 2.0, "exit_reason": "target"},
            {"net_pnl": -7.0, "exit_reason": "stop"},
        ]
    )
    assert result["worst_net_trade_rupees"] == -7.0
    assert result["worst_loss_share_pct"] == 100.0
    assert result["sample_evidence"] == "exploratory_small_sample"
