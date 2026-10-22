"""Fail-closed counterfactual labels: not live trade execution tests."""

from nifty_scalper_bot.backtesting.executable_counterfactual import (
    evaluate_executable_counterfactual,
)

CE = "NFO:NIFTY26OCT22650CE"


def quotes(points, *, symbol=CE, size=65):
    return [
        {
            "symbol": symbol,
            "timestamp": float(ts),
            "bid": float(bid),
            "ask": float(bid) + 1,
            "bid_size": size,
            "ask_size": size,
            "captured_full_depth": True,
        }
        for ts, bid in points
    ]


def evaluate(points, **kw):
    return evaluate_executable_counterfactual(
        quotes(points),
        symbol=CE,
        decision_ts=100.0,
        stop_price=90.0,
        target_price=112.0,
        quantity=65,
        horizon_seconds=20.0,
        max_quote_gap_seconds=10.0,
        **kw,
    )


def test_target_first_not_later_stop():
    result = evaluate([(100, 99), (105, 113), (110, 80)])
    assert result.status == "MODELED"
    assert result.reason == "target" and result.exit_ts == 105
    assert result.gross_pnl == 845
    assert 0 < result.estimated_fees < result.gross_pnl


def test_stop_first_not_later_recovery():
    result = evaluate([(100, 99), (103, 89), (105, 115)])
    assert result.reason == "stop" and result.exit_ts == 103
    assert result.net_pnl < 0


def test_provenance_unverified_is_unresolved():
    rows = quotes([(100, 99), (105, 114)])
    rows[0]["captured_full_depth"] = False
    result = evaluate_executable_counterfactual(
        rows, symbol=CE, decision_ts=100, stop_price=90, target_price=112, quantity=65
    )
    assert result.status == "UNRESOLVED" and result.net_pnl is None
    assert result.reason == "quote_provenance_or_freshness_unverified"


def test_late_first_executable_quote_cannot_be_backfilled():
    result = evaluate([(105, 99), (110, 114)])
    assert result.status == "UNRESOLVED"
    assert result.reason == "next_executable_ask_too_late"


def test_quote_gap_cannot_infer_barrier_order():
    result = evaluate([(100, 99), (118, 113)])
    assert result.status == "UNRESOLVED"
    assert result.reason == "unobserved_quote_gap"


def test_incomplete_forward_horizon_is_unresolved():
    result = evaluate([(100, 99), (105, 105)])
    assert result.status == "UNRESOLVED"
    assert result.reason == "forward_horizon_incomplete"


def test_contract_rotation_is_never_cross_joined():
    result = evaluate_executable_counterfactual(
        quotes([(100, 99), (105, 113)], symbol="NFO:NIFTY26OCT22700CE"),
        symbol=CE,
        decision_ts=100,
        stop_price=90,
        target_price=112,
        quantity=65,
    )
    assert result.status == "UNRESOLVED"
    assert result.reason == "no_captured_contract_quotes"


def test_lack_of_top_of_book_size_is_not_a_fill():
    rows = quotes([(100, 99), (105, 113)], size=25)
    result = evaluate_executable_counterfactual(
        rows, symbol=CE, decision_ts=100, stop_price=90, target_price=112, quantity=65
    )
    assert result.status == "UNRESOLVED"
    assert result.reason == "invalid_quote_or_insufficient_top_size"


def test_adverse_slippage_sensitivity_monotonic():
    path = [(100, 99), (105, 113)]
    a = evaluate(path, adverse_slippage_bps=0)
    b = evaluate(path, adverse_slippage_bps=50)
    assert a.status == b.status == "MODELED"
    assert b.net_pnl < a.net_pnl and b.net_r < a.net_r


def test_same_timestamp_ambiguous_quote_never_picks_favorable_path():
    result = evaluate([(100, 99), (105, 89), (105, 113)])
    assert result.status == "UNRESOLVED"
    assert result.reason == "same_timestamp_quote_order_ambiguous"
