"""Read-only trailing and order-flow shadow policies are causal."""

from nifty_scalper_bot.execution.trade_quality_shadow import (
    compare_profit_retention,
    evaluate_adverse_flow_shadow,
    replay_stop_retention,
)


def _quotes(*bids: float) -> list[dict]:
    return [
        {
            "ts": float(i + 1),
            "bid": bid,
            "source": "bid",
            "full_lot_bid_proven": True,
            "top_bid_qty": 65,
            "required_qty": 65,
            "quote_age_ms": 100.0,
            "market_age_ms": 100.0,
        }
        for i, bid in enumerate(bids)
    ]


def test_replay_evaluates_existing_stop_before_any_new_trailing() -> None:
    result = replay_stop_retention(
        _quotes(101, 103, 106, 111, 107, 102, 99),
        entry=100,
        initial_stop=95,
        quantity=65,
        round_trip_cost=56,
        min_samples=5,
    )
    assert result["ready"] is True
    assert result["status"] == "simulated_stop_breach"
    assert result["exit_bid"] <= result["stop_at_breach"]
    assert result["policy"] == "research_only"


def test_replay_does_not_treat_end_of_data_as_a_profitable_exit() -> None:
    result = replay_stop_retention(
        _quotes(101, 103, 106, 109, 112),
        entry=100,
        initial_stop=95,
        quantity=65,
        round_trip_cost=56,
    )
    assert result["status"] == "unresolved_at_last_sample"
    assert "net_r" not in result


def test_invalid_or_truncated_quote_history_is_not_promotable() -> None:
    samples = [{"ts": float(i), "bid": 110, "source": "ltp"} for i in range(10)]
    assert (
        replay_stop_retention(
            samples,
            entry=100,
            initial_stop=95,
            quantity=65,
            round_trip_cost=56,
        )["ready"]
        is False
    )
    assert (
        compare_profit_retention(
            _quotes(101, 102, 103, 104, 105),
            entry=100,
            initial_stop=95,
            quantity=65,
            round_trip_cost=56,
            samples_truncated=True,
        )["ready"]
        is False
    )


def test_adverse_flow_requires_sustained_fresh_samples() -> None:
    samples = [
        {"ts": 1.0, "ofi_3s_norm": -2.0, "fresh": True, "exit_bid": 52},
        {"ts": 2.0, "ofi_3s_norm": -2.3, "fresh": True, "exit_bid": 51},
        {"ts": 3.0, "ofi_3s_norm": -2.7, "fresh": True, "exit_bid": 50},
    ]
    result = evaluate_adverse_flow_shadow(samples, contract_side="CE")
    assert result["triggered"] is True
    assert result["first_confirmed_ts"] == 3
    assert (
        evaluate_adverse_flow_shadow(samples[:2], contract_side="CE")["triggered"]
        is False
    )
    samples[1]["fresh"] = False
    assert (
        evaluate_adverse_flow_shadow(samples, contract_side="CE")["triggered"] is False
    )


def test_adverse_flow_pe_reverses_sign_and_rejects_stale_context() -> None:
    samples = [{"ts": i, "ofi_3s_norm": 2, "fresh": True} for i in (1, 2, 3)]
    assert (
        evaluate_adverse_flow_shadow(samples, contract_side="PE")["triggered"] is True
    )
    assert (
        evaluate_adverse_flow_shadow(samples, contract_side="CE")["triggered"] is False
    )


def test_cached_best_bid_without_actual_full_lot_depth_abstains() -> None:
    samples = [
        {"ts": float(i + 1), "bid": 150.3, "source": "bid"}
        for i in range(8)
    ]
    result = compare_profit_retention(
        samples,
        entry=145.35,
        initial_stop=138.85,
        quantity=65,
        round_trip_cost=68.5,
        entry_at=1.0,
        exit_at=9.0,
        realized_net_pnl=-491.0,
    )
    assert result["ready"] is False
    assert result["reason"] == "full_lot_bid_path_unverified"
    assert result["quote_peak_retention"] is None
    assert result["policies"] == {}


def test_retention_proxy_requires_actual_fills_and_full_lot_quote_proof() -> None:
    samples = _quotes(101.0, 104.0, 110.0, 108.0, 106.0, 102.0)
    result = compare_profit_retention(
        samples,
        entry=100.0,
        initial_stop=95.0,
        quantity=65,
        round_trip_cost=65.0,
        entry_at=1.0,
        exit_at=7.0,
        realized_net_pnl=-130.0,
    )
    peak = result["quote_peak_retention"]
    assert peak is not None and peak["ready"] is True
    assert peak["peak_quote_epoch"] == 3.0
    assert peak["peak_bid"] == 110.0
    assert peak["peak_bid_mfe_r"] == 2.0
    assert peak["peak_net_using_actual_exit_costs_proxy"] == 585.0
    assert peak["peak_to_realized_cash_gap_proxy"] == 715.0
    assert peak["positive_quoted_profit_given_back_proxy"] == 715.0
    assert peak["fill_feasibility_not_proven"] is True
    # The same prices without lot proof must not generate fake cash giveback.
    samples[2]["top_bid_qty"] = 5
    other = compare_profit_retention(
        samples,
        entry=100.0,
        initial_stop=95.0,
        quantity=65,
        round_trip_cost=65.0,
        entry_at=1.0,
        exit_at=7.0,
        realized_net_pnl=-130.0,
    )
    assert other["quote_peak_retention"]["peak_bid"] == 108.0


def test_peak_gap_is_unknown_without_actual_net_or_with_truncated_path() -> None:
    samples = _quotes(101.0, 104.0, 110.0, 108.0, 106.0)
    missing = compare_profit_retention(
        samples, entry=100.0, initial_stop=95.0, quantity=65,
        round_trip_cost=65.0, entry_at=1.0, exit_at=6.0,
    )
    assert missing["quote_peak_retention"] is None
    truncated = compare_profit_retention(
        samples, entry=100.0, initial_stop=95.0, quantity=65,
        round_trip_cost=65.0, entry_at=1.0, exit_at=6.0,
        realized_net_pnl=300.0, samples_truncated=True,
    )
    assert truncated["ready"] is False
    assert truncated["quote_peak_retention"] is None
