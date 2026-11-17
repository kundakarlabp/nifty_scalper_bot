"""Read-only trailing and order-flow shadow policies are causal."""

from nifty_scalper_bot.execution.trade_quality_shadow import (
    compare_profit_retention,
    evaluate_adverse_flow_shadow,
    replay_stop_retention,
)


def _quotes(*bids: float) -> list[dict]:
    return [
        {"ts": float(i + 1), "bid": bid, "source": "bid"} for i, bid in enumerate(bids)
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
