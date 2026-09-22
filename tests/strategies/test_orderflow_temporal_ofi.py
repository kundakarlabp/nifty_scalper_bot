from nifty_scalper_bot.strategies.elite_strategies.order_flow import (
    OrderFlowStrategy,
    OrderFlowStrategyConfig,
)

SYMBOL = "NFO:NIFTY26SEP25000CE"


def _strategy() -> OrderFlowStrategy:
    return OrderFlowStrategy(
        OrderFlowStrategyConfig(enabled=True, quantity=1),
        indicator_engine=None,
    )


def _indicators(
    *,
    buy: float,
    sell: float,
    ofi_value: float,
    supports: bool,
) -> dict[str, object]:
    return {
        "bid": 100.0,
        "ask": 100.5,
        "spread_pct": 0.50,
        "depth": {
            "buy": [{"quantity": buy}],
            "sell": [{"quantity": sell}],
        },
        "tick_direction": "UP",
        "direction_bias": "CE",
        "atr": 2.0,
        "data_age_seconds": 0.2,
        "quote_update_version": 3,
        "ofi_ready": True,
        "ofi_event": ofi_value,
        "ofi_1s": ofi_value,
        "ofi_3s": ofi_value,
        "ofi_1s_normalized": ofi_value,
        "ofi_3s_normalized": ofi_value,
        "ofi_update_count_1s": 2,
        "ofi_update_count_3s": 2,
        "ofi_source": "runner_datahub_tick_updates",
        "queue_imbalance_top": (buy - sell) / max(buy + sell, 1.0),
        "_expected_support": supports,
    }


def test_orderflow_consumes_upstream_ofi_instead_of_tick_bonus() -> None:
    strategy = _strategy()
    indicators = _indicators(
        buy=160.0,
        sell=100.0,
        ofi_value=0.50,
        supports=True,
    )
    indicators.pop("_expected_support")

    signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)

    assert signal is not None
    assert signal.metadata["flow_confirmation_source"] == "temporal_ofi"
    assert signal.metadata["ofi_supports_side"] is True
    assert signal.metadata["tick_score"] == 2.0
    assert "temporal_ofi_alignment" in signal.metadata["score_reasons"]
    assert "tick_direction_alignment" not in signal.metadata["score_reasons"]
    assert not hasattr(strategy, "_ofi_state")


def test_adverse_upstream_ofi_cannot_add_context_bonus() -> None:
    strategy = _strategy()
    indicators = _indicators(
        buy=220.0,
        sell=100.0,
        ofi_value=-0.50,
        supports=False,
    )
    indicators.pop("_expected_support")

    signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)

    assert signal is not None
    assert signal.metadata["ofi_conflicts_side"] is True
    assert signal.metadata["ofi_supports_side"] is False
    assert signal.metadata["tick_score"] == 0.0
    assert signal.metadata["context_bonus_score"] == 0.0
    assert "temporal_ofi_conflict" in signal.metadata["score_reasons"]
