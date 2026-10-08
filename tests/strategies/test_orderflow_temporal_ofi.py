from unittest.mock import Mock

import nifty_scalper_bot.strategies.elite_strategies.order_flow as order_flow_module
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
        "context_age_seconds": 0.2,
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
    assert signal.metadata["flow_supports_side"] is True
    assert "temporal_ofi_alignment" in signal.metadata["setup_reasons"]
    assert "tick_direction_alignment" not in signal.metadata["setup_reasons"]
    assert not hasattr(strategy, "_ofi_state")


def test_strong_temporal_ofi_can_confirm_with_neutral_depth() -> None:
    strategy = _strategy()
    indicators = _indicators(
        buy=105.0,
        sell=100.0,
        ofi_value=0.50,
        supports=True,
    )
    indicators.pop("_expected_support")

    signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)

    assert signal is not None
    assert signal.metadata["depth_supports_side"] is False
    assert signal.metadata["ofi_supports_side"] is True
    assert signal.metadata["effective_context_alignment"] is True
    assert signal.metadata["effective_context_conflict"] is False
    assert signal.metadata["context_alignment_source"] == "temporal_ofi"


def test_strong_adverse_depth_blocks_supportive_temporal_ofi() -> None:
    strategy = _strategy()
    indicators = _indicators(
        buy=100.0,
        sell=400.0,
        ofi_value=0.50,
        supports=True,
    )
    indicators.pop("_expected_support")

    signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)

    assert signal is not None
    assert signal.metadata["ofi_supports_side"] is True
    assert signal.metadata["strong_depth_conflicts_side"] is True
    assert signal.metadata["effective_context_alignment"] is False
    assert signal.metadata["effective_context_conflict"] is True


def test_orderflow_context_log_exposes_microstructure_diagnostics(
    monkeypatch,
) -> None:
    logger = Mock()
    monkeypatch.setattr(order_flow_module, "LOGGER", logger)
    strategy = _strategy()
    indicators = _indicators(
        buy=105.0,
        sell=100.0,
        ofi_value=0.50,
        supports=True,
    )
    indicators.pop("_expected_support")

    signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)

    assert signal is not None
    calls = [
        call
        for call in logger.info.call_args_list
        if call.kwargs.get("extra", {}).get("event") == "ORDERFLOW_CONTEXT_EVIDENCE"
    ]
    assert len(calls) == 1
    extra = calls[0].kwargs["extra"]
    assert extra["context_alignment_source"] == "temporal_ofi"
    assert extra["ofi_ready"] is True
    assert extra["ofi_1s_normalized"] == 0.50
    assert extra["depth_imbalance"] == signal.metadata["depth_imbalance"]
    assert extra["strong_depth_conflicts_side"] is False


def test_live_tick_only_depth_confirmation_needs_two_distinct_fresh_snapshots(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    clock = [100.0]
    monkeypatch.setattr(order_flow_module.time, "monotonic", lambda: clock[0])
    strategy = _strategy()
    indicators = _indicators(buy=250.0, sell=100.0, ofi_value=0.0, supports=True)
    indicators.pop("_expected_support")
    indicators["ofi_ready"] = False

    first = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert first is not None
    assert first.metadata["context_alignment_candidate_source"] == "depth_plus_flow"
    assert first.metadata["context_alignment_source"] is None
    assert first.metadata["effective_context_alignment"] is False

    clock[0] = 100.5
    repeat = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert repeat is not None
    assert repeat.metadata["effective_context_alignment"] is False

    clock[0] = 101.0
    indicators["quote_update_version"] = 4
    confirmed = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert confirmed is not None
    assert confirmed.metadata["effective_context_alignment"] is True

    clock[0] = 120.0
    indicators["quote_update_version"] = 5
    expired = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert expired is not None
    assert expired.metadata["effective_context_alignment"] is False


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
    assert signal.metadata["flow_supports_side"] is False
    assert signal.metadata["effective_context_alignment"] is False
    assert "temporal_ofi_alignment" not in signal.metadata["setup_reasons"]

def test_live_depth_persistence_survives_multiple_fast_quote_versions(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    clock = [100.0]
    monkeypatch.setattr(order_flow_module.time, "monotonic", lambda: clock[0])
    strategy = _strategy()
    indicators = _indicators(buy=250.0, sell=100.0, ofi_value=0.0, supports=True)
    indicators.pop("_expected_support")
    indicators["ofi_ready"] = False

    for timestamp, version in ((100.0, 3), (100.4, 4), (100.8, 5)):
        clock[0] = timestamp
        indicators["quote_update_version"] = version
        signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
        assert signal is not None
        assert signal.metadata["effective_context_alignment"] is False

    clock[0] = 101.2
    indicators["quote_update_version"] = 6
    confirmed = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert confirmed is not None
    assert confirmed.metadata["effective_context_alignment"] is True
