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
        "ofi_update_count_3s": 3,
        "ofi_3s_span_seconds": 2.0,
        "ofi_3s_complete": True,
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
    assert extra["ofi_update_count_1s"] == 2
    assert extra["ofi_unavailable_reason"] is None
    assert extra["depth_imbalance"] == signal.metadata["depth_imbalance"]
    assert extra["strong_depth_conflicts_side"] is False


def test_live_short_ofi_cannot_authorize_vwap_or_orb_entry_context(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = _strategy()
    indicators = _indicators(
        buy=300.0, sell=100.0, ofi_value=3.14, supports=True
    )
    indicators.pop("_expected_support")
    # Two book events can flip the one-second imbalance instantly.
    indicators["ofi_update_count_3s"] = 2
    indicators["ofi_3s_span_seconds"] = 0.15

    signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)

    assert signal is not None
    assert signal.metadata["ofi_supports_side"] is True
    assert signal.metadata["ofi_sustained_confirmation"] is False
    assert signal.metadata["effective_context_alignment"] is False
    assert signal.metadata["context_alignment_source"] is None
    assert signal.metadata["effective_context_conflict"] is False


def test_live_opposing_three_second_flow_cannot_confirm_positive_one_second(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = _strategy()
    indicators = _indicators(
        buy=300.0, sell=100.0, ofi_value=0.50, supports=True
    )
    indicators.pop("_expected_support")
    indicators["ofi_3s_normalized"] = -0.40

    signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)

    assert signal is not None
    assert signal.metadata["ofi_sustained_confirmation"] is False
    assert signal.metadata["effective_context_alignment"] is False


def test_live_strong_depth_without_ofi_requires_distinct_temporal_snapshots(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    clock = [100.0]
    monkeypatch.setattr(order_flow_module.time, "monotonic", lambda: clock[0])
    strategy = _strategy()
    indicators = _indicators(
        buy=400.0, sell=100.0, ofi_value=0.0, supports=True
    )
    indicators.pop("_expected_support")
    indicators["ofi_ready"] = False

    first = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert first is not None
    assert first.metadata["context_alignment_candidate_source"] == "strong_depth"
    assert first.metadata["effective_context_alignment"] is False

    clock[0] = 101.2
    indicators["quote_update_version"] = 4
    later = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert later is not None
    assert later.metadata["effective_context_alignment"] is True


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
    assert (
        first.metadata["ofi_unavailable_reason"] == "upstream_temporal_evidence_unready"
    )

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


def test_recent_opposing_live_ofi_cannot_instantly_confirm_smc_context(
    monkeypatch,
) -> None:
    """A one-second favourable OFI flip cannot erase a recent adverse book."""
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ORDERFLOW_OFI_REVERSAL_HOLDOFF_SECONDS", "15")
    clock = [100.0]
    monkeypatch.setattr(order_flow_module.time, "monotonic", lambda: clock[0])
    strategy = _strategy()
    indicators = _indicators(buy=105.0, sell=100.0, ofi_value=-3.43, supports=False)
    indicators.pop("_expected_support")

    opposing = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert opposing is not None
    assert opposing.metadata["effective_context_conflict"] is True

    # Reproduce the observed Oct 12 transition: -3.43 at 14:15:17,
    # then +0.47 at 14:15:29 (only 12 seconds later).
    clock[0] = 112.0
    indicators["ofi_1s_normalized"] = 0.47
    indicators["quote_update_version"] = 4
    premature = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert premature is not None
    assert premature.metadata["ofi_supports_side"] is True
    assert premature.metadata["ofi_reversal_holdoff_active"] is True
    assert premature.metadata["effective_context_alignment"] is False

    clock[0] = 115.1
    indicators["quote_update_version"] = 5
    recovered = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)
    assert recovered is not None
    assert recovered.metadata["ofi_reversal_holdoff_active"] is False
    assert recovered.metadata["effective_context_alignment"] is True


def test_ofi_reversal_holdoff_does_not_modify_nonlive_confirmation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    strategy = _strategy()
    indicators = _indicators(buy=105.0, sell=100.0, ofi_value=-0.50, supports=False)
    indicators.pop("_expected_support")
    assert strategy._evaluate_signal(SYMBOL, indicators, 100.25) is not None
    indicators["ofi_1s_normalized"] = 0.50
    recovered = strategy._evaluate_signal(SYMBOL, indicators, 100.25)
    assert recovered is not None
    assert recovered.metadata["ofi_reversal_holdoff_active"] is False
    assert recovered.metadata["effective_context_alignment"] is True
