from __future__ import annotations

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    OrderFlowStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies import order_flow


SYMBOL = "NFO:NIFTY26SEP25000CE"


def _strategy() -> order_flow.OrderFlowStrategy:
    return order_flow.OrderFlowStrategy(
        OrderFlowStrategyConfig(enabled=True, quantity=1),
        indicator_engine=None,
    )


def _depth(
    buy: float,
    sell: float,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    return ([{"quantity": buy}], [{"quantity": sell}])


def _prime_ofi(
    strategy: order_flow.OrderFlowStrategy,
    monkeypatch,
    quantities: tuple[tuple[float, float], ...],
) -> dict[str, object]:
    clock = [1000.0]
    monkeypatch.setattr(order_flow.time, "monotonic", lambda: clock[0])
    snapshot: dict[str, object] = {}
    for version, (buy, sell) in enumerate(quantities, start=1):
        bids, asks = _depth(buy, sell)
        snapshot = strategy._temporal_ofi_snapshot(
            symbol=SYMBOL,
            bid=100.0,
            ask=100.5,
            bids=bids,
            asks=asks,
            update_version=version,
        )
        clock[0] += 0.2
    return snapshot


def test_temporal_ofi_aggregates_distinct_quote_updates(monkeypatch) -> None:
    strategy = _strategy()

    snapshot = _prime_ofi(
        strategy,
        monkeypatch,
        ((100.0, 100.0), (140.0, 100.0), (160.0, 100.0)),
    )

    assert snapshot["ofi_ready"] is True
    assert snapshot["ofi_event"] == 20.0
    assert snapshot["ofi_1s"] == 60.0
    assert snapshot["ofi_update_count_1s"] == 2
    assert snapshot["ofi_1s_normalized"] > 0.0
    assert snapshot["ofi_source"] == "strategy_quote_updates"


def test_temporal_ofi_does_not_double_count_same_quote_version(monkeypatch) -> None:
    strategy = _strategy()
    _prime_ofi(
        strategy,
        monkeypatch,
        ((100.0, 100.0), (140.0, 100.0), (160.0, 100.0)),
    )
    bids, asks = _depth(160.0, 100.0)

    repeated = strategy._temporal_ofi_snapshot(
        symbol=SYMBOL,
        bid=100.0,
        ask=100.5,
        bids=bids,
        asks=asks,
        update_version=3,
    )

    assert repeated["ofi_1s"] == 60.0
    assert repeated["ofi_update_count_1s"] == 2


def test_temporal_ofi_alignment_replaces_tick_bonus(monkeypatch) -> None:
    strategy = _strategy()
    _prime_ofi(
        strategy,
        monkeypatch,
        ((100.0, 100.0), (140.0, 100.0), (160.0, 100.0)),
    )
    indicators = {
        "bid": 100.0,
        "ask": 100.5,
        "spread_pct": 0.50,
        "depth": {"buy": [{"quantity": 160}], "sell": [{"quantity": 100}]},
        "tick_direction": "UP",
        "direction_bias": "CE",
        "atr": 2.0,
        "data_age_seconds": 0.2,
        "quote_update_version": 3,
    }

    signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)

    assert signal is not None
    assert signal.metadata["flow_confirmation_source"] == "temporal_ofi"
    assert signal.metadata["ofi_supports_side"] is True
    assert signal.metadata["tick_score"] == 2.0
    assert "temporal_ofi_alignment" in signal.metadata["score_reasons"]
    assert "tick_direction_alignment" not in signal.metadata["score_reasons"]


def test_temporal_ofi_conflict_cannot_add_context_bonus(monkeypatch) -> None:
    strategy = _strategy()
    _prime_ofi(
        strategy,
        monkeypatch,
        ((300.0, 100.0), (260.0, 100.0), (220.0, 100.0)),
    )
    indicators = {
        "bid": 100.0,
        "ask": 100.5,
        "spread_pct": 0.50,
        "depth": {"buy": [{"quantity": 220}], "sell": [{"quantity": 100}]},
        "tick_direction": "UP",
        "direction_bias": "CE",
        "atr": 2.0,
        "data_age_seconds": 0.2,
        "quote_update_version": 3,
    }

    signal = strategy._evaluate_signal(SYMBOL, indicators, current_price=100.25)

    assert signal is not None
    assert signal.metadata["ofi_conflicts_side"] is True
    assert signal.metadata["ofi_supports_side"] is False
    assert signal.metadata["tick_score"] == 0.0
    assert signal.metadata["context_bonus_score"] == 0.0
    assert "temporal_ofi_conflict" in signal.metadata["score_reasons"]
