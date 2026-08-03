from __future__ import annotations

import logging
from types import SimpleNamespace

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategyConfig,
    ORBProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy
from nifty_scalper_bot.strategies.elite_strategies.order_flow_live_context_patch import (
    apply_orderflow_live_context_proof,
)


class _ObservableStrategy(EliteStrategy):
    def __init__(self) -> None:
        super().__init__(EliteStrategyConfig(min_confidence=0.0), SimpleNamespace())

    def get_required_indicators(self) -> list[str]:
        return []

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, object],
        current_price: float,
        position: object | None = None,
    ) -> EliteSignal:
        del position
        return EliteSignal(
            symbol=symbol,
            signal="BUY",
            confidence=0.8,
            entry_price=current_price,
            stop_loss=current_price - 5.0,
            target=current_price + 10.0,
            strategy_name="Observable",
            metadata={
                "strategy": "Observable",
                "strategy_name": "Observable",
                "role": "trigger",
                "contract_side": "CE",
                "raw_setup_score": 8.0,
                "setup_id": "observable:ce:1",
                "latest_bar_ts": indicators["latest_bar_ts"],
                "quote_update_version": 7,
            },
        )


def test_disabled_orb_has_explicit_no_vote_reason(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_ORB_STRATEGY", "false")
    strategy = ORBProStrategy(
        ORBProStrategyConfig(enabled=True, quantity=1), indicator_engine=None
    )

    signal = strategy.generate_signal(
        "NFO:NIFTY2680724500CE",
        {"history_resolved_count": 5, "latest_bar_ts": 1_785_000_000.0},
        100.0,
    )

    assert signal is None
    assert strategy.last_no_vote_reason == "strategy_disabled_by_env"


def test_elite_signal_log_contains_structural_identity(caplog) -> None:
    caplog.set_level(logging.INFO)
    strategy = _ObservableStrategy()

    signal = strategy.generate_signal(
        "NFO:NIFTY2680724500CE",
        {"latest_bar_ts": 1_785_000_000.0},
        100.0,
    )

    assert signal is not None
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "ELITE_SIGNAL_GENERATED"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.strategy == "Observable"
    assert record.symbol == "NFO:NIFTY2680724500CE"
    assert record.side == "CE"
    assert record.raw_setup_score == 8.0
    assert record.setup_id == "observable:ce:1"
    assert record.quote_update_version == 7


def _orderflow_signal(*, bid: float = 99.5) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "strategy": "OrderFlow",
            "role": "context",
            "bid": bid,
            "ask": 100.0,
            "depth_imbalance": 0.25,
            "tick_direction": "UP",
        }
    )


def test_orderflow_quote_fingerprint_is_stable_and_changes_with_quote(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    first = apply_orderflow_live_context_proof(_orderflow_signal(), {})
    repeated = apply_orderflow_live_context_proof(_orderflow_signal(), {})
    changed = apply_orderflow_live_context_proof(_orderflow_signal(bid=99.6), {})

    first_version = first.metadata["quote_update_version"]
    assert first.metadata["quote_update_version_source"] == "microstructure_fingerprint"
    assert first_version == repeated.metadata["quote_update_version"]
    assert first_version != changed.metadata["quote_update_version"]


def test_orderflow_prefers_existing_quote_version(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    signal = _orderflow_signal()

    result = apply_orderflow_live_context_proof(
        signal, {"quote_update_version": 42}
    )

    assert result.metadata["quote_update_version"] == 42
    assert result.metadata["quote_update_version_source"] == "quote_update_version"
