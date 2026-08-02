"""Every strategy vote must carry a resolvable setup anchor (P0)."""

from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.execution.stop_rearm_contract_patch import _signal_setup_epoch
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    EliteStrategyConfig,
)
from nifty_scalper_bot.strategies.signal_identity_patch import (
    _deterministic_id,
    has_setup_anchor,
)

BAR_TS = 1_785_000_000.0


def _elite_signal(symbol: str = "NFO:NIFTY2680424400CE") -> EliteSignal:
    return EliteSignal(
        symbol=symbol,
        signal="BUY",
        confidence=0.8,
        entry_price=100.0,
        stop_loss=90.0,
        target=120.0,
        strategy_name="VWAPPro",
        metadata={"strategy": "VWAPPro"},
    )


class _RepeatingStrategy(EliteStrategy):
    def __init__(self, role: str = "trigger") -> None:
        self._role = role
        super().__init__(
            config=EliteStrategyConfig(min_confidence=0.0),
            indicator_engine=SimpleNamespace(),
        )

    def get_required_indicators(self) -> list[str]:
        return []

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: dict[str, object],
        current_price: float,
        position: object | None = None,
    ) -> EliteSignal:
        del indicators, position
        signal = _elite_signal(symbol)
        signal.entry_price = current_price
        signal.metadata.update(
            {
                "role": self._role,
                "contract_side": "CE",
            }
        )
        return signal


def test_anchor_is_stamped_from_indicator_context() -> None:
    signal = _elite_signal()
    EliteStrategy._stamp_setup_anchor(signal, {"latest_bar_ts": BAR_TS})

    assert signal.metadata["setup_candle_timestamp"] == BAR_TS
    assert signal.metadata["latest_bar_ts"] == BAR_TS
    assert has_setup_anchor(signal.metadata)


def test_stamped_signal_resolves_a_rearm_setup_epoch() -> None:
    signal = _elite_signal()
    EliteStrategy._stamp_setup_anchor(signal, {"latest_bar_ts": BAR_TS})

    assert _signal_setup_epoch(
        SimpleNamespace(symbol=signal.symbol, metadata=signal.metadata)
    ) == BAR_TS


def test_unstamped_signal_has_no_setup_epoch() -> None:
    """Documents the regression this stamp exists to prevent."""
    signal = _elite_signal()

    assert not has_setup_anchor(signal.metadata)
    assert (
        _signal_setup_epoch(
            SimpleNamespace(symbol=signal.symbol, metadata=signal.metadata)
        )
        is None
    )


def test_anchorless_identity_is_stable_across_wall_clock_time(monkeypatch) -> None:
    """Missing metadata must not mint a fresh executable setup every minute."""
    signal = _elite_signal()
    probe = SimpleNamespace(symbol=signal.symbol, action="BUY", metadata=signal.metadata)

    first = _deterministic_id(probe)
    second = _deterministic_id(probe)

    assert first == second


def test_anchorless_identity_is_stable_across_strike_rotation() -> None:
    first = _elite_signal("NFO:NIFTY2680424400CE")
    second = _elite_signal("NFO:NIFTY2680424350CE")

    assert _deterministic_id(
        SimpleNamespace(symbol=first.symbol, action="BUY", metadata=first.metadata)
    ) == _deterministic_id(
        SimpleNamespace(symbol=second.symbol, action="BUY", metadata=second.metadata)
    )


def test_identity_is_stable_across_strike_rotation() -> None:
    first = _elite_signal("NFO:NIFTY2680424400CE")
    second = _elite_signal("NFO:NIFTY2680424350CE")
    for signal in (first, second):
        EliteStrategy._stamp_setup_anchor(signal, {"latest_bar_ts": BAR_TS})

    assert _deterministic_id(
        SimpleNamespace(symbol=first.symbol, action="BUY", metadata=first.metadata)
    ) == _deterministic_id(
        SimpleNamespace(symbol=second.symbol, action="BUY", metadata=second.metadata)
    )


def test_identity_changes_on_a_new_setup_candle() -> None:
    first = _elite_signal()
    second = _elite_signal()
    EliteStrategy._stamp_setup_anchor(first, {"latest_bar_ts": BAR_TS})
    EliteStrategy._stamp_setup_anchor(second, {"latest_bar_ts": BAR_TS + 60.0})

    assert _deterministic_id(
        SimpleNamespace(symbol=first.symbol, action="BUY", metadata=first.metadata)
    ) != _deterministic_id(
        SimpleNamespace(symbol=second.symbol, action="BUY", metadata=second.metadata)
    )


def test_existing_signal_anchor_is_not_overwritten() -> None:
    signal = _elite_signal()
    signal.metadata["latest_bar_ts"] = BAR_TS
    EliteStrategy._stamp_setup_anchor(signal, {"latest_bar_ts": BAR_TS + 300.0})

    assert signal.metadata["setup_candle_timestamp"] == BAR_TS


def test_trigger_retry_is_allowed_until_entry_is_accepted() -> None:
    strategy = _RepeatingStrategy(role="trigger")
    indicators = {"latest_bar_ts": BAR_TS}
    symbol = _elite_signal().symbol

    first = strategy.generate_signal(symbol, indicators, 100.0)
    retry_before_acceptance = strategy.generate_signal(symbol, indicators, 100.5)
    strategy.notify_entry_accepted("CE")
    repeated_after_acceptance = strategy.generate_signal(symbol, indicators, 101.0)
    next_bar = strategy.generate_signal(
        symbol,
        {"latest_bar_ts": BAR_TS + 60.0},
        101.5,
    )

    assert first is not None
    assert retry_before_acceptance is not None
    assert repeated_after_acceptance is None
    assert strategy.last_no_vote_reason == "setup_anchor_already_consumed"
    assert next_bar is not None


def test_context_vote_remains_tick_responsive_on_same_anchor() -> None:
    strategy = _RepeatingStrategy(role="context")
    indicators = {"latest_bar_ts": BAR_TS}

    assert strategy.generate_signal(_elite_signal().symbol, indicators, 100.0) is not None
    strategy.notify_entry_accepted("CE")
    assert strategy.generate_signal(_elite_signal().symbol, indicators, 100.5) is not None
