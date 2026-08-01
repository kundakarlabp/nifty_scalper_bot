"""Every strategy vote must carry a resolvable setup anchor (P0)."""

from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.execution.stop_rearm_contract_patch import _signal_setup_epoch
from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.signal_identity_patch import _deterministic_id

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


def test_anchor_is_stamped_from_indicator_context() -> None:
    signal = _elite_signal()
    EliteStrategy._stamp_setup_anchor(signal, {"latest_bar_ts": BAR_TS})

    assert signal.metadata["setup_candle_timestamp"] == BAR_TS
    assert signal.metadata["latest_bar_ts"] == BAR_TS


def test_stamped_signal_resolves_a_rearm_setup_epoch() -> None:
    signal = _elite_signal()
    EliteStrategy._stamp_setup_anchor(signal, {"latest_bar_ts": BAR_TS})

    assert _signal_setup_epoch(
        SimpleNamespace(symbol=signal.symbol, metadata=signal.metadata)
    ) == BAR_TS


def test_unstamped_signal_has_no_setup_epoch() -> None:
    """Documents the regression this stamp exists to prevent."""
    signal = _elite_signal()

    assert (
        _signal_setup_epoch(
            SimpleNamespace(symbol=signal.symbol, metadata=signal.metadata)
        )
        is None
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
