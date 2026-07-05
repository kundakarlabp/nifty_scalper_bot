from __future__ import annotations

from dataclasses import dataclass

from nifty_scalper_bot.core import strategy_manager
from nifty_scalper_bot.strategies.context_builder import (
    build_strategy_history_context,
    classify_history_domain,
    collect_history_bars,
)


@dataclass
class _Bar:
    timestamp: str
    close: float = 100.0


class _Provider:
    def __init__(self, bars=None, *, raises: bool = False) -> None:
        self.bars = list(bars or [])
        self.raises = raises

    def get_ohlc_bars(self, _symbol: str):
        if self.raises:
            raise RuntimeError("provider unavailable")
        return list(self.bars)


class _HistoryOnlyProvider:
    def __init__(self, bars=None) -> None:
        self.bars = list(bars or [])

    def get_history(self, _symbol: str):
        return list(self.bars)


def test_classify_history_domain() -> None:
    assert classify_history_domain("NFO:NIFTY2662324050CE") == "options"
    assert classify_history_domain("NFO:NIFTY2662324050PE") == "options"
    assert classify_history_domain("NSE:NIFTY") == "spot"
    assert classify_history_domain("NFO:NIFTY26JULFUT") == "underlying"


def test_collect_history_bars_uses_first_available_provider_method() -> None:
    assert collect_history_bars(None, "NSE:NIFTY") == []
    assert collect_history_bars(_Provider(raises=True), "NSE:NIFTY") == []
    bars = [_Bar("09:15"), _Bar("09:16")]
    assert collect_history_bars(_HistoryOnlyProvider(bars), "NSE:NIFTY") == bars


def test_context_builder_prefers_data_hub_and_preserves_timestamps(monkeypatch) -> None:
    monkeypatch.setenv("OPTION_EVAL_MIN_BARS", "2")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "3")
    hub_bars = [{"timestamp": "09:15"}, {"timestamp": "09:16"}, {"timestamp": "09:17"}]
    engine_bars = [{"timestamp": "09:00"}]

    ctx = build_strategy_history_context(
        symbol="NFO:NIFTY2662324050CE",
        indicator_engine=_Provider(engine_bars),
        data_hub=_Provider(hub_bars),
    )

    assert ctx["history_source"] == "data_hub"
    assert ctx["history_domain_used"] == "options"
    assert ctx["history_count"] == 3
    assert ctx["option_history_count"] == 3
    assert ctx["indicator_history_count"] == 3
    assert ctx["oldest_bar_ts"] == "09:15"
    assert ctx["latest_bar_ts"] == "09:17"
    assert ctx["history_ready"] is True
    assert ctx["history_ready_for_smc"] is True


def test_context_builder_runner_context_can_supply_domain_counts(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXT_MIN_BARS", "50")
    ctx = build_strategy_history_context(
        symbol="NSE:NIFTY",
        indicator_engine=_Provider([]),
        data_hub=None,
        runner_context={"spot_history_count": 55, "option_history_count": 2},
    )

    assert ctx["history_source"] == "unavailable"
    assert ctx["history_domain_used"] == "spot"
    assert ctx["history_count"] == 55
    assert ctx["spot_history_count"] == 55
    assert ctx["history_quality"] == "warm"
    assert ctx["history_ready"] is True


def test_strategy_manager_uses_canonical_context_builder() -> None:
    assert strategy_manager.build_strategy_history_context is build_strategy_history_context
