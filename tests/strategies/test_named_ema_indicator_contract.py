"""Contract tests for requested, symbol-scoped fast/slow EMA indicators."""

from __future__ import annotations

from nifty_scalper_bot.strategies.indicators import IndicatorEngine


def test_requested_named_emas_use_existing_period_calculators(monkeypatch) -> None:
    engine = IndicatorEngine()
    calls: list[tuple[str, int]] = []

    def _ema(symbol: str, period: int = 20) -> float:
        calls.append((symbol, period))
        return {9: 109.0, 20: 120.0, 21: 121.0}[period]

    monkeypatch.setattr(engine, "get_ema", _ema)
    result = engine.get_indicators("NSE:NIFTY", ["ema_fast", "ema_slow"])

    assert result["ema_fast"] == 109.0
    assert result["ema_slow"] == 121.0
    assert result["ema_fast"] != result["ema_slow"]
    assert ("NSE:NIFTY", 9) in calls
    assert ("NSE:NIFTY", 21) in calls


def test_unrequested_named_emas_are_not_computed(monkeypatch) -> None:
    engine = IndicatorEngine()
    requested_periods: list[int] = []

    def _ema(_symbol: str, period: int = 20) -> float:
        requested_periods.append(period)
        return 100.0

    monkeypatch.setattr(engine, "get_ema", _ema)
    result = engine.get_indicators("NSE:NIFTY", ["adx"])

    assert "ema_fast" not in result
    assert "ema_slow" not in result
    assert requested_periods == [20]


def test_existing_runtime_ema_values_take_precedence(monkeypatch) -> None:
    engine = IndicatorEngine()
    engine._runtime_context["NSE:NIFTY"] = {"ema_fast": 101.0, "ema_slow": 99.0}
    periods: list[int] = []

    def _ema(_symbol: str, period: int = 20) -> float:
        periods.append(period)
        return 250.0

    monkeypatch.setattr(engine, "get_ema", _ema)
    result = engine.get_indicators("NSE:NIFTY", ["ema_fast", "ema_slow"])

    assert result["ema_fast"] == 101.0
    assert result["ema_slow"] == 99.0
    assert periods == [20]


def test_missing_history_does_not_fabricate_named_emas() -> None:
    engine = IndicatorEngine()
    result = engine.get_indicators("NSE:NIFTY", ["ema_fast", "ema_slow"])
    assert result.get("ema_fast") is None
    assert result.get("ema_slow") is None
