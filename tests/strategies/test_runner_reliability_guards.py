from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.bar_builder import OneMinuteBar
from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig


class DummyIndicatorEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def update_price(self, symbol: str, price: object, **kwargs: object) -> None:
        payload = dict(kwargs)
        payload["symbol"] = symbol
        self.calls.append(payload)

    def compute_atr(self, symbol: str, period: int = 14) -> float:  # noqa: ARG002
        return 10.0


def _runner(indicator: DummyIndicatorEngine | None = None) -> StrategyRunner:
    return StrategyRunner(
        market_data_manager=MagicMock(),
        indicator_engine=indicator or DummyIndicatorEngine(),
        strategy_manager=MagicMock(),
        risk_manager=MagicMock(),
        order_manager=MagicMock(),
        position_manager=MagicMock(),
        config=StrategyRunnerConfig(),
        data_hub=MagicMock(),
    )


def test_repair_candle_gap_skips_sparse_option_ticks() -> None:
    runner = _runner()
    symbol = "NFO:NIFTYTESTCE"
    previous = OneMinuteBar(
        open=100.0,
        high=102.0,
        low=99.0,
        close=101.0,
        volume=50,
        start=datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 9, 15, 59, tzinfo=timezone.utc),
    )
    incoming = OneMinuteBar(
        open=103.0,
        high=104.0,
        low=102.0,
        close=103.0,
        volume=60,
        start=datetime(2024, 1, 1, 9, 17, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 9, 17, 59, tzinfo=timezone.utc),
    )

    repaired = runner._repair_candle_gap(symbol, previous, incoming)

    assert repaired == []


def test_ingest_bar_ignores_synthetic_volume_for_indicators() -> None:
    indicator = DummyIndicatorEngine()
    runner = _runner(indicator)
    symbol = "NFO:NIFTYTESTCE"
    runner.add_symbol(symbol)

    bar = OneMinuteBar(
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=999,
        start=datetime(2024, 1, 1, 9, 15, tzinfo=timezone.utc),
        end=datetime(2024, 1, 1, 9, 15, 59, tzinfo=timezone.utc),
        synthetic=True,
    )

    runner._ingest_bar(symbol, bar)

    assert indicator.calls
    assert indicator.calls[-1]["volume"] == 0


def test_spot_vwap_trend_filter_allows_only_directional_option_entries() -> None:
    runner = _runner()
    base = "NSE:NIFTY"
    runner.add_symbol(base)
    with runner._lock:
        state = runner._symbol_state[base]
        state.last_tick = {"ltp": 200.0}
        state.vwap = 190.0

    assert runner._passes_spot_trend_filter(base, "NFO:NIFTY26MAR20000CE") is True
    assert runner._passes_spot_trend_filter(base, "NFO:NIFTY26MAR20000PE") is False


def test_set_trade_cooldown_tracks_per_candle_count() -> None:
    runner = _runner()
    symbol = "NSE:NIFTY"
    runner.add_symbol(symbol)
    ts = datetime.now(timezone.utc).replace(second=12, microsecond=0)

    runner._set_trade_cooldown(symbol, ts)
    runner._set_trade_cooldown(symbol, ts + timedelta(seconds=10))

    candle = ts.replace(second=0, microsecond=0)
    assert runner._trade_counter_by_symbol_candle[(symbol, candle)] == 2


def test_detect_market_regime_low_volatility() -> None:
    runner = _runner()
    runner._indicator_engine.get_atr = lambda _s: 8.0

    assert runner.detect_market_regime("NSE:NIFTY") == "low_volatility"
