from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.config_models import SMCStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy


FUTURES = "NFO:NIFTY26SEPFUT"
SPOT = "NSE:NIFTY"
CE = "NFO:NIFTY2690324100CE"
PE = "NFO:NIFTY2690324100PE"
START = datetime(2026, 9, 2, 3, 45, tzinfo=timezone.utc)


class FakeIndicatorEngine:
    def __init__(self, histories: dict[str, list[dict[str, Any]]]) -> None:
        self.histories = histories

    def get_history(self, symbol: str, count: int | None = None, *, field: str = "close"):
        rows = list(self.histories.get(symbol, []))
        if count is not None:
            rows = rows[-count:]
        if field == "bars":
            return rows
        return [row["close"] for row in rows]


def _bar(
    minute: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1000.0,
) -> dict[str, Any]:
    return {
        "timestamp": START + timedelta(minutes=minute),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "is_complete": True,
        "is_provisional": False,
    }


def _base_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for minute in range(30):
        center = 24000.0 + ((minute % 5) - 2) * 1.5
        rows.append(
            _bar(
                minute,
                open_=center - 1.0,
                high=center + 4.0,
                low=center - 4.0,
                close=center + 1.0,
                volume=1000.0,
            )
        )
    # Confirmed pivot low at minute 22 and pivot high at minute 25. Each has
    # two completed bars on either side before a later sweep can occur.
    rows[20] = _bar(20, open_=23991, high=23997, low=23988, close=23994)
    rows[21] = _bar(21, open_=23990, high=23996, low=23986, close=23992)
    rows[22] = _bar(22, open_=23988, high=23995, low=23980, close=23991)
    rows[23] = _bar(23, open_=23991, high=23999, low=23987, close=23996)
    rows[24] = _bar(24, open_=23996, high=24008, low=23992, close=24004)
    rows[25] = _bar(25, open_=24004, high=24020, low=23998, close=24008)
    rows[26] = _bar(26, open_=24008, high=24014, low=23999, close=24005)
    rows[27] = _bar(27, open_=24005, high=24012, low=23997, close=24001)
    rows[28] = _bar(28, open_=24001, high=24009, low=23995, close=24003)
    rows[29] = _bar(29, open_=24003, high=24010, low=23996, close=24002)
    return rows


def _indicators(side: str = "CE", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "open": 100.0,
        "high": 104.0,
        "low": 98.0,
        "close": 102.0,
        "atr": 4.0,
        "history_count": 10,
        "history_resolved_count": 10,
        "option_history_count": 10,
        "direction_bias": side,
        "underlying_direction_bias": side,
        "futures_symbol": FUTURES,
        "spot_symbol": SPOT,
        "premium_reclaim": True,
        "bos_confirmed": True,
        "choch_confirmed": False,
        "retest_confirmed": True,
        "spread_pct": 0.4,
        "tradable_quote": True,
        "quote_depth_valid": True,
        "stale_data_used": False,
    }
    payload.update(overrides)
    return payload


def _strategy(rows: list[dict[str, Any]], **config_overrides: Any) -> SMCStrategy:
    config = SMCStrategyConfig(min_confidence=0.0, **config_overrides)
    return SMCStrategy(config, FakeIndicatorEngine({FUTURES: rows, SPOT: rows}))


def test_option_premium_sweep_cannot_replace_underlying_structure(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    strategy = _strategy(rows)
    indicators = _indicators(
        prior_swing_low=99.0,
        low=95.0,
        close=101.0,
        liquidity_sweep_confirmed=True,
        latest_bar_ts=rows[-1]["timestamp"],
    )

    assert strategy.generate_signal(CE, indicators, 102.0) is None
    assert strategy.last_no_vote_reason in {
        "underlying_no_liquidity_sweep",
        "smc_awaiting_sweep",
    }


def test_bullish_underlying_sweep_requires_later_confirmation_bar(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    engine = FakeIndicatorEngine({FUTURES: rows, SPOT: rows})
    strategy = SMCStrategy(SMCStrategyConfig(min_confidence=0.0), engine)

    sweep = _bar(
        30,
        open_=23988.0,
        high=23991.0,
        low=23974.0,
        close=23984.0,
        volume=2400.0,
    )
    rows.append(sweep)
    first = strategy.generate_signal(
        CE,
        _indicators(latest_bar_ts=sweep["timestamp"]),
        102.0,
    )
    assert first is None
    assert strategy.last_no_vote_reason == "smc_awaiting_confirmation"

    confirm = _bar(
        31,
        open_=23984.0,
        high=23999.0,
        low=23982.0,
        close=23997.0,
        volume=2100.0,
    )
    rows.append(confirm)
    signal = strategy.generate_signal(
        CE,
        _indicators(latest_bar_ts=confirm["timestamp"]),
        103.0,
    )

    assert signal is not None
    assert signal.symbol == CE
    assert signal.metadata["trade_side"] == "CE"
    assert signal.metadata["structure_source"] == "futures"
    assert signal.metadata["source_domain"] == "underlying_price"
    assert signal.metadata["sweep_depth_points"] > 0
    assert signal.metadata["sweep_depth_atr"] > 0
    assert signal.metadata["reclaim_distance_points"] > 0
    assert signal.metadata["requires_orderflow_confirmation"] is True
    assert signal.metadata["orderflow_confirmation_owner"] == "StrategyManager"
    assert signal.metadata["underlying_invalidation_level"] < 23974.0


def test_tiny_one_tick_breach_is_not_accepted_as_liquidity_sweep(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    strategy = _strategy(rows)
    tiny = _bar(
        30,
        open_=23984.0,
        high=23989.0,
        low=23979.8,
        close=23982.0,
        volume=2500.0,
    )
    rows.append(tiny)

    assert (
        strategy.generate_signal(
            CE,
            _indicators(latest_bar_ts=tiny["timestamp"]),
            102.0,
        )
        is None
    )
    assert strategy.last_no_vote_reason == "smc_sweep_too_shallow"


def test_sweep_that_is_too_deep_is_treated_as_break_not_liquidity_grab(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    strategy = _strategy(rows)
    breakdown = _bar(
        30,
        open_=23988.0,
        high=23990.0,
        low=23945.0,
        close=23984.0,
        volume=3000.0,
    )
    rows.append(breakdown)

    assert (
        strategy.generate_signal(
            CE,
            _indicators(latest_bar_ts=breakdown["timestamp"]),
            102.0,
        )
        is None
    )
    assert strategy.last_no_vote_reason == "smc_sweep_too_deep"


def test_configured_sweep_distance_is_used_as_normalized_threshold_cap(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    # A deliberately small cap proves the config is not dead: effective minimum
    # sweep depth cannot exceed this absolute point value even when ATR expands.
    strategy = _strategy(rows, sweep_distance_points=0.5)
    sweep = _bar(
        30,
        open_=23984.0,
        high=23990.0,
        low=23979.4,
        close=23982.0,
        volume=2400.0,
    )
    rows.append(sweep)

    assert (
        strategy.generate_signal(
            CE,
            _indicators(latest_bar_ts=sweep["timestamp"]),
            102.0,
        )
        is None
    )
    assert strategy.last_no_vote_reason == "smc_awaiting_confirmation"
    assert strategy.last_sweep_diagnostics["effective_min_sweep_points"] == 0.5


def test_volume_spike_config_is_consumed_as_quality_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    engine = FakeIndicatorEngine({FUTURES: rows})
    strategy = SMCStrategy(
        SMCStrategyConfig(
            min_confidence=0.0,
            volume_spike_mult=1.5,
        ),
        engine,
    )
    sweep = _bar(
        30,
        open_=23988.0,
        high=23991.0,
        low=23974.0,
        close=23984.0,
        volume=2500.0,
    )
    rows.append(sweep)
    assert strategy.generate_signal(CE, _indicators(latest_bar_ts=sweep["timestamp"]), 102.0) is None
    confirm = _bar(31, open_=23984, high=24000, low=23982, close=23998, volume=2200)
    rows.append(confirm)

    signal = strategy.generate_signal(
        CE,
        _indicators(latest_bar_ts=confirm["timestamp"]),
        103.0,
    )
    assert signal is not None
    assert signal.metadata["volume_confirmation"] is True
    assert signal.metadata["volume_spike_threshold"] == 1.5
    assert "volume_confirmation" in signal.metadata["score_reasons"]


def test_bearish_underlying_sweep_confirms_long_pe(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    engine = FakeIndicatorEngine({FUTURES: rows})
    strategy = SMCStrategy(SMCStrategyConfig(min_confidence=0.0), engine)
    sweep = _bar(
        30,
        open_=24012.0,
        high=24027.0,
        low=24009.0,
        close=24017.0,
        volume=2400.0,
    )
    rows.append(sweep)
    assert strategy.generate_signal(PE, _indicators("PE", latest_bar_ts=sweep["timestamp"]), 98.0) is None
    confirm = _bar(
        31,
        open_=24017.0,
        high=24019.0,
        low=23999.0,
        close=24001.0,
        volume=2200.0,
    )
    rows.append(confirm)

    signal = strategy.generate_signal(
        PE,
        _indicators("PE", latest_bar_ts=confirm["timestamp"]),
        99.0,
    )
    assert signal is not None
    assert signal.symbol == PE
    assert signal.metadata["trade_side"] == "PE"
    assert signal.metadata["smc_sweep_type"] == "bearish"
    assert signal.metadata["underlying_invalidation_level"] > 24027.0


def test_same_underlying_confirmation_bar_cannot_emit_duplicate_vote(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    engine = FakeIndicatorEngine({FUTURES: rows})
    strategy = SMCStrategy(SMCStrategyConfig(min_confidence=0.0), engine)
    sweep = _bar(30, open_=23988, high=23991, low=23974, close=23984, volume=2500)
    rows.append(sweep)
    assert strategy.generate_signal(CE, _indicators(latest_bar_ts=sweep["timestamp"]), 102.0) is None
    confirm = _bar(31, open_=23984, high=24000, low=23982, close=23998, volume=2200)
    rows.append(confirm)
    indicators = _indicators(latest_bar_ts=confirm["timestamp"])

    assert strategy.generate_signal(CE, indicators, 103.0) is not None
    assert strategy.generate_signal(CE, indicators, 103.0) is None
    assert strategy.last_no_vote_reason == "smc_duplicate_confirmation_bar"
