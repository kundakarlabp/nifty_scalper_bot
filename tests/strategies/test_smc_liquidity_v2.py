from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    SMCStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy

FUTURES = "NFO:NIFTY26SEPFUT"
SPOT = "NSE:NIFTY"
CE = "NFO:NIFTY2690324100CE"
PE = "NFO:NIFTY2690324100PE"
START = datetime(2026, 9, 2, 3, 45, tzinfo=timezone.utc)


class FakeIndicatorEngine:
    def __init__(self, histories: dict[str, list[dict[str, Any]]]) -> None:
        self.histories = histories

    def get_history(
        self, symbol: str, count: int | None = None, *, field: str = "close"
    ):
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
    config = SMCStrategyConfig(**config_overrides)
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
        "smc_awaiting_confirmation",
    }
    if strategy.last_no_vote_reason == "smc_awaiting_confirmation":
        recovered = strategy._events[(FUTURES, "CE")]
        assert recovered["recovered_from_history"] is True
        assert all(not key.startswith("premium_") for key in recovered)


def test_bullish_underlying_sweep_requires_later_confirmation_bar(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    engine = FakeIndicatorEngine({FUTURES: rows, SPOT: rows})
    strategy = SMCStrategy(SMCStrategyConfig(), engine)

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
    assert signal.metadata["requires_independent_confirmation"] is True
    assert signal.metadata["confirmation_owner"] == "StrategyManager"
    assert "requires_orderflow_confirmation" not in signal.metadata
    assert "orderflow_confirmation_owner" not in signal.metadata
    assert signal.metadata["underlying_invalidation_level"] < 23974.0


@pytest.mark.parametrize("feature", ["bos", "choch"])
@pytest.mark.parametrize("mode", ["SHADOW", "LIVE"])
def test_opposite_side_option_structure_does_not_confirm_ce_sweep(
    monkeypatch, feature: str, mode: str
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", mode)
    rows = _base_rows()
    rows.append(
        _bar(
            30,
            open_=23988.0,
            high=23991.0,
            low=23974.0,
            close=23984.0,
            volume=2400.0,
        )
    )
    opposite = _strategy(rows)
    aligned = _strategy(rows)
    sweep_indicators = _indicators(latest_bar_ts=rows[-1]["timestamp"])
    assert opposite.generate_signal(CE, sweep_indicators, 102.0) is None
    assert aligned.generate_signal(CE, sweep_indicators, 102.0) is None

    rows.append(
        _bar(
            31,
            open_=23984.0,
            high=23999.0,
            low=23982.0,
            close=23997.0,
            volume=2100.0,
        )
    )
    features = {
        "latest_bar_ts": rows[-1]["timestamp"],
        "bos_confirmed": feature == "bos",
        "choch_confirmed": feature == "choch",
    }
    opposite_vote = opposite.generate_signal(
        CE, _indicators(**features, **{f"{feature}_side": "PE"}), 103.0
    )
    aligned_vote = aligned.generate_signal(
        CE, _indicators(**features, **{f"{feature}_side": "CE"}), 103.0
    )

    assert opposite_vote is not None and aligned_vote is not None
    assert opposite_vote.metadata["structure_confirmed"] is False
    assert "structure_confirmation" not in opposite_vote.metadata["setup_reasons"]
    assert aligned_vote.metadata["structure_confirmed"] is True
    assert "structure_confirmation" in aligned_vote.metadata["setup_reasons"]


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



def test_shallow_smc_sweep_needs_structural_or_volume_confirmation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    rows = _base_rows()
    strategy = _strategy(rows, sweep_distance_points=0.5)
    sweep = _bar(
        30,
        open_=23984.0,
        high=23991.0,
        low=23979.4,
        close=23984.0,
        volume=1000.0,
    )
    rows.append(sweep)
    indicators = _indicators(
        latest_bar_ts=sweep["timestamp"],
        bos_confirmed=False,
        choch_confirmed=False,
        retest_confirmed=True,
        premium_reclaim=False,
    )
    assert strategy.generate_signal(CE, indicators, 102.0) is None
    rows.append(
        _bar(
            31,
            open_=23984.0,
            high=23999.0,
            low=23982.0,
            close=23997.0,
            volume=1000.0,
        )
    )
    indicators["latest_bar_ts"] = rows[-1]["timestamp"]
    assert strategy.generate_signal(CE, indicators, 103.0) is None
    assert strategy.last_no_vote_reason == "smc_shallow_sweep_unconfirmed"



def test_sweep_that_is_too_deep_is_treated_as_break_not_liquidity_grab(
    monkeypatch,
) -> None:
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


def test_configured_sweep_distance_is_used_as_normalized_threshold_cap(
    monkeypatch,
) -> None:
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


def test_equal_low_cluster_is_preferred_over_minor_recent_swing(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    rows[13] = _bar(13, open_=23990, high=23997, low=23986, close=23993)
    rows[14] = _bar(14, open_=23989, high=23996, low=23985, close=23992)
    rows[15] = _bar(15, open_=23987, high=23994, low=23980.4, close=23991)
    rows[16] = _bar(16, open_=23991, high=23998, low=23985, close=23995)
    rows[17] = _bar(17, open_=23995, high=24000, low=23987, close=23997)
    rows[25] = _bar(25, open_=24000, high=24012, low=23994, close=24006)
    rows[26] = _bar(26, open_=24006, high=24010, low=23991, close=24002)
    rows[27] = _bar(27, open_=24002, high=24008, low=23986, close=23999)
    rows[28] = _bar(28, open_=23999, high=24007, low=23992, close=24002)
    rows[29] = _bar(29, open_=24002, high=24009, low=23993, close=24004)
    strategy = _strategy(rows)

    sweep = _bar(
        30,
        open_=23988.0,
        high=23991.0,
        low=23976.0,
        close=23984.0,
        volume=2400.0,
    )
    rows.append(sweep)

    snapshot = strategy._underlying_snapshot(
        _indicators(latest_bar_ts=sweep["timestamp"])
    )
    assert snapshot is not None
    pivot_low = snapshot["pivot_low"]
    assert pivot_low is not None
    assert pivot_low["liquidity_level_type"] == "equal_low"
    assert pivot_low["liquidity_touch_count"] >= 2
    assert pivot_low["liquidity_level_priority"] == 3
    assert pivot_low["low"] == pytest.approx(23980.0)

    bullish, _ = strategy._sweep_diagnostics(snapshot)
    assert bullish["valid"] is True
    assert bullish["liquidity_level_type"] == "equal_low"


def test_volume_spike_config_is_consumed_as_structural_confirmation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    engine = FakeIndicatorEngine({FUTURES: rows})
    strategy = SMCStrategy(
        SMCStrategyConfig(
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
    assert (
        strategy.generate_signal(
            CE, _indicators(latest_bar_ts=sweep["timestamp"]), 102.0
        )
        is None
    )
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
    assert "volume_confirmation" in signal.metadata["setup_reasons"]


def test_smc_setup_leaves_independent_confirmation_to_strategy_manager(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    rows = _base_rows()
    engine = FakeIndicatorEngine({FUTURES: rows})
    strategy = SMCStrategy(
        SMCStrategyConfig(volume_spike_mult=10.0),
        engine,
    )

    sweep = _bar(
        30,
        open_=23988.0,
        high=23991.0,
        low=23975.0,
        close=23984.0,
        volume=1000.0,
    )
    rows.append(sweep)
    assert (
        strategy.generate_signal(
            CE,
            _indicators(
                latest_bar_ts=sweep["timestamp"],
                premium_reclaim=False,
                bos_confirmed=False,
                choch_confirmed=False,
                retest_confirmed=False,
            ),
            102.0,
        )
        is None
    )

    event = strategy._events[(FUTURES, "CE")]
    assert 0.12 <= float(event["depth_atr"]) <= 0.50
    assert event["volume_confirmation"] is False

    confirm = _bar(
        31,
        open_=23984.0,
        high=23999.0,
        low=23982.0,
        close=23997.0,
        volume=1000.0,
    )
    rows.append(confirm)

    signal = strategy.generate_signal(
        CE,
        _indicators(
            latest_bar_ts=confirm["timestamp"],
            premium_reclaim=False,
            bos_confirmed=False,
            choch_confirmed=False,
            retest_confirmed=False,
        ),
        103.0,
    )

    assert signal is not None
    assert signal.metadata["requires_independent_confirmation"] is True
    assert signal.metadata["confirmation_owner"] == "StrategyManager"
    assert signal.metadata["smc_local_support_present"] is False
    assert signal.metadata["smc_local_support_sources"] == []


def test_bearish_underlying_sweep_confirms_long_pe(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    engine = FakeIndicatorEngine({FUTURES: rows})
    strategy = SMCStrategy(SMCStrategyConfig(), engine)
    sweep = _bar(
        30,
        open_=24012.0,
        high=24027.0,
        low=24009.0,
        close=24017.0,
        volume=2400.0,
    )
    rows.append(sweep)
    assert (
        strategy.generate_signal(
            PE, _indicators("PE", latest_bar_ts=sweep["timestamp"]), 98.0
        )
        is None
    )
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


def test_capacity_rejected_smc_requires_fresh_completed_bar_confirmation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    rows = _base_rows()
    strategy = _strategy(rows)
    sweep = _bar(30, open_=23988, high=23991, low=23974, close=23984, volume=2500)
    rows.append(sweep)
    assert (
        strategy.generate_signal(
            CE, _indicators(latest_bar_ts=sweep["timestamp"]), 102.0
        )
        is None
    )
    confirm = _bar(31, open_=23984, high=24000, low=23982, close=23998, volume=2200)
    rows.append(confirm)
    indicators = _indicators(latest_bar_ts=confirm["timestamp"])

    first = strategy.generate_signal(CE, indicators, 103.0)
    assert first is not None
    strategy.notify_entry_rejected(
        "CE",
        setup_id=first.metadata["setup_id"],
        reason="no_affordable_execution_candidate",
    )

    assert strategy.generate_signal(CE, indicators, 103.0) is None
    assert strategy.last_no_vote_reason == "smc_retry_requires_fresh_confirmation"

    fresh_confirm = _bar(
        32,
        open_=23994.0,
        high=24005.0,
        low=23992.0,
        close=24002.0,
        volume=2100.0,
    )
    rows.append(fresh_confirm)
    refreshed = strategy.generate_signal(
        CE,
        _indicators(latest_bar_ts=fresh_confirm["timestamp"]),
        104.0,
    )
    assert refreshed is not None
    assert refreshed.metadata["setup_id"] == first.metadata["setup_id"]
    assert refreshed.metadata["setup_candle_timestamp"] == fresh_confirm["timestamp"]
    assert refreshed.metadata["capacity_rejection_retry"] is True


def test_same_confirmation_bar_reuses_identity_until_entry_is_accepted(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _base_rows()
    engine = FakeIndicatorEngine({FUTURES: rows})
    strategy = SMCStrategy(SMCStrategyConfig(), engine)
    sweep = _bar(30, open_=23988, high=23991, low=23974, close=23984, volume=2500)
    rows.append(sweep)
    assert (
        strategy.generate_signal(
            CE, _indicators(latest_bar_ts=sweep["timestamp"]), 102.0
        )
        is None
    )
    confirm = _bar(31, open_=23984, high=24000, low=23982, close=23998, volume=2200)
    rows.append(confirm)
    indicators = _indicators(latest_bar_ts=confirm["timestamp"])

    first = strategy.generate_signal(CE, indicators, 103.0)
    repeated = strategy.generate_signal(CE, indicators, 103.0)
    assert first is not None
    assert repeated is not None
    assert repeated.deterministic_id == first.deterministic_id

    strategy.notify_entry_accepted("CE", setup_id=first.metadata["setup_id"])

    assert strategy.generate_signal(CE, indicators, 103.0) is None
    assert strategy.last_no_vote_reason == "smc_duplicate_confirmation_bar"
