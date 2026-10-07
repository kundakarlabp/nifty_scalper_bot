from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    ORBProStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_strategies.orb_pro import ORBProStrategy

FUTURE = "NFO:NIFTY26SEPFUT"
SPOT = "NSE:NIFTY"
CE = "NFO:NIFTY26SEP24050CE"
PE = "NFO:NIFTY26SEP24050PE"
SESSION_OPEN = datetime(2026, 9, 1, 3, 45, tzinfo=timezone.utc)  # 09:15 IST


class _IndicatorEngine:
    def __init__(self, rows_by_symbol: dict[str, list[dict[str, object]]]) -> None:
        self.rows_by_symbol = rows_by_symbol

    def get_history(
        self,
        symbol: str,
        count: int | None = None,
        *,
        field: str = "close",
    ) -> list[object]:
        rows = list(self.rows_by_symbol.get(symbol, []))
        if count is not None:
            rows = rows[-count:]
        if field == "bars":
            return rows
        return [float(row["close"]) for row in rows]


def _bar(
    minute: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1_000.0,
) -> dict[str, object]:
    return {
        "timestamp": SESSION_OPEN + timedelta(minutes=minute),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "is_complete": True,
        "is_provisional": False,
    }


def _opening_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for minute in range(15):
        rows.append(
            _bar(
                minute,
                open_=24_000.0,
                high=24_020.0 if minute == 8 else 24_012.0,
                low=23_980.0 if minute == 4 else 23_992.0,
                close=24_000.0,
            )
        )
    return rows


def _base_indicators(side: str, latest_ts: datetime) -> dict[str, object]:
    return {
        "history_count": 100,
        "open": 48.0,
        "high": 52.0,
        "low": 47.0,
        "close": 51.0,
        "atr": 4.0,
        "volume": 2_000.0,
        "avg_volume": 1_000.0,
        "underlying_direction_bias": side,
        "direction_bias": side,
        "regime": "TREND_UP" if side == "CE" else "TREND_DOWN",
        "spread_pct": 0.3,
        "quote_depth_valid": True,
        "tradable_quote": True,
        "stale_data_used": False,
        "futures_symbol": FUTURE,
        "spot_symbol": SPOT,
        "futures_vwap_slope": 1.0 if side == "CE" else -1.0,
        "latest_bar_ts": latest_ts.timestamp(),
        # Deliberately contradictory legacy option-premium ORB. V2 must ignore it.
        "orb_ready": True,
        "orb_high": 105.0,
        "orb_low": 95.0,
    }


def _strategy(
    rows_by_symbol: dict[str, list[dict[str, object]]], *, orb_minutes: int = 15
) -> ORBProStrategy:
    return ORBProStrategy(
        ORBProStrategyConfig(orb_minutes=orb_minutes),
        indicator_engine=_IndicatorEngine(rows_by_symbol),
    )


def test_orb_does_not_require_legacy_option_opening_range_indicators() -> None:
    required = _strategy({}).get_required_indicators()

    assert {"orb_high", "orb_low", "orb_ready"}.isdisjoint(required)


def test_orb_uses_configured_futures_opening_range_not_option_premium(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_008.0,
            high=24_034.0,
            low=24_006.0,
            close=24_030.0,
            volume=3_000.0,
        )
    )
    strategy = _strategy({FUTURE: rows})
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators["futures_price"] = 24_030.0

    signal = strategy.generate_signal(CE, indicators, 50.0)

    assert signal is not None
    assert signal.symbol == CE
    assert signal.metadata["opening_range_source"] == "futures"
    assert signal.metadata["opening_range_high"] == 24_020.0
    assert signal.metadata["opening_range_low"] == 23_980.0
    assert signal.metadata["orb_window_minutes"] == 15
    assert "legacy_option_orb_high" not in signal.metadata
    assert "legacy_option_orb_low" not in signal.metadata
    assert "legacy_option_orb_ready" not in signal.metadata
    assert signal.metadata["signal_domain"] == "NIFTY_FUTURES"
    assert signal.stop_loss is not None and 0 < signal.stop_loss < 50.0
    assert signal.take_profit is not None and signal.take_profit > 50.0


def test_same_completed_breakout_bar_cannot_vote_twice(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_008.0,
            high=24_034.0,
            low=24_006.0,
            close=24_030.0,
            volume=3_000.0,
        )
    )
    strategy = _strategy({FUTURE: rows})
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators["futures_price"] = 24_030.0

    assert strategy.generate_signal(CE, indicators, 50.0) is not None
    assert strategy.generate_signal(CE, indicators, 50.0) is None
    assert strategy.last_no_vote_reason == "orb_bar_already_evaluated"


def test_retest_must_follow_breakout_before_retest_branch_votes(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_014.0,
            high=24_034.0,
            low=24_012.0,
            close=24_030.0,
            volume=1_600.0,
        )
    )
    engine = _IndicatorEngine({FUTURE: rows})
    strategy = ORBProStrategy(
        ORBProStrategyConfig(orb_minutes=15),
        indicator_engine=engine,
    )
    first = _base_indicators("CE", rows[-1]["timestamp"])
    first["futures_price"] = 24_030.0

    assert strategy.generate_signal(CE, first, 50.0) is None
    assert strategy.last_no_vote_reason == "awaiting_orb_retest"

    rows.append(
        _bar(
            16,
            open_=24_024.0,
            high=24_026.0,
            low=24_019.0,
            close=24_021.0,
            volume=400.0,
        )
    )
    second = _base_indicators("CE", rows[-1]["timestamp"])
    second["futures_price"] = 24_021.0
    signal = strategy.generate_signal(CE, second, 50.0)

    assert signal is not None
    assert signal.metadata["entry_branch"] == "retest"
    assert signal.metadata["retest_confirmed"] is True
    assert signal.metadata["breakout_timestamp"] != signal.metadata["retest_timestamp"]
    assert signal.metadata["underlying_volume_ratio"] > 1.2
    assert signal.metadata["underlying_current_volume_ratio"] < 1.2
    assert signal.metadata["underlying_penetration_atr"] >= 0.2
    assert signal.metadata["underlying_current_penetration_atr"] < 0.2


def test_retest_cannot_rescue_low_participation_breakout(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_014.0,
            high=24_034.0,
            low=24_012.0,
            close=24_030.0,
            volume=900.0,
        )
    )
    engine = _IndicatorEngine({FUTURE: rows})
    strategy = ORBProStrategy(
        ORBProStrategyConfig(orb_minutes=15),
        indicator_engine=engine,
    )

    first = _base_indicators("CE", rows[-1]["timestamp"])
    assert strategy.generate_signal(CE, first, 50.0) is None
    assert strategy.last_no_vote_reason == "awaiting_orb_retest"

    rows.append(
        _bar(
            16,
            open_=24_024.0,
            high=24_026.0,
            low=24_019.0,
            close=24_021.0,
            volume=4_000.0,
        )
    )
    second = _base_indicators("CE", rows[-1]["timestamp"])

    assert strategy.generate_signal(CE, second, 50.0) is None
    assert strategy.last_no_vote_reason == "orb_structural_contract_not_passed"


def test_retest_tolerance_stays_anchored_to_breakout_volatility(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_014.0,
            high=24_034.0,
            low=24_012.0,
            close=24_030.0,
            volume=1_600.0,
        )
    )
    engine = _IndicatorEngine({FUTURE: rows})
    strategy = ORBProStrategy(
        ORBProStrategyConfig(orb_minutes=15),
        indicator_engine=engine,
    )

    first = _base_indicators("CE", rows[-1]["timestamp"])
    assert strategy.generate_signal(CE, first, 50.0) is None
    assert strategy.last_no_vote_reason == "awaiting_orb_retest"

    # A large upper wick inflates the later ATR. Its low is still too far above
    # the boundary to be a retest under the breakout-time tolerance.
    rows.append(
        _bar(
            16,
            open_=24_028.0,
            high=24_300.0,
            low=24_025.0,
            close=24_030.0,
            volume=400.0,
        )
    )
    second = _base_indicators("CE", rows[-1]["timestamp"])
    assert strategy.generate_signal(CE, second, 50.0) is None
    assert strategy.last_no_vote_reason == "awaiting_orb_retest"

    rows.append(
        _bar(
            17,
            open_=24_024.0,
            high=24_026.0,
            low=24_019.0,
            close=24_021.0,
            volume=400.0,
        )
    )
    third = _base_indicators("CE", rows[-1]["timestamp"])
    signal = strategy.generate_signal(CE, third, 50.0)

    assert signal is not None
    assert signal.metadata["entry_branch"] == "retest"


def test_missed_prior_retest_cannot_be_replayed_on_later_bar(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_014.0,
            high=24_034.0,
            low=24_012.0,
            close=24_030.0,
            volume=1_600.0,
        )
    )
    engine = _IndicatorEngine({FUTURE: rows})
    strategy = ORBProStrategy(
        ORBProStrategyConfig(orb_minutes=15),
        indicator_engine=engine,
    )
    first = _base_indicators("CE", rows[-1]["timestamp"])

    assert strategy.generate_signal(CE, first, 50.0) is None
    assert strategy.last_no_vote_reason == "awaiting_orb_retest"

    # Minute 16 is a valid first retest, but the strategy is not evaluated on it.
    rows.append(
        _bar(
            16,
            open_=24_024.0,
            high=24_026.0,
            low=24_019.0,
            close=24_021.0,
            volume=400.0,
        )
    )
    rows.append(
        _bar(
            17,
            open_=24_024.0,
            high=24_026.0,
            low=24_019.0,
            close=24_021.0,
            volume=400.0,
        )
    )
    later = _base_indicators("CE", rows[-1]["timestamp"])

    assert strategy.generate_signal(CE, later, 50.0) is None
    assert strategy.last_no_vote_reason == "orb_retest_missed"


def test_missed_prior_invalidation_blocks_later_retest(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_MOMENTUM_BRANCH_ENABLED", "false")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_014.0,
            high=24_034.0,
            low=24_012.0,
            close=24_030.0,
            volume=1_600.0,
        )
    )
    engine = _IndicatorEngine({FUTURE: rows})
    strategy = ORBProStrategy(
        ORBProStrategyConfig(orb_minutes=15),
        indicator_engine=engine,
    )
    first = _base_indicators("CE", rows[-1]["timestamp"])

    assert strategy.generate_signal(CE, first, 50.0) is None
    assert strategy.last_no_vote_reason == "awaiting_orb_retest"

    # Minute 16 invalidates the breakout while the strategy is not evaluated.
    rows.append(
        _bar(
            16,
            open_=24_018.0,
            high=24_020.0,
            low=24_006.0,
            close=24_010.0,
            volume=400.0,
        )
    )
    rows.append(
        _bar(
            17,
            open_=24_024.0,
            high=24_026.0,
            low=24_019.0,
            close=24_021.0,
            volume=400.0,
        )
    )
    later = _base_indicators("CE", rows[-1]["timestamp"])

    assert strategy.generate_signal(CE, later, 50.0) is None
    assert strategy.last_no_vote_reason == "orb_breakout_invalidated"


def test_option_premium_breakout_without_underlying_breakout_does_not_vote(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_000.0,
            high=24_015.0,
            low=23_995.0,
            close=24_010.0,
            volume=2_000.0,
        )
    )
    strategy = _strategy({FUTURE: rows})
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators.update({"close": 120.0, "high": 121.0, "futures_price": 24_010.0})

    assert strategy.generate_signal(CE, indicators, 120.0) is None
    assert strategy.last_no_vote_reason == "no_fresh_underlying_breakout"


def test_pe_breakout_keeps_long_option_stop_below_entry(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=23_990.0,
            high=23_992.0,
            low=23_955.0,
            close=23_962.0,
            volume=3_000.0,
        )
    )
    strategy = _strategy({FUTURE: rows})
    indicators = _base_indicators("PE", rows[-1]["timestamp"])
    indicators["futures_price"] = 23_962.0

    signal = strategy.generate_signal(PE, indicators, 48.0)

    assert signal is not None
    assert signal.stop_loss is not None and 0 < signal.stop_loss < 48.0
    assert signal.take_profit is not None and signal.take_profit > 48.0
    assert signal.metadata["underlying_invalidation"] > 23_962.0
    assert signal.metadata["breakout_side"] == "PE"


def test_futures_unavailable_does_not_use_non_traded_spot_volume(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _opening_rows()
    for row in rows:
        row["volume"] = 0.0
    rows.append(
        _bar(
            15,
            open_=24_008.0,
            high=24_034.0,
            low=24_006.0,
            close=24_030.0,
            volume=0.0,
        )
    )
    strategy = _strategy({SPOT: rows})
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators["futures_price"] = None
    indicators["spot_price"] = 24_030.0

    assert strategy.generate_signal(CE, indicators, 50.0) is None
    assert strategy.last_no_vote_reason == "underlying_orb_not_ready"


def test_prior_session_volume_cannot_dilute_current_breakout_participation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    prior_session = []
    for minute in range(5):
        row = _bar(
            minute,
            open_=23_900.0,
            high=23_910.0,
            low=23_890.0,
            close=23_900.0,
            volume=100_000.0,
        )
        row["timestamp"] = row["timestamp"] - timedelta(days=1)
        prior_session.append(row)

    rows = [*prior_session, *_opening_rows()]
    rows.append(
        _bar(
            15,
            open_=24_008.0,
            high=24_034.0,
            low=24_006.0,
            close=24_030.0,
            volume=3_000.0,
        )
    )
    strategy = _strategy({FUTURE: rows})
    signal = strategy.generate_signal(
        CE, _base_indicators("CE", rows[-1]["timestamp"]), 50.0
    )

    assert signal is not None
    assert signal.metadata["underlying_volume_ratio"] == pytest.approx(3.0)
    assert (
        signal.metadata["underlying_volume_ratio_basis"]
        == "same_session_prior_completed_bars"
    )


def test_short_orb_atr_ignores_previous_session_gap(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_BALANCED_RANGE_MAX_ATR", "2.0")
    prior_session = []
    for minute in range(10):
        row = _bar(
            minute,
            open_=23_000.0,
            high=23_005.0,
            low=22_995.0,
            close=23_000.0,
        )
        row["timestamp"] = row["timestamp"] - timedelta(days=1)
        prior_session.append(row)

    rows = [*prior_session, *_opening_rows()[:5]]
    rows.append(
        _bar(
            5,
            open_=24_008.0,
            high=24_034.0,
            low=24_006.0,
            close=24_030.0,
            volume=3_000.0,
        )
    )
    strategy = _strategy({FUTURE: rows}, orb_minutes=5)
    signal = strategy.generate_signal(
        CE, _base_indicators("CE", rows[-1]["timestamp"]), 50.0
    )

    assert signal is not None
    assert signal.metadata["underlying_atr"] < 50.0
    assert signal.metadata["underlying_atr_basis"] == "same_session_completed_bars"


def test_late_breakout_outside_orb_entry_lifetime_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ORB_MAX_ENTRY_MINUTES_AFTER_RANGE", "120")
    rows = _opening_rows()
    rows.append(_bar(164, open_=24_000.0, high=24_015.0, low=23_995.0, close=24_010.0))
    rows.append(
        _bar(
            165,
            open_=24_010.0,
            high=24_040.0,
            low=24_008.0,
            close=24_035.0,
            volume=3_000.0,
        )
    )
    strategy = _strategy({FUTURE: rows})
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators["futures_price"] = 24_035.0

    assert strategy.generate_signal(CE, indicators, 50.0) is None
    assert strategy.last_no_vote_reason == "orb_entry_window_expired"


@pytest.mark.parametrize("missing_minute", [0, 8, 14])
def test_incomplete_opening_range_cannot_generate_breakout(
    monkeypatch, missing_minute
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _opening_rows()
    rows.pop(missing_minute)
    # A duplicate must not disguise a missing minute as fifteen complete bars.
    rows.append(dict(rows[1]))
    rows.append(
        _bar(
            15,
            open_=24_008.0,
            high=24_034.0,
            low=24_006.0,
            close=24_030.0,
            volume=3_000.0,
        )
    )
    strategy = _strategy({FUTURE: rows})

    assert (
        strategy.generate_signal(
            CE, _base_indicators("CE", rows[-1]["timestamp"]), 50.0
        )
        is None
    )
    assert strategy.last_no_vote_reason == "underlying_orb_not_ready"


def test_incomplete_futures_range_cannot_be_rescued_by_spot_index(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    futures_rows = _opening_rows()
    spot_rows = [dict(row) for row in futures_rows]
    breakout = _bar(
        15,
        open_=24_008.0,
        high=24_034.0,
        low=24_006.0,
        close=24_030.0,
        volume=3_000.0,
    )
    futures_rows.append(breakout)
    spot_rows.append({**breakout, "volume": 0.0})
    strategy = _strategy({FUTURE: futures_rows[1:], SPOT: spot_rows})

    assert (
        strategy.generate_signal(
            CE, _base_indicators("CE", breakout["timestamp"]), 50.0
        )
        is None
    )
    assert strategy.last_no_vote_reason == "underlying_orb_not_ready"


@pytest.mark.parametrize("orb_minutes", [5, 10, 15])
def test_opening_range_backfill_restores_same_bar_evaluation(
    monkeypatch, orb_minutes
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    rows = _opening_rows()[:orb_minutes]
    missing = rows.pop(0)
    rows.append(
        _bar(
            orb_minutes,
            open_=24_008.0,
            high=24_034.0,
            low=24_006.0,
            close=24_030.0,
            volume=3_000.0,
        )
    )
    strategy = _strategy({FUTURE: rows}, orb_minutes=orb_minutes)
    indicators = _base_indicators("CE", rows[-1]["timestamp"])

    assert strategy.generate_signal(CE, indicators, 50.0) is None
    rows.insert(0, missing)
    signal = strategy.generate_signal(CE, indicators, 50.0)

    assert signal is not None
    assert signal.metadata["opening_range_complete"] is True
    assert signal.metadata["orb_window_minutes"] == orb_minutes
