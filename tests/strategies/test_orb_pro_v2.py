from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.strategies.elite_strategies.config_models import ORBProStrategyConfig
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
        ORBProStrategyConfig(min_confidence=0.0, orb_minutes=orb_minutes),
        indicator_engine=_IndicatorEngine(rows_by_symbol),
    )


def test_orb_uses_configured_futures_opening_range_not_option_premium(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
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
    assert signal.metadata["legacy_option_orb_high"] == 105.0
    assert signal.metadata["signal_domain"] == "NIFTY_FUTURES"
    assert signal.stop_loss is not None and 0 < signal.stop_loss < 50.0
    assert signal.take_profit is not None and signal.take_profit > 50.0


def test_same_completed_breakout_bar_cannot_vote_twice(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
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
    rows = _opening_rows()
    rows.append(
        _bar(
            15,
            open_=24_014.0,
            high=24_028.0,
            low=24_012.0,
            close=24_025.0,
            volume=1_100.0,
        )
    )
    engine = _IndicatorEngine({FUTURE: rows})
    strategy = ORBProStrategy(
        ORBProStrategyConfig(min_confidence=0.0, orb_minutes=15),
        indicator_engine=engine,
    )
    first = _base_indicators("CE", rows[-1]["timestamp"])
    first["futures_price"] = 24_025.0

    assert strategy.generate_signal(CE, first, 50.0) is None
    assert strategy.last_no_vote_reason == "awaiting_orb_retest"

    rows.append(
        _bar(
            16,
            open_=24_022.0,
            high=24_030.0,
            low=24_019.0,
            close=24_027.0,
            volume=1_400.0,
        )
    )
    second = _base_indicators("CE", rows[-1]["timestamp"])
    second["futures_price"] = 24_027.0
    signal = strategy.generate_signal(CE, second, 50.0)

    assert signal is not None
    assert signal.metadata["entry_branch"] == "retest"
    assert signal.metadata["retest_confirmed"] is True
    assert signal.metadata["breakout_timestamp"] != signal.metadata["retest_timestamp"]


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


def test_futures_unavailable_uses_spot_context_but_never_executes_spot(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
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
    strategy = _strategy({SPOT: rows})
    indicators = _base_indicators("CE", rows[-1]["timestamp"])
    indicators["futures_price"] = None
    indicators["spot_price"] = 24_030.0

    signal = strategy.generate_signal(CE, indicators, 50.0)

    assert signal is not None
    assert signal.symbol == CE
    assert signal.metadata["opening_range_source"] == "spot_fallback"
    assert signal.metadata["underlying_symbol"] == SPOT


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
