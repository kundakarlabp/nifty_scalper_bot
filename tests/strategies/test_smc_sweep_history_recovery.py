from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from nifty_scalper_bot.strategies.elite_strategies.config_models import SMCStrategyConfig
from nifty_scalper_bot.strategies.elite_strategies.smc_liquidity import SMCStrategy

FUTURES = "NFO:NIFTY26SEPFUT"
SPOT = "NSE:NIFTY"
CE = "NFO:NIFTY2690823650CE"
START = datetime(2026, 9, 8, 4, 30, tzinfo=timezone.utc)


class _Engine:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def get_history(self, symbol: str, count: int | None = None, *, field: str = "close"):
        assert symbol in {FUTURES, SPOT}
        rows = list(self.rows)
        if count is not None:
            rows = rows[-count:]
        return rows if field == "bars" else [row["close"] for row in rows]


def _bar(minute: int, *, open_: float, high: float, low: float, close: float, volume: float = 1000.0):
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
            )
        )
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


def _indicators(latest_ts: datetime) -> dict[str, Any]:
    return {
        "open": 100.0,
        "high": 104.0,
        "low": 98.0,
        "close": 103.0,
        "atr": 4.0,
        "direction_bias": "CE",
        "underlying_direction_bias": "CE",
        "futures_symbol": FUTURES,
        "spot_symbol": SPOT,
        "premium_reclaim": True,
        "bos_confirmed": True,
        "choch_confirmed": False,
        "retest_confirmed": True,
        "spread_pct": 0.3,
        "tradable_quote": True,
        "quote_depth_valid": True,
        "stale_data_used": False,
        "latest_bar_ts": latest_ts,
    }


def test_restart_recovers_unconfirmed_recent_underlying_sweep(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    rows = _base_rows()
    sweep = _bar(30, open_=23988, high=23991, low=23974, close=23984, volume=2500)
    confirm = _bar(31, open_=23984, high=24000, low=23982, close=23998, volume=2200)
    rows.extend([sweep, confirm])

    strategy = SMCStrategy(SMCStrategyConfig(min_confidence=0.0), _Engine(rows))
    signal = strategy.generate_signal(CE, _indicators(confirm["timestamp"]), 103.0)

    assert signal is not None
    assert signal.metadata["structure_source"] == "futures"
    assert signal.metadata["sweep_timestamp"] == sweep["timestamp"]
    assert signal.metadata["sweep_recovered_from_history"] is True


def test_recovery_does_not_late_enter_after_prior_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    rows = _base_rows()
    sweep = _bar(30, open_=23988, high=23991, low=23974, close=23984, volume=2500)
    prior_confirm = _bar(31, open_=23984, high=24000, low=23982, close=23998, volume=2200)
    current = _bar(32, open_=23998, high=24002, low=23995, close=24000, volume=1500)
    rows.extend([sweep, prior_confirm, current])

    strategy = SMCStrategy(SMCStrategyConfig(min_confidence=0.0), _Engine(rows))
    signal = strategy.generate_signal(CE, _indicators(current["timestamp"]), 103.0)

    assert signal is None
    assert strategy.last_no_vote_reason != "smc_awaiting_confirmation"


def test_recovery_remains_underlying_only_in_live(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    strategy = SMCStrategy(SMCStrategyConfig(min_confidence=0.0), _Engine([]))
    indicators = _indicators(START + timedelta(minutes=31))
    indicators.update(
        {
            "prior_swing_low": 99.0,
            "low": 95.0,
            "close": 101.0,
            "liquidity_sweep_confirmed": True,
        }
    )

    assert strategy.generate_signal(CE, indicators, 102.0) is None
    assert strategy.last_no_vote_reason == "underlying_context_not_ready"
