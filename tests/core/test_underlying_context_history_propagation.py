from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.core.strategy_runner_dynamic_universe_safety import apply_patches
from nifty_scalper_bot.strategies.runner import StrategyRunner


FUTURE = "NFO:NIFTY26SEPFUT"
SPOT = "NSE:NIFTY"
SESSION_OPEN = datetime(2026, 9, 3, 3, 45, tzinfo=timezone.utc)  # 09:15 IST


def _bars(count: int) -> list[dict[str, object]]:
    return [
        {
            "timestamp": SESSION_OPEN + timedelta(minutes=i),
            "open": 24_000.0 + i,
            "high": 24_001.0 + i,
            "low": 23_999.0 + i,
            "close": 24_000.5 + i,
            "volume": 1_000 + i,
        }
        for i in range(count)
    ]


class _MDM:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.requested_limits: list[int | None] = []

    def get_ohlc_bars(self, _symbol: str, *, limit: int | None = None):
        self.requested_limits.append(limit)
        if limit is None:
            return list(self.rows)
        return list(self.rows)[-int(limit) :]



def _runner(rows: list[dict[str, object]]) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._market_data = _MDM(rows)
    runner._data_hub = None
    runner._active_futures_symbol = FUTURE
    runner._spot_symbol = SPOT
    return runner


def test_futures_context_history_preserves_session_opening_range_when_orb_enabled(
    monkeypatch,
) -> None:
    """Underlying-led ORB must retain 09:15 bars after the normal context tail rolls."""
    monkeypatch.setenv("ORB_ENABLED", "true")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "30")
    apply_patches()
    runner = _runner(_bars(150))

    rows = runner._get_mdm_bars(FUTURE, 20)

    assert len(rows) == 150
    assert rows[0]["timestamp"] == SESSION_OPEN
    assert runner._market_data.requested_limits[-1] >= 150


def test_spot_fallback_history_preserves_session_opening_range_when_orb_enabled(
    monkeypatch,
) -> None:
    """ORB spot fallback needs the same structural history contract as futures."""
    monkeypatch.setenv("ORB_ENABLED", "true")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "30")
    apply_patches()
    runner = _runner(_bars(150))

    rows = runner._get_mdm_bars(SPOT, 20)

    assert len(rows) == 150
    assert rows[0]["timestamp"] == SESSION_OPEN


def test_context_history_never_falls_below_smc_structural_minimum(monkeypatch) -> None:
    """SMC must see its minimum underlying-bar window even when generic context is shorter."""
    monkeypatch.setenv("ORB_ENABLED", "false")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "30")
    apply_patches()
    runner = _runner(_bars(60))

    rows = runner._get_mdm_bars(FUTURE, 20)

    assert len(rows) == 30
    assert runner._market_data.requested_limits[-1] == 30


def test_option_history_read_limit_is_unchanged(monkeypatch) -> None:
    """The context fix must not broaden unrelated option-history reads."""
    monkeypatch.setenv("ORB_ENABLED", "false")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "30")
    apply_patches()
    runner = _runner(_bars(60))

    rows = runner._get_mdm_bars("NFO:NIFTY2690823950CE", 20)

    assert len(rows) == 20
    assert runner._market_data.requested_limits[-1] == 20
