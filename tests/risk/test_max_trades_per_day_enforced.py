"""max_trades_per_day must actually stop entries.

settings.max_trades_per_day already defaulted to 6 and the risk guard already
compared trades_today >= max_trades. But the guard resolved the count via
_call_count(position_manager, "trades_today", "daily_trade_count",
"trade_count_today") and PositionManager implemented NONE of those names, so
_call_count fell through to `return 0`. The comparison was permanently 0 >= 6
and the daily cap never fired -- a configured capital control that silently
did nothing.

Observed consequence (31 Jul session): repeated ~50-145s round trips on the
same strike, each closing for a small loss plus roughly Rs 61 of round-trip
cost (pnl -126.75 -> net -188.09; pnl -104.00 -> net -164.81).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.risk.entry_guard_patch import _daily_limit_block_reason


def _pm(tmp_path) -> PositionManager:
    return PositionManager(state_file=str(tmp_path / "positions.json"))


def _mgr(pm: PositionManager, limit: int = 6) -> SimpleNamespace:
    return SimpleNamespace(
        settings=SimpleNamespace(max_trades_per_day=limit, max_open_positions=0),
        position_manager=pm,
    )


def _count(pm: PositionManager, n: int) -> None:
    for _ in range(n):
        with pm._lock:
            pm._increment_trades_today_locked()


def test_position_manager_exposes_the_counter_the_guard_reads(tmp_path) -> None:
    """THE ROOT CAUSE: the attribute the guard looks up must exist."""
    pm = _pm(tmp_path)
    assert callable(getattr(pm, "trades_today", None))
    assert pm.trades_today() == 0


def test_entries_below_the_limit_are_allowed(tmp_path) -> None:
    pm = _pm(tmp_path)
    _count(pm, 5)
    assert pm.trades_today() == 5
    assert _daily_limit_block_reason(_mgr(pm)) is None


def test_sixth_trade_blocks_further_entries(tmp_path) -> None:
    """THE FIX: the cap fires at exactly the configured limit."""
    pm = _pm(tmp_path)
    _count(pm, 6)
    reason = _daily_limit_block_reason(_mgr(pm))
    assert reason is not None
    assert "MAX_TRADES:6/6" in reason[1]


def test_block_persists_beyond_the_limit(tmp_path) -> None:
    pm = _pm(tmp_path)
    _count(pm, 9)
    reason = _daily_limit_block_reason(_mgr(pm))
    assert reason is not None
    assert "9/6" in reason[0]


def test_counter_resets_on_ist_trading_date_rollover(tmp_path) -> None:
    """The cap is per trading day, not per process lifetime."""
    pm = _pm(tmp_path)
    _count(pm, 6)
    assert pm.trades_today() == 6

    pm._trades_today_date = "2000-01-01"  # simulate a previous trading date
    assert pm.trades_today() == 0
    assert _daily_limit_block_reason(_mgr(pm)) is None


def test_opening_a_position_increments_the_counter(tmp_path) -> None:
    """The counter must be driven by real entries, not only by tests.

    Behavioural rather than source-based: open_position is wrapped at runtime,
    so inspecting its source would only see the wrapper.
    """
    pm = _pm(tmp_path)
    assert pm.trades_today() == 0

    pm.open_position(
        symbol="NFO:NIFTY2680424400PE",
        side="LONG",
        quantity=65,
        entry_price=93.35,
    )

    assert pm.trades_today() == 1


@pytest.mark.parametrize("limit", [0, None])
def test_zero_or_unset_limit_remains_unlimited(tmp_path, limit) -> None:
    """0 keeps the documented 'unlimited' behaviour."""
    pm = _pm(tmp_path)
    _count(pm, 25)
    mgr = SimpleNamespace(
        settings=SimpleNamespace(max_trades_per_day=limit, max_open_positions=0),
        position_manager=pm,
    )
    assert _daily_limit_block_reason(mgr) is None
