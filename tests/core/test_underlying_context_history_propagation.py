from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nifty_scalper_bot.core import app
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


class _Indicator:
    def __init__(self, rows_by_symbol: dict[str, list[dict[str, object]]]) -> None:
        self.rows_by_symbol = rows_by_symbol

    def get_history(self, symbol: str, field: str = "bars"):
        del field
        return list(self.rows_by_symbol.get(symbol, []))


def _runner(rows: list[dict[str, object]]) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._market_data = _MDM(rows)
    runner._data_hub = None
    runner._active_futures_symbol = FUTURE
    runner._spot_symbol = SPOT
    return runner


async def test_futures_context_history_preserves_session_opening_range_when_orb_enabled(
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


async def test_spot_fallback_history_preserves_session_opening_range_when_orb_enabled(
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


async def test_context_history_never_falls_below_smc_structural_minimum(
    monkeypatch,
) -> None:
    """SMC must see its minimum underlying-bar window even when generic context is shorter."""
    monkeypatch.setenv("ORB_ENABLED", "false")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "30")
    apply_patches()
    runner = _runner(_bars(60))

    rows = runner._get_mdm_bars(FUTURE, 20)

    assert len(rows) == 30
    assert runner._market_data.requested_limits[-1] == 30


async def test_option_history_read_limit_is_unchanged(monkeypatch) -> None:
    """The context fix must not broaden unrelated option-history reads."""
    monkeypatch.setenv("ORB_ENABLED", "false")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "30")
    apply_patches()
    runner = _runner(_bars(60))

    rows = runner._get_mdm_bars("NFO:NIFTY2690823950CE", 20)

    assert len(rows) == 20
    assert runner._market_data.requested_limits[-1] == 20


async def test_orb_context_policy_targets_full_session_history(monkeypatch) -> None:
    """Startup hydration must fetch enough underlying bars to keep 09:15 through close."""
    monkeypatch.setenv("ORB_ENABLED", "true")
    monkeypatch.delenv("HYDRATION_MAX_BARS", raising=False)
    monkeypatch.delenv("HYDRATION_CAP_FUTURES_CONTEXT", raising=False)
    monkeypatch.delenv("HYDRATION_DEEP_FUTURES_CONTEXT", raising=False)
    apply_patches()
    ctx = SimpleNamespace(
        strategy_runner=SimpleNamespace(
            _context_required_bars=100,
            _required_candles=100,
        ),
        market_data_manager=SimpleNamespace(_min_required_bars=0),
    )

    policy = app.resolve_history_policy(
        ctx,
        FUTURE,
        role="futures_context",
        phase="startup",
        reason="startup_hydration",
    )

    assert policy.required_bars == 100
    assert policy.target_bars >= 400
    assert policy.role_cap >= 400
    assert policy.deep_cap >= 400


def _context_sync_runner(
    *,
    mdm_rows: list[dict[str, object]],
    indicator_rows: list[dict[str, object]],
) -> tuple[StrategyRunner, list[tuple[str, dict[str, object]]]]:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._market_data = _MDM(mdm_rows)
    runner._data_hub = None
    runner._active_futures_symbol = FUTURE
    runner._spot_symbol = SPOT
    runner._active_symbols = {FUTURE}
    runner._context_required_bars = 20
    runner._indicator_engine = _Indicator({FUTURE: indicator_rows})
    runner._active_context_symbols_for_history = lambda: [FUTURE]
    runner._history_count_for_symbol = lambda symbol: len(
        runner._indicator_engine.get_history(symbol)
    )
    runner._schedule_runtime_history_ensure = lambda *_args, **_kwargs: True
    calls: list[tuple[str, dict[str, object]]] = []

    def _sync(symbol: str, **kwargs: object) -> int:
        calls.append((symbol, dict(kwargs)))
        return len(mdm_rows)

    runner._sync_history_from_mdm_cache = _sync
    return runner, calls


async def test_context_history_syncs_when_mdm_completed_bar_advances(monkeypatch) -> None:
    """Warm-by-count context must still propagate a newly completed MDM bar."""
    monkeypatch.setenv("ORB_ENABLED", "false")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "30")
    apply_patches()
    mdm_rows = _bars(101)
    indicator_rows = _bars(100)
    runner, calls = _context_sync_runner(
        mdm_rows=mdm_rows,
        indicator_rows=indicator_rows,
    )

    runner._sync_context_history_if_cold(source="test_context_advance")

    assert calls, "new completed MDM bar must be mirrored into Runner/IndicatorEngine"
    assert calls[-1][0] == FUTURE
    assert int(calls[-1][1]["required_bars"]) >= 30


async def test_context_history_does_not_reseed_when_already_aligned(monkeypatch) -> None:
    """No-op when MDM and IndicatorEngine already expose the same completed bar."""
    monkeypatch.setenv("ORB_ENABLED", "false")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "30")
    apply_patches()
    rows = _bars(100)
    runner, calls = _context_sync_runner(mdm_rows=rows, indicator_rows=rows)

    runner._sync_context_history_if_cold(source="test_context_aligned")

    assert calls == []


async def test_orb_context_requests_structural_target_even_when_warm_by_count(
    monkeypatch,
) -> None:
    """100 warm bars are insufficient for ORB once the 09:15 range has rolled out."""
    monkeypatch.setenv("ORB_ENABLED", "true")
    monkeypatch.setenv("SMC_MIN_BARS_REQUIRED", "30")
    apply_patches()
    rows = _bars(100)
    runner, calls = _context_sync_runner(mdm_rows=rows, indicator_rows=rows)
    scheduled: list[dict[str, object]] = []
    runner._schedule_runtime_history_ensure = lambda _symbol, **kwargs: (
        scheduled.append(dict(kwargs)) or True
    )

    runner._sync_context_history_if_cold(source="test_orb_structural_target")

    assert calls == []  # aligned 100-bar state must not be needlessly reseeded
    assert scheduled
    assert int(scheduled[-1]["required_bars"]) >= 30
    assert int(scheduled[-1]["target_bars"]) >= 400
