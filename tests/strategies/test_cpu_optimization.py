"""CPU-optimization regression tests: hydration sync, eval cap, midday skip."""

from __future__ import annotations

from typing import Any

import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _bare_runner() -> StrategyRunner:
    runner = object.__new__(StrategyRunner)
    return runner


class _FakeIndicatorHistory:
    def __init__(self) -> None:
        self.bars: dict[str, list[dict[str, Any]]] = {}

    def count(self, symbol: str) -> int:
        return len(self.bars.get(symbol, []))


def test_hydration_sync_heals_cold_runner_history() -> None:
    """Regression: mdm_bars>=50 with runner/indicator bars=1 must heal to >=required."""
    runner = _bare_runner()
    store = _FakeIndicatorHistory()
    mdm_rows = [{"close": 100.0 + i, "ts": i} for i in range(51)]
    store.bars["NFO:NIFTY2661623450PE"] = [mdm_rows[0]]  # cold: 1 bar

    runner._normalize_symbol = lambda s: s  # type: ignore[attr-defined]
    runner._required_bars_for_symbol = lambda s: 50  # type: ignore[attr-defined]
    runner._history_count_for_symbol = lambda s: store.count(s)  # type: ignore[attr-defined]
    runner._symbol_history = {"NFO:NIFTY2661623450PE": [mdm_rows[0]]}  # type: ignore[attr-defined]
    runner._get_mdm_bars = lambda s, n: mdm_rows[-n:]  # type: ignore[attr-defined]
    runner._emit_history_hydration_trace = lambda *a, **k: None  # type: ignore[attr-defined]
    runner._request_mdm_hydration = lambda *a, **k: None  # type: ignore[attr-defined]

    def _reseed(symbol: str, rows: list[dict[str, Any]], **_: Any) -> int:
        store.bars[symbol] = list(rows)
        return len(rows)

    runner.reseed_history_from_bars = _reseed  # type: ignore[attr-defined]

    after = StrategyRunner._sync_history_from_mdm_cache(
        runner, "NFO:NIFTY2661623450PE", required_bars=50, source="test", request_if_short=False
    )
    assert after >= 50
    assert store.count("NFO:NIFTY2661623450PE") >= 50


def test_eval_option_whitelist_caps_to_nearest_strikes(monkeypatch) -> None:
    monkeypatch.setenv("MAX_LIVE_OPTION_SYMBOLS", "4")
    runner = _bare_runner()

    def _strike(sym: str) -> int | None:
        digits = "".join(ch for ch in sym if ch.isdigit())
        return int(digits[-5:]) if len(digits) >= 5 else None

    runner._extract_strike_from_symbol = _strike  # type: ignore[attr-defined]
    options = {
        "NFO:NIFTY2661623150CE", "NFO:NIFTY2661623200CE", "NFO:NIFTY2661623250CE",
        "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE", "NFO:NIFTY2661623350PE",
        "NFO:NIFTY2661623450PE",
    }
    whitelist = StrategyRunner._compute_eval_option_whitelist(
        runner, options, 23300, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE"
    )
    assert len(whitelist) == 4
    assert "NFO:NIFTY2661623300CE" in whitelist and "NFO:NIFTY2661623300PE" in whitelist
    assert "NFO:NIFTY2661623450PE" not in whitelist and "NFO:NIFTY2661623150CE" not in whitelist


def test_whitelist_uncapped_when_under_limit(monkeypatch) -> None:
    monkeypatch.setenv("MAX_LIVE_OPTION_SYMBOLS", "8")
    runner = _bare_runner()
    runner._extract_strike_from_symbol = lambda s: 23300  # type: ignore[attr-defined]
    options = {"NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE"}
    whitelist = StrategyRunner._compute_eval_option_whitelist(
        runner, options, 23300, "NFO:NIFTY2661623300CE", "NFO:NIFTY2661623300PE"
    )
    assert whitelist == options


class _FakePositionManager:
    def __init__(self, open_positions: list[Any]) -> None:
        self._open = open_positions

    def get_open_positions(self) -> list[Any]:
        return self._open


def test_midday_idle_skip_only_when_paused_and_flat(monkeypatch) -> None:
    runner = _bare_runner()
    runner._position_manager = _FakePositionManager([])  # type: ignore[attr-defined]
    monkeypatch.setenv("MIDDAY_PAUSE_ENABLED", "false")
    assert StrategyRunner._midday_idle_skip_active(runner) is False
    monkeypatch.setenv("MIDDAY_PAUSE_ENABLED", "true")
    import nifty_scalper_bot.strategies.runner as runner_mod

    monkeypatch.setattr(runner_mod, "midday_pause_block", lambda: (True, "midday_pause"))
    assert StrategyRunner._midday_idle_skip_active(runner) is True
    runner._position_manager = _FakePositionManager([object()])  # type: ignore[attr-defined]
    assert StrategyRunner._midday_idle_skip_active(runner) is False
