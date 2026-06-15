"""DataHub owns no history (PR #562 SSOT), so its bar count must NOT gate
readiness. Regression for the live blocker where SELECTED_OPTION_HISTORY_READINESS
reported both_ready=True while READINESS_BLOCKER_SUMMARY still showed
selected_option_history_cold because _hydration_status_map gated on a lagging
datahub_bars. Async so it executes under the repo conftest hook.
"""

from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.core.app import build_symbol_hydration_status


def _bars(n: int):
    # minimal OHLC rows; only the count matters for gating
    return [{"timestamp": i, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1} for i in range(n)]


class _Provider:
    def __init__(self, n):
        self._n = n
    def get_ohlc_bars(self, symbol, limit=None):
        return _bars(self._n)


class _Indicator:
    def __init__(self, n):
        self._n = n
    def get_history(self, symbol):
        return _bars(self._n)


def _ctx(*, mdm_n, datahub_n, runner_n, indicator_n, sym):
    mdm = _Provider(mdm_n)
    datahub = _Provider(datahub_n)
    runner = SimpleNamespace(
        _symbol_history={sym: _bars(runner_n)},
        _indicator_engine=_Indicator(indicator_n),
    )
    return SimpleNamespace(
        market_data_manager=mdm,
        data_hub=datahub,
        strategy_runner=runner,
        active_symbol_tokens={sym: 111},
    )


SYM = "NFO:NIFTY2661623850CE"


async def test_lagging_datahub_does_not_block_execution() -> None:
    # Exact live scenario: mdm/runner/indicator >= required(30), datahub lags (5).
    ctx = _ctx(mdm_n=51, datahub_n=5, runner_n=46, indicator_n=46, sym=SYM)
    status = build_symbol_hydration_status(ctx, SYM, "selected_ce", 30)
    # datahub still reported for observability...
    assert status.datahub_bars == 5
    # ...but it must NOT gate: history is ready because mdm/runner/indicator >= 30
    assert status.ready_for_evaluation is True
    assert "insufficient_bars" not in status.blocker_reasons
    assert "selected_ce_history_missing" not in status.blocker_reasons


async def test_genuine_history_shortfall_still_blocks() -> None:
    # If a REAL source (runner) is below required, it must still block.
    ctx = _ctx(mdm_n=51, datahub_n=51, runner_n=10, indicator_n=46, sym=SYM)
    status = build_symbol_hydration_status(ctx, SYM, "selected_ce", 30)
    assert status.ready_for_evaluation is False
    assert "insufficient_bars" in status.blocker_reasons
    assert "runner_bars_missing" in status.blocker_reasons
