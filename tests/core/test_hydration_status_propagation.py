from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nifty_scalper_bot.core import app


SYMBOL = "NFO:NIFTY26MAY23300CE"


def _bars(count: int) -> list[dict[str, object]]:
    start = datetime(2026, 5, 1, 9, 15, tzinfo=timezone.utc)
    return [
        {
            "symbol": SYMBOL,
            "timestamp": start + timedelta(minutes=idx),
            "open": 100 + idx,
            "high": 101 + idx,
            "low": 99 + idx,
            "close": 100.5 + idx,
            "volume": 10,
        }
        for idx in range(count)
    ]


class _Provider:
    def __init__(self, bars_by_symbol: dict[str, list[dict[str, object]]], quote: dict[str, object] | None = None) -> None:
        self._bars = bars_by_symbol
        self._quote = quote or {}
        self._token_by_symbol = {SYMBOL: 12345}

    def get_ohlc_bars(self, symbol: str, *, limit: int | None = None):
        rows = list(self._bars.get(symbol, []))
        return rows[-limit:] if limit else rows

    def get_quote(self, symbol: str, allow_pull: bool = False):
        return dict(self._quote) if symbol == SYMBOL else {}

    def resolve_symbol_token(self, symbol: str):
        return self._token_by_symbol.get(symbol)

    def symbol_data_age_ms(self, symbol: str) -> float:
        return 100.0


class _Indicator:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def get_history(self, symbol: str):
        return list(self.rows) if symbol == SYMBOL else []


class _Runner:
    def __init__(self, rows: list[dict[str, object]], indicator_rows: list[dict[str, object]]) -> None:
        self._symbol_history = {SYMBOL: list(rows)}
        self._indicator_engine = _Indicator(indicator_rows)

    def reseed_history_from_bars(self, symbol: str, rows, source: str = "test", min_bars: int = 1) -> int:
        self._symbol_history[symbol] = list(rows)
        self._indicator_engine.rows = list(rows)
        return len(list(rows))


def _quote() -> dict[str, object]:
    return {
        "bid": 100.0,
        "ask": 101.0,
        "tradable_quote": True,
        "depth_available": True,
        "depth": {"buy": [{"price": 100.0}], "sell": [{"price": 101.0}]},
        "tick_age_s": 1.0,
    }


def _ctx(mdm_rows: int, datahub_rows: int, runner_rows: int, indicator_rows: int, quote: dict[str, object] | None = None):
    return SimpleNamespace(
        market_data_manager=_Provider({SYMBOL: _bars(mdm_rows)}, quote),
        data_hub=_Provider({SYMBOL: _bars(datahub_rows)}, quote),
        strategy_runner=_Runner(_bars(runner_rows), _bars(indicator_rows)),
        active_symbol_tokens={SYMBOL: 12345},
    )


def test_mdm_30_bars_propagate_to_datahub_30_bars() -> None:
    status = app.build_symbol_hydration_status(_ctx(30, 30, 30, 30, _quote()), SYMBOL, "selected_ce", 30)

    assert status.mdm_bars == 30
    assert status.datahub_bars == 30


def test_datahub_30_bars_propagate_to_runner_30_bars() -> None:
    ctx = _ctx(30, 30, 0, 0, _quote())
    rows = ctx.data_hub.get_ohlc_bars(SYMBOL, limit=30)
    ctx.strategy_runner.reseed_history_from_bars(SYMBOL, rows, min_bars=30)

    status = app.build_symbol_hydration_status(ctx, SYMBOL, "selected_ce", 30)

    assert status.datahub_bars == 30
    assert status.runner_bars == 30


def test_runner_30_bars_propagate_to_indicator_30_bars() -> None:
    ctx = _ctx(30, 30, 0, 0, _quote())
    ctx.strategy_runner.reseed_history_from_bars(SYMBOL, ctx.data_hub.get_ohlc_bars(SYMBOL), min_bars=30)

    status = app.build_symbol_hydration_status(ctx, SYMBOL, "selected_ce", 30)

    assert status.runner_bars == 30
    assert status.indicator_bars == 30


def test_hydration_status_consistent_counts_across_all_layers() -> None:
    status = app.build_symbol_hydration_status(_ctx(30, 30, 30, 30, _quote()), SYMBOL, "selected_ce", 30)

    assert (status.mdm_bars, status.datahub_bars, status.runner_bars, status.indicator_bars) == (30, 30, 30, 30)
    assert status.ready_for_evaluation is True


def test_datahub_bars_do_not_gate_readiness() -> None:
    status = app.build_symbol_hydration_status(
        _ctx(30, 0, 30, 30, _quote()), SYMBOL, "selected_ce", 30
    )

    assert status.ready_for_evaluation is True
    assert "datahub_bars_missing" not in status.blocker_reasons


def test_readiness_true_when_all_layers_have_bars_and_quote_depth_valid() -> None:
    status = app.build_symbol_hydration_status(_ctx(30, 30, 30, 30, _quote()), SYMBOL, "selected_ce", 30)

    assert status.ready_for_evaluation is True
    assert status.ready_for_execution is True
    assert status.tradable_quote is True
    assert status.depth_available is True
