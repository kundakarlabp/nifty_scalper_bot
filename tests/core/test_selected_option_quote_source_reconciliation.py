from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.core.history_readiness import (
    _get_cached_quote,
    _merge_cached_quote_candidates,
    build_symbol_hydration_status,
)


SYMBOL = "NFO:NIFTY26JUN24050CE"
TOKEN = 20410882


def _bars(count: int) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": f"2026-06-26T0{index % 9}:00:00+00:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 100,
        }
        for index in range(count)
    ]


class _DataHub:
    def __init__(self, quote: dict[str, Any], bars: list[dict[str, Any]]) -> None:
        self.quote = dict(quote)
        self.bars = list(bars)
        self._token_by_symbol = {SYMBOL: TOKEN}
        self.allow_pull_values: list[bool] = []

    def get_quote(self, symbol: str, allow_pull: bool = True) -> dict[str, Any]:
        assert symbol == SYMBOL
        self.allow_pull_values.append(bool(allow_pull))
        return dict(self.quote)

    def get_ohlc_bars(self, symbol: str, limit: int = 500) -> list[dict[str, Any]]:
        assert symbol == SYMBOL
        return list(self.bars[-limit:])


@dataclass
class _Snapshot:
    ltp: float = 101.0
    bid: float | None = None
    ask: float | None = None
    spread_pct: float | None = None
    depth_available: bool = True
    depth: Any = None
    tradable_quote: bool = False
    tick_age_s: float = 0.2


class _MDM:
    def __init__(self, quote: dict[str, Any], bars: list[dict[str, Any]]) -> None:
        self.quote = dict(quote)
        self.bars = list(bars)
        self._token_by_symbol = {SYMBOL: TOKEN}
        self._symbol_to_token = {SYMBOL: TOKEN}
        self._latest_ticks = {SYMBOL: dict(quote)}
        self._tick_cache = {SYMBOL: dict(quote)}

    def get_latest_tick(self, symbol: str) -> dict[str, Any]:
        assert symbol == SYMBOL
        return dict(self.quote)

    def get_last_tick(self, symbol: str) -> dict[str, Any]:
        assert symbol == SYMBOL
        return dict(self.quote)

    def get_symbol_snapshot(self, symbol: str) -> _Snapshot:
        assert symbol == SYMBOL
        return _Snapshot()

    def get_ohlc_bars(self, symbol: str, limit: int = 500) -> list[dict[str, Any]]:
        assert symbol == SYMBOL
        return list(self.bars[-limit:])

    def symbol_data_age_ms(self, symbol: str) -> float:
        assert symbol == SYMBOL
        return 200.0


class _Indicator:
    def __init__(self, bars: list[dict[str, Any]]) -> None:
        self.bars = list(bars)

    def get_history(self, symbol: str) -> list[dict[str, Any]]:
        assert symbol == SYMBOL
        return list(self.bars)


class _Runner:
    def __init__(self, bars: list[dict[str, Any]]) -> None:
        self._symbol_history = {SYMBOL: list(bars)}
        self._indicator_engine = _Indicator(bars)


def _ctx(datahub_quote: dict[str, Any], mdm_quote: dict[str, Any]) -> SimpleNamespace:
    bars = _bars(40)
    datahub = _DataHub(datahub_quote, bars)
    mdm = _MDM(mdm_quote, bars)
    return SimpleNamespace(
        data_hub=datahub,
        datahub=datahub,
        market_data_manager=mdm,
        strategy_runner=_Runner(bars),
        active_symbol_tokens={SYMBOL: TOKEN},
    )


def test_cached_quote_uses_richer_mdm_depth_when_datahub_projection_is_incomplete() -> None:
    ctx = _ctx(
        {
            "symbol": SYMBOL,
            "ltp": 101.0,
            "depth_available": True,
            "tradable_quote": False,
            "tick_age_s": 0.2,
        },
        {
            "symbol": SYMBOL,
            "instrument_token": TOKEN,
            "ltp": 101.0,
            "timestamp_ms": 1_782_443_000_000,
            "depth": {
                "buy": [{"price": 100.9, "quantity": 75}],
                "sell": [{"price": 101.1, "quantity": 50}],
            },
            "source": "ws_full",
        },
    )

    quote = _get_cached_quote(ctx, SYMBOL)

    assert quote["bid"] == 100.9
    assert quote["ask"] == 101.1
    assert quote["tradable_quote"] is True
    assert quote["depth_available"] is True
    assert "mdm.get_latest_tick" in quote["quote_cache_sources"]
    assert ctx.data_hub.allow_pull_values == [False]


def test_hydration_status_becomes_execution_ready_from_actual_two_sided_depth() -> None:
    ctx = _ctx(
        {
            "symbol": SYMBOL,
            "ltp": 101.0,
            "depth_available": True,
            "tradable_quote": False,
        },
        {
            "symbol": SYMBOL,
            "instrument_token": TOKEN,
            "ltp": 101.0,
            "depth": {
                "buy": [{"price": 100.9, "quantity": 75}],
                "sell": [{"price": 101.1, "quantity": 50}],
            },
            "source": "ws_full",
        },
    )

    status = build_symbol_hydration_status(ctx, SYMBOL, "selected_ce", 30)

    assert status.mdm_bars == 40
    assert status.runner_bars == 40
    assert status.indicator_bars == 40
    assert status.live_tick_fresh is True
    assert status.bid == 100.9
    assert status.ask == 101.1
    assert status.tradable_quote is True
    assert status.depth_available is True
    assert status.ready_for_evaluation is True
    assert status.ready_for_execution is True
    assert "selected_ce_quote_missing" not in status.blocker_reasons


def test_boolean_depth_flag_without_bid_ask_remains_fail_closed() -> None:
    ctx = _ctx(
        {
            "symbol": SYMBOL,
            "ltp": 101.0,
            "depth_available": True,
            "tradable_quote": False,
        },
        {
            "symbol": SYMBOL,
            "instrument_token": TOKEN,
            "ltp": 101.0,
            "depth_available": True,
            "tradable_quote": False,
            "source": "ws",
        },
    )

    status = build_symbol_hydration_status(ctx, SYMBOL, "selected_ce", 30)

    assert status.tradable_quote is False
    assert status.ready_for_execution is False
    assert "selected_ce_quote_missing" in status.blocker_reasons


def test_one_sided_depth_does_not_manufacture_tradable_quote() -> None:
    quote = _merge_cached_quote_candidates(
        SYMBOL,
        [
            (
                "mdm",
                {
                    "symbol": SYMBOL,
                    "ltp": 101.0,
                    "depth": {
                        "buy": [{"price": 100.9, "quantity": 75}],
                        "sell": [],
                    },
                    "depth_available": True,
                },
            )
        ],
    )

    assert quote["tradable_quote"] is False
    assert quote["depth_available"] is False
    assert quote.get("bid") is None or quote.get("ask") is None
