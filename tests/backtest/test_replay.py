from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from nifty_scalper_bot.backtest.replay import (
    HistoricalContractCatalog,
    ReplayContractSnapshot,
    ReplayHarness,
)
from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine


class _DummyDataHub:
    def __init__(self) -> None:
        self.quotes: dict[str, dict[str, float]] = {}
        self.baskets: list[dict[str, object]] = []

    def get_quote(self, symbol: str, allow_pull: bool = False) -> dict[str, float]:
        return self.quotes.setdefault(symbol, {"ltp": 100.0})

    def store_quote(self, symbol: str, quote: dict[str, float], *, source: str) -> None:
        self.quotes[symbol] = dict(quote)

    def set_active_contract_basket(self, basket: dict[str, object]) -> None:
        self.baskets.append(dict(basket))


class _DummyResolver:
    def lot_size_for_symbol(self, symbol: str) -> int:  # pragma: no cover - not used
        return 75


class _TradeRecord:
    def __init__(self, action: str) -> None:
        self.action = action

    def to_dict(self) -> dict[str, str]:
        return {"action": self.action}


class _DummyRunner:
    def __init__(self) -> None:
        self.ticks: list[tuple[str, dict[str, float]]] = []
        self.baskets: list[dict[str, object]] = []

    def on_tick_event(self, tick: dict[str, float]) -> None:
        self.ticks.append((str(tick["symbol"]), tick))

    def on_replay_tick(self, symbol: str, tick: dict[str, float]) -> None:
        raise AssertionError("replay must use the production tick ingress")

    def set_active_trading_universe(self, basket: dict[str, object]) -> None:
        self.baskets.append(dict(basket))

    def get_status(self) -> dict[str, dict[str, list[_TradeRecord]]]:
        return {"symbols": {"OPT": {"trade_history": [_TradeRecord("BUY")]}}}


def _sample_frame() -> pd.DataFrame:
    index = pd.date_range(datetime(2024, 1, 1, 9, 30), periods=3, freq="T")
    return pd.DataFrame(
        {
            "option_open": [100.0, 101.0, 102.0],
            "option_high": [101.0, 102.0, 103.0],
            "option_low": [99.5, 100.5, 101.5],
            "option_close": [100.5, 101.5, 102.5],
            "option_volume": [1000, 1100, 1200],
            "option_bid": [100.4, 101.4, 102.4],
            "option_ask": [100.6, 101.6, 102.6],
            "index_open": [19500.0, 19510.0, 19520.0],
            "index_high": [19520.0, 19530.0, 19540.0],
            "index_low": [19480.0, 19490.0, 19500.0],
            "index_close": [19510.0, 19520.0, 19530.0],
            "index_volume": [5000, 5100, 5200],
        },
        index=index,
    )


def test_replay_harness_dispatches_ticks(tmp_path: Path) -> None:
    hub = _DummyDataHub()
    resolver = _DummyResolver()
    paper = PaperFillEngine(hub, resolver)
    runner = _DummyRunner()
    harness = ReplayHarness(runner, paper, option_symbol="OPT", index_symbol="IDX")

    frame = _sample_frame()
    result = harness.run_dataframe(frame)

    assert result.bars_processed == len(frame)
    assert len(runner.ticks) == len(frame) * 2
    assert [symbol for symbol, _ in runner.ticks[:2]] == ["IDX", "OPT"]
    assert runner.ticks[1][1]["last_price"] == 100.5
    assert runner.ticks[1][1]["bid"] == 100.4
    assert runner.ticks[1][1]["ask"] == 100.6
    assert runner.ticks[1][1]["source"] == "historical_replay"
    assert paper.hub.get_quote("OPT", allow_pull=False)["ltp"] == 102.5
    assert result.trades == [{"action": "BUY"}]
    assert result.orders == []

    csv_path = tmp_path / "20240101.csv"
    frame.to_csv(csv_path)
    result_from_file = harness.run_day(tmp_path, "20240101")
    assert result_from_file.bars_processed == len(frame)


class _ReplayClock:
    def __init__(self) -> None:
        self.timestamps: list[datetime] = []
        self.current: datetime | None = None

    def advance_to(self, timestamp: datetime) -> None:
        self.timestamps.append(timestamp)
        self.current = timestamp

    def time(self) -> float:
        assert self.current is not None
        return self.current.timestamp()


def test_replay_advances_clock_once_per_synchronised_timestamp() -> None:
    hub = _DummyDataHub()
    paper = PaperFillEngine(hub, _DummyResolver())
    runner = _DummyRunner()
    clock = _ReplayClock()
    harness = ReplayHarness(
        runner,
        paper,
        option_symbol="OPT",
        index_symbol="IDX",
        clock=clock,
    )

    frame = _sample_frame()
    harness.run_dataframe(frame)

    assert clock.timestamps == [timestamp.to_pydatetime() for timestamp in frame.index]
    order = paper.place_order(
        {
            "symbol": "OPT",
            "transaction_type": "BUY",
            "quantity": 1,
            "order_type": "MARKET",
        }
    )
    assert order["timestamp"] == frame.index[-1].to_pydatetime().timestamp()


def test_replay_rotates_only_time_available_historical_contracts() -> None:
    timestamps = pd.date_range(datetime(2024, 1, 1, 9, 30), periods=2, freq="T")
    frame = _sample_frame().iloc[:2].copy()
    frame.index = timestamps
    frame["option_symbol"] = ["NFO:NIFTY24JAN19500CE", "NFO:NIFTY24JAN19550CE"]
    first = {
        "selected_ce": "NFO:NIFTY24JAN19500CE",
        "selected_pe": "NFO:NIFTY24JAN19500PE",
        "option_symbols": (
            "NFO:NIFTY24JAN19500CE",
            "NFO:NIFTY24JAN19500PE",
        ),
        "option_expiry": "2024-01-04",
    }
    second = {
        "selected_ce": "NFO:NIFTY24JAN19550CE",
        "selected_pe": "NFO:NIFTY24JAN19550PE",
        "option_symbols": (
            "NFO:NIFTY24JAN19550CE",
            "NFO:NIFTY24JAN19550PE",
        ),
        "option_expiry": "2024-01-04",
    }
    catalog = HistoricalContractCatalog(
        [
            ReplayContractSnapshot(timestamps[0].to_pydatetime(), first),
            ReplayContractSnapshot(timestamps[1].to_pydatetime(), second),
        ]
    )
    hub = _DummyDataHub()
    runner = _DummyRunner()
    harness = ReplayHarness(
        runner,
        PaperFillEngine(hub, _DummyResolver()),
        option_symbol="legacy-option",
        index_symbol="IDX",
        contract_catalog=catalog,
    )

    harness.run_dataframe(frame)

    assert [symbol for symbol, _ in runner.ticks if symbol != "IDX"] == [
        "NFO:NIFTY24JAN19500CE",
        "NFO:NIFTY24JAN19550CE",
    ]
    assert [basket["selected_ce"] for basket in runner.baskets] == [
        first["selected_ce"],
        second["selected_ce"],
    ]
    assert hub.baskets == runner.baskets


def test_replay_rejects_contract_not_available_in_historical_basket() -> None:
    timestamp = datetime(2024, 1, 1, 9, 30)
    frame = _sample_frame().iloc[:1].copy()
    frame.index = pd.DatetimeIndex([timestamp])
    frame["option_symbol"] = ["NFO:NIFTY24JAN19600CE"]
    catalog = HistoricalContractCatalog(
        [
            ReplayContractSnapshot(
                timestamp,
                {
                    "selected_ce": "NFO:NIFTY24JAN19500CE",
                    "selected_pe": "NFO:NIFTY24JAN19500PE",
                    "option_symbols": (
                        "NFO:NIFTY24JAN19500CE",
                        "NFO:NIFTY24JAN19500PE",
                    ),
                    "option_expiry": "2024-01-04",
                },
            )
        ]
    )
    harness = ReplayHarness(
        _DummyRunner(),
        PaperFillEngine(_DummyDataHub(), _DummyResolver()),
        option_symbol="legacy-option",
        contract_catalog=catalog,
    )

    with pytest.raises(ValueError, match="not present in historical basket"):
        harness.run_dataframe(frame)


def test_historical_contract_catalog_rejects_lookahead_and_expired_baskets() -> None:
    catalog = HistoricalContractCatalog(
        [
            ReplayContractSnapshot(
                datetime(2024, 1, 2, 9, 30),
                {
                    "selected_ce": "NFO:NIFTY24JAN19500CE",
                    "selected_pe": "NFO:NIFTY24JAN19500PE",
                    "option_symbols": (
                        "NFO:NIFTY24JAN19500CE",
                        "NFO:NIFTY24JAN19500PE",
                    ),
                    "option_expiry": "2024-01-04",
                },
            )
        ]
    )

    with pytest.raises(LookupError, match="no contract basket"):
        catalog.resolve(datetime(2024, 1, 2, 9, 29))
    with pytest.raises(LookupError, match="expired"):
        catalog.resolve(datetime(2024, 1, 5, 9, 30))

    missing_expiry = HistoricalContractCatalog(
        [
            ReplayContractSnapshot(
                datetime(2024, 1, 2, 9, 30),
                {
                    "selected_ce": "NFO:NIFTY24JAN19500CE",
                    "selected_pe": "NFO:NIFTY24JAN19500PE",
                    "option_symbols": (
                        "NFO:NIFTY24JAN19500CE",
                        "NFO:NIFTY24JAN19500PE",
                    ),
                },
            )
        ]
    )
    with pytest.raises(ValueError, match="requires option_expiry"):
        missing_expiry.resolve(datetime(2024, 1, 2, 9, 30))
