from __future__ import annotations

import datetime as dt
import logging
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping

import pytest
import pandas as pd

from src.config import settings
from src.data.broker_source import BrokerDataSource
from src.data.base_source import BaseDataSource
from src.data.market_data import get_best_ask
from src.data.types import HistResult, HistStatus
from src.features.health import check
from src.features.range import range_score
from src.features.indicators import atr_pct
from src.options.instruments_cache import (
    InstrumentsCache,
    nearest_weekly_expiry,
    _safe_int,
)
from src.execution.order_manager import OrderManager
from src.risk.risk_gates import AccountState, evaluate as evaluate_gates
from src.risk.risk_manager import RiskManager
import src.strategies.scalper as scalper_module
from src.strategies.scalper import ScalperStrategy, StaleMarketDataError
from src.utils.helpers import get_next_thursday, get_weekly_expiry


@pytest.fixture(autouse=True)
def _allow_offhours(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "allow_offhours_testing", True)
    monkeypatch.setattr(settings, "trade_window_start", "00:00")
    monkeypatch.setattr(settings, "trade_window_end", "23:59")
    monkeypatch.setattr(settings, "max_data_staleness_ms", 30_000)


def test_get_weekly_expiry_same_day_before_cutoff() -> None:
    now = dt.datetime(2024, 6, 6, 10, 0)
    assert get_weekly_expiry(now=now) == "240606"


def test_get_weekly_expiry_rolls_forward_after_cutoff() -> None:
    now = dt.datetime(2024, 6, 6, 16, 0)
    assert get_weekly_expiry(now=now) == "240613"


def test_get_next_thursday_rolls_from_weekend() -> None:
    saturday = dt.datetime(2024, 6, 8)
    expected = dt.datetime(2024, 6, 13).strftime("%y%m%d")
    assert get_next_thursday(saturday) == expected


def test_get_best_ask_prefers_depth() -> None:
    def fetcher(symbol: str) -> Mapping[str, object]:
        assert symbol == "NFO:NIFTY24"
        return {
            "ask": 110.0,
            "depth": {"sell": [{"price": 99.5, "quantity": 50}]},
        }

    assert get_best_ask("NFO:NIFTY24", depth_fetcher=fetcher) == pytest.approx(99.55)


def test_get_best_ask_falls_back_to_ask() -> None:
    payload = {
        "ask": "102.5",
        "depth": {"sell": [{"price": "bad"}, "oops", {"quantity": 10}]},
    }

    assert get_best_ask("NFO:BANKNIFTY", depth_fetcher=lambda _: payload) == pytest.approx(
        102.55
    )


def test_get_best_ask_raises_when_missing() -> None:
    with pytest.raises(ValueError) as exc:
        get_best_ask("NFO:FOO", depth_fetcher=lambda _s: {})
    assert "depth_missing" in str(exc.value)


def test_get_best_ask_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError) as exc:
        get_best_ask("NFO:FOO", depth_fetcher=lambda _s: None)
    assert "quote_invalid" in str(exc.value)


def test_get_best_ask_requires_symbol() -> None:
    with pytest.raises(ValueError):
        get_best_ask("", depth_fetcher=lambda _s: {})


def test_get_best_ask_handles_mapping_sell_levels() -> None:
    payload = {
        "depth": {"sell": {"a": {"price": 101.5}, "b": {"price": 103.0}}},
    }

    ask = get_best_ask("NFO:NIFTYX", depth_fetcher=lambda _: payload)
    assert ask == pytest.approx(101.55)


def test_get_best_ask_handles_invalid_ask_type() -> None:
    with pytest.raises(ValueError) as exc:
        get_best_ask("NFO:BADASK", depth_fetcher=lambda _s: {"ask": object()})
    assert "ask_invalid" in str(exc.value)


def test_get_best_ask_handles_non_numeric_string() -> None:
    with pytest.raises(ValueError) as exc:
        get_best_ask("NFO:BADASK2", depth_fetcher=lambda _s: {"ask": "oops"})
    assert "ask_invalid" in str(exc.value)


def test_get_best_ask_falls_back_to_ltp_when_depth_missing() -> None:
    payload = {"last_price": 210.5, "tick_size": 0.05}
    price = get_best_ask("NFO:NIFTY", depth_fetcher=lambda _s: payload)
    assert price == pytest.approx(210.55)


def test_get_best_ask_logs_fallback_reason(caplog: pytest.LogCaptureFixture) -> None:
    payload = {
        "last_price": 150.25,
        "depth": {"sell": []},
        "tick_size": 0.05,
    }

    with caplog.at_level(logging.WARNING):
        price = get_best_ask("NFO:NIFTY24", depth_fetcher=lambda _s: payload)

    assert price == pytest.approx(150.3)
    assert any("sell_empty" in record.message for record in caplog.records)


def test_order_manager_confirmation_success() -> None:
    placed: list[Dict[str, Any]] = []
    statuses: Iterator[str] = iter(["OPEN", "COMPLETE"])

    def place(payload: Mapping[str, Any]) -> str:
        placed.append(dict(payload))
        return "OID1"

    def status(_: str) -> Mapping[str, Any]:
        return {"status": next(statuses)}

    cancelled: list[str] = []
    manager = OrderManager(
        place,
        status_fetcher=status,
        cancel_order=cancelled.append,
        poll_interval=0.0,
    )
    order_id = manager.place_order_with_confirmation({"symbol": "NFO:TEST"})
    assert order_id == "OID1"
    assert cancelled == []
    assert placed[0]["symbol"] == "NFO:TEST"


def test_order_manager_returns_immediately_without_status_fetcher() -> None:
    def place(_: Mapping[str, Any]) -> str:
        return "OID100"

    manager = OrderManager(place, poll_interval=0.0)
    assert (
        manager.place_order_with_confirmation({"symbol": "NFO:NO_STATUS"})
        == "OID100"
    )


def test_order_manager_confirmation_timeout_triggers_cancel() -> None:
    def place(_: Mapping[str, Any]) -> str:
        return "OID42"

    def status(_: str) -> str:
        return "OPEN"

    cancelled: list[str] = []
    manager = OrderManager(
        place,
        status_fetcher=status,
        cancel_order=cancelled.append,
        poll_interval=0.0,
    )
    assert (
        manager.place_order_with_confirmation({"symbol": "NFO:FAIL"}, max_wait_sec=0.0)
        is None
    )
    assert cancelled == ["OID42"]


def test_order_manager_cancelled_status_aborts() -> None:
    def place(_: Mapping[str, Any]) -> str:
        return "OID_CANCEL"

    statuses = iter([{"state": "OPEN"}, {"order_status": "cancelled"}])

    def status(_: str) -> Mapping[str, Any]:
        return next(statuses)

    manager = OrderManager(place, status_fetcher=status, poll_interval=0.0)
    assert (
        manager.place_order_with_confirmation({"symbol": "NFO:CANCEL"}, max_wait_sec=1.0)
        is None
    )


def test_order_manager_square_off_fallback_places_market_order() -> None:
    placed: list[Dict[str, Any]] = []

    def place(payload: Dict[str, Any]) -> str:
        placed.append(dict(payload))
        return "SO1"

    manager = OrderManager(place, poll_interval=0.0)
    manager.square_off_position("NIFTY24CE", side="SELL", quantity=25)

    assert placed == [
        {
            "symbol": "NIFTY24CE",
            "transaction_type": "SELL",
            "order_type": "MARKET",
            "quantity": 25,
        }
    ]


def test_order_manager_square_off_uses_callback_when_available() -> None:
    calls: list[tuple[str, str, int | None]] = []

    def place(_: Dict[str, Any]) -> str:
        return "IGNORED"

    def custom_square_off(symbol: str, side: str, qty: int | None) -> None:
        calls.append((symbol, side, qty))

    manager = OrderManager(
        place,
        square_off=custom_square_off,
        poll_interval=0.0,
    )
    manager.square_off_position("NIFTY24PE", side="SELL", quantity=None)
    assert calls == [("NIFTY24PE", "SELL", None)]


def test_order_manager_place_straddle_handles_partial_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    order_ids = iter(["CE1", "PE1"])
    square_calls: list[tuple[str, str, int | None]] = []

    def place(_: MutableMapping[str, Any]) -> str:
        return next(order_ids)

    manager = OrderManager(
        place,
        poll_interval=0.0,
        square_off=lambda symbol, side, qty: square_calls.append((symbol, side, qty)),
    )

    monkeypatch.setattr(
        manager,
        "_confirm_order",
        lambda oid, timeout=15: {"status": "COMPLETE" if oid == "CE1" else "REJECTED", "order_id": oid},
    )

    ok = manager.place_straddle_orders(
        {"symbol": "NIFTYCE", "transaction_type": "BUY", "quantity": 25},
        {"symbol": "NIFTYPE", "transaction_type": "BUY", "quantity": 25},
    )

    assert not ok
    assert square_calls == [("NIFTYCE", "SELL", 25)]


def test_order_manager_place_straddle_success(monkeypatch: pytest.MonkeyPatch) -> None:
    order_ids = iter(["CE2", "PE2"])

    def place(_: MutableMapping[str, Any]) -> str:
        return next(order_ids)

    manager = OrderManager(place, poll_interval=0.0)

    monkeypatch.setattr(
        manager,
        "_confirm_order",
        lambda oid, timeout=15: {"status": "COMPLETE", "order_id": oid},
    )

    ok = manager.place_straddle_orders(
        {"symbol": "NIFTYCE", "transaction_type": "BUY", "quantity": 25},
        {"symbol": "NIFTYPE", "transaction_type": "BUY", "quantity": 25},
    )

    assert ok


def test_order_manager_square_off_requires_symbol() -> None:
    manager = OrderManager(lambda payload: "OK", poll_interval=0.0)
    with pytest.raises(ValueError):
        manager.square_off_position("", side="BUY")


class DummyOrderManager:
    def __init__(self, results: Iterable[str | None], *, square_off_ok: bool = True) -> None:
        self._results = iter(results)
        self.orders: list[Dict[str, Any]] = []
        self.square_off_calls: list[Dict[str, Any]] = []
        self._square_off_ok = square_off_ok

    def place_order_with_confirmation(
        self, params: Mapping[str, Any], *, max_wait_sec: float = 10.0
    ) -> str | None:
        self.orders.append(dict(params))
        return next(self._results, None)

    def square_off_position(
        self, symbol: str, *, side: str, quantity: int | None = None
    ) -> None:
        if not self._square_off_ok:
            raise AssertionError("square off should not be called")
        self.square_off_calls.append({"symbol": symbol, "side": side, "quantity": quantity})


def _make_strategy(
    order_results: Iterable[str | None],
    *,
    prices: Mapping[str, float],
    tick: float = 0.05,
    market_data: Any | None = None,
) -> ScalperStrategy:
    def fetch_price(symbol: str) -> float:
        return prices[symbol]

    return ScalperStrategy(
        order_manager=DummyOrderManager(order_results),  # type: ignore[arg-type]
        risk_manager=RiskManager(),
        price_fetcher=fetch_price,
        expiry_resolver=lambda: "240620",
        underlying="NIFTY",
        tick_size=tick,
        market_data=market_data,
    )


def test_scalper_strategy_complete_flow() -> None:
    prices = {
        "NFO:NIFTY24062020000CE": 101.23,
        "NFO:NIFTY24062020000PE": 98.77,
    }
    strategy = _make_strategy(["CE123", "PE456"], prices=prices)

    result = strategy.trade_straddle(20000, quantity=50, atr=10.0, side="BUY")

    assert result["status"] == "complete"
    assert result["ce_order_id"] == "CE123"
    assert result["pe_order_id"] == "PE456"


def test_scalper_strategy_partial_fill_ce_squares_off() -> None:
    prices = {
        "NFO:NIFTY24062020000CE": 101.23,
        "NFO:NIFTY24062020000PE": 98.77,
    }
    manager = DummyOrderManager(["CE123", None])
    strategy = ScalperStrategy(
        order_manager=manager,  # type: ignore[arg-type]
        risk_manager=RiskManager(),
        price_fetcher=lambda symbol: prices[symbol],
        expiry_resolver=lambda: "240620",
    )

    result = strategy.trade_straddle(20000, quantity=50, atr=10.0, side="BUY")

    assert result["status"] == "partial_ce"
    assert manager.square_off_calls == [
        {"symbol": "NIFTY24062020000CE", "side": "SELL", "quantity": 50}
    ]


def test_execute_trade_skips_when_market_data_stale() -> None:
    prices = {
        "NFO:NIFTY24062020000CE": 101.23,
        "NFO:NIFTY24062020000PE": 98.77,
    }
    strategy = _make_strategy(["CE123", "PE456"], prices=prices)
    strategy.market_data = SimpleNamespace(last_tick_age_ms=45_000)

    result = strategy.execute_trade(20000, quantity=50, atr=10.0, side="BUY")

    assert result == {
        "status": "skipped",
        "reason": "data_stale",
        "last_tick_age_ms": 45_000,
    }
    assert strategy.order_manager.orders == []  # type: ignore[attr-defined]


def test_execute_trade_skips_outside_market_hours(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = {
        "NFO:NIFTY24062020000CE": 101.23,
        "NFO:NIFTY24062020000PE": 98.77,
    }
    strategy = _make_strategy(["CE", "PE"], prices=prices)
    monkeypatch.setattr(settings, "allow_offhours_testing", False)
    monkeypatch.setattr(scalper_module, "_is_market_hours", lambda now=None: False)

    result = strategy.execute_trade(20000, quantity=50, atr=10.0, side="BUY")

    assert result == {"status": "skipped", "reason": "off_hours"}


def test_execute_trade_delegates_when_market_data_fresh() -> None:
    prices = {
        "NFO:NIFTY24062020000CE": 101.23,
        "NFO:NIFTY24062020000PE": 98.77,
    }
    strategy = _make_strategy(["CE321", "PE654"], prices=prices)
    strategy.market_data = SimpleNamespace(last_tick_age_ms=500)

    result = strategy.execute_trade(20000, quantity=50, atr=10.0, side="BUY")

    assert result["status"] == "complete"
    assert result["ce_order_id"] == "CE321"
    assert result["pe_order_id"] == "PE654"


def test_scalper_strategy_partial_fill_pe_squares_off() -> None:
    prices = {
        "NFO:NIFTY24062020000CE": 101.23,
        "NFO:NIFTY24062020000PE": 98.77,
    }
    manager = DummyOrderManager([None, "PE999"])
    strategy = ScalperStrategy(
        order_manager=manager,  # type: ignore[arg-type]
        risk_manager=RiskManager(),
        price_fetcher=lambda symbol: prices[symbol],
        expiry_resolver=lambda: "240620",
    )

    result = strategy.trade_straddle(20000, quantity=25, atr=5.0, side="BUY")

    assert result["status"] == "partial_pe"
    assert manager.square_off_calls == [
        {"symbol": "NIFTY24062020000PE", "side": "SELL", "quantity": 25}
    ]


def test_scalper_strategy_failure_status() -> None:
    prices = {
        "NFO:NIFTY24062020000CE": 101.23,
        "NFO:NIFTY24062020000PE": 98.77,
    }
    manager = DummyOrderManager([None, None])
    strategy = ScalperStrategy(
        order_manager=manager,  # type: ignore[arg-type]
        risk_manager=RiskManager(),
        price_fetcher=lambda symbol: prices[symbol],
        expiry_resolver=lambda: "240620",
    )

    result = strategy.trade_straddle(20000, quantity=10, atr=3.0, side="SELL")

    assert result["status"] == "failed"
    assert manager.square_off_calls == []


def test_scalper_strategy_requires_positive_quantity() -> None:
    strategy = _make_strategy([], prices={})
    with pytest.raises(ValueError):
        strategy.trade_straddle(20000, quantity=0, atr=1.0)


def test_scalper_strategy_rejects_stale_market_data() -> None:
    class DataStub(BaseDataSource):
        pass

    stale_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=90)
    data = DataStub()
    data._last_tick_ts = stale_time

    prices = {"NFO:NIFTY24062020000CE": 101.23, "NFO:NIFTY24062020000PE": 98.77}
    strategy = _make_strategy(["CE", "PE"], prices=prices, market_data=data)

    with pytest.raises(StaleMarketDataError):
        strategy.trade_straddle(20000, quantity=50, atr=10.0)


def test_scalper_strategy_accepts_fresh_market_data() -> None:
    class DataStub(BaseDataSource):
        pass

    fresh_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=5)
    data = DataStub()
    data._last_tick_ts = fresh_time

    prices = {"NFO:NIFTY24062020000CE": 101.23, "NFO:NIFTY24062020000PE": 98.77}
    strategy = _make_strategy(["CE123", "PE456"], prices=prices, market_data=data)

    result = strategy.trade_straddle(20000, quantity=50, atr=10.0)
    assert result["status"] == "complete"


def test_scalper_strategy_fetch_price_handles_tick() -> None:
    prices = {"NFO:NIFTY24062020000CE": 100.07}
    strategy = _make_strategy([], prices=prices, tick=0.0)
    assert strategy._fetch_price("NIFTY24062020000CE") == pytest.approx(100.07)


def test_scalper_strategy_fetch_price_validates_positive() -> None:
    prices = {"NFO:NIFTY24062020000CE": 0.0}
    strategy = _make_strategy([], prices=prices)
    with pytest.raises(ValueError):
        strategy._fetch_price("NIFTY24062020000CE")


def test_risk_manager_stop_loss_bounds() -> None:
    risk = RiskManager()
    assert risk.calculate_stop_loss(100.0, 2.0, side="LONG") == pytest.approx(97.0)
    assert risk.calculate_stop_loss(100.0, 2.0, side="SHORT") == pytest.approx(103.0)


def test_risk_manager_validates_inputs() -> None:
    risk = RiskManager()
    with pytest.raises(ValueError):
        risk.calculate_stop_loss(0.0, 1.0)
    with pytest.raises(ValueError):
        risk.calculate_stop_loss(100.0, -1.0)


def test_range_score_prefers_low_momentum() -> None:
    class Features:
        mom_norm = 0.2
        atr_pct = 0.08

    score = range_score(Features())
    assert score == pytest.approx((1 - abs(0.2)) * 1.0)


def test_range_score_handles_extremes() -> None:
    class Features:
        mom_norm = -2.0
        atr_pct = 0.5

    score = range_score(Features())
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.0)


def test_base_data_source_accessors() -> None:
    class Source(BaseDataSource):
        _last_tick_ts = dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
        _last_bar_open_ts = dt.datetime(2024, 6, 1, 9, tzinfo=dt.timezone.utc)
        _tf_seconds = 300

    src = Source()
    assert src.last_tick_dt() == dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
    assert src.last_bar_open_ts() == dt.datetime(2024, 6, 1, 9, tzinfo=dt.timezone.utc)
    assert src.timeframe_seconds == 300


def test_base_data_source_defaults() -> None:
    class Source(BaseDataSource):
        pass

    src = Source()
    assert src.last_tick_dt() is None
    assert src.last_bar_open_ts() is None
    assert src.timeframe_seconds == 60


def test_hist_result_bool_behaviour() -> None:
    ok = HistResult(HistStatus.OK, pd.DataFrame())
    assert ok
    missing = HistResult(HistStatus.NO_DATA, pd.DataFrame(), reason="gap")
    assert not missing


def test_feature_health_success() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    data = pd.DataFrame({"close": range(20)})
    health = check(data, now, atr_period=14, max_age_s=300)
    assert health.bars_ok and health.atr_ok and health.fresh_ok
    assert health.reasons == []


def test_feature_health_detects_issues() -> None:
    old = dt.datetime(2024, 1, 1, 9, 15)
    data = pd.DataFrame({"close": range(5)})
    health = check(data, old, atr_period=10, max_age_s=30)
    assert not health.bars_ok
    assert not health.fresh_ok
    assert "bars_short" in health.reasons
    assert "data_stale" in health.reasons


def test_atr_pct_returns_percentage() -> None:
    ohlc = pd.DataFrame(
        {
            "high": [101, 102, 103, 104, 105],
            "low": [99, 99.5, 100, 101, 102],
            "close": [100, 101, 102, 103, 104],
        }
    )
    result = atr_pct(ohlc, period=3)
    assert result is not None and result > 0


def test_atr_pct_requires_sufficient_bars() -> None:
    ohlc = pd.DataFrame({"high": [100], "low": [99], "close": [99.5]})
    assert atr_pct(ohlc, period=2) is None


def test_atr_pct_handles_non_positive_close() -> None:
    ohlc = pd.DataFrame(
        {
            "high": [101, 102, 103, 104],
            "low": [99, 100, 101, 102],
            "close": [100, 99, 0.0, 0.0],
        }
    )
    assert atr_pct(ohlc, period=2) is None


class DummyBroker:
    def __init__(self) -> None:
        self.connected = False
        self.subscribe_calls: List[List[int]] = []
        self.handlers: List[Any] = []
        self.disconnect_called = False
        self.reconnect_cb = None
        self.disconnect_cb = None

    def connect(self) -> None:
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def subscribe_ticks(self, instruments: List[int], on_tick: Any) -> None:
        self.subscribe_calls.append(list(instruments))
        self.handlers.append(on_tick)

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_called = True

    def on_reconnect(self, cb: Any) -> None:
        self.reconnect_cb = cb

    def on_disconnect(self, cb: Any) -> None:
        self.disconnect_cb = cb


def test_broker_data_source_flow() -> None:
    broker = DummyBroker()
    source = BrokerDataSource(broker)
    ticks: List[Any] = []
    source.set_tick_callback(lambda tick: ticks.append(tick))

    source.subscribe([1, 2])
    assert broker.subscribe_calls[-1] == [1, 2]
    assert broker.handlers[-1] is not None

    source.start()
    assert broker.connected
    assert callable(broker.reconnect_cb)
    broker.reconnect_cb()
    assert broker.subscribe_calls[-1] == [1, 2]

    handler = broker.handlers[-1]
    handler({"ltp": 123})
    assert ticks[-1] == {"ltp": 123}

    source.stop()
    assert broker.disconnect_called


def test_broker_data_source_requires_callback() -> None:
    broker = DummyBroker()
    source = BrokerDataSource(broker)
    with pytest.raises(RuntimeError):
        source.subscribe([1])


def test_broker_data_source_handle_tick_guards() -> None:
    broker = DummyBroker()
    source = BrokerDataSource(broker)
    # Without callback the handler should no-op
    source._handle_tick({"ltp": 1})

    captured: list[Any] = []
    source.set_tick_callback(lambda tick: captured.append(tick) or (_ for _ in ()).throw(RuntimeError("boom")))
    source._handle_tick({"ltp": 2})
    assert captured == [{"ltp": 2}]


def test_risk_gates_flags_reasons() -> None:
    acct = AccountState(equity_rupees=100000, dd_rupees=5000, max_daily_loss=4000, loss_streak=3)

    class Cfg:
        rr_threshold = 1.5

        class risk:
            max_consec_losses = 2

    ok, reasons = evaluate_gates({"rr": 1.0}, acct, Cfg())
    assert not ok
    assert set(reasons) == {"daily_dd", "loss_streak", "rr_low"}


def test_risk_gates_pass_when_clear() -> None:
    acct = AccountState(equity_rupees=100000, dd_rupees=1000, max_daily_loss=4000, loss_streak=1)

    class Cfg:
        rr_threshold = 1.0

        class risk:
            max_consec_losses = 3

    ok, reasons = evaluate_gates({"rr": 2.0}, acct, Cfg())
    assert ok
    assert reasons == []


def test_interfaces_imports_protocols() -> None:
    from src import interfaces

    assert hasattr(interfaces, "DataSource")
    assert interfaces.DataSource.__module__ == "src.interfaces"


def test_server_main_invokes_components(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.server.main as server_main

    calls: list[str] = []
    monkeypatch.setattr(server_main, "setup_root_logger", lambda: calls.append("setup"))
    monkeypatch.setattr(server_main, "run_health_server", lambda: calls.append("run"))

    server_main.main()
    assert calls == ["setup", "run"]


def test_safe_int_parses_values() -> None:
    assert _safe_int("10") == 10
    assert _safe_int(None) is None
    assert _safe_int("bad") is None
    assert _safe_int(float("nan")) is None


def test_instruments_cache_populates_entries() -> None:
    data = [
        {
            "name": "NIFTY",
            "expiry": dt.datetime(2024, 6, 13, 15, 30),
            "strike": 20000,
            "instrument_type": "CE",
            "instrument_token": 123456,
            "tradingsymbol": "NIFTY24JUN20000CE",
            "lot_size": 50,
        }
    ]
    cache = InstrumentsCache(instruments=data)
    meta = cache.get("nifty", "2024-06-13", 20000, "ce")
    assert meta == {
        "token": 123456,
        "tradingsymbol": "NIFTY24JUN20000CE",
        "lot_size": 50,
    }


def test_instruments_cache_skips_invalid_entries() -> None:
    data = [
        {"name": "", "expiry": "", "strike": "bad", "instrument_type": "CE"},
        {
            "name": "BANKNIFTY",
            "expiry": dt.date(2024, 6, 20),
            "strike": 45000,
            "instrument_type": "PE",
            "instrument_token": None,
            "tradingsymbol": "BANKNIFTY24JUN45000PE",
            "lot_size": 25,
        },
        {
            "name": "BANKNIFTY",
            "expiry": dt.date(2024, 6, 20),
            "strike": 45000,
            "instrument_type": "CE",
            "instrument_token": 555,
            "tradingsymbol": "BANKNIFTY24JUN45000CE",
            "lot_size": None,
        },
    ]
    cache = InstrumentsCache(instruments=data)
    assert cache.get("BANKNIFTY", "2024-06-20", 45000, "PE") is None


def test_nearest_weekly_expiry_rolls_forward() -> None:
    now = dt.datetime(2024, 6, 11, 16, 0, tzinfo=dt.timezone.utc)
    expiry = nearest_weekly_expiry(now)
    assert expiry >= now.date().isoformat()


def test_instruments_cache_fetches_from_kite() -> None:
    class KiteStub:
        def __init__(self, data: list[dict], *, fail: bool = False) -> None:
            self.data = data
            self.fail = fail
            self.calls = 0

        def instruments(self, segment: str) -> list[dict]:
            self.calls += 1
            if self.fail:
                raise RuntimeError("boom")
            return self.data

    sample = [
        {
            "name": "NIFTY",
            "expiry": dt.date(2024, 6, 27),
            "strike": 21000,
            "instrument_type": "PE",
            "instrument_token": 999,
            "tradingsymbol": "NIFTY24JUN21000PE",
            "lot_size": 50,
        }
    ]
    kite = KiteStub(sample)
    cache = InstrumentsCache(kite=kite)
    assert kite.calls == 1
    assert cache.get("NIFTY", "2024-06-27", 21000, "PE") is not None

    kite_fail = KiteStub(sample, fail=True)
    cache_fail = InstrumentsCache(kite=kite_fail)
    assert cache_fail.get("NIFTY", "2024-06-27", 21000, "PE") is None
