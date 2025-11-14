import logging
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.storage import HubStore


class _StubMarketDataManager:
    def __init__(self) -> None:
        self._latest: dict[str, dict[str, Any]] = {}
        self.subscribed: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
        self.unsubscribed: list[tuple[str, Callable[[dict[str, Any]], None]]] = []
        self._balance_snapshot: dict[str, Any] = {}
        self._balance_value: float | None = None

    def set_tick(self, symbol: str, tick: dict[str, Any]) -> None:
        self._latest[symbol] = tick

    def subscribe(
        self, symbol: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        self.subscribed.setdefault(symbol, []).append(callback)

    def unsubscribe(
        self, symbol: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        self.unsubscribed.append((symbol, callback))

    def get_latest_tick(self, symbol: str) -> dict[str, Any] | None:
        return self._latest.get(symbol)

    def pull_quote(self, symbol: str) -> dict[str, Any] | None:
        return self._latest.get(symbol)

    def set_balance_snapshot(
        self, snapshot: Mapping[str, Any], *, available: float | None = None
    ) -> None:
        self._balance_snapshot = dict(snapshot)
        self._balance_value = available

    def get_available_balance(self, *, force: bool = False) -> float | None:
        return self._balance_value

    def get_account_snapshot(self, *, force: bool = False) -> dict[str, Any]:
        return dict(self._balance_snapshot)


class _StubResolver:
    def __init__(self, mapping: dict[int, str] | None = None) -> None:
        self._mapping = mapping or {}

    def resolve_token_to_symbol(self, token: int) -> str | None:
        return self._mapping.get(int(token))


@pytest.fixture()
def stub_mdm() -> _StubMarketDataManager:
    return _StubMarketDataManager()


@pytest.fixture()
def hub(stub_mdm: _StubMarketDataManager, resolver: _StubResolver) -> DataHub:
    return DataHub(stub_mdm, resolver, options_only=True)


@pytest.fixture()
def resolver() -> _StubResolver:
    return _StubResolver()


@pytest.fixture()
def hub_store(tmp_path: Path) -> HubStore:
    store = HubStore(tmp_path / "hub.db")
    yield store
    store.close()


@pytest.fixture()
def hub_with_store(
    stub_mdm: _StubMarketDataManager, resolver: _StubResolver, hub_store: HubStore
) -> DataHub:
    return DataHub(
        stub_mdm,
        resolver,
        options_only=True,
        store=hub_store,
        checkpoint_interval=0.1,
    )


def test_get_available_balance_uses_available_cash(
    hub: DataHub, stub_mdm: _StubMarketDataManager
) -> None:
    stub_mdm.set_balance_snapshot({"available_cash": 275_000.0})

    balance = hub.get_available_balance()

    assert balance == pytest.approx(275_000.0)


def test_replace_positions_filters_and_normalises(hub: DataHub) -> None:
    hub.replace_positions(
        [
            {"symbol": "nifty25oct25000ce", "quantity": "50", "average_price": "201.5"},
            {"symbol": "BANKNIFTY25NOVFUT", "quantity": 15, "average_price": 42000},
            {"symbol": "NIFTY", "quantity": 0, "average_price": ""},
            {"symbol": "nifty25oct25000pe", "quantity": "-25", "average_price": ""},
        ]
    )

    positions = {row["symbol"]: row for row in hub.positions()}

    assert "NIFTY25OCT25000CE" in positions
    assert positions["NIFTY25OCT25000CE"]["quantity"] == 50
    assert positions["NIFTY25OCT25000CE"]["average_price"] == pytest.approx(201.5)
    assert "BANKNIFTY25NOVFUT" not in positions
    assert positions["NIFTY"]["quantity"] == 0
    assert positions["NIFTY"]["side"] == "FLAT"
    assert positions["NIFTY25OCT25000PE"]["side"] == "SHORT"
    assert positions["NIFTY25OCT25000PE"]["quantity"] == 25


def test_upsert_order_coerces_payload_fields(hub: DataHub) -> None:
    hub.upsert_order(
        {
            "order_id": "OID123",
            "symbol": "nifty25oct25000ce",
            "side": "buy",
            "quantity": "50",
            "filled_quantity": "25",
            "status": "OPEN",
            "price": "",
            "trigger_price": " ",
            "timestamp": "1735000000000",
        }
    )

    orders = hub.orders()
    assert len(orders) == 1
    order = orders[0]
    assert order["symbol"] == "NIFTY25OCT25000CE"
    assert order["quantity"] == 50
    assert order["filled_quantity"] == 25
    assert order["price"] is None
    assert order["trigger_price"] is None
    assert order["ts"] == pytest.approx(1_735_000_000.0)


def test_subscribe_ticks_warm_pushes_cached_quote(
    hub: DataHub, stub_mdm: _StubMarketDataManager
) -> None:
    symbol = "NIFTY25OCT25000CE"
    now = time.time()
    stub_mdm.set_tick(symbol, {"symbol": symbol, "ltp": "99.5", "timestamp": now})
    quote = hub.pull_quote(symbol)
    assert quote is not None

    received: list[dict[str, Any]] = []
    hub.subscribe_ticks(symbol, lambda tick: received.append(tick))

    assert len(received) == 1
    assert received[0]["ltp"] == pytest.approx(99.5)
    assert symbol in stub_mdm.subscribed


def test_get_quote_pulls_when_missing(
    hub: DataHub, stub_mdm: _StubMarketDataManager
) -> None:
    symbol = "NIFTY25DEC25000CE"
    stub_mdm.set_tick(symbol, {"symbol": symbol, "ltp": 150.0, "timestamp": 111})

    quote = hub.get_quote(symbol)
    assert quote is not None
    assert quote["ltp"] == pytest.approx(150.0)


def test_upsert_order_drops_duplicate_sequences(hub: DataHub) -> None:
    payload = {
        "order_id": "OID1",
        "symbol": "nifty25oct25000ce",
        "quantity": 50,
        "filled_quantity": 0,
        "status": "SUBMITTED",
        "status_sequence": 2,
    }
    hub.upsert_order(payload)
    duplicate = dict(payload)
    duplicate.update({"status": "FILLED", "status_sequence": 1})
    hub.upsert_order(duplicate)

    orders = hub.orders()
    assert len(orders) == 1
    assert orders[0]["status"] == "submitted"


def test_is_fresh_returns_false_without_tick(hub: DataHub) -> None:
    ok, meta = hub.is_fresh("NIFTY")

    assert ok is False
    assert meta["reason"] == "no_tick"
    assert meta["threshold_ms"] > 0


def test_is_fresh_reports_metrics_for_cached_tick(hub: DataHub) -> None:
    now = time.time()
    hub.ingest_tick(
        {
            "symbol": "NIFTY",
            "last_price": 100.0,
            "best_bid": 99.5,
            "best_ask": 100.5,
            "timestamp": now,
            "server_ts_s": now,
        }
    )

    ok, meta = hub.is_fresh("NIFTY")

    assert ok is True
    assert meta["effective_ms"] is not None
    assert meta["threshold_ms"] >= 0
    assert meta["reason"] in {None, "warmup_grace"}


def test_upsert_order_rejects_invalid_transition(hub: DataHub) -> None:
    base = {
        "order_id": "OID2",
        "symbol": "nifty25oct25000pe",
        "quantity": 25,
    }
    hub.upsert_order({**base, "status": "submitted", "status_sequence": 1})
    hub.upsert_order({**base, "status": "cancelled", "status_sequence": 2})
    hub.upsert_order({**base, "status": "submitted", "status_sequence": 3})

    orders = hub.orders()
    assert len(orders) == 1
    assert orders[0]["status"] == "cancelled"


def test_store_restores_snapshot(
    stub_mdm: _StubMarketDataManager,
    resolver: _StubResolver,
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "hub.db"
    store = HubStore(store_path)
    hub = DataHub(
        stub_mdm,
        resolver,
        options_only=True,
        store=store,
        checkpoint_interval=0.1,
    )
    tick = {"symbol": "NIFTY", "ltp": 99.5, "timestamp": time.time()}
    hub.ingest_tick(tick)
    hub.replace_positions(
        [
            {"symbol": "NIFTY", "quantity": 0, "average_price": 0.0},
            {"symbol": "NIFTY25OCT25000CE", "quantity": 50, "average_price": 100.0},
        ]
    )
    hub.upsert_order(
        {
            "order_id": "OID3",
            "symbol": "nifty25oct25000ce",
            "quantity": 50,
            "status": "submitted",
            "status_sequence": 1,
        }
    )
    hub.checkpoint()
    store.close()

    restored_store = HubStore(store_path)
    restored = DataHub(
        stub_mdm,
        resolver,
        options_only=True,
        store=restored_store,
        checkpoint_interval=0.1,
    )
    restored_quotes = restored.get_quote("NIFTY", allow_pull=False)
    assert restored_quotes is not None
    assert restored_quotes["ltp"] == pytest.approx(99.5)
    orders = {row["order_id"]: row for row in restored.orders()}
    assert orders["OID3"]["status"] == "submitted"
    positions = {row["symbol"]: row for row in restored.positions()}
    assert "NIFTY25OCT25000CE" in positions
    restored_store.close()


def test_stale_requires_two_consecutive_polls(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stub_mdm: _StubMarketDataManager,
    resolver: _StubResolver,
) -> None:
    monkeypatch.setenv("STALE_TICK_MAX_AGE_MS", "100")
    monkeypatch.setenv("DRIFT_BUDGET_MS", "0")
    monkeypatch.setenv("WARMUP_GRACE_S", "0")
    hub = DataHub(stub_mdm, resolver, options_only=True)
    base = 1_000.0
    wall = [base]
    mono = [0.0]
    setattr(hub, "_now", lambda: wall[0])
    setattr(hub, "_monotonic", lambda: mono[0])
    hub._start_mono = mono[0]  # type: ignore[assignment]

    symbol = "NIFTY25OCT25000CE"
    hub.ingest_tick({"symbol": symbol, "timestamp": base})

    wall[0] = base + 0.6
    mono[0] = 0.6
    stale_tick = {"symbol": symbol, "timestamp": base}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hub.ingest_tick(stale_tick)

    assert not any("stale_tick_dropped" in record.message for record in caplog.records)

    wall[0] = base + 1.2
    mono[0] = 1.2
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hub.ingest_tick(stale_tick)

    records = [rec for rec in caplog.records if "stale_tick_dropped" in rec.message]
    assert len(records) == 1
    cached = hub.get_quote(symbol, allow_pull=False)
    assert cached is not None
    assert cached.get("timestamp") == pytest.approx(base)


def test_adaptive_staleness_allows_1p7s_when_poll_0p7s(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stub_mdm: _StubMarketDataManager,
    resolver: _StubResolver,
) -> None:
    monkeypatch.delenv("STALE_TICK_MAX_AGE_MS", raising=False)
    monkeypatch.setenv("POLL_INTERVAL_MS", "700")
    monkeypatch.setenv("WARMUP_GRACE_S", "0")
    monkeypatch.setenv("DRIFT_BUDGET_MS", "0")
    hub = DataHub(stub_mdm, resolver, options_only=True)
    assert hub._stale_tick_max_age_ms == pytest.approx(2000.0)
    base = 2_000.0
    wall = [base]
    mono = [0.0]
    setattr(hub, "_now", lambda: wall[0])
    setattr(hub, "_monotonic", lambda: mono[0])
    hub._start_mono = mono[0]  # type: ignore[assignment]

    symbol = "NIFTY25OCT25000CE"
    hub.ingest_tick({"symbol": symbol, "timestamp": base})

    wall[0] = base + 1.7
    mono[0] = 1.7
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hub.ingest_tick({"symbol": symbol, "timestamp": base})

    assert not any("stale_tick_dropped" in record.message for record in caplog.records)
    cached = hub.get_quote(symbol, allow_pull=False)
    assert cached is not None
    assert cached.get("timestamp") == pytest.approx(base)


def test_dual_clock_requires_both_ages_old_for_stale(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stub_mdm: _StubMarketDataManager,
    resolver: _StubResolver,
) -> None:
    monkeypatch.setenv("STALE_TICK_MAX_AGE_MS", "500")
    monkeypatch.setenv("DRIFT_BUDGET_MS", "0")
    monkeypatch.setenv("WARMUP_GRACE_S", "0")
    hub = DataHub(stub_mdm, resolver, options_only=True)
    base = 3_000.0
    wall = [base]
    mono = [0.0]
    setattr(hub, "_now", lambda: wall[0])
    setattr(hub, "_monotonic", lambda: mono[0])
    hub._start_mono = mono[0]  # type: ignore[assignment]

    symbol = "NIFTY25OCT25000CE"
    hub.ingest_tick({"symbol": symbol, "timestamp": base})

    wall[0] = base + 2.0
    mono[0] = 0.3
    stale_tick = {"symbol": symbol, "timestamp": base}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hub.ingest_tick(stale_tick)

    assert not any("stale_tick_dropped" in record.message for record in caplog.records)

    wall[0] = base + 2.4
    mono[0] = 0.6
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hub.ingest_tick(stale_tick)

    assert not any("stale_tick_dropped" in record.message for record in caplog.records)


def test_warmup_grace_skips_and_drops_after_hard_cap(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stub_mdm: _StubMarketDataManager,
    resolver: _StubResolver,
) -> None:
    monkeypatch.setenv("STALE_TICK_MAX_AGE_MS", "1500")
    monkeypatch.setenv("DRIFT_BUDGET_MS", "0")
    monkeypatch.setenv("WARMUP_GRACE_S", "5")
    hub = DataHub(stub_mdm, resolver, options_only=True)
    base = 4_000.0
    wall = [base]
    mono = [0.0]
    setattr(hub, "_now", lambda: wall[0])
    setattr(hub, "_monotonic", lambda: mono[0])
    hub.reset_warmup()

    symbol = "NIFTY25OCT25000CE"
    hub.ingest_tick({"symbol": symbol, "timestamp": base})

    wall[0] = base + 2.0
    mono[0] = 2.0
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hub.ingest_tick({"symbol": symbol, "timestamp": base})

    assert not any("stale_tick_dropped" in record.message for record in caplog.records)

    wall[0] = base + 12.0
    mono[0] = 4.0
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        hub.ingest_tick({"symbol": symbol, "timestamp": base})

    records = [rec for rec in caplog.records if "stale_tick_dropped" in rec.message]
    assert len(records) == 1
    cached = hub.get_quote(symbol, allow_pull=False)
    assert cached is not None
    assert cached.get("timestamp") == pytest.approx(base)


def test_tick_listener_error_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stub_mdm: _StubMarketDataManager,
    resolver: _StubResolver,
) -> None:
    monkeypatch.setenv("TICK_ERROR_LOG_COOLDOWN_SEC", "10")
    monkeypatch.setenv("STALE_TICK_MAX_AGE_MS", "5000")
    hub = DataHub(stub_mdm, resolver, options_only=True)
    base_time = time.time()
    now_box = {"value": base_time}
    hub._now = lambda: now_box["value"]  # type: ignore[assignment]
    symbol = "NIFTY25OCT25000CE"

    def failing_listener(_tick: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    hub.subscribe_ticks(symbol, failing_listener)
    tick = {"symbol": symbol, "timestamp": base_time}

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        hub.ingest_tick(tick)

    records_first = [
        record for record in caplog.records if "tick_listener_failed" in record.message
    ]
    assert len(records_first) == 1
    assert "consecutive=1" in records_first[0].message

    caplog.clear()
    with caplog.at_level(logging.ERROR):
        hub.ingest_tick(tick)

    assert not any(
        "tick_listener_failed" in record.message for record in caplog.records
    )
    assert hub._listener_error_streak == 2  # type: ignore[attr-defined]

    caplog.clear()
    now_box["value"] = base_time + 15
    tick["timestamp"] = now_box["value"]
    with caplog.at_level(logging.ERROR):
        hub.ingest_tick(tick)

    records_third = [
        record for record in caplog.records if "tick_listener_failed" in record.message
    ]
    assert len(records_third) == 1
    assert "consecutive=3" in records_third[0].message
