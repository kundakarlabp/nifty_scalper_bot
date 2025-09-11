from src.execution import order_executor as oe


def _executor() -> oe.OrderExecutor:
    return oe.OrderExecutor(kite=None)


def test_place_order_blocks_when_micro_bad(monkeypatch) -> None:
    ex = _executor()
    monkeypatch.setattr(oe, "ENTRY_WAIT_S", 0.0)
    payload = {
        "action": "BUY",
        "quantity": ex.lot_size,
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "take_profit": 110.0,
        "symbol": "SYM",
        "bid": 100.0,
        "ask": 101.0,
        "depth": (ex.lot_size * 10, ex.lot_size * 10),
    }
    rid = ex.place_order(payload)
    assert rid is None
    assert ex.last_error == "micro_block"


def test_place_order_waits_and_succeeds(monkeypatch) -> None:
    ex = _executor()
    monkeypatch.setattr(oe, "ENTRY_WAIT_S", 4.0)
    monkeypatch.setattr(oe, "MICRO_SPREAD_CAP", 0.5)

    fake_time = {"t": 0.0}

    def fake_monotonic() -> float:  # pragma: no cover - deterministic fake
        return fake_time["t"]

    def fake_sleep(dt: float) -> None:  # pragma: no cover - deterministic fake
        fake_time["t"] += dt

    monkeypatch.setattr(oe.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(oe.time, "sleep", fake_sleep)

    def refresh():
        return (100.0, 100.1, (ex.lot_size * 10, ex.lot_size * 10))

    payload = {
        "action": "BUY",
        "quantity": ex.lot_size,
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "take_profit": 110.0,
        "symbol": "SYM",
        "bid": 100.0,
        "ask": 101.0,
        "depth": (ex.lot_size * 10, ex.lot_size * 10),
        "refresh_market": refresh,
    }
    rid = ex.place_order(payload)
    assert rid is not None
    assert ex.last_error == "micro_wait"


def test_price_uses_half_spread_slippage(monkeypatch) -> None:
    ex = _executor()
    monkeypatch.setattr(oe, "ENTRY_WAIT_S", 0.0)
    monkeypatch.setattr(oe, "MICRO_SPREAD_CAP", 2.0)
    payload = {
        "action": "BUY",
        "quantity": ex.lot_size,
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "take_profit": 110.0,
        "symbol": "SYM",
        "bid": 100.0,
        "ask": 101.0,
        "depth": (ex.lot_size * 10, ex.lot_size * 10),
    }
    rid = ex.place_order(payload)
    assert rid is not None
    rec = ex._active[rid]
    assert rec.entry_price == 101.0

