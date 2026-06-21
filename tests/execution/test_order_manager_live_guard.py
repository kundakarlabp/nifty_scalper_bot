import pytest

from nifty_scalper_bot.execution.order_manager import OrderIntent, OrderManager


class _Broker:
    def __init__(self, connected=True):
        self.connected = connected

    def is_connected(self):
        return self.connected


class _Positions:
    def has_open_position(self, symbol):
        return False


class _Limiter:
    pass


def _manager(monkeypatch, *, mode="LIVE", enable_live=None, enable_live_trading=None, connected=True):
    monkeypatch.setenv("EXECUTION_MODE", mode)
    if enable_live is None:
        monkeypatch.delenv("ENABLE_LIVE", raising=False)
    else:
        monkeypatch.setenv("ENABLE_LIVE", enable_live)

    if enable_live_trading is None:
        monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    else:
        monkeypatch.setenv("ENABLE_LIVE_TRADING", enable_live_trading)

    return OrderManager(_Broker(connected=connected), _Positions(), _Limiter())


def test_live_guard_allows_enable_live(monkeypatch):
    om = _manager(monkeypatch, mode="LIVE", enable_live="true")
    assert om._validate_live_execution_safety() is True


def test_live_guard_allows_enable_live_trading(monkeypatch):
    om = _manager(monkeypatch, mode="LIVE", enable_live="false", enable_live_trading="true")
    assert om._validate_live_execution_safety() is True


def test_live_guard_blocks_paper_even_with_enable_live(monkeypatch):
    om = _manager(monkeypatch, mode="PAPER", enable_live="true")
    assert om._validate_live_execution_safety() is False


def test_live_guard_blocks_live_without_any_live_flag(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.delenv("ENABLE_LIVE", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)

    with pytest.raises(RuntimeError, match="LIVE mode requires"):
        OrderManager(_Broker(), _Positions(), _Limiter())


def test_live_guard_blocks_disconnected_broker(monkeypatch):
    om = _manager(monkeypatch, mode="LIVE", enable_live_trading="true", connected=False)
    assert om._validate_live_execution_safety() is False


def test_live_selected_option_ltp_only_rejected_before_broker(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    broker = _Broker(connected=True)
    broker.calls = 0

    def _place_order(**_kwargs):
        broker.calls += 1
        return {"order_id": "OID"}

    broker.place_order = _place_order
    om = OrderManager(broker, _Positions(), _Limiter())
    monkeypatch.setattr(om, "_validate_live_execution_safety", lambda: True)
    om.set_entry_execution_guard(lambda: (True, None))
    selected = "NFO:NIFTY26MAY23750CE"
    om._market_data = type(
        "MDM",
        (),
        {
            "get_active_contract_basket": lambda self: {"selected_ce": selected, "selected_pe": "NFO:NIFTY26MAY23750PE"},
            "get_quote": lambda self, symbol: {"ltp": 100.0, "last_price": 100.0},
        },
    )()

    result = om.place_order(
        selected,
        "BUY",
        1,
        stop_loss=90.0,
        take_profit=120.0,
    )

    assert result is None
    assert broker.calls == 0
    assert om.get_last_skip_reason() == "selected_option_bid_ask_missing"
    assert om._last_order_decision["block_reason"] == "selected_option_bid_ask_missing"


def test_auth_latch_blocks_order_before_broker(monkeypatch):
    from nifty_scalper_bot.utils.errors import OrderPlacementError

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    broker = _Broker(connected=True)
    broker.auth_invalid = True
    broker.calls = 0

    def _place_order(**_kwargs):
        broker.calls += 1
        return {"order_id": "OID"}

    broker.place_order = _place_order
    om = OrderManager(broker, _Positions(), _Limiter())

    with pytest.raises(OrderPlacementError, match="broker_auth_invalid"):
        om.place_order(
            "NFO:NIFTY26MAY23750CE",
            "BUY",
            75,
            stop_loss=90.0,
            take_profit=120.0,
        )

    assert broker.calls == 0
    assert om.get_last_skip_reason() == "broker_auth_invalid"


def test_entry_execution_guard_blocks_entry_before_broker(monkeypatch):
    from nifty_scalper_bot.utils.errors import OrderPlacementError

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    broker = _Broker(connected=True)
    broker.calls = 0
    broker.place_order = lambda **_kwargs: {"order_id": "OID"}
    om = OrderManager(broker, _Positions(), _Limiter())
    monkeypatch.setattr(om, "_validate_live_execution_safety", lambda: True)
    om.set_entry_execution_guard(lambda: (False, "position_reconciliation_incomplete"))

    with pytest.raises(OrderPlacementError, match="position_reconciliation_incomplete"):
        om.place_order("NFO:NIFTY26MAY23750CE", "BUY", 75, stop_loss=90, take_profit=120, intent=OrderIntent.ENTRY)

    assert broker.calls == 0
    assert om.get_last_skip_reason() == "position_reconciliation_incomplete"


def test_entry_execution_guard_does_not_block_protective_exit(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    broker = _Broker(connected=True)
    broker.calls = 0

    def _place_order(**_kwargs):
        broker.calls += 1
        return {"order_id": "OID"}

    broker.place_order = _place_order
    om = OrderManager(broker, _Positions(), _Limiter())
    monkeypatch.setattr(om, "_validate_live_execution_safety", lambda: True)
    om.set_entry_execution_guard(lambda: (False, "selected_option_subscription_missing"))

    result = om.place_order("NFO:NIFTY26MAY23750CE", "SELL", 75, tag="protective_exit", intent=OrderIntent.PROTECTIVE)

    assert result == "OID"
    assert broker.calls == 1


def test_live_entry_fails_closed_when_entry_guard_missing(monkeypatch):
    from nifty_scalper_bot.utils.errors import OrderPlacementError

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    broker = _Broker(connected=True)
    broker.calls = 0
    broker.place_order = lambda **_kwargs: {"order_id": "OID"}
    om = OrderManager(broker, _Positions(), _Limiter())
    monkeypatch.setattr(om, "_validate_live_execution_safety", lambda: True)

    with pytest.raises(OrderPlacementError, match="entry_execution_guard_missing"):
        om.place_order(
            "NFO:NIFTY26MAY23750CE",
            "BUY",
            75,
            stop_loss=90,
            take_profit=120,
            intent=OrderIntent.ENTRY,
        )

    assert broker.calls == 0


def test_explicit_exit_intent_bypasses_entry_guard(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    broker = _Broker(connected=True)
    broker.calls = 0

    def _place_order(**_kwargs):
        broker.calls += 1
        return {"order_id": "OID"}

    broker.place_order = _place_order
    om = OrderManager(broker, _Positions(), _Limiter())
    monkeypatch.setattr(om, "_validate_live_execution_safety", lambda: True)
    om.set_entry_execution_guard(lambda: (False, "readiness_snapshot_missing"))

    assert om.place_order(
        "NFO:NIFTY26MAY23750CE",
        "SELL",
        75,
        intent=OrderIntent.EXIT,
    ) == "OID"
    assert broker.calls == 1
