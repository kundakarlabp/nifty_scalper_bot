import pytest

from nifty_scalper_bot.execution.order_manager import OrderManager


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
