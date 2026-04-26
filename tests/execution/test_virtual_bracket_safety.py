"""Safety checks for pending-entry and exit confirmation lifecycle."""

from __future__ import annotations

from nifty_scalper_bot.execution.bracket_manager import BracketManager


class _Broker:
    def __init__(self, status: str = 'COMPLETE') -> None:
        self._status = status

    def get_order_status(self, _order_id: str) -> dict[str, str]:
        return {'status': self._status}

    def get_positions(self) -> list[dict[str, int]]:
        return []


class _OrderManager:
    def __init__(self, *, wait_fill: bool, order_id: str | None = 'exit-1', status: str = 'COMPLETE') -> None:
        self._wait_fill = wait_fill
        self._order_id = order_id
        self._broker = _Broker(status=status)

    def place_order(self, **_: object) -> str | None:
        return self._order_id

    def wait_for_fill(self, _order_id: str, timeout_sec: float = 2.0) -> bool:
        _ = timeout_sec
        return self._wait_fill


def test_pending_entry_bracket_does_not_fire_exit() -> None:
    manager = BracketManager(order_manager=_OrderManager(wait_fill=True))
    manager.register_virtual_bracket(
        order_id='entry-pending',
        symbol='NFO:NIFTYTEST',
        side='BUY',
        qty=1,
        price=100.0,
        sl=99.0,
        tp=110.0,
    )
    manager.on_tick('NFO:NIFTYTEST', 98.0)
    bracket = manager.get_bracket('entry-pending')
    assert bracket is not None
    assert bracket.entry_confirmed is False
    assert bracket.exit_executed is False
    assert bracket.remaining_quantity == 1


def test_confirmed_fill_activates_bracket() -> None:
    manager = BracketManager(order_manager=_OrderManager(wait_fill=True))
    manager.register_virtual_bracket(
        order_id='entry-active',
        symbol='NFO:NIFTYACTIVE',
        side='BUY',
        qty=1,
        price=100.0,
        sl=99.0,
        tp=110.0,
    )
    manager.confirm_entry_fill('entry-active', 101.0)
    bracket = manager.get_bracket('entry-active')
    assert bracket is not None
    assert bracket.active is True
    assert bracket.entry_confirmed is True
    assert bracket.entry_status == 'ACTIVE'


def test_unfilled_limit_exit_does_not_close_bracket() -> None:
    manager = BracketManager(order_manager=_OrderManager(wait_fill=False, order_id='exit-2', status='OPEN'))
    manager.register_virtual_bracket(
        order_id='entry-exit',
        symbol='NFO:NIFTYEXIT',
        side='BUY',
        qty=2,
        price=100.0,
        sl=99.0,
        tp=110.0,
    )
    manager.confirm_entry_fill('entry-exit', 100.0)
    result = manager._execute_exit(manager.get_bracket('entry-exit'), 1, 'SL', is_partial=True)  # type: ignore[arg-type]
    bracket = manager.get_bracket('entry-exit')
    assert result.confirmed is False
    assert bracket is not None
    assert bracket.remaining_quantity == 2
    assert bracket.entry_status in {'ACTIVE', 'EXIT_PENDING'}
