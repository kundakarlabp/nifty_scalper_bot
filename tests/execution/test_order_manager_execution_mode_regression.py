import pytest

from nifty_scalper_bot.execution.order_manager import OrderManager


class DummyBroker:
    pass


class DummyPositionManager:
    pass


class DummyRateLimiter:
    pass


def test_execution_mode_helpers_belong_to_order_manager_not_guardpair():
    assert hasattr(OrderManager, "_env_truthy")
    assert hasattr(OrderManager, "_execution_mode_env")
    assert hasattr(OrderManager, "_live_flag_enabled")
    assert hasattr(OrderManager, "_order_live_execution_enabled")


def test_execution_mode_env_does_not_raise_attribute_error(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    manager = OrderManager(
        broker_client=DummyBroker(),
        position_manager=DummyPositionManager(),
        rate_limiter=DummyRateLimiter(),
    )

    assert manager.execution_mode == "SHADOW"
    assert manager.get_execution_mode() == "SHADOW"
    assert manager.is_live_mode() is False


def test_live_mode_requires_live_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    with pytest.raises(RuntimeError, match="LIVE mode requires"):
        OrderManager(
            broker_client=DummyBroker(),
            position_manager=DummyPositionManager(),
            rate_limiter=DummyRateLimiter(),
        )


class _SubmittingBroker:
    def __init__(self) -> None:
        self.calls = 0

    def place_order(self, **kwargs):
        self.calls += 1
        return {"order_id": "SIM-1", "status": "success"}


class _MarkedSubmittingBroker(_SubmittingBroker):
    is_simulated_adapter = True


def _live_sim_order_manager(tmp_path, broker):
    from nifty_scalper_bot.execution.position_manager import PositionManager
    from nifty_scalper_bot.utils.rate_limiter import RateLimiter

    return OrderManager(
        broker_client=broker,
        position_manager=PositionManager(str(tmp_path / "positions.json")),
        rate_limiter=RateLimiter(),
    )


def test_live_simulation_order_submission_rejects_unmarked_broker(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _SubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)

    with pytest.raises(RuntimeError, match="non-simulated broker"):
        manager._submit_order_with_retry(  # noqa: SLF001 - canonical broker submission guard
            {"symbol": "NFO:NIFTY26AUG25000CE", "side": "BUY", "quantity": 75}
        )

    assert broker.calls == 0


def test_live_simulation_order_submission_allows_marked_broker(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)

    response = manager._submit_order_with_retry(  # noqa: SLF001 - canonical broker submission guard
        {"symbol": "NFO:NIFTY26AUG25000CE", "side": "BUY", "quantity": 75}
    )

    assert response["order_id"] == "SIM-1"
    assert broker.calls == 1


def test_order_status_parser_preserves_submitted_and_known_broker_states(tmp_path):
    from nifty_scalper_bot.execution.order_manager import OrderStatus

    manager = _live_sim_order_manager(tmp_path, _MarkedSubmittingBroker())

    cases = {
        "SUBMITTED": OrderStatus.SUBMITTED,
        "OPEN": OrderStatus.SUBMITTED,
        "TRIGGER PENDING": OrderStatus.SUBMITTED,
        "PARTIALLY FILLED": OrderStatus.PARTIALLY_FILLED,
        "COMPLETE": OrderStatus.FILLED,
        "REJECTED": OrderStatus.REJECTED,
        "CANCELLED": OrderStatus.CANCELLED,
        "SOME NEW NONTERMINAL": OrderStatus.SUBMITTED,
    }
    for raw, expected in cases.items():
        assert manager._parse_status(raw) is expected


def test_nifty_option_entry_rejects_unit_quantity_not_lot_multiple(
    monkeypatch, tmp_path
):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=1,
        order_type=OrderType.LIMIT,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        check_risk=False,
        intent="ENTRY",
        signal_id="bad-qty",
    )

    assert order_id is None
    assert broker.calls == 0


@pytest.mark.parametrize("quantity", [65, 130])
def test_nifty_option_entry_uses_unit_quantity_at_broker_boundary(
    monkeypatch, tmp_path, quantity
):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class Broker(_MarkedSubmittingBroker):
        def __init__(self):
            super().__init__()
            self.payloads = []

        def place_order(self, **kwargs):
            self.payloads.append(dict(kwargs))
            return {"order_id": f"SIM-{len(self.payloads)}", "status": "SUBMITTED"}

    broker = Broker()
    manager = _live_sim_order_manager(tmp_path, broker)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=quantity,
        order_type=OrderType.LIMIT,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        check_risk=False,
        intent="ENTRY",
        signal_id=f"qty-{quantity}",
    )

    assert order_id is not None
    assert broker.payloads[-1]["quantity"] == quantity


@pytest.mark.parametrize("quantity", [1, 50, 75, 129])
def test_nifty_option_entry_rejects_invalid_unit_quantities_before_broker(
    monkeypatch, tmp_path, quantity
):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=quantity,
        order_type=OrderType.LIMIT,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        check_risk=False,
        intent="ENTRY",
        signal_id=f"bad-qty-{quantity}",
    )

    assert order_id is None
    assert broker.calls == 0


def test_non_option_quantity_validator_remains_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = _live_sim_order_manager(tmp_path, _MarkedSubmittingBroker())
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    manager._validate_quantity("NSE:SBIN", 1)


def test_full_protective_exit_uses_open_position_units_when_lot_lookup_unavailable(
    monkeypatch, tmp_path
):
    from nifty_scalper_bot.execution.order_manager import OrderPlacementError, OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class Broker(_MarkedSubmittingBroker):
        def __init__(self):
            super().__init__()
            self.payloads = []

        def place_order(self, **kwargs):
            self.payloads.append(dict(kwargs))
            return {"order_id": "EXIT-1", "status": "SUBMITTED"}

    broker = Broker()
    manager = _live_sim_order_manager(tmp_path, broker)
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )

    def _raise_unresolved(_symbol):
        raise OrderPlacementError("lot_size_unresolved")

    monkeypatch.setattr(manager, "_lot_size_for_symbol", _raise_unresolved)

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="protective-exit",
    )

    assert order_id == "EXIT-1"
    assert broker.payloads[-1]["quantity"] == 65


def test_option_exit_without_position_is_blocked_before_broker(monkeypatch, tmp_path):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="protective-exit",
    )

    assert order_id is None
    assert broker.calls == 0
    assert manager._last_order_decision["block_reason"] == "exit_without_open_position"


def test_option_exit_exceeding_position_is_blocked_before_broker(monkeypatch, tmp_path):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=130,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="protective-exit",
    )

    assert order_id is None
    assert broker.calls == 0
    assert (
        manager._last_order_decision["block_reason"] == "exit_quantity_exceeds_position"
    )


def test_option_exit_wrong_side_is_blocked_before_broker(monkeypatch, tmp_path):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="protective-exit",
    )

    assert order_id is None
    assert broker.calls == 0
    assert manager._last_order_decision["block_reason"] == "exit_side_not_reducing"


def test_valid_partial_option_exit_requires_lot_multiple(monkeypatch, tmp_path):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class Broker(_MarkedSubmittingBroker):
        def __init__(self):
            super().__init__()
            self.payloads = []

        def place_order(self, **kwargs):
            self.payloads.append(dict(kwargs))
            return {"order_id": "EXIT-PART", "status": "SUBMITTED"}

    broker = Broker()
    manager = _live_sim_order_manager(tmp_path, broker)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 130, 100.0, order_id="entry-1"
    )

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="manual-square-off",
    )

    assert order_id == "EXIT-PART"
    assert broker.payloads[-1]["quantity"] == 65


def test_valid_full_option_exit_accepts_lot_multiple(monkeypatch, tmp_path):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 130, 100.0, order_id="entry-1"
    )

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=130,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="manual-square-off",
    )

    assert order_id == "SIM-1"
    assert broker.calls == 1


@pytest.mark.parametrize(
    ("open_units", "exit_units"),
    [(65, 32), (65, 1), (130, 100)],
)
def test_option_exit_rejects_non_lot_multiple_quantities(
    monkeypatch, tmp_path, open_units, exit_units
):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", open_units, 100.0, order_id="entry-1"
    )

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=exit_units,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="manual-square-off",
    )

    assert order_id is None
    assert broker.calls == 0
    assert (
        manager._last_order_decision["block_reason"] == "exit_quantity_not_lot_multiple"
    )


def test_partial_option_exit_blocks_when_lot_lookup_unavailable(monkeypatch, tmp_path):
    from nifty_scalper_bot.execution.order_manager import OrderPlacementError, OrderType

    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    broker = _MarkedSubmittingBroker()
    manager = _live_sim_order_manager(tmp_path, broker)
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 130, 100.0, order_id="entry-1"
    )

    def _raise_unresolved(_symbol):
        raise OrderPlacementError("lot_size_unresolved")

    monkeypatch.setattr(manager, "_lot_size_for_symbol", _raise_unresolved)

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="manual-square-off",
    )

    assert order_id is None
    assert broker.calls == 0
    assert manager._last_order_decision["block_reason"] == "exit_lot_size_unresolved"


class _ExactLotResolver:
    def __init__(self, lots):
        self.lots = dict(lots)

    def lot_size_for_symbol(self, symbol: str):
        return self.lots.get(symbol.upper()) or self.lots.get(
            symbol.split(":", 1)[-1].upper()
        )


def test_live_nifty_lot_size_resolves_exact_ce_from_instrument_dump(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = _live_sim_order_manager(tmp_path, _MarkedSubmittingBroker())
    manager._resolver = _ExactLotResolver({"NIFTY2671423950CE": 65})
    monkeypatch.setattr(manager, "is_live_mode", lambda: True)

    assert manager._lot_size_for_symbol("NFO:NIFTY2671423950CE") == 65


def test_live_nifty_lot_size_resolves_exact_pe_from_instrument_dump(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = _live_sim_order_manager(tmp_path, _MarkedSubmittingBroker())
    manager._resolver = _ExactLotResolver({"NIFTY2671423950PE": 65})
    monkeypatch.setattr(manager, "is_live_mode", lambda: True)

    assert manager._lot_size_for_symbol("NFO:NIFTY2671423950PE") == 65


def test_live_nifty_lot_size_missing_exact_contract_blocks_fallback(
    monkeypatch, tmp_path
):
    from nifty_scalper_bot.execution.order_manager import OrderPlacementError

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("NIFTY_LOT_SIZE", "65")
    manager = _live_sim_order_manager(tmp_path, _MarkedSubmittingBroker())
    manager._resolver = _ExactLotResolver({"NIFTY2671424000CE": 65})
    monkeypatch.setattr(manager, "is_live_mode", lambda: True)

    with pytest.raises(OrderPlacementError, match="lot_size_unresolved"):
        manager._lot_size_for_symbol("NFO:NIFTY2671423950CE")


def test_non_nifty_option_uses_its_exact_metadata_lot_size(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = _live_sim_order_manager(tmp_path, _MarkedSubmittingBroker())
    manager._resolver = _ExactLotResolver({"BANKNIFTY2671452000CE": 35})
    monkeypatch.setattr(manager, "is_live_mode", lambda: True)

    assert manager._lot_size_for_symbol("NFO:BANKNIFTY2671452000CE") == 35


def test_exit_order_raw_broker_receives_only_zerodha_supported_tag(monkeypatch, tmp_path):
    from nifty_scalper_bot.execution.order_manager import OrderType

    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class StrictBroker:
        is_simulated_adapter = True

        def __init__(self) -> None:
            self.payload = None

        def place_order(
            self,
            *,
            symbol,
            side,
            quantity,
            product,
            order_type,
            price=None,
            trigger_price=None,
            tag=None,
            variety="regular",
        ):
            self.payload = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "product": product,
                "order_type": order_type,
                "price": price,
                "trigger_price": trigger_price,
                "tag": tag,
                "variety": variety,
            }
            return {"order_id": "strict-exit-1", "status": "SUBMITTED", "tag": tag}

    broker = StrictBroker()
    manager = _live_sim_order_manager(tmp_path, broker)
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 95.0, order_id="entry-1"
    )
    manager._order_live_execution_enabled = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _symbol: 65)

    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.LIMIT,
        price=100.0,
        check_risk=False,
        intent="EXIT",
        bracket_id="entry-1",
        tag="EXIT_abcd1234_1",
    )

    assert order_id == "strict-exit-1"
    assert broker.payload is not None
    assert broker.payload["tag"] == "EXIT_abcd1234_1"
