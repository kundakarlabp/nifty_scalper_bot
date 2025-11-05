from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nifty_scalper_bot.options.strike_selector import SelectedContract
import nifty_scalper_bot.strategies.runner as runner_module
from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils.errors import OrderPlacementError


class StubMetric:
    def __init__(self) -> None:
        self.reasons: list[str] = []

    def labels(self, **kwargs):  # noqa: D401 - minimal stub
        reason = kwargs.get("reason")
        if reason is not None:
            self.reasons.append(reason)
        return self

    def inc(self) -> None:  # noqa: D401 - minimal stub
        return None


class StubOrderManager:
    def __init__(self, message: str) -> None:
        self._message = message

    def place_order(self, **kwargs):  # noqa: D401 - minimal stub
        raise OrderPlacementError(self._message)

    def place_reduce_only_exit(self, *_args, **_kwargs):  # noqa: D401 - minimal stub
        raise OrderPlacementError(self._message)

    def consume_skip_reason(self) -> str | None:  # noqa: D401 - minimal stub
        return None


class StubRiskManager:
    def suggest_position_size(self, **kwargs) -> int:  # noqa: D401 - minimal stub
        return int(kwargs.get("requested_quantity", 0))

    def validate_new_position(self, **kwargs):  # noqa: D401 - minimal stub
        return True, None

    def validate_close_position(self, **kwargs):  # noqa: D401 - minimal stub
        return True, None


class EntryPositionManager:
    def get_active_contract(self, symbol: str):  # noqa: D401 - minimal stub
        return None

    def is_flat(self, symbol: str) -> bool:  # noqa: D401 - minimal stub
        return True

    def clear_active_contract(self, symbol: str) -> None:  # noqa: D401 - minimal stub
        return None

    def get_position(self, symbol: str):  # noqa: D401 - minimal stub
        return None

    def set_active_contract(self, symbol: str, selection) -> None:  # noqa: D401 - stub
        return None


class ExitPositionManager:
    def __init__(self) -> None:
        self._contract = SimpleNamespace(
            symbol="NFO:NIFTY24OCT20000CE",
            option_type="CE",
            strike=20_000,
            expiry=datetime.now(timezone.utc),
        )
        self._position = SimpleNamespace(quantity=25, current_price=205.0)

    def get_active_contract(self, symbol: str):  # noqa: D401 - minimal stub
        return self._contract

    def is_flat(self, symbol: str) -> bool:  # noqa: D401 - minimal stub
        return False

    def clear_active_contract(self, symbol: str) -> None:  # noqa: D401 - minimal stub
        return None

    def set_active_contract(
        self, symbol: str, selection
    ) -> None:  # noqa: D401 - minimal stub
        self._contract = selection

    def get_position(self, symbol: str):  # noqa: D401 - minimal stub
        if symbol == self._contract.symbol:
            return self._position
        return None


class StubStrategyManager:
    pass


class StubIndicatorEngine:
    pass


class StubMarketDataManager:
    pass


def _build_runner(order_manager, position_manager):
    return StrategyRunner(
        market_data_manager=StubMarketDataManager(),
        indicator_engine=StubIndicatorEngine(),
        strategy_manager=StubStrategyManager(),
        risk_manager=StubRiskManager(),
        order_manager=order_manager,
        position_manager=position_manager,
    )


def test_runner_logs_and_counts_no_reference_price(caplog, monkeypatch) -> None:
    metric = StubMetric()
    monkeypatch.setattr(runner_module, "_STRATEGY_SKIP_COUNTER", metric)
    monkeypatch.setattr(runner_module.metrics, "strategy_skips_total", metric)
    runner = _build_runner(
        StubOrderManager("NO_REFERENCE_PRICE"), EntryPositionManager()
    )
    signal = Signal(
        action="BUY",
        symbol="NIFTY",
        quantity=1,
        confidence=1.0,
        reason="test",
        stop_loss=None,
        take_profit=None,
        metadata={"option_type": "CE"},
    )
    with caplog.at_level("INFO", logger=runner._logger.name):
        runner._handle_signal(signal, price=200.0, timestamp=datetime.now(timezone.utc))

    assert any(
        record.msg == "skip_order" and record.reason == "no_reference_price"
        for record in caplog.records
    )
    assert "no_reference_price" in metric.reasons


def test_runner_logs_and_counts_margin_no_qty(caplog, monkeypatch) -> None:
    metric = StubMetric()
    monkeypatch.setattr(runner_module, "_STRATEGY_SKIP_COUNTER", metric)
    monkeypatch.setattr(runner_module.metrics, "strategy_skips_total", metric)
    runner = _build_runner(StubOrderManager("margin_no_qty"), ExitPositionManager())
    signal = Signal(
        action="CLOSE_LONG",
        symbol="NIFTY",
        quantity=1,
        confidence=1.0,
        reason="test",
        stop_loss=None,
        take_profit=None,
        metadata={"option_type": "CE"},
    )
    with caplog.at_level("INFO", logger=runner._logger.name):
        runner._handle_signal(signal, price=200.0, timestamp=datetime.now(timezone.utc))

    assert any(
        record.msg == "skip_order" and record.reason == "margin_no_qty"
        for record in caplog.records
    )
    assert "margin_no_qty" in metric.reasons


def test_runner_skips_near_monthly_expiry(monkeypatch) -> None:
    class NoopOrderManager:
        def __init__(self) -> None:
            self.placed = False

        def place_order(self, **_kwargs) -> None:  # noqa: D401 - test stub
            self.placed = True
            raise AssertionError("order placement should be blocked")

        def place_reduce_only_exit(self, *_args, **_kwargs) -> None:  # noqa: D401
            raise AssertionError("exit placement should be blocked")

        def consume_skip_reason(self) -> str | None:  # noqa: D401 - minimal stub
            return None

    monthly_expiry = datetime(2024, 7, 25, 15, 0, tzinfo=timezone.utc)

    class LockoutSelector:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(
                long_only=True,
                legacy_side_to_type=False,
                monthly_halt_minutes=60,
            )

        def select_contract(self, **_kwargs):  # noqa: D401 - test stub
            return SelectedContract(
                symbol="NFO:NIFTY24JUL19800CE",
                option_type="CE",
                strike=19800.0,
                expiry=monthly_expiry,
                ltp=150.0,
                delta=0.5,
                metadata={},
            )

    order_manager = NoopOrderManager()
    selector = LockoutSelector()
    runner = StrategyRunner(
        market_data_manager=StubMarketDataManager(),
        indicator_engine=StubIndicatorEngine(),
        strategy_manager=StubStrategyManager(),
        risk_manager=StubRiskManager(),
        order_manager=order_manager,
        position_manager=EntryPositionManager(),
        strike_selector=selector,
    )
    runner.add_symbol("NIFTY")
    signal = Signal(
        action="BUY",
        symbol="NIFTY",
        quantity=1,
        confidence=1.0,
        reason="test",
        stop_loss=None,
        take_profit=None,
        metadata={"option_type": "CE"},
    )

    runner._handle_signal(
        signal, price=200.0, timestamp=monthly_expiry - timedelta(minutes=30)
    )

    assert not order_manager.placed
    state = runner._symbol_state.get("NIFTY")
    assert state is not None
    assert state.trade_history
    assert state.trade_history[-1].reason == "monthly_expiry_lockout"
