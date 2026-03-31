from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from nifty_scalper_bot.execution.order_manager import ExitIntent
from nifty_scalper_bot.options.strike_selector import SelectedContract
from nifty_scalper_bot.strategies.runner import (
    StrategyRunner,
    StrategyRunnerConfig,
    SymbolRuntimeState,
)
from nifty_scalper_bot.strategies.signal_generator import Signal


class _StubSelector:
    def __init__(
        self, *, long_only: bool = True, legacy_side_to_type: bool = False
    ) -> None:
        self.selected_for: list[tuple[str, str, str, float]] = []
        self.selection: SelectedContract | None = None
        self.registered: list[tuple[str, SelectedContract]] = []
        self.resolved: SelectedContract | None = None
        self.cleared: list[str] = []
        self.settings = SimpleNamespace(
            long_only=long_only, legacy_side_to_type=legacy_side_to_type
        )

    def select_contract(
        self,
        *,
        underlying: str,
        side: str,
        option_type: str,
        underlying_price: float,
        **_: object,
    ) -> SelectedContract | None:
        self.selected_for.append((underlying, side, option_type, underlying_price))
        return self.selection

    def register_open(self, underlying: str, contract: SelectedContract) -> None:
        self.registered.append((underlying, contract))

    def resolve_active(self, underlying: str) -> SelectedContract | None:
        return self.resolved

    def clear_active(self, underlying: str) -> None:
        self.cleared.append(underlying)


class _StubDataHub:
    def __init__(self) -> None:
        self.quote = {"ltp": 98.5}
        self.requests: list[str] = []

    def get_quote(self, symbol: str) -> dict[str, float]:
        self.requests.append(symbol)
        return dict(self.quote)


def _make_runner(selector: _StubSelector) -> StrategyRunner:
    market_data = MagicMock()
    indicator = MagicMock()
    strategy_manager = MagicMock()
    risk_manager = MagicMock()
    order_manager = MagicMock()
    position_manager = MagicMock()
    data_hub = _StubDataHub()

    cfg = StrategyRunnerConfig(max_trade_history=5)
    runner = StrategyRunner(
        market_data_manager=market_data,
        indicator_engine=indicator,
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        config=cfg,
        data_hub=data_hub,
        strike_selector=selector,
    )
    runner._symbol_state["NIFTY"] = SymbolRuntimeState(symbol="NIFTY", history_limit=5)
    runner._symbol_state["NIFTY"].active = True
    runner._symbol_state["NIFTY"].strategy_data["last_signal"] = {}
    return runner


def test_runner_skips_trade_when_selector_rejects() -> None:
    selector = _StubSelector()
    selector.selection = None
    runner = _make_runner(selector)
    signal = Signal(
        action="BUY",
        symbol="NIFTY",
        quantity=75,
        confidence=0.8,
        reason="entry",
        stop_loss=19500.0,
        take_profit=20000.0,
        metadata={"option_type": "CE"},
    )

    runner._handle_signal(signal, price=19800.0, timestamp=datetime.now(timezone.utc))

    assert selector.selected_for == [("NIFTY", "BUY", "CE", 19800.0)]
    runner._order_manager.place_order.assert_not_called()
    runner._risk_manager.suggest_position_size.assert_not_called()


def test_runner_places_order_with_selected_contract() -> None:
    selector = _StubSelector()
    selection = SelectedContract(
        symbol="NIFTY24JUL19900CE",
        option_type="CE",
        strike=19900,
        expiry=datetime(2024, 7, 18, tzinfo=timezone.utc),
        ltp=112.5,
        delta=0.42,
        metadata={},
    )
    selector.selection = selection
    runner = _make_runner(selector)
    runner._risk_manager.suggest_position_size.return_value = 75
    runner._risk_manager.validate_new_position.return_value = (True, "")
    runner._order_manager.place_order.return_value = "OID123"

    signal = Signal(
        action="BUY",
        symbol="NIFTY",
        quantity=75,
        confidence=0.8,
        reason="entry",
        stop_loss=19500.0,
        take_profit=20000.0,
        metadata={"option_type": "CE"},
    )

    runner._handle_signal(signal, price=19850.0, timestamp=datetime.now(timezone.utc))

    runner._order_manager.place_order.assert_called_once()
    kwargs = runner._order_manager.place_order.call_args.kwargs
    assert kwargs["symbol"] == "NIFTY24JUL19900CE"
    assert kwargs["price"] == selection.ltp
    runner._risk_manager.suggest_position_size.assert_called_once()
    risk_kwargs = runner._risk_manager.suggest_position_size.call_args.kwargs
    assert risk_kwargs["symbol"] == "NIFTY24JUL19900CE"
    assert selector.registered == [("NIFTY", selection)]


def test_runner_buy_signal_allows_put_selection() -> None:
    selector = _StubSelector()
    selection = SelectedContract(
        symbol="NIFTY24JUL19800PE",
        option_type="PE",
        strike=19800,
        expiry=datetime(2024, 7, 18, tzinfo=timezone.utc),
        ltp=108.5,
        delta=-0.35,
        metadata={},
    )
    selector.selection = selection
    runner = _make_runner(selector)
    runner._risk_manager.suggest_position_size.return_value = 60
    runner._risk_manager.validate_new_position.return_value = (True, "")
    runner._order_manager.place_order.return_value = "OID555"

    signal = Signal(
        action="BUY",
        symbol="NIFTY",
        quantity=60,
        confidence=0.75,
        reason="long_put_entry",
        stop_loss=19650.0,
        take_profit=19980.0,
        metadata={"option_type": "PE"},
    )

    runner._handle_signal(signal, price=19820.0, timestamp=datetime.now(timezone.utc))

    runner._order_manager.place_order.assert_called_once()
    kwargs = runner._order_manager.place_order.call_args.kwargs
    assert kwargs["side"] == "BUY"
    assert kwargs["symbol"] == "NIFTY24JUL19800PE"
    assert selector.selected_for == [("NIFTY", "BUY", "PE", 19820.0)]


def test_runner_sell_signal_defaults_to_buy_side() -> None:
    selector = _StubSelector()
    selection = SelectedContract(
        symbol="NIFTY24JUL19800PE",
        option_type="PE",
        strike=19800,
        expiry=datetime(2024, 7, 18, tzinfo=timezone.utc),
        ltp=125.0,
        delta=-0.35,
        metadata={},
    )
    selector.selection = selection
    runner = _make_runner(selector)
    runner._risk_manager.suggest_position_size.return_value = 50
    runner._risk_manager.validate_new_position.return_value = (True, "")
    runner._order_manager.place_order.return_value = "OID321"

    signal = Signal(
        action="SELL",
        symbol="NIFTY",
        quantity=50,
        confidence=0.7,
        reason="bearish_entry",
        stop_loss=19950.0,
        take_profit=19500.0,
        metadata={"option_type": "PE"},
    )

    runner._handle_signal(signal, price=19880.0, timestamp=datetime.now(timezone.utc))

    runner._order_manager.place_order.assert_called_once()
    kwargs = runner._order_manager.place_order.call_args.kwargs
    assert kwargs["side"] == "BUY"
    risk_kwargs = runner._risk_manager.suggest_position_size.call_args.kwargs
    assert risk_kwargs["side"] == "BUY"
    assert selector.selected_for == [("NIFTY", "SELL", "PE", 19880.0)]


def test_runner_bearish_metadata_defaults_to_buy_pe() -> None:
    selector = _StubSelector()
    selection = SelectedContract(
        symbol="NIFTY24JUL19800PE",
        option_type="PE",
        strike=19800,
        expiry=datetime(2024, 7, 18, tzinfo=timezone.utc),
        ltp=118.0,
        delta=-0.34,
        metadata={},
    )
    selector.selection = selection
    runner = _make_runner(selector)
    runner._risk_manager.suggest_position_size.return_value = 30
    runner._risk_manager.validate_new_position.return_value = (True, "")
    runner._order_manager.place_order.return_value = "OID777"

    signal = Signal(
        action="SELL",
        symbol="NIFTY",
        quantity=30,
        confidence=0.65,
        reason="bearish_metadata",
        stop_loss=19920.0,
        take_profit=19550.0,
        metadata={"bias": "bearish", "option_type": "PE"},
    )

    runner._handle_signal(signal, price=19870.0, timestamp=datetime.now(timezone.utc))

    runner._order_manager.place_order.assert_called_once()
    kwargs = runner._order_manager.place_order.call_args.kwargs
    assert kwargs["side"] == "BUY"
    selector_call = selector.selected_for
    assert selector_call == [("NIFTY", "SELL", "PE", 19870.0)]


def test_runner_bearish_sell_premium_places_short_option() -> None:
    selector = _StubSelector(long_only=False)
    selection = SelectedContract(
        symbol="NIFTY24JUL19800PE",
        option_type="PE",
        strike=19800,
        expiry=datetime(2024, 7, 18, tzinfo=timezone.utc),
        ltp=118.0,
        delta=-0.34,
        metadata={},
    )
    selector.selection = selection
    runner = _make_runner(selector)
    runner._risk_manager.suggest_position_size.return_value = 35
    runner._risk_manager.validate_new_position.return_value = (True, "")
    runner._order_manager.place_order.return_value = "OID888"

    signal = Signal(
        action="SELL",
        symbol="NIFTY",
        quantity=35,
        confidence=0.7,
        reason="bearish_premium",
        stop_loss=19940.0,
        take_profit=19520.0,
        metadata={"bias": "bearish", "sell_premium": True, "option_type": "PE"},
    )

    runner._handle_signal(signal, price=19875.0, timestamp=datetime.now(timezone.utc))

    runner._order_manager.place_order.assert_called_once()
    kwargs = runner._order_manager.place_order.call_args.kwargs
    assert kwargs["side"] == "SELL"
    assert selector.selected_for == [("NIFTY", "SELL", "PE", 19875.0)]


def test_runner_sell_premium_enabled_when_allowed() -> None:
    selector = _StubSelector(long_only=False)
    selection = SelectedContract(
        symbol="NIFTY24JUL19900CE",
        option_type="CE",
        strike=19900,
        expiry=datetime(2024, 7, 18, tzinfo=timezone.utc),
        ltp=110.0,
        delta=0.38,
        metadata={},
    )
    selector.selection = selection
    runner = _make_runner(selector)
    runner._risk_manager.suggest_position_size.return_value = 40
    runner._risk_manager.validate_new_position.return_value = (True, "")
    runner._order_manager.place_order.return_value = "OID654"

    signal = Signal(
        action="SELL",
        symbol="NIFTY",
        quantity=40,
        confidence=0.6,
        reason="premium_short",
        stop_loss=20100.0,
        take_profit=19400.0,
        metadata={"sell_premium": True, "option_type": "CE"},
    )

    runner._handle_signal(signal, price=19890.0, timestamp=datetime.now(timezone.utc))

    runner._order_manager.place_order.assert_called_once()
    kwargs = runner._order_manager.place_order.call_args.kwargs
    assert kwargs["side"] == "SELL"
    risk_kwargs = runner._risk_manager.suggest_position_size.call_args.kwargs
    assert risk_kwargs["side"] == "SELL"
    assert selector.selected_for == [("NIFTY", "SELL", "CE", 19890.0)]


def test_runner_closes_position_with_selector_mapping() -> None:
    selector = _StubSelector()
    selection = SelectedContract(
        symbol="NIFTY24JUL19900CE",
        option_type="CE",
        strike=19900,
        expiry=datetime(2024, 7, 18, tzinfo=timezone.utc),
        ltp=105.0,
        delta=0.4,
        metadata={},
    )
    selector.resolved = selection
    runner = _make_runner(selector)
    runner._position_manager.get_position.return_value = SimpleNamespace(quantity=75)
    runner._risk_manager.validate_close_position.return_value = (True, "")
    runner._order_manager.place_reduce_only_exit.return_value = "OID999"

    signal = Signal(
        action="CLOSE_LONG",
        symbol="NIFTY",
        quantity=75,
        confidence=0.9,
        reason="exit",
        stop_loss=0.0,
        take_profit=0.0,
    )

    runner._handle_signal(signal, price=19840.0, timestamp=datetime.now(timezone.utc))

    runner._order_manager.place_order.assert_not_called()
    runner._order_manager.place_reduce_only_exit.assert_called_once()
    intent_arg = runner._order_manager.place_reduce_only_exit.call_args.args[0]
    assert isinstance(intent_arg, ExitIntent)
    assert intent_arg.symbol == "NIFTY24JUL19900CE"
    assert selector.cleared == ["NIFTY"]
