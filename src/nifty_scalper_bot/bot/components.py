"""Core runtime components for the trading bot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dtime, timedelta
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from nifty_scalper_bot.data.rest.client import DummyBrokerClient
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TradeSignal:
    """Simple trade signal container."""

    symbol: str
    side: str
    quantity: int
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ZerodhaKiteClient(DummyBrokerClient):
    """Lightweight Zerodha client placeholder backed by the dummy broker."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str | None = None,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._api_secret = api_secret
        self._access_token = access_token
        self._profile = {"user_name": api_key, "broker": "zerodha"}
        self._positions: list[dict[str, Any]] = []
        self._instruments_loaded = False

    def get_profile(self) -> dict[str, Any]:
        """Return a dummy profile to simulate broker connectivity."""

        return dict(self._profile)

    def load_instruments(self, exchange: str) -> None:
        """Simulate loading an instrument list."""

        LOGGER.info("Loaded instruments for exchange %s", exchange)
        self._instruments_loaded = True

    def get_positions(self) -> list[dict[str, Any]]:
        """Return cached broker positions."""

        return list(self._positions)

    def set_positions(self, positions: Iterable[dict[str, Any]]) -> None:
        """Update cached broker positions (testing helper)."""

        self._positions = list(positions)


class ZerodhaKiteWebSocket:
    """Minimal WebSocket wrapper keeping track of subscriptions."""

    def __init__(
        self,
        *,
        api_key: str,
        access_token: str | None,
        on_tick: Callable[[dict[str, Any]], None] | None,
        on_error: Callable[[Exception], None],
    ) -> None:
        self._api_key = api_key
        self._access_token = access_token
        self._on_tick = on_tick
        self._on_error = on_error
        self._connected = False
        self._subscriptions: set[str] = set()
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def subscriptions(self) -> set[str]:
        return set(self._subscriptions)

    @property
    def on_tick(self) -> Callable[[dict[str, Any]], None] | None:
        return self._on_tick

    @on_tick.setter
    def on_tick(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self._on_tick = callback

    def connect(self) -> None:
        self._connected = True
        LOGGER.info("WebSocket connection established")

    def disconnect(self) -> None:
        self._connected = False
        LOGGER.info("WebSocket connection closed")

    def subscribe(self, symbols: Iterable[str]) -> None:
        with self._lock:
            self._subscriptions.update(symbols)
        LOGGER.info("Subscribed to symbols: %s", ", ".join(sorted(symbols)))

    def unsubscribe(self, symbols: Iterable[str]) -> None:
        with self._lock:
            for symbol in symbols:
                self._subscriptions.discard(symbol)
        LOGGER.info("Unsubscribed symbols: %s", ", ".join(sorted(symbols)))

    def emit_tick(self, tick: dict[str, Any]) -> None:
        callback = self._on_tick
        if callback is None:
            return
        try:
            callback(tick)
        except Exception as exc:  # noqa: BLE001
            self._on_error(exc)


class MarketDataManager:
    """Manage quote retrieval and WebSocket subscriptions."""

    def __init__(
        self,
        broker_client: ZerodhaKiteClient,
        websocket_client: ZerodhaKiteWebSocket,
    ) -> None:
        self._broker = broker_client
        self._ws = websocket_client
        self._latest: dict[str, dict[str, Any]] = {}
        self._ws.on_tick = self._handle_tick

    def _handle_tick(self, tick: dict[str, Any]) -> None:
        symbol = tick.get("symbol")
        if symbol:
            self._latest[symbol] = tick

    def subscribe(self, symbol: str) -> None:
        self._ws.subscribe([symbol])

    def unsubscribe(self, symbol: str) -> None:
        self._ws.unsubscribe([symbol])
        self._latest.pop(symbol, None)

    def start(self) -> None:
        if not self._ws.connected:
            self._ws.connect()

    def stop(self) -> None:
        if self._ws.connected:
            self._ws.disconnect()

    def get_latest_price(self, symbol: str) -> float:
        tick = self._latest.get(symbol)
        if tick:
            return float(tick["ltp"])
        quote = self._broker.get_quote(symbol)
        self._latest[symbol] = quote
        return float(quote["ltp"])


class IndicatorEngine:
    """Compute indicators for symbols (placeholder implementation)."""

    def calculate(self, symbol: str, price: float) -> dict[str, float]:
        return {"price": price, "symbol_hash": float(hash(symbol) % 1000)}


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: int
    average_price: float


@dataclass(slots=True)
class PendingOrder:
    order_id: str
    symbol: str
    side: str
    quantity: int
    status: str = "open"


class PositionManager:
    """Persist and manage open positions and pending orders."""

    def __init__(self, state_file: str) -> None:
        self._state_file = Path(state_file)
        self._positions: dict[str, Position] = {}
        self._pending: dict[str, PendingOrder] = {}
        self._lock = threading.Lock()

    def load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
        except json.JSONDecodeError as exc:  # noqa: PERF203 - executed rarely
            LOGGER.warning("Failed to load state file %s: %s", self._state_file, exc)
            return
        positions = {
            item["symbol"]: Position(
                symbol=item["symbol"],
                quantity=int(item["quantity"]),
                average_price=float(item["average_price"]),
            )
            for item in data.get("positions", [])
        }
        pending = {
            item["order_id"]: PendingOrder(
                order_id=item["order_id"],
                symbol=item["symbol"],
                side=item["side"],
                quantity=int(item["quantity"]),
                status=item.get("status", "open"),
            )
            for item in data.get("pending", [])
        }
        with self._lock:
            self._positions = positions
            self._pending = pending

    def save_state(self) -> None:
        data = {
            "positions": [asdict(pos) for pos in self._positions.values()],
            "pending": [asdict(order) for order in self._pending.values()],
        }
        self._state_file.write_text(json.dumps(data, indent=2))

    def get_all_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    def get_pending_orders(self) -> list[PendingOrder]:
        with self._lock:
            return list(self._pending.values())

    def record_position(self, symbol: str, quantity: int, price: float) -> None:
        with self._lock:
            self._positions[symbol] = Position(
                symbol=symbol, quantity=quantity, average_price=price
            )

    def close_position(self, symbol: str) -> None:
        with self._lock:
            self._positions.pop(symbol, None)

    def record_pending_order(self, order: PendingOrder) -> None:
        with self._lock:
            self._pending[order.order_id] = order

    def update_order_status(self, order_id: str, status: str) -> None:
        with self._lock:
            order = self._pending.get(order_id)
            if not order:
                return
            order.status = status
            if status in {"cancelled", "filled"}:
                self._pending.pop(order_id, None)

    def clear_pending_orders(self) -> None:
        with self._lock:
            self._pending.clear()


class RiskManager:
    """Perform rudimentary risk checks before order placement."""

    def __init__(
        self,
        *,
        risk_config: Any,
        position_manager: PositionManager,
        initial_balance: float | None = None,
    ) -> None:
        self._cfg = risk_config
        self._positions = position_manager
        self._initial_balance = float(initial_balance or 0.0)

    def allow_order(self, signal: TradeSignal, price: float) -> bool:
        max_notional = getattr(self._cfg, "max_order_notional", None)
        if max_notional is not None and price * signal.quantity > float(max_notional):
            LOGGER.warning(
                "Order blocked due to max notional limit for %s", signal.symbol
            )
            return False
        if signal.side.lower() == "sell" and not getattr(
            self._cfg, "allow_short", True
        ):
            position_symbols = {
                pos.symbol for pos in self._positions.get_all_positions()
            }
            if signal.symbol not in position_symbols:
                LOGGER.warning("Short selling disabled for %s", signal.symbol)
                return False
        return True


class OrderManager:
    """Submit and monitor orders via the broker."""

    def __init__(
        self,
        *,
        broker_client: ZerodhaKiteClient,
        position_manager: PositionManager,
        rate_limiter: Any,
    ) -> None:
        self._broker = broker_client
        self._positions = position_manager
        self._rate_limiter = rate_limiter
        self._monitoring = False

    def start_monitoring(self) -> None:
        self._monitoring = True
        LOGGER.info("Order monitoring started")

    def stop_monitoring(self) -> None:
        self._monitoring = False
        LOGGER.info("Order monitoring stopped")

    def submit_order(self, signal: TradeSignal, price: float) -> str:
        payload = {
            "symbol": signal.symbol,
            "side": signal.side,
            "qty": signal.quantity,
            "price": price,
        }
        response = self._broker.place_order(payload)
        order_id = str(response.get("order_id", f"SIM-{int(time.time() * 1000)}"))
        pending = PendingOrder(
            order_id=order_id,
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
        )
        self._positions.record_pending_order(pending)
        return order_id

    def cancel_order(self, order_id: str) -> None:
        self._positions.update_order_status(order_id, "cancelled")
        LOGGER.info("Cancelled order %s", order_id)


@dataclass(slots=True)
class StrategyConfig:
    name: str
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)


class BaseStrategy:
    """Base class for trading strategies."""

    def __init__(self, name: str, parameters: dict[str, Any] | None = None) -> None:
        self.name = name
        self.parameters = parameters or {}
        self.enabled = True

    def generate_signals(
        self, symbol: str, indicators: dict[str, float]
    ) -> list[TradeSignal]:
        return []


class RSIMeanReversionStrategy(BaseStrategy):
    """Placeholder RSI mean reversion strategy."""

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        super().__init__("RSIMeanReversion", parameters)


class EMACrossoverStrategy(BaseStrategy):
    """Placeholder EMA crossover strategy."""

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        super().__init__("EMACrossover", parameters)


class StrategyManager:
    """Combine multiple strategies and generate aggregate signals."""

    def __init__(
        self,
        *,
        strategies: Iterable[BaseStrategy],
        indicator_engine: IndicatorEngine,
    ) -> None:
        self._strategies = list(strategies)
        self._indicator_engine = indicator_engine

    def generate_signals(self, symbol: str, price: float) -> list[TradeSignal]:
        indicators = self._indicator_engine.calculate(symbol, price)
        signals: list[TradeSignal] = []
        for strategy in self._strategies:
            if not getattr(strategy, "enabled", True):
                continue
            signals.extend(strategy.generate_signals(symbol, indicators))
        return signals


@dataclass(slots=True)
class StrategyRunnerConfig:
    poll_interval: float = 1.0
    signal_cooldown: float = 0.0
    trailing_stop_pct: float = 0.0
    max_position_age: float | None = None
    market_close_time: dtime | None = dtime(hour=15, minute=25)
    market_close_timezone: str = "Asia/Kolkata"
    market_close_buffer_minutes: int = 10


class StrategyRunner:
    """Background loop evaluating strategies for subscribed symbols."""

    def __init__(
        self,
        *,
        market_data_manager: MarketDataManager,
        indicator_engine: IndicatorEngine,
        signal_generator: StrategyManager,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        position_manager: PositionManager,
        config: StrategyRunnerConfig | None = None,
    ) -> None:
        self._market_data = market_data_manager
        self._indicator_engine = indicator_engine
        self._signals = signal_generator
        self._risk = risk_manager
        self._orders = order_manager
        self._positions = position_manager
        self._config = config or StrategyRunnerConfig()
        self._symbols: set[str] = set()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_signal_time: dict[str, float] = {}
        self._position_state: dict[str, dict[str, Any]] = {}
        self._trade_results: list[float] = []
        self._gross_profit: float = 0.0
        self._gross_loss: float = 0.0
        self._net_profit: float = 0.0
        self._winning_trades: int = 0
        self._losing_trades: int = 0
        self._breakeven_trades: int = 0
        self._market_close_exit_triggered = False
        self._emergency_exit_requested = False

    def add_symbol(self, symbol: str) -> None:
        if symbol in self._symbols:
            return
        self._symbols.add(symbol)
        self._market_data.subscribe(symbol)

    def remove_symbol(self, symbol: str) -> None:
        if symbol not in self._symbols:
            return
        self._symbols.remove(symbol)
        self._market_data.unsubscribe(symbol)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._market_data.start()
        self._market_close_exit_triggered = False
        self._emergency_exit_requested = False
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="StrategyRunner"
        )
        self._thread.start()
        LOGGER.info("Strategy runner started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._market_data.stop()
        LOGGER.info("Strategy runner stopped")

    def pause_trading(self) -> None:
        self._pause_event.set()
        LOGGER.info("Strategy runner paused")

    def resume_trading(self) -> None:
        self._pause_event.clear()
        LOGGER.info("Strategy runner resumed")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.5)
                continue
            for symbol in list(self._symbols):
                try:
                    price = self._market_data.get_latest_price(symbol)
                    self._on_tick(symbol, price)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Strategy runner error for %s: %s", symbol, exc)
            time.sleep(self._config.poll_interval)

    def _on_tick(self, symbol: str, price: float) -> None:
        """Process a new market data tick for *symbol*."""

        open_positions = {
            pos.symbol: pos for pos in self._positions.get_all_positions()
        }
        position = open_positions.get(symbol)
        if position and symbol not in self._position_state:
            direction = "LONG" if position.quantity >= 0 else "SHORT"
            self._position_state[symbol] = {
                "side": direction,
                "quantity": abs(position.quantity),
                "entry_price": position.average_price,
                "entry_time": time.time(),
                "stop_loss": None,
                "take_profit": None,
                "trailing_stop": None,
                "extreme_price": position.average_price,
            }

        self._update_trailing_stop(symbol, price, position)
        if self._check_exit_conditions(symbol, price, position, open_positions):
            return

        if not self._should_generate_signal(symbol):
            return

        signals = self._signals.generate_signals(symbol, price) or []
        for signal in signals:
            if signal.quantity <= 0:
                LOGGER.debug("Ignoring non-positive quantity signal for %s", symbol)
                continue
            if not self._risk.allow_order(signal, price):
                LOGGER.debug(
                    "Risk manager blocked %s signal for %s", signal.side, symbol
                )
                continue
            self._execute_signal(signal, price, open_positions)

    def _should_generate_signal(self, symbol: str) -> bool:
        """Return ``True`` if cooldown permits generating a new signal."""

        cooldown = max(self._config.signal_cooldown, 0.0)
        if cooldown == 0.0:
            return True
        last_signal_ts = self._last_signal_time.get(symbol)
        if last_signal_ts is None:
            return True
        return (time.time() - last_signal_ts) >= cooldown

    def _update_trailing_stop(
        self, symbol: str, price: float, position: Position | None
    ) -> None:
        """Update trailing stop levels when positions move favourably."""

        if position is None:
            return
        state = self._position_state.get(symbol)
        if state is None:
            return
        trailing_pct = max(self._config.trailing_stop_pct, 0.0)
        if trailing_pct == 0.0:
            return
        side = state.get("side", "LONG")
        extreme_price = state.get("extreme_price", state.get("entry_price", price))
        if side == "LONG":
            if price <= extreme_price:
                return
            state["extreme_price"] = price
            new_stop = price * (1 - trailing_pct)
            current_stop = state.get("trailing_stop")
            if current_stop is None or new_stop > current_stop:
                state["trailing_stop"] = new_stop
                prev_stop_loss = state.get("stop_loss")
                state["stop_loss"] = (
                    new_stop
                    if prev_stop_loss is None
                    else max(prev_stop_loss, new_stop)
                )
                LOGGER.debug("Raised trailing stop for %s to %.2f", symbol, new_stop)
        else:  # SHORT
            if price >= extreme_price:
                return
            state["extreme_price"] = price
            new_stop = price * (1 + trailing_pct)
            current_stop = state.get("trailing_stop")
            if current_stop is None or new_stop < current_stop:
                state["trailing_stop"] = new_stop
                prev_stop_loss = state.get("stop_loss")
                state["stop_loss"] = (
                    new_stop
                    if prev_stop_loss is None
                    else min(prev_stop_loss, new_stop)
                )
                LOGGER.debug("Tightened trailing stop for %s to %.2f", symbol, new_stop)

    def _check_exit_conditions(
        self,
        symbol: str,
        price: float,
        position: Position | None,
        open_positions: dict[str, Position],
    ) -> bool:
        """Evaluate whether existing positions require an exit order."""

        if self._emergency_exit_requested:
            executed = self._liquidate_positions(
                open_positions, reason="emergency_exit"
            )
            self._emergency_exit_requested = False
            if executed:
                LOGGER.warning("Emergency exit triggered; closed open positions")
                return True

        if position is None:
            return False

        state = self._position_state.get(symbol, {})
        side = state.get("side") or ("LONG" if position.quantity >= 0 else "SHORT")
        stop_loss = state.get("stop_loss")
        take_profit = state.get("take_profit")
        entry_time = state.get("entry_time")

        if stop_loss is not None:
            if (side == "LONG" and price <= stop_loss) or (
                side == "SHORT" and price >= stop_loss
            ):
                return self._dispatch_exit(
                    symbol, position, price, "stop_loss", open_positions
                )

        if take_profit is not None:
            if (side == "LONG" and price >= take_profit) or (
                side == "SHORT" and price <= take_profit
            ):
                return self._dispatch_exit(
                    symbol, position, price, "take_profit", open_positions
                )

        max_age = self._config.max_position_age
        if max_age is not None and entry_time is not None:
            if time.time() - entry_time >= max_age:
                return self._dispatch_exit(
                    symbol, position, price, "time_stop", open_positions
                )

        if self.is_running() and self._close_all_positions_before_market_close(
            open_positions
        ):
            return True

        return False

    def _dispatch_exit(
        self,
        symbol: str,
        position: Position,
        price: float,
        reason: str,
        open_positions: dict[str, Position],
    ) -> bool:
        quantity = abs(position.quantity)
        if quantity == 0:
            return False
        exit_side = "SELL" if position.quantity >= 0 else "BUY"
        exit_signal = TradeSignal(symbol=symbol, side=exit_side, quantity=quantity)
        if not self._risk.allow_order(exit_signal, price):
            LOGGER.warning("Risk manager rejected %s exit for %s", reason, symbol)
            return False
        LOGGER.info("Exit condition '%s' triggered for %s", reason, symbol)
        self._execute_signal(exit_signal, price, open_positions)
        return True

    def _liquidate_positions(
        self, open_positions: dict[str, Position], reason: str
    ) -> bool:
        executed = False
        for sym, pos in list(open_positions.items()):
            price = self._market_data.get_latest_price(sym)
            if self._dispatch_exit(sym, pos, price, reason, open_positions):
                executed = True
        return executed

    def _close_all_positions_before_market_close(
        self, open_positions: dict[str, Position]
    ) -> bool:
        if self._market_close_exit_triggered:
            return False
        close_time = self._config.market_close_time
        if close_time is None:
            return False
        tz = self._get_market_timezone()
        now = datetime.now(tz)
        close_dt = datetime.combine(now.date(), close_time, tz)
        buffer = timedelta(minutes=max(self._config.market_close_buffer_minutes, 0))
        if now < close_dt - buffer:
            return False
        executed = self._liquidate_positions(open_positions, reason="market_close")
        if executed:
            self._market_close_exit_triggered = True
            LOGGER.info("Closed positions ahead of market close")
        return executed

    def _get_market_timezone(self) -> ZoneInfo:
        tz_name = self._config.market_close_timezone or "UTC"
        try:
            return ZoneInfo(tz_name)
        except Exception:  # noqa: BLE001 - fallback to UTC
            LOGGER.warning("Invalid timezone '%s'; defaulting to UTC", tz_name)
            return ZoneInfo("UTC")

    def _execute_signal(
        self,
        signal: TradeSignal,
        price: float,
        open_positions: dict[str, Position],
    ) -> str | None:
        """Submit *signal* as an order and update performance trackers."""

        normalized_side = signal.side.upper()
        if normalized_side == "HOLD":
            return None
        if normalized_side in {"CLOSE_LONG", "CLOSE_SHORT"}:
            normalized_side = "SELL" if normalized_side == "CLOSE_LONG" else "BUY"

        exec_signal = TradeSignal(
            symbol=signal.symbol,
            side=normalized_side,
            quantity=signal.quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata=dict(signal.metadata),
        )
        if exec_signal.quantity <= 0:
            LOGGER.debug("Ignoring zero-quantity signal for %s", exec_signal.symbol)
            return None

        order_id = self._orders.submit_order(exec_signal, price)
        LOGGER.info(
            "Submitted order %s for %s %s @ %.2f",
            order_id,
            exec_signal.side,
            exec_signal.symbol,
            price,
        )
        self._last_signal_time[exec_signal.symbol] = time.time()

        position = open_positions.get(exec_signal.symbol)
        side = exec_signal.side.upper()
        if side == "BUY":
            if position and position.quantity < 0:
                self._finalize_trade(
                    exec_signal.symbol, price, position, open_positions
                )
            else:
                qty = exec_signal.quantity
                self._positions.record_position(exec_signal.symbol, qty, price)
                open_positions[exec_signal.symbol] = Position(
                    symbol=exec_signal.symbol,
                    quantity=qty,
                    average_price=price,
                )
                self._position_state[exec_signal.symbol] = self._build_position_state(
                    symbol=exec_signal.symbol,
                    side="LONG",
                    quantity=qty,
                    entry_price=price,
                    stop_loss=exec_signal.stop_loss,
                    take_profit=exec_signal.take_profit,
                )
        elif side == "SELL":
            if position and position.quantity > 0:
                self._finalize_trade(
                    exec_signal.symbol, price, position, open_positions
                )
            else:
                qty = exec_signal.quantity
                self._positions.record_position(exec_signal.symbol, -qty, price)
                open_positions[exec_signal.symbol] = Position(
                    symbol=exec_signal.symbol,
                    quantity=-qty,
                    average_price=price,
                )
                self._position_state[exec_signal.symbol] = self._build_position_state(
                    symbol=exec_signal.symbol,
                    side="SHORT",
                    quantity=qty,
                    entry_price=price,
                    stop_loss=exec_signal.stop_loss,
                    take_profit=exec_signal.take_profit,
                )
        return order_id

    def _build_position_state(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> dict[str, Any]:
        state = {
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_time": time.time(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop": None,
            "extreme_price": entry_price,
        }
        trailing_pct = max(self._config.trailing_stop_pct, 0.0)
        if trailing_pct > 0:
            if side == "LONG":
                trail = entry_price * (1 - trailing_pct)
                state["trailing_stop"] = trail
                if stop_loss is None or trail > stop_loss:
                    state["stop_loss"] = trail
            else:
                trail = entry_price * (1 + trailing_pct)
                state["trailing_stop"] = trail
                if stop_loss is None or trail < stop_loss:
                    state["stop_loss"] = trail
        return state

    def _finalize_trade(
        self,
        symbol: str,
        exit_price: float,
        position: Position,
        open_positions: dict[str, Position],
    ) -> None:
        pnl = self._calculate_pnl(symbol, exit_price, position)
        self._record_trade_outcome(symbol, pnl)
        self._positions.close_position(symbol)
        open_positions.pop(symbol, None)
        self._position_state.pop(symbol, None)

    def _calculate_pnl(
        self, symbol: str, exit_price: float, position: Position | None
    ) -> float:
        state = self._position_state.get(symbol, {})
        entry_price = float(
            state.get("entry_price")
            or (position.average_price if position else exit_price)
        )
        quantity = int(
            state.get("quantity") or (abs(position.quantity) if position else 0)
        )
        if quantity <= 0:
            return 0.0
        side = state.get("side") or (
            "LONG" if position and position.quantity >= 0 else "SHORT"
        )
        direction = 1 if side == "LONG" else -1
        return (exit_price - entry_price) * quantity * direction

    def _record_trade_outcome(self, symbol: str, pnl: float) -> None:
        self._trade_results.append(pnl)
        self._net_profit += pnl
        if pnl > 0:
            self._winning_trades += 1
            self._gross_profit += pnl
            LOGGER.info("Closed %s with profit %.2f", symbol, pnl)
        elif pnl < 0:
            self._losing_trades += 1
            self._gross_loss += abs(pnl)
            LOGGER.info("Closed %s with loss %.2f", symbol, pnl)
        else:
            self._breakeven_trades += 1
            LOGGER.info("Closed %s breakeven", symbol)

    def trigger_emergency_exit(self) -> None:
        """Request liquidation of all positions at the next tick."""

        self._emergency_exit_requested = True

    def get_performance_metrics(self) -> dict[str, float]:
        """Return aggregated trade performance metrics."""

        total_trades = len(self._trade_results)
        win_rate = (self._winning_trades / total_trades) if total_trades else 0.0
        avg_profit = (
            (self._gross_profit / self._winning_trades) if self._winning_trades else 0.0
        )
        avg_loss = (
            -(self._gross_loss / self._losing_trades) if self._losing_trades else 0.0
        )
        profit_factor = 0.0
        if self._gross_loss == 0.0:
            profit_factor = float("inf") if self._gross_profit > 0 else 0.0
        else:
            profit_factor = self._gross_profit / self._gross_loss
        average_trade = (self._net_profit / total_trades) if total_trades else 0.0
        return {
            "total_trades": float(total_trades),
            "winning_trades": float(self._winning_trades),
            "losing_trades": float(self._losing_trades),
            "breakeven_trades": float(self._breakeven_trades),
            "win_rate": win_rate,
            "average_profit": avg_profit,
            "average_loss": avg_loss,
            "net_profit": self._net_profit,
            "profit_factor": profit_factor,
            "average_trade": average_trade,
        }


__all__ = [
    "EMACrossoverStrategy",
    "IndicatorEngine",
    "MarketDataManager",
    "OrderManager",
    "PendingOrder",
    "Position",
    "PositionManager",
    "RiskManager",
    "RSIMeanReversionStrategy",
    "StrategyConfig",
    "StrategyManager",
    "StrategyRunner",
    "StrategyRunnerConfig",
    "TradeSignal",
    "ZerodhaKiteClient",
    "ZerodhaKiteWebSocket",
]
