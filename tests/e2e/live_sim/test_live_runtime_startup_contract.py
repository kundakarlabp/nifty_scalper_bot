from __future__ import annotations

import asyncio
import inspect
from dataclasses import asdict
from datetime import date, timedelta
import time
from typing import Any

import pandas as pd
import pytest

from nifty_scalper_bot.core import app as core_app
from nifty_scalper_bot.core.app import BotContext, initialize_components
from nifty_scalper_bot.core.instrument_manager import InstrumentManager
from nifty_scalper_bot.strategies.signal_generator import RSIMeanReversionStrategy
from nifty_scalper_bot.utils.market_hours import MarketState
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.risk.risk_manager import RiskManager
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner

pytestmark = pytest.mark.e2e_live_sim


class _NoNetworkBroker:
    is_simulated_adapter = True
    """Broker stand-in for production composition startup only.

    It implements the broker methods that app.initialize_components wires into
    InstrumentManager, MDM, RiskManager, OrderManager and reconciliation. Any
    unknown method access fails loudly so a real network surface cannot be
    reached silently.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.api_key = kwargs.get("api_key") or (args[0] if args else "sim")
        self.client = self
        self.auth_invalid = False
        self._positions: list[dict[str, Any]] = []
        self._orders: list[dict[str, Any]] = []
        self._order_updates: list[dict[str, Any]] = []
        self._next_order_id = 1
        self._order_update_callback = None
        self._instruments = _instrument_dump()

    def preload_instruments(self) -> None:
        return None

    def instruments(self, exchange: str | None = None) -> list[dict[str, Any]]:
        rows = list(self._instruments)
        if exchange:
            rows = [r for r in rows if str(r.get("exchange")) == exchange]
        return rows

    def get_available_balance(self, *args: Any, **kwargs: Any) -> float:
        return 1_000_000.0

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self._positions)

    def register_order_update_callback(self, callback: Any) -> None:
        self._order_update_callback = callback

    def place_order(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        payload = dict(args[0]) if args and isinstance(args[0], dict) else dict(kwargs)
        order_id = f"SIM-{self._next_order_id:04d}"
        self._next_order_id += 1
        symbol = (
            payload.get("symbol")
            or payload.get("tradingsymbol")
            or payload.get("exchange_symbol")
            or "UNKNOWN"
        )
        side = (payload.get("side") or payload.get("transaction_type") or "BUY").upper()
        qty = int(float(payload.get("quantity") or 0))
        price = float(payload.get("price") or payload.get("limit_price") or 100.0)
        row = {
            "order_id": order_id,
            "symbol": symbol,
            "tradingsymbol": symbol,
            "transaction_type": side,
            "side": side,
            "quantity": qty,
            "filled_quantity": 0,
            "average_price": 0.0,
            "price": price,
            "status": "OPEN",
            "tag": payload.get("tag"),
            "payload": payload,
        }
        self._orders.append(row)
        self._order_updates.append(dict(row))
        return {"order_id": order_id, "status": "success"}

    def fill_order(self, order_id: str, *, average_price: float | None = None) -> None:
        for order in self._orders:
            if order["order_id"] != order_id:
                continue
            order["status"] = "COMPLETE"
            order["filled_quantity"] = int(order["quantity"])
            order["average_price"] = float(average_price or order.get("price") or 100.0)
            qty = int(order["quantity"])
            signed = qty if order["transaction_type"] == "BUY" else -qty
            existing = next(
                (
                    p
                    for p in self._positions
                    if p["tradingsymbol"] == order["tradingsymbol"]
                ),
                None,
            )
            if existing is None:
                existing = {
                    "tradingsymbol": order["tradingsymbol"],
                    "symbol": order["symbol"],
                    "quantity": 0,
                    "average_price": order["average_price"],
                    "product": "MIS",
                }
                self._positions.append(existing)
            existing["quantity"] = int(existing.get("quantity", 0)) + signed
            existing["average_price"] = order["average_price"]
            update = dict(order)
            self._order_updates.append(update)
            if self._order_update_callback is not None:
                self._order_update_callback(order_id, update)
            return
        raise KeyError(order_id)

    def cancel_order(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        order_id = str(kwargs.get("order_id") or (args[0] if args else ""))
        for order in self._orders:
            if order["order_id"] == order_id:
                order["status"] = "CANCELLED"
                return {"order_id": order_id, "status": "cancelled"}
        return {"order_id": order_id, "status": "not_found"}

    def modify_order(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        order_id = str(kwargs.get("order_id") or (args[0] if args else ""))
        return {"order_id": order_id, "status": "modified"}

    def get_orders(self) -> list[dict[str, Any]]:
        return list(self._orders)

    def is_connected(self) -> bool:
        return True

    def set_auth_failure_callback(self, callback: Any) -> None:
        self._auth_failure_callback = callback

    def get_instrument_token(self, symbol: str) -> int:
        if symbol == "NSE:NIFTY":
            return 256265
        normalized = str(symbol).split(":", 1)[-1]
        for row in self._instruments:
            if row["tradingsymbol"] == normalized:
                return int(row["instrument_token"])
        raise KeyError(symbol)

    def get_ltp(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        return {symbol: {"last_price": 25000.0} for symbol in symbols}

    def get_ltp_bulk(self, tokens: list[int]) -> dict[int, dict[str, float]]:
        return {int(token): {"last_price": 25000.0} for token in tokens}

    def ltp(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        return self.get_ltp(symbols)


class _NoNetworkWebSocketManager:
    is_simulated_adapter = True
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._callbacks: dict[str, Any] = {}
        self._subscribed_tokens: set[int] = set()
        self._market_data_manager = None
        self.connected = False

    def set_callbacks(self, **callbacks: Any) -> None:
        self._callbacks.update(callbacks)

    def connect(self) -> None:
        self.connected = True
        callback = self._callbacks.get("on_connect") or self._callbacks.get("on_open")
        if callback is not None:
            result = callback()
            if inspect.isawaitable(result):
                asyncio.get_event_loop().run_until_complete(result)

    def publish_tick(self, tick: dict[str, Any]) -> None:
        if not self.connected:
            raise RuntimeError("simulated websocket is not connected")
        callback = self._callbacks.get("on_ticks") or self._callbacks.get("on_tick")
        if callback is None:
            # Production WebSocketManager routes through its injected MDM reference.
            mdm = getattr(self, "_market_data_manager", None)
            if mdm is None:
                raise RuntimeError("production websocket tick callback was not registered")
            mdm.process_ticks([tick])
            drain = getattr(mdm, "_drain_tick_queue_sync", None)
            if callable(drain):
                drain()
            return
        try:
            result = callback([tick])
        except TypeError:
            result = callback(tick)
        if inspect.isawaitable(result):
            asyncio.get_event_loop().run_until_complete(result)

    def set_fallback_callbacks(self, **callbacks: Any) -> None:
        self._fallback_callbacks = callbacks

    def subscribe_tokens(
        self, tokens: list[int] | tuple[int, ...], mode: str = "full"
    ) -> None:
        self._subscribed_tokens.update(int(token) for token in tokens)

    async def subscribe(self, tokens: list[int] | tuple[int, ...]) -> None:
        self.subscribe_tokens(tokens)

    def is_connected(self) -> bool:
        return bool(self.connected)

    def backlog_size(self) -> int:
        return 0


class _NoNetworkRobustProvider:
    is_simulated_adapter = True
    def __init__(self, broker_client: Any, *args: Any, **kwargs: Any) -> None:
        self.client = broker_client
        self._broker = broker_client
        self.auth_invalid = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def is_connected(self) -> bool:
        return True


def _instrument_dump() -> list[dict[str, Any]]:
    expiry = date.today() + timedelta(days=30)
    rows: list[dict[str, Any]] = [
        {
            "instrument_token": 900001,
            "exchange": "NFO",
            "tradingsymbol": "NIFTY26AUGFUT",
            "name": "NIFTY",
            "expiry": expiry,
            "instrument_type": "FUT",
            "strike": 0,
            "lot_size": 65,
            "tick_size": 0.05,
        }
    ]
    token = 910000
    for strike in (24900, 24950, 25000, 25050, 25100):
        for side in ("CE", "PE"):
            token += 1
            rows.append(
                {
                    "instrument_token": token,
                    "exchange": "NFO",
                    "tradingsymbol": f"NIFTY26AUG{strike}{side}",
                    "name": "NIFTY",
                    "expiry": expiry,
                    "instrument_type": side,
                    "strike": float(strike),
                    "lot_size": 65,
                    "tick_size": 0.05,
                }
            )
    return rows


def _runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from nifty_scalper_bot.config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("ALLOW_REAL_BROKER", "false")
    monkeypatch.setenv("ALLOW_NETWORK", "false")
    monkeypatch.setenv("BROKER_SIMULATION", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STREAM__MODE", "websocket")
    monkeypatch.setenv("WEBSOCKET__DISABLED", "false")
    monkeypatch.setenv("TELEGRAM__ENABLED", "false")
    monkeypatch.setenv("TELEGRAM__WEBHOOK_ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("ALLOW_OFFHOURS_TESTING", "true")
    monkeypatch.setenv("READINESS_CONTEXT_MIN_BARS", "20")
    monkeypatch.setenv("READINESS_OPTION_EXEC_MIN_BARS", "20")
    monkeypatch.setenv("READINESS_OPTION_EVAL_MIN_BARS", "5")
    monkeypatch.setenv("GLOBAL_MIN_SIGNAL_CONFIDENCE", "0.1")
    monkeypatch.setenv("REGIME_GATE_ENABLED", "false")
    monkeypatch.setenv("ALLOW_MARKET_ENTRY", "false")
    monkeypatch.setenv("MAX_ACTIVE_OPTION_SYMBOLS", "2")
    monkeypatch.setenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", "true")
    monkeypatch.setenv("STRATEGY_SINGLE_VOTE_SCALP_MIN", "0")
    monkeypatch.setenv("STRATEGY_SINGLE_VOTE_MIN_CONFIDENCE", "0")
    monkeypatch.setenv("STRATEGY_TRIGGER_MIN_SCORE", "0")
    monkeypatch.setenv("STRATEGY_ALLOW_SELECTED_OPTION_SINGLE_VOTE", "true")
    monkeypatch.setenv("STRATEGY_SELECTED_OPTION_SINGLE_VOTE_MIN_SCORE", "0")
    monkeypatch.setenv("STRATEGY_MIN_TRADE_QUALITY_LIVE", "0")
    monkeypatch.setenv("STRATEGY_MIN_TRADE_QUALITY_LIVE_SIMULATION", "0")
    monkeypatch.setenv("STRATEGY_MIN_TRADE_QUALITY_SHADOW", "0")
    get_settings.cache_clear()


def _patch_no_network_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core_app, "ZerodhaKiteClient", _NoNetworkBroker)
    monkeypatch.setattr(core_app, "RobustDataProvider", _NoNetworkRobustProvider)
    monkeypatch.setattr(core_app, "WebSocketManager", _NoNetworkWebSocketManager)
    monkeypatch.setattr(core_app, "start_background_tasks", lambda *a, **k: [])
    monkeypatch.setattr(core_app, "schedule_instrument_refresh", lambda *a, **k: None)
    monkeypatch.setattr(core_app, "start_watchdog", lambda *a, **k: None)
    monkeypatch.setattr(core_app, "get_market_state", lambda: MarketState.OPEN)
    monkeypatch.setattr(core_app, "get_runtime_market_mode", lambda: "OPEN")
    import nifty_scalper_bot.strategies.runner as runner_mod
    import nifty_scalper_bot.execution.order_manager_core as om_core

    monkeypatch.setattr(runner_mod, "is_market_hours_cached", lambda: True)
    monkeypatch.setattr(runner_mod, "is_market_open_now", lambda: True)
    monkeypatch.setattr(om_core, "get_time_status", lambda: (True, "open"))

    def _build_test_strategies(settings, indicator_engine):
        del settings, indicator_engine
        strategy = RSIMeanReversionStrategy(
            oversold_threshold=95, overbought_threshold=99, default_quantity=1
        )
        strategy.get_required_indicators = lambda: ["rsi", "atr"]  # type: ignore[method-assign]
        return [strategy]

    monkeypatch.setattr(core_app, "build_elite_strategies", _build_test_strategies)


def _bars(
    end: pd.Timestamp, count: int, start: float, step: float
) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": end - pd.Timedelta(minutes=count - idx),
            "open": start + idx * step,
            "high": start + idx * step + abs(step) + 0.2,
            "low": start + idx * step - abs(step) - 0.2,
            "close": start + idx * step + step,
            "volume": 1000 + idx,
        }
        for idx in range(count)
    ]


def _as_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dict("records")


async def _initialize_runtime() -> BotContext:
    return initialize_components()


def _publish_tick(ctx: BotContext, symbol: str, token: int, price: float) -> None:
    now = pd.Timestamp.now(tz="UTC")
    tick = {
        "instrument_token": int(token),
        "symbol": symbol,
        "last_price": float(price),
        "ltp": float(price),
        "bid": float(price) - 0.05,
        "ask": float(price) + 0.05,
        "depth": {
            "buy": [{"price": float(price) - 0.05, "quantity": 1000}],
            "sell": [{"price": float(price) + 0.05, "quantity": 1000}],
        },
        "volume": 10000,
        "timestamp": now.to_pydatetime(),
        "exchange_timestamp": now.to_pydatetime(),
        "timestamp_ms": float(now.timestamp() * 1000.0),
        "source": "websocket",
    }
    assert ctx.websocket_manager is not None
    ctx.websocket_manager.publish_tick(tick)


@pytest.mark.simulation_component
def test_live_simulation_rejects_unmarked_broker(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")

    class Broker:
        pass

    with pytest.raises(RuntimeError, match="simulated broker"):
        core_app._assert_live_simulation_adapter_safe(  # noqa: SLF001 - focused safety hook
            Broker(), component="broker"
        )


@pytest.mark.simulation_component
def test_live_simulation_blocks_real_websocket_adapter(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")

    class WebSocketManager:
        pass

    with pytest.raises(RuntimeError, match="simulated websocket"):
        core_app._assert_live_simulation_adapter_safe(  # noqa: SLF001 - focused safety hook
            WebSocketManager(), component="websocket"
        )


@pytest.mark.simulation_component
def test_live_simulation_accepts_marked_broker(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")
    class Broker:
        is_simulated_adapter = True

    core_app._assert_live_simulation_adapter_safe(  # noqa: SLF001 - focused safety hook
        Broker(), component="broker"
    )


@pytest.mark.live_runtime_e2e
def test_live_runtime_starts_from_production_composition_with_simulated_adapters(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE_SIMULATION")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    monkeypatch.setenv("ALLOW_REAL_BROKER", "false")
    monkeypatch.setenv("ALLOW_NETWORK", "false")
    monkeypatch.setenv("BROKER_SIMULATION", "true")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STREAM__MODE", "websocket")
    monkeypatch.setenv("WEBSOCKET__DISABLED", "false")
    monkeypatch.setenv("TELEGRAM__ENABLED", "false")
    monkeypatch.setenv("TELEGRAM__WEBHOOK_ENABLED", "false")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("PAPER_MODE", "false")

    monkeypatch.setattr(core_app, "ZerodhaKiteClient", _NoNetworkBroker)
    monkeypatch.setattr(core_app, "RobustDataProvider", _NoNetworkRobustProvider)
    monkeypatch.setattr(core_app, "WebSocketManager", _NoNetworkWebSocketManager)
    monkeypatch.setattr(core_app, "start_background_tasks", lambda *a, **k: [])
    monkeypatch.setattr(core_app, "schedule_instrument_refresh", lambda *a, **k: None)
    monkeypatch.setattr(core_app, "start_watchdog", lambda *a, **k: None)

    async def _run():
        return initialize_components()

    ctx = asyncio.run(_run())
    assert isinstance(ctx, BotContext)
    assert isinstance(ctx.instrument_manager, InstrumentManager)
    assert isinstance(ctx.market_data_manager, MarketDataManager)
    assert isinstance(ctx.data_hub, DataHub)
    assert isinstance(ctx.indicator_engine, IndicatorEngine)
    assert isinstance(ctx.strategy_runner, StrategyRunner)
    assert isinstance(ctx.risk_manager, RiskManager)
    assert isinstance(ctx.order_manager, OrderManager)
    assert isinstance(ctx.position_manager, PositionManager)
    assert isinstance(ctx.bracket_manager, BracketManager)
    assert isinstance(ctx.broker_client, _NoNetworkRobustProvider)
    assert isinstance(ctx.websocket_manager, _NoNetworkWebSocketManager)
    assert ctx.shadow_mode_enabled is False
    assert ctx.settings.enable_live is True

    ctx.instrument_manager.load()
    basket = ctx.instrument_manager.get_active_nifty_contracts(25000.0)
    assert basket.atm_strike == 25000
    assert basket.selected_ce.endswith("25000CE")
    assert basket.selected_pe.endswith("25000PE")
    assert basket.selected_ce_token in basket.all_tokens
    assert basket.selected_pe_token in basket.all_tokens
    assert ctx.live_orders_armed is False


@pytest.mark.live_runtime_e2e
def test_live_runtime_bullish_spot_future_selects_ce_and_exits_target(
    monkeypatch, tmp_path
):
    """Runtime smoke: RSI mean-reversion warm history + bullish context selects ATM CE.

    Strategy used: existing ``RSIMeanReversionStrategy`` via the production
    ``build_elite_strategies`` seam.  The selected CE history is below the
    test-only oversold threshold while spot/future history and fresh ticks are bullish.
    """
    _runtime_env(monkeypatch, tmp_path)
    _patch_no_network_runtime(monkeypatch)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        ctx = loop.run_until_complete(_initialize_runtime())
        ctx.instrument_manager.load()
        basket = ctx.instrument_manager.get_active_nifty_contracts(25000.0)
        basket_dict = {
            **asdict(basket),
            "option_symbols": list(basket.option_symbols),
            "option_tokens": list(basket.option_tokens),
            "all_symbols": list(basket.all_symbols),
            "all_tokens": list(basket.all_tokens),
            "token_by_symbol": dict(basket.token_by_symbol),
        }
        selected_ce, selected_pe = (
            core_app._commit_active_dynamic_basket(  # noqa: SLF001 - production app commit helper
                ctx,
                basket=basket_dict,
                option_symbols=list(basket.option_symbols),
                symbols=list(basket.all_symbols),
                atm_strike=basket.atm_strike,
            )
        )
        assert selected_ce == basket.selected_ce
        assert selected_pe == basket.selected_pe

        end = pd.Timestamp.now(tz="Asia/Kolkata").floor("min") - pd.Timedelta(minutes=2)
        histories = {
            "NSE:NIFTY": _bars(end, 80, 24900.0, 1.25),
            basket.futures_symbol: _bars(end, 80, 24920.0, 1.20),
            basket.selected_ce: _bars(end, 60, 130.0, -0.45),
            basket.selected_pe: _bars(end, 60, 95.0, 0.05),
        }

        async def fetch_history(symbol: str, interval: str, days: int = 3):
            del interval, days
            rows = histories.get(symbol) or histories.get(str(symbol).upper())
            assert rows, f"unexpected history request for {symbol}"
            return list(rows)

        ctx.market_data_manager.fetch_history = fetch_history  # type: ignore[method-assign]
        awaitable = core_app.ensure_symbol_runtime_history
        loop.run_until_complete(
            awaitable(ctx, "NSE:NIFTY", role="spot", phase="startup", reason="live_sim")
        )
        loop.run_until_complete(
            awaitable(
                ctx,
                basket.futures_symbol,
                role="futures_context",
                phase="startup",
                reason="live_sim",
            )
        )
        loop.run_until_complete(
            awaitable(
                ctx,
                basket.selected_ce,
                role="selected_option",
                phase="startup",
                reason="live_sim",
            )
        )
        loop.run_until_complete(
            awaitable(
                ctx,
                basket.selected_pe,
                role="selected_option",
                phase="startup",
                reason="live_sim",
            )
        )

        for sym in (
            "NSE:NIFTY",
            basket.futures_symbol,
            basket.selected_ce,
            basket.selected_pe,
        ):
            assert ctx.market_data_manager.get_latest_closed_bar(sym) is not None
            assert len(ctx.market_data_manager.get_ohlc_bars(sym) or []) >= 20
            assert len(ctx.indicator_engine.get_history(sym) or []) >= 20

        assert ctx.broker_balance_valid is True
        loop.run_until_complete(core_app._reconcile_state(ctx))  # noqa: SLF001
        assert ctx.position_reconciliation_completed is True
        assert ctx.websocket_manager is not None
        ctx.websocket_manager.connect()
        ctx.strategy_runner.start()
        broker = ctx.broker_client.client
        broker.register_order_update_callback(
            ctx.order_manager.apply_broker_order_update
        )

        for sym, tok, price in (
            ("NSE:NIFTY", basket.spot_token, 25000.0),
            (basket.futures_symbol, basket.futures_token, 25020.0),
            (basket.selected_ce, basket.selected_ce_token, 112.0),
            (basket.selected_pe, basket.selected_pe_token, 96.0),
        ):
            assert core_app._register_and_subscribe_live_symbol(  # noqa: SLF001 - production app subscribe helper
                ctx,
                sym,
                tok,
                "live_sim",
                role="tradable_option" if sym.endswith(("CE", "PE")) else "context",
            )
            _publish_tick(ctx, sym, int(tok), price)

        assert ctx.live_orders_armed is False
        loop.run_until_complete(
            core_app._recompute_and_push_runtime_readiness(ctx, reason="live_sim")
        )  # noqa: SLF001
        assert ctx.live_orders_armed is True, ctx.live_block_reason

        submitted_before = len(broker.get_orders())
        _publish_tick(ctx, basket.selected_ce, basket.selected_ce_token, 116.0)
        orders = broker.get_orders()
        for _ in range(50):
            if len(orders) == submitted_before + 1:
                break
            ctx.market_data_manager._drain_tick_queue_sync()  # noqa: SLF001 - drain production queue
            time.sleep(0.01)
            orders = broker.get_orders()
        assert len(orders) == submitted_before + 1, {
            "orders": orders,
            "no_signal": ctx.strategy_manager.get_last_no_signal_decision(
                basket.selected_ce
            ),
            "mdm_tracked": ctx.market_data_manager.is_symbol_tracked(
                basket.selected_ce
            ),
            "tick_ready": ctx.market_data_manager.classify_live_tick_readiness(
                basket.selected_ce, basket.selected_ce_token, max_age_s=5.0
            ),
        }
        entry = orders[-1]
        assert entry["symbol"] == basket.selected_ce
        assert entry["transaction_type"] == "BUY"
        for _ in range(50):
            if entry["order_id"] in getattr(ctx.position_manager, "_orders", {}):
                break
            time.sleep(0.01)
        assert entry["order_id"] in getattr(ctx.position_manager, "_orders", {})

        broker.fill_order(
            entry["order_id"], average_price=float(entry["price"] or 116.0)
        )
        position = ctx.position_manager.get_position(basket.selected_ce)
        assert position is not None
        assert int(getattr(position, "quantity", 0)) > 0
        assert ctx.bracket_manager.is_symbol_managed(basket.selected_ce)

        # Let production bracket logic see the target-side market move through the
        # normal MDM/DataHub path, then fill the resulting simulated exit order.
        _publish_tick(ctx, basket.selected_ce, basket.selected_ce_token, 200.0)
        target_orders: list[dict[str, Any]] = []
        for _ in range(50):
            target_orders = [
                order
                for order in broker.get_orders()
                if order["order_id"] != entry["order_id"]
                and order.get("transaction_type") == "SELL"
            ]
            if target_orders:
                break
            ctx.market_data_manager._drain_tick_queue_sync()  # noqa: SLF001 - drain production queue
            time.sleep(0.01)
        assert target_orders, broker.get_orders()
        target_id = target_orders[-1]["order_id"]
        for _ in range(50):
            if target_id in getattr(ctx.order_manager, "_orders", {}):
                break
            time.sleep(0.01)
        assert target_id in getattr(ctx.order_manager, "_orders", {})
        broker.fill_order(target_id, average_price=200.0)
        loop.run_until_complete(core_app._reconcile_state(ctx))  # noqa: SLF001
        assert ctx.position_reconciliation_completed is True

        final_qty = sum(
            int(p.get("quantity", 0))
            for p in broker.get_positions()
            if p.get("symbol") == basket.selected_ce
        )
        assert final_qty == 0
        final_position = ctx.position_manager.get_position(basket.selected_ce)
        assert (
            final_position is None or int(getattr(final_position, "quantity", 0)) == 0
        )
        if ctx.bracket_manager.is_symbol_managed(basket.selected_ce):
            ctx.bracket_manager.reconcile_symbol_flat(basket.selected_ce)
        assert not ctx.bracket_manager.is_symbol_managed(basket.selected_ce)
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
        asyncio.set_event_loop(None)
