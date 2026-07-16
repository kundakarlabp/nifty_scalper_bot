from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

import pytest

from nifty_scalper_bot.core import app as core_app
from nifty_scalper_bot.core.app import BotContext, initialize_components
from nifty_scalper_bot.core.instrument_manager import InstrumentManager
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.risk.risk_manager import RiskManager
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner

pytestmark = [pytest.mark.e2e_live_sim, pytest.mark.live_runtime_e2e]


class _NoNetworkBroker:
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
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._callbacks: dict[str, Any] = {}
        self._subscribed_tokens: set[int] = set()
        self._market_data_manager = None
        self.connected = False

    def set_callbacks(self, **callbacks: Any) -> None:
        self._callbacks.update(callbacks)

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
