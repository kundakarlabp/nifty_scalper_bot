"""Offline production-composition replay of archived live observations.

Uses initialize_components and real decision/risk/order/bracket owners. Only
external broker/stream adapters are substituted. This is modeled execution,
not evidence of exchange fills or a reproduction of uncaptured decision state.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import socket
from contextlib import ExitStack
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from nifty_scalper_bot.storage.replay_archive import archive_fingerprint, iter_session


def validate_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Fail closed on unavailable inputs instead of substituting live instruments."""
    if not snapshot.get("instruments"):
        raise ValueError("replay_instrument_catalog_missing")
    if snapshot.get("initial_positions"):
        raise ValueError("replay_initial_positions_unsupported")
    if not snapshot.get("effective_settings") or not snapshot.get("runner_config"):
        raise ValueError("replay_configuration_missing")
    balance = snapshot.get("initial_balance")
    if (
        not isinstance(balance, (int, float))
        or not math.isfinite(balance)
        or balance <= 0
    ):
        raise ValueError("replay_initial_balance_missing")
    if not snapshot.get("basket"):
        raise ValueError("replay_contract_basket_missing")


def _restore_settings(current: Any, captured: Any) -> Any:
    if (
        is_dataclass(current)
        and not isinstance(current, type)
        and isinstance(captured, dict)
    ):
        updates = {
            field.name: _restore_settings(
                getattr(current, field.name), captured[field.name]
            )
            for field in fields(current)
            if field.name in captured
        }
        return replace(current, **updates)
    if hasattr(current, "model_validate") and isinstance(captured, dict):
        return type(current).model_validate(captured)
    if isinstance(current, tuple) and isinstance(captured, list):
        return tuple(captured)
    if isinstance(current, (set, frozenset)) and isinstance(captured, list):
        return type(current)(captured)
    return captured


class ArchivedBroker:
    """No-network identity/history adapter delegating fills to PaperFillEngine."""

    is_simulated_adapter = True

    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        self.snapshot = snapshot
        self.client = self
        self.api_key = "offline_replay"
        self.auth_invalid = False
        self.paper: Any = None
        self._callback: Any = None
        self._last_updates: dict[str, tuple[Any, ...]] = {}

    def preload_instruments(self) -> None:
        return None

    def instruments(self, exchange: str | None = None) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.snapshot["instruments"]
            if exchange is None or row.get("exchange") == exchange
        ]

    def get_available_balance(self, *args: Any, **kwargs: Any) -> float:
        return max(0.0, self.account()["cash"])

    def account(self) -> dict[str, float]:
        orders = self.get_orders()
        fees = sum(float(order.get("fees") or 0) for order in orders)
        cash_flow = (
            sum(
                (1 if order.get("transaction_type") == "SELL" else -1)
                * float(order.get("average_price") or 0)
                * int(order.get("filled_quantity") or 0)
                for order in orders
            )
            - fees
        )
        open_value = 0.0
        for position in self.get_positions():
            quote = self.paper.hub.get_quote(position["symbol"], allow_pull=False) or {}
            open_value += position["quantity"] * float(
                quote.get("bid") or quote.get("ltp") or 0
            )
        return {
            "cash": float(self.snapshot["initial_balance"]) + cash_flow,
            "net_equity_pnl": cash_flow + open_value,
            "fees": fees,
            "open_position_value": open_value,
        }

    def get_instrument_token(self, symbol: str) -> int:
        if symbol in {"NSE:NIFTY", "NSE:NIFTY 50"}:
            return 256265
        for row in self.instruments():
            if str(row.get("tradingsymbol")) == symbol.split(":")[-1]:
                return int(row["instrument_token"])
        raise LookupError("replay_instrument_missing")

    def historical_data(self, token: int, *args: Any, **kwargs: Any) -> list[Any]:
        # Bootstrap history is injected explicitly with availability checks.
        return []

    def set_auth_failure_callback(self, callback: Any) -> None:
        self._auth_callback = callback

    def register_order_update_callback(self, callback: Any) -> None:
        self._callback = callback

    def is_connected(self) -> bool:
        return True

    def get_ltp(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        if self.paper is None:
            return {}
        return {
            symbol: dict(self.paper.hub.get_quote(symbol, allow_pull=False) or {})
            for symbol in symbols
        }

    def get_orders(self) -> list[dict[str, Any]]:
        return self.paper.get_orders() if self.paper else []

    def get_positions(self) -> list[dict[str, Any]]:
        quantities: dict[str, int] = {}
        for order in self.get_orders():
            symbol = str(order.get("symbol") or "")
            quantity = int(order.get("filled_quantity") or 0)
            sign = 1 if order.get("transaction_type") == "BUY" else -1
            quantities[symbol] = quantities.get(symbol, 0) + sign * quantity
        return [
            {
                "symbol": symbol,
                "tradingsymbol": symbol.split(":")[-1],
                "quantity": qty,
                "product": "MIS",
            }
            for symbol, qty in quantities.items()
            if qty
        ]

    def place_order(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if self.paper is None:
            raise RuntimeError("replay_paper_engine_unbound")
        return self.paper.place_order(dict(args[0]) if args else kwargs)

    def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        return self.paper.cancel_order(*args, **kwargs)

    def modify_order(self, *args: Any, **kwargs: Any) -> Any:
        return self.paper.modify_order(*args, **kwargs)

    def publish_updates(self) -> None:
        for order in self.get_orders():
            key = str(order["order_id"])
            state = (
                order.get("status"),
                order.get("filled_quantity"),
                order.get("average_price"),
            )
            if state == self._last_updates.get(key):
                continue
            self._last_updates[key] = state
            if self._callback:
                update = dict(order)
                update["status"] = {
                    "filled": "COMPLETE",
                    "open": "OPEN",
                    "cancelled": "CANCELLED",
                    "rejected": "REJECTED",
                }.get(str(order.get("status")), str(order.get("status")).upper())
                self._callback(key, update)


class ArchivedStream:
    """No-network stream adapter; events enter the production MDM directly."""

    is_simulated_adapter = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.connected = True
        self._subscribed_tokens: set[int] = set()
        self._callbacks: dict[str, Any] = {}

    def set_callbacks(self, **kwargs: Any) -> None:
        self._callbacks.update(kwargs)

    def set_fallback_callbacks(self, **kwargs: Any) -> None:
        return None

    def subscribe_tokens(self, tokens: Any, mode: str = "full") -> None:
        self._subscribed_tokens.update(int(token) for token in tokens)

    async def subscribe(self, tokens: Any) -> None:
        self.subscribe_tokens(tokens)

    def is_connected(self) -> bool:
        return self.connected

    def connect(self) -> None:
        self.connected = True

    def stop(self) -> None:
        self.connected = False

    def backlog_size(self) -> int:
        return 0


class ArchivedProvider:
    is_simulated_adapter = True

    def __init__(
        self, broker_client: ArchivedBroker, *args: Any, **kwargs: Any
    ) -> None:
        self.client = broker_client
        self.auth_invalid = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


def run_runtime_session(
    path: Path, output: Path, *, slippage_bps: float = 10.0
) -> dict[str, Any]:
    """Run one captured session in a dedicated process with no external sockets."""
    if os.getenv("REPLAY_ISOLATED_PROCESS") != "true":
        raise ValueError("replay_requires_isolated_process")
    if not math.isfinite(slippage_bps) or not 0 <= slippage_bps <= 100:
        raise ValueError("replay_slippage_invalid")
    # First pass validates integrity without holding a session in RAM.
    event_count = sum(1 for _ in iter_session(path))
    events = iter_session(path)
    first_event = next(events)
    snapshot = first_event["payload"]
    validate_snapshot(snapshot)
    from freezegun import freeze_time

    output.mkdir(parents=True, exist_ok=True)
    broker = ArchivedBroker(snapshot)
    overrides = {
        **snapshot.get("configuration", {}),
        "DATA_DIR": str(output.resolve()),
        "HUB_STORE_PATH": str((output / "hub.db").resolve()),
        "EXECUTION_MODE": "LIVE_SIMULATION",
        "ENABLE_LIVE": "false",
        "ENABLE_LIVE_TRADING": "false",
        "ALLOW_REAL_BROKER": "false",
        "ALLOW_NETWORK": "false",
        "BROKER_SIMULATION": "true",
        "SHADOW_MODE": "false",
        "PAPER_MODE": "false",
        "PAPER__ENABLED": "false",
        "PAPER__SLIPPAGE_BPS": str(slippage_bps),
        "REPLAY_CAPTURE_ENABLED": "false",
        "TELEGRAM__ENABLED": "false",
        "TELEGRAM__WEBHOOK_ENABLED": "false",
        "SUPABASE_TRADE_REPLICATION_ENABLED": "false",
        "SUPABASE_LOG_ARCHIVE_ENABLED": "false",
        "BROKER_API_KEY": "offline_replay",
        "BROKER_API_SECRET": "offline_replay",
        "BROKER_ACCESS_TOKEN": "offline_replay",
        "STREAM__MODE": "websocket",
    }
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ctx = None
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, overrides))
        stack.enter_context(
            patch.object(
                socket.socket,
                "connect",
                side_effect=RuntimeError("replay_network_forbidden"),
            )
        )
        stack.enter_context(
            patch.object(
                socket,
                "create_connection",
                side_effect=RuntimeError("replay_network_forbidden"),
            )
        )
        stack.enter_context(
            patch.object(
                socket.socket,
                "connect_ex",
                side_effect=RuntimeError("replay_network_forbidden"),
            )
        )
        stack.enter_context(
            patch.object(
                socket,
                "getaddrinfo",
                side_effect=RuntimeError("replay_network_forbidden"),
            )
        )
        frozen = stack.enter_context(
            freeze_time(first_event["available_at"], real_asyncio=True)
        )
        # Production import order installs runtime hardening and prevents config cycles.
        import nifty_scalper_bot.core.app as app
        from nifty_scalper_bot.config.settings import get_settings
        from nifty_scalper_bot.strategies.runner import StrategyRunnerConfig
        from nifty_scalper_bot.utils.serialization import to_json_safe

        stack.enter_context(
            patch.object(app, "ZerodhaKiteClient", lambda *a, **k: broker)
        )
        stack.enter_context(patch.object(app, "RobustDataProvider", ArchivedProvider))
        stack.enter_context(patch.object(app, "WebSocketManager", ArchivedStream))
        for name in (
            "start_background_tasks",
            "schedule_instrument_refresh",
            "start_watchdog",
        ):
            stack.enter_context(patch.object(app, name, lambda *a, **k: []))

        async def captured_history_only(ctx: Any, **kwargs: Any) -> dict[str, Any]:
            # Historical-data boundary: missing history stays missing. No REST,
            # no synthetic candles, and no weakening of the readiness decision.
            basket = ctx.active_contract_basket or {}
            for symbol in basket.get("all_symbols", []):
                role = "option" if symbol.endswith(("CE", "PE")) else "context"
                required = (
                    kwargs["option_required_bars"]
                    if role == "option"
                    else kwargs["context_required_bars"]
                )
                ctx.strategy_runner.sync_history_from_mdm(
                    symbol,
                    required_bars=required,
                    reason="captured_history",
                    role=role,
                    request_if_short=False,
                )
            return {}

        stack.enter_context(
            patch.object(app, "_ensure_active_basket_history", captured_history_only)
        )
        settings = get_settings()
        for name, value in snapshot["effective_settings"].items():
            if name not in {
                "risk",
                "orders",
                "elite",
                "execution",
                "regime",
                "selector",
                "liquidity",
                "option_universe",
            }:
                raise ValueError("replay_settings_section_invalid")
            setattr(settings, name, _restore_settings(getattr(settings, name), value))
        if snapshot.get("app_risk"):
            settings.app.risk = type(settings.app.risk).model_validate(
                snapshot["app_risk"]
            )
        if snapshot.get("quote_stale_threshold_ms"):
            settings.app.quote_stale_threshold_ms = int(
                snapshot["quote_stale_threshold_ms"]
            )
        settings.enable_live = (
            True  # strategy profile only; adapters and network remain isolated
        )
        runner_config = StrategyRunnerConfig(**snapshot["runner_config"])
        stack.enter_context(
            patch.object(app, "_get_strategy_config", lambda config: runner_config)
        )
        try:

            async def initialize() -> Any:
                return app.initialize_components(settings)

            ctx = loop.run_until_complete(initialize())
            broker.paper = ctx.paper_engine
            ctx.order_manager.set_broker_client(broker)

            async def attach_loop() -> None:
                ctx.strategy_runner.attach_runtime_loop(loop)

            loop.run_until_complete(attach_loop())
            equity_peak = max_drawdown = 0.0
            for event in chain((first_event,), events):
                frozen.move_to(event["available_at"])
                payload = event["payload"]
                if event["kind"] == "snapshot":
                    if payload.get("initial_positions") and event is first_event:
                        raise ValueError("replay_initial_positions_unsupported")
                    basket = payload["basket"]
                    ctx.active_contract_basket = basket
                    ctx.active_trading_universe = basket
                    ctx.market_data_manager.set_active_contract_basket(basket)
                    ctx.strategy_runner.set_active_trading_universe(basket)
                    if ctx.data_hub:
                        ctx.data_hub.set_active_contract_basket(basket)
                    now = datetime.fromisoformat(event["available_at"])
                    for symbol, bars in payload.get("history", {}).items():

                        completed = [
                            bar
                            for bar in bars
                            if datetime.fromisoformat(str(bar["timestamp"]))
                            + timedelta(minutes=1)
                            <= now
                        ]
                        if completed:
                            ctx.market_data_manager.ingest_historical_ohlc(
                                symbol, completed
                            )
                            ctx.strategy_runner.sync_history_from_mdm(
                                symbol,
                                required_bars=len(completed),
                                reason="recorded_warmup",
                                request_if_short=False,
                            )
                        ctx.market_data_manager.subscribe(
                            symbol, ctx.strategy_runner.on_datahub_tick
                        )

                    async def start_runner() -> None:
                        ctx.strategy_runner.start()

                    loop.run_until_complete(start_runner())
                    loop.run_until_complete(
                        app._recompute_and_push_runtime_readiness(
                            ctx, reason="recorded_snapshot"
                        )
                    )
                else:
                    tick = dict(payload)
                    symbol = str(tick["symbol"])
                    # Archive boundary is the normalized, delivered observation.
                    tick["_volume_delta_normalized"] = True
                    ctx.market_data_manager._process_queued_tick(tick)
                    broker.paper.process_quote(symbol)
                    broker.publish_updates()

                # Let production queued callbacks run before advancing the clock.
                async def drain_evaluations() -> None:
                    for _ in range(500):
                        await asyncio.sleep(0.001)
                        runner = ctx.strategy_runner
                        with runner._eval_gate_lock:
                            pending = bool(
                                runner._pending_entry_eval_symbols
                                or runner._entry_eval_active
                                or runner._entry_eval_drain_scheduled
                            )
                        if not pending:
                            return
                    raise RuntimeError("replay_evaluation_drain_timeout")

                loop.run_until_complete(drain_evaluations())
                broker.publish_updates()
                equity = broker.account()["net_equity_pnl"]
                equity_peak = max(equity_peak, equity)
                max_drawdown = max(max_drawdown, equity_peak - equity)
            orders = broker.get_orders()
            report = {
                "scope": "production_composition_recorded_feed_replay",
                "live_equivalent": False,
                "evidence_label": "RESEARCH_CANDIDATE",
                "events_processed": event_count,
                "candle_counts": {
                    symbol: len(ctx.market_data_manager.get_ohlc_bars(symbol))
                    for symbol in (snapshot["basket"].get("all_symbols") or [])
                },
                "runtime_owners": {
                    name: type(getattr(ctx, name)).__name__
                    for name in (
                        "strategy_runner",
                        "risk_manager",
                        "order_manager",
                        "bracket_manager",
                        "position_manager",
                    )
                },
                "archive_sha256": archive_fingerprint(path),
                "captured_revision": snapshot.get("code_revision"),
                "slippage_bps_per_side": slippage_bps,
                "orders": to_json_safe(orders),
                "modeled_account": broker.account(),
                "max_equity_drawdown": max_drawdown,
                "strategy_mode_profile": (
                    ctx.strategy_manager.get_strategy_mode_profile()
                ),
                "positions": to_json_safe(broker.get_positions()),
                "runner_status": to_json_safe(ctx.strategy_runner.get_status()),
                "limitations": [
                    "Simulated fills, not exchange execution",
                    "Initial decision/cooldown/risk state is not restored",
                    "Captured baskets are replayed, not independently reselected",
                    "Background recovery and scheduling differ from production",
                ],
            }
            (output / "report.json").write_text(json.dumps(report, allow_nan=False))
            return report
        finally:
            if ctx is not None:
                loop.run_until_complete(
                    app.shutdown_sequence(ctx, reason="historical_replay")
                )
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()
            asyncio.set_event_loop(None)
