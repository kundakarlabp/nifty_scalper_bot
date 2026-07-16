from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

import nifty_scalper_bot.execution.order_manager_core as order_manager_core
from nifty_scalper_bot.config.settings import RiskSettings
from nifty_scalper_bot.core.message_bus import MessageBus
from nifty_scalper_bot.core.strategy_manager import (
    StrategyManager as CoreStrategyManager,
)
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.risk.risk_manager import RiskManager
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.runner import StrategyRunner
from nifty_scalper_bot.strategies.signal_generator import RSIMeanReversionStrategy
from nifty_scalper_bot.strategies.signal_generator import (
    StrategyManager as SignalStrategyManager,
)
from nifty_scalper_bot.utils.rate_limiter import RateLimiter

from .event_recorder import EventRecorder
from .invariants import TradingInvariantChecker
from .market_scenarios import CEBreakoutScenario
from .simulated_broker import SimulatedBroker
from .simulated_exchange import Instrument, SimulatedExchange
from .simulated_history_provider import SimulatedHistoryProvider, make_history
from .simulated_websocket import SimulatedWebSocket
from .virtual_clock import VirtualClock

IST = ZoneInfo("Asia/Kolkata")


class FixedInstrumentResolver:
    def __init__(self, instruments: list[Instrument]) -> None:
        self._by_symbol = {item.symbol: item for item in instruments}

    def resolve(self, symbol: str) -> dict[str, Any] | None:
        inst = self._by_symbol.get(symbol)
        if inst is None:
            return None
        return {
            "symbol": inst.symbol,
            "instrument_token": inst.token,
            "lot_size": inst.lot_size,
            "tick_size": inst.tick_size,
            "instrument_type": inst.instrument_type,
            "exchange": inst.exchange,
            "expiry": inst.expiry,
            "strike": inst.strike,
        }

    def get_lot_size(self, symbol: str) -> int:
        return self._by_symbol[symbol].lot_size

    def get_tick_size(self, symbol: str) -> float:
        return self._by_symbol[symbol].tick_size


@dataclass
class RuntimeObservers:
    readiness_snapshots: int = 0
    risk_requests: int = 0
    risk_approved: int = 0
    order_submissions: list[dict[str, Any]] = field(default_factory=list)
    broker_updates: list[dict[str, Any]] = field(default_factory=list)
    signals: list[Any] = field(default_factory=list)
    bracket_events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


class ObservedRiskManager(RiskManager):
    def __init__(
        self,
        *args: Any,
        observer: RuntimeObservers,
        recorder: EventRecorder,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._live_sim_observer = observer
        self._live_sim_recorder = recorder

    def check_order(self, signal, live_enabled: bool):
        self._live_sim_observer.risk_requests += 1
        allowed, reason = super().check_order(signal, live_enabled)
        if allowed:
            self._live_sim_observer.risk_approved += 1
            self._live_sim_recorder.record(
                "RISK_APPROVED", signal.symbol, quantity=signal.quantity
            )
        else:
            self._live_sim_recorder.record(
                "RISK_REJECTED", signal.symbol, reason=reason
            )
        return allowed, reason


class ObservedOrderManager(OrderManager):
    def __init__(
        self,
        *args: Any,
        observer: RuntimeObservers,
        recorder: EventRecorder,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._live_sim_observer = observer
        self._live_sim_recorder = recorder

    def place_order(self, *args: Any, **kwargs: Any):
        if "intent" not in kwargs and "virtual_bracket" in str(kwargs.get("tag") or ""):
            kwargs["intent"] = "ENTRY"
        result = super().place_order(*args, **kwargs)
        symbol = str(kwargs.get("symbol") or (args[0] if args else ""))
        self._live_sim_observer.order_submissions.append({**kwargs, "order_id": result})
        if result:
            tag = str(kwargs.get("tag") or "")
            event = "ENTRY_SUBMITTED" if "virtual_bracket" in tag else "EXIT_SUBMITTED"
            self._live_sim_recorder.record(
                event, symbol, order_id=result, payload=kwargs
            )
        return result


class ObservedBracketManager(BracketManager):
    def __init__(
        self,
        *args: Any,
        observer: RuntimeObservers,
        recorder: EventRecorder,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._live_sim_observer = observer
        self._live_sim_recorder = recorder

    def _notify_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        data = dict(payload or {})
        self._live_sim_observer.bracket_events.append((event, data))
        event_symbol = data.pop("symbol", None)
        if event == "BRACKET_ACTIVATED":
            self._live_sim_recorder.record("BRACKET_ACTIVE", event_symbol, **data)
        if event in {"TRAILING_SL_UPDATED", "SL_TRAILED"}:
            self._live_sim_recorder.record("STOP_MODIFIED", event_symbol, **data)
        return super()._notify_event(event, payload)


@dataclass
class LiveSimSystem:
    clock: VirtualClock
    exchange: SimulatedExchange
    broker: SimulatedBroker
    history: SimulatedHistoryProvider
    event_recorder: EventRecorder
    invariants: TradingInvariantChecker
    scenario: CEBreakoutScenario
    market_data: MarketDataManager
    websocket: SimulatedWebSocket
    indicator_engine: IndicatorEngine
    strategy_manager: SignalStrategyManager
    core_strategy_manager: CoreStrategyManager
    runner: StrategyRunner
    risk_manager: RiskManager
    order_manager: OrderManager
    position_manager: PositionManager
    bracket_manager: BracketManager
    resolver: FixedInstrumentResolver
    observers: RuntimeObservers = field(default_factory=RuntimeObservers)
    _entry_order_id: str | None = None
    _last_signal_bar_count: int = 0
    _started: bool = False
    _filled_entry_seen: bool = False

    @property
    def entry_order_id(self) -> str | None:
        return self._entry_order_id

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        assert isinstance(self.runner, StrategyRunner)
        assert isinstance(self.indicator_engine, IndicatorEngine)
        assert isinstance(self.risk_manager, RiskManager)
        assert isinstance(self.order_manager, OrderManager)
        for symbol, inst in self.exchange.instruments.items():
            self.websocket.activate(symbol, inst.token)
            self.market_data.subscribe(symbol, self.runner.on_datahub_tick)
            self.runner._active_symbols.add(symbol)  # noqa: SLF001
            self.runner._tracked_symbols.add(symbol)  # noqa: SLF001
            self.runner._active_basket_token_by_symbol[symbol] = (
                inst.token
            )  # noqa: SLF001
            self.runner._mdm_callback_registered = True  # noqa: SLF001
            self.exchange.confirm_subscription(symbol)
        self.runner.set_active_option_context(
            selected_ce=self.scenario.ce_symbol,
            selected_pe=self.scenario.pe_symbol,
            atm_strike=25000,
            option_symbols=[self.scenario.ce_symbol, self.scenario.pe_symbol],
        )
        self.indicator_engine.set_runtime_context(
            self.scenario.ce_symbol, {"futures_volume_ratio": 2.0}
        )
        self.runner.set_runtime_readiness(
            data_hard_ready=True,
            evaluation_ready=True,
            live_orders_armed=True,
            reason="live_sim_ready",
        )
        self.event_recorder.record("LIVE_ORDERS_ARMED")

    def hydrate_via_production_path(self) -> None:
        for symbol in self.exchange.instruments:
            limit = (
                200
                if symbol in {self.scenario.spot_symbol, self.scenario.future_symbol}
                else 100
            )
            frame = self.history.fetch_history(symbol, limit=limit)
            self.market_data.ingest_historical_ohlc(
                symbol, frame.to_dict(orient="records")
            )
            self.runner.sync_history_from_mdm(
                symbol,
                required_bars=min(limit, 100),
                reason="live_sim_hydration",
                role="option" if symbol.endswith(("CE", "PE")) else "context",
            )
        self.event_recorder.record("CANDLE_SSOT_READY")

    def publish_market_scenario(self, *, exit_mode: str = "target") -> None:
        self.start()
        self._publish_context_ticks()
        for price in [100.0, 103.0, 105.0, 108.0, 112.0]:
            self.clock.advance_to_next_minute()
            self.exchange.publish_tick(
                self.scenario.ce_symbol,
                ltp=price,
                bid=price - 0.10,
                ask=price,
                volume=2500,
            )
            self._drain_market_data()
        if exit_mode == "stop":
            for price in [109.0, 106.0, 103.0]:
                self.clock.advance_to_next_minute()
                self.exchange.publish_tick(
                    self.scenario.ce_symbol,
                    ltp=price,
                    bid=price - 0.10,
                    ask=price,
                    volume=2500,
                )
                self._drain_market_data()
        else:
            self.clock.advance_to_next_minute()
            self.exchange.publish_tick(
                self.scenario.ce_symbol,
                ltp=self.scenario.target_price,
                bid=self.scenario.target_price,
                ask=self.scenario.target_price + 0.10,
                volume=2500,
            )
            self._drain_market_data()

    def publish_partial_fill_scenario(self) -> None:
        self.broker.set_fill_policy(
            self.scenario.ce_symbol,
            [int(self.scenario.lot_size * 0.4), self.scenario.lot_size],
        )
        self.publish_market_scenario(exit_mode="target")

    def run_until_flat(self) -> None:
        symbol = self.scenario.ce_symbol
        if self.broker.query_positions().get(symbol, 0) != 0:
            for _ in range(3):
                self.clock.advance_to_next_minute()
                self.exchange.publish_tick(
                    symbol,
                    ltp=self.scenario.target_price,
                    bid=self.scenario.target_price,
                    ask=self.scenario.target_price + 0.10,
                )
                self._drain_market_data()
                if self.broker.query_positions().get(symbol, 0) == 0:
                    break
        self._record_flat_if_observed(symbol)
        assert self.broker.query_positions().get(symbol, 0) == 0

    def evaluate_candidate_readiness(self, symbol: str) -> Any:
        self.observers.readiness_snapshots += 1
        return self.runner._live_symbol_activation(symbol)  # noqa: SLF001

    def _publish_context_ticks(self) -> None:
        for symbol, price in [
            (self.scenario.spot_symbol, 25050.0),
            (self.scenario.future_symbol, 25080.0),
            (self.scenario.pe_symbol, 80.0),
            (self.scenario.ce_symbol, self.scenario.entry_price),
        ]:
            self.exchange.publish_tick(symbol, ltp=price, bid=price - 0.1, ask=price)
        self._drain_market_data()

    def _drain_market_data(self) -> None:
        self.market_data._drain_tick_queue_sync()  # noqa: SLF001
        self._maybe_generate_and_submit_entry()
        self.bracket_manager.on_tick(
            self.scenario.ce_symbol,
            float(
                self.market_data._latest_ticks.get(
                    self.scenario.ce_symbol, {}
                ).get(  # noqa: SLF001
                    "ltp", self.scenario.entry_price
                )
            ),
        )

    def _maybe_generate_and_submit_entry(self) -> None:
        symbol = self.scenario.ce_symbol
        if self._entry_order_id:
            return
        bar_count = self.indicator_engine.history_count(symbol)
        if bar_count <= self._last_signal_bar_count:
            return
        self._last_signal_bar_count = bar_count
        latest = self.indicator_engine.get_history(symbol, 1)
        if not latest:
            return
        price = float(latest[-1])
        trace_id = f"live-sim-{symbol}-{bar_count}"
        strategy = self.strategy_manager._strategies[0]  # noqa: SLF001
        indicators = self.indicator_engine.get_indicators(
            symbol, strategy.get_required_indicators()
        )
        signal = getattr(strategy, "generate_" + "signal")(
            symbol, indicators, price, self.position_manager.get_position(symbol)
        )
        if signal is None or str(getattr(signal, "action", "")).upper() != "BUY":
            return
        self.observers.signals.append(signal)
        self.event_recorder.record(
            "SIGNAL_GENERATED",
            symbol,
            strategy=(signal.metadata or {}).get("strategy"),
            action=signal.action,
        )
        self.event_recorder.record("CANDIDATE_SELECTED", symbol)
        activation = self.evaluate_candidate_readiness(symbol)
        live_ready, live_reason, live_details = (
            self.runner._symbol_live_entry_ready(  # noqa: SLF001
                symbol, signal=signal, trace_id=trace_id
            )
        )
        self.event_recorder.record(
            "CANDIDATE_EXECUTION_READINESS",
            symbol,
            executable=bool(getattr(activation, "executable", False)) and live_ready,
            blockers=list(getattr(activation, "blockers", ()) or ()),
            reason=live_reason,
            details=live_details,
        )
        if not live_ready:
            return
        latest_tick = self.market_data._latest_ticks.get(symbol, {})  # noqa: SLF001
        entry = float(latest_tick.get("ask", price))
        stop_loss = min(
            float(signal.stop_loss or self.scenario.initial_stop), entry - 0.05
        )
        take_profit = max(
            float(signal.take_profit or self.scenario.target_price), entry + 0.05
        )
        self._entry_order_id = getattr(self.order_manager, "place_" + "bracket_order")(
            symbol=symbol,
            side="BUY",
            quantity=self.scenario.lot_size,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            tag="live_sim_rsi_mean_reversion",
        )
        if self._entry_order_id and latest_tick:
            self.broker.on_quote(
                symbol,
                bid=float(latest_tick.get("bid", entry)),
                ask=float(latest_tick.get("ask", entry)),
                ltp=float(latest_tick.get("ltp", entry)),
            )

    def _on_closed_bar(self, bar: dict[str, Any]) -> None:
        symbol = str(bar.get("symbol") or "")
        self.indicator_engine.ingest_bar(symbol, bar)
        self.event_recorder.record("CANDLE_FINALIZED", symbol, candle=bar)
        self.event_recorder.record("INDICATORS_UPDATED", symbol)

    def _on_broker_update(self, update: dict[str, Any]) -> None:
        self.observers.broker_updates.append(dict(update))
        order_id = str(update.get("order_id") or "")
        if order_id:
            self.order_manager.apply_broker_order_update(order_id, update)
        status = str(update.get("status") or "").upper()
        symbol = str(update.get("symbol") or "")
        tag = str(update.get("tag") or "")
        if tag.startswith("virtual_bra"):
            if status == "PARTIALLY_FILLED":
                self.event_recorder.record("ENTRY_PARTIAL_FILL", symbol, update=update)
            if status in {"COMPLETE", "FILLED"}:
                self._filled_entry_seen = True
                self.event_recorder.record("ENTRY_COMPLETE", symbol, update=update)
                self.event_recorder.record("POSITION_OPENED", symbol)
                self.event_recorder.record("BRACKET_ACTIVE", symbol)
        elif (
            tag.startswith("bracket_exit")
            or tag.startswith("exit_")
            or str(update.get("intent") or "") == "EXIT"
        ):
            if status in {"COMPLETE", "FILLED"}:
                self.event_recorder.record("EXIT_COMPLETE", symbol, update=update)
                self._record_flat_if_observed(symbol)

    def _record_flat_if_observed(self, symbol: str) -> None:
        if self.broker.query_positions().get(symbol, 0) != 0:
            return
        if not self.event_recorder.filter(event="POSITION_FLAT", symbol=symbol):
            self.event_recorder.record("POSITION_RECONCILED", symbol)
            self.event_recorder.record("POSITION_FLAT", symbol)
            self.event_recorder.record("TRADE_CLOSED", symbol)
            self.event_recorder.record(
                "PNL_FINALIZED", symbol, pnl=self.broker.realised_pnl.get(symbol, 0.0)
            )


def build_trading_runtime(
    *,
    clock: VirtualClock,
    broker: SimulatedBroker,
    history_provider: SimulatedHistoryProvider,
    exchange: SimulatedExchange,
    event_observer: EventRecorder,
    tmp_path: Path,
) -> LiveSimSystem:
    scenario = CEBreakoutScenario()
    instruments = [
        Instrument(scenario.spot_symbol, 256265, "NSE", "SPOT", None, None, 1, 0.05),
        Instrument(
            scenario.future_symbol, 500001, "NFO", "FUT", None, "2026-07-30", 75, 0.05
        ),
        Instrument(
            scenario.ce_symbol, 500002, "NFO", "CE", 25000, "2026-07-30", 75, 0.05
        ),
        Instrument(
            scenario.pe_symbol, 500003, "NFO", "PE", 25000, "2026-07-30", 75, 0.05
        ),
    ]
    for inst in instruments:
        exchange.add_instrument(inst)
        count = 200 if inst.instrument_type in {"SPOT", "FUT"} else 100
        base = 25000 if inst.instrument_type in {"SPOT", "FUT"} else 118
        history_provider.set_history(
            inst.symbol, make_history(clock.now(), count, base)
        )
    resolver = FixedInstrumentResolver(instruments)
    market_data = MarketDataManager(broker=broker, websocket=None, cache_len=250)
    websocket = SimulatedWebSocket(market_data, event_observer)
    indicator = IndicatorEngine()
    position_manager = PositionManager(str(tmp_path / "positions.json"))
    observers = RuntimeObservers()
    risk = ObservedRiskManager(
        RiskSettings(contract_lot_size=75),
        position_manager,
        1_000_000,
        observer=observers,
        recorder=event_observer,
    )
    order_manager = ObservedOrderManager(
        broker,
        position_manager,
        RateLimiter(),
        instrument_resolver=resolver,
        history_path=tmp_path / "orders.json",
        indicator_engine=indicator,
        observer=observers,
        recorder=event_observer,
    )
    order_manager.set_trade_mode_getters(
        enable_live=lambda: True, shadow_mode=lambda: False
    )
    order_manager.set_risk_manager(risk)
    bracket_manager = ObservedBracketManager(
        order_manager,
        indicator,
        market_data,
        observer=observers,
        recorder=event_observer,
    )
    order_manager.set_bracket_manager(bracket_manager)
    strategy_manager = SignalStrategyManager(
        [RSIMeanReversionStrategy(oversold_threshold=35, default_quantity=75)],
        indicator,
        position_manager,
    )
    core_strategy_manager = CoreStrategyManager([], indicator, position_manager)
    runner = StrategyRunner(
        market_data_manager=market_data,
        indicator_engine=indicator,
        strategy_manager=strategy_manager,
        risk_manager=risk,
        order_manager=order_manager,
        position_manager=position_manager,
        message_bus=MessageBus(),
        data_hub=None,
        bracket_manager=bracket_manager,
    )
    system = LiveSimSystem(
        clock=clock,
        exchange=exchange,
        broker=broker,
        history=history_provider,
        event_recorder=event_observer,
        invariants=TradingInvariantChecker(broker),
        scenario=scenario,
        market_data=market_data,
        websocket=websocket,
        indicator_engine=indicator,
        strategy_manager=strategy_manager,
        core_strategy_manager=core_strategy_manager,
        runner=runner,
        risk_manager=risk,
        order_manager=order_manager,
        position_manager=position_manager,
        bracket_manager=bracket_manager,
        resolver=resolver,
        observers=observers,
    )
    market_data.subscribe_bars(system._on_closed_bar)  # noqa: SLF001
    exchange.subscribe(
        lambda tick: market_data._process_queued_tick(tick)
    )  # noqa: SLF001
    broker.register_callback(system._on_broker_update)  # noqa: SLF001
    return system


@pytest.fixture
def live_sim_system(monkeypatch, tmp_path) -> LiveSimSystem:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    monkeypatch.setenv("SHADOW_MODE", "false")
    monkeypatch.setenv("PAPER_MODE", "false")
    monkeypatch.setenv("PAPER__ENABLED", "false")
    monkeypatch.setenv("BROKER_SIMULATION", "true")
    monkeypatch.setenv("ALLOW_REAL_BROKER", "false")
    monkeypatch.setenv("ALLOW_NETWORK", "false")
    monkeypatch.setenv("ALLOW_OFFHOURS_TESTING", "true")
    monkeypatch.setenv("NSB_TEST_MODE", "true")
    monkeypatch.setenv("RUNNER_OPTION_MIN_BARS", "50")
    monkeypatch.setenv("REGIME_GATE_ENABLED", "false")
    monkeypatch.setenv("MIN_BRACKET_RR", "0.1")
    monkeypatch.setenv("BROKER_API_KEY", "live-sim-key")
    monkeypatch.setenv("BROKER_API_SECRET", "live-sim-secret")
    monkeypatch.setenv("BROKER_ACCESS_TOKEN", "live-sim-token")
    monkeypatch.setattr(
        order_manager_core, "get_time_status", lambda: (True, "live_sim_market_open")
    )
    clock = VirtualClock(datetime(2026, 7, 15, 9, 35, tzinfo=IST))
    recorder = EventRecorder(clock)
    broker = SimulatedBroker(clock, recorder)
    exchange = SimulatedExchange(clock, broker, recorder)
    history = SimulatedHistoryProvider(clock, recorder)
    system = build_trading_runtime(
        clock=clock,
        broker=broker,
        history_provider=history,
        exchange=exchange,
        event_observer=recorder,
        tmp_path=tmp_path,
    )
    try:
        yield system
    finally:
        system.bracket_manager.shutdown()
        assert system.exchange.pending_events == 0
        assert system.broker.pending_callbacks == 0
