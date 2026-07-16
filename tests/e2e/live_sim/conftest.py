from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from nifty_scalper_bot.data.candle_engine import CandleEngine

from .event_recorder import EventRecorder
from .invariants import InternalState, TradingInvariantChecker
from .market_scenarios import CEBreakoutScenario
from .simulated_broker import SimulatedBroker
from .simulated_exchange import Instrument, SimulatedExchange
from .simulated_history_provider import SimulatedHistoryProvider, make_history
from .virtual_clock import VirtualClock

IST = ZoneInfo("Asia/Kolkata")


class SimIndicatorEngine:
    def __init__(self) -> None:
        self.history: dict[str, pd.DataFrame] = {}

    def sync(self, symbol: str, frame: pd.DataFrame) -> None:
        self.history[symbol] = frame.copy()


class SimRunner:
    def __init__(self) -> None:
        self.history: dict[str, pd.DataFrame] = {}

    def sync(self, symbol: str, frame: pd.DataFrame) -> None:
        self.history[symbol] = frame.copy()


@dataclass
class LiveSimSystem:
    clock: VirtualClock
    exchange: SimulatedExchange
    broker: SimulatedBroker
    history: SimulatedHistoryProvider
    event_recorder: EventRecorder
    invariants: TradingInvariantChecker
    scenario: CEBreakoutScenario
    candle_engines: dict[str, CandleEngine]
    indicator_engine: SimIndicatorEngine
    runner: SimRunner
    internal: InternalState
    bracket_orders: dict[str, str] = field(default_factory=dict)
    target_orders: dict[str, str] = field(default_factory=dict)
    entry_order_id: str | None = None

    def hydrate(self) -> None:
        for symbol, engine in self.candle_engines.items():
            limit = (
                200
                if symbol in {self.scenario.spot_symbol, self.scenario.future_symbol}
                else 100
            )
            frame = self.history.fetch_history(symbol, limit=limit)
            engine.replace_history(frame)
            self.indicator_engine.sync(symbol, engine.get_df())
            self.runner.sync(symbol, engine.get_df())
        self.event_recorder.record("CANDLE_SSOT_READY")

    def subscribe_all(self) -> None:
        for symbol in self.candle_engines:
            self.event_recorder.record("SUBSCRIPTION_REQUESTED", symbol)
            self.exchange.confirm_subscription(symbol)
            self.exchange.publish_tick(
                symbol,
                ltp=100 if "CE" in symbol else 80 if "PE" in symbol else 25000,
                bid=99.8,
                ask=100.0,
            )
            self.event_recorder.record("FIRST_CURRENT_GENERATION_TICK", symbol)
        self.event_recorder.record("LIVE_ORDERS_ARMED")

    def on_tick(self, tick: dict) -> None:
        symbol = tick["symbol"]
        finalized = self.candle_engines[symbol].on_tick(tick)
        if finalized is not None:
            frame = self.candle_engines[symbol].get_df()
            self.indicator_engine.sync(symbol, frame)
            self.runner.sync(symbol, frame)
            self.event_recorder.record("CANDLE_FINALIZED", symbol, candle=finalized)
            self.event_recorder.record("INDICATORS_UPDATED", symbol)
        if symbol == self.scenario.spot_symbol:
            self.event_recorder.record("SPOT_CONTEXT_UPDATED", symbol)
        elif symbol == self.scenario.future_symbol:
            self.event_recorder.record("FUTURES_CONTEXT_UPDATED", symbol)
        else:
            self.event_recorder.record("OPTION_CONTEXT_UPDATED", symbol)

    def evaluate_and_enter_ce(self, *, partial: bool = False) -> None:
        s = self.scenario
        self.event_recorder.record("STRATEGY_EVALUATED", s.spot_symbol)
        self.event_recorder.record("SIGNAL_GENERATED", s.ce_symbol, side="CE")
        self.event_recorder.record(
            "CANDIDATE_EXECUTION_READINESS", s.ce_symbol, ready=True
        )
        self.event_recorder.record("RISK_APPROVED", s.ce_symbol, quantity=s.lot_size)
        self.entry_order_id = self.broker.place_order(
            symbol=s.ce_symbol,
            side="BUY",
            quantity=s.lot_size,
            order_type="LIMIT",
            price=s.entry_price,
            tag="entry",
        )
        self.internal.active_entries[s.ce_symbol] = 1
        self.event_recorder.record(
            "ENTRY_SUBMITTED", s.ce_symbol, order_id=self.entry_order_id
        )
        self.event_recorder.record(
            "ENTRY_ACKNOWLEDGED", s.ce_symbol, order_id=self.entry_order_id
        )
        if partial:
            q1 = int(s.lot_size * 0.4)
            self.broker.fill(self.entry_order_id, q1, s.entry_price)
            self.internal.positions[s.ce_symbol] = q1
            self.event_recorder.record("ENTRY_PARTIAL_FILL", s.ce_symbol, quantity=q1)
            self.internal.active_stop[s.ce_symbol] = q1
            self.internal.active_target[s.ce_symbol] = q1
            self.broker.fill(self.entry_order_id, s.lot_size - q1, s.entry_price)
        else:
            self.broker.fill(self.entry_order_id, s.lot_size, s.entry_price)
        self.internal.positions[s.ce_symbol] = s.lot_size
        self.event_recorder.record("ENTRY_COMPLETE", s.ce_symbol)
        self.event_recorder.record("POSITION_OPENED", s.ce_symbol, quantity=s.lot_size)
        self.submit_bracket()
        self.invariants.check_all()

    def submit_bracket(self) -> None:
        s = self.scenario
        stop_id = self.broker.place_order(
            symbol=s.ce_symbol,
            side="SELL",
            quantity=s.lot_size,
            order_type="SL",
            price=s.initial_stop,
            trigger_price=s.initial_stop,
            tag="stop",
        )
        target_id = self.broker.place_order(
            symbol=s.ce_symbol,
            side="SELL",
            quantity=s.lot_size,
            order_type="LIMIT",
            price=s.target_price,
            tag="target",
        )
        self.bracket_orders[s.ce_symbol] = stop_id
        self.target_orders[s.ce_symbol] = target_id
        self.internal.active_stop[s.ce_symbol] = s.lot_size
        self.internal.active_target[s.ce_symbol] = s.lot_size
        self.internal.stop_prices.setdefault(s.ce_symbol, []).append(s.initial_stop)
        self.event_recorder.record("STOP_SUBMITTED", s.ce_symbol, order_id=stop_id)
        self.event_recorder.record("TARGET_SUBMITTED", s.ce_symbol, order_id=target_id)
        self.event_recorder.record("BRACKET_ACTIVE", s.ce_symbol)

    def trail_stop(self, price: float) -> None:
        symbol = self.scenario.ce_symbol
        self.broker.modify_order(
            self.bracket_orders[symbol], price=price, trigger_price=price
        )
        self.internal.stop_prices.setdefault(symbol, []).append(price)
        self.event_recorder.record("STOP_MODIFIED", symbol, price=price)
        self.invariants.check_all()

    def close_via_target(self) -> None:
        symbol = self.scenario.ce_symbol
        self.broker.fill(
            self.target_orders[symbol],
            self.scenario.lot_size,
            self.scenario.target_price,
        )
        self.event_recorder.record(
            "EXIT_COMPLETE", symbol, order_id=self.target_orders[symbol]
        )
        self.broker.cancel_order(self.bracket_orders[symbol])
        self.event_recorder.record("SIBLING_CANCELLED", symbol)
        self._flat(symbol)

    def close_via_stop(self, price: float) -> None:
        symbol = self.scenario.ce_symbol
        self.broker.fill(self.bracket_orders[symbol], self.scenario.lot_size, price)
        self.event_recorder.record(
            "EXIT_COMPLETE", symbol, order_id=self.bracket_orders[symbol]
        )
        self.broker.cancel_order(self.target_orders[symbol])
        self.event_recorder.record("SIBLING_CANCELLED", symbol)
        self._flat(symbol)

    def _flat(self, symbol: str) -> None:
        self.internal.positions[symbol] = 0
        self.internal.active_stop[symbol] = 0
        self.internal.active_target[symbol] = 0
        self.internal.risk_reserved[symbol] = 0
        self.event_recorder.record("POSITION_RECONCILED", symbol)
        self.event_recorder.record("POSITION_FLAT", symbol)
        self.event_recorder.record("TRADE_CLOSED", symbol)
        self.event_recorder.record(
            "PNL_FINALIZED", symbol, pnl=self.broker.realised_pnl.get(symbol, 0.0)
        )
        self.invariants.check_all()


@pytest.fixture
def live_sim_system(monkeypatch) -> LiveSimSystem:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("BROKER_SIMULATION", "true")
    monkeypatch.setenv("ALLOW_REAL_BROKER", "false")
    monkeypatch.setenv("ALLOW_NETWORK", "false")
    clock = VirtualClock(datetime(2026, 7, 15, 8, 55, tzinfo=IST))
    recorder = EventRecorder(clock)
    broker = SimulatedBroker(clock, recorder)
    exchange = SimulatedExchange(clock, broker, recorder)
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
    history = SimulatedHistoryProvider(clock, recorder)
    for inst in instruments:
        count = 200 if inst.instrument_type in {"SPOT", "FUT"} else 100
        history.set_history(
            inst.symbol,
            make_history(
                clock.now(),
                count,
                25000 if inst.instrument_type in {"SPOT", "FUT"} else 80,
            ),
        )
    engines = {
        inst.symbol: CandleEngine(symbol=inst.symbol, max_bars=500)
        for inst in instruments
    }
    indicator = SimIndicatorEngine()
    runner = SimRunner()
    internal = InternalState()
    system = LiveSimSystem(
        clock,
        exchange,
        broker,
        history,
        recorder,
        TradingInvariantChecker(broker, internal),
        scenario,
        engines,
        indicator,
        runner,
        internal,
    )
    exchange.subscribe(system.on_tick)
    return system
