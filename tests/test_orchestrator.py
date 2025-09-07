from __future__ import annotations

import time
from decimal import Decimal
from typing import List

from src.brokers.mock import MockBroker
from src.data.broker_source import BrokerDataSource
from src.execution.broker_executor import BrokerOrderExecutor
from src.strategies.runner import Orchestrator
from src.broker.interface import Tick


def test_orchestrator_drops_oldest() -> None:
    broker = MockBroker()
    ds = BrokerDataSource(broker)
    execu = BrokerOrderExecutor(broker)
    orch = Orchestrator(ds, execu, lambda _t: None, max_ticks=2)
    for i in range(3):
        orch._enqueue_tick(Tick(instrument_id=i, ts=time.time(), ltp=Decimal("1")))
    ids = [t.instrument_id for t in orch._tick_queue]
    assert ids == [1, 2]


def test_orchestrator_min_eval_interval() -> None:
    broker = MockBroker()
    ds = BrokerDataSource(broker)
    execu = BrokerOrderExecutor(broker)
    times: List[float] = []

    def strat(_t: Tick) -> None:
        times.append(time.time())

    orch = Orchestrator(ds, execu, strat, min_eval_interval_s=0.2)
    orch.start()
    ds.subscribe([1])
    for _ in range(3):
        broker.push_tick(1, Decimal("1"))
        time.sleep(0.05)
    time.sleep(0.5)
    diffs = [t2 - t1 for t1, t2 in zip(times, times[1:])]
    assert all(d >= 0.2 for d in diffs)
    orch.stop()


def test_orchestrator_stale_watchdog() -> None:
    broker = MockBroker()
    ds = BrokerDataSource(broker)
    execu = BrokerOrderExecutor(broker)
    triggered: List[str] = []

    orch = Orchestrator(
        ds,
        execu,
        lambda _t: None,
        stale_tick_timeout_s=0.2,
        on_stale=lambda: triggered.append("x"),
    )
    orch.start()
    ds.subscribe([1])
    broker.push_tick(1, Decimal("1"))
    time.sleep(0.3)
    assert triggered == ["x"]
    assert orch._paused
    broker.push_tick(1, Decimal("1"))
    time.sleep(0.1)
    assert not orch._paused
    orch.stop()
