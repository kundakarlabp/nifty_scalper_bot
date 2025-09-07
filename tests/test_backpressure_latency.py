from __future__ import annotations

import time
from decimal import Decimal

from src.brokers.mock import MockBroker
from src.data.broker_source import BrokerDataSource
from src.execution.broker_executor import BrokerOrderExecutor
from src.strategies.runner import Orchestrator


def test_backpressure_handles_flood() -> None:
    broker = MockBroker()
    ds = BrokerDataSource(broker)
    execu = BrokerOrderExecutor(broker)
    processed = 0

    def strat(_tick):
        nonlocal processed
        processed += 1
        return None

    orch = Orchestrator(ds, execu, strat, max_ticks=60_000)
    orch.start()
    ds.subscribe([1])
    start = time.time()
    for _ in range(50_000):
        broker.push_tick(1, Decimal("1"))
    deadline = start + 2.0
    while processed < 50_000 and time.time() < deadline:
        time.sleep(0.01)
    orch.stop()
    assert processed == 50_000
    assert time.time() - start < 2.0
