from __future__ import annotations

import time
from decimal import Decimal
from pathlib import Path

from src.broker.instruments import InstrumentStore
from src.brokers.mock import MockBroker
from src.data.broker_source import BrokerDataSource
from src.execution.broker_executor import BrokerOrderExecutor
from src.strategies.runner import Orchestrator


def test_end_to_end_paper_mode() -> None:
    store = InstrumentStore.from_csv(str(Path("data/instruments_sample.csv")))
    broker = MockBroker()
    ds = BrokerDataSource(broker)
    execu = BrokerOrderExecutor(broker)
    seen: list[str | None] = []

    def strat(tick):
        inst = store.by_token(tick.instrument_id)
        seen.append(inst.symbol if inst else None)
        return None

    orch = Orchestrator(ds, execu, strat)
    orch.start()
    ds.subscribe([101])
    broker.push_tick(101, Decimal("123.45"))
    time.sleep(0.05)
    orch.stop()
    assert seen == ["NIFTY24SEP18000CE"]
