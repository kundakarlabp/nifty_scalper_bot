from __future__ import annotations

import csv
import time
from decimal import Decimal
from pathlib import Path

from src.brokers.mock import MockBroker
from src.data.broker_source import BrokerDataSource
from src.execution.broker_executor import BrokerOrderExecutor
from src.strategies.runner import Orchestrator
from src.backtesting.parity import parity_report
import pandas as pd


def test_tick_replay_one_day() -> None:
    data_path = Path(__file__).with_name("data") / "replay_ticks_day.csv"
    broker = MockBroker()
    ds = BrokerDataSource(broker)
    execu = BrokerOrderExecutor(broker)
    seen: list[float] = []

    def strat(tick):
        seen.append(float(tick.ltp))
        return None

    orch = Orchestrator(ds, execu, strat)
    orch.start()
    ds.subscribe([101])
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            broker.push_tick(int(row["instrument_id"]), Decimal(row["ltp"]))
    time.sleep(0.1)
    orch.stop()
    expected = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5]
    rpt = parity_report(pd.Series(expected), pd.Series(seen))
    assert rpt.mae == 0
    assert rpt.mfe == 0
    assert rpt.slippage_bps == 0
