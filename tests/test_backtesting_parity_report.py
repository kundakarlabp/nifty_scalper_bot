from __future__ import annotations

import pandas as pd

from src.backtesting.parity import parity_report


def test_parity_report_within_tolerance() -> None:
    premium = pd.Series([100.0, 101.0, 102.0])
    live = pd.Series([100.1, 100.9, 101.8])
    rpt = parity_report(premium, live)
    assert rpt.mae < 0.2
    assert rpt.mfe > 0
    assert abs(rpt.slippage_bps) < 100
