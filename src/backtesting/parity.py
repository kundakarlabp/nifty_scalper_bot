from __future__ import annotations

"""Utilities to compare backtest output with live metrics."""

from dataclasses import dataclass
import pandas as pd


@dataclass
class ParityReport:
    """Simple parity stats between two series."""

    mae: float
    mfe: float
    slippage_bps: float


def parity_report(premium: pd.Series, live: pd.Series) -> ParityReport:
    """Return MAE/MFE/slippage stats for ``premium`` vs ``live`` series.

    Both inputs must be aligned ``pd.Series`` with numeric values.  The
    returned dataclass exposes:
    - ``mae``: mean absolute error
    - ``mfe``: maximum favourable excursion
    - ``slippage_bps``: average slippage in basis points
    """

    if premium.empty or live.empty:
        raise ValueError("inputs must be non-empty")
    if len(premium) != len(live):
        raise ValueError("series length mismatch")

    diff = premium - live
    mae = float(diff.abs().mean())
    mfe = float(diff.max())
    ref = (premium + live) / 2.0
    slippage_bps = float(((diff / ref).mean()) * 10_000)
    return ParityReport(mae=mae, mfe=mfe, slippage_bps=slippage_bps)
