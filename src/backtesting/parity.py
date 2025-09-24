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

    # ``ref`` can be exactly zero for deep ITM/OTM legs. Pandas would emit
    # ``inf`` for those divisions which then poisons the mean and results in a
    # bogus slippage number.  We treat such points as contributing zero
    # slippage instead of blowing up the metric.
    mask = ref.abs() > 1e-9
    with pd.option_context("mode.use_inf_as_na", True):
        slippage_ratio = (diff[mask] / ref[mask]).dropna()
    slippage_bps = float((slippage_ratio.mean() if not slippage_ratio.empty else 0.0) * 10_000)
    return ParityReport(mae=mae, mfe=mfe, slippage_bps=slippage_bps)
