#!/usr/bin/env python3
"""Independent Backtrader gross-P&L oracle for historical research ledgers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import backtrader as bt
import pandas as pd


class _LedgerReplay(bt.Strategy):
    params = (("quantities", ()),)

    def next(self) -> None:
        bar_index = len(self) - 1
        trade_index = bar_index // 3
        if trade_index >= len(self.p.quantities):
            return
        phase = bar_index % 3
        quantity = int(self.p.quantities[trade_index])
        if phase == 0:
            if self.position:
                raise RuntimeError("external_oracle_position_overlap")
            self.buy(size=quantity)
        elif phase == 1:
            if not self.position:
                raise RuntimeError("external_oracle_entry_not_filled")
            self.sell(size=quantity)

    def stop(self) -> None:
        if self.position:
            raise RuntimeError("external_oracle_position_open")


def backtrader_gross_pnl(trades: list[dict[str, Any]]) -> float:
    """Replay resolved long round trips without using the bot P&L formula."""
    if not trades:
        return 0.0
    prices: list[float] = []
    quantities: list[int] = []
    for trade in trades:
        entry = float(trade["entry_price"])
        exit_price = float(trade["exit_price"])
        quantity = int(trade["quantity"])
        if (
            not math.isfinite(entry)
            or not math.isfinite(exit_price)
            or entry <= 0
            or exit_price <= 0
            or quantity <= 0
        ):
            raise ValueError("external_oracle_trade_invalid")
        # Default Backtrader market orders fill at the next bar open.\n        # Bar 0 queues BUY, bar 1 opens at entry and queues SELL, and bar 2\n        # opens at exit. This independently reproduces the resolved ledger fill\n        # chronology instead of accidentally buying at the exit price.\n        prices.extend((entry, entry, exit_price))
        quantities.append(quantity)

    index = pd.date_range("2000-01-01", periods=len(prices), freq="min")
    frame = pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1.0] * len(prices),
            "openinterest": [0.0] * len(prices),
        },
        index=index,
    )
    starting_cash = 1_000_000_000_000.0
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(starting_cash)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.broker.set_coc(True)
    cerebro.adddata(bt.feeds.PandasData(dataname=frame))
    cerebro.addstrategy(_LedgerReplay, quantities=tuple(quantities))
    cerebro.run(runonce=False, preload=False)
    return float(cerebro.broker.getvalue() - starting_cash)


def validate_result_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for result in payload:
        trades = list(result.get("trades", []))
        external = backtrader_gross_pnl(trades)
        reported = sum(float(trade["gross_pnl"]) for trade in trades)
        tolerance = max(1e-6, abs(reported) * 1e-10)
        parity = math.isclose(external, reported, rel_tol=1e-10, abs_tol=tolerance)
        rows.append(
            {
                "candidate": result["candidate"],
                "slippage_bps_per_side": result["slippage_bps_per_side"],
                "trade_count": len(trades),
                "reported_gross_pnl": reported,
                "backtrader_gross_pnl": external,
                "gross_pnl_parity": parity,
            }
        )
        if not parity:
            raise ValueError("external_backtrader_gross_pnl_mismatch")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phases: dict[str, list[dict[str, Any]]] = {}
    for phase in ("development", "validation", "final"):
        path = args.study_dir / f"{phase}_results.json"
        if path.is_file():
            phases[phase] = validate_result_file(path)
    if not phases:
        raise ValueError("external_oracle_results_unavailable")
    report = {
        "engine": "backtrader",
        "scope": "independent_resolved_trade_gross_pnl_oracle",
        "signal_generation_independent": False,
        "fee_model_independent": False,
        "all_rows_match": all(
            row["gross_pnl_parity"] for rows in phases.values() for row in rows
        ),
        "phases": phases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
