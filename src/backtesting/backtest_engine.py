from __future__ import annotations

"""Event-driven backtest engine."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import csv
import json
import os

from .data_feed import SpotFeed
from .sim_connector import SimConnector
from src.strategies.strategy_config import StrategyConfig
from src.risk.limits import RiskEngine
from src.utils import strike_selector


@dataclass
class BTParams:
    """Basic backtest parameters."""

    lot_size: int = 50
    time_stop_min: int = 12
    slippage_entry_frac: float = 0.50
    slippage_exit_frac: float = 0.40
    tz: str = "Asia/Kolkata"


class BacktestEngine:
    """Drive a strategy over historical data and record trades."""

    def __init__(
        self,
        feed: SpotFeed,
        cfg: StrategyConfig,
        risk: RiskEngine,
        sim: SimConnector,
        outdir: str,
    ) -> None:
        self.feed = feed
        self.cfg = cfg
        self.risk = risk
        self.sim = sim
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.trades: List[Dict[str, object]] = []

    def run(self, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, float]:
        """Execute the backtest between ``start`` and ``end`` timestamps."""

        bars = self.feed.window(start, end)
        from src.strategies.scalping_strategy import ScalpingStrategy

        strategy = ScalpingStrategy()
        equity = 100_000.0

        for ts, o, h, l, c, v in bars.iter_bars():
            plan = strategy.evaluate_from_backtest(ts, o, h, l, c, v)
            plan["bar_count"] = max(plan.get("bar_count", 0), 20)
            plan["atr_pct"] = plan.get("atr_pct", 0.45)

            if not plan.get("has_signal"):
                continue

            tsym = plan.get("strike")
            parsed = strike_selector.parse_nfo_symbol(tsym) if tsym else None
            if not parsed:
                plan["has_signal"] = False
                plan["reason_block"] = "no_option_token"
                continue
            K = parsed["strike"]
            opt = parsed["option_type"]
            ob = self.sim.synth_option_book(spot=c, strike=K, opt_type=opt, now=ts, atr_pct=plan["atr_pct"])
            spread = ob["ask"] - ob["bid"]
            mid = ob["mid"]
            ladd0, ladd1 = self.sim.ladder_prices(mid, spread)

            ok, _, _ = self.risk.pre_trade_check(
                equity_rupees=equity,
                plan=plan,
                exposure=self._exposure_snapshot(),
                intended_symbol=tsym,
                intended_lots=plan.get("qty_lots", 1),
                lot_size=self.sim.lot_size,
                entry_price=ladd0,
                stop_loss_price=plan.get("sl") or max(0.5, ladd0 - (spread * 2)),
            )
            if not ok:
                plan["has_signal"] = False
                plan["reason_block"] = "risk_block"
                continue

            side = "BUY" if plan.get("action") == "BUY" else "SELL"
            if side == "BUY":
                ok0, px0 = self.sim.fill_limit_buy(ladd0, ob["bid"], ob["ask"])
                if not ok0:
                    ok1, px1 = self.sim.fill_limit_buy(ladd1, ob["bid"], ob["ask"])
                    if not ok1:
                        continue
                    entry_px = px1
                else:
                    entry_px = px0
            else:
                ok0, px0 = self.sim.fill_limit_sell(ladd0, ob["bid"], ob["ask"])
                if not ok0:
                    ok1, px1 = self.sim.fill_limit_sell(ladd1, ob["bid"], ob["ask"])
                    if not ok1:
                        continue
                    entry_px = px1
                else:
                    entry_px = px0

            qty = self.sim.lot_size * plan.get("qty_lots", 1)
            costs_in = self.sim.apply_costs("BUY" if side == "BUY" else "SELL", entry_px, qty)
            R = abs((plan.get("sl") or (entry_px - spread)) - entry_px)
            tp1 = entry_px + plan.get("tp1_R", 1.0) * R * (1 if side == "BUY" else -1)
            tp2 = entry_px + plan.get("tp2_R", 1.6) * R * (1 if side == "BUY" else -1)
            time_stop_ts = ts + timedelta(minutes=plan.get("time_stop_min", 12))

            exit_px = None
            exit_ts = None
            exit_reason = "time"
            for fts, fo, fh, fl, fc, fv in self.feed.iter_bars():
                if fts <= ts:
                    continue
                hi, lo = (fh, fl) if side == "BUY" else (fl, fh)
                if (side == "BUY" and fh >= tp1) or (side == "SELL" and fl <= tp1):
                    exit_px, exit_ts, exit_reason = tp1, fts, "TP1"
                    break
                if (side == "BUY" and fl <= (plan.get("sl") or 0.0)) or (
                    side == "SELL" and fh >= (plan.get("sl") or 9e9)
                ):
                    exit_px, exit_ts, exit_reason = plan.get("sl"), fts, "SL"
                    break
                if fts >= time_stop_ts:
                    exit_px, exit_ts, exit_reason = fc, fts, "TIME"
                    break

            if exit_px is None or exit_ts is None:
                continue

            costs_out = self.sim.apply_costs("SELL" if side == "BUY" else "BUY", exit_px, qty)
            gross = (exit_px - entry_px) * (1 if side == "BUY" else -1) * qty
            net = gross - (costs_in + costs_out)
            pnl_R = ((exit_px - entry_px) * (1 if side == "BUY" else -1)) / max(1e-6, R)

            self.trades.append(
                {
                    "ts_entry": ts.isoformat(),
                    "ts_exit": exit_ts.isoformat(),
                    "side": side,
                    "strike": tsym,
                    "qty": qty,
                    "entry": round(entry_px, 2),
                    "exit": round(exit_px, 2),
                    "exit_reason": exit_reason,
                    "R": round(R, 2),
                    "pnl_R": round(pnl_R, 2),
                    "pnl_rupees": round(net, 2),
                }
            )
            self.risk.on_trade_closed(pnl_R=pnl_R)

        trades_csv = os.path.join(self.outdir, "trades.csv")
        with open(trades_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(self.trades[0].keys())
                if self.trades
                else [
                    "ts_entry",
                    "ts_exit",
                    "side",
                    "strike",
                    "qty",
                    "entry",
                    "exit",
                    "exit_reason",
                    "R",
                    "pnl_R",
                    "pnl_rupees",
                ],
            )
            writer.writeheader()
            for t in self.trades:
                writer.writerow(t)

        summary = self._summary_metrics(self.trades)
        with open(os.path.join(self.outdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _summary_metrics(self, trades: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute a few aggregate performance metrics."""

        if not trades:
            return {"trades": 0}
        rs = [float(t["pnl_R"]) for t in trades]
        pos = [r for r in rs if r > 0]
        neg = [-r for r in rs if r < 0]
        pf = (sum(pos) / sum(neg)) if neg else float("inf")
        win = (sum(1 for r in rs if r > 0) / len(rs)) * 100.0
        avgR = sum(rs) / len(rs)
        eq = 0.0
        peak = 0.0
        mdd = 0.0
        for r in rs:
            eq += r
            peak = max(peak, eq)
            mdd = min(mdd, eq - peak)
        return {
            "trades": len(rs),
            "PF": round(pf, 2),
            "Win%": round(win, 1),
            "AvgR": round(avgR, 2),
            "MaxDD_R": round(-mdd, 2),
        }

    def _exposure_snapshot(self) -> object:
        """Return a dummy exposure snapshot (flat)."""

        from src.risk.limits import Exposure

        return Exposure()
