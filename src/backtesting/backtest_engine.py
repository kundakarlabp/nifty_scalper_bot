"""Event-driven backtest engine."""

from __future__ import annotations

import csv
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast

from src.risk.greeks import OptionType
from src.risk.limits import Exposure, RiskEngine
from src.strategies.parameters import StrategyParameters
from src.strategies.strategy_config import StrategyConfig
from src.utils import strike_selector

from .data_feed import SpotFeed
from .sim_connector import SimConnector

if TYPE_CHECKING:  # pragma: no cover
    pass


class BacktestStrategy(Protocol):
    """Minimal interface required by the backtest engine."""

    def evaluate_from_backtest(
        self, ts: datetime, o: float, h: float, low: float, c: float, v: float
    ) -> dict[str, Any] | None:
        ...


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
        *,
        strategy_factory: Callable[[StrategyParameters | None], BacktestStrategy] | None = None,
        write_results: bool = True,
    ) -> None:
        self.feed = feed
        self.cfg = cfg
        self.risk = risk
        self.sim = sim
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.trades: list[dict[str, Any]] = []
        self.equity = 100_000.0
        self.equity_curve: list[tuple[str, float]] = []
        self._strategy_factory = strategy_factory
        self.write_results = write_results

    def run(
        self,
        start: str | None = None,
        end: str | None = None,
        *,
        params: StrategyParameters | None = None,
    ) -> dict[str, float]:
        """Execute the backtest between ``start`` and ``end`` timestamps."""

        bars = self.feed.window(start, end)
        strategy = self._build_strategy(params)

        for ts, o, h, low, c, v in bars.iter_bars():
            ts_local = ts.replace(tzinfo=self.feed.tz) if ts.tzinfo is None else ts
            plan = strategy.evaluate_from_backtest(ts, o, h, low, c, v)
            if plan is None:
                continue
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
            tsym = str(tsym)
            K = parsed["strike"]
            opt = cast(OptionType, parsed["option_type"])
            ob = self.sim.synth_option_book(
                spot=c,
                strike=K,
                opt_type=opt,
                now=ts_local,
                atr_pct=plan["atr_pct"],
            )
            spread = ob["ask"] - ob["bid"]
            mid = ob["mid"]

            ok, _, _ = self.risk.pre_trade_check(
                equity_rupees=self.equity,
                plan=plan,
                exposure=self._exposure_snapshot(),
                intended_symbol=tsym,
                intended_lots=plan.get("qty_lots", 1),
                lot_size=self.sim.lot_size,
                entry_price=mid,
                stop_loss_price=plan.get("sl") or max(0.5, mid - (spread * 2)),
                spot_price=c,
                quote={"mid": mid},
                now=ts_local,
            )
            if not ok:
                plan["has_signal"] = False
                plan["reason_block"] = "risk_block"
                continue

            side = "BUY" if plan.get("action") == "BUY" else "SELL"
            jitter = random.uniform(0.0, 0.25)
            entry_ts = ts + timedelta(seconds=plan.get("entry_wait", 0) + jitter)
            qty = self.sim.lot_size * plan.get("qty_lots", 1)
            ok0, entry_px, slippage = self.sim.ioc_fill(side, ob, qty)
            if not ok0:
                plan["has_signal"] = False
                plan["reason_block"] = "depth_reject"
                continue
            costs_in = self.sim.apply_costs(
                "BUY" if side == "BUY" else "SELL", entry_px, qty
            )
            R = abs((plan.get("sl") or (entry_px - spread)) - entry_px)
            tp1 = entry_px + plan.get("tp1_R", 1.0) * R * (1 if side == "BUY" else -1)
            _tp2 = entry_px + plan.get("tp2_R", 1.6) * R * (1 if side == "BUY" else -1)
            time_stop_ts = entry_ts + timedelta(minutes=plan.get("time_stop_min", 12))

            exit_px = None
            exit_ts = None
            exit_reason = "time"
            mae = 0.0
            mfe = 0.0
            for fts, _fo, fh, fl, fc, _fv in self.feed.iter_bars():
                if fts <= entry_ts:
                    continue
                if side == "BUY":
                    mae = max(mae, entry_px - fl)
                    mfe = max(mfe, fh - entry_px)
                else:
                    mae = max(mae, fh - entry_px)
                    mfe = max(mfe, entry_px - fl)
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

            costs_out = self.sim.apply_costs(
                "SELL" if side == "BUY" else "BUY", exit_px, qty
            )
            gross = (exit_px - entry_px) * (1 if side == "BUY" else -1) * qty
            net = gross - (costs_in + costs_out)
            pnl_R = ((exit_px - entry_px) * (1 if side == "BUY" else -1)) / max(1e-6, R)

            self.equity += net
            self.equity_curve.append((exit_ts.isoformat(), round(self.equity, 2)))

            slippage_bps = (slippage / mid * 10_000) if mid else 0.0
            bucket = f"{entry_ts.hour:02d}:{(entry_ts.minute // 30) * 30:02d}"

            self.trades.append(
                {
                    "ts_entry": entry_ts.isoformat(),
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
                    "regime": plan.get("regime", "TREND"),
                    "slippage": round(slippage, 2),
                    "mae": round(mae, 2),
                    "mfe": round(mfe, 2),
                    "slippage_bps": round(slippage_bps, 2),
                    "bucket": bucket,
                }
            )
            self.risk.on_trade_closed(pnl_R=pnl_R)

        if self.write_results:
            trades_csv = os.path.join(self.outdir, "trades.csv")
            with open(trades_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=(
                        list(self.trades[0].keys())
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
                        ]
                    ),
                )
                writer.writeheader()
                for t in self.trades:
                    writer.writerow(t)
            eq_path = os.path.join(self.outdir, "equity_curve.csv")
            with open(eq_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ts", "equity"])
                w.writerows(self.equity_curve)

        summary: dict[str, Any] = self._summary_metrics(self.trades)
        by_regime: dict[str, list[float]] = {"TREND": [], "RANGE": []}
        for t in self.trades:
            reg = str(t.get("regime", "TREND"))
            by_regime.setdefault(reg, []).append(
                float(cast(float, t.get("pnl_R", 0.0)))
            )

        def pf(rs: list[float]) -> float:
            pos = [r for r in rs if r > 0]
            neg = [-r for r in rs if r < 0]
            return (sum(pos) / sum(neg)) if neg else float("inf")

        summary.update(
            {
                "PF_trend": round(pf(by_regime.get("TREND", [])), 2),
                "PF_range": round(pf(by_regime.get("RANGE", [])), 2),
                "trades_trend": len(by_regime.get("TREND", [])),
                "trades_range": len(by_regime.get("RANGE", [])),
            }
        )

        bucket_stats: dict[tuple[str, str], dict[str, Any]] = {}
        for t in self.trades:
            reg = str(t.get("regime", "TREND"))
            bkt = str(t.get("bucket", "00:00"))
            key = (reg, bkt)
            stats = bucket_stats.setdefault(
                key,
                {
                    "pnl": 0.0,
                    "hit": 0,
                    "trades": 0,
                    "slippage": [],
                    "mae": [],
                    "mfe": [],
                },
            )
            stats["pnl"] += float(t.get("pnl_rupees", 0.0))
            stats["hit"] += 1 if float(t.get("pnl_rupees", 0.0)) > 0 else 0
            stats["trades"] += 1
            stats["slippage"].append(float(t.get("slippage_bps", 0.0)))
            stats["mae"].append(float(t.get("mae", 0.0)))
            stats["mfe"].append(float(t.get("mfe", 0.0)))

        bucket_report: dict[str, dict[str, Any]] = {}
        for (reg, bkt), s in bucket_stats.items():
            bucket_report.setdefault(reg, {})[bkt] = {
                "PnL": round(s["pnl"], 2),
                "HitRate": round(100 * s["hit"] / s["trades"], 1)
                if s["trades"]
                else 0.0,
                "slippage_bps": round(sum(s["slippage"]) / s["trades"], 2)
                if s["trades"]
                else 0.0,
                "MAE": round(sum(s["mae"]) / s["trades"], 2) if s["trades"] else 0.0,
                "MFE": round(sum(s["mfe"]) / s["trades"], 2) if s["trades"] else 0.0,
            }

        summary["by_bucket"] = bucket_report
        if self.write_results:
            with open(os.path.join(self.outdir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            with open(os.path.join(self.outdir, "bucket_report.json"), "w") as f:
                json.dump(bucket_report, f, indent=2)
        return summary

    def _summary_metrics(self, trades: list[dict[str, Any]]) -> dict[str, float]:
        """Compute a few aggregate performance metrics."""

        if not trades:
            return {"trades": 0}
        rs = [float(t["pnl_R"]) for t in trades]
        pos = [r for r in rs if r > 0]
        neg = [-r for r in rs if r < 0]
        pf = (sum(pos) / sum(neg)) if neg else float("inf")
        win = (sum(1 for r in rs if r > 0) / len(rs)) * 100.0
        avgR = sum(rs) / len(rs)
        pnl = sum(float(t.get("pnl_rupees", 0.0)) for t in trades)
        slippages = [float(t.get("slippage", 0.0)) for t in trades]
        max_ae = max(float(t.get("mae", 0.0)) for t in trades)
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
            "PnL": round(pnl, 2),
            "HitRate": round(win, 1),
            "AvgSlippage": round(sum(slippages) / len(slippages), 2),
            "MaxAE": round(max_ae, 2),
        }

    def _exposure_snapshot(self) -> Exposure:
        """Return a dummy exposure snapshot (flat)."""

        return Exposure()

    def _build_strategy(
        self, params: StrategyParameters | None
    ) -> BacktestStrategy:
        if self._strategy_factory is not None:
            return self._strategy_factory(params)
        from src.strategies.scalping_strategy import ScalpingStrategy

        if params is None:
            return ScalpingStrategy()
        return ScalpingStrategy(params=params)
