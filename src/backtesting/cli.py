from __future__ import annotations

"""CLI entrypoint for running walk-forward backtests."""

import argparse
import copy
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Tuple

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.data_feed import SpotFeed
from src.backtesting.metrics import reject
from src.backtesting.sim_connector import SimConnector
from src.risk.limits import LimitConfig, RiskEngine
from src.strategies.strategy_config import resolve_config_path, try_load

log = logging.getLogger(__name__)


def daterange(start: datetime, end: datetime):
    """Yield dates from ``start`` to ``end`` (exclusive)."""

    cur = start
    while cur < end:
        yield cur
        cur = cur + timedelta(days=1)


def build_walkforward_windows(
    start: datetime, end: datetime, train_w: int, test_w: int
) -> List[Tuple[datetime, datetime, datetime]]:
    """Compute walk-forward train/test windows."""

    windows: List[Tuple[datetime, datetime, datetime]] = []
    cur = start
    while True:
        tr_start = cur
        tr_end = tr_start + timedelta(weeks=train_w)
        te_end = tr_end + timedelta(weeks=test_w)
        if tr_end >= end:
            break
        windows.append((tr_start, tr_end, te_end))
        cur = tr_end
    return windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        required=True,
        help="CSV with 1m OHLC (timestamp,open,high,low,close,volume)",
    )
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--wf", default="6:2", help="train_weeks:test_weeks")
    ap.add_argument("--out", default="out_bt")
    args = ap.parse_args()

    feed = SpotFeed.from_csv(args.data)
    cfg = try_load(resolve_config_path(), None)
    risk = RiskEngine(LimitConfig(tz=cfg.tz))
    sim = SimConnector()

    train_w, test_w = map(int, args.wf.split(":"))
    start = datetime.fromisoformat(args.start).replace(tzinfo=None)
    end = datetime.fromisoformat(args.end).replace(tzinfo=None)

    windows = build_walkforward_windows(start, end, train_w, test_w)
    os.makedirs(args.out, exist_ok=True)
    best_overall = None

    for i, (tr_s, tr_e, te_e) in enumerate(windows, 1):
        fold_dir = os.path.join(args.out, f"wf_{i:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        candidates = []
        for min_score in [0.3, 0.35, 0.4]:
            for tp2R in [1.6, 1.8, 2.0]:
                local_cfg = copy.deepcopy(cfg)
                local_cfg.raw.setdefault("strategy", {})["min_score"] = min_score
                local_cfg.tp2_R_trend = tp2R
                risk_tr = RiskEngine(LimitConfig(tz=cfg.tz))
                bt_tr = BacktestEngine(
                    feed,
                    local_cfg,
                    risk_tr,
                    sim,
                    outdir=os.path.join(fold_dir, "train"),
                )
                s_tr = bt_tr.run(start=tr_s.isoformat(), end=tr_e.isoformat())
                if reject(s_tr):
                    continue
                candidates.append((s_tr, {"min_score": min_score, "tp2_R_trend": tp2R}))

        if not candidates:
            with open(os.path.join(fold_dir, "NO_CANDIDATES"), "w") as f:
                f.write("all rejected")
            continue
        best = sorted(
            candidates, key=lambda x: (x[0]["PF"], x[0]["AvgR"]), reverse=True
        )[0]

        risk_te = RiskEngine(LimitConfig(tz=cfg.tz))
        cfg.raw.setdefault("strategy", {})["min_score"] = best[1]["min_score"]
        cfg.tp2_R_trend = best[1]["tp2_R_trend"]
        bt_te = BacktestEngine(
            feed, cfg, risk_te, sim, outdir=os.path.join(fold_dir, "test")
        )
        s_te = bt_te.run(start=tr_e.isoformat(), end=te_e.isoformat())
        with open(os.path.join(fold_dir, "best_config.yaml"), "w") as f:
            import yaml  # type: ignore[import]

            yaml.safe_dump(best[1], f)
        with open(os.path.join(fold_dir, "test_summary.json"), "w") as f:
            json.dump(s_te, f, indent=2)

        if not best_overall or s_te.get("PF", 0) > best_overall.get("PF", 0):
            best_overall = {**s_te, **best[1]}

    if best_overall:
        with open(os.path.join(args.out, "best_overall.json"), "w") as f:
            json.dump(best_overall, f, indent=2)
    log.info("DONE")


if __name__ == "__main__":
    main()
