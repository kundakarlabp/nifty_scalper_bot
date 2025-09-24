"""CLI entrypoint for running walk-forward backtests."""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.data_feed import SpotFeed
from src.backtesting.sim_connector import SimConnector
from src.backtesting.tuning import WalkForwardResult, WalkForwardSplit, WalkForwardValidator
from src.risk.limits import LimitConfig, RiskEngine
from src.strategies.parameters import StrategyParameters, StrategyParameterSpace
from src.strategies.scalping_strategy import ScalpingStrategy
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
) -> list[tuple[datetime, datetime, datetime]]:
    """Compute walk-forward train/test windows."""

    windows: list[tuple[datetime, datetime, datetime]] = []
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
    ap.add_argument(
        "--trials",
        type=int,
        default=12,
        help="Tuning iterations per walk-forward fold",
    )
    ap.add_argument(
        "--metric",
        default="PF",
        help="Summary metric to optimise during tuning",
    )
    args = ap.parse_args()

    feed = SpotFeed.from_csv(args.data)
    cfg = try_load(resolve_config_path(), None)
    sim = SimConnector()

    train_w, test_w = map(int, args.wf.split(":"))
    start = datetime.fromisoformat(args.start).replace(tzinfo=None)
    end = datetime.fromisoformat(args.end).replace(tzinfo=None)

    windows = build_walkforward_windows(start, end, train_w, test_w)
    os.makedirs(args.out, exist_ok=True)

    splits = [
        WalkForwardSplit(
            train_start=tr_s.isoformat(),
            train_end=tr_e.isoformat(),
            test_start=tr_e.isoformat(),
            test_end=te_e.isoformat(),
        )
        for tr_s, tr_e, te_e in windows
    ]

    if not splits:
        log.error("No walk-forward windows computed for the provided range")
        return

    tuning_cfg = cfg.raw.get("tuning", {}) if isinstance(cfg.raw, dict) else {}
    space = StrategyParameterSpace.from_config(tuning_cfg)

    def _strategy_factory(params: StrategyParameters | None) -> ScalpingStrategy:
        if params is None:
            return ScalpingStrategy()
        return ScalpingStrategy(params=params)

    def engine_factory(
        feed_segment: SpotFeed, outdir: str, write_results: bool
    ) -> BacktestEngine:
        risk = RiskEngine(LimitConfig(tz=cfg.tz))
        return BacktestEngine(
            feed_segment,
            cfg,
            risk,
            sim,
            outdir=outdir,
            strategy_factory=_strategy_factory,
            write_results=write_results,
        )

    validator = WalkForwardValidator(
        feed,
        space,
        engine_factory,
        metric=args.metric,
        maximize=True,
    )

    results = validator.run(splits, trials=args.trials, outdir=args.out)

    best_overall: WalkForwardResult | None = None
    best_metric = float("-inf")

    for idx, result in enumerate(results, 1):
        fold_dir = os.path.join(args.out, f"wf_{idx:02d}")
        params_payload = result.tuning.best.params.as_settings_update()
        with open(os.path.join(fold_dir, "best_params.json"), "w") as f:
            json.dump(params_payload, f, indent=2)
        with open(os.path.join(fold_dir, "test_summary.json"), "w") as f:
            json.dump(result.test_summary, f, indent=2)
        history = [
            {
                "score": tr.score,
                "params": tr.params.as_settings_update(),
                "summary": tr.summary,
            }
            for tr in result.tuning.trials
        ]
        with open(os.path.join(fold_dir, "tuning_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        metric_val = float(result.test_summary.get(args.metric, 0.0))
        if metric_val > best_metric:
            best_metric = metric_val
            best_overall = result

    if best_overall is not None:
        payload = {
            "metric": args.metric,
            "metric_value": best_metric,
            "params": best_overall.tuning.best.params.as_settings_update(),
            "summary": best_overall.test_summary,
        }
        with open(os.path.join(args.out, "best_overall.json"), "w") as f:
            json.dump(payload, f, indent=2)
    log.info("DONE")


if __name__ == "__main__":
    main()
