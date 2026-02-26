"""Adaptive walk-forward calibration utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from statistics import mean, pstdev
from typing import Deque


@dataclass(slots=True)
class TradeStats:
    """Rolling trade metrics. Args: None. Returns: None. Raises: None."""

    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    rolling_sharpe: float = 0.0
    max_drawdown: float = 0.0
    signal_frequency: float = 0.0


@dataclass(slots=True)
class AdaptiveParameterStore:
    """Rolling per-strategy stats store. Args: window_trades. Returns: None. Raises: None."""

    window_trades: int = 200
    _pnl: dict[str, Deque[float]] = field(default_factory=dict)
    _stats: dict[str, TradeStats] = field(default_factory=dict)

    def record_trade(self, strategy: str, pnl: float) -> TradeStats:
        """Record strategy pnl. Args: strategy,pnl. Returns: TradeStats. Raises: None."""

        bucket = self._pnl.setdefault(strategy, deque(maxlen=max(1, self.window_trades)))
        bucket.append(float(pnl))
        values = list(bucket)
        wins = [v for v in values if v > 0]
        losses = [abs(v) for v in values if v < 0]
        win_rate = len(wins) / max(len(values), 1)
        avg_win = mean(wins) if wins else 0.0
        avg_loss = mean(losses) if losses else 0.0
        sharpe = 0.0
        if len(values) >= 2:
            std = pstdev(values)
            if std > 0:
                sharpe = (mean(values) / std) * sqrt(len(values))
        eq = 0.0
        peak = 0.0
        dd = 0.0
        for v in values:
            eq += v
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        stats = TradeStats(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            rolling_sharpe=sharpe,
            max_drawdown=dd,
            signal_frequency=len(values) / max(float(self.window_trades), 1.0),
        )
        self._stats[strategy] = stats
        return stats

    def get_stats(self, strategy: str) -> TradeStats:
        """Get current stats. Args: strategy. Returns: TradeStats. Raises: None."""

        return self._stats.get(strategy, TradeStats())


@dataclass(slots=True)
class WalkForwardOptimizer:
    """Walk-forward parameter tuner. Args: recalibrate_every,alpha. Returns: None. Raises: None."""

    recalibrate_every: int = 50
    alpha: float = 0.2
    drawdown_threshold: float = 0.0
    _trade_count: dict[str, int] = field(default_factory=dict)
    _params: dict[str, dict[str, float]] = field(default_factory=dict)
    _regime_params: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    _frozen: bool = False
    risk_scale: float = 1.0

    def should_recalibrate(self, strategy: str) -> bool:
        """Check cadence. Args: strategy. Returns: bool. Raises: None."""

        count = self._trade_count.get(strategy, 0) + 1
        self._trade_count[strategy] = count
        return count % max(1, self.recalibrate_every) == 0

    def optimize(
        self,
        strategy: str,
        regime: str,
        stats: TradeStats,
        current: dict[str, float],
    ) -> dict[str, float]:
        """Optimize local grid. Args: strategy,regime,stats,current. Returns: params. Raises: None."""

        if self._frozen or stats.rolling_sharpe < 0:
            self._frozen = True
            return current
        if self.drawdown_threshold > 0 and stats.max_drawdown > self.drawdown_threshold:
            self.risk_scale = min(self.risk_scale, 0.5)

        mz = current.get('momentum_z_threshold', 0.5)
        mv = current.get('microvol_percentile', 60.0)
        sp = current.get('spread_threshold_pct', 0.3)
        candidates: list[dict[str, float]] = []
        for dm in (-0.1, 0.0, 0.1):
            for dv in (-5.0, 0.0, 5.0):
                for ds in (-0.05, 0.0, 0.05):
                    candidates.append({
                        'momentum_z_threshold': max(0.1, mz + dm),
                        'microvol_percentile': min(95.0, max(5.0, mv + dv)),
                        'spread_threshold_pct': max(0.01, sp + ds),
                    })
        best = max(candidates, key=lambda p: self._objective(stats, p))
        prev = self._params.get(strategy, current)
        blended = {
            key: (1.0 - self.alpha) * float(prev.get(key, value)) + self.alpha * float(value)
            for key, value in best.items()
        }
        self._params[strategy] = blended
        self._regime_params.setdefault(strategy, {})[regime] = blended
        return blended

    def on_regime_change(self, strategy: str, regime: str, current: dict[str, float]) -> dict[str, float]:
        """Load regime params with EMA blend. Args: strategy,regime,current. Returns: params. Raises: None."""

        target = self._regime_params.get(strategy, {}).get(regime)
        if not target:
            return current
        return {
            key: (1.0 - self.alpha) * float(current.get(key, value)) + self.alpha * float(value)
            for key, value in target.items()
        }

    @staticmethod
    def _objective(stats: TradeStats, params: dict[str, float]) -> float:
        """Objective score. Args: stats,params. Returns: float. Raises: None."""

        spread_penalty = params['spread_threshold_pct'] * 0.2
        return stats.rolling_sharpe + stats.win_rate - spread_penalty
