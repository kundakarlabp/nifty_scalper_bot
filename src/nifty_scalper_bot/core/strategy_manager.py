"""Strategy scoring and dynamic allocation manager."""

from __future__ import annotations

import logging
import os
import re
import time
import typing as t
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import sqrt
from statistics import mean, pstdev

from nifty_scalper_bot.config import settings as app_settings
from nifty_scalper_bot.core.adaptive_calibration import (
    AdaptiveParameterStore,
    WalkForwardOptimizer,
)
from nifty_scalper_bot.core.market_regime import RegimeSnapshot
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy
from nifty_scalper_bot.strategies.signal_quality import infer_option_side
from nifty_scalper_bot.strategies.signal_generator import (
    Signal,
)
from nifty_scalper_bot.strategies.signal_generator import (
    StrategyManager as _BaseStrategyManager,
)
from nifty_scalper_bot.utils.logging import get_logger, log_state_change, log_throttled
from nifty_scalper_bot.utils.symbols import normalize_symbol

log = get_logger(__name__)


def classify_symbol_role(symbol: str) -> str:
    """Classify symbols for strategy routing."""
    s = normalize_symbol(str(symbol or ""))
    u = s.upper()
    if u in {"NSE:NIFTY", "NIFTY", "NSE:NIFTY50", "NIFTY50"}:
        return "spot_context"
    if u.startswith("NFO:NIFTY") and u.endswith("FUT"):
        return "futures_context"
    if u.startswith("NFO:NIFTY") and (u.endswith("CE") or u.endswith("PE")):
        return "tradable_option"
    return "unknown"

REGIME_STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "TREND_UP": {"SMC": 1.2, "VWAPPro": 1.2, "ORBPro": 1.15, "BBSqueeze": 1.1, "OrderFlow": 1.15, "RSIDivergence": 0.8},
    "TREND_DOWN": {"SMC": 1.2, "VWAPPro": 1.2, "ORBPro": 1.15, "BBSqueeze": 1.1, "OrderFlow": 1.15, "RSIDivergence": 0.8},
    "RANGE": {"RSIDivergence": 1.15, "OIMaxPain": 1.05, "ORBPro": 0.7, "VWAPPro": 0.8, "SMC": 0.85},
    "CHOPPY": {"SMC": 0.2, "VWAPPro": 0.2, "ORBPro": 0.2, "BBSqueeze": 0.3, "OrderFlow": 0.25},
    "EXPIRY_GAMMA": {"GammaScalping": 1.25, "EliteTuesdayGammaBuyer": 1.25},
    "LOW_VOLATILITY": {"BBSqueeze": 0.6, "VWAPPro": 0.7, "ORBPro": 0.6},
}


class StrategyInterface(ABC):
    """Define the standardised contract expected from all strategies."""

    name: str

    @abstractmethod
    def should_trade(self, market_state: t.Mapping[str, t.Any]) -> bool:
        """Decide whether the strategy should trade the current *market_state*.

        Args:
            market_state: Mapping describing the evaluated market environment.

        Returns:
            bool: ``True`` if the strategy wants to participate in the cycle.

        Raises:
            NotImplementedError: Raised when subclasses omit the override.
        """

        try:
            raise NotImplementedError("Concrete strategies must implement should_trade")
        except Exception as exc:  # noqa: BLE001 - intentional propagation
            log.error(
                "Failure in StrategyInterface.should_trade: %s",
                exc,
                exc_info=exc,
            )
            raise

    @abstractmethod
    def generate_signals(self, market_data: t.Mapping[str, t.Any]) -> list[Signal]:
        """Produce a list of signals for the provided *market_data* snapshot.

        Args:
            market_data: Mapping containing enriched indicator and price data.

        Returns:
            list[Signal]: Collection of generated trading signals.

        Raises:
            NotImplementedError: Raised when subclasses omit the override.
        """

        try:
            raise NotImplementedError(
                "Concrete strategies must implement generate_signals"
            )
        except Exception as exc:  # noqa: BLE001 - intentional propagation
            log.error(
                "Failure in StrategyInterface.generate_signals: %s",
                exc,
                exc_info=exc,
            )
            raise

    @abstractmethod
    def score_performance(
        self, trade_history: t.Sequence[t.Mapping[str, t.Any]]
    ) -> dict[str, float]:
        """Score the strategy using *trade_history* derived metrics.

        Args:
            trade_history: Ordered trade snapshots with realised PnL values.

        Returns:
            dict[str, float]: Dictionary containing PnL, hit rate, Sharpe, drawdown.

        Raises:
            NotImplementedError: Raised when subclasses omit the override.
        """

        try:
            raise NotImplementedError(
                "Concrete strategies must implement score_performance"
            )
        except Exception as exc:  # noqa: BLE001 - intentional propagation
            log.error(
                "Failure in StrategyInterface.score_performance: %s",
                exc,
                exc_info=exc,
            )
            raise


class StrategyAdapter(StrategyInterface):
    """Bridge legacy strategy implementations to the registry contract."""

    def __init__(self, strategy: t.Any) -> None:
        """Initialise the adapter with the provided *strategy* instance.

        Args:
            strategy: Concrete strategy instance to bridge into the registry.

        Returns:
            None.

        Raises:
            ValueError: Raised when *strategy* is ``None``.
        """

        log.debug("Entered StrategyAdapter.__init__")
        try:
            if strategy is None:
                raise ValueError("StrategyAdapter requires a strategy instance")
            self._strategy = strategy
            self.name = getattr(strategy, "name", strategy.__class__.__name__)
        except Exception as exc:  # noqa: BLE001 - defensive logging
            log.error("Failure in StrategyAdapter.__init__: %s", exc, exc_info=exc)
            raise

    def should_trade(self, market_state: t.Mapping[str, t.Any]) -> bool:
        """Return readiness to trade using optional underlying hook.

        Args:
            market_state: Mapping describing the evaluated market environment.

        Returns:
            bool: ``True`` when the underlying strategy approves trading.

        Raises:
            None.
        """

        log.debug("Entered StrategyAdapter.should_trade")
        try:
            hook = getattr(self._strategy, "should_trade", None)
            if callable(hook):
                return bool(hook(market_state))
            return True
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error("Failure in StrategyAdapter.should_trade: %s", exc, exc_info=exc)
            return False

    def generate_signals(self, market_data: t.Mapping[str, t.Any]) -> list[Signal]:
        """Generate signals using either bulk or single-signal hooks.

        Args:
            market_data: Mapping containing enriched indicator and price data.

        Returns:
            list[Signal]: Generated signals filtered for validity.

        Raises:
            None.
        """

        log.debug("Entered StrategyAdapter.generate_signals")
        signals: list[Signal] = []
        try:
            multi_hook = getattr(self._strategy, "generate_signals", None)
            if callable(multi_hook):
                generated = multi_hook(market_data)
                if isinstance(generated, list):
                    return [
                        signal for signal in generated if isinstance(signal, Signal)
                    ]
            single_hook = getattr(self._strategy, "generate_signal", None)
            if callable(single_hook) and isinstance(market_data, t.Mapping):
                symbol_raw = market_data.get("symbol", "")
                symbol = str(symbol_raw) if symbol_raw is not None else ""
                indicators_raw = market_data.get("indicators", {})
                if isinstance(indicators_raw, t.Mapping):
                    indicators = dict(indicators_raw)
                else:
                    indicators = {}
                current_price_raw = market_data.get("current_price", 0.0)
                try:
                    current_price = float(current_price_raw)
                except Exception as price_exc:  # noqa: BLE001 - defensive cast
                    log.error(
                        "Failure casting current_price in "
                        "StrategyAdapter.generate_signals: %s",
                        price_exc,
                        exc_info=price_exc,
                    )
                    current_price = 0.0
                position = market_data.get("position")
                signal = single_hook(symbol, indicators, current_price, position)
                if isinstance(signal, Signal):
                    signals.append(signal)
            return signals
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in StrategyAdapter.generate_signals: %s",
                exc,
                exc_info=exc,
            )
            return []

    def score_performance(
        self, trade_history: t.Sequence[t.Mapping[str, t.Any]]
    ) -> dict[str, float]:
        """Return standardised performance metrics for *trade_history*.

        Args:
            trade_history: Sequence of trade metadata dictionaries.

        Returns:
            dict[str, float]: Dictionary with pnl, hit_rate, sharpe, drawdown.

        Raises:
            None.
        """

        log.debug("Entered StrategyAdapter.score_performance")
        try:
            hook = getattr(self._strategy, "score_performance", None)
            if callable(hook):
                result = hook(trade_history)
                if isinstance(result, dict):
                    return result
        except Exception as exc:  # noqa: BLE001 - continue with fallback
            log.error(
                "Failure in StrategyAdapter score hook: %s",
                exc,
                exc_info=exc,
            )
        performance = StrategyPerformance()
        try:
            for trade in trade_history:
                pnl_raw = trade.get("pnl", 0.0)
                pnl = float(pnl_raw)
                performance.record(pnl)
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in StrategyAdapter.score_performance: %s", exc, exc_info=exc
            )
        return {
            "pnl": performance.total_pnl,
            "hit_rate": performance.hit_rate(),
            "sharpe": performance.sharpe_ratio(),
            "drawdown": performance.max_drawdown(),
        }


StrategyFactory = t.Callable[..., StrategyInterface | t.Any]


@dataclass(slots=True)
class StrategyVote:
    """Args: strategy vote fields. Returns: structured vote. Raises: none."""

    strategy: str
    side: str
    score: float
    confidence: float
    reasons: list[str]
    metadata: dict[str, t.Any]


def signal_to_vote(signal: Signal, strategy_name: str) -> StrategyVote:
    """Args: signal + strategy name. Returns: normalized vote. Raises: none."""
    metadata = dict(signal.metadata or {})
    symbol_side = infer_option_side(signal.symbol, metadata)
    metadata_side = str(metadata.get("trade_side") or metadata.get("side") or "").upper()
    if symbol_side in {"CE", "PE"}:
        if metadata_side in {"CE", "PE"} and metadata_side != symbol_side:
            metadata["side_conflict"] = True
            metadata["side_from_metadata"] = metadata_side
        side = symbol_side
    else:
        side = metadata_side or symbol_side
    if signal.action == "HOLD" and side not in {"CE", "PE"}:
        side = "NO_TRADE"
    score_candidate = float(
        metadata.get("raw_setup_score")
        if metadata.get("raw_setup_score") is not None
        else metadata.get("setup_score")
        if metadata.get("setup_score") is not None
        else metadata.get("strategy_score")
        if metadata.get("strategy_score") is not None
        else metadata.get("setup_quality")
        if metadata.get("setup_quality") is not None
        else 0.0
    )
    confidence_score = float(signal.confidence or 0.0) * 10.0

    raw_setup_score = max(0.0, min(10.0, score_candidate))
    vote_score = max(raw_setup_score, max(0.0, min(10.0, confidence_score)))
    reason = str(signal.reason or '').strip()
    reason_list = list(metadata.get("score_reasons") or [])
    if reason and reason not in reason_list:
        reason_list.append(reason)
    metadata.setdefault("strategy", strategy_name)
    metadata.setdefault("required_data_present", True)
    metadata.setdefault("setup_quality", raw_setup_score)
    metadata["vote_score_raw_strategy"] = raw_setup_score
    metadata["vote_score_from_confidence"] = max(0.0, min(10.0, confidence_score))
    metadata["vote_score"] = vote_score
    metadata["raw_setup_score"] = raw_setup_score
    metadata["setup_score"] = raw_setup_score
    metadata["raw_confidence"] = max(0.0, min(1.0, float(signal.confidence or 0.0)))
    metadata["score_reasons"] = reason_list
    return StrategyVote(
        strategy=strategy_name,
        side=side if side in {'CE', 'PE', 'NO_TRADE'} else 'UNKNOWN',
        score=vote_score,
        confidence=max(0.0, min(1.0, float(signal.confidence or 0.0))),
        reasons=reason_list,
        metadata=metadata,
    )


@dataclass(slots=True)
class StrategyRegistry:
    """Maintain a registry of available strategy factories."""

    _factories: dict[str, StrategyFactory] = field(default_factory=dict)

    def register(self, name: str, factory: StrategyFactory) -> None:
        """Register *factory* under the provided *name*.

        Args:
            name: Human readable identifier for the strategy.
            factory: Callable returning a strategy instance.

        Returns:
            None.

        Raises:
            ValueError: Raised when *name* or *factory* is invalid.
        """

        log.debug(
            "Entered StrategyRegistry.register",
            extra={"event": "registry_register", "strategy_name": name},
        )
        try:
            if not name:
                raise ValueError("Strategy name must be provided")
            if factory is None:
                raise ValueError("Strategy factory must not be None")
            self._factories[name] = factory
            log.info(
                "Condition met: strategy_registered",
                extra={"event": "strategy_registered", "strategy": name},
            )
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error("Failure in StrategyRegistry.register: %s", exc, exc_info=exc)
            raise

    def create(self, name: str, **kwargs: t.Any) -> StrategyInterface:
        """Instantiate the strategy named *name* using *kwargs*.

        Args:
            name: Registered strategy identifier.
            **kwargs: Keyword arguments forwarded to the factory.

        Returns:
            StrategyInterface: Instantiated strategy complying with the contract.

        Raises:
            KeyError: Raised when no factory is registered for *name*.
            Exception: Propagated when the factory fails to instantiate.
        """

        log.debug(
            "Entered StrategyRegistry.create",
            extra={
                "event": "registry_create",
                "strategy_name": name,
                "kwargs": list(kwargs.keys()),
            },
        )
        try:
            factory = self._factories[name]
        except KeyError as exc:
            log.error("Failure in StrategyRegistry.create: %s", exc, exc_info=exc)
            raise
        try:
            instance = factory(**kwargs)
            if isinstance(instance, StrategyInterface):
                return instance
            return StrategyAdapter(instance)
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in StrategyRegistry.create factory: %s", exc, exc_info=exc
            )
            raise

    def get_factories(self) -> dict[str, StrategyFactory]:
        """Return a shallow copy of the registered factories.

        Args:
            None.

        Returns:
            dict[str, StrategyFactory]: Mapping of strategy names to factories.

        Raises:
            None.
        """

        log.debug("Entered StrategyRegistry.get_factories")
        try:
            return dict(self._factories)
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in StrategyRegistry.get_factories: %s", exc, exc_info=exc
            )
            raise

    def available_strategies(self) -> list[str]:
        """Return the list of registered strategy names.

        Args:
            None.

        Returns:
            list[str]: Sorted list of registered strategy identifiers.

        Raises:
            None.
        """

        log.debug("Entered StrategyRegistry.available_strategies")
        try:
            return sorted(self._factories.keys())
        except Exception as exc:  # noqa: BLE001 - defensive
            log.error(
                "Failure in StrategyRegistry.available_strategies: %s",
                exc,
                exc_info=exc,
            )
            raise


STRATEGY_REGISTRY = StrategyRegistry()


def register_strategy(name: str, factory: StrategyFactory) -> None:
    """Helper to register *factory* under *name* on the global registry.

    Args:
        name: Strategy identifier registered on the global registry.
        factory: Callable returning the strategy instance.

    Returns:
        None.

    Raises:
        Exception: Propagated when registration fails.
    """

    log.debug(
        "Entered register_strategy",
        extra={"event": "register_strategy", "strategy_name": name},
    )
    try:
        STRATEGY_REGISTRY.register(name, factory)
    except Exception as exc:  # noqa: BLE001 - defensive
        log.error("Failure in register_strategy: %s", exc, exc_info=exc)
        raise


def instantiate_strategy(name: str, **kwargs: t.Any) -> StrategyInterface:
    """Instantiate registered strategy *name* with *kwargs*.

    Args:
        name: Strategy identifier.
        **kwargs: Keyword arguments forwarded to the factory.

    Returns:
        StrategyInterface: Instantiated strategy complying with the interface.

    Raises:
        Exception: Propagated from the registry when instantiation fails.
    """

    log.debug(
        "Entered instantiate_strategy",
        extra={
            "event": "instantiate_strategy",
            "name": name,
            "kwargs": list(kwargs.keys()),
        },
    )
    try:
        return STRATEGY_REGISTRY.create(name, **kwargs)
    except Exception as exc:  # noqa: BLE001 - defensive
        log.error("Failure in instantiate_strategy: %s", exc, exc_info=exc)
        raise


@dataclass(slots=True)
class StrategyScoreWeights:
    """Weight configuration used when computing composite scores."""

    pnl: float = 0.55
    sharpe: float = 0.25
    win_rate: float = 0.1
    drawdown: float = 0.1

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return the weight quadruple for normalisation.

        Args:
            None.

        Returns:
            tuple[float, float, float, float]: Tuple containing the raw weights.

        Raises:
            None.
        """

        return self.pnl, self.sharpe, self.win_rate, self.drawdown

    def normalised(self) -> "StrategyScoreWeights":
        """Return a copy with weights normalised to sum to one.

        Args:
            None.

        Returns:
            StrategyScoreWeights: Normalised weights summing to one.

        Raises:
            None.
        """

        total = self.pnl + self.sharpe + self.win_rate + self.drawdown
        if total <= 0:
            return StrategyScoreWeights(1.0, 0.0, 0.0, 0.0)
        return StrategyScoreWeights(
            pnl=self.pnl / total,
            sharpe=self.sharpe / total,
            win_rate=self.win_rate / total,
            drawdown=self.drawdown / total,
        )


@dataclass(slots=True)
class StrategyPerformance:
    """Rolling performance snapshot for a strategy."""

    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    trade_returns: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    regime_buckets: dict[str, "RegimePerformanceBucket"] = field(default_factory=dict)

    def record(self, pnl: float, regime: str | None = None) -> None:
        """Update the performance snapshot with realised *pnl*.

        Args:
            pnl: Realised profit or loss for the completed trade.
            regime: Optional market regime associated with the trade.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            self.total_pnl += pnl
            if pnl >= 0:
                self.wins += 1
            else:
                self.losses += 1
            self.trade_returns.append(pnl)
            bucket_key = "unknown"
            if isinstance(regime, str) and regime.strip():
                bucket_key = regime.strip().lower()
            bucket = self.regime_buckets.setdefault(
                bucket_key, RegimePerformanceBucket()
            )
            bucket.record(pnl)
            self.last_updated = datetime.now(timezone.utc)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyPerformance.record: %s",
                exc,
                exc_info=exc,
            )

    def rolling_pnl(self) -> float:
        """Return rolling realised PnL over the tracked trade window.

        Args:
            None.

        Returns:
            float: Rolling profit or loss from recorded trades.

        Raises:
            None.
        """

        cumulative = 0.0
        try:
            cumulative = sum(self.trade_returns)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyPerformance.rolling_pnl: %s",
                exc,
                exc_info=exc,
            )
        return cumulative

    def snapshot(self) -> dict[str, float]:
        """Return aggregate statistics for the performance window.

        Args:
            None.

        Returns:
            dict[str, float]: Mapping of metric names to values.

        Raises:
            None.
        """

        try:
            trade_count = float(self.wins + self.losses)
            rolling_value = float(self.rolling_pnl())
            win_rate_value = float(self.win_rate())
            sharpe_value = float(self.sharpe_ratio())
            drawdown_value = float(self.max_drawdown())
            return {
                "pnl": float(self.total_pnl),
                "rolling_pnl": rolling_value,
                "win_rate": win_rate_value,
                "hit_rate": win_rate_value,
                "sharpe": sharpe_value,
                "drawdown": drawdown_value,
                "trades": trade_count,
                "wins": float(self.wins),
                "losses": float(self.losses),
            }
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyPerformance.snapshot: %s",
                exc,
                exc_info=exc,
            )
            return {
                "pnl": 0.0,
                "rolling_pnl": 0.0,
                "win_rate": 0.0,
                "hit_rate": 0.0,
                "sharpe": 0.0,
                "drawdown": 0.0,
                "trades": 0.0,
                "wins": 0.0,
                "losses": 0.0,
            }

    def win_rate(self) -> float:
        """Return historical win rate computed from wins/losses.

        Args:
            None.

        Returns:
            float: Historical win rate expressed as 0.0–1.0 fraction.

        Raises:
            None.
        """

        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return self.wins / total

    def win_loss_ratio(self) -> float:
        """Return win-to-loss ratio for recorded trades.

        Args:
            None.

        Returns:
            float: Ratio of wins to losses using defensive divisor.

        Raises:
            None.
        """

        try:
            if self.wins == 0 and self.losses == 0:
                return 0.0
            return self.wins / max(1, self.losses)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyPerformance.win_loss_ratio: %s",
                exc,
                exc_info=exc,
            )
            return 0.0

    def hit_rate(self) -> float:
        """Return alias for win rate aiding external score reporters.

        Args:
            None.

        Returns:
            float: Alias for ``win_rate`` reflecting trade hit ratio.

        Raises:
            None.
        """

        try:
            return self.win_rate()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyPerformance.hit_rate: %s",
                exc,
                exc_info=exc,
            )
            return 0.0

    def sharpe_ratio(self) -> float:
        """Return a simple Sharpe ratio approximation for the strategy.

        Args:
            None.

        Returns:
            float: Sharpe-like score derived from recorded returns.

        Raises:
            None.
        """

        samples = list(self.trade_returns)
        if len(samples) < 2:
            return 0.0
        avg = mean(samples)
        std_dev = pstdev(samples)
        if std_dev == 0:
            return 0.0
        return (avg / std_dev) * sqrt(len(samples))

    def max_drawdown(self) -> float:
        """Return maximum drawdown computed from rolling equity curve.

        Args:
            None.

        Returns:
            float: Maximum drawdown magnitude observed in the window.

        Raises:
            None.
        """

        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        try:
            for pnl in self.trade_returns:
                equity += pnl
                if equity > peak:
                    peak = equity
                drawdown = peak - equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyPerformance.max_drawdown: %s",
                exc,
                exc_info=exc,
            )
            return 0.0
        return max_drawdown


@dataclass(slots=True)
class RegimePerformanceBucket:
    """Rolling performance snapshot limited to a specific regime."""

    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    trades: int = 0
    trade_returns: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record(self, pnl: float) -> None:
        """Record a trade outcome for the regime bucket.

        Args:
            pnl: Realised profit or loss attributed to the regime.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            self.total_pnl += pnl
            self.trades += 1
            if pnl >= 0:
                self.wins += 1
            else:
                self.losses += 1
            self.trade_returns.append(pnl)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in RegimePerformanceBucket.record: %s",
                exc,
                exc_info=exc,
            )

    def snapshot(self) -> dict[str, float]:
        """Return aggregate statistics for the regime bucket.

        Args:
            None.

        Returns:
            dict[str, float]: Summary containing pnl, win rate, Sharpe, and drawdown.

        Raises:
            None.
        """

        try:
            returns = list(self.trade_returns)
            win_rate = self.wins / max(1, self.trades)
            sharpe = 0.0
            if len(returns) >= 2:
                std_dev = pstdev(returns)
                if std_dev > 0:
                    sharpe = (mean(returns) / std_dev) * sqrt(len(returns))
            drawdown = 0.0
            equity = 0.0
            peak = 0.0
            for value in returns:
                equity += value
                peak = max(peak, equity)
                drawdown = max(drawdown, peak - equity)
            return {
                "pnl": float(self.total_pnl),
                "win_rate": float(win_rate),
                "hit_rate": float(win_rate),
                "sharpe": float(sharpe),
                "drawdown": float(drawdown),
                "trades": float(self.trades),
            }
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in RegimePerformanceBucket.snapshot: %s",
                exc,
                exc_info=exc,
            )
            return {
                "pnl": 0.0,
                "win_rate": 0.0,
                "hit_rate": 0.0,
                "sharpe": 0.0,
                "drawdown": 0.0,
                "trades": float(self.trades),
            }


def _regime_breakdown(performance: StrategyPerformance) -> dict[str, dict[str, float]]:
    """Return per-regime statistics for *performance*.

    Args:
        performance: Strategy performance tracker containing buckets.

    Returns:
        dict[str, dict[str, float]]: Mapping of regime name to snapshot metrics.

    Raises:
        None.
    """

    log.debug(
        "Entered _regime_breakdown",
        extra={"event": "strategy_regime_breakdown"},
    )
    summary: dict[str, dict[str, float]] = {}
    try:
        for regime, bucket in performance.regime_buckets.items():
            summary[regime] = bucket.snapshot()
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Failure in _regime_breakdown: %s",
            exc,
            exc_info=exc,
        )
    return summary


@dataclass(slots=True)
class RegimeState:
    """Track current market regime information."""

    regime: str | None = None
    confidence: float = 0.0
    updated_at: datetime | None = None


@dataclass(slots=True)
class StrategyScore:
    """Composite score snapshot for a single strategy."""

    strategy: str
    weight: float
    allocation: float
    score: float
    regime_score: float
    pnl: float
    sharpe: float
    win_rate: float
    drawdown: float
    rolling_pnl: float
    win_loss_ratio: float
    manual_allocation: float | None
    regime_bias: float
    enabled: bool
    active_regime_stats: dict[str, t.Any]
    regime_breakdown: dict[str, dict[str, float]]
    dynamic_disabled: bool


class StrategyManager(_BaseStrategyManager):
    """Augment the base manager with performance scoring and allocations."""

    def __init__(
        self,
        strategies: list[t.Any],
        indicator_engine: t.Any,
        position_manager: t.Any,
        min_confidence: float = 0.35,
        data_hub: t.Any | None = None,
        orchestrator: t.Any | None = None,
        futures_symbol: str | None = None,
        *,
        score_weights: StrategyScoreWeights | None = None,
        regime_signal_getter: (
            t.Callable[[], t.Mapping[str, t.Any] | None] | None
        ) = None,
        regime_bias_map: t.Mapping[str, t.Mapping[str, float]] | None = None,
        market_regime_manager: MarketRegimeManager | None = None,
    ) -> None:
        """Initialise strategy manager with scoring configuration.

        Args:
            strategies: Strategy instances producing signals.
            indicator_engine: Indicator engine shared across strategies.
            position_manager: Position manager used for exposure checks.
            min_confidence: Minimum confidence threshold for signals.
            data_hub: Optional data hub providing futures context.
            orchestrator: Optional orchestrator enforcing allocations.
            futures_symbol: Futures symbol used for futures metrics.
            score_weights: Optional override for score weighting.
            regime_signal_getter: Callable returning regime snapshots.
            regime_bias_map: Optional per-regime weighting overrides.
            market_regime_manager: Optional central regime manager used for
                gating decisions.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug("Entered StrategyManager.__init__")
        super().__init__(
            strategies,
            indicator_engine,
            position_manager,
            min_confidence,
            data_hub,
            orchestrator,
            futures_symbol,
        )
        self._score_weights = (
            score_weights.normalised()
            if score_weights
            else StrategyScoreWeights().normalised()
        )
        self._regime_signal_getter = regime_signal_getter
        self._regime_bias_map = {
            regime.lower(): {k: float(v) for k, v in mapping.items()}
            for regime, mapping in (regime_bias_map or {}).items()
        }
        self._regime_manager = market_regime_manager
        self._performance: dict[str, StrategyPerformance] = {}
        self._manual_allocations: dict[str, float] = {}
        self._disabled_strategies: set[str] = set()
        self._dynamic_disabled: set[str] = set()
        self._regime_state = RegimeState()
        self._score_cache: dict[str, StrategyScore] = {}
        self._score_floor = 0.05
        self._score_ceiling = 3.0
        self._allocation_state: dict[str, float] = {}
        self._regime_last_key: str | None = None
        self._dynamic_disable_threshold = 0.2
        self._dynamic_enable_threshold = 0.35
        self._dynamic_trade_threshold = 5
        self._dynamic_confidence_floor = 0.45
        self._last_regime_gate: tuple[bool, tuple[str, ...], str | None] | None = None
        self._last_regime_gate_at: float = 0.0
        self._regime_gate_cooldown = 10.0
        self._use_regime_adaptive = bool(app_settings.USE_REGIME_ADAPTIVE)
        self._observability_counters: dict[str, int] = {
            "signals_generated": 0,
            "signals_blocked_by_regime": 0,
            "signals_blocked_by_risk": 0,
            "orders_submitted": 0,
        }
        self._last_metrics_log_ts = time.time()
        self._adaptive_store = AdaptiveParameterStore(
            window_trades=app_settings.ADAPTIVE_WINDOW_TRADES
        )
        self._optimizer = WalkForwardOptimizer(
            recalibrate_every=app_settings.ADAPTIVE_RECALIBRATE_EVERY
        )
        self._regime_fallback_scale = float(app_settings.REGIME_FALLBACK_SCALE)
        self._avg_confidence_window: deque[float] = deque(maxlen=500)
        self._avg_kelly_window: deque[float] = deque(maxlen=500)
        self._market_open_since_ts: float | None = None
        self._last_zero_signal_check_ts = 0.0
        self._no_signal_summary: dict[str, int] = {
            "total": 0,
            "missing_indicators": 0,
            "volume_zero": 0,
            "avg_volume_zero": 0,
            "error_strategies": 0,
        }
        self._no_signal_missing: dict[str, int] = {}
        self._no_signal_last_summary_ts = time.time()
        self._symbol_invalid_counts: dict[str, int] = {}
        self._symbol_temporarily_ineligible: dict[str, str] = {}
        self._symbol_invalid_threshold = 10  # ✅ FIX #2a: Raised from 3→10; options have legitimate data gaps at open/reconnect
        self._required_indicators: set[str] = {"volume", "avg_volume", "minutes_since_open", "minutes_until_close"}
        self._strategy_required_indicators: dict[str, set[str]] = {
            getattr(strategy, "name", strategy.__class__.__name__): set(strategy.get_required_indicators())
            for strategy in strategies
        }
        allow_scalp_single = str(os.getenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", "false")).lower() in {"1", "true", "yes", "on"}
        vwap_min_score, vwap_min_conf = self._single_vote_thresholds("VWAPPro")
        generic_min_score, generic_min_conf = self._single_vote_thresholds("generic")
        log.info(
            "STRATEGY_SINGLE_VOTE_CONFIG allow_scalp_single=%s vwap_min_score=%s vwap_min_conf=%s generic_min_score=%s generic_min_conf=%s",
            allow_scalp_single,
            vwap_min_score,
            vwap_min_conf,
            generic_min_score,
            generic_min_conf,
        )

    def record_trade_result(
        self,
        strategy_name: str,
        pnl: float,
        *,
        metadata: t.Mapping[str, t.Any] | None = None,
    ) -> None:
        """Record realised *pnl* for *strategy_name*.

        Args:
            strategy_name: Strategy identifier whose metrics update.
            pnl: Realised profit or loss for the trade.
            metadata: Optional metadata associated with the trade.

        Returns:
            None.

        Raises:
            None.
        """

        log.debug(
            "Entered StrategyManager.record_trade_result",
            extra={"event": "strategy_record_trade", "strategy": strategy_name},
        )
        perf = self._performance.setdefault(strategy_name, StrategyPerformance())
        regime_label: str | None = None
        if metadata is not None:
            try:
                candidate = metadata.get("regime")
                if isinstance(candidate, str):
                    regime_label = candidate
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in StrategyManager.record_trade_result metadata parse: %s",
                    exc,
                    exc_info=exc,
                )
        if regime_label is None:
            regime_label = self._regime_state.regime
        perf.record(pnl, regime=regime_label)
        if metadata:
            log.info(
                "Condition met: strategy_trade_recorded",
                extra={
                    "event": "strategy_trade_recorded",
                    "strategy": strategy_name,
                    "pnl": pnl,
                    "metadata": dict(metadata),
                },
            )
        self._score_cache.pop(strategy_name, None)

    def get_strategy_scores(self, limit: int | None = None) -> list[StrategyScore]:
        """Return ordered strategy scores for observability.

        Args:
            limit: Optional limit on the number of scores returned.

        Returns:
            list[StrategyScore]: Strategy scores sorted by composite score.

        Raises:
            None.
        """

        log.debug("Entered StrategyManager.get_strategy_scores")
        scores = list(self._recompute_scores().values())
        scores.sort(key=lambda entry: entry.score, reverse=True)
        if limit is not None:
            return scores[: max(limit, 0)]
        return scores

    def get_performance_snapshot(self) -> dict[str, dict[str, t.Any]]:
        """Return rolling performance metrics per strategy.

        Args:
            None.

        Returns:
            dict[str, dict[str, t.Any]]: Snapshot of aggregate and regime metrics.

        Raises:
            None.
        """

        log.debug("Entered StrategyManager.get_performance_snapshot")
        snapshot: dict[str, dict[str, t.Any]] = {}
        try:
            for name, performance in self._performance.items():
                snapshot[name] = {
                    "aggregate": performance.snapshot(),
                    "regimes": _regime_breakdown(performance),
                }
            return snapshot
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager.get_performance_snapshot: %s",
                exc,
                exc_info=exc,
            )
            return snapshot

    def get_top_strategies(self, limit: int = 3) -> list[str]:
        """Return the names of the highest scoring strategies.

        Args:
            limit: Maximum number of strategy names to return.

        Returns:
            list[str]: Ordered list of strategy names by score.

        Raises:
            None.
        """

        log.debug("Entered StrategyManager.get_top_strategies")
        limit = max(limit, 0)
        ordered = self.get_strategy_scores(limit)
        return [entry.strategy for entry in ordered]

    def select_strategy_for_activation(
        self, *, weight_floor: float = 0.1
    ) -> StrategyScore | None:
        """Return the best candidate strategy above ``weight_floor``.

        Args:
            weight_floor: Minimum allocation weight required to qualify.

        Returns:
            StrategyScore | None: Selected strategy score snapshot when
            available.

        Raises:
            None.
        """

        log.debug("Entered StrategyManager.select_strategy_for_activation")
        candidates = self.get_strategy_scores()
        for entry in candidates:
            if entry.allocation >= weight_floor:
                log.info(
                    "Condition met: strategy_selected",
                    extra={
                        "event": "strategy_selected",
                        "strategy": entry.strategy,
                        "weight": entry.allocation,
                    },
                )
                return entry
        return None

    def get_allocation_snapshot(self) -> dict[str, float]:
        """Return capital allocation fractions derived from scores.

        Args:
            None.

        Returns:
            dict[str, float]: Mapping of strategy names to allocation
            fractions summing to one when possible.

        Raises:
            None.
        """

        log.debug("Entered StrategyManager.get_allocation_snapshot")
        scores = self._recompute_scores()
        active_scores = {name: entry for name, entry in scores.items() if entry.enabled}
        if not active_scores:
            return {}
        total = sum(max(entry.allocation, 0.0) for entry in active_scores.values())
        if total <= 0:
            even = 1.0 / len(active_scores)
            return {name: even for name in active_scores}
        return {
            name: max(entry.allocation, 0.0) / total
            for name, entry in active_scores.items()
        }

    def set_manual_allocation(self, strategy_name: str, fraction: float | None) -> None:
        """Set manual capital *fraction* override for *strategy_name*.

        Args:
            strategy_name: Strategy whose allocation override changes.
            fraction: Desired capital fraction or ``None`` to clear.

        Returns:
            None.

        Raises:
            ValueError: If ``fraction`` is negative.
        """

        log.debug(
            "Entered StrategyManager.set_manual_allocation",
            extra={
                "event": "strategy_manual_alloc",
                "strategy": strategy_name,
                "fraction": fraction,
            },
        )
        if fraction is None:
            self._manual_allocations.pop(strategy_name, None)
            self._score_cache.pop(strategy_name, None)
            return
        if fraction < 0:
            raise ValueError("Manual allocation fraction must be non-negative")
        self._manual_allocations[strategy_name] = fraction
        self._score_cache.pop(strategy_name, None)

    def configure_score_thresholds(
        self,
        *,
        disable: float | None = None,
        enable: float | None = None,
        min_trades: int | None = None,
        confidence_floor: float | None = None,
    ) -> None:
        """Configure dynamic score thresholds used for auto toggles.

        Args:
            disable: Optional disable threshold override.
            enable: Optional enable threshold override.
            min_trades: Minimum trades required before toggles activate.
            confidence_floor: Minimum regime confidence before auto actions.

        Returns:
            None.

        Raises:
            ValueError: If supplied thresholds are invalid.
        """

        log.debug(
            "Entered StrategyManager.configure_score_thresholds",
            extra={
                "event": "strategy_configure_thresholds",
                "disable": disable,
                "enable": enable,
                "min_trades": min_trades,
                "confidence_floor": confidence_floor,
            },
        )
        try:
            if disable is not None:
                if disable < 0.0:
                    raise ValueError("Disable threshold must be non-negative")
                self._dynamic_disable_threshold = float(disable)
            if enable is not None:
                if enable < 0.0:
                    raise ValueError("Enable threshold must be non-negative")
                self._dynamic_enable_threshold = float(enable)
            if (
                disable is not None
                and enable is not None
                and self._dynamic_enable_threshold <= self._dynamic_disable_threshold
            ):
                raise ValueError("Enable threshold must exceed disable threshold")
            if min_trades is not None:
                if min_trades < 0:
                    raise ValueError("Minimum trades must be non-negative")
                self._dynamic_trade_threshold = int(min_trades)
            if confidence_floor is not None:
                if not 0.0 <= confidence_floor <= 1.0:
                    raise ValueError("Confidence floor must be between 0 and 1")
                self._dynamic_confidence_floor = float(confidence_floor)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager.configure_score_thresholds: %s",
                exc,
                exc_info=exc,
            )
            raise

    def get_score_thresholds(self) -> dict[str, float | int]:
        """Return the currently configured score thresholds.

        Args:
            None.

        Returns:
            dict[str, float | int]: Mapping of threshold names to values.

        Raises:
            None.
        """

        log.debug("Entered StrategyManager.get_score_thresholds")
        try:
            return {
                "disable": float(self._dynamic_disable_threshold),
                "enable": float(self._dynamic_enable_threshold),
                "min_trades": int(self._dynamic_trade_threshold),
                "confidence_floor": float(self._dynamic_confidence_floor),
            }
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager.get_score_thresholds: %s",
                exc,
                exc_info=exc,
            )
            return {
                "disable": 0.0,
                "enable": 0.0,
                "min_trades": 0,
                "confidence_floor": 0.0,
            }

    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable strategy execution for *strategy_name*.

        Args:
            strategy_name: Strategy identifier to disable.

        Returns:
            bool: ``True`` when the strategy transitioned to disabled.

        Raises:
            None.
        """

        log.debug(
            "Entered StrategyManager.disable_strategy",
            extra={
                "event": "strategy_disable_request",
                "strategy": strategy_name,
            },
        )
        try:
            resolved = str(strategy_name)
            registered = {strategy.name for strategy in self._strategies}
            if resolved not in registered:
                return False
            if resolved in self._disabled_strategies:
                return False
            self._disabled_strategies.add(resolved)
            self._dynamic_disabled.discard(resolved)
            self._score_cache.pop(resolved, None)
            log.info(
                "Condition met: strategy_disabled",
                extra={
                    "event": "strategy_disabled",
                    "strategy": resolved,
                },
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager.disable_strategy: %s",
                exc,
                exc_info=exc,
            )
            return False

    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable strategy execution for *strategy_name*.

        Args:
            strategy_name: Strategy identifier to enable.

        Returns:
            bool: ``True`` when the strategy transitioned to enabled.

        Raises:
            None.
        """

        log.debug(
            "Entered StrategyManager.enable_strategy",
            extra={
                "event": "strategy_enable_request",
                "strategy": strategy_name,
            },
        )
        try:
            resolved = str(strategy_name)
            registered = {strategy.name for strategy in self._strategies}
            if resolved not in registered:
                return False
            if resolved not in self._disabled_strategies:
                return False
            self._disabled_strategies.discard(resolved)
            self._dynamic_disabled.discard(resolved)
            self._score_cache.pop(resolved, None)
            log.info(
                "Condition met: strategy_enabled",
                extra={
                    "event": "strategy_enabled",
                    "strategy": resolved,
                },
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager.enable_strategy: %s",
                exc,
                exc_info=exc,
            )
            return False

    def disabled_strategies(self) -> tuple[str, ...]:
        """Return tuple of strategies currently disabled.

        Args:
            None.

        Returns:
            tuple[str, ...]: Sorted strategy names that are disabled.

        Raises:
            None.
        """

        try:
            return tuple(sorted(self._disabled_strategies))
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager.disabled_strategies: %s",
                exc,
                exc_info=exc,
            )
            return tuple()

    def get_elite_strategy_stats(self) -> list[dict[str, t.Any]]:
        """Return diagnostic stats for elite strategies.

        Args:
            None.

        Returns:
            list[dict[str, t.Any]]: Snapshot dictionaries for elite strategies.

        Raises:
            None.
        """

        log.debug(
            "Entered StrategyManager.get_elite_strategy_stats",
            extra={"event": "elite_stats_collect"},
        )
        stats: list[dict[str, t.Any]] = []
        for strategy in self._strategies:
            if not isinstance(strategy, EliteStrategy):
                continue
            getter = getattr(strategy, "get_stats", None)
            if not callable(getter):
                continue
            try:
                snapshot = getter()
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure collecting elite stats for %s: %s",
                    strategy.name,
                    exc,
                    exc_info=exc,
                    extra={
                        "event": "elite_stats_error",
                        "strategy": strategy.name,
                    },
                )
                continue
            if isinstance(snapshot, dict):
                stats.append(snapshot)
        log.info(
            "Condition met: elite_stats_collected",
            extra={"event": "elite_stats_collected", "count": len(stats)},
        )
        return stats

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Return ``True`` when *strategy_name* is currently enabled.

        Args:
            strategy_name: Strategy identifier to inspect.

        Returns:
            bool: ``True`` if enabled else ``False``.

        Raises:
            None.
        """

        try:
            resolved = str(strategy_name)
            return (
                resolved not in self._disabled_strategies
                and resolved not in self._dynamic_disabled
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager.is_strategy_enabled: %s",
                exc,
                exc_info=exc,
            )
            return False

    def refresh_regime_state(self) -> RegimeState:
        """Refresh and return cached regime information.

        Args:
            None.

        Returns:
            RegimeState: Updated regime snapshot containing metadata.

        Raises:
            None.
        """

        log.debug("Entered StrategyManager.refresh_regime_state")
        getter = self._regime_signal_getter
        if getter is None:
            return self._regime_state
        try:
            snapshot = getter()
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager.refresh_regime_state: %s",
                exc,
                exc_info=exc,
            )
            return self._regime_state

        if snapshot is None:
            return self._regime_state

        # [FIX] Robust Normalization: Handle Dict, Object, or String inputs
        regime_raw = None
        confidence_raw = 0.0

        # Case 1: It's a Dictionary (e.g. from JSON/API)
        if isinstance(snapshot, dict):
            regime_raw = snapshot.get("regime")
            confidence_raw = snapshot.get("confidence", 0.0)

        # Case 2: It's a Raw String (e.g. "TRENDING")
        elif isinstance(snapshot, str):
            regime_raw = snapshot
            confidence_raw = 0.0  # Strings carry no confidence data

        # Case 3: It's an Object (RegimeSnapshot)
        elif hasattr(snapshot, "regime"):
            regime_raw = snapshot.regime
            confidence_raw = getattr(snapshot, "confidence", 0.0)

        # Case 4: Fallback
        else:
            regime_raw = str(snapshot)

        # Handle Enum objects if present
        if hasattr(regime_raw, "value"):
            regime_raw = regime_raw.value

        # Final Cleaning
        regime = str(regime_raw or "").strip().lower()
        try:
            confidence = float(confidence_raw or 0.0)
        except (ValueError, TypeError):
            confidence = 0.0

        self._regime_state = RegimeState(
            regime=regime or None,
            confidence=max(0.0, min(confidence, 1.0)),
            updated_at=datetime.now(timezone.utc),
        )

        # Logging & Metrics (Kept from your original code)
        payload = {
            "event": "regime_state_refreshed",
            "regime": self._regime_state.regime,
            "confidence": self._regime_state.confidence,
        }
        change_payload = dict(payload)
        emitted_change = log_state_change(
            log,
            key="strategy_manager.regime_state",
            value=(self._regime_state.regime, self._regime_state.confidence),
            msg="Condition met: regime_state_refreshed",
            extra=change_payload,
        )
        if not emitted_change:
            log_throttled(
                log,
                key="strategy_manager.regime_state_refreshed",
                msg="Condition met: regime_state_refreshed",
                interval_sec=10.0,
                extra=dict(payload),
            )
        try:
            METRICS.update_regime_confidence(
                regime=self._regime_state.regime,
                confidence=self._regime_state.confidence,
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager regime metrics: %s",
                exc,
                exc_info=exc,
                extra={"event": "strategy_regime_metric_error"},
            )
        return self._regime_state

    def _record_no_signal_summary(
        self,
        *,
        symbol: str,
        missing: list[str],
        indicators: t.Mapping[str, t.Any],
        error_strategies: list[str],
    ) -> None:
        """Args: symbol, missing, indicators, error_strategies. Returns: None. Raises: Exception."""
        log.debug(
            "Entered StrategyManager._record_no_signal_summary",
            extra={"event": "strategy_no_signal_summary_enter", "symbol": symbol},
        )
        try:
            self._no_signal_summary["total"] += 1
            if missing:
                self._no_signal_summary["missing_indicators"] += len(missing)
                for name in missing:
                    self._no_signal_missing[name] = (
                        self._no_signal_missing.get(name, 0) + 1
                    )
            volume = indicators.get("volume")
            avg_volume = indicators.get("avg_volume")
            try:
                if volume is None or float(volume) <= 0.0:
                    self._no_signal_summary["volume_zero"] += 1
            except (TypeError, ValueError) as exc:
                log.debug(
                    "Failure in volume coercion: %s",
                    exc,
                    extra={
                        "event": "strategy_no_signal_volume_error",
                        "symbol": symbol,
                    },
                )
                self._no_signal_summary["volume_zero"] += 1
            try:
                if avg_volume is None or float(avg_volume) <= 0.0:
                    self._no_signal_summary["avg_volume_zero"] += 1
            except (TypeError, ValueError) as exc:
                log.debug(
                    "Failure in avg_volume coercion: %s",
                    exc,
                    extra={
                        "event": "strategy_no_signal_avg_volume_error",
                        "symbol": symbol,
                    },
                )
                self._no_signal_summary["avg_volume_zero"] += 1
            if error_strategies:
                self._no_signal_summary["error_strategies"] += len(error_strategies)
            now_ts = time.time()
            if now_ts - self._no_signal_last_summary_ts >= 60.0:
                top_missing = sorted(
                    self._no_signal_missing.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
                log.debug(
                    "Condition met: strategy_manager_no_signal_summary",
                    extra={
                        "event": "strategy_manager_no_signal_summary",
                        "symbol": symbol,
                        "total": self._no_signal_summary["total"],
                        "missing_indicators": self._no_signal_summary[
                            "missing_indicators"
                        ],
                        "volume_zero": self._no_signal_summary["volume_zero"],
                        "avg_volume_zero": self._no_signal_summary["avg_volume_zero"],
                        "error_strategies": self._no_signal_summary["error_strategies"],
                        "top_missing": top_missing,
                    },
                )
                self._no_signal_last_summary_ts = now_ts
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager no-signal summary: %s",
                exc,
                exc_info=exc,
                extra={"event": "strategy_no_signal_summary_error", "symbol": symbol},
            )


    def _update_context_snapshot(
        self,
        *,
        symbol: str,
        indicators: t.Mapping[str, t.Any],
        role: str,
    ) -> None:
        """Store latest spot/futures context snapshots for option strategies."""
        if not hasattr(self, "_latest_context_snapshots"):
            self._latest_context_snapshots: dict[str, dict[str, t.Any]] = {}

        def _num(*keys: str) -> float | None:
            for key in keys:
                value = indicators.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        context_kind = (
            "price_direction"
            if role == "spot_context"
            else "volume_flow"
            if role == "futures_context"
            else "unknown"
        )
        snapshot = {
            "symbol": symbol, "role": role, "context_kind": context_kind, "timestamp": time.time(),
            "ltp": _num("ltp", "close", "price"), "close": _num("close", "ltp", "price"),
            "vwap": _num("vwap", "exchange_vwap", "session_vwap"),
            "ema_fast": _num("ema_fast", "ema_9", "ema9"),
            "ema_slow": _num("ema_slow", "ema_21", "ema21"), "ema_50": _num("ema_50", "ema50"),
            "adx": _num("adx"), "atr": _num("atr"), "volume": _num("volume"),
            "avg_volume": _num("avg_volume"), "futures_volume_ratio": _num("futures_volume_ratio"),
            "vwap_slope": _num("vwap_slope"), "ema_slope": _num("ema_slope"),
            "direction_bias": indicators.get("direction_bias") or indicators.get("underlying_direction_bias"),
            "regime": indicators.get("regime") or indicators.get("market_regime"),
        }
        self._latest_context_snapshots[role] = snapshot

    def generate_signal(
        self,
        symbol: str,
        current_price: float,
        *,
        trace_id: str | None = None,
    ) -> Signal | None:
        """Generate signal with confidence adjusted by strategy scores.

        Args:
            symbol: Symbol evaluated for trading opportunities.
            current_price: Latest trade price for the symbol.

        Returns:
            Signal | None: Weighted trading signal or ``None`` when absent.

        Raises:
            None.
        """

        symbol = normalize_symbol(symbol)
        log.debug(
            "Entered StrategyManager.generate_signal",
            extra={"event": "scored_strategy_generate", "symbol": symbol},
        )
        log.debug(
            "strategy_evaluation_start",
            extra={"event": "strategy_evaluation_start", "symbol": symbol},
        )
        no_signal_reasons: list[str] = []
        error_strategies: list[str] = []
        disabled_strategies_snapshot: list[str] = []
        signal_action: str | None = None
        signal_score: float | None = None
        signal_confidence: float | None = None
        exit_result = "no_signal"
        required_bars = int(getattr(self, "_required_candles", 20) or 20)
        bars_available = len(self._indicator_engine.get_history(symbol) or [])
        indicators_ready = bars_available >= required_bars
        hub_ready = None
        if self._data_hub is not None:
            hub_ready = bool(getattr(self._data_hub, "indicators_ready", False))
        indicators_raw = self._indicator_engine.get_indicators(
            symbol, self._required_indicators
        )
        indicators: dict[str, t.Any] = dict(indicators_raw)
        vwap = indicators.get("vwap")
        avg_volume = indicators.get("avg_volume")
        required_missing = sorted(
            name for name in self._required_indicators if indicators.get(name) is None
        )
        log.debug(
            "STRATEGY_MANAGER_ENTER",
            extra={
                "event": "STRATEGY_MANAGER_ENTER",
                "symbol": symbol,
                "trace_id": trace_id,
                "indicators_ready": indicators_ready,
                "bars_available": bars_available,
                "required_bars": required_bars,
                "hub_indicators_ready": hub_ready,
                "vwap": vwap,
                "avg_volume": avg_volume,
                "required_indicators_missing": required_missing,
                "active_strategies_count": len(self._strategies),
            },
        )
        if not indicators_ready:
            no_signal_reasons.append("global_indicators_not_ready_diagnostic_only")
            log_throttled(
                log,
                key=f"indicators_not_ready_diagnostic:{symbol}",
                msg="indicators_not_ready_diagnostic",
                interval_sec=120.0,
                level=10,
                extra={
                    "event": "indicators_not_ready_diagnostic",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "bars": bars_available,
                    "required": required_bars,
                },
            )

        def _emit_strategy_exit() -> None:
            """Emit terminal strategy manager summary. Args: none. Returns: none. Raises: none."""
            log.debug(
                "STRATEGY_MANAGER_EXIT",
                extra={
                    "event": "STRATEGY_MANAGER_EXIT",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "result": exit_result,
                    "signal_action": signal_action,
                    "signal_score": signal_score,
                    "signal_confidence": signal_confidence,
                    "no_signal_reasons": no_signal_reasons,
                    "disabled_strategies": disabled_strategies_snapshot,
                    "error_strategies": error_strategies,
                },
            )

        def _log_reject(
            reason_code: str, context: dict[str, t.Any] | None = None
        ) -> None:
            """Args: reason_code, context. Returns: None. Raises: Exception."""
            try:
                payload = {
                    "event": "strategy_no_signal_reject",
                    "symbol": symbol,
                    "reason_code": reason_code,
                }
                if context:
                    payload.update(context)
                log.debug(
                    "VWAP_PRO_REJECT | reason=%s",
                    reason_code,
                    extra=payload,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in StrategyManager._log_reject: %s",
                    exc,
                    exc_info=exc,
                    extra={
                        "event": "strategy_no_signal_reject_error",
                        "symbol": symbol,
                        "reason_code": reason_code,
                    },
                )

        def _emit_no_signal(
            reason_code: str, context: dict[str, t.Any] | None = None
        ) -> None:
            """Args: reason_code, context. Returns: None. Raises: Exception."""
            try:
                payload = {
                    "event": "strategy_no_signal",
                    "symbol": symbol,
                    "reason_code": reason_code,
                }
                if context:
                    payload.update(context)
                throttle_key = f"strategy_no_signal_{symbol}_{reason_code}"
                log_throttled(
                    log,
                    throttle_key,
                    f"📉 NO SIGNAL | symbol={symbol} reason={reason_code}",
                    level=10,
                    interval_sec=60.0,
                    extra=payload,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in StrategyManager._emit_no_signal: %s",
                    exc,
                    exc_info=exc,
                    extra={
                        "event": "strategy_no_signal_emit_error",
                        "symbol": symbol,
                        "reason_code": reason_code,
                    },
                )

        regime_manager = self._regime_manager
        regime_snapshot: RegimeSnapshot | None = None
        adjustments: dict[str, t.Any] = {}
        regime_scale = 1.0
        if regime_manager is not None:
            gate_context = {"component": "strategy_manager", "symbol": symbol}
            try:
                allowed = regime_manager.can_trade(context=gate_context)
                reasons = tuple(regime_manager.get_filter_reasons())
                regime_snapshot = regime_manager.get_latest_snapshot()
                self._log_regime_gate_decision(
                    symbol=symbol,
                    allowed=allowed,
                    reasons=reasons,
                    snapshot=regime_snapshot,
                )
                if not allowed:
                    self._observability_counters["signals_blocked_by_regime"] += 1
                    log_throttled(
                        log,
                        key=f"regime_scale_fallback:{symbol}",
                        msg="strategy_regime_scale_fallback",
                        interval_sec=30.0,
                        extra={
                            "event": "strategy_regime_scale_fallback",
                            "symbol": symbol,
                            "gate_reasons": list(reasons),
                            "scale": 1.0,
                        },
                    )
                if not allowed and not self._use_regime_adaptive:
                    _log_reject(
                        "regime_gate_block",
                        {"gate_reasons": reasons, "gate": "regime_manager"},
                    )
                    _emit_no_signal("regime_gate_block", {"gate_reasons": reasons})
                    no_signal_reasons.append("regime_gate_block")
                    _emit_strategy_exit()
                    return None
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in StrategyManager regime gate: %s",
                    exc,
                    exc_info=exc,
                    extra={"event": "strategy_regime_gate_error", "symbol": symbol},
                )
        if regime_snapshot is not None and isinstance(
            regime_snapshot.adjustments, t.Mapping
        ):
            adjustments = dict(regime_snapshot.adjustments)
        regime_scale = self._extract_regime_scale(adjustments)
        if regime_manager is not None:
            try:
                if (
                    not regime_manager.can_trade(
                        context={
                            "component": "strategy_manager_fallback",
                            "symbol": symbol,
                        }
                    )
                    and self._use_regime_adaptive
                ):
                    regime_scale = min(
                        regime_scale, max(self._regime_fallback_scale, 0.0)
                    )
            except Exception as exc:
                log.error(
                    "Failure in StrategyManager.generate_signal regime fallback: %s",
                    exc,
                    exc_info=exc,
                )
        regime_name = (
            regime_snapshot.regime
            if isinstance(regime_snapshot, RegimeSnapshot)
            else None
        )
        log.debug(
            "strategy_regime_scaling",
            extra={
                "event": "strategy_regime_scaling",
                "symbol": symbol,
                "regime": regime_name,
                "scale": regime_scale,
                "use_regime_adaptive": self._use_regime_adaptive,
            },
        )
        score_map = self._recompute_scores()
        indicators = dict(indicators)
        symbol_role = classify_symbol_role(symbol)
        indicators["symbol_role"] = symbol_role
        indicators["_regime_adjustments"] = adjustments
        self._augment_futures_metrics(indicators)
        # ✅ FIX S3: Inject exchange VWAP for options
        if self._data_hub is not None:
            try:
                opt_quote = self._data_hub.get_quote(symbol)
                if opt_quote:
                    _exch_vwap = self._extract_float(
                        opt_quote, ("vwap", "average_price")
                    )
                    if _exch_vwap and _exch_vwap > 0:
                        indicators["exchange_vwap"] = _exch_vwap
                        if not indicators.get("vwap") or indicators["vwap"] is None:
                            indicators["vwap"] = _exch_vwap
                    # Inject bid/ask spread so strategies can gate on execution cost.
                    # Options can have spreads of 5-30%+ of premium — entering without
                    # checking spread means the trade is already deeply underwater at fill.
                    _bid = self._extract_float(
                        opt_quote, ("bid", "best_bid", "buy_price")
                    )
                    _ask = self._extract_float(
                        opt_quote, ("ask", "best_ask", "sell_price")
                    )
                    if _bid and _bid > 0:
                        indicators["bid"] = _bid
                    if _ask and _ask > 0:
                        indicators["ask"] = _ask
                    if _bid and _ask and _bid > 0 and _ask > _bid:
                        _mid = (_bid + _ask) / 2.0
                        indicators["spread_pct"] = float((_ask - _bid) / _mid * 100.0)
            except Exception as e:
                log.warning(
                    "quote_enrichment_failed",
                    extra={
                        "event": "quote_enrichment_failed",
                        "symbol": symbol,
                        "error": str(e),
                    },
                    exc_info=e,
                )
        invalid_reason: str | None = None
        vwap = indicators.get("vwap") or indicators.get("exchange_vwap")
        volume = indicators.get("volume")
        avg_volume = indicators.get("avg_volume")
        # ✅ FIX #2b: avg_volume grace period (120s) — options legitimately have avg_volume=0
        # at market open (9:15–9:20 AM). Cache last valid value and use it within grace window.
        _now_ts = time.time()
        if not hasattr(self, "_sm_last_valid_avg_vol"):
            self._sm_last_valid_avg_vol: dict = {}
            self._sm_last_valid_avg_vol_ts: dict = {}
        try:
            _avg_raw = float(avg_volume) if avg_volume is not None else 0.0
            if _avg_raw > 0:
                self._sm_last_valid_avg_vol[symbol] = _avg_raw
                self._sm_last_valid_avg_vol_ts[symbol] = _now_ts
            elif (_now_ts - self._sm_last_valid_avg_vol_ts.get(symbol, 0)) < 120.0:
                avg_volume = self._sm_last_valid_avg_vol.get(symbol, avg_volume)
        except (TypeError, ValueError):
            pass
        is_nifty_context_symbol = symbol in {"NSE:NIFTY", "NIFTY"}
        try:
            if vwap is None or float(vwap) <= 0.0:
                if symbol_role == "spot_context" and (volume is None or float(volume) <= 0.0):
                    log_throttled(
                        log,
                        key=f"context_vwap_missing:{symbol}:spot",
                        msg=f"CONTEXT_VWAP_UNAVAILABLE_NON_FATAL symbol={symbol}",
                        interval_sec=300.0,
                        level=logging.INFO,
                        extra={"event": "CONTEXT_VWAP_UNAVAILABLE_NON_FATAL", "symbol": symbol, "reason": "spot_vwap_optional", "context_kind": "price_direction"},
                    )
                elif symbol_role == "futures_context":
                    log_throttled(
                        log,
                        key=f"context_vwap_missing:{symbol}:futures",
                        msg=f"CONTEXT_VWAP_UNAVAILABLE_NON_FATAL symbol={symbol}",
                        interval_sec=30.0,
                        level=logging.WARNING,
                        extra={"event": "CONTEXT_VWAP_UNAVAILABLE_NON_FATAL", "symbol": symbol, "reason": "futures_vwap_unavailable", "context_kind": "volume_flow"},
                    )
                else:
                    invalid_reason = "vwap_zero_or_invalid"
            elif avg_volume is None or float(avg_volume) <= 0.0:
                # ✅ FIX #2b: avg_volume=0 is transient (options with no live bars yet).
                # Delegate to individual strategies which have their own grace periods.
                log_throttled(
                    log,
                    key=f"avg_vol_zero_gate:{symbol}",
                    msg=f"avg_volume=0 for {symbol}, delegating to strategies",
                    interval_sec=60.0,
                    extra={"event": "avg_volume_zero_delegated", "symbol": symbol},
                )
            # ✅ FIX I: Remove hard volume==0 block at strategy_manager level.
            # NFO options have sparse ticks (once per 13+ min) and never complete
            # a live 1-min bar, so indicators["volume"] is always 0.  Blocking here
            # prevents ALL option signal evaluation.  Individual strategies (vwap_pro)
            # already handle volume gracefully with their own cached-volume fallback.
            # This pipeline-level gate is redundant and harmful for sparse instruments.
        except (TypeError, ValueError):
            invalid_reason = "data_invalid"

        if invalid_reason is not None:
            self._observability_counters["signals_blocked_by_risk"] += 1
            count = self._symbol_invalid_counts.get(symbol, 0) + 1
            self._symbol_invalid_counts[symbol] = count
            if invalid_reason == "vwap_zero_or_invalid" and is_nifty_context_symbol:
                log.info(
                    "CONTEXT_SYMBOL_INVALID_NOT_SUSPENDED symbol=%s reason=%s",
                    symbol,
                    invalid_reason,
                    extra={
                        "event": "CONTEXT_SYMBOL_INVALID_NOT_SUSPENDED",
                        "symbol": symbol,
                        "reason_code": invalid_reason,
                        "invalid_streak": count,
                    },
                )
                _log_reject(
                    "data_invalid",
                    {
                        "reason_code": invalid_reason,
                        "invalid_streak": count,
                        "ltp": current_price,
                        "vwap": vwap,
                    },
                )
                return None
            if count > self._symbol_invalid_threshold:
                if symbol not in self._symbol_temporarily_ineligible:
                    self._symbol_temporarily_ineligible[symbol] = invalid_reason
                    log.info(
                        "⛔ SYMBOL SUSPENDED — Data invalid | symbol=%s reason=%s",
                        symbol,
                        invalid_reason,
                        extra={
                            "event": "symbol_suspended_data_invalid",
                            "symbol": symbol,
                            "reason_code": invalid_reason,
                            "invalid_streak": count,
                        },
                    )
                _log_reject(
                    "data_invalid",
                    {
                        "reason_code": invalid_reason,
                        "invalid_streak": count,
                        "ltp": current_price,
                        "vwap": vwap,
                        "volume": volume,
                        "avg_volume": avg_volume,
                    },
                )
                _emit_no_signal("data_invalid", {"reason_code": invalid_reason})
                no_signal_reasons.append("data_invalid")
                _emit_strategy_exit()
                return None
            log_throttled(
                log,
                key=f"signal_invalid_streak:{symbol}",
                msg="signal_invalid_data_streak",
                interval_sec=30.0,
                extra={
                    "event": "signal_invalid_data_streak",
                    "symbol": symbol,
                    "reason_code": invalid_reason,
                    "invalid_streak": count,
                },
            )
        else:
            self._symbol_invalid_counts.pop(symbol, None)
            if symbol in self._symbol_temporarily_ineligible:
                self._symbol_temporarily_ineligible.pop(symbol, None)

        if symbol in self._symbol_temporarily_ineligible:
            _log_reject(
                "no_strategy_signal",
                {
                    "reason_code": self._symbol_temporarily_ineligible.get(symbol),
                    "invalid_streak": self._symbol_invalid_counts.get(symbol, 0),
                    "ltp": current_price,
                    "vwap": vwap,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "no_vote_reason_counts": no_vote_reason_counts,
                    "symbol_role": symbol_role,
                },
            )
            _emit_no_signal(
                "data_invalid",
                {
                    "reason_code": self._symbol_temporarily_ineligible.get(symbol),
                },
            )
            no_signal_reasons.append("data_invalid")
            _emit_strategy_exit()
            return None
        if symbol_role in {"spot_context", "futures_context"}:
            self._update_context_snapshot(symbol=symbol, indicators=indicators, role=symbol_role)
            log_throttled(
                log,
                key=f"context_symbol_strategy_eval_skipped:{symbol}",
                msg=(
                    "CONTEXT_SYMBOL_STRATEGY_EVAL_SKIPPED "
                    f"symbol={symbol} role={symbol_role} "
                    "reason=context_only_no_trade_strategy_eval"
                ),
                interval_sec=30.0,
                level=logging.INFO,
                extra={"event": "CONTEXT_SYMBOL_STRATEGY_EVAL_SKIPPED", "symbol": symbol, "symbol_role": symbol_role, "reason": "context_only_no_trade_strategy_eval"},
            )
            return None

        if symbol_role == "tradable_option":
            context_snapshots = getattr(self, "_latest_context_snapshots", {})
            spot_ctx = context_snapshots.get("spot_context", {})
            fut_ctx = context_snapshots.get("futures_context", {})
            indicators.setdefault("spot_context", spot_ctx)
            indicators.setdefault("futures_context", fut_ctx)
            direction_bias = (
                indicators.get("direction_bias")
                or indicators.get("underlying_direction_bias")
                or spot_ctx.get("direction_bias")
                or fut_ctx.get("direction_bias")
            )
            if str(direction_bias or "").upper() in {"CE", "PE"}:
                indicators["direction_bias"] = str(direction_bias).upper()
                indicators["underlying_direction_bias"] = str(direction_bias).upper()
            if fut_ctx.get("futures_volume_ratio") is not None:
                indicators.setdefault("futures_volume_ratio", fut_ctx.get("futures_volume_ratio"))
            if fut_ctx.get("vwap") is not None:
                indicators.setdefault("futures_vwap", fut_ctx.get("vwap"))
            if fut_ctx.get("vwap_slope") is not None:
                indicators.setdefault("futures_vwap_slope", fut_ctx.get("vwap_slope"))

        position = self._position_manager.get_position(symbol)

        signals: list[Signal] = []
        signal_votes: list[tuple[Signal, StrategyVote]] = []
        max_votes = max(1, int(getattr(app_settings, "MAX_STRATEGY_VOTES", 5)))
        disabled: list[str] = []
        empty: list[str] = []
        no_vote_reason_counts: dict[str, int] = {}
        errors: list[str] = []
        disabled_strategies_snapshot = disabled
        evaluation_start = time.monotonic()
        symbol_upper = symbol.upper()
        symbol_is_option = bool(re.search(r"(CE|PE)$", symbol_upper))
        symbol_is_future = symbol_upper.endswith("FUT")
        direction_bias = str(indicators.get("direction_bias") or "").upper()
        for strategy in self._strategies:
            if strategy.name in self._disabled_strategies:
                disabled.append(strategy.name)
                log.debug(
                    "strategy_disabled_skipped",
                    extra={
                        "event": "strategy_disabled_skipped",
                        "strategy": strategy.name,
                        "symbol": symbol,
                    },
                )
                continue
            strategy_name = str(getattr(strategy, "name", "") or "")
            if strategy_name in {"VWAPPro", "OrderFlow"} and not symbol_is_option:
                no_vote_reason_counts["context_symbol_skipped_for_option_strategy"] = (
                    no_vote_reason_counts.get(
                        "context_symbol_skipped_for_option_strategy", 0
                    )
                    + 1
                )
                log_throttled(
                    log,
                    f"strategy_domain_skip:{symbol}:{strategy_name}",
                    (
                        "STRATEGY_DOMAIN_SKIPPED strategy=%s symbol=%s domain=%s "
                        "reason=context_symbol_skipped_for_option_strategy"
                    ),
                    strategy_name,
                    symbol,
                    "underlying_or_futures",
                    interval_sec=30.0,
                    level=logging.INFO,
                )
                continue
            if strategy_name == "BBSqueeze" and symbol_is_option:
                no_vote_reason_counts["context_symbol_skipped_for_underlying_strategy"] = (
                    no_vote_reason_counts.get(
                        "context_symbol_skipped_for_underlying_strategy", 0
                    )
                    + 1
                )
                log_throttled(
                    log,
                    f"strategy_domain_skip:{symbol}:{strategy_name}",
                    (
                        "STRATEGY_DOMAIN_SKIPPED strategy=%s symbol=%s domain=%s "
                        "reason=context_symbol_skipped_for_underlying_strategy"
                    ),
                    strategy_name,
                    symbol,
                    "options",
                    interval_sec=30.0,
                    level=logging.INFO,
                )
                continue
            try:
                base_signal = strategy.generate_signal(
                    symbol, indicators, current_price, position
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(strategy.name)
                error_strategies.append(strategy.name)
                log.error(
                    "Failure in strategy generate for %s: %s",
                    strategy.name,
                    exc,
                    exc_info=exc,
                )
                continue
            if base_signal is None:
                empty.append(strategy.name)
                reason = str(getattr(strategy, "last_no_vote_reason", "none") or "none")
                no_vote_reason_counts[reason] = no_vote_reason_counts.get(reason, 0) + 1
                log_throttled(
                    log,
                    key=f"strategy_no_vote:{symbol}:{strategy.name}:{reason}",
                    msg=f"STRATEGY_NO_VOTE strategy={strategy.name} symbol={symbol} reason={reason}",
                    interval_sec=15.0,
                    level=logging.INFO,
                )
                continue
            entry = score_map.get(strategy.name)
            adjusted = self._apply_weighted_confidence(base_signal, strategy.name, entry)
            signals.append(adjusted)
            vote = signal_to_vote(adjusted, strategy.name)
            vote = self._apply_regime_vote_weight(vote=vote, regime_name=regime_name)
            signal_votes.append((adjusted, vote))
            if len(signals) >= max_votes:
                break

        elapsed = time.monotonic() - evaluation_start
        if elapsed > 3.0:
            log.warning(
                "Strategy evaluation exceeded watchdog threshold",
                extra={
                    "event": "strategy_eval_watchdog",
                    "symbol": symbol,
                    "elapsed_seconds": elapsed,
                },
            )

        if not signals:
            missing = sorted(
                name
                for name in self._required_indicators
                if indicators.get(name) is None
            )
            log_throttled(
                log,
                key=f"strategy_manager_no_signal:{symbol}",
                msg="Condition met: strategy_manager_no_signal",
                level=logging.DEBUG,
                interval_sec=10.0,
                extra={
                    "event": "strategy_manager_no_signal",
                    "symbol": symbol,
                    "disabled_strategies": disabled,
                    "no_signal_strategies": empty[:8],
                    "no_signal_count": len(empty),
                    "error_strategies": errors,
                    "missing_indicators": missing,
                    "volume": indicators.get("volume"),
                    "avg_volume": indicators.get("avg_volume"),
                    "no_vote_reason_counts": no_vote_reason_counts,
                },
            )
            self._record_no_signal_summary(
                symbol=symbol,
                missing=missing,
                indicators=indicators,
                error_strategies=errors,
            )
            _log_reject(
                "no_strategy_signal",
                {
                    "missing_indicators": missing,
                    "no_signal_strategies": empty[:8],
                    "error_strategies": errors,
                    "ltp": current_price,
                    "vwap": vwap,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "no_vote_reason_counts": no_vote_reason_counts,
                    "symbol_role": symbol_role,
                },
            )
            _emit_no_signal(
                "no_strategy_signal",
                {
                    "missing_indicators": missing,
                    "no_signal_strategies": empty[:8],
                    "error_strategies": errors,
                    "no_vote_reason_counts": no_vote_reason_counts,
                    "symbol_role": symbol_role,
                },
            )
            no_signal_reasons.append("no_strategy_signal")
            _emit_strategy_exit()
            return None

        combined = self._combine_strategy_votes(
            symbol=symbol,
            signals=signal_votes,
            indicators=indicators,
        )
        if combined and bool(getattr(combined, "metadata", {}).get("is_approved")):
            exit_result = "signal"
            signal_action = combined.action
            signal_score = float(combined.quantity)
            signal_confidence = float(combined.confidence)
            _emit_strategy_exit()
            return combined
        if combined and self._filter_signal(combined):
            orchestrator = self._orchestrator
            if orchestrator is not None:
                try:
                    combined = orchestrator.filter_signal(
                        combined, indicators, self._position_manager
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in orchestrator.filter_signal: %s",
                        exc,
                        exc_info=exc,
                    )
                    combined = None
            if combined:
                scaled_quantity = max(1, int(round(combined.quantity * regime_scale)))
                combined = Signal(
                    action=combined.action,
                    symbol=combined.symbol,
                    quantity=scaled_quantity,
                    confidence=combined.confidence,  # ✅ FIX #6: Was metadata.get("probability", combined.confidence); that bypassed score weighting
                    reason=combined.reason,
                    stop_loss=combined.stop_loss,
                    take_profit=combined.take_profit,
                    metadata={
                        **dict(combined.metadata),
                        "regime_scale": regime_scale,
                        "regime": regime_name,
                    },
                )
                log.info(
                    "Condition met: scored_signal_ready",
                    extra={
                        "event": "scored_signal_ready",
                        "symbol": symbol,
                        "action": combined.action,
                        "confidence": combined.confidence,
                        "quantity": combined.quantity,
                    },
                )
                self._observability_counters["signals_generated"] += 1
                self._avg_confidence_window.append(float(combined.confidence))
                self._avg_kelly_window.append(
                    float(dict(combined.metadata).get("kelly_fraction", 0.0))
                )
                self._emit_metrics_snapshot()
                exit_result = "signal"
                signal_action = combined.action
                signal_score = float(combined.quantity)
                signal_confidence = float(combined.confidence)
                _emit_strategy_exit()
                return combined
        elif combined is None:
            log_throttled(
                log,
                key=f"strategy_manager_no_combined:{symbol}",
                msg="strategy_manager_no_combined_signal",
                interval_sec=30.0,
                extra={
                    "event": "strategy_manager_no_combined_signal",
                    "symbol": symbol,
                    "no_vote_reason_counts": no_vote_reason_counts,
                },
            )
            _log_reject(
                "no_strategy_signal",
                {
                    "stage": "combine",
                    "ltp": current_price,
                    "vwap": vwap,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "no_vote_reason_counts": no_vote_reason_counts,
                    "symbol_role": symbol_role,
                },
            )
            _emit_no_signal("data_invalid", {"stage": "combine"})
            no_signal_reasons.append("combine_none")
        else:
            log_throttled(
                log,
                key=f"strategy_manager_filtered:{symbol}",
                msg="strategy_manager_filtered_signal",
                interval_sec=30.0,
                extra={
                    "event": "strategy_manager_filtered_signal",
                    "symbol": symbol,
                    "action": combined.action,
                    "confidence": combined.confidence,
                },
            )
            _log_reject(
                "no_strategy_signal",
                {
                    "stage": "filter",
                    "action": combined.action,
                    "confidence": combined.confidence,
                    "ltp": current_price,
                    "vwap": vwap,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "no_vote_reason_counts": no_vote_reason_counts,
                    "symbol_role": symbol_role,
                },
            )
            _emit_no_signal(
                "data_invalid",
                {
                    "stage": "filter",
                    "action": combined.action,
                    "confidence": combined.confidence,
                },
            )
            no_signal_reasons.append("filtered_signal")
        _emit_strategy_exit()
        return None

    def _emit_metrics_snapshot(self) -> None:
        """Args: None. Returns: None. Raises: Exception."""

        now_ts = time.time()
        if now_ts - self._last_metrics_log_ts < 300.0:
            return
        self._last_metrics_log_ts = now_ts
        log.info(
            "Condition met: strategy_metrics_snapshot",
            extra={
                "event": "strategy_metrics_snapshot",
                "metrics": dict(self._observability_counters),
                "avg_confidence": (
                    (
                        sum(self._avg_confidence_window)
                        / len(self._avg_confidence_window)
                    )
                    if self._avg_confidence_window
                    else 0.0
                ),
                "avg_kelly_fraction": (
                    (sum(self._avg_kelly_window) / len(self._avg_kelly_window))
                    if self._avg_kelly_window
                    else 0.0
                ),
                "regime": getattr(self._regime_state, "regime", None),
                "rolling_sharpe": float(
                    self._performance.get(
                        "_aggregate", StrategyPerformance()
                    ).sharpe_ratio()
                ),
            },
        )

    def _combine_strategy_votes(
        self,
        *,
        symbol: str,
        signals: list[tuple[Signal, StrategyVote]],
        indicators: t.Mapping[str, t.Any],
    ) -> Signal | None:
        """Args: symbol/signals/indicators. Returns: consensus signal or None. Raises: none."""
        symbol_norm = str(symbol or "").strip().upper()
        indicator_map = dict(indicators or {})
        selected_symbols = {
            str(indicator_map.get("selected_ce") or "").strip().upper(),
            str(indicator_map.get("selected_pe") or "").strip().upper(),
        }
        selected_symbols.discard("")
        indicator_selected_option = bool(
            indicator_map.get("is_selected_option")
            or symbol_norm in selected_symbols
        )
        indicator_near_atm = False
        try:
            if indicator_map.get("strike_distance_from_atm") is not None:
                indicator_near_atm = float(indicator_map.get("strike_distance_from_atm")) <= 50.0
        except (TypeError, ValueError):
            indicator_near_atm = False
        if not signals:
            return None
        def _raw_score(vote: StrategyVote) -> float:
            payload = dict(vote.metadata or {})
            try:
                return float(
                    payload.get("raw_vote_score")
                    if payload.get("raw_vote_score") is not None
                    else payload.get("raw_setup_score")
                    if payload.get("raw_setup_score") is not None
                    else payload.get("vote_score")
                    if payload.get("vote_score") is not None
                    else vote.score
                )
            except (TypeError, ValueError):
                return float(vote.score or 0.0)
        def _weighted_score(vote: StrategyVote) -> float:
            try:
                return float(vote.score or 0.0)
            except (TypeError, ValueError):
                return 0.0
        def _regime_weight(vote: StrategyVote) -> float:
            try:
                return float((vote.metadata or {}).get("regime_weight", 1.0) or 1.0)
            except (TypeError, ValueError):
                return 1.0
        def _context_bonus_score(vote: StrategyVote) -> float:
            payload = dict(vote.metadata or {})
            for key in ("context_bonus_score", "context_score", "raw_setup_score", "raw_vote_score", "vote_score"):
                try:
                    if payload.get(key) is not None:
                        return max(0.0, float(payload.get(key)))
                except (TypeError, ValueError):
                    continue
            return max(0.0, _raw_score(vote))
        def _context_veto_score(vote: StrategyVote) -> float:
            payload = dict(vote.metadata or {})
            for key in ("context_veto_score", "context_score", "raw_setup_score", "raw_vote_score", "vote_score"):
                try:
                    if payload.get(key) is not None:
                        return max(0.0, float(payload.get(key)))
                except (TypeError, ValueError):
                    continue
            return max(0.0, _raw_score(vote))
        mode_profile = self.get_strategy_mode_profile()
        trigger_votes: list[tuple[Signal, StrategyVote]] = []
        context_votes: list[tuple[Signal, StrategyVote]] = []
        for signal, vote in signals:
            if signal.action in {"CLOSE_LONG", "CLOSE_SHORT"}:
                return signal
            role = str((vote.metadata or {}).get("role") or "trigger").lower()
            if role == "context":
                context_votes.append((signal, vote))
            else:
                trigger_votes.append((signal, vote))
        if not trigger_votes:
            allow_context_promotion = bool(mode_profile.get("allow_context_promotion", True))
            live_promotion_allowed = str(os.getenv("STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED", "false")).lower() in {"1", "true", "yes", "on"}
            execution_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
            allowed_strategies = {s.strip() for s in str(os.getenv("STRATEGY_CONTEXT_PROMOTION_ALLOWED_STRATEGIES", "OrderFlow,VWAPPro")).split(",") if s.strip()}
            min_score = float(os.getenv("STRATEGY_CONTEXT_PROMOTION_MIN_SCORE", "5.0") or "5.0")
            min_conf = float(os.getenv("STRATEGY_CONTEXT_PROMOTION_MIN_CONFIDENCE", "0.45") or "0.45")
            if context_votes and allow_context_promotion and (execution_mode != "LIVE" or live_promotion_allowed):
                best_signal, best_vote = max(context_votes, key=lambda pair: _raw_score(pair[1]))
                opp = [v for _, v in context_votes if v.side in {"CE", "PE"} and v.side != best_vote.side]
                vetoed = bool(opp and max(_context_veto_score(v) for v in opp) >= 8.0)
                raw_score = _raw_score(best_vote)
                near_atm_ok = indicator_near_atm
                selected_ok = indicator_selected_option or near_atm_ok
                if best_vote.strategy in allowed_strategies and raw_score >= min_score and best_vote.confidence >= min_conf and best_vote.side in {"CE","PE"} and selected_ok and not vetoed:
                    md = dict(best_signal.metadata or {})
                    md.update({"role":"trigger","promoted_from_context":True,"consensus_stage":"context_promoted_controlled","promotion_reason":"context_only_quality_pass","raw_setup_score":round(raw_score,3),"setup_score":round(raw_score,3),"strategy_score":round(raw_score,3),"final_trade_score":round(raw_score,3),"trade_side":best_vote.side,"side":best_vote.side,"contract_side":best_vote.side,"confirming_votes":[best_vote.strategy]})
                    return Signal(action="BUY",symbol=best_signal.symbol,quantity=best_signal.quantity,confidence=best_signal.confidence,reason=best_signal.reason,stop_loss=best_signal.stop_loss,take_profit=best_signal.take_profit,metadata=md)
            log_throttled(
                log,
                f"strategy_no_trigger_vote:{symbol_norm}",
                "STRATEGY_NO_TRIGGER_VOTE symbol=%s context_votes=%s",
                symbol_norm,
                [v.strategy for _, v in context_votes],
                interval_sec=30.0,
                level=logging.INFO,
                extra={
                    "event": "STRATEGY_NO_TRIGGER_VOTE",
                    "symbol": symbol_norm,
                    "context_votes": [v.strategy for _, v in context_votes],
                },
            )
            log.info(
                "TRADE_DECISION_TRACE symbol=%s strategy=%s side=%s allowed=%s blocked_at=%s blocked_reason=%s context_votes=%s",
                symbol_norm,
                "context_only",
                "UNKNOWN",
                False,
                "strategy_manager_trigger_gate",
                "no_trigger_vote",
                [v.strategy for _, v in context_votes],
                extra={
                    "event": "TRADE_DECISION_TRACE",
                    "symbol": symbol_norm,
                    "strategy": "context_only",
                    "allowed": False,
                    "blocked_at": "strategy_manager_trigger_gate",
                    "blocked_reason": "no_trigger_vote",
                    "context_votes": [v.strategy for _, v in context_votes],
                },
            )
            return None
        best_signal, best_vote = max(trigger_votes, key=lambda pair: _raw_score(pair[1]))
        metadata = dict(best_signal.metadata or {})
        threshold = float(os.getenv("STRATEGY_TRIGGER_MIN_SCORE", "4.5") or "4.5")
        single_high = float(os.getenv("STRATEGY_SINGLE_VOTE_HIGH_CONVICTION", "8.8") or "8.8")
        allow_scalp_single = bool(mode_profile.get("allow_single_vote", True)) and str(os.getenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", "false")).lower() in {"1", "true", "yes", "on"}
        raw_trigger_score = _raw_score(best_vote)
        weighted_trigger_score = _weighted_score(best_vote)
        vetoed = False
        same_side_context = [v for _, v in context_votes if v.side == best_vote.side]
        opposite_context = [v for _, v in context_votes if v.side in {"CE", "PE"} and v.side != best_vote.side]
        positive_context = sum(_context_bonus_score(v) for v in same_side_context)
        negative_context = sum(_context_veto_score(v) for v in opposite_context)
        context_bonus = min(1.5, 0.45 * positive_context)
        context_penalty = min(1.5, 0.60 * negative_context)
        final_score = raw_trigger_score + context_bonus - context_penalty
        final_score = max(0.0, min(10.0, final_score))
        if opposite_context and max(_context_veto_score(v) for v in opposite_context) >= 8.0:
            vetoed = True
        selected_ok = True
        near_atm = indicator_near_atm
        if len(trigger_votes) == 1:
            selected_ce = str(
                metadata.get("selected_ce")
                or indicator_map.get("selected_ce")
                or ""
            )
            selected_pe = str(
                metadata.get("selected_pe")
                or indicator_map.get("selected_pe")
                or ""
            )
            selected_symbol_set = {
                str(selected_ce).strip().upper(),
                str(selected_pe).strip().upper(),
            }
            selected_symbol_set.discard("")
            selected_option = bool(
                metadata.get("is_selected_option")
                or indicator_selected_option
                or symbol_norm in selected_symbol_set
            )
            near_atm_threshold = float(os.getenv("STRATEGY_NEAR_ATM_THRESHOLD_POINTS", "50") or "50")
            try:
                strike_distance_from_atm = float(metadata.get("strike_distance_from_atm"))
                near_atm = strike_distance_from_atm <= near_atm_threshold
            except (TypeError, ValueError):
                strike_distance_from_atm = None
                near_atm = indicator_near_atm
            def _extract_option_strike(sym: str) -> int | None:
                match = re.search(r"(\d{5})(CE|PE)$", str(sym).upper())
                return int(match.group(1)) if match else None
            symbol_strike = _extract_option_strike(symbol_norm)
            same_side_selected = selected_ce if symbol_norm.endswith("CE") else selected_pe if symbol_norm.endswith("PE") else ""
            selected_strike = _extract_option_strike(same_side_selected)
            if strike_distance_from_atm is None and symbol_strike is not None and selected_strike is not None:
                strike_distance_from_atm = abs(float(symbol_strike) - float(selected_strike))
                near_atm = strike_distance_from_atm <= near_atm_threshold
            score_min, conf_min = self._single_vote_thresholds(best_vote.strategy)
            regime_weight = _regime_weight(best_vote)
            score_ok = raw_trigger_score >= score_min
            conf_ok = best_vote.confidence >= conf_min
            selected_ok = selected_option or near_atm
            selected_ok_reason = "selected_option" if selected_option else "near_atm" if near_atm else "not_selected_or_near_atm"
            metadata["selected_ce"] = selected_ce
            metadata["selected_pe"] = selected_pe
            metadata["strike_distance_from_atm"] = strike_distance_from_atm
            metadata["is_selected_option"] = selected_option
            metadata["selected_ok_reason"] = selected_ok_reason
            log.info(
                "SINGLE_VOTE_DECISION strategy=%s raw_score=%.2f weighted_score=%.2f regime_weight=%.2f score_min=%.2f confidence=%.2f conf_min=%.2f selected_ok=%s vetoed=%s",
                best_vote.strategy,
                raw_trigger_score,
                weighted_trigger_score,
                regime_weight,
                score_min,
                best_vote.confidence,
                conf_min,
                selected_ok,
                vetoed,
                extra={
                    "event": "SINGLE_VOTE_DECISION",
                    "symbol": symbol_norm,
                    "strategy": best_vote.strategy,
                    "raw_score": raw_trigger_score,
                    "weighted_score": weighted_trigger_score,
                    "regime_weight": regime_weight,
                    "score_min": score_min,
                    "confidence": best_vote.confidence,
                    "conf_min": conf_min,
                    "selected_ok": selected_ok,
                    "selected_ok_reason": selected_ok_reason,
                    "vetoed": vetoed,
                },
            )
            if raw_trigger_score >= single_high and selected_ok and not vetoed:
                metadata_stage = "preliminary_single_high_conviction"
            elif allow_scalp_single and score_ok and conf_ok and selected_ok and not vetoed:
                metadata_stage = "single_vote_scalp_controlled"
            else:
                allow_candidate_switch = str(os.getenv("STRATEGY_ALLOW_CANDIDATE_SWITCH_ON_HIGH_SCORE", "false")).lower() in {"1", "true", "yes", "on"}
                max_candidate_switch_distance = float(os.getenv("CANDIDATE_SWITCH_MAX_DISTANCE_POINTS", "100") or "100")
                quote_depth_valid = bool(metadata.get("quote_depth_valid", True))
                switch_allowed = (
                    allow_candidate_switch
                    and (strike_distance_from_atm is not None)
                    and raw_trigger_score >= 7.0
                    and best_vote.confidence >= 0.60
                    and quote_depth_valid
                    and strike_distance_from_atm <= max_candidate_switch_distance
                )
                if switch_allowed:
                    metadata_stage = "single_vote_candidate_switch"
                    metadata["candidate_switch_requested"] = True
                    metadata["candidate_switch_reason"] = "high_raw_score_nearby_strike"
                else:
                    if not score_ok:
                        blocked_reason = "raw_score_below_min"
                    elif not conf_ok:
                        blocked_reason = "confidence_below_min"
                    elif not selected_ok:
                        blocked_reason = "not_selected_or_near_atm"
                    elif vetoed:
                        blocked_reason = "context_veto"
                    elif not allow_scalp_single:
                        blocked_reason = "single_vote_scalp_disabled"
                    else:
                        blocked_reason = "single_vote_gate_failed_unknown"
                    log.info(
                        "TRADE_DECISION_TRACE symbol=%s strategy=%s side=%s allowed=%s blocked_at=%s blocked_reason=%s selected_ce=%s selected_pe=%s strike_distance_from_atm=%s near_atm_threshold=%s selected_ok_reason=%s raw_score=%.2f confidence=%.2f",
                        symbol_norm,
                        best_vote.strategy,
                        best_vote.side,
                        False,
                        "single_vote_gate",
                        blocked_reason,
                        selected_ce,
                        selected_pe,
                        strike_distance_from_atm,
                        near_atm_threshold,
                        selected_ok_reason,
                        raw_trigger_score,
                        best_vote.confidence,
                        extra={"event": "TRADE_DECISION_TRACE", "symbol": symbol_norm, "strategy": best_vote.strategy, "side": best_vote.side, "allowed": False, "blocked_at": "single_vote_gate", "blocked_reason": blocked_reason, "selected_ce": selected_ce, "selected_pe": selected_pe, "strike_distance_from_atm": strike_distance_from_atm, "near_atm_threshold": near_atm_threshold, "selected_ok_reason": selected_ok_reason, "raw_score": raw_trigger_score, "confidence": best_vote.confidence},
                    )
                    return None
        else:
            metadata_stage = "multi_vote_confirmed"
        metadata["is_selected_option"] = bool(metadata.get("is_selected_option") or indicator_selected_option)
        metadata["indicator_near_atm"] = bool(indicator_near_atm)
        metadata["selected_ok_reason"] = metadata.get("selected_ok_reason", "selected_or_near_atm")
        metadata["near_atm_threshold"] = float(os.getenv("STRATEGY_NEAR_ATM_THRESHOLD_POINTS", "50") or "50")
        metadata["raw_setup_score"] = round(raw_trigger_score, 3)
        metadata["setup_score"] = round(raw_trigger_score, 3)
        metadata["regime_weighted_score"] = round(weighted_trigger_score, 3)
        metadata["regime_weight"] = round(_regime_weight(best_vote), 3)
        metadata["context_bonus"] = round(context_bonus, 3)
        metadata["context_penalty"] = round(context_penalty, 3)
        metadata["final_trade_score"] = round(final_score, 3)
        if vetoed or final_score < threshold:
            blocked_reason = "context_veto" if vetoed else "final_trade_score_below_threshold"
            log.info(
                "TRADE_DECISION_TRACE symbol=%s strategy=%s side=%s data_gate=%s setup_score=%.2f setup_min=%.2f weighted_score=%.2f regime_weight=%.2f context_bonus=%.2f context_penalty=%.2f final_score=%.2f final_min=%.2f allowed=%s blocked_at=%s blocked_reason=%s",
                symbol_norm, best_vote.strategy, best_vote.side, True, raw_trigger_score, threshold, weighted_trigger_score, _regime_weight(best_vote), context_bonus, context_penalty, final_score, threshold, False, "strategy_manager_combine", blocked_reason,
                extra={"event": "TRADE_DECISION_TRACE", "symbol": symbol_norm, "strategy": best_vote.strategy, "side": best_vote.side, "data_gate": True, "setup_score": raw_trigger_score, "setup_min": threshold, "weighted_score": weighted_trigger_score, "regime_weight": _regime_weight(best_vote), "context_bonus": context_bonus, "context_penalty": context_penalty, "final_score": final_score, "final_min": threshold, "allowed": False, "blocked_at": "strategy_manager_combine", "blocked_reason": blocked_reason},
            )
            return None
        quality_score, quality_meta = self._compute_trade_quality_score(
            best_vote,
            indicator_map,
            symbol=symbol_norm,
            selected_ok=selected_ok,
            near_atm_ok=near_atm,
            context_votes=[v for _, v in context_votes],
        )
        quality_min_required = float(mode_profile.get("min_trade_quality", 5.0))
        quality_pass = quality_score >= quality_min_required
        metadata.update(quality_meta)
        metadata["quality_min_required"] = quality_min_required
        metadata["quality_pass"] = quality_pass
        if not quality_pass:
            metadata["quality_block_reason"] = str(metadata.get("quality_block_reason") or "trade_quality_below_threshold")
            log.info(
                "STRATEGY_QUALITY_REJECT symbol=%s strategy=%s side=%s score=%.2f reason=%s",
                symbol_norm,
                best_vote.strategy,
                best_vote.side,
                quality_score,
                metadata["quality_block_reason"],
            )
            return None

        metadata.setdefault("trigger_strategy_score", metadata.get("strategy_score"))
        metadata["setup_score"] = round(raw_trigger_score, 3)
        metadata["raw_setup_score"] = round(raw_trigger_score, 3)
        metadata["strategy_score"] = round(raw_trigger_score, 3)
        metadata["consensus_score"] = round(final_score, 3)
        metadata["final_trade_score"] = round(final_score, 3)
        metadata["trade_side"] = best_vote.side
        metadata["contract_side"] = best_vote.side
        metadata["side"] = best_vote.side
        metadata["confirming_votes"] = [best_vote.strategy] + [v.strategy for v in same_side_context]
        existing_direction_bias = str(metadata.get("direction_bias") or "").upper()
        if existing_direction_bias not in {"CE", "PE"}:
            metadata.pop("direction_bias", None)
        metadata["confidence"] = float(best_vote.confidence)
        metadata["consensus_stage"] = metadata_stage
        return Signal(
            action="BUY",
            symbol=best_signal.symbol,
            quantity=best_signal.quantity,
            confidence=best_signal.confidence,
            reason=best_signal.reason,
            stop_loss=best_signal.stop_loss,
            take_profit=best_signal.take_profit,
            metadata=metadata,
        )

    def get_strategy_mode_profile(self) -> dict[str, t.Any]:
        """Args: none. Returns: strategy mode profile. Raises: none."""
        execution_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        if execution_mode == "LIVE":
            defaults = {"allow_context_promotion": False, "allow_single_vote": False, "min_trade_quality": 7.0}
        elif execution_mode == "PAPER":
            defaults = {"allow_context_promotion": True, "allow_single_vote": True, "min_trade_quality": 5.8}
        else:
            defaults = {"allow_context_promotion": True, "allow_single_vote": True, "min_trade_quality": 5.0}
        return {
            "mode": execution_mode,
            "allow_context_promotion": str(os.getenv("STRATEGY_ALLOW_CONTEXT_PROMOTION", str(defaults["allow_context_promotion"]).lower())).lower() in {"1", "true", "yes", "on"},
            "allow_single_vote": str(os.getenv("STRATEGY_ALLOW_SINGLE_VOTE_BY_MODE", str(defaults["allow_single_vote"]).lower())).lower() in {"1", "true", "yes", "on"},
            "min_trade_quality": float(os.getenv(f"STRATEGY_MIN_TRADE_QUALITY_{execution_mode}", str(defaults["min_trade_quality"])) or defaults["min_trade_quality"]),
        }

    def _compute_trade_quality_score(
        self,
        vote: StrategyVote,
        indicators: t.Mapping[str, t.Any],
        *,
        symbol: str,
        selected_ok: bool,
        near_atm_ok: bool,
        context_votes: list[StrategyVote],
    ) -> tuple[float, dict[str, t.Any]]:
        """Args: vote+indicators. Returns: trade quality score and metadata. Raises: none."""
        payload = dict(vote.metadata or {})
        components = {
            "strategy_score": min(3.0, max(0.0, float(payload.get("strategy_score", vote.score or 0.0)) / 3.0)),
            "direction_alignment": float(payload.get("direction_alignment_score", 1.0)),
            "liquidity_spread_quality": min(2.0, max(0.0, float(payload.get("liquidity_score", 1.0)))),
            "freshness_tick_quality": 1.0 if not bool(indicators.get("stale_data_used")) else 0.0,
            "same_side_context_confirmation": 1.0 if any(v.side == vote.side for v in context_votes) else 0.0,
            "market_regime_time_suitability": 1.0,
        }
        penalties: dict[str, float] = {}
        if bool(indicators.get("stale_data_used")):
            penalties["stale_data"] = -3.0
        spread_pct = float(payload.get("spread_pct") or indicators.get("spread_pct") or 0.0)
        if spread_pct > float(os.getenv("STRATEGY_MAX_SPREAD_PCT", "28.0")):
            penalties["spread_above_max"] = -3.0
        if any(v.side in {"CE", "PE"} and v.side != vote.side and float((v.metadata or {}).get("context_veto_score", v.score)) >= 8.0 for v in context_votes):
            penalties["opposite_context_veto"] = -4.0
        if not selected_ok and not near_atm_ok:
            penalties["non_selected_far_otm"] = -3.0
        if str(vote.strategy).lower() == "orderflow" and str(payload.get("role", "")).lower() == "trigger":
            if not bool(payload.get("quote_depth_valid")):
                penalties["orderflow_missing_depth"] = -2.0
        score = max(0.0, min(10.0, sum(components.values()) + sum(penalties.values())))
        block_reason = "ok" if score >= 0 else "invalid"
        if penalties:
            block_reason = ",".join(penalties.keys())
        return score, {
            "trade_quality_score": round(score, 3),
            "trade_quality_components": components,
            "trade_quality_penalties": penalties,
            "quality_block_reason": block_reason,
            "trade_quality_symbol": symbol,
        }

    def increment_observability_counter(self, key: str) -> None:
        """Args: key. Returns: None. Raises: None."""

        if key in self._observability_counters:
            self._observability_counters[key] += 1

    def _extract_regime_scale(self, adjustments: t.Mapping[str, t.Any]) -> float:
        """Args: adjustments. Returns: float. Raises: Exception."""

        try:
            raw = (
                adjustments.get("position_scale")
                or adjustments.get("size_multiplier")
                or adjustments.get("sizing_multiplier")
                or 1.0
            )
            scale = float(raw)
            if scale <= 0:
                return 1.0
            return min(scale, 3.0)
        except Exception as exc:  # noqa: BLE001
            log.error("Failure in StrategyManager._extract_regime_scale: %s", exc)
            return 1.0

    def _single_vote_thresholds(self, strategy_name: str) -> tuple[float, float]:
        """Args: strategy_name. Returns: score/conf thresholds. Raises: none."""
        # Keep live defaults conservative:
        # STRATEGY_ALLOW_SINGLE_VOTE_SCALP=false
        # STRATEGY_SINGLE_VOTE_VWAP_MIN_SCORE=5.8
        # STRATEGY_SINGLE_VOTE_VWAP_MIN_CONFIDENCE=0.45
        # Paper/shadow only (opt-in):
        # STRATEGY_ALLOW_SINGLE_VOTE_SCALP=true
        # STRATEGY_SINGLE_VOTE_VWAP_MIN_SCORE=5.5
        # STRATEGY_SINGLE_VOTE_VWAP_MIN_CONFIDENCE=0.45
        execution_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        is_live = execution_mode == "LIVE"
        key = str(strategy_name or "").strip().lower().replace(" ", "_")
        if key in {"vwappro", "vwap_pro"}:
            return (
                float(os.getenv("STRATEGY_SINGLE_VOTE_VWAP_MIN_SCORE", "5.8" if is_live else "4.8") or ("5.8" if is_live else "4.8")),
                float(os.getenv("STRATEGY_SINGLE_VOTE_VWAP_MIN_CONFIDENCE", "0.45" if is_live else "0.40") or ("0.45" if is_live else "0.40")),
            )
        return (
            float(os.getenv("STRATEGY_SINGLE_VOTE_SCALP_MIN", "6.6" if is_live else "4.5") or ("6.6" if is_live else "4.5")),
            float(os.getenv("STRATEGY_SINGLE_VOTE_MIN_CONFIDENCE", "0.60" if is_live else "0.45") or ("0.60" if is_live else "0.45")),
        )

    def _apply_regime_vote_weight(
        self, *, vote: StrategyVote, regime_name: str | None
    ) -> StrategyVote:
        """Args: vote + regime_name. Returns: weighted StrategyVote. Raises: none."""
        try:
            regime_key = str(regime_name or "").upper()
            weight_map = REGIME_STRATEGY_WEIGHTS.get(regime_key, {})
            weight = float(weight_map.get(vote.strategy, 1.0))
            weighted_score = max(0.0, min(10.0, vote.score * weight))
            log.debug(
                "STRATEGY_REGIME_WEIGHT strategy=%s regime=%s weight=%.3f",
                vote.strategy,
                regime_key or "UNKNOWN",
                weight,
                extra={
                    "event": "STRATEGY_REGIME_WEIGHT",
                    "strategy": vote.strategy,
                    "regime": regime_key or "UNKNOWN",
                    "weight": weight,
                },
            )
            metadata = dict(vote.metadata or {})
            raw_score = float((vote.metadata or {}).get("raw_setup_score", vote.score))
            metadata["raw_vote_score"] = raw_score
            metadata["raw_setup_score"] = raw_score
            metadata["regime_weight"] = weight
            metadata["regime_weighted_vote_score"] = weighted_score
            metadata["regime_name"] = regime_key or "UNKNOWN"
            return StrategyVote(
                strategy=vote.strategy,
                side=vote.side,
                score=weighted_score,
                confidence=vote.confidence,
                reasons=list(vote.reasons),
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Failure in StrategyManager._apply_regime_vote_weight: %s", exc)
            return vote

    def _bounded_confidence(self, candidate: float | None) -> float:
        """Clamp candidate confidence to an acceptable range.

        Args:
            candidate: Proposed confidence multiplier for the signal.

        Returns:
            float: Confidence bounded between the configured minimum and one.

        Raises:
            None.
        """

        try:
            value = float(candidate if candidate is not None else 0.0)
        except Exception as exc:  # noqa: BLE001 - defensive conversion
            log.error(
                "Failure in StrategyManager._bounded_confidence: %s",
                exc,
                exc_info=exc,
            )
            lower_bound = max(0.0, float(getattr(self, "_min_confidence", 0.0)))
            return lower_bound
        lower_bound = max(0.0, float(getattr(self, "_min_confidence", 0.0)))
        bounded = max(lower_bound, min(value, 1.0))
        if bounded != value:
            log.info(
                "Condition met: confidence_bounded",
                extra={
                    "event": "confidence_bounded",
                    "candidate": value,
                    "bounded": bounded,
                },
            )
        return bounded

    def _log_regime_gate_decision(
        self,
        *,
        symbol: str,
        allowed: bool,
        reasons: tuple[str, ...],
        snapshot: RegimeSnapshot | None,
    ) -> None:
        """Log the regime gating decision while avoiding repeated noise.

        Args:
            symbol: Trading symbol evaluated by the manager.
            allowed: ``True`` when the gate approved trading.
            reasons: Tuple describing the reasons from the regime gate.
            snapshot: Latest snapshot sourced from the regime manager.

        Returns:
            None.
        """

        log.debug(
            "Entered StrategyManager._log_regime_gate_decision",
            extra={"event": "strategy_regime_gate_log", "symbol": symbol},
        )

        now = time.time()

        # ---------- SAFE NORMALIZATION (NO ASSUMPTIONS) ----------
        regime_val: str | None = None
        confidence_val: float | None = None

        if isinstance(snapshot, RegimeSnapshot):
            regime_val = snapshot.regime
            confidence_val = snapshot.confidence
        elif isinstance(snapshot, dict):
            regime_val = snapshot.get("regime")
            confidence_val = snapshot.get("confidence")
        elif isinstance(snapshot, str):
            regime_val = snapshot
            confidence_val = None
        # snapshot may also be None -> leave as None
        # ---------------------------------------------------------

        gate_key = (allowed, reasons, regime_val)

        extras = {
            "event": (
                "strategy_regime_gate_allow"
                if allowed
                else "strategy_regime_gate_block"
            ),
            "symbol": symbol,
            "regime": regime_val,
            "confidence": confidence_val,
            "reasons": list(reasons),
        }

        if self._last_regime_gate == gate_key:
            if (now - self._last_regime_gate_at) < self._regime_gate_cooldown:
                return
            self._last_regime_gate_at = now
            log.debug("Condition met: strategy_regime_gate_unchanged", extra=extras)
            return

        self._last_regime_gate = gate_key
        self._last_regime_gate_at = now
        if allowed:
            log.info("Condition met: strategy_regime_gate_allow", extra=extras)
        else:
            log.info("Condition met: strategy_regime_gate_block", extra=extras)

    def _apply_weighted_confidence(
        self,
        signal: Signal,
        strategy_name: str,
        score_entry: StrategyScore | None,
    ) -> Signal:
        """Return signal with confidence adjusted by *score_entry*.

        Args:
            signal: Raw signal generated by the strategy.
            strategy_name: Strategy name associated with the signal.
            score_entry: Score snapshot used for weighting.

        Returns:
            Signal: Weighted signal containing additional metadata.

        Raises:
            None.
        """

        base_metadata: dict[str, t.Any] = {}
        if isinstance(signal.metadata, dict):
            base_metadata = dict(signal.metadata)
        base_metadata.update({"strategy": strategy_name})
        if score_entry is None:
            return Signal(
                action=signal.action,
                symbol=signal.symbol,
                quantity=signal.quantity,
                confidence=signal.confidence,
                reason=signal.reason,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata={**base_metadata, "strategy_weight": 1.0},
            )
        weight = max(self._score_floor, min(score_entry.weight, self._score_ceiling))
        weighted_confidence_hint = self._bounded_confidence(signal.confidence * weight)
        base_metadata.update(
            {
                "strategy_weight": weight,
                "strategy_allocation": score_entry.allocation,
                "adaptive_strategy_score": score_entry.score,
                "strategy_manual_allocation": score_entry.manual_allocation,
                "strategy_regime_bias": score_entry.regime_bias,
                "weighted_confidence_hint": weighted_confidence_hint,
            }
        )
        base_metadata.setdefault("strategy_score", score_entry.score)
        return Signal(
            action=signal.action,
            symbol=signal.symbol,
            quantity=signal.quantity,
            confidence=signal.confidence,
            reason=signal.reason,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata=base_metadata,
        )

    def _recompute_scores(self) -> dict[str, StrategyScore]:
        """Compute and cache per-strategy score snapshots.

        Args:
            None.

        Returns:
            dict[str, StrategyScore]: Mapping of strategy name to scores.

        Raises:
            None.
        """
        self.refresh_regime_state()
        scores: dict[str, StrategyScore] = {}
        pnl_values: dict[str, float] = {}
        total_pnl_values: dict[str, float] = {}
        sharpe_values: dict[str, float] = {}
        win_values: dict[str, float] = {}
        drawdown_values: dict[str, float] = {}
        rolling_values: dict[str, float] = {}
        ratio_values: dict[str, float] = {}
        aggregate_snapshots: dict[str, dict[str, float]] = {}
        regime_pnl_values: dict[str, float] = {}
        regime_win_values: dict[str, float] = {}
        regime_sharpe_values: dict[str, float] = {}
        regime_drawdown_values: dict[str, float] = {}
        active_regime_snapshots: dict[str, dict[str, float]] = {}
        performances: dict[str, StrategyPerformance] = {}
        regime_key = (self._regime_state.regime or "").strip().lower()
        for strategy in self._strategies:
            name = strategy.name
            perf = self._performance.setdefault(name, StrategyPerformance())
            performances[name] = perf
            aggregate_snapshot = perf.snapshot()
            aggregate_snapshots[name] = aggregate_snapshot
            rolling_pnl_value = float(aggregate_snapshot.get("rolling_pnl", 0.0))
            pnl_values[name] = rolling_pnl_value
            total_pnl_values[name] = float(
                aggregate_snapshot.get("pnl", perf.total_pnl)
            )
            sharpe_values[name] = float(
                aggregate_snapshot.get("sharpe", perf.sharpe_ratio())
            )
            win_values[name] = float(
                aggregate_snapshot.get("win_rate", perf.win_rate())
            )
            drawdown_values[name] = -float(
                aggregate_snapshot.get("drawdown", perf.max_drawdown())
            )
            rolling_values[name] = rolling_pnl_value
            ratio_values[name] = perf.win_loss_ratio()
            bucket = None
            try:
                if regime_key:
                    bucket = perf.regime_buckets.get(regime_key)
                if bucket is None:
                    bucket = perf.regime_buckets.get("unknown")
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in StrategyManager regime bucket fetch: %s",
                    exc,
                    exc_info=exc,
                )
                bucket = None
            regime_stats_raw: dict[str, float] = {}
            if bucket is not None:
                regime_stats_raw = bucket.snapshot()
            active_snapshot = {
                "pnl": float(regime_stats_raw.get("pnl", 0.0)),
                "win_rate": float(regime_stats_raw.get("win_rate", 0.0)),
                "hit_rate": float(regime_stats_raw.get("win_rate", 0.0)),
                "sharpe": float(regime_stats_raw.get("sharpe", 0.0)),
                "drawdown": float(regime_stats_raw.get("drawdown", 0.0)),
                "trades": float(regime_stats_raw.get("trades", 0.0)),
            }
            active_regime_snapshots[name] = active_snapshot
            regime_pnl_values[name] = active_snapshot["pnl"]
            regime_win_values[name] = active_snapshot["win_rate"]
            regime_sharpe_values[name] = active_snapshot["sharpe"]
            regime_drawdown_values[name] = -active_snapshot["drawdown"]

        pnl_norm = self._normalise_metric(pnl_values)
        sharpe_norm = self._normalise_metric(sharpe_values)
        win_norm = self._normalise_metric(win_values)
        drawdown_norm = self._normalise_metric(drawdown_values)
        regime_pnl_norm = self._normalise_metric(regime_pnl_values)
        regime_sharpe_norm = self._normalise_metric(regime_sharpe_values)
        regime_win_norm = self._normalise_metric(regime_win_values)
        regime_drawdown_norm = self._normalise_metric(regime_drawdown_values)

        regime_bias = self._regime_bias_map.get(regime_key, {})
        regime_changed = regime_key != self._regime_last_key
        self._regime_last_key = regime_key
        confidence = self._regime_state.confidence
        base_smoothing = 0.55 if regime_changed else 0.35
        smoothing = max(0.2, min(0.9, base_smoothing + confidence * 0.3))
        for strategy in self._strategies:
            name = strategy.name
            manual = self._manual_allocations.get(name)
            base_score = (
                pnl_norm.get(name, 0.5) * self._score_weights.pnl
                + sharpe_norm.get(name, 0.5) * self._score_weights.sharpe
                + win_norm.get(name, 0.5) * self._score_weights.win_rate
                + drawdown_norm.get(name, 0.5) * self._score_weights.drawdown
            )
            regime_score = (
                regime_pnl_norm.get(name, 0.5) * self._score_weights.pnl
                + regime_sharpe_norm.get(name, 0.5) * self._score_weights.sharpe
                + regime_win_norm.get(name, 0.5) * self._score_weights.win_rate
                + regime_drawdown_norm.get(name, 0.5) * self._score_weights.drawdown
            )
            blend_weight = 0.0
            if regime_key:
                blend_weight = min(0.6, 0.25 + confidence * 0.35)
            composite_score = (
                base_score * (1.0 - blend_weight) + regime_score * blend_weight
            )
            composite_score = max(self._score_floor, float(composite_score))
            base_weight = manual if manual is not None else composite_score
            bias = float(regime_bias.get(name, 1.0) or 1.0)
            dynamic_bias = 1.0 + (bias - 1.0) * confidence
            performance = performances[name]
            trade_count = performance.wins + performance.losses
            rolling_pnl_value = rolling_values.get(name, 0.0)
            stability_factor = 1.0
            apply_penalties = manual is None and trade_count >= max(
                1, self._dynamic_trade_threshold
            )
            if rolling_pnl_value < 0 and apply_penalties:
                try:
                    penalty = min(
                        0.35,
                        abs(rolling_pnl_value) / (abs(performance.total_pnl) + 1.0),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in stability penalty calculation: %s",
                        exc,
                        exc_info=exc,
                    )
                    penalty = 0.0
                stability_factor -= penalty * 0.5
            ratio_value = ratio_values.get(name, 0.0)
            if ratio_value < 1.0 and apply_penalties:
                stability_factor -= min(0.2, (1.0 - ratio_value) * 0.3)
            drawdown_value = abs(drawdown_values.get(name, 0.0))
            if drawdown_value > 0 and apply_penalties:
                try:
                    stability_factor -= min(
                        0.25,
                        drawdown_value / (abs(performance.total_pnl) + 1.0) * 0.3,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in drawdown adjustment: %s",
                        exc,
                        exc_info=exc,
                    )
            stability_factor = max(0.25, stability_factor)
            weight = max(base_weight * dynamic_bias * stability_factor, 0.0)
            enabled = name not in self._disabled_strategies
            dynamic_flag = name in self._dynamic_disabled
            should_disable = (
                enabled
                and not dynamic_flag
                and manual is None
                and trade_count >= max(1, self._dynamic_trade_threshold)
                and confidence >= self._dynamic_confidence_floor
                and composite_score < self._dynamic_disable_threshold
                and rolling_pnl_value <= 0
                and drawdown_value > 0
            )
            should_enable = dynamic_flag and (
                composite_score >= self._dynamic_enable_threshold
                or rolling_pnl_value > 0
                or ratio_value >= 1.1
            )
            if should_disable:
                self._dynamic_disabled.add(name)
                enabled = False
                weight = 0.0
                log.info(
                    "Condition met: strategy_dynamic_disabled",
                    extra={
                        "event": "strategy_dynamic_disabled",
                        "strategy": name,
                        "regime": regime_key,
                        "score": round(composite_score, 4),
                        "threshold": self._dynamic_disable_threshold,
                        "rolling_pnl": round(rolling_pnl_value, 4),
                        "drawdown": round(drawdown_value, 4),
                    },
                )
                try:
                    METRICS.increment_strategy_allocation_change(
                        strategy=name,
                        direction="suspend",
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in dynamic disable metric: %s",
                        exc,
                        exc_info=exc,
                    )
            elif should_enable:
                self._dynamic_disabled.discard(name)
                enabled = name not in self._disabled_strategies
                log.info(
                    "Condition met: strategy_dynamic_restored",
                    extra={
                        "event": "strategy_dynamic_restored",
                        "strategy": name,
                        "regime": regime_key,
                        "score": round(composite_score, 4),
                        "threshold": self._dynamic_enable_threshold,
                    },
                )
                try:
                    METRICS.increment_strategy_allocation_change(
                        strategy=name,
                        direction="restore",
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in dynamic restore metric: %s",
                        exc,
                        exc_info=exc,
                    )
            elif dynamic_flag:
                enabled = False
                weight = 0.0
            if not enabled:
                weight = 0.0
            previous_allocation = self._allocation_state.get(name, weight)
            if manual is not None:
                allocation = max(manual, 0.0)
            else:
                allocation = max(
                    0.0, smoothing * weight + (1.0 - smoothing) * previous_allocation
                )
            self._allocation_state[name] = allocation
            if allocation <= 0.0 and previous_allocation > 0.0:
                log.info(
                    "Condition met: strategy_allocation_zero",
                    extra={
                        "event": "strategy_allocation_zero",
                        "strategy": name,
                        "previous": round(previous_allocation, 4),
                        "manual": manual is not None,
                        "dynamic_disabled": name in self._dynamic_disabled,
                    },
                )
            elif allocation > 0.0 and previous_allocation <= 0.0:
                log.info(
                    "Condition met: strategy_allocation_restored",
                    extra={
                        "event": "strategy_allocation_restored",
                        "strategy": name,
                        "allocation": round(allocation, 4),
                        "manual": manual is not None,
                        "dynamic_disabled": name in self._dynamic_disabled,
                    },
                )
            if abs(allocation - previous_allocation) > 0.05 or regime_changed:
                log.info(
                    "Condition met: strategy_allocation_updated",
                    extra={
                        "event": "strategy_allocation_updated",
                        "strategy": name,
                        "allocation": round(allocation, 4),
                        "previous": round(previous_allocation, 4),
                        "regime": regime_key,
                        "score": round(composite_score, 4),
                        "regime_score": round(regime_score, 4),
                        "confidence": round(confidence, 4),
                    },
                )
                try:
                    direction = "increase"
                    if allocation < previous_allocation:
                        direction = "decrease"
                    elif abs(allocation - previous_allocation) <= 1e-6:
                        direction = "hold"
                    METRICS.increment_strategy_allocation_change(
                        strategy=name,
                        direction=direction,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "Failure in strategy allocation change metric: %s",
                        exc,
                        exc_info=exc,
                    )
            try:
                METRICS.record_strategy_allocation(
                    strategy=name,
                    weight=allocation,
                    score=composite_score,
                    regime=regime_key or None,
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in StrategyManager allocation metrics: %s",
                    exc,
                    extra={
                        "event": "strategy_allocation_metric_error",
                        "strategy": name,
                    },
                )
            aggregate_snapshot = aggregate_snapshots.get(name, {})
            active_snapshot = active_regime_snapshots.get(name, {})
            scores[name] = StrategyScore(
                strategy=name,
                weight=weight,
                allocation=allocation,
                score=composite_score,
                regime_score=regime_score,
                pnl=total_pnl_values.get(name, 0.0),
                sharpe=sharpe_values.get(name, 0.0),
                win_rate=win_values.get(name, 0.0),
                drawdown=abs(drawdown_values.get(name, 0.0)),
                rolling_pnl=rolling_values.get(name, 0.0),
                win_loss_ratio=ratio_values.get(name, 0.0),
                manual_allocation=manual,
                regime_bias=dynamic_bias,
                enabled=enabled,
                active_regime_stats=active_snapshot,
                regime_breakdown=_regime_breakdown(performances[name]),
                dynamic_disabled=name in self._dynamic_disabled,
            )
            try:
                scores[name].active_regime_stats.setdefault(
                    "confidence", float(confidence)
                )
                scores[name].active_regime_stats.setdefault(
                    "regime", regime_key or "unknown"
                )
                scores[name].active_regime_stats.setdefault(
                    "hit_rate", scores[name].active_regime_stats.get("win_rate", 0.0)
                )
                scores[name].active_regime_stats.setdefault(
                    "score", round(composite_score, 4)
                )
                aggregate_snapshot.setdefault(
                    "hit_rate", aggregate_snapshot.get("win_rate", 0.0)
                )
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "Failure in StrategyManager score enrichment: %s",
                    exc,
                    exc_info=exc,
                )
        self._score_cache = scores
        return scores

    @staticmethod
    def _normalise_metric(
        values: dict[str, float],
        zero_is_neutral: bool = True,
    ) -> dict[str, float]:
        """Return normalised mapping with optional zero-neutral treatment.

        Args:
            values: Mapping of identifiers to metric values.
            zero_is_neutral: When ``True`` map zero to 0.5 with symmetric scaling.

        Returns:
            dict[str, float]: Normalised mapping with entries in [0, 1].

        Raises:
            None.
        """

        if not values:
            return {}
        try:
            all_values = list(values.values())
            if zero_is_neutral:
                positives = [value for value in all_values if value > 0]
                negatives = [value for value in all_values if value < 0]
                if positives and negatives:
                    max_pos = max(positives) or 1.0
                    min_neg = min(negatives) or -1.0

                    def _scale(value: float) -> float:
                        if value > 0 and max_pos > 0:
                            return 0.5 + (value / max_pos) * 0.5
                        if value < 0 and min_neg < 0:
                            return 0.5 + (value / abs(min_neg)) * 0.5
                        return 0.5

                    return {name: _scale(value) for name, value in values.items()}
            min_value = min(all_values)
            max_value = max(all_values)
            if max_value == min_value:
                return {name: 0.5 for name in values}
            span = max_value - min_value
            return {name: (value - min_value) / span for name, value in values.items()}
        except Exception as exc:  # noqa: BLE001
            log.error(
                "Failure in StrategyManager._normalise_metric: %s",
                exc,
                exc_info=exc,
                extra={"event": "strategy_metric_normalise_error"},
            )
            return {}


    @staticmethod
    def _extract_strike_from_symbol(symbol: str) -> int | None:
        """Extract option strike from symbol. Args: symbol. Returns: strike/None. Raises: none."""
        raw = str(symbol or "").strip().upper()
        if ":" in raw:
            raw = raw.split(":", 1)[1]
        match = re.search(r"(\d{4,6})(CE|PE)$", raw.replace("-", "").replace("_", ""))
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None


__all__ = [
    "StrategyManager",
    "StrategyScore",
    "StrategyScoreWeights",
    "StrategyPerformance",
    "RegimeState",
    "RegimePerformanceBucket",
]
