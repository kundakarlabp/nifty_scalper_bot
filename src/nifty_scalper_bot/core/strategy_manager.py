"""Strategy scoring and dynamic allocation manager.

Runtime role:
- Evaluates strategies using prepared DataHub/ActiveContractBasket context.
- Propagates selected option/futures context to strategy code.
- Must not select contracts or call broker instruments."""

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
from nifty_scalper_bot.instruments.active_contracts import canonical_nifty_future_symbol
from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy
from nifty_scalper_bot.strategies.signal_quality import infer_option_side
from nifty_scalper_bot.strategies.signal_generator import (
    Signal,
    build_strategy_history_context,
)
from nifty_scalper_bot.strategies.signal_generator import (
    StrategyManager as _BaseStrategyManager,
)
from nifty_scalper_bot.utils.log_throttle import log_throttled as log_throttled_live
from nifty_scalper_bot.utils.logging import get_logger, log_state_change, log_throttled
from nifty_scalper_bot.utils.symbols import normalize_symbol
from nifty_scalper_bot.execution.readiness import HistoryReadinessPolicy

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


def _safe_float_value(value: t.Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _normalise_ohlcv_bars(bars: t.Any) -> list[dict[str, float]]:
    normalised: list[dict[str, float]] = []
    rows: t.Iterable[t.Any]
    if bars is None:
        rows = []
    elif hasattr(bars, "tail") and hasattr(bars, "to_dict"):
        try:
            rows = bars.tail(100).to_dict("records")
        except Exception as exc:  # noqa: BLE001 - malformed provider data is non-fatal
            log.debug(
                "SMC_OHLCV_NORMALISE_DATAFRAME_FAILED error=%s",
                exc,
                extra={"event": "SMC_OHLCV_NORMALISE_DATAFRAME_FAILED", "error": str(exc)},
            )
            rows = []
    elif isinstance(bars, (list, tuple, deque)):
        rows = bars
    else:
        rows = []
    if not isinstance(rows, (list, tuple, deque)):
        rows = []
    for bar in rows:
        if not isinstance(bar, t.Mapping):
            continue
        raw_open = bar.get("open") if bar.get("open") is not None else bar.get("o")
        raw_high = bar.get("high") if bar.get("high") is not None else bar.get("h")
        raw_low = bar.get("low") if bar.get("low") is not None else bar.get("l")
        raw_close = bar.get("close") if bar.get("close") is not None else bar.get("c")
        raw_volume = bar.get("volume") if bar.get("volume") is not None else bar.get("v")
        open_v = _safe_float_value(raw_open)
        high_v = _safe_float_value(raw_high)
        low_v = _safe_float_value(raw_low)
        close_v = _safe_float_value(raw_close)
        volume_v = _safe_float_value(raw_volume)
        if open_v is None or high_v is None or low_v is None or close_v is None:
            continue
        normalised.append({"open": open_v, "high": high_v, "low": low_v, "close": close_v, "volume": float(volume_v if volume_v is not None else 0.0)})
    return normalised


def _extract_option_strike(symbol: str) -> int | None:
    match = re.search(r"(\d{4,6})(CE|PE)$", str(symbol or "").strip().upper())
    return int(match.group(1)) if match else None


def _enrich_option_candidate_metadata(symbol: str, indicators: t.Mapping[str, t.Any]) -> dict[str, t.Any]:
    enriched = dict(indicators or {})
    symbol_norm = str(symbol or "").strip().upper()
    selected_ce = str(enriched.get("selected_ce") or "").strip().upper()
    selected_pe = str(enriched.get("selected_pe") or "").strip().upper()
    selected_same_side = selected_ce if symbol_norm.endswith("CE") else selected_pe if symbol_norm.endswith("PE") else ""
    selected_symbols = {selected_ce, selected_pe}
    selected_symbols.discard("")
    if enriched.get("is_selected_option") is None:
        enriched["is_selected_option"] = bool(symbol_norm in selected_symbols) if selected_symbols else False
    if enriched.get("strike_distance_from_atm") is None:
        symbol_strike = _extract_option_strike(symbol_norm)
        atm_strike = _safe_float_value(enriched.get("atm_strike"))
        if symbol_strike is not None and atm_strike is not None:
            enriched["strike_distance_from_atm"] = abs(float(symbol_strike) - atm_strike)
        else:
            selected_strike = _extract_option_strike(selected_same_side)
            if symbol_strike is not None and selected_strike is not None:
                enriched["strike_distance_from_atm"] = abs(float(symbol_strike) - float(selected_strike))
    return enriched


def _enrich_smc_pre_strategy(
    symbol: str,
    indicators: t.Mapping[str, t.Any],
    bars: t.Sequence[t.Mapping[str, t.Any]],
) -> dict[str, t.Any]:
    enriched = dict(indicators or {})
    ohlcv = _normalise_ohlcv_bars(bars)
    bar_count = len(ohlcv)
    if bar_count < 20:
        enriched.setdefault("smc_enrichment_bars", bar_count)
        return enriched

    window = 5
    scan_start = max(0, bar_count - 100)
    latest_high: float | None = None
    latest_low: float | None = None
    for idx in range(bar_count - 1 - window, scan_start - 1, -1):
        left = max(0, idx - window)
        right = min(bar_count, idx + window + 1)
        segment = ohlcv[left:right]
        candidate_high = ohlcv[idx]["high"]
        candidate_low = ohlcv[idx]["low"]
        if latest_high is None and candidate_high == max(item["high"] for item in segment):
            latest_high = candidate_high
        if latest_low is None and candidate_low == min(item["low"] for item in segment):
            latest_low = candidate_low
        if latest_high is not None and latest_low is not None:
            break

    latest = ohlcv[-1]
    previous = ohlcv[-2]
    closes = [bar["close"] for bar in ohlcv[-5:]]
    symbol_side = "CE" if str(symbol).upper().endswith("CE") else "PE" if str(symbol).upper().endswith("PE") else ""
    direction_bias = str(enriched.get("direction_bias") or enriched.get("underlying_direction_bias") or "").upper()
    side = symbol_side if symbol_side in {"CE", "PE"} else direction_bias if direction_bias in {"CE", "PE"} else ""

    bos_side: str | None = None
    bos_confirmed = False
    if latest_high is not None and any(close > latest_high for close in closes):
        if side in {"", "CE"}:
            bos_confirmed = True
            bos_side = "CE"
    if latest_low is not None and any(close < latest_low for close in closes):
        if side in {"", "PE"}:
            bos_confirmed = True
            bos_side = "PE"

    choch_side: str | None = None
    choch_confirmed = False
    if direction_bias == "CE" and latest_low is not None and any(close < latest_low for close in closes):
        choch_confirmed = True
        choch_side = "PE"
    elif direction_bias == "PE" and latest_high is not None and any(close > latest_high for close in closes):
        choch_confirmed = True
        choch_side = "CE"

    premium_current = latest["close"]
    fallback_price = _safe_float_value(enriched.get("current_price") or enriched.get("ltp") or enriched.get("price"))
    if premium_current <= 0 and fallback_price is not None:
        premium_current = fallback_price
    premium_prev_close = previous["close"]
    premium_vwap = _safe_float_value(enriched.get("vwap"))
    if premium_vwap is None:
        premium_vwap = _safe_float_value(enriched.get("exchange_vwap"))
    if premium_vwap is None:
        volume_sum = sum(max(0.0, bar["volume"]) for bar in ohlcv)
        if volume_sum > 0:
            premium_vwap = sum(((bar["high"] + bar["low"] + bar["close"]) / 3.0) * max(0.0, bar["volume"]) for bar in ohlcv) / volume_sum
        else:
            premium_vwap = premium_current

    tolerance = max(abs(premium_current) * 0.003, 0.05)
    recent_bars = ohlcv[-5:]
    retest_confirmed = False
    if latest_high is not None and side in {"", "CE"}:
        retest_confirmed = any(abs(bar["low"] - latest_high) <= tolerance or abs(bar["close"] - latest_high) <= tolerance for bar in recent_bars)
    if not retest_confirmed and latest_low is not None and side in {"", "PE"}:
        retest_confirmed = any(abs(bar["high"] - latest_low) <= tolerance or abs(bar["close"] - latest_low) <= tolerance for bar in recent_bars)
    if not retest_confirmed and premium_vwap is not None:
        retest_confirmed = any(bar["low"] - tolerance <= premium_vwap <= bar["high"] + tolerance for bar in recent_bars)

    current_body = abs(latest["close"] - latest["open"])
    previous_body = abs(previous["close"] - previous["open"])
    body_floor = max(current_body, 0.05)
    lower_wick = min(latest["open"], latest["close"]) - latest["low"]
    upper_wick = latest["high"] - max(latest["open"], latest["close"])
    bullish_reversal = bool(latest["close"] > latest["open"] and current_body >= previous_body and lower_wick >= 0.3 * body_floor)
    bearish_reversal = bool(latest["close"] < latest["open"] and current_body >= previous_body and upper_wick >= 0.3 * body_floor)
    premium_reclaim = bool(premium_vwap is not None and premium_prev_close < premium_vwap and premium_current >= premium_vwap)

    derived: dict[str, t.Any] = {
        "swing_high": latest_high,
        "swing_low": latest_low,
        "bos_confirmed": bos_confirmed,
        "bos_side": bos_side,
        "choch_confirmed": choch_confirmed,
        "choch_side": choch_side,
        "retest_confirmed": retest_confirmed,
        "premium_current": premium_current,
        "premium_prev_close": premium_prev_close,
        "premium_vwap": premium_vwap,
        "premium_reclaim": premium_reclaim,
        "bullish_reversal": bullish_reversal,
        "bearish_reversal": bearish_reversal,
        "smc_enrichment_source": "strategy_manager_pre_strategy",
        "smc_enrichment_bars": bar_count,
        "smc_enrichment_lookahead_safe": True,
    }
    for key, value in derived.items():
        if enriched.get(key) is None:
            enriched[key] = value

    required = ("premium_reclaim", "bullish_reversal", "choch_confirmed", "bos_confirmed", "retest_confirmed")
    presence_ratio = sum(1 for name in required if enriched.get(name) is not None) / float(len(required))
    positive_features = [name for name in required if enriched.get(name) is True]
    positive_feature_count = len(positive_features)
    if enriched.get("feature_completeness") is None:
        enriched["feature_completeness"] = presence_ratio
    enriched["smc_feature_presence_ratio"] = presence_ratio
    enriched["smc_positive_feature_count"] = positive_feature_count
    enriched["smc_positive_features"] = positive_features
    log_throttled(
        log,
        f"smc_feature_enriched:{symbol}",
        "SMC_FEATURE_ENRICHED symbol=%s feature_completeness=%.3f presence_ratio=%.3f positive_feature_count=%s positive_features=%s swing_high=%s swing_low=%s premium_current=%s premium_vwap=%s premium_reclaim=%s bos_confirmed=%s choch_confirmed=%s retest_confirmed=%s",
        symbol,
        presence_ratio,
        presence_ratio,
        positive_feature_count,
        positive_features,
        enriched.get("swing_high"),
        enriched.get("swing_low"),
        enriched.get("premium_current"),
        enriched.get("premium_vwap"),
        enriched.get("premium_reclaim"),
        enriched.get("bos_confirmed"),
        enriched.get("choch_confirmed"),
        enriched.get("retest_confirmed"),
        interval_sec=30.0,
        level=logging.INFO,
        extra={"event": "SMC_FEATURE_ENRICHED", "symbol": symbol, "feature_completeness": presence_ratio, "presence_ratio": presence_ratio, "positive_feature_count": positive_feature_count, "positive_features": positive_features},
    )
    return enriched


@dataclass(frozen=True)
class StrategyNoSignalDecision:
    symbol: str
    eval_id: str | None
    final_block_reason: str | None
    category: str
    reason: str
    blocked_at: str
    no_vote_reason_counts: dict[str, int]
    strategy_reasons: dict[str, str]
    direction_bias: str | None
    underlying_direction_bias: str | None
    context_age_seconds: float | None
    trigger_vote_count: int
    context_vote_count: int
    selected_ce: str | None
    selected_pe: str | None
    trace_id: str | None = None
    created_ts: float = field(default_factory=time.time)


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
            "strategy_name": name,
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
        log.info(
            "RUNTIME_STRATEGY_MANAGER_CLASS strategy_manager_module=%s strategy_manager_class=%s strategy_manager_id=%s",
            self.__class__.__module__,
            self.__class__.__name__,
            id(self),
            extra={
                "event": "RUNTIME_STRATEGY_MANAGER_CLASS",
                "strategy_manager_module": self.__class__.__module__,
                "strategy_manager_class": self.__class__.__name__,
                "strategy_manager_id": id(self),
            },
        )
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
        self._last_no_signal_decision_by_symbol: dict[str, StrategyNoSignalDecision] = {}
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

    def _emit_smc_history_no_vote_summary(
        self,
        *,
        summary: dict[str, t.Any],
        indicators: t.Mapping[str, t.Any],
        window_seconds: float,
    ) -> None:
        total_no_votes = int(summary.get("total_no_votes", 0) or 0)
        if total_no_votes <= 0:
            return
        symbols = summary.get("symbols")
        symbol_count = len(symbols) if isinstance(symbols, set) else 0
        symbol_counts = summary.get("symbol_counts", {})
        top_symbols = sorted(symbol_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        log.info(
            "SMC_HISTORY_NO_VOTE_SUMMARY window_seconds=%s symbol_count=%s total_no_votes=%s reason_counts=%s",
            int(window_seconds),
            symbol_count,
            total_no_votes,
            summary.get("reason_counts", {}),
            extra={
                "event": "SMC_HISTORY_NO_VOTE_SUMMARY",
                "window_seconds": int(window_seconds),
                "symbol_count": symbol_count,
                "total_no_votes": total_no_votes,
                "reason_counts": dict(summary.get("reason_counts", {})),
                "history_domain_used_counts": dict(summary.get("domain_counts", {})),
                "top_symbols": top_symbols,
                "min_bars": int(os.getenv("SMC_MIN_BARS_REQUIRED", "30") or "30"),
                "live_mode": str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper() == "LIVE",
                "data_phase": indicators.get("data_phase"),
            },
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
        previous_snapshot = dict(self._latest_context_snapshots.get(role, {}) or {})

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

        derived_direction, derived_confidence, derived_reasons = self._derive_context_direction(indicators, role=role)
        context_kind = (
            "price_direction"
            if role == "spot_context"
            else "volume_flow"
            if role == "futures_context"
            else "unknown"
        )
        current_volume = _num("volume")
        current_avg_volume = _num("avg_volume")
        current_vwap = _num("vwap", "exchange_vwap", "session_vwap")
        current_close = _num("close", "ltp", "price")
        futures_volume_ratio = _num("futures_volume_ratio")
        futures_volume_ratio_source = "indicator"
        if role == "futures_context" and (futures_volume_ratio is None or futures_volume_ratio <= 0):
            if (
                current_volume is not None
                and current_avg_volume is not None
                and current_avg_volume > 0
            ):
                futures_volume_ratio = current_volume / max(current_avg_volume, 1e-9)
                futures_volume_ratio_source = "derived_volume_avg"
        vwap_slope = _num("vwap_slope")
        vwap_slope_source = "indicator"
        if role == "futures_context" and (vwap_slope is None or abs(vwap_slope) <= 1e-12):
            prev_vwap = previous_snapshot.get("vwap")
            try:
                prev_vwap_f = float(prev_vwap) if prev_vwap is not None else None
            except (TypeError, ValueError):
                prev_vwap_f = None
            if current_vwap is not None and prev_vwap_f and prev_vwap_f > 0:
                vwap_slope = (current_vwap - prev_vwap_f) / prev_vwap_f
                vwap_slope_source = "derived_previous_snapshot"
        ema_slope = _num("ema_slope")
        if role == "futures_context" and (ema_slope is None or abs(ema_slope) <= 1e-12):
            prev_close = previous_snapshot.get("close") or previous_snapshot.get("ltp")
            try:
                prev_close_f = float(prev_close) if prev_close is not None else None
            except (TypeError, ValueError):
                prev_close_f = None
            if current_close is not None and prev_close_f and prev_close_f > 0:
                ema_slope = (current_close - prev_close_f) / prev_close_f
        snapshot = {
            "symbol": symbol, "role": role, "context_kind": context_kind, "timestamp": time.time(),
            "ltp": _num("ltp", "close", "price"), "close": _num("close", "ltp", "price"),
            "vwap": _num("vwap", "exchange_vwap", "session_vwap"),
            "ema_fast": _num("ema_fast", "ema_9", "ema9"),
            "ema_slow": _num("ema_slow", "ema_21", "ema21"), "ema_50": _num("ema_50", "ema50"),
            "adx": _num("adx"), "atr": _num("atr"), "volume": _num("volume"),
            "avg_volume": _num("avg_volume"), "futures_volume_ratio": futures_volume_ratio,
            "vwap_slope": vwap_slope, "ema_slope": ema_slope,
            "direction_bias": derived_direction,
            "underlying_direction_bias": derived_direction,
            "underlying_direction_confidence": derived_confidence,
            "direction_context_reasons": derived_reasons,
            "context_timestamp_epoch": time.time(),
            "regime": indicators.get("regime") or indicators.get("market_regime"),
            "futures_volume_ratio_source": futures_volume_ratio_source,
            "vwap_slope_source": vwap_slope_source,
        }
        self._latest_context_snapshots[role] = snapshot
        if derived_direction not in {"CE", "PE"}:
            log_throttled(
                log,
                f"context_direction_unavailable:{role}:{symbol}",
                "CONTEXT_DIRECTION_UNAVAILABLE role=%s symbol=%s",
                role,
                symbol,
                interval_sec=30.0,
                level=logging.INFO,
                extra={"event": "CONTEXT_DIRECTION_UNAVAILABLE", "role": role, "symbol": symbol},
            )

    def _derive_context_direction(
        self, indicators: t.Mapping[str, t.Any], *, role: str
    ) -> tuple[str | None, float, list[str]]:
        def _f(key: str) -> float | None:
            value = indicators.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        explicit = str(indicators.get("direction_bias") or indicators.get("underlying_direction_bias") or "").upper()
        if explicit in {"CE", "PE"}:
            try:
                conf = float(indicators.get("underlying_direction_confidence") or indicators.get("direction_confidence") or 0.75)
            except (TypeError, ValueError):
                conf = 0.75
            return explicit, max(0.0, min(1.0, conf)), ["explicit_direction_bias"]
        close = _f("close") or _f("ltp") or _f("price")
        ltp = _f("ltp") or _f("last_price") or _f("price")
        previous_close = _f("previous_close") or _f("prev_close") or _f("previous_price")
        day_open = _f("open") or _f("day_open") or _f("first_ltp")
        recent_ltp_delta = _f("recent_ltp_delta") or _f("net_change") or _f("price_change_pct")
        tick_slope = _f("tick_slope")
        vwap = _f("vwap") or _f("exchange_vwap") or _f("session_vwap")
        ema_fast = _f("ema_fast") or _f("ema_9") or _f("ema9")
        ema_slow = _f("ema_slow") or _f("ema_21") or _f("ema21")
        ema_50 = _f("ema_50") or _f("ema50")
        vwap_slope = _f("vwap_slope") or 0.0
        ema_slope = _f("ema_slope") or 0.0
        futures_volume_ratio = _f("futures_volume_ratio") or 0.0
        ce_score = pe_score = 0.0
        reasons: list[str] = []
        if close is not None and vwap is not None and vwap > 0:
            if close >= vwap: ce_score += 1.0; reasons.append("close_above_vwap")
            else: pe_score += 1.0; reasons.append("close_below_vwap")
        if ema_fast is not None and ema_slow is not None:
            if ema_fast >= ema_slow: ce_score += 1.0; reasons.append("ema_fast_above_slow")
            else: pe_score += 1.0; reasons.append("ema_fast_below_slow")
        if close is not None and ema_50 is not None:
            if close >= ema_50: ce_score += 0.5; reasons.append("close_above_ema50")
            else: pe_score += 0.5; reasons.append("close_below_ema50")
        if vwap_slope > 0: ce_score += 0.5; reasons.append("vwap_slope_positive")
        elif vwap_slope < 0: pe_score += 0.5; reasons.append("vwap_slope_negative")
        if ema_slope > 0: ce_score += 0.5; reasons.append("ema_slope_positive")
        elif ema_slope < 0: pe_score += 0.5; reasons.append("ema_slope_negative")
        if role == "futures_context" and futures_volume_ratio >= 1.0:
            reasons.append("futures_volume_active")
        if close is not None and previous_close is not None:
            if close > previous_close:
                ce_score += 0.8
                reasons.append("close_above_previous_close")
            elif close < previous_close:
                pe_score += 0.8
                reasons.append("close_below_previous_close")
        if ltp is not None and day_open is not None:
            if ltp > day_open:
                ce_score += 0.7
                reasons.append("ltp_above_open")
            elif ltp < day_open:
                pe_score += 0.7
                reasons.append("ltp_below_open")
        delta_signal = recent_ltp_delta if recent_ltp_delta not in (None, 0.0) else tick_slope
        if delta_signal is not None:
            if delta_signal > 0:
                ce_score += 0.6
                reasons.append("tick_slope_positive")
            elif delta_signal < 0:
                pe_score += 0.6
                reasons.append("tick_slope_negative")
        if (
            ce_score + pe_score <= 0
            and ltp is not None
            and close is not None
            and close > 0
        ):
            ltp_close_delta_pct = (ltp - close) / close * 100.0
            min_delta_pct = self._env_float(
                "STRATEGY_CONTEXT_MIN_LTP_CLOSE_DELTA_PCT", 0.03
            )
            if ltp_close_delta_pct >= min_delta_pct:
                ce_score += 0.5
                reasons.append("ltp_above_close_fallback")
            elif ltp_close_delta_pct <= -min_delta_pct:
                pe_score += 0.5
                reasons.append("ltp_below_close_fallback")
                
        total = ce_score + pe_score
        if total <= 0:
            return None, 0.0, ["direction_unavailable"]
        margin = abs(ce_score - pe_score)
        if margin < 0.5:
            return None, min(0.55, total / 4.0), reasons + ["direction_tie"]
        side = "CE" if ce_score > pe_score else "PE"
        strong_reason_tags = {
            "close_above_vwap",
            "close_below_vwap",
            "ema_fast_above_slow",
            "ema_fast_below_slow",
            "vwap_slope_positive",
            "vwap_slope_negative",
            "ema_slope_positive",
            "ema_slope_negative",
            "close_above_ema50",
            "close_below_ema50",
        }
        has_strong_reasons = any(tag in strong_reason_tags for tag in reasons)
        raw_confidence = 0.50 + margin / max(total, 1.0) * 0.45
        if has_strong_reasons:
            confidence = min(0.95, max(0.50, raw_confidence))
        else:
            confidence = min(0.70, max(0.55, raw_confidence))
        return side, confidence, reasons

    @staticmethod
    def _context_direction_valid(ctx: t.Mapping[str, t.Any]) -> bool:
        return str(ctx.get("direction_bias") or ctx.get("underlying_direction_bias") or "").upper() in {"CE", "PE"}

    @staticmethod
    def _context_tick_age_seconds(ctx: t.Mapping[str, t.Any]) -> float | None:
        raw = ctx.get("tick_age_ms")
        if raw is None:
            return None
        try:
            age_ms = float(raw)
        except (TypeError, ValueError):
            return None
        return age_ms / 1000.0 if age_ms >= 0 else None

    def _strategy_required_indicator_union(self) -> set[str]:
        required = set(getattr(self, "_required_indicators", set()) or set())
        try:
            for names in getattr(self, "_strategy_required_indicators", {}).values():
                required.update(str(name) for name in names if name)
        except Exception as exc:
            log.warning("STRATEGY_REQUIRED_INDICATOR_UNION_FAILED error=%s", exc, extra={"event": "STRATEGY_REQUIRED_INDICATOR_UNION_FAILED"})
        required.update({"symbol","ltp","price","close","open","high","low","vwap","exchange_vwap","atr","volume","avg_volume","direction_bias","underlying_direction_bias","underlying_direction_confidence","context_age_seconds","futures_volume_ratio","futures_vwap","futures_vwap_slope","ema_fast","ema_slow","ema_50","ema_slope","vwap_slope","bos_confirmed","choch_confirmed","retest_confirmed","premium_reclaim","bullish_reversal","liquidity_sweep_confirmed","liquidity_sweep_confirmed_bear","prior_swing_low","prior_swing_high","spread_pct","bid","ask","selected_ce","selected_pe","is_selected_option","strike_distance_from_atm","data_age_seconds","stale_data_used"})
        return required

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
        symbol_norm = str(symbol or "").strip().upper()
        self._last_no_signal_decision_by_symbol.pop(symbol_norm, None)
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
        signals: list[Signal] = []
        signal_votes: list[tuple[Signal, StrategyVote]] = []
        disabled: list[str] = []
        empty: list[str] = []
        no_vote_reason_counts: dict[str, int] = {}
        errors: list[str] = []
        disabled_strategies_snapshot: list[str] = []
        signal_action: str | None = None
        signal_score: float | None = None
        signal_confidence: float | None = None
        exit_result = "no_signal"
        policy = HistoryReadinessPolicy.from_env()
        required_bars = int(getattr(self, "_required_candles", policy.option_eval_min_bars) or policy.option_eval_min_bars)
        bars_available = len(self._indicator_engine.get_history(symbol) or [])
        indicators_ready = bars_available >= required_bars
        hub_ready = None
        if self._data_hub is not None:
            hub_ready = bool(getattr(self._data_hub, "indicators_ready", False))
        required_for_eval = self._strategy_required_indicator_union()
        indicators_raw = self._indicator_engine.get_indicators(symbol, required_for_eval)
        if isinstance(indicators_raw, dict):
            indicators: dict[str, t.Any] = dict(indicators_raw)
        elif hasattr(indicators_raw, "items"):
            indicators = dict(indicators_raw)
        else:
            indicators = {}
            log_throttled(
                log,
                f"strategy_empty_indicators:{symbol}",
                "STRATEGY_EMPTY_INDICATORS symbol=%s",
                symbol,
                interval_sec=30.0,
                level=logging.INFO,
                extra={
                    "event": "STRATEGY_EMPTY_INDICATORS",
                    "symbol": symbol,
                    "required_indicator_count": len(required_for_eval),
                    "indicators_raw_type": type(indicators_raw).__name__,
                },
            )
        symbol_role = classify_symbol_role(symbol)
        indicators["symbol_role"] = symbol_role
        history_ctx = build_strategy_history_context(
            symbol=symbol,
            indicator_engine=self._indicator_engine,
            data_hub=self._data_hub,
            runner_context=indicators,
        )
        indicators.update(history_ctx)
        vwap = indicators.get("vwap")
        avg_volume = indicators.get("avg_volume")
        required_missing = sorted(
            name for name in self._required_indicators if indicators.get(name) is None
        )
        strategy_missing = sorted(name for name in required_for_eval if indicators.get(name) is None and name not in {"symbol", "selected_ce", "selected_pe"})
        log_throttled(
            log,
            f"strategy_missing_indicators:{symbol}",
            "STRATEGY_REQUIRED_FIELDS_MISSING symbol=%s count=%s",
            symbol,
            len(strategy_missing),
            interval_sec=30.0,
            level=logging.DEBUG,
            extra={
                "event": "STRATEGY_REQUIRED_FIELDS_MISSING",
                "symbol": symbol,
                "strategy_missing": strategy_missing[:60],
            },
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
                "required_indicator_count": len(required_for_eval),
                "strategy_required_indicator_count": len(required_for_eval - set(self._required_indicators)),
                "required_indicators_sample": sorted(required_for_eval)[:40],
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
        regime_allowed: bool | None = None
        regime_reasons: tuple[str, ...] = ()
        if regime_manager is not None:
            gate_context = {"component": "strategy_manager", "symbol": symbol}
            try:
                allowed = regime_manager.can_trade(context=gate_context)
                reasons = tuple(regime_manager.get_filter_reasons())
                regime_allowed = bool(allowed)
                regime_reasons = tuple(reasons)
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
                    if symbol_role in {"spot_context", "futures_context"}:
                        log_throttled(
                            log,
                            f"context_regime_gate_bypassed:{symbol}",
                            (
                                "CONTEXT_REGIME_GATE_BYPASSED "
                                "symbol=%s role=%s reason=context_snapshot_update_not_trade_entry"
                            ),
                            symbol,
                            symbol_role,
                            interval_sec=60.0,
                            level=logging.INFO,
                            extra={
                                "event": "CONTEXT_REGIME_GATE_BYPASSED",
                                "symbol": symbol,
                                "symbol_role": symbol_role,
                                "gate_reasons": list(reasons),
                                "reason": "context_snapshot_update_not_trade_entry",
                            },
                        )
                    else:
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
                    regime_allowed is False
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
        indicators.setdefault("ltp", current_price)
        indicators.setdefault("price", current_price)
        indicators.setdefault("close", current_price)
        indicators["symbol_role"] = symbol_role
        indicators["_regime_adjustments"] = adjustments
        self._augment_futures_metrics(indicators)
        # ✅ FIX S3: Inject exchange VWAP for options
        if self._data_hub is not None:
            try:
                opt_quote = _get_cached_quote_for_eval(self._data_hub, symbol)
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
                extra={
                    "event": "CONTEXT_SYMBOL_STRATEGY_EVAL_SKIPPED",
                    "symbol": symbol,
                    "symbol_role": symbol_role,
                    "reason": "context_only_no_trade_strategy_eval",
                },
            )
            return None
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
        if symbol_role == "tradable_option":
            context_snapshots = getattr(self, "_latest_context_snapshots", {})
            spot_ctx = context_snapshots.get("spot_context", {})
            fut_ctx = context_snapshots.get("futures_context", {})
            active_fut = self._resolve_active_futures_symbol_for_metrics()
            fut_ctx_symbol = canonical_nifty_future_symbol((fut_ctx or {}).get("symbol") if isinstance(fut_ctx, dict) else None)
            active_fut_canonical = canonical_nifty_future_symbol(active_fut)
            if fut_ctx_symbol and active_fut_canonical and fut_ctx_symbol != active_fut_canonical:
                fut_ctx = {}
                indicators.setdefault("context_discard_reasons", []).append("stale_futures_context_discarded")
                log_throttled(
                    log,
                    f"stale_futures_context_discarded:{symbol}",
                    "STALE_FUTURES_CONTEXT_DISCARDED symbol=%s stale=%s active=%s",
                    symbol,
                    fut_ctx_symbol,
                    active_fut_canonical,
                    interval_sec=60.0,
                    level=logging.WARNING,
                    extra={"event": "STALE_FUTURES_CONTEXT_DISCARDED", "symbol": symbol, "stale_symbol": fut_ctx_symbol, "active_symbol": active_fut_canonical},
                )
            max_context_age = self._live_context_max_age_seconds()
            now_ts = time.time()
            def _fresh(ctx: t.Mapping[str, t.Any]) -> bool:
                try:
                    ts = float(ctx.get("timestamp") or ctx.get("context_timestamp_epoch") or 0.0)
                    return ts > 0 and (now_ts - ts) <= max_context_age
                except (TypeError, ValueError):
                    return False
            spot_fresh = _fresh(spot_ctx)
            fut_fresh = _fresh(fut_ctx)
            spot_direction_valid = self._context_direction_valid(spot_ctx)
            fut_direction_valid = self._context_direction_valid(fut_ctx)
            spot_tick_age_s = self._context_tick_age_seconds(spot_ctx)
            fut_tick_age_s = self._context_tick_age_seconds(fut_ctx)
            spot_usable = spot_fresh and spot_direction_valid
            fut_usable = fut_fresh and fut_direction_valid
            if spot_fresh and not spot_direction_valid:
                spot_direction_reasons = spot_ctx.get("direction_context_reasons")
                log_throttled(
                    log, f"option_context_fresh_but_directionless:{symbol}",
                    "OPTION_CONTEXT_FRESH_BUT_DIRECTIONLESS symbol=%s spot_fresh=%s spot_direction_bias=%s spot_underlying_direction_bias=%s spot_confidence=%s spot_direction_reasons=%s spot_ctx_keys=%s",
                    symbol, spot_fresh, spot_ctx.get("direction_bias"), spot_ctx.get("underlying_direction_bias"), spot_ctx.get("underlying_direction_confidence"), spot_direction_reasons, sorted(list(spot_ctx.keys())),
                    interval_sec=30.0, level=logging.INFO,
                    extra={"event": "OPTION_CONTEXT_FRESH_BUT_DIRECTIONLESS", "symbol": symbol, "spot_fresh": spot_fresh, "spot_direction_valid": spot_direction_valid, "spot_ctx_keys": sorted(list(spot_ctx.keys())), "spot_direction_bias": spot_ctx.get("direction_bias"), "spot_underlying_direction_bias": spot_ctx.get("underlying_direction_bias"), "spot_confidence": spot_ctx.get("underlying_direction_confidence"), "spot_direction_reasons": spot_direction_reasons, "spot_ltp": spot_ctx.get("ltp"), "spot_close": spot_ctx.get("close"), "spot_vwap": spot_ctx.get("vwap"), "spot_ema_fast": spot_ctx.get("ema_fast"), "spot_ema_slow": spot_ctx.get("ema_slow"), "spot_ema_50": spot_ctx.get("ema_50"), "spot_vwap_slope": spot_ctx.get("vwap_slope"), "spot_ema_slope": spot_ctx.get("ema_slope"), "fut_fresh": fut_fresh, "fut_direction_valid": fut_direction_valid, "fut_direction_bias": fut_ctx.get("direction_bias"), "fut_underlying_direction_bias": fut_ctx.get("underlying_direction_bias"), "fut_confidence": fut_ctx.get("underlying_direction_confidence"), "fut_direction_reasons": fut_ctx.get("direction_context_reasons"), "fut_ctx_keys": sorted(list(fut_ctx.keys())), "fut_ltp": fut_ctx.get("ltp"), "fut_close": fut_ctx.get("close"), "fut_vwap": fut_ctx.get("vwap"), "fut_ema_fast": fut_ctx.get("ema_fast"), "fut_ema_slow": fut_ctx.get("ema_slow"), "fut_ema_50": fut_ctx.get("ema_50"), "fut_vwap_slope": fut_ctx.get("vwap_slope"), "fut_ema_slope": fut_ctx.get("ema_slope")},
                )
            if spot_usable:
                indicators.setdefault("spot_context", spot_ctx)
            if fut_usable:
                indicators.setdefault("futures_context", fut_ctx)
            direction_bias = (indicators.get("direction_bias") or indicators.get("underlying_direction_bias") or (spot_ctx.get("direction_bias") if spot_usable else None) or (fut_ctx.get("direction_bias") if fut_usable else None))
            direction_confidence = (indicators.get("underlying_direction_confidence") or (spot_ctx.get("underlying_direction_confidence") if spot_usable else None) or (fut_ctx.get("underlying_direction_confidence") if fut_usable else None) or 0.0)
            context_resolved = False
            direction_context_source: str | None = None
            if str(direction_bias or "").upper() in {"CE", "PE"}:
                indicators["direction_bias"] = str(direction_bias).upper()
                indicators["underlying_direction_bias"] = str(direction_bias).upper()
                try:
                    indicators["underlying_direction_confidence"] = float(direction_confidence)
                except (TypeError, ValueError):
                    indicators["underlying_direction_confidence"] = 0.0
                indicators["context_age_seconds"] = min(now_ts - float(spot_ctx.get("timestamp", now_ts)) if spot_usable else max_context_age + 1, now_ts - float(fut_ctx.get("timestamp", now_ts)) if fut_usable else max_context_age + 1)
                indicators["context_fresh"] = bool(float(indicators.get("context_age_seconds") or max_context_age + 1) <= max_context_age)
                indicators["direction_context_source"] = "primary"
                context_resolved = bool(indicators["context_fresh"])
                direction_context_source = "primary"
                log_throttled(
                    log,
                    f"direction_context_resolved_primary:{symbol}",
                    "DIRECTION_CONTEXT_RESOLVED source=primary bias=%s confidence=%.2f age_s=%.2f symbol=%s",
                    indicators["direction_bias"],
                    float(indicators.get("underlying_direction_confidence") or 0.0),
                    float(indicators.get("context_age_seconds") or 999.0),
                    symbol,
                    interval_sec=30.0,
                    level=logging.INFO,
                )
            else:
                fallback_ctx = spot_ctx if spot_fresh else fut_ctx if fut_fresh else {}
                fallback_direction, fallback_conf, fallback_reasons = self._derive_context_direction(
                    fallback_ctx,
                    role=str(fallback_ctx.get("role") or "spot_context"),
                )
                if str(fallback_direction or "").upper() in {"CE", "PE"}:
                    fallback_age = (
                        min(v for v in [spot_tick_age_s, fut_tick_age_s] if v is not None)
                        if any(v is not None for v in [spot_tick_age_s, fut_tick_age_s])
                        else (
                            now_ts - float(fallback_ctx.get("timestamp", now_ts))
                            if fallback_ctx
                            else max_context_age + 1
                        )
                    )
                    indicators["direction_bias"] = str(fallback_direction).upper()
                    indicators["underlying_direction_bias"] = str(fallback_direction).upper()
                    indicators["underlying_direction_confidence"] = float(max(fallback_conf, 0.05))
                    indicators["context_age_seconds"] = float(max(0.0, fallback_age))
                    indicators["context_fresh"] = bool(indicators["context_age_seconds"] <= max_context_age)
                    indicators["direction_context_source"] = "fallback"
                    indicators["direction_context_reasons"] = fallback_reasons
                    direction_context_source = "fallback"
                    context_resolved = bool(indicators["context_fresh"])
                    log_throttled(
                        log,
                        f"direction_context_resolved_fallback:{symbol}",
                        "DIRECTION_CONTEXT_RESOLVED source=fallback bias=%s confidence=%.2f age_s=%.2f symbol=%s reasons=%s",
                        indicators["direction_bias"],
                        float(indicators.get("underlying_direction_confidence") or 0.0),
                        float(indicators.get("context_age_seconds") or 999.0),
                        symbol,
                        fallback_reasons,
                        interval_sec=30.0,
                        level=logging.INFO,
                    )
                if not context_resolved:
                    log_throttled(
                        log,
                        f"option_underlying_context_missing:{symbol}",
                        "OPTION_UNDERLYING_CONTEXT_MISSING symbol=%s spot_ctx_present=%s futures_ctx_present=%s spot_ctx_age=%s futures_ctx_age=%s selected_ce=%s selected_pe=%s",
                        symbol,
                        bool(spot_ctx),
                        bool(fut_ctx),
                        (now_ts - float(spot_ctx.get("timestamp", now_ts))) if spot_ctx else None,
                        (now_ts - float(fut_ctx.get("timestamp", now_ts))) if fut_ctx else None,
                        indicators.get("selected_ce"),
                        indicators.get("selected_pe"),
                        interval_sec=30.0,
                        level=logging.INFO,
                    )
                    log_throttled(
                        log,
                        f"option_context_missing_direction_bias:{symbol}",
                        "OPTION_CONTEXT_MISSING_DIRECTION_BIAS symbol=%s spot_fresh=%s fut_fresh=%s spot_direction_valid=%s fut_direction_valid=%s",
                        symbol,
                        spot_fresh,
                        fut_fresh,
                        spot_direction_valid,
                        fut_direction_valid,
                        interval_sec=30.0,
                        level=logging.INFO,
                        extra={
                            "event": "OPTION_CONTEXT_MISSING_DIRECTION_BIAS",
                            "symbol": symbol,
                            "spot_fresh": spot_fresh,
                            "fut_fresh": fut_fresh,
                            "spot_direction_valid": spot_direction_valid,
                            "fut_direction_valid": fut_direction_valid,
                            "spot_direction_reasons": spot_ctx.get("direction_context_reasons"),
                            "fut_direction_reasons": fut_ctx.get("direction_context_reasons"),
                            "spot_ctx_keys": sorted(list(spot_ctx.keys())),
                            "fut_ctx_keys": sorted(list(fut_ctx.keys())),
                            "spot_tick_age_ms": spot_ctx.get("tick_age_ms"),
                            "futures_tick_age_ms": fut_ctx.get("tick_age_ms"),
                            "direction_context_source": direction_context_source,
                        },
                    )
            if fut_fresh and fut_ctx.get("futures_volume_ratio") is not None:
                indicators.setdefault("futures_volume_ratio", fut_ctx.get("futures_volume_ratio"))
            if fut_fresh and fut_ctx.get("vwap") is not None:
                indicators.setdefault("futures_vwap", fut_ctx.get("vwap"))
            if fut_fresh and fut_ctx.get("vwap_slope") is not None:
                indicators.setdefault("futures_vwap_slope", fut_ctx.get("vwap_slope"))

        if symbol_role == "tradable_option":
            indicators = _enrich_option_candidate_metadata(symbol, indicators)
            recent_bars: list[dict[str, float]] = []
            for owner, method_name in (
                (self._data_hub, "get_ohlc_bars"),
                (self._indicator_engine, "get_history"),
                (self._indicator_engine, "get_bars"),
            ):
                if recent_bars or owner is None:
                    continue
                method = getattr(owner, method_name, None)
                if not callable(method):
                    continue
                provider_name = owner.__class__.__name__
                raw_bars: t.Any = []
                try:
                    try:
                        raw_bars = method(symbol, limit=100) if method_name == "get_ohlc_bars" else method(symbol)
                    except TypeError:
                        raw_bars = method(symbol)
                except Exception as exc:  # noqa: BLE001 - provider failure must not crash strategy evaluation
                    raw_bars = []
                    log_throttled(
                        log,
                        f"smc_history_provider_failed:{symbol}:{provider_name}:{method_name}",
                        "SMC_HISTORY_PROVIDER_FAILED symbol=%s provider=%s method=%s error=%s",
                        symbol,
                        provider_name,
                        method_name,
                        str(exc),
                        interval_sec=30.0,
                        level=logging.WARNING,
                        extra={
                            "event": "SMC_HISTORY_PROVIDER_FAILED",
                            "symbol": symbol,
                            "provider": provider_name,
                            "method": method_name,
                            "error": str(exc),
                        },
                    )
                recent_bars = _normalise_ohlcv_bars(raw_bars)[-100:]
            indicators = _enrich_smc_pre_strategy(symbol, indicators, recent_bars)

        position = self._position_manager.get_position(symbol)

        max_votes = max(1, int(getattr(app_settings, "MAX_STRATEGY_VOTES", 5)))
        disabled_strategies_snapshot = disabled
        evaluation_start = time.monotonic()
        symbol_upper = symbol.upper()
        symbol_is_option = bool(re.search(r"(CE|PE)$", symbol_upper))
        symbol_is_future = symbol_upper.endswith("FUT")
        direction_bias = str(indicators.get("direction_bias") or "").upper()
        eval_id = f"{symbol}:{int(time.time())}"
        indicators.setdefault("eval_id", eval_id)
        strategy_reasons: dict[str, str] = {}
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
                strategy_reasons[strategy_name] = "context_symbol_skipped_for_option_strategy"
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
                strategy_reasons[strategy_name] = "context_symbol_skipped_for_underlying_strategy"
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
                if strategy_name == "SMC":
                    now = time.monotonic()
                    last_diag_map = getattr(self, "_smc_history_diag_last_emitted", {})
                    if not isinstance(last_diag_map, dict):
                        last_diag_map = {}
                    ready_for_smc = bool(indicators.get("history_ready_for_smc"))
                    interval_sec = 30.0 if not ready_for_smc else 60.0
                    should_emit_diag = now - float(last_diag_map.get(symbol, 0.0) or 0.0) >= interval_sec
                    if should_emit_diag:
                        last_diag_map[symbol] = now
                        self._smc_history_diag_last_emitted = last_diag_map
                        log.debug(
                            "SMC_HISTORY_INPUT_DIAGNOSTICS symbol=%s eval_id=%s history_domain_used=%s history_count=%s history_resolved_count=%s option_history_count=%s",
                            symbol,
                            eval_id,
                            indicators.get("history_domain_used"),
                            indicators.get("history_count"),
                            indicators.get("history_resolved_count"),
                            indicators.get("option_history_count"),
                            extra={
                                "event": "SMC_HISTORY_INPUT_DIAGNOSTICS",
                                "symbol": symbol,
                                "eval_id": eval_id,
                                "history_domain_used": indicators.get("history_domain_used"),
                                "history_count": indicators.get("history_count"),
                                "history_resolved_count": indicators.get("history_resolved_count"),
                                "indicator_history_count": indicators.get("indicator_history_count"),
                                "option_history_count": indicators.get("option_history_count"),
                                "spot_history_count": indicators.get("spot_history_count"),
                                "underlying_history_count": indicators.get("underlying_history_count"),
                                "history_source": indicators.get("history_source"),
                                "history_symbol_key": indicators.get("history_symbol_key"),
                                "history_ready_for_smc": indicators.get("history_ready_for_smc"),
                                "history_quality": indicators.get("history_quality"),
                                "history_required_min": indicators.get("history_required_min"),
                                "oldest_bar_ts": indicators.get("oldest_bar_ts"),
                                "latest_bar_ts": indicators.get("latest_bar_ts"),
                                "available_indicator_keys_count": len(indicators),
                                "has_open": indicators.get("open") is not None,
                                "has_high": indicators.get("high") is not None,
                                "has_low": indicators.get("low") is not None,
                                "has_close": indicators.get("close") is not None,
                                "has_vwap": indicators.get("vwap") is not None,
                                "has_volume": indicators.get("volume") is not None,
                                "data_phase": indicators.get("data_phase"),
                                "quote_update_version": indicators.get("quote_update_version"),
                                "live_candle_version": indicators.get("live_candle_version"),
                            },
                        )
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
                strategy_reasons[strategy.name] = reason
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

        if no_vote_reason_counts:
            log.debug(
                "STRATEGY_NO_VOTE_SUMMARY symbol=%s eval_id=%s no_vote_reason_counts=%s strategy_reasons=%s trigger_vote_count=%s context_vote_count=%s final_block_reason=%s",
                symbol,
                eval_id,
                no_vote_reason_counts,
                strategy_reasons,
                len(signals),
                0,
                "no_strategy_signal" if not signals else "partial",
                extra={"event": "STRATEGY_NO_VOTE_SUMMARY", "symbol": symbol, "eval_id": eval_id, "no_vote_reason_counts": no_vote_reason_counts, "strategy_reasons": strategy_reasons, "trigger_vote_count": len(signals), "context_vote_count": 0, "final_block_reason": "no_strategy_signal" if not signals else "partial"},
            )
            smc_reasons = {"smc_history_count_missing", "smc_insufficient_history"}
            smc_reason_counts = {k: v for k, v in no_vote_reason_counts.items() if k in smc_reasons}
            if smc_reason_counts:
                now = time.monotonic()
                win = 30.0
                summary = getattr(self, "_smc_history_no_vote_summary", None)
                if isinstance(summary, dict):
                    window_start = float(summary.get("window_start", now) or now)
                    if now - window_start >= win:
                        self._emit_smc_history_no_vote_summary(
                            summary=summary,
                            indicators=indicators,
                            window_seconds=win,
                        )
                        summary = None
                if not isinstance(summary, dict):
                    summary = {"window_start": now, "symbols": set(), "total_no_votes": 0, "reason_counts": {}, "domain_counts": {}, "symbol_counts": {}}
                    self._smc_history_no_vote_summary = summary
                summary["symbols"].add(symbol)
                domain = str(indicators.get("history_domain_used") or "unknown")
                summary["domain_counts"][domain] = int(summary["domain_counts"].get(domain, 0)) + sum(smc_reason_counts.values())
                for reason, count in smc_reason_counts.items():
                    summary["reason_counts"][reason] = int(summary["reason_counts"].get(reason, 0)) + int(count)
                    summary["total_no_votes"] = int(summary["total_no_votes"]) + int(count)
                    summary["symbol_counts"][symbol] = int(summary["symbol_counts"].get(symbol, 0)) + int(count)
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
            primary_reason = "no_strategy_signal"
            category = "strategy_no_trigger"
            if no_vote_reason_counts.get("underlying_direction_conflict"):
                primary_reason = "underlying_direction_conflict"
                category = "context_direction_conflict"
            elif no_vote_reason_counts.get("tick_direction_missing_or_neutral"):
                primary_reason = "tick_direction_missing_or_neutral"
                category = "option_tick_direction_neutral"
            elif no_vote_reason_counts.get("negative_premium_flow"):
                primary_reason = "negative_premium_flow"
                category = "premium_flow_negative"
            elif no_vote_reason_counts.get("raw_score_below_min"):
                primary_reason = "raw_score_below_min"
                category = "strategy_score_below_threshold"
            elif no_vote_reason_counts.get("weak_score"):
                primary_reason = "weak_score"
                category = "strategy_score_below_threshold"
            elif no_vote_reason_counts.get("single_vote_scalp_disabled"):
                primary_reason = "single_vote_scalp_disabled"
                category = "strategy_single_vote_disabled"
            elif no_vote_reason_counts.get("not_selected_or_near_atm"):
                primary_reason = "not_selected_or_near_atm"
                category = "candidate_not_selected_or_near_atm"
            elif any(no_vote_reason_counts.get(r) for r in ("no_liquidity_sweep", "premium_not_reversing_up", "smc_structure_required_live")):
                primary_reason = next((r for r in ("no_liquidity_sweep", "premium_not_reversing_up", "smc_structure_required_live") if no_vote_reason_counts.get(r)), "no_strategy_signal")
                category = "strategy_structure_not_confirmed" if primary_reason == "smc_structure_required_live" else "strategy_no_trigger"
            elif errors:
                primary_reason = "strategy_error"
                category = "strategy_error"
            self._record_no_signal_decision(
                symbol=symbol_norm,
                category=category,
                reason=primary_reason,
                blocked_at="generate_signal_no_signals",
                indicators=indicators,
                no_vote_reason_counts=no_vote_reason_counts,
                strategy_reasons=strategy_reasons,
                trigger_vote_count=0,
                context_vote_count=0,
                trace_id=trace_id,
                eval_id=eval_id,
                final_block_reason="no_strategy_signal",
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
            blocker_reason = "strategy_no_vote_present:" + ",".join(sorted(str(k) for k in no_vote_reason_counts))
            self._log_strategy_combiner_blocker(
                symbol=symbol,
                signal_votes=signal_votes,
                indicators=indicators,
                combined=None,
                blocked_reason=blocker_reason,
                no_vote_reason_counts=no_vote_reason_counts,
            )
            no_signal_reasons.append("no_strategy_signal")
            _emit_strategy_exit()
            return None

        combined = self._combine_strategy_votes(
            symbol=symbol,
            signals=signal_votes,
            indicators=indicators,
            no_vote_reason_counts=no_vote_reason_counts,
        )
        if combined is None or not bool(getattr(combined, "metadata", {}).get("is_approved")):
            decision = self.get_last_no_signal_decision(symbol)
            blocked_reason = str(getattr(decision, "reason", "") or "") or ("combined_none" if combined is None else "combined_not_approved")
            self._log_strategy_combiner_blocker(
                symbol=symbol,
                signal_votes=signal_votes,
                indicators=indicators,
                combined=combined,
                blocked_reason=blocked_reason,
                no_vote_reason_counts=no_vote_reason_counts,
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
                f"strategy_manager_no_combined:{symbol}",
                (
                    "strategy_manager_no_combined_signal symbol=%s no_vote_reason_counts=%s "
                    "direction_bias=%s underlying_direction_bias=%s context_age_seconds=%s"
                ),
                symbol,
                no_vote_reason_counts,
                indicators.get("direction_bias"),
                indicators.get("underlying_direction_bias"),
                indicators.get("context_age_seconds"),
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


    def _is_live_mode(self) -> bool:
        """Return True only when strategy logic is allowed to behave as live."""
        execution_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        enable_live = self._env_bool("ENABLE_LIVE", False) or self._env_bool(
            "ENABLE_LIVE_TRADING", False
        )
        paper_enabled = self._env_bool("PAPER__ENABLED", False) or self._env_bool(
            "PAPER_MODE", False
        )
        shadow_enabled = self._env_bool("SHADOW_MODE", False)
        return (
            execution_mode == "LIVE"
            and enable_live
            and not paper_enabled
            and not shadow_enabled
        )

    def _env_bool(self, name: str, default: bool) -> bool:
        """Args: env key/default. Returns: parsed bool. Raises: none."""
        return str(os.getenv(name, str(default).lower())).strip().lower() in {"1", "true", "yes", "on"}

    def _env_float(self, name: str, default: float) -> float:
        """Args: env key/default. Returns: parsed float. Raises: none."""
        from nifty_scalper_bot.config.env_utils import parse_float_env
        return parse_float_env(os.getenv(name), default)


    def _live_context_max_age_seconds(self) -> float:
        """Args: none. Returns: live context max age seconds. Raises: none."""
        default_age = self._env_float("MAX_CONTEXT_AGE_SECONDS", 5.0)
        if self._is_live_mode():
            return max(0.1, default_age)
        return max(0.1, self._env_float("STRATEGY_CONTEXT_MAX_AGE_SECONDS", 120.0))

    def _extract_raw_score(self, vote: StrategyVote) -> float:
        """Args: vote. Returns: raw score. Raises: none."""
        payload = dict(vote.metadata or {})
        for key in ("raw_vote_score", "raw_setup_score", "vote_score"):
            try:
                if payload.get(key) is not None:
                    return float(payload.get(key))
            except (TypeError, ValueError):
                continue
        try:
            return float(vote.score or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _extract_context_score(self, vote: StrategyVote) -> float:
        """Args: vote. Returns: context score. Raises: none."""
        payload = dict(vote.metadata or {})
        for key in ("context_bonus_score", "context_score", "raw_setup_score", "raw_vote_score", "vote_score"):
            try:
                if payload.get(key) is not None:
                    return max(0.0, float(payload.get(key)))
            except (TypeError, ValueError):
                continue
        return max(0.0, self._extract_raw_score(vote))

    def _extract_context_veto_score(self, vote: StrategyVote) -> float:
        """Args: vote. Returns: context veto score. Raises: none."""
        payload = dict(vote.metadata or {})
        for key in ("context_veto_score", "context_score", "raw_setup_score", "raw_vote_score", "vote_score"):
            try:
                if payload.get(key) is not None:
                    return max(0.0, float(payload.get(key)))
            except (TypeError, ValueError):
                continue
        return max(0.0, self._extract_raw_score(vote))

    def _is_selected_or_near_atm(self, symbol: str, metadata: dict[str, t.Any], indicators: t.Mapping[str, t.Any]) -> tuple[bool, dict[str, t.Any]]:
        """Args: symbol/metadata/indicators. Returns: selected bool + audit metadata. Raises: none."""
        symbol_norm = str(symbol or "").strip().upper()
        selected_ce = str(metadata.get("selected_ce") or indicators.get("selected_ce") or "")
        selected_pe = str(metadata.get("selected_pe") or indicators.get("selected_pe") or "")
        selected_set = {selected_ce.strip().upper(), selected_pe.strip().upper()}
        selected_set.discard("")
        selected_option = bool(metadata.get("is_selected_option") or indicators.get("is_selected_option") or symbol_norm in selected_set)
        near_atm_threshold = self._env_float("STRATEGY_NEAR_ATM_THRESHOLD_POINTS", 50.0)
        strike_distance_from_atm = metadata.get("strike_distance_from_atm", indicators.get("strike_distance_from_atm"))
        try:
            strike_distance = float(strike_distance_from_atm)
            near_atm = strike_distance <= near_atm_threshold
        except (TypeError, ValueError):
            strike_distance = None
            near_atm = False
        selected_ok = selected_option or near_atm
        return selected_ok, {"selected_ce": selected_ce, "selected_pe": selected_pe, "strike_distance_from_atm": strike_distance, "near_atm_threshold": near_atm_threshold, "is_selected_option": selected_option, "selected_ok_reason": "selected_option" if selected_option else "near_atm" if near_atm else "not_selected_or_near_atm", "near_atm": near_atm}


    def _log_strategy_combiner_blocker(
        self,
        *,
        symbol: str,
        signal_votes: list[tuple[Signal, StrategyVote]],
        indicators: t.Mapping[str, t.Any],
        combined: Signal | None,
        blocked_reason: str | None,
        no_vote_reason_counts: t.Mapping[str, int] | None = None,
    ) -> None:
        """Log exact strategy combiner blocker details without changing vote outcomes."""
        indicator_map = dict(indicators or {})
        trigger_votes: list[StrategyVote] = []
        context_votes: list[StrategyVote] = []
        for _signal, vote in signal_votes:
            role = str((vote.metadata or {}).get("role") or "trigger").lower()
            if role == "context":
                context_votes.append(vote)
            else:
                trigger_votes.append(vote)
        all_votes = trigger_votes + context_votes
        best_vote = max(all_votes, key=self._extract_raw_score) if all_votes else None
        combined_md = dict(getattr(combined, "metadata", {}) or {}) if combined is not None else {}
        final_trade_score = combined_md.get("final_trade_score")
        if final_trade_score is None and best_vote is not None:
            final_trade_score = self._extract_raw_score(best_vote)
        try:
            final_trade_threshold = float(os.getenv("STRATEGY_TRIGGER_MIN_SCORE", "4.5") or "4.5")
        except (TypeError, ValueError):
            final_trade_threshold = 4.5
        mode_profile = self.get_strategy_mode_profile()
        selected_ok = bool(
            indicator_map.get("is_selected_option")
            or str(symbol or "").strip().upper()
            in {
                str(indicator_map.get("selected_ce") or "").strip().upper(),
                str(indicator_map.get("selected_pe") or "").strip().upper(),
            }
        )
        selected_ok_reason = "selected_option" if selected_ok else "not_selected_or_near_atm"
        if combined_md.get("selected_ok_reason"):
            selected_ok_reason = str(combined_md.get("selected_ok_reason"))
            selected_ok = selected_ok_reason != "not_selected_or_near_atm"
        log.debug(
            "STRATEGY_COMBINER_BLOCKER symbol=%s strategy_vote_count=%s trigger_vote_count=%s context_vote_count=%s best_strategy=%s best_score=%s best_confidence=%s single_vote_allowed=%s selected_ok=%s selected_ok_reason=%s final_trade_score=%s final_trade_threshold=%s blocked_reason=%s",
            symbol,
            len(signal_votes),
            len(trigger_votes),
            len(context_votes),
            best_vote.strategy if best_vote else None,
            self._extract_raw_score(best_vote) if best_vote else None,
            float(best_vote.confidence) if best_vote else None,
            bool(mode_profile.get("allow_single_vote", True)) and str(os.getenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", "false")).lower() in {"1", "true", "yes", "on"},
            selected_ok,
            selected_ok_reason,
            final_trade_score,
            final_trade_threshold,
            blocked_reason or combined_md.get("blocked_reason") or "combined_not_approved",
            extra={
                "event": "STRATEGY_COMBINER_BLOCKER",
                "symbol": symbol,
                "strategy_vote_count": len(signal_votes),
                "trigger_vote_count": len(trigger_votes),
                "context_vote_count": len(context_votes),
                "best_strategy": best_vote.strategy if best_vote else None,
                "best_score": self._extract_raw_score(best_vote) if best_vote else None,
                "best_confidence": float(best_vote.confidence) if best_vote else None,
                "single_vote_allowed": bool(mode_profile.get("allow_single_vote", True)) and str(os.getenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", "false")).lower() in {"1", "true", "yes", "on"},
                "selected_ok": selected_ok,
                "selected_ok_reason": selected_ok_reason,
                "final_trade_score": final_trade_score,
                "final_trade_threshold": final_trade_threshold,
                "blocked_reason": blocked_reason or combined_md.get("blocked_reason") or "combined_not_approved",
                "no_vote_reason_counts": dict(no_vote_reason_counts or {}),
                "combined_present": combined is not None,
                "combined_is_approved": bool(combined_md.get("is_approved")),
            },
        )

    def _combine_strategy_votes(
        self,
        *,
        symbol: str,
        signals: list[tuple[Signal, StrategyVote]],
        indicators: t.Mapping[str, t.Any],
        no_vote_reason_counts: t.Mapping[str, int] | None = None,
    ) -> Signal | None:
        """Args: symbol/signals/indicators. Returns: consensus signal or None. Raises: none."""
        symbol_norm = str(symbol or "").strip().upper()
        def _record_no_signal(category: str, reason_text: str, blocked_at: str, *, no_vote_reason_counts: dict[str, int] | None = None, strategy_reasons: dict[str, str] | None = None, trigger_vote_count: int = 0, context_vote_count: int = 0, final_block_reason: str | None = None) -> None:
            self._last_no_signal_decision_by_symbol[symbol_norm] = StrategyNoSignalDecision(
                symbol=symbol_norm,
                eval_id=str(indicator_map.get("eval_id") or indicator_map.get("trace_id") or "") or None,
                final_block_reason=final_block_reason,
                category=category,
                reason=reason_text,
                blocked_at=blocked_at,
                no_vote_reason_counts=dict(no_vote_reason_counts or {}),
                strategy_reasons=dict(strategy_reasons or {}),
                direction_bias=str(indicator_map.get("direction_bias") or "").upper() or None,
                underlying_direction_bias=str(indicator_map.get("underlying_direction_bias") or "").upper() or None,
                context_age_seconds=float(indicator_map.get("context_age_seconds")) if indicator_map.get("context_age_seconds") is not None else None,
                trigger_vote_count=trigger_vote_count,
                context_vote_count=context_vote_count,
                selected_ce=str(indicator_map.get("selected_ce") or "") or None,
                selected_pe=str(indicator_map.get("selected_pe") or "") or None,
                trace_id=str(indicator_map.get("trace_id") or "") or None,
            )
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
        mode_profile = self.get_strategy_mode_profile()
        trigger_votes: list[tuple[Signal, StrategyVote]] = []
        context_votes: list[tuple[Signal, StrategyVote]] = []
        for signal, vote in signals:
            if signal.action in {"CLOSE_LONG", "CLOSE_SHORT"}:
                signal.metadata = dict(signal.metadata or {})
                signal.metadata["approval_path"] = "close_signal"
                log.info("TRADE_DECISION_TRACE approval_path=%s symbol=%s", "close_signal", symbol_norm)
                return signal
            role = str((vote.metadata or {}).get("role") or "trigger").lower()
            if role == "context":
                context_votes.append((signal, vote))
            else:
                trigger_votes.append((signal, vote))
        approval_path = "multi_trigger"
        if not trigger_votes:
            promoted_candidate = self._try_context_promotion(symbol_norm, context_votes, indicator_map, mode_profile)
            if promoted_candidate:
                best_signal, best_vote = promoted_candidate
                approval_path = "context_promotion"
            else:
                approval_path = "no_trade"
                context_vote_names = [v.strategy for _, v in context_votes]
                context_sides = [v.side for _, v in context_votes]
                context_scores = [self._extract_raw_score(v) for _, v in context_votes]
                context_confidences = [float(v.confidence) for _, v in context_votes]
                context_trigger_details = []
                for signal, vote in context_votes:
                    md = dict(getattr(signal, "metadata", {}) or {})
                    vote_md = dict(getattr(vote, "metadata", {}) or {})
                    merged_md = {**md, **vote_md}
                    context_trigger_details.append(
                        {
                            "strategy": vote.strategy,
                            "side": vote.side,
                            "score": self._extract_raw_score(vote),
                            "confidence": float(vote.confidence),
                            "role": merged_md.get("role"),
                            "can_trigger": merged_md.get("can_trigger"),
                            "trigger_conditions_met": merged_md.get("trigger_conditions_met"),
                            "trigger_block_reason": merged_md.get("trigger_block_reason"),
                            "quote_depth_valid": merged_md.get("quote_depth_valid"),
                            "tradable_quote": merged_md.get("tradable_quote"),
                            "spread_pct": merged_md.get("spread_pct"),
                            "tick_age_ms": merged_md.get("tick_age_ms") or indicator_map.get("tick_age_ms") or merged_md.get("context", {}).get("tick_age_ms"),
                            "quote_update_version": merged_md.get("quote_update_version") or indicator_map.get("quote_update_version") or merged_md.get("context", {}).get("quote_update_version"),
                            "depth_available": merged_md.get("depth_available"),
                            "tick_direction": merged_md.get("tick_direction"),
                        }
                    )
                log_throttled_live(
                    log,
                    logging.INFO,
                    "TRADE_DECISION_TRACE",
                    f"TRADE_DECISION_TRACE:{symbol_norm}:no_trigger_vote:{indicator_map.get('direction_bias')}",
                    float(os.getenv("LOG_THROTTLE_NO_TRADE_TRACE_SECONDS", "30") or "30"),
                    "TRADE_DECISION_TRACE approval_path=%s blocked_at=%s blocked_reason=%s "
                    "symbol=%s context_votes=%s context_sides=%s context_scores=%s "
                    "direction_bias=%s underlying_direction_bias=%s context_age_seconds=%s context_trigger_details=%s",
                    approval_path,
                    "no_trigger_vote",
                    "no_trigger_vote",
                    symbol_norm,
                    context_vote_names,
                    context_sides,
                    context_scores,
                    indicator_map.get("direction_bias"),
                    indicator_map.get("underlying_direction_bias"),
                    indicator_map.get("context_age_seconds"),
                    context_trigger_details,
                    extra={
                        "event": "TRADE_DECISION_TRACE",
                        "symbol": symbol_norm,
                        "approval_path": approval_path,
                        "blocked_at": "no_trigger_vote",
                        "blocked_reason": "no_trigger_vote",
                        "context_vote_count": len(context_votes),
                        "context_votes": context_vote_names,
                        "context_sides": context_sides,
                        "context_scores": context_scores,
                        "context_confidences": context_confidences,
                        "context_trigger_details": context_trigger_details,
                        "allow_context_promotion": bool(mode_profile.get("allow_context_promotion", False)),
                        "execution_mode": mode_profile.get("mode"),
                        "direction_bias": indicator_map.get("direction_bias"),
                        "underlying_direction_bias": indicator_map.get("underlying_direction_bias"),
                        "context_age_seconds": indicator_map.get("context_age_seconds"),
                        "direction_context_source": indicator_map.get("direction_context_source"),
                        "direction_context_age_s": indicator_map.get("context_age_seconds"),
                        "direction_context_confidence": indicator_map.get("underlying_direction_confidence"),
                        "spot_tick_age_ms": (indicator_map.get("spot_context") or {}).get("tick_age_ms") if isinstance(indicator_map.get("spot_context"), dict) else None,
                        "futures_tick_age_ms": (indicator_map.get("futures_context") or {}).get("tick_age_ms") if isinstance(indicator_map.get("futures_context"), dict) else None,
                        "direction_missing_reason": "direction_bias_none" if str(indicator_map.get("direction_bias") or "").upper() not in {"CE", "PE"} else None,
                    },
                )
                log_throttled(
                    log,
                    f"strategy_no_trigger_vote:{symbol_norm}",
                    "STRATEGY_NO_TRIGGER_VOTE symbol=%s context_votes=%s context_sides=%s live_context_promotion_allowed=%s context_trigger_details=%s",
                    symbol_norm,
                    context_vote_names,
                    context_sides,
                    bool(mode_profile.get("allow_context_promotion", False)),
                    context_trigger_details,
                    interval_sec=30.0,
                    level=logging.INFO,
                    extra={
                        "event": "STRATEGY_NO_TRIGGER_VOTE",
                        "symbol": symbol_norm,
                        "context_vote_count": len(context_votes),
                        "context_votes": context_vote_names,
                        "context_sides": context_sides,
                        "context_scores": context_scores,
                        "context_confidences": context_confidences,
                        "context_trigger_details": context_trigger_details,
                        "allow_context_promotion": bool(mode_profile.get("allow_context_promotion", False)),
                        "execution_mode": mode_profile.get("mode"),
                        "direction_bias": indicator_map.get("direction_bias"),
                        "underlying_direction_bias": indicator_map.get("underlying_direction_bias"),
                        "context_age_seconds": indicator_map.get("context_age_seconds"),
                    },
                )
                _record_no_signal("strategy_partial_no_consensus", "no_trigger_vote", "no_trigger_vote", trigger_vote_count=0, context_vote_count=len(context_votes), final_block_reason="partial")
                return None
        else:
            best_signal, best_vote = max(trigger_votes, key=lambda pair: self._extract_raw_score(pair[1]))
        metadata = dict(best_signal.metadata or {})
        threshold = float(os.getenv("STRATEGY_TRIGGER_MIN_SCORE", "4.5") or "4.5")
        single_high = float(os.getenv("STRATEGY_SINGLE_VOTE_HIGH_CONVICTION", "8.8") or "8.8")
        allow_scalp_single = bool(mode_profile.get("allow_single_vote", True)) and str(os.getenv("STRATEGY_ALLOW_SINGLE_VOTE_SCALP", "false")).lower() in {"1", "true", "yes", "on"}
        raw_trigger_score = self._extract_raw_score(best_vote)
        weighted_trigger_score = _weighted_score(best_vote)
        vetoed = False
        same_side_context = [v for _, v in context_votes if v.side == best_vote.side]
        opposite_context = [v for _, v in context_votes if v.side in {"CE", "PE"} and v.side != best_vote.side]
        positive_context = sum(self._extract_context_score(v) for v in same_side_context)
        negative_context = sum(self._extract_context_veto_score(v) for v in opposite_context)
        context_bonus = min(1.5, 0.45 * positive_context)
        context_penalty = min(1.5, 0.60 * negative_context)
        final_score = raw_trigger_score + context_bonus - context_penalty
        final_score = max(0.0, min(10.0, final_score))
        context_confidence_floor = float(os.getenv("STRATEGY_CONTEXT_HARD_VETO_MIN_CONFIDENCE", "0.80") or "0.80")
        context_freshness_max_age_s = float(os.getenv("STRATEGY_CONTEXT_HARD_VETO_MAX_AGE_SECONDS", "120") or "120")
        now_epoch = time.time()
        hard_veto_candidates = []
        for vote in opposite_context:
            md = dict(vote.metadata or {})
            vote_ts = float(md.get("timestamp_epoch") or md.get("ts") or now_epoch)
            age_s = max(0.0, now_epoch - vote_ts)
            if (
                self._extract_context_veto_score(vote) >= 8.0
                and float(vote.confidence) >= context_confidence_floor
                and age_s <= context_freshness_max_age_s
            ):
                hard_veto_candidates.append(vote)
        vetoed = bool(hard_veto_candidates)
        selected_ok = True
        near_atm = indicator_near_atm
        if trigger_votes and len(trigger_votes) == 1:
            approval_path = "single_trigger"
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
            threshold_passed = bool(raw_trigger_score >= score_min and best_vote.confidence >= conf_min and selected_ok and not vetoed)
            # STRATEGY_ALLOW_SINGLE_VOTE_SCALP=false is the master default block.
            # Single-vote high-conviction/selected-option paths require explicit,
            # narrower operator overrides to avoid accidental lone-vote approvals.
            allow_high_conviction = self._env_bool(
                "STRATEGY_ALLOW_HIGH_CONVICTION_SINGLE_VOTE",
                self._env_bool("STRATEGY_ALLOW_SINGLE_VOTE_HIGH_CONVICTION", False),
            )
            allow_selected_option = self._env_bool(
                "STRATEGY_ALLOW_SELECTED_OPTION_SINGLE_VOTE",
                self._env_bool("STRATEGY_ALLOW_SINGLE_VOTE_SELECTED_OPTION", False),
            )
            high_conviction_allowed = bool(raw_trigger_score >= single_high and selected_ok and not vetoed and allow_high_conviction)
            # A single trigger on the actually-SELECTED ATM option that already
            # cleared score/confidence/veto gates is the core scalp this platform
            # exists to take. Allow it without the global scalp flag, since
            # selected_option is a stronger guarantee than mere near_atm.
            selected_option_scalp_allowed = bool(threshold_passed and selected_option and allow_selected_option)
            scalp_fallback_allowed = bool((allow_scalp_single and threshold_passed) or selected_option_scalp_allowed)
            final_allowed = bool(high_conviction_allowed or scalp_fallback_allowed)
            approval_path_single = (
                "single_vote_high_conviction" if high_conviction_allowed
                else "single_vote_selected_option" if selected_option_scalp_allowed
                else "single_vote_fallback" if scalp_fallback_allowed
                else None
            )
            blocked_reason = None
            if not final_allowed:
                if threshold_passed and not allow_scalp_single:
                    blocked_reason = "single_vote_scalp_disabled"
                elif not score_ok:
                    blocked_reason = "raw_score_below_min"
                elif not conf_ok:
                    blocked_reason = "confidence_below_min"
                elif not selected_ok:
                    blocked_reason = "not_selected_or_near_atm"
                elif vetoed:
                    blocked_reason = "hard_context_veto"
                else:
                    blocked_reason = "single_vote_gate_failed_unknown"
            log.info(
                "SINGLE_VOTE_DECISION threshold_passed=%s high_conviction_allowed=%s scalp_fallback_allowed=%s final_allowed=%s allow_scalp_single=%s approval_path=%s reason=%s strategy=%s raw_score=%.2f weighted_score=%.2f regime_weight=%.2f score_min=%.2f confidence=%.2f conf_min=%.2f selected_ok=%s vetoed=%s",
                threshold_passed,
                high_conviction_allowed,
                scalp_fallback_allowed,
                final_allowed,
                allow_scalp_single,
                approval_path_single,
                blocked_reason or "ok",
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
                    "allowed": final_allowed,
                    "threshold_passed": threshold_passed,
                    "high_conviction_allowed": high_conviction_allowed,
                    "scalp_fallback_allowed": scalp_fallback_allowed,
                    "final_allowed": final_allowed,
                    "allow_scalp_single": allow_scalp_single,
                    "approval_path": approval_path_single,
                    "blocked_reason": blocked_reason,
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
            if high_conviction_allowed:
                metadata_stage = "preliminary_single_high_conviction"
                approval_path = "single_vote_high_conviction"
            elif scalp_fallback_allowed:
                metadata_stage = "single_vote_scalp_controlled"
                approval_path = approval_path_single or "single_vote_fallback"
            else:
                allow_candidate_switch = str(os.getenv("STRATEGY_ALLOW_CANDIDATE_SWITCH_ON_HIGH_SCORE", "false")).lower() in {"1", "true", "yes", "on"}
                max_candidate_switch_distance = float(os.getenv("CANDIDATE_SWITCH_MAX_DISTANCE_POINTS", "100") or "100")
                raw_qdv = metadata.get("quote_depth_valid")
                if raw_qdv is None:
                    raw_qdv = indicator_map.get("quote_depth_valid")
                quote_depth_valid = bool(raw_qdv)
                raw_spread = metadata.get("spread_pct")
                if raw_spread is None:
                    raw_spread = indicator_map.get("spread_pct")
                try:
                    switch_spread_pct = float(raw_spread) if raw_spread is not None else 999.0
                except (TypeError, ValueError):
                    switch_spread_pct = 999.0
                switch_min_score = 8.0 if best_vote.strategy == "OrderFlow" else 8.5
                switch_allowed = (
                    allow_candidate_switch
                    and (strike_distance_from_atm is not None)
                    and raw_trigger_score >= switch_min_score
                    and quote_depth_valid
                    and switch_spread_pct <= self._env_float("LIVE_MAX_SPREAD_PCT", 0.75)
                    and strike_distance_from_atm <= max_candidate_switch_distance
                )
                if switch_allowed:
                    metadata_stage = "single_vote_candidate_switch"
                    metadata["candidate_switch_requested"] = True
                    metadata["candidate_switch_reason"] = "high_score_nearby_option_candidate"
                else:
                    if not score_ok:
                        blocked_reason = "raw_score_below_min"
                    elif not conf_ok:
                        blocked_reason = "confidence_below_min"
                    elif not selected_ok:
                        blocked_reason = "not_selected_or_near_atm"
                    elif not bool(metadata.get("quote_depth_valid", True)):
                        blocked_reason = "quote_depth_invalid"
                    elif vetoed:
                        blocked_reason = "hard_context_veto"
                    elif not allow_scalp_single:
                        blocked_reason = "single_vote_scalp_disabled"
                    else:
                        blocked_reason = "single_vote_gate_failed_unknown"
                    log_throttled_live(
                        log,
                        logging.INFO,
                        "TRADE_DECISION_TRACE",
                        f"TRADE_DECISION_TRACE:{best_vote.strategy}:{symbol_norm}:{blocked_reason}",
                        float(os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120") or "120"),
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
                    category = "strategy_single_vote_disabled" if blocked_reason == "single_vote_scalp_disabled" else ("context_direction_conflict" if blocked_reason == "underlying_direction_conflict" else "strategy_no_trigger")
                    _record_no_signal(category, blocked_reason, "single_vote_gate", trigger_vote_count=len(trigger_votes), context_vote_count=len(context_votes))
                    return None
        elif trigger_votes:
            metadata_stage = "multi_vote_confirmed"
            approval_path = "multi_trigger"
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
            blocked_reason = "hard_context_veto" if vetoed else "final_trade_score_below_threshold"
            log_throttled_live(
                log,
                logging.INFO,
                "TRADE_DECISION_TRACE",
                f"TRADE_DECISION_TRACE:{best_vote.strategy}:{symbol_norm}:{blocked_reason}",
                float(os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120") or "120"),
                "TRADE_DECISION_TRACE symbol=%s strategy=%s side=%s data_gate=%s setup_score=%.2f setup_min=%.2f weighted_score=%.2f regime_weight=%.2f context_bonus=%.2f context_penalty=%.2f final_score=%.2f final_min=%.2f allowed=%s blocked_at=%s blocked_reason=%s",
                symbol_norm, best_vote.strategy, best_vote.side, True, raw_trigger_score, threshold, weighted_trigger_score, _regime_weight(best_vote), context_bonus, context_penalty, final_score, threshold, False, "strategy_manager_combine", blocked_reason,
                extra={"event": "TRADE_DECISION_TRACE", "symbol": symbol_norm, "strategy": best_vote.strategy, "side": best_vote.side, "data_gate": True, "setup_score": raw_trigger_score, "setup_min": threshold, "weighted_score": weighted_trigger_score, "regime_weight": _regime_weight(best_vote), "context_bonus": context_bonus, "context_penalty": context_penalty, "final_score": final_score, "final_min": threshold, "allowed": False, "blocked_at": "strategy_manager_combine", "blocked_reason": blocked_reason},
            )
            _record_no_signal("strategy_score_below_threshold", blocked_reason, "strategy_manager_combine", trigger_vote_count=len(trigger_votes), context_vote_count=len(context_votes))
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
            raw_reason = str(metadata.get("quality_block_reason") or "").strip().lower()
            metadata["quality_block_reason"] = "trade_quality_below_threshold" if raw_reason in {"", "ok"} else str(metadata.get("quality_block_reason"))
            log_throttled_live(
                log,
                logging.INFO,
                "STRATEGY_QUALITY_REJECT",
                f"STRATEGY_QUALITY_REJECT:{best_vote.strategy}:{symbol_norm}:{metadata['quality_block_reason']}",
                float(os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120") or "120"),
                "STRATEGY_QUALITY_REJECT symbol=%s strategy=%s side=%s score=%.2f reason=%s",
                symbol_norm,
                best_vote.strategy,
                best_vote.side,
                quality_score,
                metadata["quality_block_reason"],
                extra={"event": "STRATEGY_QUALITY_REJECT", "symbol": symbol_norm, "strategy": best_vote.strategy, "side": best_vote.side, "score": quality_score, "reason": metadata["quality_block_reason"]},
            )
            log_throttled_live(
                log,
                logging.INFO,
                "TRADE_DECISION_TRACE",
                f"TRADE_DECISION_TRACE:{best_vote.strategy}:{symbol_norm}:{metadata['quality_block_reason']}",
                float(os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120") or "120"),
                "TRADE_DECISION_TRACE symbol=%s strategy=%s side=%s allowed=%s blocked_at=%s blocked_reason=%s",
                symbol_norm,
                best_vote.strategy,
                best_vote.side,
                False,
                "trade_quality_gate",
                metadata["quality_block_reason"],
                extra={"event": "TRADE_DECISION_TRACE", "symbol": symbol_norm, "strategy": best_vote.strategy, "side": best_vote.side, "allowed": False, "blocked_at": "trade_quality_gate", "blocked_reason": metadata["quality_block_reason"]},
            )
            _record_no_signal("strategy_no_trigger", str(metadata["quality_block_reason"]), "trade_quality_gate", trigger_vote_count=len(trigger_votes), context_vote_count=len(context_votes))
            return None

        direction_bias = str(indicator_map.get("direction_bias") or indicator_map.get("underlying_direction_bias") or "").upper()
        max_context_age = self._live_context_max_age_seconds()
        try:
            context_age_seconds = float(indicator_map.get("context_age_seconds"))
        except (TypeError, ValueError):
            context_age_seconds = 999.0
        spread_pct = float(metadata.get("spread_pct") or indicator_map.get("spread_pct") or 999.0)
        selected_ok_combined = bool(selected_ok or near_atm)
        quote_depth_ok = bool(metadata.get("quote_depth_valid") or indicator_map.get("quote_depth_valid") or metadata.get("tradable_quote") or indicator_map.get("tradable_quote"))
        no_vote_counts = dict(no_vote_reason_counts or indicator_map.get("no_vote_reason_counts") or {})
        neutral_no_votes = {"smc_insufficient_history", "strategy_feature_unavailable"}
        hard_veto_reasons: list[str] = []
        if no_vote_counts.get("underlying_direction_conflict"):
            hard_veto_reasons.append("underlying_direction_conflict")
        if no_vote_counts.get("negative_premium_flow"):
            hard_veto_reasons.append("negative_premium_flow")
        if no_vote_counts.get("smc_structure_required_live"):
            hard_veto_reasons.append("smc_structure_required_live")
        two_trigger_aligned = bool(
            len(trigger_votes) >= 2
            and best_vote.side in {"CE", "PE"}
            and direction_bias == str(best_vote.side).upper()
            and context_age_seconds <= max_context_age
            and selected_ok_combined
            and quote_depth_ok
            and spread_pct <= self._env_float("LIVE_MAX_SPREAD_PCT", 0.75)
            and quality_pass
            and not hard_veto_reasons
            and not no_vote_counts.get("negative_premium_flow")
        )
        if two_trigger_aligned:
            approval_path = "aligned_two_trigger_consensus"
            log.info("CONSENSUS_SIGNAL_APPROVED symbol=%s side=%s approval_path=%s trigger_vote_count=%s context_vote_count=%s", symbol_norm, best_vote.side, approval_path, len(trigger_votes), len(context_votes), extra={"event":"CONSENSUS_SIGNAL_APPROVED","symbol":symbol_norm,"side":best_vote.side,"approval_path":"aligned_two_trigger_consensus","trigger_vote_count":len(trigger_votes),"context_vote_count":len(context_votes),"ignored_neutral_no_votes":[r for r in neutral_no_votes if no_vote_counts.get(r)],"hard_veto_reasons":hard_veto_reasons,"trade_quality_score":quality_score,"direction_bias":direction_bias,"context_age_seconds":context_age_seconds,"spread_pct":spread_pct,"selected_ok":bool(selected_ok),"near_atm_ok":bool(near_atm)})

        metadata.setdefault("trigger_strategy_score", metadata.get("strategy_score"))
        metadata["setup_score"] = round(raw_trigger_score, 3)
        metadata["raw_setup_score"] = round(raw_trigger_score, 3)
        metadata.setdefault("strategy_score", round(raw_trigger_score, 3))
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
        metadata["manager_score_source"] = "strategy_score_plus_context_arbitration"
        metadata["approval_path"] = approval_path
        metadata["is_approved"] = True
        log.info("TRADE_DECISION_TRACE approval_path=%s symbol=%s strategy=%s", approval_path, symbol_norm, best_vote.strategy)
        self._last_no_signal_decision_by_symbol.pop(symbol_norm, None)
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

    def get_last_no_signal_decision(self, symbol: str) -> StrategyNoSignalDecision | None:
        return self._last_no_signal_decision_by_symbol.get(str(symbol or "").strip().upper())

    def _record_no_signal_decision(
        self,
        *,
        symbol: str,
        category: str,
        reason: str,
        blocked_at: str,
        indicators: t.Mapping[str, t.Any],
        no_vote_reason_counts: t.Mapping[str, int] | None = None,
        strategy_reasons: t.Mapping[str, str] | None = None,
        trigger_vote_count: int = 0,
        context_vote_count: int = 0,
        trace_id: str | None = None,
        eval_id: str | None = None,
        final_block_reason: str | None = None,
    ) -> None:
        symbol_norm = str(symbol or "").strip().upper()
        self._last_no_signal_decision_by_symbol[symbol_norm] = StrategyNoSignalDecision(
            symbol=symbol_norm,
            eval_id=eval_id,
            final_block_reason=final_block_reason,
            category=category,
            reason=reason,
            blocked_at=blocked_at,
            no_vote_reason_counts=dict(no_vote_reason_counts or {}),
            strategy_reasons=dict(strategy_reasons or {}),
            direction_bias=str(indicators.get("direction_bias") or "").upper() or None,
            underlying_direction_bias=str(indicators.get("underlying_direction_bias") or "").upper() or None,
            context_age_seconds=float(indicators.get("context_age_seconds")) if indicators.get("context_age_seconds") is not None else None,
            trigger_vote_count=trigger_vote_count,
            context_vote_count=context_vote_count,
            selected_ce=str(indicators.get("selected_ce") or "") or None,
            selected_pe=str(indicators.get("selected_pe") or "") or None,
            trace_id=trace_id,
        )

    def get_strategy_mode_profile(self) -> dict[str, t.Any]:
        """Return strategy gating profile based on effective execution mode."""
        raw_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        live_effective = self._is_live_mode()
        execution_mode = "LIVE" if live_effective else raw_mode
        if execution_mode not in {"LIVE", "PAPER", "SHADOW", "SIMULATION"}:
            execution_mode = "SHADOW"

        if execution_mode == "LIVE":
            defaults = {
                "allow_context_promotion": False,
                "allow_single_vote": True,
                "min_trade_quality": 7.0,
            }
        elif execution_mode == "PAPER":
            defaults = {
                "allow_context_promotion": True,
                "allow_single_vote": True,
                "min_trade_quality": 5.8,
            }
        else:
            defaults = {
                "allow_context_promotion": True,
                "allow_single_vote": True,
                "min_trade_quality": 5.0,
            }

        allow_context_key = (
            "STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED"
            if execution_mode == "LIVE"
            else "STRATEGY_ALLOW_CONTEXT_PROMOTION"
        )
        allow_single_key = (
            "STRATEGY_ALLOW_SINGLE_VOTE_LIVE"
            if execution_mode == "LIVE"
            else "STRATEGY_ALLOW_SINGLE_VOTE_BY_MODE"
        )

        try:
            min_quality = float(
                os.getenv(
                    f"STRATEGY_MIN_TRADE_QUALITY_{execution_mode}",
                    str(defaults["min_trade_quality"]),
                )
                or defaults["min_trade_quality"]
            )
        except (TypeError, ValueError):
            min_quality = float(defaults["min_trade_quality"])

        return {
            "mode": execution_mode,
            "raw_mode": raw_mode,
            "live_effective": live_effective,
            "allow_context_promotion": self._env_bool(
                allow_context_key,
                bool(defaults["allow_context_promotion"]),
            ),
            "allow_single_vote": self._env_bool(
                allow_single_key,
                bool(defaults["allow_single_vote"]),
            ),
            "min_trade_quality": min_quality,
        }

    def _try_context_promotion(
        self,
        symbol: str,
        context_votes: list[tuple[Signal, StrategyVote]],
        indicators: t.Mapping[str, t.Any],
        mode_profile: dict[str, t.Any],
    ) -> tuple[Signal, StrategyVote] | None:
        """Args: symbol/context votes/indicators/profile. Returns: promoted candidate or None. Raises: none."""
        if not context_votes:
            return None
        if not bool(mode_profile.get("allow_context_promotion", True)):
            return None
        allowed_strategies = {s.strip() for s in str(os.getenv("STRATEGY_CONTEXT_PROMOTION_ALLOWED_STRATEGIES", "OrderFlow,VWAPPro")).split(",") if s.strip()}
        min_score = self._env_float("STRATEGY_CONTEXT_PROMOTION_MIN_SCORE", 5.0)
        min_conf = self._env_float("STRATEGY_CONTEXT_PROMOTION_MIN_CONFIDENCE", 0.45)
        min_direction_conf = self._env_float("STRATEGY_CONTEXT_PROMOTION_MIN_DIRECTION_CONF", 0.05)
        best_signal, best_vote = max(context_votes, key=lambda pair: self._extract_raw_score(pair[1]))
        raw_score = self._extract_raw_score(best_vote)
        md0 = dict(best_signal.metadata or {})
        selected_ok, selected_meta = self._is_selected_or_near_atm(symbol, md0, indicators)
        if self._is_live_mode():
            if not self._env_bool("STRATEGY_CONTEXT_PROMOTION_LIVE_ALLOWED", False):
                return None
            live_min_score = self._env_float("STRATEGY_CONTEXT_PROMOTION_LIVE_MIN_SCORE", 8.5)
            live_min_conf = self._env_float("STRATEGY_CONTEXT_PROMOTION_LIVE_MIN_CONFIDENCE", 0.75)
            max_spread = self._env_float("STRATEGY_MAX_SPREAD_PCT", 12.0)
            max_context_age = self._live_context_max_age_seconds()
            try:
                spread_pct = float(md0.get("spread_pct") or indicators.get("spread_pct") or 999.0)
            except (TypeError, ValueError):
                spread_pct = 999.0
            try:
                context_age = float(md0.get("context_age_seconds") or indicators.get("context_age_seconds") or 999.0)
            except (TypeError, ValueError):
                context_age = 999.0
            try:
                bid = float(md0.get("bid") or indicators.get("bid") or 0.0)
            except (TypeError, ValueError):
                bid = 0.0
            try:
                ask = float(md0.get("ask") or indicators.get("ask") or 0.0)
            except (TypeError, ValueError):
                ask = 0.0
            bid_ask_valid = bid > 0.0 and ask > 0.0 and ask >= bid
            depth_or_tradable = bool(
                md0.get("quote_depth_valid")
                or indicators.get("quote_depth_valid")
                or md0.get("tradable_quote")
                or indicators.get("tradable_quote")
            )
            quote_depth_valid = bool(bid_ask_valid and depth_or_tradable)
            if spread_pct >= 999.0 and bid_ask_valid:
                midpoint = (bid + ask) / 2.0
                if midpoint > 0.0:
                    spread_pct = ((ask - bid) / midpoint) * 100.0
            live_reject_reason = None
            if raw_score < live_min_score:
                live_reject_reason = "live_context_score_below_min"
            elif float(best_vote.confidence) < live_min_conf:
                live_reject_reason = "live_context_confidence_below_min"
            elif best_vote.side not in {"CE", "PE"}:
                live_reject_reason = "live_context_invalid_side"
            elif not selected_ok:
                live_reject_reason = "live_context_not_selected_or_near_atm"
            elif not quote_depth_valid:
                live_reject_reason = "live_context_quote_depth_missing"
            elif spread_pct > max_spread:
                live_reject_reason = "live_context_spread_too_wide"
            elif context_age > max_context_age:
                live_reject_reason = "live_context_stale"
            if live_reject_reason:
                log.info(
                    "CONTEXT_PROMOTION_REJECTED_LIVE symbol=%s strategy=%s reason=%s "
                    "raw_score=%.2f confidence=%.2f selected_ok=%s quote_depth_valid=%s "
                    "spread_pct=%.2f context_age_seconds=%.2f",
                    symbol,
                    best_vote.strategy,
                    live_reject_reason,
                    raw_score,
                    best_vote.confidence,
                    selected_ok,
                    quote_depth_valid,
                    spread_pct,
                    context_age,
                    extra={
                        "event": "CONTEXT_PROMOTION_REJECTED_LIVE",
                        "symbol": symbol,
                        "strategy": best_vote.strategy,
                        "reason": live_reject_reason,
                        "raw_score": raw_score,
                        "min_score": live_min_score,
                        "confidence": best_vote.confidence,
                        "min_confidence": live_min_conf,
                        "selected_ok": selected_ok,
                        "quote_depth_valid": quote_depth_valid,
                        "spread_pct": spread_pct,
                        "max_spread": max_spread,
                        "context_age_seconds": context_age,
                        "max_context_age": max_context_age,
                    },
                )
                return None
        opposite = [v for _, v in context_votes if v.side in {"CE", "PE"} and v.side != best_vote.side]
        vetoed = bool(opposite and max(self._extract_context_veto_score(v) for v in opposite) >= 8.0)
        direction_bias = str(
            md0.get("direction_bias")
            or md0.get("underlying_direction_bias")
            or indicators.get("direction_bias")
            or indicators.get("underlying_direction_bias")
            or ""
        ).upper()
        direction_aligned = direction_bias in {"CE", "PE"} and direction_bias == str(best_vote.side).upper()
        age_raw = md0.get("context_age_seconds")
        if age_raw is None:
            age_raw = indicators.get("context_age_seconds")
        try:
            context_age_seconds = float(age_raw) if age_raw is not None else 999.0
        except (TypeError, ValueError):
            context_age_seconds = 999.0
        context_fresh = bool(
            md0.get("context_fresh")
            if md0.get("context_fresh") is not None
            else indicators.get("context_fresh")
        ) and context_age_seconds <= self._live_context_max_age_seconds()
        conf_raw = md0.get("underlying_direction_confidence")
        if conf_raw is None:
            conf_raw = indicators.get("underlying_direction_confidence")
        try:
            direction_conf = float(conf_raw) if conf_raw is not None else 0.0
        except (TypeError, ValueError):
            direction_conf = 0.0
        if best_vote.strategy not in allowed_strategies or raw_score < min_score or best_vote.confidence < min_conf or best_vote.side not in {"CE", "PE"} or not selected_ok or vetoed or not direction_aligned or not context_fresh or direction_conf < min_direction_conf:
            return None
        md = dict(best_signal.metadata or {})
        md.update(selected_meta)
        md.update({
            "role": "trigger",
            "promoted_from_context": True,
            "consensus_stage": "context_promoted_controlled",
            "promotion_reason": "direction_context_aligned",
        })
        for key in (
            "direction_bias",
            "underlying_direction_bias",
            "underlying_direction_confidence",
            "context_age_seconds",
            "context_fresh",
            "direction_context_source",
            "direction_context_reasons",
        ):
            value = indicators.get(key)
            if value is not None:
                md[key] = value
        promoted = Signal(action="BUY", symbol=best_signal.symbol, quantity=best_signal.quantity, confidence=best_signal.confidence, reason=best_signal.reason, stop_loss=best_signal.stop_loss, take_profit=best_signal.take_profit, metadata=md)
        promoted_vote = StrategyVote(strategy=best_vote.strategy, side=best_vote.side, score=best_vote.score, confidence=best_vote.confidence, reasons=list(best_vote.reasons), metadata=md)
        log.info(
            "ORDERFLOW_TRIGGER_PROMOTED side=%s score=%.2f reason=direction_context_aligned symbol=%s direction_context_source=%s context_age_seconds=%.2f underlying_direction_confidence=%.2f",
            best_vote.side,
            raw_score,
            symbol,
            md.get("direction_context_source"),
            context_age_seconds,
            direction_conf,
            extra={
                "event": "ORDERFLOW_TRIGGER_PROMOTED",
                "symbol": symbol,
                "side": best_vote.side,
                "score": raw_score,
                "reason": "direction_context_aligned",
            },
        )
        return promoted, promoted_vote

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
        score_reasons = {str(r) for r in (payload.get("score_reasons") or [])}
        already_blocked_by_strategy = False
        strategy_block_reason = str(payload.get("trigger_block_reason") or "")
        if not bool(payload.get("trigger_conditions_met", True)) or not bool(payload.get("quote_depth_valid", True)):
            already_blocked_by_strategy = True
        if "direction_conflict" in score_reasons:
            already_blocked_by_strategy = True
        if bool(payload.get("stale_data_used")):
            already_blocked_by_strategy = True
        if bool(indicators.get("stale_data_used")) and not already_blocked_by_strategy:
            penalties["stale_data"] = -3.0
        spread_pct = float(payload.get("spread_pct") or indicators.get("spread_pct") or 0.0)
        if spread_pct > float(os.getenv("STRATEGY_MAX_SPREAD_PCT", "28.0")) and not already_blocked_by_strategy:
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
        if already_blocked_by_strategy and strategy_block_reason:
            block_reason = strategy_block_reason
        return score, {
            "trade_quality_score": round(score, 3),
            "trade_quality_components": components,
            "trade_quality_penalties": penalties,
            "quality_block_reason": block_reason,
            "already_blocked_by_strategy": already_blocked_by_strategy,
            "strategy_block_reason": strategy_block_reason or None,
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
def _get_cached_quote_for_eval(hub: t.Any, symbol: str) -> t.Mapping[str, t.Any] | None:
    if hub is None:
        return None
    get_quote = getattr(hub, "get_quote", None)
    if callable(get_quote):
        try:
            quote = get_quote(symbol, allow_pull=False)
            if quote:
                return t.cast(t.Mapping[str, t.Any], quote)
        except TypeError:
            pass
    for method_name in ("get_latest_tick", "get_last_tick"):
        method = getattr(hub, method_name, None)
        if callable(method):
            quote = method(symbol)
            if quote:
                return t.cast(t.Mapping[str, t.Any], dict(quote))
    return None
