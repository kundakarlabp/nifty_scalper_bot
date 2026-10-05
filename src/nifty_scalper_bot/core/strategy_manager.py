# fmt: off
# ruff: noqa: E501,I001,F841,E701,E702
# mypy: ignore-errors
"""Strategy arbitration and performance telemetry manager.

Runtime role:
- Evaluates strategies using prepared DataHub/ActiveContractBasket context.
- Propagates selected option/futures context to strategy code.
- Must not select contracts or call broker instruments.

Direction-authority invariant:
- OPTION PREMIUM DATA MUST NEVER AUTHORIZE UNDERLYING DIRECTION.
- Only fresh NIFTY spot/futures context may authorize CE/PE direction.
- Direction, freshness age and source are one atomic observation.
- Fresh spot/futures disagreement fails closed; source order never breaks ties.
- The final option combiner gate independently revalidates direction alignment."""

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
from math import isfinite, sqrt
from statistics import mean, pstdev

from nifty_scalper_bot.config import settings as app_settings
from nifty_scalper_bot.config.regime_ontology import normalize_regime
from nifty_scalper_bot.config.strategy_taxonomy import (
    canonical_signal_family,
    canonical_strategy_role,
    normalize_strategy_name,
)
from nifty_scalper_bot.core.adaptive_calibration import (
    AdaptiveParameterStore,
    WalkForwardOptimizer,
)
from nifty_scalper_bot.core.market_regime import RegimeSnapshot
from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager
from nifty_scalper_bot.core.strategy_vote_policy import (
    independent_same_side_confirmation,
    partition_votes,
)
from nifty_scalper_bot.core.underlying_direction import (
    UnderlyingDirectionObservation,
    UnderlyingDirectionState,
    arbitrate_underlying_direction,
)
from nifty_scalper_bot.infra.metrics import METRICS
from nifty_scalper_bot.instruments.active_contracts import canonical_nifty_future_symbol
from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy
from nifty_scalper_bot.strategies.entry_evidence import infer_option_side
from nifty_scalper_bot.strategies.setup_lifecycle import SetupStage, transition_setup
from nifty_scalper_bot.core.strategy_context_builder import (
    build_strategy_history_context,
)
from nifty_scalper_bot.core.strategy_context_fast_path import _generate_context_only
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.strategies.signal_generator import (
    StrategyManager as _BaseStrategyManager,
)
from nifty_scalper_bot.utils.logging import get_logger, log_state_change, log_throttled
from nifty_scalper_bot.utils.symbols import normalize_symbol
from nifty_scalper_bot.execution.readiness import HistoryReadinessPolicy
from nifty_scalper_bot.execution.quote_readiness import resolve_tick_age_seconds

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


def resolve_canonical_vwap(indicators: t.Mapping[str, t.Any] | None) -> float | None:
    """Single VWAP authority for every strategy-layer consumer.

    Precedence is exchange/session cumulative VWAP first, then the rolling
    proxy. Direction gating, SMC enrichment and VWAPPro previously resolved in
    different orders and could classify the same snapshot in opposite
    directions. Returns None when no VWAP evidence exists — never the current
    price, which manufactures a reclaim out of nothing.
    """
    payload = indicators or {}
    for key in ("exchange_vwap", "session_vwap", "vwap"):
        value = _safe_float_value(payload.get(key))
        if value is not None and value > 0:
            return value
    return None


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
    premium_vwap = resolve_canonical_vwap(enriched)
    if premium_vwap is None:
        volume_sum = sum(max(0.0, bar["volume"]) for bar in ohlcv)
        if volume_sum > 0:
            premium_vwap = sum(((bar["high"] + bar["low"] + bar["close"]) / 3.0) * max(0.0, bar["volume"]) for bar in ohlcv) / volume_sum
        # No volume-backed VWAP exists. Falling back to the current premium made
        # prev_close < price read as a VWAP reclaim on zero evidence, so the
        # feature stays absent instead of being manufactured.

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
    def performance_metrics(
        self, trade_history: t.Sequence[t.Mapping[str, t.Any]]
    ) -> dict[str, float]:
        """Summarize strategy performance from *trade_history* metrics.

        Args:
            trade_history: Ordered trade snapshots with realised PnL values.

        Returns:
            dict[str, float]: Dictionary containing PnL, hit rate, Sharpe, drawdown.

        Raises:
            NotImplementedError: Raised when subclasses omit the override.
        """

        try:
            raise NotImplementedError(
                "Concrete strategies must implement performance_metrics"
            )
        except Exception as exc:  # noqa: BLE001 - intentional propagation
            log.error(
                "Failure in StrategyInterface.performance_metrics: %s",
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

    def performance_metrics(
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

        log.debug("Entered StrategyAdapter.performance_metrics")
        try:
            hook = getattr(self._strategy, "performance_metrics", None)
            if callable(hook):
                result = hook(trade_history)
                if isinstance(result, dict):
                    return result
        except Exception as exc:  # noqa: BLE001 - continue with fallback
            log.error(
                "Failure in StrategyAdapter performance hook: %s",
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
                "Failure in StrategyAdapter.performance_metrics: %s", exc, exc_info=exc
            )
        return {
            "pnl": performance.total_pnl,
            "hit_rate": performance.hit_rate(),
            "sharpe": performance.sharpe_ratio(),
            "drawdown": performance.max_drawdown(),
        }


StrategyFactory = t.Callable[..., StrategyInterface | t.Any]


@dataclass(slots=True)
class StrategyEvidence:
    """One strategy's structural evidence; no numeric vote or confidence."""

    strategy: str
    side: str
    reasons: list[str]
    metadata: dict[str, t.Any]


def signal_to_evidence(signal: Signal, strategy_name: str) -> StrategyEvidence:
    """Normalize a strategy signal into explicit structural evidence."""
    metadata = dict(signal.metadata or {})
    symbol_side = infer_option_side(signal.symbol, metadata)
    metadata_side = str(
        metadata.get("trade_side") or metadata.get("side") or ""
    ).upper()
    if symbol_side in {"CE", "PE"}:
        if metadata_side in {"CE", "PE"} and metadata_side != symbol_side:
            metadata["side_conflict"] = True
            metadata["side_from_metadata"] = metadata_side
            metadata["no_vote_reason"] = "strategy_contract_side_conflict"
        side = symbol_side
    else:
        side = metadata_side or symbol_side
    if signal.action == "HOLD" and side not in {"CE", "PE"}:
        side = "NO_TRADE"

    reason = str(signal.reason or "").strip()
    reason_list = list(
        metadata.get("evidence_reasons")
        or metadata.get("setup_reasons")
        or metadata.get("reasons")
        or []
    )
    if reason and reason not in reason_list:
        reason_list.append(reason)

    strategy_key = normalize_strategy_name(strategy_name)
    metadata["strategy"] = strategy_name
    metadata["strategy_key"] = strategy_key
    metadata["strategy_role"] = canonical_strategy_role(strategy_key)
    metadata["signal_family"] = canonical_signal_family(strategy_key)
    setup_name = str(
        metadata.get("setup_name")
        or metadata.get("setup_type")
        or metadata.get("feature")
        or signal.reason
        or strategy_key
    ).strip()
    metadata["setup_name"] = setup_name or strategy_key
    metadata.setdefault("required_data_present", True)
    metadata["evidence_reasons"] = reason_list
    return StrategyEvidence(
        strategy=strategy_name,
        side=side if side in {"CE", "PE", "NO_TRADE"} else "UNKNOWN",
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
class StrategyPerformance:
    """Rolling performance snapshot for a strategy."""

    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    trades: int = 0
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
            self.trades += 1
            if pnl > 0:
                self.wins += 1
            elif pnl < 0:
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
            trade_count = float(self.trades)
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

        if self.trades == 0:
            return 0.0
        return self.wins / self.trades

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
        """Return alias for win rate aiding external performance reports.

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
            float: Sharpe-like performance ratio derived from recorded returns.

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
            if pnl > 0:
                self.wins += 1
            elif pnl < 0:
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


class StrategyManager(_BaseStrategyManager):
    """Augment the base manager with structural arbitration and performance telemetry."""

    _context_only_fast_path_native = True

    def __init__(
        self,
        strategies: list[t.Any],
        indicator_engine: t.Any,
        position_manager: t.Any,
        data_hub: t.Any | None = None,
        orchestrator: t.Any | None = None,
        futures_symbol: str | None = None,
        *,
        regime_signal_getter: (
            t.Callable[[], t.Mapping[str, t.Any] | None] | None
        ) = None,
        regime_bias_map: t.Mapping[str, t.Mapping[str, float]] | None = None,
        market_regime_manager: MarketRegimeManager | None = None,
    ) -> None:
        """Initialise the structural strategy manager.

        Args:
            strategies: Strategy instances producing signals.
            indicator_engine: Indicator engine shared across strategies.
            position_manager: Position manager used for exposure checks.
            data_hub: Optional data hub providing futures context.
            orchestrator: Optional orchestrator enforcing allocations.
            futures_symbol: Futures symbol used for futures metrics.
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
            data_hub=data_hub,
            orchestrator=orchestrator,
            futures_symbol=futures_symbol,
        )
        self._regime_signal_getter = regime_signal_getter
        self._regime_manager = market_regime_manager
        self._performance: dict[str, StrategyPerformance] = {}
        self._manual_allocations: dict[str, float] = {}
        self._disabled_strategies: set[str] = set()
        self._regime_state = RegimeState()
        self._last_no_signal_decision_by_symbol: dict[str, StrategyNoSignalDecision] = {}
        self._regime_last_key: str | None = None
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

    def get_last_no_signal_decision(
        self, symbol: str
    ) -> StrategyNoSignalDecision | None:
        """Return the latest fail-closed decision for *symbol*, if present."""
        symbol_norm = str(normalize_symbol(symbol) or symbol or "").strip().upper()
        return self._last_no_signal_decision_by_symbol.get(symbol_norm)

    @staticmethod
    def _canonical_no_signal_root_cause(
        indicators: t.Mapping[str, t.Any],
    ) -> tuple[str, str] | None:
        """Return an explicit upstream structural no-trade cause when present."""
        if (
            bool(indicators.get("direction_transition"))
            and str(indicators.get("direction_resolution_reason") or "")
            == "fresh_spot_futures_disagreement"
        ):
            return (
                "context_direction_transition",
                "underlying_direction_transition",
            )
        return None

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
        """Persist one canonical fail-closed strategy decision."""
        symbol_norm = str(normalize_symbol(symbol) or symbol or "").strip().upper()
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
            underlying_direction_bias=(
                str(indicators.get("underlying_direction_bias") or "").upper() or None
            ),
            context_age_seconds=(
                float(indicators.get("context_age_seconds"))
                if indicators.get("context_age_seconds") is not None
                else None
            ),
            trigger_vote_count=trigger_vote_count,
            context_vote_count=context_vote_count,
            selected_ce=str(indicators.get("selected_ce") or "") or None,
            selected_pe=str(indicators.get("selected_pe") or "") or None,
            trace_id=trace_id,
        )

    def get_strategy_mode_profile(self) -> dict[str, t.Any]:
        """Return the structural admission profile for the effective mode."""
        raw_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        live_effective = self._is_live_mode()
        execution_mode = "LIVE" if live_effective else raw_mode
        if execution_mode not in {
            "LIVE",
            "LIVE_SIMULATION",
            "PAPER",
            "SHADOW",
            "SIMULATION",
        }:
            execution_mode = "SHADOW"
        return {
            "mode": execution_mode,
            "raw_mode": raw_mode,
            "live_effective": live_effective,
            "context_promotion": False,
            "single_trigger_requires_confirmation": True,
            "countertrend_requires_reversal_contract": True,
        }

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
        resolved_pnl = float(pnl)
        if not isfinite(resolved_pnl):
            raise ValueError("pnl must be finite")
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
        self._record_performance_observation(
            strategy_name,
            resolved_pnl,
            regime_label=regime_label,
        )
        if metadata:
            log.info(
                "Condition met: strategy_trade_recorded",
                extra={
                    "event": "strategy_trade_recorded",
                    "strategy": strategy_name,
                    "pnl": resolved_pnl,
                    "metadata": dict(metadata),
                },
            )

    def _record_performance_observation(
        self,
        strategy_name: str,
        pnl: float,
        *,
        regime_label: str | None,
    ) -> None:
        perf = self._performance.setdefault(strategy_name, StrategyPerformance())
        perf.record(pnl, regime=regime_label)
        self._adaptive_store.record_trade(strategy_name, pnl)

    def restore_performance_history(
        self,
        outcomes: t.Sequence[t.Mapping[str, t.Any]],
    ) -> int:
        """Restore bounded completed-trade history before live feedback begins."""

        if any(performance.trades > 0 for performance in self._performance.values()):
            raise RuntimeError("performance history already initialised")

        restored = 0
        for outcome in outcomes:
            strategy_name = str(outcome.get("strategy") or "").strip()
            if not strategy_name:
                continue
            try:
                resolved_pnl = float(outcome.get("net_pnl"))
            except (TypeError, ValueError):
                continue
            if not isfinite(resolved_pnl):
                continue
            candidate_regime = outcome.get("regime")
            regime_label = (
                str(candidate_regime).strip()
                if isinstance(candidate_regime, str) and candidate_regime.strip()
                else None
            )
            self._record_performance_observation(
                strategy_name,
                resolved_pnl,
                regime_label=regime_label,
            )
            restored += 1

        log.info(
            "STRATEGY_PERFORMANCE_HISTORY_RESTORED trades=%d strategies=%d",
            restored,
            len(self._performance),
            extra={
                "event": "strategy_performance_history_restored",
                "trades": restored,
                "strategies": len(self._performance),
            },
        )
        return restored

    def notify_entry_accepted(
        self,
        strategy_name: str,
        side: str,
        *,
        setup_id: str | None = None,
    ) -> None:
        """Notify the originating strategy after an entry order is accepted."""
        resolved_name = str(strategy_name or "").strip()
        if not resolved_name:
            return
        for strategy in self._strategies:
            if str(getattr(strategy, "name", "") or "") != resolved_name:
                continue
            hook = getattr(strategy, "notify_entry_accepted", None)
            if not callable(hook):
                return
            try:
                hook(side, setup_id=setup_id)
            except Exception as exc:  # noqa: BLE001 - order is already accepted
                log.error(
                    "Failure in strategy entry-accepted hook: %s",
                    exc,
                    exc_info=exc,
                    extra={
                        "event": "strategy_entry_accepted_hook_error",
                        "strategy": resolved_name,
                        "side": side,
                    },
                )
            return

    def get_allocation_snapshot(self) -> dict[str, float]:
        """Return deterministic manual/equal allocation across enabled strategies.

        Allocation is not inferred from signal quality or historical performance.
        Explicit manual fractions are honored and remaining capacity is shared
        equally among enabled strategies without an override.
        """
        active = [
            str(strategy.name)
            for strategy in self._strategies
            if str(strategy.name) not in self._disabled_strategies
        ]
        if not active:
            return {}
        manual = {
            name: max(0.0, float(self._manual_allocations.get(name, 0.0)))
            for name in active
            if name in self._manual_allocations
        }
        manual_total = sum(manual.values())
        if manual_total >= 1.0 and manual_total > 0.0:
            return {name: manual.get(name, 0.0) / manual_total for name in active}
        unspecified = [name for name in active if name not in manual]
        remaining = max(0.0, 1.0 - manual_total)
        equal = remaining / len(unspecified) if unspecified else 0.0
        return {name: manual.get(name, equal) for name in active}

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
            return
        if fraction < 0:
            raise ValueError("Manual allocation fraction must be non-negative")
        self._manual_allocations[strategy_name] = fraction

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
            return resolved not in self._disabled_strategies
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

        # Final cleaning through the canonical regime ontology.
        canonical_regime = normalize_regime(regime_raw)
        regime = canonical_regime.value.lower()
        try:
            confidence = float(confidence_raw or 0.0)
        except (ValueError, TypeError):
            confidence = 0.0

        self._regime_state = RegimeState(
            regime=regime,
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

        def _history_ema(period: int) -> float | None:
            # Spot and futures are co-equal underlying-direction authorities.
            # Both must receive the same history-backed EMA evidence when the
            # fast-path indicator payload omits precomputed EMA fields.
            if role not in {"spot_context", "futures_context"}:
                return None
            getter = getattr(getattr(self, "_indicator_engine", None), "get_ema", None)
            if not callable(getter):
                return None
            try:
                value = getter(symbol, period=period)
            except TypeError:
                value = getter(symbol, period)
            except Exception as exc:  # noqa: BLE001 - context remains fail-closed
                log.debug(
                    "FUTURES_CONTEXT_EMA_FALLBACK_FAILED symbol=%s period=%s error=%s",
                    symbol,
                    period,
                    exc,
                )
                return None
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        def _history_vwap_slope() -> float | None:
            # Preserve evidence symmetry across the two underlying sources;
            # option-premium history remains excluded from direction authority.
            if role not in {"spot_context", "futures_context"}:
                return None
            getter = getattr(
                getattr(self, "_indicator_engine", None),
                "get_session_vwap_slope",
                None,
            )
            if not callable(getter):
                return None
            try:
                value = getter(symbol, lookback=3)
            except Exception as exc:  # noqa: BLE001 - context remains fail-closed
                log.debug(
                    "FUTURES_CONTEXT_VWAP_SLOPE_FALLBACK_FAILED symbol=%s error=%s",
                    symbol,
                    exc,
                )
                return None
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        context_kind = (
            "price_direction"
            if role == "spot_context"
            else "volume_flow"
            if role == "futures_context"
            else "unknown"
        )
        current_volume = _num("volume")
        current_avg_volume = _num("avg_volume")
        futures_volume_ratio = _num("futures_volume_ratio")
        futures_volume_ratio_source = "indicator" if futures_volume_ratio is not None else "unavailable"
        if (
            role == "futures_context"
            and futures_volume_ratio is None
            and current_volume is not None
            and current_avg_volume is not None
            and current_avg_volume > 0
        ):
            futures_volume_ratio = current_volume / current_avg_volume
            futures_volume_ratio_source = "derived_volume_avg"
        vwap_slope = _num("vwap_slope")
        ema_slope = _num("ema_slope")
        vwap_slope_source = "indicator" if vwap_slope is not None else "unavailable"
        ema_fast = _num("ema_fast", "ema_9", "ema9")
        ema_slow = _num("ema_slow", "ema_21", "ema21")
        ema_50 = _num("ema_50", "ema50")
        ema_fast_source = "indicator" if ema_fast is not None else "unavailable"
        ema_slow_source = "indicator" if ema_slow is not None else "unavailable"
        ema_50_source = "indicator" if ema_50 is not None else "unavailable"
        if role in {"spot_context", "futures_context"}:
            if vwap_slope is None:
                vwap_slope = _history_vwap_slope()
                if vwap_slope is not None:
                    vwap_slope_source = "indicator_engine_history"
            if ema_fast is None:
                ema_fast = _history_ema(9)
                if ema_fast is not None:
                    ema_fast_source = "indicator_engine_history"
            if ema_slow is None:
                ema_slow = _history_ema(21)
                if ema_slow is not None:
                    ema_slow_source = "indicator_engine_history"
            if ema_50 is None:
                ema_50 = _history_ema(50)
                if ema_50 is not None:
                    ema_50_source = "indicator_engine_history"
        direction_inputs = dict(indicators)
        direction_inputs.update(
            futures_volume_ratio=futures_volume_ratio,
            vwap_slope=vwap_slope,
            ema_slope=ema_slope,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_50=ema_50,
        )
        derived_direction, derived_confidence, derived_reasons = self._derive_context_direction(
            direction_inputs,
            role=role,
        )
        stabilised = self._stabilize_context_direction(
            symbol=symbol,
            role=role,
            direction=derived_direction,
            confidence=derived_confidence,
            reasons=derived_reasons,
        )
        derived_direction = stabilised["direction"]
        derived_confidence = stabilised["confidence"]
        derived_reasons = stabilised["reasons"]
        self._context_snapshot_version = int(getattr(self, "_context_snapshot_version", 0) or 0) + 1
        snapshot_version = self._context_snapshot_version
        snapshot = {
            "symbol": symbol, "role": role, "context_kind": context_kind, "timestamp": time.time(),
            "context_snapshot_version": snapshot_version,
            "ltp": _num("ltp", "close", "price"), "close": _num("close", "ltp", "price"),
            "vwap": _num("exchange_vwap", "session_vwap", "vwap"),
            "ema_fast": ema_fast,
            "ema_slow": ema_slow, "ema_50": ema_50,
            "ema_fast_source": ema_fast_source,
            "ema_slow_source": ema_slow_source,
            "ema_50_source": ema_50_source,
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
            "direction_last_conclusive_at": stabilised["last_conclusive_at"],
            "direction_reversal_candidate": stabilised["reversal_candidate"],
            "direction_reversal_since": stabilised["reversal_since"],
            "direction_reversal_observations": stabilised["reversal_observations"],
        }
        self._latest_context_snapshots[role] = snapshot
        direction_available = derived_direction in {"CE", "PE"}
        log_state_change(
            log,
            f"context_direction_state:{role}:{symbol}",
            (direction_available, derived_direction if direction_available else None),
            msg=(
                f"CONTEXT_DIRECTION_STATE role={role} symbol={symbol} "
                f"available={direction_available} direction={derived_direction}"
            ),
            extra={
                "event": "CONTEXT_DIRECTION_STATE",
                "role": role,
                "symbol": symbol,
                "available": direction_available,
                "direction": derived_direction,
            },
        )

    def _stabilize_context_direction(
        self,
        *,
        symbol: str,
        role: str,
        direction: str | None,
        confidence: float,
        reasons: list[str],
    ) -> dict[str, t.Any]:
        """Keep each underlying source stable while preserving fail-closed expiry."""
        now_ts = time.time()
        previous = getattr(self, "_latest_context_snapshots", {}).get(role, {})
        if str(previous.get("symbol") or "") != symbol:
            previous = {}
        previous_direction = str(previous.get("direction_bias") or "").upper()
        if previous_direction not in {"CE", "PE"}:
            previous_direction = ""
        candidate = str(direction or "").upper()
        if candidate not in {"CE", "PE"}:
            candidate = ""
        try:
            previous_confidence = float(
                previous.get("underlying_direction_confidence") or 0.0
            )
        except (TypeError, ValueError):
            previous_confidence = 0.0
        try:
            last_conclusive_at = float(
                previous.get("direction_last_conclusive_at") or now_ts
            )
        except (TypeError, ValueError):
            last_conclusive_at = now_ts

        stable = {
            "direction": direction,
            "confidence": confidence,
            "reasons": list(reasons),
            "last_conclusive_at": now_ts if candidate else last_conclusive_at,
            "reversal_candidate": None,
            "reversal_since": None,
            "reversal_observations": 0,
        }
        if not previous_direction:
            return stable
        if candidate == previous_direction:
            return stable
        if not candidate:
            tie_grace_seconds = max(
                0.0,
                self._env_float("STRATEGY_CONTEXT_TIE_GRACE_SECONDS", 5.0),
            )
            stable["last_conclusive_at"] = last_conclusive_at
            if now_ts - last_conclusive_at <= tie_grace_seconds:
                stable.update(
                    direction=previous_direction,
                    confidence=min(previous_confidence, max(0.50, confidence)),
                    reasons=[*reasons, "direction_tie_hysteresis"],
                )
            return stable

        prior_candidate = str(
            previous.get("direction_reversal_candidate") or ""
        ).upper()
        try:
            prior_since = float(previous.get("direction_reversal_since") or now_ts)
        except (TypeError, ValueError):
            prior_since = now_ts
        try:
            prior_observations = int(
                previous.get("direction_reversal_observations") or 0
            )
        except (TypeError, ValueError):
            prior_observations = 0
        reversal_since = prior_since if prior_candidate == candidate else now_ts
        reversal_observations = (
            prior_observations + 1 if prior_candidate == candidate else 1
        )
        confirm_seconds = max(
            0.0,
            self._env_float("STRATEGY_CONTEXT_REVERSAL_CONFIRM_SECONDS", 15.0),
        )
        try:
            min_observations = max(
                1,
                int(
                    float(
                        os.getenv(
                            "STRATEGY_CONTEXT_REVERSAL_MIN_OBSERVATIONS", "3"
                        )
                        or "3"
                    )
                ),
            )
        except (TypeError, ValueError):
            min_observations = 3
        if (
            now_ts - reversal_since >= confirm_seconds
            and reversal_observations >= min_observations
        ):
            stable["reasons"] = [*reasons, "direction_reversal_confirmed"]
            return stable

        stable.update(
            direction=previous_direction,
            confidence=min(previous_confidence, confidence, 0.60),
            reasons=[*reasons, "direction_reversal_pending"],
            last_conclusive_at=last_conclusive_at,
            reversal_candidate=candidate,
            reversal_since=reversal_since,
            reversal_observations=reversal_observations,
        )
        return stable

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
        close = _f("close") or _f("ltp") or _f("price")
        ltp = _f("ltp") or _f("last_price") or _f("price")
        previous_close = _f("previous_close") or _f("prev_close") or _f("previous_price")
        # Session location must use a session anchor. Generic candle-open data
        # is too short-horizon to define the underlying market direction.
        day_open = _f("day_open") or _f("session_open") or _f("first_ltp")
        recent_ltp_delta = _f("recent_ltp_delta") or _f("net_change") or _f("price_change_pct")
        tick_slope = _f("tick_slope")
        vwap = _f("exchange_vwap") or _f("session_vwap") or _f("vwap")
        ema_fast = _f("ema_fast") or _f("ema_9") or _f("ema9")
        ema_slow = _f("ema_slow") or _f("ema_21") or _f("ema21")
        ema_50 = _f("ema_50") or _f("ema50")
        vwap_slope = _f("vwap_slope")
        ema_slope = _f("ema_slope")
        futures_volume_ratio = _f("futures_volume_ratio")
        reasons: list[str] = []
        location_votes: list[str] = []
        trend_votes: list[str] = []

        def _vote_from_pair(
            lhs: float | None,
            rhs: float | None,
            *,
            up_reason: str,
            down_reason: str,
            bucket: list[str],
        ) -> None:
            if lhs is None or rhs is None:
                return
            if lhs > rhs:
                bucket.append("CE")
                reasons.append(up_reason)
            elif lhs < rhs:
                bucket.append("PE")
                reasons.append(down_reason)

        _vote_from_pair(
            close,
            vwap,
            up_reason="close_above_vwap",
            down_reason="close_below_vwap",
            bucket=location_votes,
        )
        _vote_from_pair(
            close,
            ema_50,
            up_reason="close_above_ema50",
            down_reason="close_below_ema50",
            bucket=location_votes,
        )
        _vote_from_pair(
            close,
            previous_close,
            up_reason="close_above_previous_close",
            down_reason="close_below_previous_close",
            bucket=location_votes,
        )
        _vote_from_pair(
            ltp,
            day_open,
            up_reason="ltp_above_open",
            down_reason="ltp_below_open",
            bucket=location_votes,
        )
        _vote_from_pair(
            ema_fast,
            ema_slow,
            up_reason="ema_fast_above_slow",
            down_reason="ema_fast_below_slow",
            bucket=trend_votes,
        )
        if vwap_slope is not None:
            if vwap_slope > 0:
                trend_votes.append("CE")
                reasons.append("vwap_slope_positive")
            elif vwap_slope < 0:
                trend_votes.append("PE")
                reasons.append("vwap_slope_negative")
        if ema_slope is not None:
            if ema_slope > 0:
                trend_votes.append("CE")
                reasons.append("ema_slope_positive")
            elif ema_slope < 0:
                trend_votes.append("PE")
                reasons.append("ema_slope_negative")
        if role == "futures_context" and futures_volume_ratio is not None and futures_volume_ratio >= 1.0:
            reasons.append("futures_volume_active")

        delta_signal = recent_ltp_delta if recent_ltp_delta not in (None, 0.0) else tick_slope
        tick_side: str | None = None
        if delta_signal is not None:
            if delta_signal > 0:
                tick_side = "CE"
                reasons.append("tick_slope_positive")
            elif delta_signal < 0:
                tick_side = "PE"
                reasons.append("tick_slope_negative")

        def _family_side(votes: list[str], family: str) -> str | None:
            if not votes:
                return None
            sides = set(votes)
            if len(sides) == 1:
                return votes[0]
            reasons.append(f"{family}_conflict")
            return None

        location_side = _family_side(location_votes, "price_location")
        trend_side = _family_side(trend_votes, "trend_structure")
        if location_side is None or trend_side is None:
            return None, 0.0, [*reasons, "direction_requires_location_and_trend"]
        if location_side != trend_side:
            return None, 0.0, [*reasons, "direction_family_conflict"]

        side = trend_side
        # Confidence is provenance telemetry only. Admission depends on the
        # explicit structural agreement above, never on a numeric threshold.
        confidence = 0.80
        if tick_side == side:
            confidence = min(0.90, confidence + 0.05)
        elif tick_side is not None:
            confidence = max(0.50, confidence - 0.05)
            reasons.append("tick_direction_disagrees")
        return side, confidence, reasons

    @staticmethod
    def _context_direction_valid(ctx: t.Mapping[str, t.Any]) -> bool:
        return str(ctx.get("direction_bias") or ctx.get("underlying_direction_bias") or "").upper() in {"CE", "PE"}

    @staticmethod
    def _context_tick_age_seconds(ctx: t.Mapping[str, t.Any]) -> float | None:
        return resolve_tick_age_seconds(ctx)

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
        """Generate a structurally qualified strategy signal.

        Args:
            symbol: Symbol evaluated for trading opportunities.
            current_price: Latest trade price for the symbol.

        Returns:
            Signal | None: Structurally qualified signal or ``None`` when absent.

        Raises:
            None.
        """

        symbol = normalize_symbol(symbol)
        symbol_role = classify_symbol_role(symbol)
        if symbol_role in {"spot_context", "futures_context"}:
            return _generate_context_only(self, symbol, current_price, symbol_role)
        symbol_norm = str(symbol or "").strip().upper()
        self._last_no_signal_decision_by_symbol.pop(symbol_norm, None)
        log.debug(
            "Entered StrategyManager.generate_signal",
            extra={"event": "strategy_generate", "symbol": symbol},
        )
        log.debug(
            "strategy_evaluation_start",
            extra={"event": "strategy_evaluation_start", "symbol": symbol},
        )
        no_signal_reasons: list[str] = []
        error_strategies: list[str] = []
        signals: list[Signal] = []
        signal_votes: list[tuple[Signal, StrategyEvidence]] = []
        disabled: list[str] = []
        empty: list[str] = []
        no_vote_reason_counts: dict[str, int] = {}
        errors: list[str] = []
        disabled_strategies_snapshot: list[str] = []
        signal_action: str | None = None
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
                    "approval_path": None,
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
                age_seconds = resolve_tick_age_seconds(ctx)
                if age_seconds is not None:
                    return age_seconds <= max_context_age
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
            if fut_fresh:
                indicators["futures_context"] = fut_ctx
            # Underlying direction is an independent market-context authority.
            # Option-premium indicators may trigger a setup but must never authorize
            # the NIFTY direction used to choose CE versus PE.
            def _resolve_underlying_observation(
                ctx: t.Mapping[str, t.Any],
                *,
                source: str,
                fresh: bool,
                direction_valid: bool,
                tick_age_s: float | None,
            ) -> UnderlyingDirectionObservation | None:
                if not fresh or not ctx:
                    return None
                bias: t.Any = None
                confidence: t.Any = 0.0
                if direction_valid:
                    bias = ctx.get("direction_bias") or ctx.get("underlying_direction_bias")
                    confidence = ctx.get("underlying_direction_confidence") or 0.0
                else:
                    # Re-derive only from raw source evidence. Removing previously
                    # derived fields prevents a stale/self-referential bias from
                    # being accepted as independent fallback evidence.
                    raw_ctx = dict(ctx)
                    for key in (
                        "direction_bias",
                        "underlying_direction_bias",
                        "underlying_direction_confidence",
                        "direction_context_reasons",
                    ):
                        raw_ctx.pop(key, None)
                    bias, confidence, _reasons = self._derive_context_direction(
                        raw_ctx,
                        role=str(ctx.get("role") or source),
                    )
                bias_norm = str(bias or "").upper()
                if bias_norm not in {"CE", "PE"}:
                    return None
                age_s = tick_age_s
                if age_s is None:
                    try:
                        ts = float(ctx.get("timestamp") or ctx.get("context_timestamp_epoch") or 0.0)
                        age_s = max(0.0, now_ts - ts) if ts > 0 else None
                    except (TypeError, ValueError):
                        age_s = None
                if age_s is None or age_s > max_context_age:
                    return None
                try:
                    confidence_value = float(confidence or 0.0)
                except (TypeError, ValueError):
                    confidence_value = 0.0
                return UnderlyingDirectionObservation(
                    bias=bias_norm,
                    confidence=confidence_value,
                    age_seconds=max(0.0, float(age_s)),
                    source=source,
                )

            # Discard any option-local direction before resolving the underlying.
            # This is deliberate even if upstream copied a context bias into the
            # option snapshot: the authoritative value is reconstructed atomically
            # from the current spot/futures snapshots below.
            indicators.pop("direction_bias", None)
            indicators.pop("underlying_direction_bias", None)
            indicators.pop("underlying_direction_confidence", None)
            indicators.pop("context_age_seconds", None)
            indicators.pop("context_fresh", None)
            indicators.pop("direction_context_source", None)
            indicators.pop("underlying_direction_state", None)
            indicators.pop("direction_transition", None)
            indicators.pop("direction_resolution_reason", None)

            spot_observation = _resolve_underlying_observation(
                spot_ctx,
                source="spot_context",
                fresh=spot_fresh,
                direction_valid=spot_direction_valid,
                tick_age_s=spot_tick_age_s,
            )
            futures_observation = _resolve_underlying_observation(
                fut_ctx,
                source="futures_context",
                fresh=fut_fresh,
                direction_valid=fut_direction_valid,
                tick_age_s=fut_tick_age_s,
            )
            pending_reversal = bool(
                spot_ctx.get("direction_reversal_candidate")
                or fut_ctx.get("direction_reversal_candidate")
            )
            resolution = arbitrate_underlying_direction(
                spot_observation,
                futures_observation,
            )
            indicators["context_snapshot_version"] = max(
                int(spot_ctx.get("context_snapshot_version") or 0),
                int(fut_ctx.get("context_snapshot_version") or 0),
            )
            spot_snapshot_version = int(spot_ctx.get("context_snapshot_version") or 0)
            futures_snapshot_version = int(fut_ctx.get("context_snapshot_version") or 0)
            indicators["context_snapshot_pair"] = (
                spot_snapshot_version,
                futures_snapshot_version,
            )
            indicators["context_snapshot_version_skew"] = abs(
                spot_snapshot_version - futures_snapshot_version
            )
            context_resolved = False
            context_available = False
            direction_context_source: str | None = None
            if pending_reversal:
                pending_ages = [
                    obs.age_seconds
                    for obs in (spot_observation, futures_observation)
                    if obs is not None
                ]
                indicators["context_fresh"] = bool(pending_ages)
                if pending_ages:
                    indicators["context_age_seconds"] = max(pending_ages)
                indicators["underlying_direction_state"] = UnderlyingDirectionState.TRANSITION.value
                indicators["direction_transition"] = True
                indicators["direction_resolution_reason"] = "underlying_reversal_pending"
                indicators["direction_context_source"] = "reversal_transition"
                direction_context_source = "reversal_transition"
                context_available = True
                log_throttled(
                    log,
                    f"direction_reversal_pending:{symbol}",
                    "DIRECTION_CONTEXT_TRANSITION symbol=%s reason=underlying_reversal_pending "
                    "spot_candidate=%s futures_candidate=%s",
                    symbol,
                    spot_ctx.get("direction_reversal_candidate"),
                    fut_ctx.get("direction_reversal_candidate"),
                    interval_sec=30.0,
                    level=logging.WARNING,
                    extra={
                        "event": "DIRECTION_CONTEXT_TRANSITION",
                        "symbol": symbol,
                        "reason": "underlying_reversal_pending",
                        "spot_candidate": spot_ctx.get("direction_reversal_candidate"),
                        "futures_candidate": fut_ctx.get("direction_reversal_candidate"),
                    },
                )
            elif resolution.conflict:
                # Fresh contradictory evidence is a transition, not stale/missing
                # context. Keep execution fail-closed (no CE/PE bias) while
                # preserving freshness/provenance for strategy diagnostics.
                conflict_ages = [
                    obs.age_seconds
                    for obs in (spot_observation, futures_observation)
                    if obs is not None
                ]
                indicators["context_fresh"] = bool(conflict_ages)
                if conflict_ages:
                    indicators["context_age_seconds"] = max(conflict_ages)
                indicators["underlying_direction_state"] = resolution.state.value
                indicators["direction_transition"] = True
                indicators["direction_resolution_reason"] = resolution.reason
                indicators["direction_context_source"] = "spot_futures_transition"
                direction_context_source = "spot_futures_transition"
                context_available = True
                log_throttled(
                    log,
                    f"direction_context_transition:{symbol}",
                    "DIRECTION_CONTEXT_TRANSITION symbol=%s spot_bias=%s futures_bias=%s spot_age_s=%s futures_age_s=%s spot_snapshot_version=%s futures_snapshot_version=%s snapshot_version_skew=%s reason=%s",
                    symbol,
                    spot_observation.bias if spot_observation else None,
                    futures_observation.bias if futures_observation else None,
                    spot_observation.age_seconds if spot_observation else None,
                    futures_observation.age_seconds if futures_observation else None,
                    spot_snapshot_version,
                    futures_snapshot_version,
                    indicators["context_snapshot_version_skew"],
                    resolution.reason,
                    interval_sec=30.0,
                    level=logging.WARNING,
                    extra={
                        "event": "DIRECTION_CONTEXT_TRANSITION",
                        "symbol": symbol,
                        "spot_bias": spot_observation.bias if spot_observation else None,
                        "futures_bias": futures_observation.bias if futures_observation else None,
                    },
                )
            elif (
                resolution.observation is not None
                and (
                    str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper() != "LIVE"
                    or resolution.reason == "spot_futures_agree"
                )
            ):
                observation = resolution.observation
                indicators["direction_bias"] = observation.bias
                indicators["underlying_direction_bias"] = observation.bias
                indicators["underlying_direction_confidence"] = observation.confidence
                indicators["context_age_seconds"] = observation.age_seconds
                indicators["context_fresh"] = True
                indicators["direction_context_source"] = observation.source
                indicators["underlying_direction_state"] = resolution.state.value
                indicators["direction_transition"] = False
                indicators["direction_resolution_reason"] = resolution.reason
                if resolution.confirming_source:
                    indicators["direction_context_confirming_source"] = resolution.confirming_source
                direction_context_source = observation.source
                context_resolved = True
                context_available = True
                log_throttled(
                    log,
                    f"direction_context_resolved:{symbol}",
                    "DIRECTION_CONTEXT_RESOLVED source=%s bias=%s confidence=%.2f age_s=%.2f symbol=%s confirming_source=%s spot_snapshot_version=%s futures_snapshot_version=%s snapshot_version_skew=%s reason=%s",
                    observation.source,
                    observation.bias,
                    observation.confidence,
                    observation.age_seconds,
                    symbol,
                    resolution.confirming_source,
                    spot_snapshot_version,
                    futures_snapshot_version,
                    indicators["context_snapshot_version_skew"],
                    resolution.reason,
                    interval_sec=30.0,
                    level=logging.INFO,
                )
            else:
                indicators.pop("direction_bias", None)
                indicators.pop("underlying_direction_bias", None)
                indicators["context_fresh"] = False
                indicators["underlying_direction_state"] = UnderlyingDirectionState.UNAVAILABLE.value
                indicators["direction_transition"] = False
                indicators["direction_resolution_reason"] = resolution.reason
                indicators["direction_context_source"] = "unresolved"
                direction_context_source = "unresolved"

            if not context_available:
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
                indicators["futures_volume_ratio"] = fut_ctx.get("futures_volume_ratio")
            if fut_fresh and fut_ctx.get("vwap") is not None:
                indicators["futures_vwap"] = fut_ctx.get("vwap")
            if fut_fresh and fut_ctx.get("vwap_slope") is not None:
                indicators["futures_vwap_slope"] = fut_ctx.get("vwap_slope")

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
            adjusted = base_signal
            signals.append(adjusted)
            vote = signal_to_evidence(adjusted, strategy.name)
            if vote.metadata.get("side_conflict"):
                signals.pop()
                reason = "strategy_contract_side_conflict"
                no_vote_reason_counts[reason] = no_vote_reason_counts.get(reason, 0) + 1
                strategy_reasons[strategy.name] = reason
                recorder = getattr(strategy, "_record_evaluation_failure", None)
                if callable(recorder):
                    # A contract violation is a strategy fault, not a no-trade.
                    recorder(ValueError(reason))
                log.error(
                    "STRATEGY_CONTRACT_SIDE_CONFLICT strategy=%s symbol=%s "
                    "metadata_side=%s contract_side=%s",
                    strategy.name,
                    symbol,
                    vote.metadata.get("side_from_metadata"),
                    vote.side,
                    extra={
                        "event": "STRATEGY_CONTRACT_SIDE_CONFLICT",
                        "strategy": strategy.name,
                        "symbol": symbol,
                    },
                )
                continue
            vote.metadata["regime_name"] = str(regime_name or "UNKNOWN")
            vote.metadata["regime_routing_mode"] = "observe_only"
            signal_votes.append((adjusted, vote))

        # Preserve every structurally valid vote. MAX_STRATEGY_VOTES is kept
        # as observability only: numeric ranking must not decide which strategy
        # evidence is discarded before consensus.
        if len(signal_votes) > max_votes:
            log.info(
                "STRATEGY_VOTE_COUNT_ABOVE_REFERENCE symbol=%s votes=%s reference_max=%s",
                symbol,
                len(signal_votes),
                max_votes,
                extra={
                    "event": "STRATEGY_VOTE_COUNT_ABOVE_REFERENCE",
                    "symbol": symbol,
                    "votes": len(signal_votes),
                    "reference_max": max_votes,
                },
            )

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
            # Preserve explicit upstream context causes before strategy
            # no-vote reasons so a healthy fail-closed transition is not
            # misclassified as alpha failure.
            canonical_cause = self._canonical_no_signal_root_cause(indicators)
            if canonical_cause is not None:
                category, primary_reason = canonical_cause
            elif no_vote_reason_counts.get("underlying_direction_conflict"):
                primary_reason = "underlying_direction_conflict"
                category = "context_direction_conflict"
            elif no_vote_reason_counts.get("tick_direction_missing_or_neutral"):
                primary_reason = "tick_direction_missing_or_neutral"
                category = "option_tick_direction_neutral"
            elif no_vote_reason_counts.get("negative_premium_flow"):
                primary_reason = "negative_premium_flow"
                category = "premium_flow_negative"
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
                # Quantity belongs to the risk engine. Scaling it here could
                # never protect capital anyway: the requested size is one lot,
                # and max(1, round(1 * 0.6)) is still one lot, so a defensive
                # regime multiplier was a no-op while an expansive one could
                # still raise size outside the 2% risk owner. The regime scale
                # stays in metadata as sizing evidence for the risk layer.
                combined = Signal(
                    action=combined.action,
                    symbol=combined.symbol,
                    quantity=combined.quantity,
                    confidence=combined.confidence,
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
                    "Condition met: structural_signal_ready",
                    extra={
                        "event": "structural_signal_ready",
                        "symbol": symbol,
                        "action": combined.action,
                        "confidence": combined.confidence,
                        "quantity": combined.quantity,
                    },
                )
                self._observability_counters["signals_generated"] += 1
                self._avg_kelly_window.append(
                    float(dict(combined.metadata).get("kelly_fraction", 0.0))
                )
                self._emit_metrics_snapshot()
                exit_result = "signal"
                signal_action = combined.action
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
            (execution_mode == "LIVE_SIMULATION" or (execution_mode == "LIVE" and enable_live))
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

    def _context_vote_is_timestamped(
        self,
        vote: StrategyEvidence,
        *,
        max_age_s: float | None = None,
    ) -> bool:
        """Return whether context evidence proves it is current."""
        raw = (vote.metadata or {}).get("vote_timestamp")
        try:
            stamped = float(raw)
        except (TypeError, ValueError):
            return False
        if not isfinite(stamped) or stamped <= 0:
            return False
        max_age = max(0.0, self._env_float("CONTEXT_VOTE_MAX_AGE_SEC", 30.0))
        if max_age_s is not None:
            max_age = min(max_age, max(0.0, max_age_s))
        return 0.0 <= (time.time() - stamped) <= max_age

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
        signal_votes: list[tuple[Signal, StrategyEvidence]],
        indicators: t.Mapping[str, t.Any],
        combined: Signal | None,
        blocked_reason: str | None,
        no_vote_reason_counts: t.Mapping[str, int] | None = None,
    ) -> None:
        """Log structural strategy-arbitration blockers."""
        indicator_map = dict(indicators or {})
        trigger_votes, context_votes, rejected = partition_votes(signal_votes)
        combined_md = (
            dict(getattr(combined, "metadata", {}) or {})
            if combined is not None
            else {}
        )
        selected_ok, selected_meta = self._is_selected_or_near_atm(
            symbol, combined_md, indicator_map
        )
        log.debug(
            "STRATEGY_COMBINER_BLOCKER symbol=%s trigger_count=%s context_count=%s "
            "rejected_setups=%s selected_ok=%s selected_reason=%s blocked_reason=%s",
            symbol,
            len(trigger_votes),
            len(context_votes),
            rejected,
            selected_ok,
            selected_meta.get("selected_ok_reason"),
            blocked_reason
            or combined_md.get("blocked_reason")
            or "combined_not_approved",
            extra={
                "event": "STRATEGY_COMBINER_BLOCKER",
                "symbol": symbol,
                "trigger_count": len(trigger_votes),
                "context_count": len(context_votes),
                "rejected_setups": rejected,
                "selected_ok": selected_ok,
                "selected_ok_reason": selected_meta.get("selected_ok_reason"),
                "direction_bias": indicator_map.get("underlying_direction_bias")
                or indicator_map.get("direction_bias"),
                "direction_state": indicator_map.get("underlying_direction_state"),
                "blocked_reason": blocked_reason
                or combined_md.get("blocked_reason")
                or "combined_not_approved",
                "no_vote_reason_counts": dict(no_vote_reason_counts or {}),
                "combined_present": combined is not None,
                "combined_is_approved": bool(combined_md.get("is_approved")),
            },
        )

    def _combine_strategy_votes(
        self,
        *,
        symbol: str,
        signals: list[tuple[Signal, StrategyEvidence]],
        indicators: t.Mapping[str, t.Any],
        no_vote_reason_counts: t.Mapping[str, int] | None = None,
    ) -> Signal | None:
        """Approve only structurally valid, direction-aligned entry evidence."""
        symbol_norm = str(symbol or "").strip().upper()
        indicator_map = dict(indicators or {})

        def _record_no_signal(
            category: str,
            reason_text: str,
            blocked_at: str,
            *,
            trigger_vote_count: int = 0,
            context_vote_count: int = 0,
        ) -> None:
            self._last_no_signal_decision_by_symbol[symbol_norm] = StrategyNoSignalDecision(
                symbol=symbol_norm,
                eval_id=str(
                    indicator_map.get("eval_id") or indicator_map.get("trace_id") or ""
                )
                or None,
                final_block_reason=reason_text,
                category=category,
                reason=reason_text,
                blocked_at=blocked_at,
                no_vote_reason_counts=dict(no_vote_reason_counts or {}),
                strategy_reasons={},
                direction_bias=str(indicator_map.get("direction_bias") or "").upper()
                or None,
                underlying_direction_bias=str(
                    indicator_map.get("underlying_direction_bias") or ""
                ).upper()
                or None,
                context_age_seconds=(
                    float(indicator_map.get("context_age_seconds"))
                    if indicator_map.get("context_age_seconds") is not None
                    else None
                ),
                trigger_vote_count=trigger_vote_count,
                context_vote_count=context_vote_count,
                selected_ce=str(indicator_map.get("selected_ce") or "") or None,
                selected_pe=str(indicator_map.get("selected_pe") or "") or None,
                trace_id=str(indicator_map.get("trace_id") or "") or None,
            )

        if not signals:
            return None

        for signal, _evidence in signals:
            if signal.action in {"CLOSE_LONG", "CLOSE_SHORT"}:
                metadata = dict(signal.metadata or {})
                metadata["approval_path"] = "close_signal"
                metadata["is_approved"] = True
                return Signal(
                    action=signal.action,
                    symbol=signal.symbol,
                    quantity=signal.quantity,
                    confidence=signal.confidence,
                    reason=signal.reason,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    metadata=metadata,
                )

        trigger_votes, context_votes, rejected_setups = partition_votes(signals)
        if rejected_setups:
            log.info(
                "STRUCTURAL_SETUP_REJECTED symbol=%s rejected=%s",
                symbol_norm,
                rejected_setups,
                extra={
                    "event": "STRUCTURAL_SETUP_REJECTED",
                    "symbol": symbol_norm,
                    "rejected": rejected_setups,
                },
            )
        if not trigger_votes:
            _record_no_signal(
                "strategy_no_trigger",
                "no_setup_valid_trigger",
                "strategy_structural_contract",
                context_vote_count=len(context_votes),
            )
            return None

        trigger_sides = {
            str(evidence.side or "").upper()
            for _signal, evidence in trigger_votes
            if str(evidence.side or "").upper() in {"CE", "PE"}
        }
        if len(trigger_sides) != 1:
            _record_no_signal(
                "strategy_direction_conflict",
                "trigger_side_conflict",
                "strategy_structural_contract",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
            )
            return None
        side = next(iter(trigger_sides))

        direction = str(
            indicator_map.get("underlying_direction_bias")
            or indicator_map.get("direction_bias")
            or ""
        ).upper()
        direction_state = str(
            indicator_map.get("underlying_direction_state") or ""
        ).upper()
        if direction not in {"CE", "PE"}:
            _record_no_signal(
                "context_direction_unavailable",
                "underlying_direction_unavailable",
                "direction_contract",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
            )
            return None
        if direction_state == UnderlyingDirectionState.TRANSITION.value:
            _record_no_signal(
                "context_direction_transition",
                "underlying_direction_transition",
                "direction_contract",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
            )
            return None

        # Continuation-only live admission. Counter-trend entries require a future,
        # separately modeled reversal contract based on underlying structure; option
        # premium behaviour or OrderFlow may never override underlying direction.
        if side != direction:
            _record_no_signal(
                "countertrend_block",
                "countertrend_requires_structural_reversal_contract",
                "direction_contract",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
            )
            log.info(
                "COUNTERTREND_ENTRY_BLOCKED symbol=%s trigger_side=%s underlying=%s",
                symbol_norm,
                side,
                direction,
                extra={
                    "event": "COUNTERTREND_ENTRY_BLOCKED",
                    "symbol": symbol_norm,
                    "trigger_side": side,
                    "underlying_direction": direction,
                    "direction_state": direction_state,
                },
            )
            return None

        selected_ok, selected_meta = self._is_selected_or_near_atm(
            symbol_norm, dict(trigger_votes[0][0].metadata or {}), indicator_map
        )
        if not selected_ok:
            _record_no_signal(
                "candidate_not_selected_or_near_atm",
                "not_selected_or_near_atm",
                "candidate_contract",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
            )
            return None

        opposite_context = [
            evidence
            for _signal, evidence in context_votes
            if str(evidence.side or "").upper() in {"CE", "PE"}
            and str(evidence.side or "").upper() != side
            and self._context_vote_is_timestamped(evidence)
            and bool((evidence.metadata or {}).get("context_quality_eligible"))
            and bool((evidence.metadata or {}).get("effective_context_conflict"))
        ]
        if opposite_context:
            _record_no_signal(
                "context_conflict",
                "fresh_opposing_context",
                "context_contract",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
            )
            return None

        same_side_context = [
            evidence
            for _signal, evidence in context_votes
            if str(evidence.side or "").upper() == side
            and self._context_vote_is_timestamped(evidence)
            and bool((evidence.metadata or {}).get("context_quality_eligible"))
            and bool((evidence.metadata or {}).get("effective_context_alignment"))
        ]

        independent_trigger_confirmation, confirming_trigger_strategies = (
            independent_same_side_confirmation(trigger_votes)
        )
        if not independent_trigger_confirmation and not same_side_context:
            _record_no_signal(
                "strategy_confirmation_missing",
                "independent_confirmation_missing",
                "context_contract",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
            )
            return None

        primary_signal, primary_evidence = trigger_votes[0]
        metadata = dict(primary_signal.metadata or {})
        approval_path = (
            "aligned_trigger_consensus"
            if independent_trigger_confirmation
            else "single_trigger_context_confirmed"
        )
        context_strategies = sorted(
            {str(evidence.strategy) for evidence in same_side_context}
        )
        metadata.update(selected_meta)
        metadata.update(
            {
                "is_approved": True,
                "approval_path": approval_path,
                "strategy": primary_evidence.strategy,
                "strategy_name": primary_evidence.strategy,
                "strategy_role": canonical_strategy_role(primary_evidence.strategy),
                "signal_family": canonical_signal_family(primary_evidence.strategy),
                "requires_runner_execution_validation": True,
                "direction_contract": {
                    "passed": True,
                    "side": side,
                    "underlying_direction": direction,
                    "underlying_state": direction_state or None,
                    "source": indicator_map.get("direction_context_source"),
                    "age_seconds": indicator_map.get("context_age_seconds"),
                },
                "setup_contract": {
                    "passed": True,
                    "strategy": primary_evidence.strategy,
                    "setup_id": metadata.get("setup_id"),
                    "reasons": list(primary_evidence.reasons),
                },
                "confirmation_contract": {
                    "passed": True,
                    "trigger_consensus": independent_trigger_confirmation,
                    "confirming_trigger_strategies": confirming_trigger_strategies,
                    "context_strategies": context_strategies,
                },
                "confirming_trigger_strategies": confirming_trigger_strategies,
                "context_confirmation_strategies": context_strategies,
            }
        )
        transition_setup(
            SetupStage.MANAGER_QUALIFIED,
            metadata,
            strategy=primary_evidence.strategy,
            symbol=primary_signal.symbol,
            side=side,
            reason=approval_path,
        )
        log.info(
            "STRUCTURAL_CANDIDATE_QUALIFIED symbol=%s strategy=%s side=%s approval_path=%s",
            symbol_norm,
            primary_evidence.strategy,
            side,
            approval_path,
            extra={
                "event": "STRUCTURAL_CANDIDATE_QUALIFIED",
                "symbol": symbol_norm,
                "strategy": primary_evidence.strategy,
                "side": side,
                "approval_path": approval_path,
                "direction_contract": metadata["direction_contract"],
                "setup_contract": metadata["setup_contract"],
                "confirmation_contract": metadata["confirmation_contract"],
            },
        )
        return Signal(
            action=primary_signal.action,
            symbol=primary_signal.symbol,
            quantity=primary_signal.quantity,
            confidence=1.0,
            reason=primary_signal.reason,
            stop_loss=primary_signal.stop_loss,
            take_profit=primary_signal.take_profit,
            metadata=metadata,
        )

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
    "StrategyEvidence",
    "signal_to_evidence",
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
