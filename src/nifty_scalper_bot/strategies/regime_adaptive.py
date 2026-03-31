"""Regime adaptive strategy blending volatility and flow filters."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Mapping, MutableMapping

from nifty_scalper_bot.config.settings import REGIME_FILTER_BYPASS
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.data.market_regime import (
    MarketRegimeDetector,
    RegimeParameters,
    RegimeSnapshot,
)
from nifty_scalper_bot.strategies.signal_generator import (
    IndicatorMap,
    Position,
    Signal,
    Strategy,
)
from nifty_scalper_bot.utils.env import coalesce_bool
from nifty_scalper_bot.utils.logging import get_logger, log_throttled

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class AdaptiveThresholds:
    """Strategy threshold configuration exposed for tests."""

    min_open_interest: float = 5_000.0
    iv_rank_min: float = 30.0
    iv_rank_max: float = 70.0
    atr_floor: float = 5.0
    atr_ceiling: float = 35.0
    volume_multiplier: float = 1.5
    base_lots: int = 1


class RegimeAdaptiveStrategy(Strategy):
    """Blend regime detection with Kelly sizing across option strikes."""

    def __init__(
        self,
        *,
        market_data_manager: MarketDataManager,
        thresholds: AdaptiveThresholds | None = None,
        description: str | None = None,
        market_regime: MarketRegimeDetector | None = None,
    ) -> None:
        """Create the strategy with market data hooks.

        Args:
            market_data_manager: Data manager used for chain/volume lookups.
            thresholds: Optional overrides for the built-in defaults.
            description: Human readable description exposed to orchestrator.
            market_regime: Optional shared regime detector used for fan-out.
        """
        params = {
            "description": description
            or "Adaptive regime strategy combining OI/IV/ATR filters",
        }
        super().__init__("Regime Adaptive", params)
        self._market_data = market_data_manager
        self._data_hub: Any | None = None  # Injected via set_data_hub
        self._thresholds = thresholds or AdaptiveThresholds()
        self._pnl_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self._logger = LOGGER
        self._regime_filter_bypass = bool(REGIME_FILTER_BYPASS)
        self._enforce_blocklist = bool(
            coalesce_bool("STRATEGY_ENFORCE_BLOCKLIST", default=False)
        )
        self._filter_stats: Dict[str, int] = {
            "passed": 0,
            "blocked": 0,
            "bypassed": 0,
        }
        self._last_regime: str | None = None
        self._strategy_slug = self.name.replace(" ", "_").lower()
        regime_thresholds = self._thresholds
        event_threshold = max(
            regime_thresholds.atr_ceiling * 1.2, regime_thresholds.atr_ceiling + 1.0
        )
        self._regime_detector = market_regime or MarketRegimeDetector(
            RegimeParameters(
                atr_range_threshold=regime_thresholds.atr_floor,
                atr_volatile_threshold=regime_thresholds.atr_ceiling,
                atr_event_threshold=event_threshold,
                volume_trend_multiple=max(regime_thresholds.volume_multiplier, 1.0),
                volume_event_multiple=max(
                    regime_thresholds.volume_multiplier * 1.5, 1.5
                ),
            )
        )

    def set_data_hub(self, data_hub: Any) -> None:
        """Inject DataHub for access to calculated Greeks and IV."""
        self._data_hub = data_hub

    # ------------------------------------------------------------------
    def get_required_indicators(self) -> list[str]:
        """Return indicator keys required by the strategy."""
        return [
            "atr",
            "atr_14",
            "volume",
            "avg_volume",
            "vwap",
            "ema_fast",
            "ema_slow",
            "iv_rank",
            "trend_score",
        ]

    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorMap,
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate adaptive entry signals when filters pass."""
        self._logger.debug(
            "Entered RegimeAdaptiveStrategy.generate_signal",
            extra={"event": "regime_adaptive_generate", "symbol": symbol},
        )
        if position is not None:
            log_throttled(
                self._logger,
                f"no_signal_position_{symbol}",
                "No signal | symbol=%s | rsi=%s | ema_fast=%s | ema_slow=%s"
                % (symbol, indicators.get("rsi"), indicators.get("ema_fast"), indicators.get("ema_slow")),
                interval_sec=60.0,
                level=logging.DEBUG,
                extra={"event": "signal_evaluated"},
            )
            return None
        snapshot = self._build_market_snapshot(symbol)
        regime_snapshot = self._resolve_regime(symbol, indicators, snapshot)
        if regime_snapshot is None or snapshot is None:
            log_throttled(
                self._logger,
                f"no_signal_snapshot_{symbol}",
                "No signal | symbol=%s | rsi=%s | ema_fast=%s | ema_slow=%s"
                % (symbol, indicators.get("rsi"), indicators.get("ema_fast"), indicators.get("ema_slow")),
                interval_sec=60.0,
                level=logging.DEBUG,
                extra={"event": "signal_evaluated"},
            )
            return None
        adjustments = self._resolve_adjustments(indicators, regime_snapshot)
        regime = regime_snapshot.regime
        if regime != self._last_regime:
            self._logger.info(
                "Condition met: regime_transition",
                extra={
                    "event": "regime_transition",
                    "symbol": symbol,
                    "previous": self._last_regime,
                    "current": regime,
                    "confidence": regime_snapshot.confidence,
                    "bypass": self._regime_filter_bypass,
                },
            )
            self._last_regime = regime
        passes_filters, reasons = self._passes_filters(regime, snapshot, indicators)
        if not passes_filters:
            self._logger.debug(
                "Condition met: regime_filters_withheld",
                extra={
                    "event": "regime_filters_withheld",
                    "symbol": symbol,
                    "regime": regime,
                    "reasons": reasons,
                    "stats": dict(self._filter_stats),
                },
            )
            log_throttled(
                self._logger,
                f"no_signal_filters_{symbol}",
                "No signal | symbol=%s | rsi=%s | ema_fast=%s | ema_slow=%s"
                % (symbol, indicators.get("rsi"), indicators.get("ema_fast"), indicators.get("ema_slow")),
                interval_sec=60.0,
                level=logging.DEBUG,
                extra={"event": "signal_evaluated"},
            )
            return None

        block_entries = self._normalize_block_entries(adjustments)
        block_hit = self._strategy_in_blocklist(block_entries)
        high_risk = self._is_high_risk_regime(regime_snapshot)
        enforce_blocklist = self._enforce_blocklist or high_risk
        if block_hit and enforce_blocklist:
            self._logger.info(
                "Condition met: regime_blocklist_veto",
                extra={
                    "event": "regime_blocklist_veto",
                    "symbol": symbol,
                    "regime": regime,
                    "blocklist": list(block_entries),
                    "high_risk": high_risk,
                    "confidence": regime_snapshot.confidence,
                },
            )
            return None

        kelly_fraction = self._kelly_fraction(symbol)
        volatility_adjustment = self._volatility_adjustment(indicators)
        sizing_multiplier = self._extract_sizing_multiplier(adjustments)
        regime_size_multiplier = 0.5 if regime.lower() == "volatile" else 1.0
        raw_quantity = (
            self._thresholds.base_lots
            * kelly_fraction
            * volatility_adjustment
            * sizing_multiplier
            * regime_size_multiplier
        )
        quantity = max(1, int(round(max(raw_quantity, 1.0))))
        block_multiplier = 1.0
        if block_hit and not enforce_blocklist:
            block_multiplier = 0.4
            self._logger.info(
                "Condition met: regime_blocklist_downweight",
                extra={
                    "event": "regime_blocklist_downweight",
                    "symbol": symbol,
                    "regime": regime,
                    "blocklist": list(block_entries),
                    "multiplier": block_multiplier,
                },
            )
        self._logger.info(
            "Condition met: regime_signal_ready",
            extra={
                "event": "regime_signal_ready",
                "symbol": symbol,
                "regime": regime,
                "kelly_fraction": round(kelly_fraction, 4),
                "volatility_adjustment": round(volatility_adjustment, 4),
                "quantity": quantity,
                "confidence": regime_snapshot.confidence,
                "sizing_multiplier": sizing_multiplier,
            "regime_size_multiplier": regime_size_multiplier,
                "regime_size_multiplier": regime_size_multiplier,
                "block_multiplier": block_multiplier,
            },
        )
        reason = f"Regime {regime} filters passed"
        stop_loss = snapshot.get("stop_loss")
        take_profit = snapshot.get("take_profit")
        if stop_loss is None or take_profit is None:
            # Default 2.0 RR if metrics unavailable
            stop_loss = current_price * 0.99
            take_profit = current_price + (current_price - stop_loss) * 2.0
            
        confidence = self._bounded_confidence(0.6 + kelly_fraction / 2.0)
        if block_multiplier != 1.0:
            confidence = self._bounded_confidence(confidence * block_multiplier)
        metadata = {
            "regime": regime,
            "kelly_fraction": kelly_fraction,
            "volatility_adjustment": volatility_adjustment,
            "open_interest": snapshot.get("open_interest"),
            "iv_rank": snapshot.get("iv_rank"),
            "atr": snapshot.get("atr"),
            "volume_ratio": snapshot.get("volume_ratio"),
            "regime_confidence": regime_snapshot.confidence,
            "regime_reason": regime_snapshot.reason,
            "regime_adjustments": adjustments,
            "block_multiplier": block_multiplier,
            "sizing_multiplier": sizing_multiplier,
            "blocklist_enforced": block_hit and enforce_blocklist,
        }
        return Signal(
            action="BUY",
            symbol=symbol,
            quantity=quantity,
            confidence=confidence,
            reason=reason,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    def record_trade_result(self, symbol: str, pnl: float) -> None:
        """Record realised *pnl* used for Kelly sizing adjustments."""
        self._logger.debug(
            "Entered RegimeAdaptiveStrategy.record_trade_result",
            extra={"event": "regime_record_trade", "symbol": symbol},
        )
        history = self._pnl_history[symbol]
        history.append(float(pnl))
        self._logger.info(
            "Condition met: regime_trade_recorded",
            extra={
                "event": "regime_trade_recorded",
                "symbol": symbol,
                "pnl": pnl,
                "history_size": len(history),
            },
        )

    # ------------------------------------------------------------------
    def _build_market_snapshot(self, symbol: str) -> MutableMapping[str, float] | None:
        """Return snapshot including OI/IV/ATR/volume from DataHub or MDM."""

        self._logger.debug(
            'Entered RegimeAdaptiveStrategy._build_market_snapshot',
            extra={'event': 'regime_snapshot_enter', 'symbol': symbol},
        )
        try:
            # 1. Try DataHub (Primary Source - Enriched)
            latest: dict[str, Any] = {}
            iv = None
            if self._data_hub:
                try:
                    iv = self._data_hub.get_iv(symbol)
                    latest = self._data_hub.get_quote(symbol) or {}
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        'Failure in RegimeAdaptiveStrategy._build_market_snapshot: %s',
                        exc,
                        extra={'event': 'regime_snapshot_datahub_error'},
                        exc_info=exc,
                    )

            # 2. Fallback to MarketDataManager (Secondary Source - Raw)
            if not latest:
                latest = self._market_data.get_latest_tick(symbol) or {}

            if not latest:
                return None

            snapshot: MutableMapping[str, float] = {}

            # Extract Open Interest
            oi_value = latest.get('open_interest') or latest.get('oi')
            if isinstance(oi_value, (int, float)):
                snapshot['open_interest'] = float(oi_value)

            # Extract Implied Volatility
            # Use DataHub calculation if available, else raw tick
            if isinstance(iv, (int, float)):
                snapshot['iv_rank'] = float(iv) * 100.0
            else:
                raw_iv = latest.get('implied_volatility')
                if not isinstance(raw_iv, (int, float)):
                    raw = (
                        latest.get('raw')
                        if isinstance(latest.get('raw'), Mapping)
                        else None
                    )
                    if isinstance(raw, Mapping):
                        raw_iv = raw.get('implied_volatility')
                if isinstance(raw_iv, (int, float)):
                    snapshot['iv_rank'] = float(raw_iv) * 100.0

            # Extract ATR
            atr = latest.get('atr')
            if isinstance(atr, (int, float)):
                snapshot['atr'] = float(atr)

            # Extract Volume & Ratio
            volume = (
                latest.get('volume')
                or latest.get('trades')
                or latest.get('volume_traded')
            )
            avg_volume = latest.get('avg_volume')
            if isinstance(volume, (int, float)):
                vol_val = float(volume)
                snapshot['volume'] = vol_val
                if isinstance(avg_volume, (int, float)) and avg_volume > 0:
                    snapshot['volume_ratio'] = vol_val / float(avg_volume)

            # Extract SL/TP hints from tick (if any)
            stop_loss = latest.get('stop_loss')
            take_profit = latest.get('take_profit')
            if isinstance(stop_loss, (int, float)):
                snapshot['stop_loss'] = float(stop_loss)
            if isinstance(take_profit, (int, float)):
                snapshot['take_profit'] = float(take_profit)

            return snapshot
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                'Failure in RegimeAdaptiveStrategy._build_market_snapshot: %s',
                exc,
                extra={'event': 'regime_snapshot_error', 'symbol': symbol},
                exc_info=exc,
            )
            return None

    def _passes_filters(
        self,
        regime: str,
        snapshot: Mapping[str, float],
        indicators: Mapping[str, float | tuple[float, float, float] | None],
    ) -> tuple[bool, list[str]]:
        """Return ``(passed, reasons)`` if the regime/flow filters pass."""
        self._logger.debug(
            "Entered RegimeAdaptiveStrategy._passes_filters",
            extra={"event": "regime_filters_enter", "regime": regime},
        )
        try:
            thresholds = self._thresholds
            reasons: list[str] = []

            oi_value = snapshot.get("open_interest", 0.0)
            if oi_value < thresholds.min_open_interest:
                reasons.append("low_oi")

            volume_ratio = snapshot.get("volume_ratio")
            if volume_ratio is None:
                volume_ratio = self._compute_volume_ratio(indicators)
            
            # Safe cast for volume check
            vol_val = float(volume_ratio) if isinstance(volume_ratio, (int, float)) else None
            if vol_val is None or vol_val < thresholds.volume_multiplier:
                reasons.append("low_volume")

            spread_pct_raw = snapshot.get("spread_pct")
            if spread_pct_raw is None:
                spread_pct_raw = self._extract_float(
                    indicators, ("spread_pct", "option_spread_pct", "liq_spread_pct")
                )
            if isinstance(spread_pct_raw, (int, float)) and float(spread_pct_raw) > 1.0:
                reasons.append("wide_spread")

            atr_candidate = snapshot.get("atr")
            if atr_candidate is None:
                atr_candidate = self._extract_float(indicators, ("atr", "atr_14"))
            
            atr_value = float(atr_candidate) if isinstance(atr_candidate, (int, float)) else None
            if (
                atr_value is None
                or not thresholds.atr_floor <= atr_value <= thresholds.atr_ceiling
            ):
                reasons.append("atr_out_of_range")

            bypassed_regime = False
            if regime.lower() == "consolidation":
                bypassed_regime = True

            if reasons:
                self._filter_stats["blocked"] += 1
                self._logger.debug(
                    "regime_filter_block",
                    extra={
                        "event": "regime_filter_block",
                        "regime": regime,
                        "reasons": reasons,
                        "stats": dict(self._filter_stats),
                        "snapshot": dict(snapshot),
                    },
                )
                return False, reasons

            if bypassed_regime:
                self._filter_stats["bypassed"] += 1
                self._logger.info("Condition met: regime_filters_bypassed", extra={"regime": regime})
            else:
                self._filter_stats["passed"] += 1
                self._logger.debug("regime_filters_passed", extra={"regime": regime})
            return True, []
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in RegimeAdaptiveStrategy._passes_filters: %s",
                exc,
                extra={"event": "regime_filters_error", "regime": regime},
                exc_info=exc,
            )
            self._filter_stats["blocked"] += 1
            return False, ["filter_error"]

    def _resolve_adjustments(
        self,
        indicators: Mapping[str, float | tuple[float, float, float] | None],
        regime_snapshot: RegimeSnapshot,
    ) -> MutableMapping[str, object]:
        """Merge regime adjustments from indicators and snapshot."""
        resolved: MutableMapping[str, object] = {}
        indicator_adjustments = indicators.get("_regime_adjustments")
        if isinstance(indicator_adjustments, Mapping):
            resolved.update(indicator_adjustments)
        snapshot_adjustments = getattr(regime_snapshot, "adjustments", {})
        if isinstance(snapshot_adjustments, Mapping):
            for key, value in snapshot_adjustments.items():
                resolved.setdefault(key, value)
        return resolved

    def _normalize_block_entries(
        self, adjustments: Mapping[str, object]
    ) -> tuple[str, ...]:
        """Return normalised blocklist tokens from *adjustments*."""
        entries = adjustments.get("block") or adjustments.get("blocklist")
        tokens: list[str] = []
        if isinstance(entries, (list, tuple, set)):
            iterable = entries
        elif isinstance(entries, str):
            iterable = [entries]
        else:
            iterable = []
        for entry in iterable:
            token = str(entry).strip()
            if token:
                tokens.append(token)
        return tuple(tokens)

    def _strategy_in_blocklist(self, entries: Iterable[str]) -> bool:
        """Return ``True`` when this strategy matches *entries*."""
        slug = self._strategy_slug
        aliases = {slug, "regime_adaptive", "adaptive"}
        for entry in entries:
            token = entry.strip().lower().replace(" ", "_")
            if token in aliases:
                return True
        return False

    def _is_high_risk_regime(self, snapshot: RegimeSnapshot) -> bool:
        """Return ``True`` when *snapshot* denotes a high-risk regime."""
        regime = snapshot.regime.strip().lower()
        thresholds = {"event": 0.8, "volatile": 0.95}
        threshold = thresholds.get(regime)
        if threshold is None:
            return False
        return snapshot.confidence >= threshold

    def _extract_sizing_multiplier(self, adjustments: Mapping[str, object]) -> float:
        """Return validated sizing multiplier from *adjustments*."""
        candidate = adjustments.get("sizing_multiplier")
        multiplier = 1.0
        if isinstance(candidate, (int, float)):
            multiplier = float(candidate)
        elif isinstance(candidate, str):
            try:
                multiplier = float(candidate)
            except ValueError:
                multiplier = 1.0
        return max(0.1, min(multiplier, 3.0))

    def _kelly_fraction(self, symbol: str) -> float:
        """Return Kelly fraction derived from the rolling trade history."""
        self._logger.debug(
            "Entered RegimeAdaptiveStrategy._kelly_fraction",
            extra={"event": "kelly_fraction_enter", "symbol": symbol},
        )
        try:
            history = list(self._pnl_history[symbol])
            if not history:
                return 0.0

            wins = [pnl for pnl in history if pnl > 0]
            losses = [abs(pnl) for pnl in history if pnl < 0]
            
            # FIX: Safety check for zero losses (no division)
            if not wins or not losses:
                return 0.0

            win_rate = len(wins) / len(history)
            if win_rate <= 0.0 or win_rate >= 1.0:
                return 0.0

            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            if avg_loss <= 0.0: # Should be impossible now, but defensive
                return 0.0

            win_loss_ratio = avg_win / avg_loss
            if win_loss_ratio <= 0.0:
                return 0.0

            kelly_raw = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            kelly_half = kelly_raw * 0.5
            bounded = max(0.0, min(kelly_half, 0.25))
            self._logger.debug(
                "Condition met: kelly_fraction_computed",
                extra={
                    "event": "kelly_fraction_computed",
                    "symbol": symbol,
                    "win_rate": round(win_rate, 4),
                    "win_loss_ratio": round(win_loss_ratio, 4),
                    "kelly": round(bounded, 4),
                },
            )
            return bounded
        except Exception as exc:
            self._logger.error(
                "Failure in RegimeAdaptiveStrategy._kelly_fraction: %s",
                exc,
                extra={"event": "kelly_fraction_error", "symbol": symbol},
                exc_info=exc,
            )
            return 0.0

   
    
    def _volatility_adjustment(
        self, indicators: Mapping[str, float | tuple[float, float, float] | None]
    ) -> float:
        """Return volatility adjustment based on ATR regime."""
        atr_value = self._extract_float(indicators, ("atr", "atr_14"))
        if atr_value is None:
            return 1.0
        thresholds = self._thresholds
        if atr_value < thresholds.atr_floor:
            return 0.5
        if atr_value > thresholds.atr_ceiling:
            return 1.5
        return 1.0

    def _resolve_regime(
        self,
        symbol: str,
        indicators: Mapping[str, float | tuple[float, float, float] | None],
        snapshot: Mapping[str, float] | None,
    ) -> RegimeSnapshot | None:
        """Delegate regime evaluation to the shared detector."""
        self._logger.debug(
            "Entered RegimeAdaptiveStrategy._resolve_regime",
            extra={"event": "regime_resolve_enter", "symbol": symbol},
        )
        try:
            return self._regime_detector.evaluate(symbol, indicators, snapshot)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _resolve_regime: %s",
                exc,
                extra={"event": "regime_resolve_error", "symbol": symbol},
            )
            return None

    def _extract_float(
        self,
        indicators: Mapping[str, float | tuple[float, float, float] | None],
        keys: Iterable[str],
    ) -> float | None:
        """Extract first numeric indicator value for *keys*."""
        for key in keys:
            value = indicators.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, tuple) and value:
                candidate = value[0]
                if isinstance(candidate, (int, float)):
                    return float(candidate)
        return None

    def _compute_volume_ratio(
        self, indicators: Mapping[str, float | tuple[float, float, float] | None]
    ) -> float | None:
        """Return volume spike ratio derived from indicator payload."""
        volume = self._extract_float(indicators, ("volume",))
        avg_volume = self._extract_float(indicators, ("avg_volume",))
        if volume is None or avg_volume is None or avg_volume <= 0:
            return None
        return volume / avg_volume

    # ------------------------------------------------------------------
    def set_regime_filter_bypass(self, enabled: bool) -> None:
        """Toggle the regime filter bypass at runtime."""
        self._logger.debug(
            "Entered RegimeAdaptiveStrategy.set_regime_filter_bypass",
            extra={"event": "regime_set_bypass", "enabled": bool(enabled)},
        )
        self._regime_filter_bypass = bool(enabled)

    def get_regime_filter_bypass(self) -> bool:
        """Return whether the regime filter bypass is enabled."""
        return self._regime_filter_bypass

    def get_regime_filter_stats(self) -> dict[str, int]:
        """Return a copy of the regime filter pass/fail counters."""
        return dict(self._filter_stats)


__all__ = ["AdaptiveThresholds", "RegimeAdaptiveStrategy"]
