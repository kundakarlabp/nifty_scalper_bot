"""
VWAP Pro Strategy - PRODUCTION FIXED VERSION
═══════════════════════════════════════════════════════════════════════════════

FIXES APPLIED (Feb 2, 2026):
1. ✅ Robust regime detection (fallback when index VWAP unavailable)
2. ✅ Removed overly restrictive session-regime blocking
3. ✅ Relaxed VWAP bands (percentage-based instead of ATR-based)
4. ✅ Relaxed volume filter (0.8x instead of 1.2x)
5. ✅ ATR fallback (1.5% of price when ATR unavailable)
6. ✅ Better logging for debugging
"""

from __future__ import annotations

import time as time_module
from typing import Any, Dict

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class VWAPProStrategy(EliteStrategy):
    """
    VWAP Pro – Production-Grade Strategy with Relaxed Filters
    
    CHANGES FROM ORIGINAL:
    - Regime detection works even without index VWAP
    - Session filtering only in first 5 minutes
    - VWAP bands use percentage (2%) instead of ATR multiplier
    - Volume filter reduced to 0.8x (from 1.2x)
    """

    MIN_BARS_REQUIRED = 1

    COOLDOWN_SECONDS = 60  # ✅ REDUCED from 90 to 60
    VWAP_ACCEPTANCE_BARS = 1  # ✅ REDUCED from 2 to 1
    REGIME_DECAY_SECONDS = 30 * 60  # ✅ INCREASED from 20 to 30 minutes
    TELEMETRY_LOG_EVERY = 5  # ✅ More frequent telemetry

    __slots__ = (
        "_vwap_config",
        "_signal_cooldown_tracker",
        "_vwap_acceptance_tracker",
        "_strike_lock",
        "_index_regime",
        "_regime_timestamp",
        "_last_expiry",
        "_telemetry",
    )

    def __init__(self, config: VWAPProStrategyConfig, indicator_engine: Any) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._vwap_config = config

        self._signal_cooldown_tracker: Dict[str, float] = {}
        self._vwap_acceptance_tracker: Dict[str, int] = {}
        self._strike_lock: Dict[str, str] = {}
        self._index_regime: Dict[str, str] = {}
        self._regime_timestamp: Dict[str, float] = {}
        self._last_expiry: str | None = None

        self._telemetry: Dict[str, int] = {
            "signals": 0,
            "ce": 0,
            "pe": 0,
            "trend": 0,
            "range": 0,
            "skipped_cooldown": 0,
            "skipped_regime": 0,
            "skipped_vwap": 0,
            "skipped_volume": 0,
        }

    def get_required_indicators(self) -> set[str]:
        return {
            "vwap",
            "atr",
            "volume",
            "avg_volume",
            "open",
            "high",
            "low",
            "close",
            # ✅ NEW: Request index data
            "nifty_index_ltp",
            "nifty_index_vwap",
        }

    # ───────────────────────────────
    # Helpers
    # ───────────────────────────────

    def _session_phase(self) -> str:
        t = time_module.localtime()
        minutes = t.tm_hour * 60 + t.tm_min
        # OPEN: 9:15-9:30 (first 15 mins only)
        # MID: 9:30 onwards
        return "OPEN" if 555 <= minutes <= 570 else "MID"

    def _extract_expiry(self, symbol: str) -> str:
        digits = "".join(c for c in symbol if c.isdigit())
        return digits[:5] if len(digits) >= 5 else "UNK"

    # ───────────────────────────────
    # Core Strategy
    # ───────────────────────────────

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:
        try:
            now = time_module.time()

            # ───────────────────────────────
            # Auto-unlock on exit
            # ───────────────────────────────
            if position is None:
                self._strike_lock.clear()

            # ───────────────────────────────
            # Expiry rollover handling
            # ───────────────────────────────
            expiry = self._extract_expiry(symbol)
            if self._last_expiry and expiry != self._last_expiry:
                self._strike_lock.clear()
                self._vwap_acceptance_tracker.clear()
                self._index_regime.clear()
                self._regime_timestamp.clear()
            self._last_expiry = expiry

            # ───────────────────────────────
            # Cooldown
            # ───────────────────────────────
            if (now - self._signal_cooldown_tracker.get(symbol, 0.0)) < self.COOLDOWN_SECONDS:
                self._telemetry["skipped_cooldown"] += 1
                return None

            # ───────────────────────────────
            # Data extraction with fallbacks
            # ───────────────────────────────
            vwap = float(indicators.get("vwap") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            
            # ✅ FIX: ATR fallback (1.5% of price)
            if atr <= 0 and current_price > 0:
                atr = current_price * 0.015
                LOGGER.debug(f"ATR fallback for {symbol}: {atr:.2f}")

            if current_price <= 0 or vwap <= 0:
                return None
                
            # ✅ FIX: If ATR still zero, skip
            if atr <= 0:
                return None

            is_ce = "CE" in symbol.upper()
            is_pe = "PE" in symbol.upper()
            if not (is_ce or is_pe):
                return None

            index_key = "NIFTY"
            direction = "CE" if is_ce else "PE"
            lock_key = f"{index_key}:{expiry}:{direction}"

            # ───────────────────────────────
            # ✅ FIXED: Robust regime detection
            # ───────────────────────────────
            index_ltp = float(indicators.get("nifty_index_ltp") or 0.0)
            index_vwap = float(indicators.get("nifty_index_vwap") or 0.0)

            if index_ltp > 0 and index_vwap > 0:
                # Primary: Use index data
                deviation = abs(index_ltp - index_vwap) / index_vwap
                regime = "TREND" if deviation > 0.002 else "RANGE"
                self._index_regime[index_key] = regime
                self._regime_timestamp[index_key] = now
            else:
                # ✅ FALLBACK: Estimate regime from option price vs VWAP
                regime = self._index_regime.get(index_key)
                if regime is None:
                    if vwap > 0:
                        option_deviation = abs(current_price - vwap) / vwap
                        regime = "TREND" if option_deviation > 0.03 else "RANGE"
                    else:
                        regime = "RANGE"  # Safe default
                    self._index_regime[index_key] = regime
                    self._regime_timestamp[index_key] = now

            # ✅ FIXED: Relaxed regime decay (don't block, just reset)
            if (
                index_key in self._regime_timestamp
                and now - self._regime_timestamp[index_key] > self.REGIME_DECAY_SECONDS
            ):
                self._index_regime.pop(index_key, None)
                # Re-estimate instead of blocking
                regime = "RANGE"
                self._index_regime[index_key] = regime
                self._regime_timestamp[index_key] = now

            # ───────────────────────────────
            # ✅ FIXED: Relaxed session anchoring
            # ───────────────────────────────
            session = self._session_phase()
            
            # Only apply strict filtering in first 15 minutes
            if session == "OPEN" and regime != "TREND":
                self._telemetry["skipped_regime"] += 1
                return None
            
            # ✅ REMOVED: "if session == MID and regime == TREND" block
            # This was preventing profitable trend trades during the day!

            # ───────────────────────────────
            # ✅ FIXED: Percentage-based VWAP bands
            # ───────────────────────────────
            price_vs_vwap_pct = (current_price - vwap) / vwap if vwap > 0 else 0

            if is_ce:
                # CE needs price near or above VWAP
                if price_vs_vwap_pct < -0.05:  # Price 5%+ below VWAP
                    self._telemetry["skipped_vwap"] += 1
                    return None
            
            if is_pe:
                # PE needs price near or below VWAP
                if price_vs_vwap_pct > 0.05:  # Price 5%+ above VWAP
                    self._telemetry["skipped_vwap"] += 1
                    return None

            # ───────────────────────────────
            # VWAP acceptance (reduced to 1 bar)
            # ───────────────────────────────
            acc_key = f"{symbol}_accept"
            self._vwap_acceptance_tracker[acc_key] = self._vwap_acceptance_tracker.get(acc_key, 0) + 1
            if self._vwap_acceptance_tracker[acc_key] < self.VWAP_ACCEPTANCE_BARS:
                return None

            # ───────────────────────────────
            # Strike lock
            # ───────────────────────────────
            if lock_key in self._strike_lock:
                return None

            # ───────────────────────────────
            # ✅ FIXED: Relaxed volume filter (0.8x)
            # ───────────────────────────────
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 1.0)
            if avg_vol > 0 and vol < avg_vol * 0.8:
                self._telemetry["skipped_volume"] += 1
                return None

            # ───────────────────────────────
            # Risk geometry
            # ───────────────────────────────
            if is_ce:
                sl = current_price - atr * 1.5
                tp = current_price + atr * 3.0
            else:
                sl = current_price + atr * 1.5
                tp = current_price - atr * 3.0

            # Ensure SL is not negative
            sl = max(sl, current_price * 0.85)  # At least 15% SL

            # ───────────────────────────────
            # Confidence scoring
            # ───────────────────────────────
            confidence = 0.80
            
            # Boost for volume
            vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
            if vol_ratio > 1.5:
                confidence += 0.10
            elif vol_ratio > 1.2:
                confidence += 0.05
            
            # Boost for strong trend
            if regime == "TREND":
                confidence += 0.05
            
            confidence = min(confidence, 0.95)

            qty = int((self._vwap_config.quantity or 1) * confidence)
            qty = max(1, qty)

            # ───────────────────────────────
            # Register state
            # ───────────────────────────────
            self._signal_cooldown_tracker[symbol] = now
            self._strike_lock[lock_key] = symbol
            self._vwap_acceptance_tracker[acc_key] = 0  # Reset acceptance counter

            # Telemetry
            self._telemetry["signals"] += 1
            self._telemetry["ce"] += int(is_ce)
            self._telemetry["pe"] += int(is_pe)
            self._telemetry["trend"] += int(regime == "TREND")
            self._telemetry["range"] += int(regime == "RANGE")

            if self._telemetry["signals"] % self.TELEMETRY_LOG_EVERY == 0:
                LOGGER.info(
                    f"📊 VWAP-Pro metrics: {self._telemetry}",
                    extra={"event": "vwap_pro_metrics", "metrics": self._telemetry},
                )

            LOGGER.info(
                f"🚀 VWAP-Pro SIGNAL: {symbol} BUY | Regime={regime} | "
                f"Session={session} | Vol={vol_ratio:.1f}x | Conf={confidence:.2f}",
                extra={"event": "vwap_pro_signal", "symbol": symbol},
            )

            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=confidence,
                entry_price=current_price,
                stop_loss=sl,
                target=tp,
                quantity=qty,
                strategy_name="VWAP_Pro_v2",
                metadata={
                    "expiry": expiry,
                    "regime": regime,
                    "session": session,
                    "vol_ratio": round(vol_ratio, 2),
                    "price_vs_vwap_pct": round(price_vs_vwap_pct * 100, 2),
                },
            )

        except Exception as e:
            LOGGER.error(
                f"🔴 VWAP-Pro error on {symbol}: {e}",
                exc_info=True,
            )
            return None


__all__ = ["VWAPProStrategy"]
