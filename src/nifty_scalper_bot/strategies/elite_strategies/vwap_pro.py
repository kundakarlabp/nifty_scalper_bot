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
    VWAP Pro – Final Institutional Grade Strategy

    Features:
    ✔ Expiry rollover handling
    ✔ Expiry-aware strike locking
    ✔ VWAP deviation bands
    ✔ VWAP acceptance window
    ✔ Session anchoring
    ✔ Regime-aware VWAP
    ✔ Time-based regime decay
    ✔ Confidence-weighted sizing
    ✔ Auto-unlock on exit
    ✔ Performance telemetry hooks
    """

    MIN_BARS_REQUIRED = 1

    COOLDOWN_SECONDS = 90
    VWAP_ACCEPTANCE_BARS = 2
    REGIME_DECAY_SECONDS = 20 * 60
    VWAP_BAND_MULTIPLIER = 0.6
    TELEMETRY_LOG_EVERY = 10

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
            "open": 0,
            "mid": 0,
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
        }

    # ───────────────────────────────
    # Helpers
    # ───────────────────────────────

    def _session_phase(self) -> str:
        t = time_module.localtime()
        minutes = t.tm_hour * 60 + t.tm_min
        return "OPEN" if 555 <= minutes <= 600 else "MID"

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
            # Auto-unlock on exit (SAFE)
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
                return None

            vwap = float(indicators.get("vwap") or 0.0)
            atr = float(indicators.get("atr") or 0.0)
            if current_price <= 0 or vwap <= 0 or atr <= 0:
                return None

            is_ce = "CE" in symbol.upper()
            is_pe = "PE" in symbol.upper()
            if not (is_ce or is_pe):
                return None

            index_key = "NIFTY"
            direction = "CE" if is_ce else "PE"
            lock_key = f"{index_key}:{expiry}:{direction}"

            # ───────────────────────────────
            # Regime detection + decay
            # ───────────────────────────────
            index_ltp = float(indicators.get("nifty_index_ltp") or 0.0)
            index_vwap = float(indicators.get("nifty_index_vwap") or 0.0)

            if index_ltp > 0 and index_vwap > 0:
                regime = "TREND" if abs(index_ltp - index_vwap) > (index_vwap * 0.002) else "RANGE"
                self._index_regime[index_key] = regime
                self._regime_timestamp[index_key] = now
            else:
                regime = self._index_regime.get(index_key)

            if (
                index_key in self._regime_timestamp
                and now - self._regime_timestamp[index_key] > self.REGIME_DECAY_SECONDS
            ):
                self._index_regime.pop(index_key, None)
                return None

            # ───────────────────────────────
            # Session anchoring
            # ───────────────────────────────
            session = self._session_phase()
            if session == "OPEN" and regime != "TREND":
                return None
            if session == "MID" and regime == "TREND":
                return None

            # ───────────────────────────────
            # VWAP deviation bands
            # ───────────────────────────────
            upper = vwap + atr * self.VWAP_BAND_MULTIPLIER
            lower = vwap - atr * self.VWAP_BAND_MULTIPLIER

            if is_ce and current_price <= upper:
                return None
            if is_pe and current_price >= lower:
                return None

            # ───────────────────────────────
            # VWAP acceptance
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
            # Volume filter
            # ───────────────────────────────
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 1.0)
            if vol < avg_vol * 1.2:
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

            # ───────────────────────────────
            # Confidence-weighted sizing
            # ───────────────────────────────
            confidence = 0.85
            qty = int((self._vwap_config.quantity or 1) * confidence)
            qty = max(1, qty)

            # ───────────────────────────────
            # Register state
            # ───────────────────────────────
            self._signal_cooldown_tracker[symbol] = now
            self._strike_lock[lock_key] = symbol

            # Telemetry
            self._telemetry["signals"] += 1
            self._telemetry["ce"] += int(is_ce)
            self._telemetry["pe"] += int(is_pe)
            self._telemetry["trend"] += int(regime == "TREND")
            self._telemetry["range"] += int(regime == "RANGE")
            self._telemetry["open"] += int(session == "OPEN")
            self._telemetry["mid"] += int(session == "MID")

            if self._telemetry["signals"] % self.TELEMETRY_LOG_EVERY == 0:
                LOGGER.info(
                    "📊 VWAP-Pro metrics",
                    extra={"event": "vwap_pro_metrics", "metrics": self._telemetry},
                )

            LOGGER.info(
                f"🚀 VWAP-Pro FINAL SIGNAL: {symbol} BUY | {regime} | {session}",
                extra={"event": "vwap_pro_signal"},
            )

            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=confidence,
                entry_price=current_price,
                stop_loss=sl,
                target=tp,
                quantity=qty,
                strategy_name="VWAP_Pro_Ultimate",
                metadata={
                    "expiry": expiry,
                    "regime": regime,
                    "session": session,
                },
            )

        except Exception as e:
            LOGGER.error(
                f"🔴 VWAP-Pro fatal error on {symbol}: {e}",
                exc_info=True,
            )
            return None


__all__ = ["VWAPProStrategy"]
