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
    VWAP Pro – Production Hardened Version (World-Class Enhanced)
    -------------------------------------------------------------
    • Single option per expiry+direction
    • No laddering
    • No CE+PE overlap
    • Stable cooldown + strike lock
    • ATR-based virtual trailing SL
    • TP1 partial exit + runner
    • Expiry-day tightening
    • Index-bias CE/PE suppression
    • Bracket-safe (virtual)
    """

    MIN_BARS_REQUIRED = 10
    COOLDOWN_SECONDS = 60
    VWAP_ACCEPTANCE_BARS = 2
    TELEMETRY_LOG_EVERY = 5

    __slots__ = (
        "_vwap_config",
        "_signal_cooldown_tracker",
        "_vwap_acceptance_tracker",
        "_strike_lock",
        "_last_expiry",
        "_telemetry",
    )

    def __init__(
        self,
        config: VWAPProStrategyConfig,
        indicator_engine: Any,
    ) -> None:
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._vwap_config = config

        self._signal_cooldown_tracker: Dict[str, float] = {}
        self._vwap_acceptance_tracker: Dict[str, int] = {}
        self._strike_lock: Dict[str, str] = {}
        self._last_expiry: str | None = None

        self._telemetry: Dict[str, int] = {
            "signals": 0,
            "skipped_cooldown": 0,
            "skipped_vwap": 0,
            "skipped_volume": 0,
            "skipped_overextended": 0,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _extract_expiry(self, symbol: str) -> str:
        digits = "".join(c for c in symbol if c.isdigit())
        return digits[:5] if len(digits) >= 5 else "UNK"

    def _dynamic_volume_threshold(self) -> float:
        """Clamped U-shape volume logic (SAFE)."""
        t = time_module.localtime()
        minutes = t.tm_hour * 60 + t.tm_min
        if 630 < minutes <= 870:  # Mid-day
            return 1.2
        return 0.9

    def _is_expiry_day(self) -> bool:
        """Weekly expiry tightening (Thursday)."""
        return time_module.localtime().tm_wday == 3

    # ------------------------------------------------------------------ #
    # Core Signal Logic
    # ------------------------------------------------------------------ #

    def _evaluate_signal(
        self,
        symbol: str,
        indicators: Dict[str, Any],
        current_price: float,
        position: Any | None = None,
    ) -> EliteSignal | None:

        try:
            now = time_module.time()

            expiry = self._extract_expiry(symbol)
            is_ce = "CE" in symbol.upper()
            direction = "CE" if is_ce else "PE"

            lock_key = f"NIFTY:{expiry}:{direction}"
            cooldown_key = f"{expiry}:{direction}"

            # -------------------------------
            # 🔐 Strike Lock Management
            # -------------------------------
            if position is not None and getattr(position, "quantity", 0) > 0:
                self._strike_lock[lock_key] = symbol
            else:
                self._strike_lock.pop(lock_key, None)

            if lock_key in self._strike_lock and self._strike_lock[lock_key] != symbol:
                return None

            # -------------------------------
            # ⏳ Cooldown (expiry+side scoped)
            # -------------------------------
            last_fire = self._signal_cooldown_tracker.get(cooldown_key, 0.0)
            if (now - last_fire) < self.COOLDOWN_SECONDS:
                self._telemetry["skipped_cooldown"] += 1
                return None

            # -------------------------------
            # 📊 Indicator Extraction
            # -------------------------------
            vwap = float(indicators.get("vwap") or 0.0)
            vwap_std = float(indicators.get("vwap_std") or 0.0)
            vwap_15m = float(indicators.get("vwap_15m") or vwap)
            atr = float(indicators.get("atr") or (current_price * 0.015))
            entropy = float(indicators.get("entropy_5") or 0.5)

            index_ltp = float(indicators.get("nifty_index_ltp") or 0.0)
            index_vwap = float(indicators.get("nifty_index_vwap") or 0.0)

            if current_price <= 0 or vwap <= 0:
                return None

            # -------------------------------
            # 🧭 Index Bias (CE/PE suppression)
            # -------------------------------
            if index_ltp > 0 and index_vwap > 0:
                if is_ce and index_ltp < index_vwap:
                    return None
                if not is_ce and index_ltp > index_vwap:
                    return None

            else:
                LOGGER.warning(
                    "Index bias unavailable — proceeding without index confirmation",
                    extra={
                        "event": "index_bias_missing",
                        "symbol": symbol,
                        "index_ltp": index_ltp,
                        "index_vwap": index_vwap,
                    },
                )

            # -------------------------------
            # 📐 VWAP Acceptance (2 bars)
            # -------------------------------
            acc_key = f"{symbol}_accept"
            if acc_key not in self._vwap_acceptance_tracker:
                self._vwap_acceptance_tracker[acc_key] = 0

            if is_ce and current_price < vwap:
                self._telemetry["skipped_vwap"] += 1
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            if not is_ce and current_price > vwap:
                self._telemetry["skipped_vwap"] += 1
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            self._vwap_acceptance_tracker[acc_key] += 1
            if self._vwap_acceptance_tracker[acc_key] < self.VWAP_ACCEPTANCE_BARS:
                return None

            # -------------------------------
            # 📏 Over-extension filter
            # -------------------------------
            if vwap_std > 0:
                z = abs(current_price - vwap) / vwap_std
                if z > 2.2:
                    self._telemetry["skipped_overextended"] += 1
                    return None

            # -------------------------------
            # 🔊 Volume Filter
            # -------------------------------
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 1.0)
            if vol < avg_vol * self._dynamic_volume_threshold():
                self._telemetry["skipped_volume"] += 1
                return None

            # -------------------------------
            # 🎯 Quantity (LOT SAFE)
            # -------------------------------
            base_qty = int(self._vwap_config.quantity or 1)
            confidence = 0.85 if entropy < 0.7 else 0.65

            # Strategy emits intent, execution layer enforces lot sizing
            qty = max(1, int(base_qty * confidence))

            # -------------------------------
            # 🛑 SL / 🎯 TP (Expiry aware)
            # -------------------------------
            sl_mult = (1.2 if self._is_expiry_day() else 1.5) * (0.85 if entropy > 0.75 else 1.0)
            tp1_mult = 1.5
            tp2_mult = 3.0

            sl = current_price - (atr * sl_mult)
            tp = current_price + (atr * tp2_mult)

            # -------------------------------
            # ✅ Register State
            # -------------------------------
            self._signal_cooldown_tracker[cooldown_key] = now
            self._strike_lock[lock_key] = symbol
            self._telemetry["signals"] += 1

            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=confidence,
                entry_price=current_price,
                stop_loss=sl,
                target=tp,
                quantity=qty,
                strategy_name="VWAP_Pro_WorldClass",
                metadata={
                    # ---- Virtual Bracket Plan ----
                    "bracket_type": "VIRTUAL",
                    "sl_mode": "ATR_TRAIL",
                    "sl_atr_mult": sl_mult,
                    "tp1_atr_mult": tp1_mult,
                    "tp2_atr_mult": tp2_mult,
                    "tp1_qty_pct": 0.5,
                    "runner_trail_after_tp1": True,
                    # ---- Context ----
                    "expiry": expiry,
                    "direction": direction,
                    "expiry_day": self._is_expiry_day(),
                    "index_bias": "FOLLOW",
                    "entropy": round(entropy, 2),
                },
            )

        except Exception as exc:
            LOGGER.error("VWAP-Pro fatal error: %s", exc, exc_info=True)
            return None


__all__ = ["VWAPProStrategy"]
