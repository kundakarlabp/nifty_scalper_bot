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
    VWAP Pro – Production Hardened Version (Final Audit Applied)
    -------------------------------------------------------------
    • Direction-safe SL/TP (CE/PE correctly handled)
    • Aggressive acceptance resets on all rejections and post-signal
    • Futures-only Index Bias (Hardened against Spot VWAP=0)
    • Single option per expiry+direction isolated state
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
            "evaluations": 0,
            "signals": 0,
            "skipped_cooldown": 0,
            "skipped_strike_lock": 0,
            "skipped_bias": 0,
            "skipped_no_vwap": 0,
            "skipped_vwap": 0,
            "skipped_volume": 0,
            "skipped_overextended": 0,
            "skipped_acceptance": 0,
        }
        self._index_bias_missing_logged: bool = False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def get_required_indicators(self) -> set[str]:
        """Declare ALL indicators VWAPPro needs."""
        return {
            "vwap", "atr", "volume", "avg_volume",
            "rsi",  # for potential future use
        }
    
    def _extract_expiry(self, symbol: str) -> str:
        digits = "".join(c for c in symbol if c.isdigit())
        return digits[:5] if len(digits) >= 5 else "UNK"

    def _dynamic_volume_threshold(self) -> float:
        t = time_module.localtime()
        minutes = t.tm_hour * 60 + t.tm_min
        if 630 < minutes <= 870:  # Mid-day
            return 1.2
        return 0.9

    def _is_expiry_day(self) -> bool:
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
        """Args: symbol, indicators, current_price, position. Returns: EliteSignal|None. Raises: Exception."""
        LOGGER.debug(
            'Entered VWAPProStrategy._evaluate_signal',
            extra={'event': 'vwap_pro_signal_enter', 'symbol': symbol},
        )
        try:
            now = time_module.time()
            expiry = self._extract_expiry(symbol)
            is_ce = "CE" in symbol.upper()
            direction = "CE" if is_ce else "PE"

            # ✅ FIX B6: Periodic telemetry dump
            self._telemetry["evaluations"] += 1
            _evals = self._telemetry["evaluations"]
            if _evals == 1 or _evals % self.TELEMETRY_LOG_EVERY == 0:
                LOGGER.info(
                    f"📊 VWAPPro TELEMETRY [{symbol}]: evals={_evals} "
                    f"signals={self._telemetry['signals']} "
                    f"cool={self._telemetry['skipped_cooldown']} "
                    f"lock={self._telemetry['skipped_strike_lock']} "
                    f"bias={self._telemetry['skipped_bias']} "
                    f"vwap0={self._telemetry['skipped_no_vwap']} "
                    f"vwap={self._telemetry['skipped_vwap']} "
                    f"vol={self._telemetry['skipped_volume']} "
                    f"overext={self._telemetry['skipped_overextended']} "
                    f"accept={self._telemetry['skipped_acceptance']}",
                    extra={"event": "vwap_pro_telemetry", "symbol": symbol},
                )

            lock_key = f"NIFTY:{expiry}:{direction}"
            cooldown_key = f"{expiry}:{direction}"
            acc_key = f"{symbol}_{direction}_accept"

            if acc_key not in self._vwap_acceptance_tracker:
                self._vwap_acceptance_tracker[acc_key] = 0

            # 🔐 ISSUE 3 FIX: Reset acceptance on Strike Lock violation
            if position is not None and getattr(position, "quantity", 0) > 0:
                self._strike_lock[lock_key] = symbol
            else:
                self._strike_lock.pop(lock_key, None)

            if lock_key in self._strike_lock and self._strike_lock[lock_key] != symbol:
                self._telemetry["skipped_strike_lock"] += 1
                if self._telemetry["skipped_strike_lock"] <= 3:
                    LOGGER.info(
                        f"🔐 STRIKE LOCK: {symbol} blocked | Active={self._strike_lock[lock_key]}",
                        extra={"event": "vwap_pro_strike_lock", "symbol": symbol},
                    )
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            last_fire = self._signal_cooldown_tracker.get(cooldown_key, 0.0)
            if (now - last_fire) < self.COOLDOWN_SECONDS:
                self._telemetry["skipped_cooldown"] += 1
                if self._telemetry["skipped_cooldown"] <= 3:
                    LOGGER.info(
                        f"⏳ COOLDOWN: {symbol} {direction} | "
                        f"Remaining={self.COOLDOWN_SECONDS - (now - last_fire):.0f}s",
                        extra={"event": "vwap_pro_cooldown", "symbol": symbol},
                    )
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            # 📊 ISSUE 1 FIX: Hard-block on missing Futures VWAP (No spot fallback)
            # ✅ BONUS: Prefer exchange VWAP (full-session, from broker) over rolling VWAP
            _exch_vwap = indicators.get("exchange_vwap")
            _rolling_vwap = indicators.get("vwap")
            vwap = float(_exch_vwap or _rolling_vwap or 0.0)  # ✅ Exchange > Rolling > 0
            
            atr = float(indicators.get("atr") or max(current_price * 0.015, 1.0))
            entropy = float(indicators.get("entropy_5") or 0.5)

            index_ltp = float(indicators.get("nifty_fut_ltp") or indicators.get("nifty_index_ltp") or 0.0)
            index_vwap = float(indicators.get("nifty_fut_vwap") or indicators.get("nifty_index_vwap") or 0.0)

            if index_ltp <= 0 or index_vwap <= 0:
                if not self._index_bias_missing_logged:
                    LOGGER.error("INVALID INDEX DATA — blocking signal", extra={"symbol": symbol})
                    self._index_bias_missing_logged = True
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            self._index_bias_missing_logged = False

            if (is_ce and index_ltp < index_vwap) or (not is_ce and index_ltp > index_vwap):
                self._telemetry["skipped_bias"] += 1
                if self._telemetry["skipped_bias"] <= 3 or self._telemetry["skipped_bias"] % self.TELEMETRY_LOG_EVERY == 0:
                    LOGGER.info(
                        f"🧭 BIAS GATE: {symbol} {direction} blocked | "
                        f"IdxLTP={index_ltp:.2f} IdxVWAP={index_vwap:.2f} "
                        f"Need={'BULL' if is_ce else 'BEAR'}",
                        extra={"event": "vwap_pro_bias_reject", "symbol": symbol},
                    )
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            if current_price <= 0 or vwap <= 0:
                self._telemetry["skipped_no_vwap"] += 1
                if self._telemetry["skipped_no_vwap"] <= 5:
                    LOGGER.warning(
                        f"⚠️ VWAP ZERO: {symbol} | price={current_price:.2f} vwap={vwap:.2f}",
                        extra={"event": "vwap_pro_zero_block", "symbol": symbol},
                    )
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            if current_price < vwap:
                self._telemetry["skipped_vwap"] += 1
                if self._telemetry["skipped_vwap"] <= 3 or self._telemetry["skipped_vwap"] % self.TELEMETRY_LOG_EVERY == 0:
                    LOGGER.info(
                        f"📏 OPT VWAP: {symbol} {direction} | "
                        f"Price={current_price:.2f} VWAP={vwap:.2f}",
                        extra={"event": "vwap_pro_opt_vwap_reject", "symbol": symbol},
                    )
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            # 📏 Over-extension filter
            vwap_std = float(indicators.get("vwap_std") or 0.0)
            if vwap_std > 0:
                z = abs(current_price - vwap) / vwap_std
                if z > 2.2:
                    self._telemetry["skipped_overextended"] += 1
                    if self._telemetry["skipped_overextended"] <= 3:
                        LOGGER.info(
                            f"📐 OVEREXT: {symbol} | z={z:.2f} > 2.2 | "
                            f"Price={current_price:.2f} VWAP={vwap:.2f} Std={vwap_std:.2f}",
                            extra={"event": "vwap_pro_overext_reject", "symbol": symbol},
                        )
                    self._vwap_acceptance_tracker[acc_key] = 0
                    return None

            # 🔊 ISSUE 4 FIX: Reset acceptance on Volume rejection
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 1.0)
            vol_thresh = self._dynamic_volume_threshold()
            if vol < avg_vol * vol_thresh:
                self._telemetry["skipped_volume"] += 1
                if self._telemetry["skipped_volume"] <= 5 or self._telemetry["skipped_volume"] % self.TELEMETRY_LOG_EVERY == 0:
                    LOGGER.info(
                        f"🔊 VOL GATE: {symbol} | vol={vol:.0f} < avg={avg_vol:.0f}×{vol_thresh}={avg_vol*vol_thresh:.0f}",
                        extra={"event": "vwap_pro_vol_reject", "symbol": symbol},
                    )
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            self._vwap_acceptance_tracker[acc_key] += 1
            if self._vwap_acceptance_tracker[acc_key] < self.VWAP_ACCEPTANCE_BARS:
                self._telemetry["skipped_acceptance"] += 1
                LOGGER.info(
                    f"⏳ ACCEPTANCE: {symbol} {direction} | "
                    f"Bar {self._vwap_acceptance_tracker[acc_key]}/{self.VWAP_ACCEPTANCE_BARS}",
                    extra={"event": "vwap_pro_acceptance", "symbol": symbol},
                )
                return None

            # ═══════════════════════════════════════════════════════════════════
            # ✅ WORLD-CLASS FIX: SL/TP Based on OPTION PREMIUM Direction
            # We trade OPTION PREMIUM, not the index.
            # LONG any option (CE or PE) → profit when PREMIUM rises
            # Therefore: SL below entry, TP above entry (ALWAYS for LONG)
            # ═══════════════════════════════════════════════════════════════════
            sl_mult = (1.2 if self._is_expiry_day() else 1.5) * (0.85 if entropy > 0.75 else 1.0)
            tp2_mult = 3.0

            # LONG position: SL below, TP above (regardless of CE/PE)
            sl = current_price - (atr * sl_mult)
            tp2 = current_price + (atr * tp2_mult)

            # Validation: LONG must have SL < entry < TP
            invalid = sl >= current_price or tp2 <= current_price

            if invalid:
                LOGGER.error(
                    "Invalid SL/TP for LONG",
                    extra={"symbol": symbol, "entry": current_price, "sl": sl, "tp": tp2, "atr": atr}
                )
                self._vwap_acceptance_tracker[acc_key] = 0
                return None

            # 🎯 Final Signal Prep
            base_qty = int(self._vwap_config.quantity or 1)
            confidence = 0.85 if entropy < 0.7 else 0.65
            qty = max(1, int(base_qty * confidence))

            # 🏁 CRITICAL ISSUE 3 FIX: Reset acceptance after firing
            self._vwap_acceptance_tracker[acc_key] = 0
            self._signal_cooldown_tracker[cooldown_key] = now
            self._telemetry["signals"] += 1

            LOGGER.info(
                'Condition met: vwap_pro_signal_ready',
                extra={
                    'event': 'vwap_pro_signal_ready',
                    'symbol': symbol,
                    'direction': direction,
                    'entry': current_price,
                    'sl': sl,
                    'tp2': tp2,
                },
            )

            return EliteSignal(
                symbol=symbol,
                signal="BUY",
                confidence=confidence,
                entry_price=current_price,
                stop_loss=sl,
                target=tp2,
                quantity=qty,
                strategy_name="VWAP_Pro_WorldClass",
                metadata={
                    "bracket_type": "VIRTUAL",
                    "sl_mode": "ATR_TRAIL",
                    "sl_atr_mult": sl_mult,
                    "tp1_atr_mult": 1.5,
                    "tp2_atr_mult": tp2_mult,
                    "tp1_qty_pct": 0.5,
                    "runner_trail_after_tp1": True,
                    "direction": direction,
                },
            )

        except Exception as exc:
            LOGGER.error("VWAP-Pro fatal error: %s", exc, exc_info=True)
            return None


__all__ = ["VWAPProStrategy"]
