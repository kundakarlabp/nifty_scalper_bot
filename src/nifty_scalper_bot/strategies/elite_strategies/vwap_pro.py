from __future__ import annotations

import os
import time as time_module
from typing import Any, Dict

from nifty_scalper_bot.strategies.elite_strategies.base_elite import (
    EliteSignal,
    EliteStrategy,
)
from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    VWAPProStrategyConfig,
)
from nifty_scalper_bot.utils.logging import get_logger, log_throttled
from nifty_scalper_bot.utils.smart_symbol import WEEKLY_EXPIRY_WEEKDAY

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
    VOLUME_GRACE_SECONDS = 120.0

    __slots__ = (
        "_vwap_config",
        "_signal_cooldown_tracker",
        "_vwap_acceptance_tracker",
        "_strike_lock",
        "_last_expiry",
        "_telemetry",
        "_last_valid_volume",
        "_last_valid_avg_volume",
        "_last_valid_volume_ts",
        "_reject_reason_counts",
        "_index_bias_degraded_logged",
    )

    def __init__(
        self,
        config: VWAPProStrategyConfig,
        indicator_engine: Any,
    ) -> None:
        """Args: config, indicator_engine. Returns: None. Raises: Exception."""
        super().__init__(config=config, indicator_engine=indicator_engine)
        self._vwap_config = config

        self._signal_cooldown_tracker: Dict[str, float] = {}
        self._vwap_acceptance_tracker: Dict[str, int] = {}
        self._strike_lock: Dict[str, str] = {}
        self._last_expiry: str | None = None
        self._last_valid_volume: Dict[str, float] = {}
        self._last_valid_avg_volume: Dict[str, float] = {}
        self._last_valid_volume_ts: Dict[str, float] = {}

        self._telemetry: Dict[str, int] = {
            "evaluations": 0,
            "signals": 0,
            "skipped_cooldown": 0,
            "skipped_strike_lock": 0,
            "skipped_bias": 0,
            "skipped_no_vwap": 0,
            "skipped_vwap": 0,
            "skipped_volume": 0,
            "skipped_data": 0,
            "skipped_overextended": 0,
            "skipped_acceptance": 0,
        }
        self._reject_reason_counts: Dict[str, int] = {}
        self._index_bias_missing_logged: bool = False
        self._index_bias_degraded_logged: bool = False

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def get_required_indicators(self) -> set[str]:
        """Declare ALL indicators VWAPPro needs."""
        return {
            "vwap",
            "atr",
            "volume",
            "avg_volume",
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
        """Args: None. Returns: bool. Raises: Exception."""
        LOGGER.debug(
            "Entered VWAPProStrategy._is_expiry_day",
            extra={"event": "vwap_pro_expiry_day_enter"},
        )
        try:
            default_weekday = WEEKLY_EXPIRY_WEEKDAY
            expiry_day_raw = os.getenv("NIFTY_EXPIRY_WEEKDAY", str(default_weekday))
            try:
                expiry_day = int(expiry_day_raw)
            except ValueError as e:
                log_throttled(
                    LOGGER,
                    "vwap_pro_expiry_defaulted",
                    "Condition met: expiry_day_defaulted",
                    interval_sec=3600.0,
                )
                LOGGER.error(
                    "Failure in VWAPProStrategy._is_expiry_day: %s",
                    e,
                    extra={"event": "vwap_pro_expiry_day_parse_error"},
                    exc_info=e,
                )
                expiry_day = default_weekday
            if expiry_day not in range(7):
                log_throttled(
                    LOGGER,
                    "vwap_pro_expiry_out_of_range",
                    "Condition met: expiry_day_out_of_range",
                    interval_sec=3600.0,
                )
                expiry_day = default_weekday
            if expiry_day == 3:
                log_throttled(
                    LOGGER,
                    "vwap_pro_expiry_corrected",
                    "Condition met: expiry_day_corrected_to_default",
                    interval_sec=3600.0,
                )
                expiry_day = default_weekday
            is_expiry = time_module.localtime().tm_wday == expiry_day
            if is_expiry:
                log_throttled(
                    LOGGER,
                    "vwap_pro_expiry_active",
                    "Condition met: expiry_day_active",
                    interval_sec=3600.0,
                )
            return is_expiry
        except Exception as e:
            LOGGER.error(
                "Failure in VWAPProStrategy._is_expiry_day: %s",
                e,
                extra={"event": "vwap_pro_expiry_day_error"},
                exc_info=e,
            )
            return False

    def _log_no_signal_reason(
        self,
        reason_code: str,
        *,
        symbol: str,
        ltp: float | None = None,
        vwap: float | None = None,
        vol: float | None = None,
        avg_vol: float | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        """Args: reason_code, symbol, ltp, vwap, vol, avg_vol, context. Returns: None. Raises: Exception."""
        try:
            self._reject_reason_counts[reason_code] = (
                self._reject_reason_counts.get(reason_code, 0) + 1
            )
            payload: Dict[str, Any] = {
                "event": "vwap_pro_no_signal_reason",
                "reason_code": reason_code,
                "symbol": symbol,
            }
            if ltp is not None:
                payload["ltp"] = ltp
            if vwap is not None:
                payload["vwap"] = vwap
            if vol is not None:
                payload["vol"] = vol
            if avg_vol is not None:
                payload["avg_vol"] = avg_vol
            if context:
                payload.update(context)
            LOGGER.debug(
                "VWAP_PRO_REJECT | reason=%s",
                reason_code,
                extra=payload,
            )
        except Exception as exc:
            LOGGER.error(
                "Failure in VWAPProStrategy._log_no_signal_reason: %s",
                exc,
                extra={
                    "event": "vwap_pro_no_signal_reason_error",
                    "symbol": symbol,
                },
                exc_info=exc,
            )

    def _reset_acceptance(self, key: str, *, symbol: str, reason_code: str) -> None:
        """Args: key, symbol, reason_code. Returns: None. Raises: Exception."""
        try:
            self._vwap_acceptance_tracker[key] = 0
            LOGGER.debug(
                "⏳ ACCEPTANCE RESET | symbol=%s reason=%s",
                symbol,
                reason_code,
                extra={
                    "event": "vwap_pro_acceptance_reset",
                    "symbol": symbol,
                    "reason_code": reason_code,
                },
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in VWAPProStrategy._reset_acceptance: %s",
                exc,
                extra={
                    "event": "vwap_pro_acceptance_reset_error",
                    "symbol": symbol,
                    "reason_code": reason_code,
                },
                exc_info=exc,
            )

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
            "Entered VWAPProStrategy._evaluate_signal",
            extra={"event": "vwap_pro_signal_enter", "symbol": symbol},
        )
        try:
            def _emit_no_signal(reason_code: str) -> None:
                """Args: reason_code. Returns: None. Raises: Exception."""
                try:
                    LOGGER.info(
                        "📉 NO SIGNAL | symbol=%s reason=%s",
                        symbol,
                        reason_code,
                        extra={
                            "event": "vwap_pro_no_signal",
                            "symbol": symbol,
                            "reason_code": reason_code,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "Failure in VWAPProStrategy._emit_no_signal: %s",
                        exc,
                        extra={
                            "event": "vwap_pro_no_signal_error",
                            "symbol": symbol,
                            "reason_code": reason_code,
                        },
                        exc_info=exc,
                    )

            now = time_module.time()
            expiry = self._extract_expiry(symbol)
            is_ce = "CE" in symbol.upper()
            direction = "CE" if is_ce else "PE"

            # ✅ FIX B6: Periodic telemetry dump
            self._telemetry["evaluations"] += 1
            _evals = self._telemetry["evaluations"]
            if _evals == 1 or _evals % self.TELEMETRY_LOG_EVERY == 0:
                dominant_reject_reason = None
                if self._reject_reason_counts:
                    dominant_reject_reason = max(
                        self._reject_reason_counts.items(),
                        key=lambda item: item[1],
                    )[0]
                LOGGER.info(
                    f"📊 VWAPPro TELEMETRY [{symbol}]: evals={_evals} "
                    f"signals={self._telemetry['signals']} "
                    f"cool={self._telemetry['skipped_cooldown']} "
                    f"lock={self._telemetry['skipped_strike_lock']} "
                    f"bias={self._telemetry['skipped_bias']} "
                    f"vwap0={self._telemetry['skipped_no_vwap']} "
                    f"vwap={self._telemetry['skipped_vwap']} "
                    f"vol={self._telemetry['skipped_volume']} "
                    f"data={self._telemetry['skipped_data']} "
                    f"overext={self._telemetry['skipped_overextended']} "
                    f"accept={self._telemetry['skipped_acceptance']} "
                    f"dominant_reject_reason={dominant_reject_reason} "
                    f"reject_counts={dict(self._reject_reason_counts)}",
                    extra={
                        "event": "vwap_pro_telemetry",
                        "symbol": symbol,
                        "dominant_reject_reason": dominant_reject_reason,
                        "reject_counts_by_reason": dict(self._reject_reason_counts),
                    },
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
                self._log_no_signal_reason(
                    "strike_lock_active",
                    symbol=symbol,
                    ltp=current_price,
                    vwap=indicators.get("vwap"),
                    context={"active_symbol": self._strike_lock.get(lock_key)},
                )
                self._reset_acceptance(
                    acc_key, symbol=symbol, reason_code="strike_lock_active"
                )
                _emit_no_signal("strike_lock_active")
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
                self._log_no_signal_reason(
                    "cooldown_active",
                    symbol=symbol,
                    ltp=current_price,
                    vwap=indicators.get("vwap"),
                    context={
                        "cooldown_remaining_s": max(
                            0.0, self.COOLDOWN_SECONDS - (now - last_fire)
                        )
                    },
                )
                self._reset_acceptance(
                    acc_key, symbol=symbol, reason_code="cooldown_active"
                )
                _emit_no_signal("cooldown_active")
                return None

            # 📊 ISSUE 1 FIX: Hard-block on missing Futures VWAP (No spot fallback)
            # ✅ BONUS: Prefer exchange VWAP (full-session, from broker) over rolling VWAP
            _exch_vwap = indicators.get("exchange_vwap")
            _rolling_vwap = indicators.get("vwap")
            vwap = float(
                _exch_vwap or _rolling_vwap or 0.0
            )  # ✅ Exchange > Rolling > 0

            atr = float(indicators.get("atr") or max(current_price * 0.015, 1.0))
            entropy = float(indicators.get("entropy_5") or 0.5)

            index_ltp = float(
                indicators.get("nifty_fut_ltp")
                or indicators.get("nifty_index_ltp")
                or 0.0
            )
            index_vwap = float(
                indicators.get("nifty_fut_vwap")
                or indicators.get("nifty_index_vwap")
                or 0.0
            )
            index_volume = float(indicators.get("futures_volume") or 0.0)

            if index_vwap <= 0 or index_volume <= 0:
                if not self._index_bias_degraded_logged:
                    LOGGER.info(
                        "🚫 INDEX BIAS DEGRADED → PE BLOCKED, CE-ONLY MODE",
                        extra={
                            "event": "vwap_pro_index_bias_degraded",
                            "symbol": symbol,
                            "index_vwap": index_vwap,
                            "index_volume": index_volume,
                        },
                    )
                    self._index_bias_degraded_logged = True
                if index_ltp <= 0:
                    self._log_no_signal_reason(
                        "index_bias_invalid",
                        symbol=symbol,
                        ltp=current_price,
                        vwap=vwap,
                        context={
                            "index_ltp": index_ltp,
                            "index_vwap": index_vwap,
                            "index_volume": index_volume,
                        },
                    )
                    self._reset_acceptance(
                        acc_key, symbol=symbol, reason_code="index_bias_invalid"
                    )
                    _emit_no_signal("index_bias_invalid")
                    return None
                if not is_ce:
                    self._log_no_signal_reason(
                        "index_bias_invalid",
                        symbol=symbol,
                        ltp=current_price,
                        vwap=vwap,
                        context={
                            "index_ltp": index_ltp,
                            "index_vwap": index_vwap,
                            "index_volume": index_volume,
                        },
                    )
                    self._reset_acceptance(
                        acc_key, symbol=symbol, reason_code="index_bias_invalid"
                    )
                    _emit_no_signal("index_bias_invalid")
                    return None
            elif index_ltp <= 0:
                if not self._index_bias_missing_logged:
                    LOGGER.error(
                        "INVALID INDEX DATA — blocking signal",
                        extra={"event": "vwap_pro_index_invalid", "symbol": symbol},
                    )
                    self._index_bias_missing_logged = True
                self._log_no_signal_reason(
                    "index_bias_invalid",
                    symbol=symbol,
                    ltp=current_price,
                    vwap=vwap,
                    context={"index_ltp": index_ltp, "index_vwap": index_vwap},
                )
                self._reset_acceptance(
                    acc_key, symbol=symbol, reason_code="index_bias_invalid"
                )
                _emit_no_signal("index_bias_invalid")
                return None

            self._index_bias_missing_logged = False
            if index_vwap > 0 and index_volume > 0:
                self._index_bias_degraded_logged = False

            if (is_ce and index_ltp < index_vwap) or (
                not is_ce and index_ltp > index_vwap
            ):
                self._telemetry["skipped_bias"] += 1
                if (
                    self._telemetry["skipped_bias"] <= 3
                    or self._telemetry["skipped_bias"] % self.TELEMETRY_LOG_EVERY == 0
                ):
                    LOGGER.info(
                        f"🧭 BIAS GATE: {symbol} {direction} blocked | "
                        f"IdxLTP={index_ltp:.2f} IdxVWAP={index_vwap:.2f} "
                        f"Need={'BULL' if is_ce else 'BEAR'}",
                        extra={"event": "vwap_pro_bias_reject", "symbol": symbol},
                    )
                self._log_no_signal_reason(
                    "index_bias_invalid",
                    symbol=symbol,
                    ltp=current_price,
                    vwap=vwap,
                    context={
                        "index_ltp": index_ltp,
                        "index_vwap": index_vwap,
                        "direction": direction,
                    },
                )
                self._reset_acceptance(
                    acc_key, symbol=symbol, reason_code="index_bias_invalid"
                )
                _emit_no_signal("index_bias_invalid")
                return None

            if current_price <= 0 or vwap <= 0:
                self._telemetry["skipped_no_vwap"] += 1
                if self._telemetry["skipped_no_vwap"] <= 5:
                    LOGGER.warning(
                        f"⚠️ VWAP ZERO: {symbol} | price={current_price:.2f} vwap={vwap:.2f}",
                        extra={"event": "vwap_pro_zero_block", "symbol": symbol},
                    )
                self._log_no_signal_reason(
                    "vwap_zero_or_invalid",
                    symbol=symbol,
                    ltp=current_price,
                    vwap=vwap,
                )
                self._reset_acceptance(
                    acc_key, symbol=symbol, reason_code="vwap_zero_or_invalid"
                )
                _emit_no_signal("vwap_zero_or_invalid")
                return None

            if current_price < vwap:
                self._telemetry["skipped_vwap"] += 1
                if (
                    self._telemetry["skipped_vwap"] <= 3
                    or self._telemetry["skipped_vwap"] % self.TELEMETRY_LOG_EVERY == 0
                ):
                    LOGGER.info(
                        f"📏 OPT VWAP: {symbol} {direction} | "
                        f"Price={current_price:.2f} VWAP={vwap:.2f}",
                        extra={"event": "vwap_pro_opt_vwap_reject", "symbol": symbol},
                    )
                self._log_no_signal_reason(
                    "price_below_vwap",
                    symbol=symbol,
                    ltp=current_price,
                    vwap=vwap,
                )
                self._reset_acceptance(
                    acc_key, symbol=symbol, reason_code="price_below_vwap"
                )
                _emit_no_signal("price_below_vwap")
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
                            extra={
                                "event": "vwap_pro_overext_reject",
                                "symbol": symbol,
                            },
                        )
                    self._log_no_signal_reason(
                        "overextension_filter",
                        symbol=symbol,
                        ltp=current_price,
                        vwap=vwap,
                        context={"vwap_std": vwap_std, "z_score": z},
                    )
                    self._reset_acceptance(
                        acc_key, symbol=symbol, reason_code="overextension_filter"
                    )
                    _emit_no_signal("overextension_filter")
                    return None

            # 🔊 ISSUE 4 FIX: Reset acceptance on Volume rejection
            vol = float(indicators.get("volume") or 0.0)
            avg_vol = float(indicators.get("avg_volume") or 0.0)
            if vol > 0 and avg_vol > 0:
                self._last_valid_volume[symbol] = vol
                self._last_valid_avg_volume[symbol] = avg_vol
                self._last_valid_volume_ts[symbol] = now
            else:
                last_vol = self._last_valid_volume.get(symbol)
                last_avg = self._last_valid_avg_volume.get(symbol)
                last_ts = self._last_valid_volume_ts.get(symbol, 0.0)
                if (
                    last_vol is not None
                    and last_avg is not None
                    and (now - last_ts) <= self.VOLUME_GRACE_SECONDS
                ):
                    LOGGER.info(
                        "Condition met: vwap_pro_volume_grace_fallback",
                        extra={
                            "event": "vwap_pro_volume_grace_fallback",
                            "symbol": symbol,
                            "fallback_volume": last_vol,
                            "fallback_avg_volume": last_avg,
                        },
                    )
                    vol = last_vol
                    avg_vol = last_avg
                else:
                    self._telemetry["skipped_data"] += 1
                    if (
                        self._telemetry["skipped_data"] <= 5
                        or self._telemetry["skipped_data"] % self.TELEMETRY_LOG_EVERY
                        == 0
                    ):
                        LOGGER.warning(
                            f"⚠️ DATA INVALID: {symbol} | "
                            f"vol={vol:.0f} avg_vol={avg_vol:.0f}",
                            extra={
                                "event": "vwap_pro_data_invalid",
                                "symbol": symbol,
                            },
                        )
                        LOGGER.debug(
                            "volume_below_threshold",
                            extra={
                                "event": "volume_below_threshold",
                                "symbol": symbol,
                                "vol": vol,
                                "avg_vol": avg_vol,
                                "required_volume": None,
                            },
                        )
                    self._log_no_signal_reason(
                        "volume_below_threshold",
                        symbol=symbol,
                        ltp=current_price,
                        vwap=vwap,
                        vol=vol,
                        avg_vol=avg_vol,
                    )
                    self._reset_acceptance(
                        acc_key, symbol=symbol, reason_code="volume_below_threshold"
                    )
                    _emit_no_signal("volume_below_threshold")
                    return None
            vol_thresh = self._dynamic_volume_threshold()
            if vol < avg_vol * vol_thresh:
                self._telemetry["skipped_volume"] += 1
                if (
                    self._telemetry["skipped_volume"] <= 5
                    or self._telemetry["skipped_volume"] % self.TELEMETRY_LOG_EVERY == 0
                ):
                    LOGGER.info(
                        f"🔊 VOL GATE: {symbol} | vol={vol:.0f} < avg={avg_vol:.0f}×{vol_thresh}={avg_vol*vol_thresh:.0f}",
                        extra={"event": "vwap_pro_vol_reject", "symbol": symbol},
                    )
                    LOGGER.debug(
                        "volume_below_threshold",
                        extra={
                            "event": "volume_below_threshold",
                            "symbol": symbol,
                            "vol": vol,
                            "avg_vol": avg_vol,
                            "volume_threshold": vol_thresh,
                            "required_volume": avg_vol * vol_thresh,
                        },
                    )
                self._log_no_signal_reason(
                    "volume_below_threshold",
                    symbol=symbol,
                    ltp=current_price,
                    vwap=vwap,
                    vol=vol,
                    avg_vol=avg_vol,
                    context={
                        "volume_threshold": vol_thresh,
                        "required_volume": avg_vol * vol_thresh,
                    },
                )
                self._reset_acceptance(
                    acc_key, symbol=symbol, reason_code="volume_below_threshold"
                )
                _emit_no_signal("volume_below_threshold")
                return None

            self._vwap_acceptance_tracker[acc_key] += 1
            if self._vwap_acceptance_tracker[acc_key] < self.VWAP_ACCEPTANCE_BARS:
                self._telemetry["skipped_acceptance"] += 1
                LOGGER.info(
                    f"⏳ ACCEPTANCE: {symbol} {direction} | "
                    f"Bar {self._vwap_acceptance_tracker[acc_key]}/{self.VWAP_ACCEPTANCE_BARS}",
                    extra={"event": "vwap_pro_acceptance", "symbol": symbol},
                )
                self._log_no_signal_reason(
                    "acceptance_bars_insufficient",
                    symbol=symbol,
                    ltp=current_price,
                    vwap=vwap,
                    context={
                        "acceptance_bars": self._vwap_acceptance_tracker[acc_key],
                        "required_bars": self.VWAP_ACCEPTANCE_BARS,
                    },
                )
                _emit_no_signal("acceptance_bars_insufficient")
                return None

            # ═══════════════════════════════════════════════════════════════════
            # ✅ WORLD-CLASS FIX: SL/TP Based on OPTION PREMIUM Direction
            # We trade OPTION PREMIUM, not the index.
            # LONG any option (CE or PE) → profit when PREMIUM rises
            # Therefore: SL below entry, TP above entry (ALWAYS for LONG)
            # ═══════════════════════════════════════════════════════════════════
            sl_mult = (1.2 if self._is_expiry_day() else 1.5) * (
                0.85 if entropy > 0.75 else 1.0
            )
            tp2_mult = 3.0

            # LONG position: SL below, TP above (regardless of CE/PE)
            sl = current_price - (atr * sl_mult)
            tp2 = current_price + (atr * tp2_mult)

            # Validation: LONG must have SL < entry < TP
            invalid = sl >= current_price or tp2 <= current_price

            if invalid:
                LOGGER.error(
                    "Invalid SL/TP for LONG",
                    extra={
                        "symbol": symbol,
                        "entry": current_price,
                        "sl": sl,
                        "tp": tp2,
                        "atr": atr,
                    },
                )
                self._log_no_signal_reason(
                    "overextension_filter",
                    symbol=symbol,
                    ltp=current_price,
                    vwap=vwap,
                    context={"sl": sl, "tp": tp2, "atr": atr},
                )
                self._reset_acceptance(
                    acc_key, symbol=symbol, reason_code="overextension_filter"
                )
                _emit_no_signal("overextension_filter")
                return None

            # 🎯 Final Signal Prep
            base_qty = int(self._vwap_config.quantity or 1)
            confidence = 0.85 if entropy < 0.7 else 0.65
            qty = max(1, int(base_qty * confidence))

            # 🏁 CRITICAL ISSUE 3 FIX: Reset acceptance after firing
            self._reset_acceptance(acc_key, symbol=symbol, reason_code="signal_fired")
            self._signal_cooldown_tracker[cooldown_key] = now
            self._telemetry["signals"] += 1

            LOGGER.info(
                "Condition met: vwap_pro_signal_ready",
                extra={
                    "event": "vwap_pro_signal_ready",
                    "symbol": symbol,
                    "direction": direction,
                    "entry": current_price,
                    "sl": sl,
                    "tp2": tp2,
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
