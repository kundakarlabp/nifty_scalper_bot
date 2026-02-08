"""Microstructure-driven entry and exit signals for NIFTY options."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

PositionSide = Literal["FLAT", "LONG", "SHORT"]


@dataclass(slots=True)
class SignalDecision:
    """Signal output produced by :class:`OptionMicroSignal`.

    Example:
        >>> signal = SignalDecision(enter=True, exit=False, side="LONG", reason="snmom")
        >>> bool(signal.enter)
        True
    """

    enter: bool
    exit: bool
    side: Optional[Literal["LONG", "SHORT"]]
    reason: str | None
    features: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class OptionMicroSignal:
    """Compute microstructure features and trading signals for scalping.

    Args:
        spread_limit: Maximum allowed bid/ask spread in INR.
        min_depth: Minimum aggregated depth required to trade.
        snmom_threshold: Spread-normalized momentum threshold.
        microvol_threshold: Minimum EMA of mid-price volatility required.
        adverse_ticks: Stop loss trigger in ticks against the position.
        tick_size: Minimum tick size for comparisons (defaults to 0.05).
        ema_half_life_ms: Approximate half-life for the EMA in milliseconds.
    """

    spread_limit: float
    min_depth: int
    snmom_threshold: float
    microvol_threshold: float
    adverse_ticks: int
    tick_size: float = 0.05
    ema_half_life_ms: float = 500.0
    _position: PositionSide = field(init=False, default="FLAT")
    _last_mid: float | None = field(init=False, default=None)
    _microvol: float = field(init=False, default=0.0)
    _last_ts_ns: int = field(init=False, default=0)
    _entry_mid: float | None = field(init=False, default=None)

    def on_tick(
        self,
        *,
        bid: float,
        ask: float,
        last_price: float,
        last_size: int,
        depth: int,
        ts_ns: int,
    ) -> SignalDecision:
        """Process a new tick and return the resulting decision.

        Args:
            bid: Latest best bid price.
            ask: Latest best ask price.
            last_price: Price of the last trade.
            last_size: Reported size of the last trade.
            depth: Aggregate visible depth at top-of-book.
            ts_ns: Monotonic nanosecond timestamp of the tick.

        Returns:
            SignalDecision capturing entry/exit intent and feature values.

        Raises:
            None.
        """
        LOGGER.debug(
            'Entered OptionMicroSignal.on_tick',
            extra={'event': 'option_micro_signal_tick'},
        )
        features: Dict[str, float] = {'last_size': float(last_size)}
        try:
            if (
                not math.isfinite(bid)
                or not math.isfinite(ask)
                or not math.isfinite(last_price)
            ):
                LOGGER.info(
                    'Condition met: option_micro_signal_invalid_price',
                    extra={'event': 'option_micro_signal_invalid_price'},
                )
                return SignalDecision(False, False, None, None, features)
            if ask <= 0.0 or bid <= 0.0 or ask < bid:
                LOGGER.info(
                    'Condition met: option_micro_signal_invalid_quote',
                    extra={
                        'event': 'option_micro_signal_invalid_quote',
                        'bid': bid,
                        'ask': ask,
                    },
                )
                return SignalDecision(False, False, None, None, features)
            if last_size < 0 or depth < 0:
                LOGGER.info(
                    'Condition met: option_micro_signal_invalid_depth',
                    extra={
                        'event': 'option_micro_signal_invalid_depth',
                        'last_size': last_size,
                        'depth': depth,
                    },
                )
                return SignalDecision(False, False, None, None, features)

            spread = max(ask - bid, self.tick_size)
            mid = (ask + bid) / 2.0
            if self._last_mid is None:
                self._last_mid = mid
                self._last_ts_ns = ts_ns
                return SignalDecision(False, False, None, None, features)

            delta_mid = mid - self._last_mid
            delta_abs = abs(delta_mid)
            features['spread'] = spread
            features['mid'] = mid
            features['delta_mid'] = delta_mid
            features['depth'] = float(depth)

            elapsed_ns = max(ts_ns - self._last_ts_ns, 1)
            elapsed_ms = elapsed_ns / 1_000_000
            decay = math.exp(-elapsed_ms / max(self.ema_half_life_ms, 1e-6))
            alpha = 1.0 - decay
            self._microvol = (1.0 - alpha) * self._microvol + alpha * delta_abs
            features['microvol'] = self._microvol

            snmom = delta_mid / max(spread, self.tick_size)
            features['snmom'] = snmom

            ltt_flag = math.isclose(
                last_price, bid, abs_tol=self.tick_size / 2
            ) or math.isclose(last_price, ask, abs_tol=self.tick_size / 2)
            features['ltt'] = 1.0 if ltt_flag else 0.0

            decision = SignalDecision(False, False, None, None, features)
            if spread > self.spread_limit or depth < self.min_depth:
                LOGGER.info(
                    'Condition met: option_micro_signal_liquidity_block',
                    extra={
                        'event': 'option_micro_signal_liquidity_block',
                        'spread': spread,
                        'depth': depth,
                    },
                )
                self._update_state(mid, ts_ns)
                return decision

            if self._position == 'FLAT':
                if (
                    snmom > self.snmom_threshold
                    and self._microvol > self.microvol_threshold
                    and ltt_flag
                    and math.isclose(last_price, ask, abs_tol=self.tick_size / 2)
                ):
                    self._position = 'LONG'
                    self._entry_mid = mid
                    LOGGER.info(
                        'Condition met: option_micro_signal_long_entry',
                        extra={'event': 'option_micro_signal_long_entry'},
                    )
                    decision = SignalDecision(
                        True, False, 'LONG', 'LONG_ENTRY', features
                    )
                elif (
                    snmom < -self.snmom_threshold
                    and self._microvol > self.microvol_threshold
                    and ltt_flag
                    and math.isclose(last_price, bid, abs_tol=self.tick_size / 2)
                ):
                    self._position = 'SHORT'
                    self._entry_mid = mid
                    LOGGER.info(
                        'Condition met: option_micro_signal_short_entry',
                        extra={'event': 'option_micro_signal_short_entry'},
                    )
                    decision = SignalDecision(
                        True, False, 'SHORT', 'SHORT_ENTRY', features
                    )
            else:
                exit_reason = self._should_exit(mid, snmom)
                if exit_reason:
                    LOGGER.info(
                        'Condition met: option_micro_signal_exit',
                        extra={
                            'event': 'option_micro_signal_exit',
                            'reason': exit_reason,
                            'side': self._position,
                        },
                    )
                    decision = SignalDecision(
                        False, True, self._position, exit_reason, features
                    )
                    self._position = 'FLAT'
                    self._entry_mid = None

            self._update_state(mid, ts_ns)
            return decision
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                'Failure in OptionMicroSignal.on_tick: %s',
                exc,
                extra={'event': 'option_micro_signal_error'},
                exc_info=exc,
            )
            return SignalDecision(False, False, None, None, features)

    def _should_exit(self, mid: float, snmom: float) -> str | None:
        if self._position == "LONG":
            if (
                self._entry_mid is not None
                and (self._entry_mid - mid) >= self.tick_size * self.adverse_ticks
            ):
                return "HARD_STOP"
            if snmom < -self.snmom_threshold:
                return "MOMENTUM_REVERSAL"
        elif self._position == "SHORT":
            if (
                self._entry_mid is not None
                and (mid - self._entry_mid) >= self.tick_size * self.adverse_ticks
            ):
                return "HARD_STOP"
            if snmom > self.snmom_threshold:
                return "MOMENTUM_REVERSAL"
        return None

    def _update_state(self, mid: float, ts_ns: int) -> None:
        self._last_mid = mid
        self._last_ts_ns = ts_ns


__all__ = ["OptionMicroSignal", "SignalDecision"]
