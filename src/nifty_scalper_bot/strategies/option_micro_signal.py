"""Microstructure-driven entry and exit signals for NIFTY options."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Literal, Optional

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
    probability: float = 0.0
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
    spread_threshold_pct: float | None = None
    momentum_z_threshold: float | None = None
    microvol_percentile: float = 60.0
    signal_weight_momentum: float = 1.0
    signal_weight_vol: float = 0.8
    signal_weight_depth: float = 0.4
    signal_weight_spread: float = 1.1
    microvol_buffer_len: int = 1500
    _last_mid: float | None = field(init=False, default=None)
    _microvol: float = field(init=False, default=0.0)
    _last_ts_ns: int = field(init=False, default=0)
    _microvol_buffer: Deque[float] = field(init=False)

    def __post_init__(self) -> None:
        """Args: None. Returns: None. Raises: None."""

        self._microvol_buffer = deque(maxlen=max(int(self.microvol_buffer_len), 1))

    def on_tick(
        self,
        *,
        bid: float,
        ask: float,
        last_price: float,
        last_size: int,
        depth: int,
        ts_ns: int,
        symbol: str | None = None,
        portfolio: object | None = None,
        position_side: PositionSide = "FLAT",
        entry_mid: float | None = None,
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
                return SignalDecision(False, False, None, None, 0.0, features)
            if ask <= 0.0 or bid <= 0.0 or ask < bid:
                LOGGER.info(
                    'Condition met: option_micro_signal_invalid_quote',
                    extra={
                        'event': 'option_micro_signal_invalid_quote',
                        'bid': bid,
                        'ask': ask,
                    },
                )
                return SignalDecision(False, False, None, None, 0.0, features)
            if last_size < 0 or depth < 0:
                LOGGER.info(
                    'Condition met: option_micro_signal_invalid_depth',
                    extra={
                        'event': 'option_micro_signal_invalid_depth',
                        'last_size': last_size,
                        'depth': depth,
                    },
                )
                return SignalDecision(False, False, None, None, 0.0, features)

            spread = max(ask - bid, self.tick_size)
            mid = (ask + bid) / 2.0
            spread_pct = (spread / max(mid, self.tick_size)) * 100.0
            if self._last_mid is None:
                self._last_mid = mid
                self._last_ts_ns = ts_ns
                return SignalDecision(False, False, None, None, 0.0, features)

            delta_mid = mid - self._last_mid
            delta_abs = abs(delta_mid)
            features['spread'] = spread
            features['spread_pct'] = spread_pct
            features['mid'] = mid
            features['delta_mid'] = delta_mid
            features['depth'] = float(depth)

            elapsed_ns = max(ts_ns - self._last_ts_ns, 1)
            elapsed_ms = elapsed_ns / 1_000_000
            decay = math.exp(-elapsed_ms / max(self.ema_half_life_ms, 1e-6))
            alpha = 1.0 - decay
            self._microvol = (1.0 - alpha) * self._microvol + alpha * delta_abs
            self._microvol_buffer.append(self._microvol)
            features['microvol'] = self._microvol

            snmom = delta_mid / max(spread, self.tick_size)
            momentum_z = delta_mid / max(self._microvol, self.tick_size)
            features['snmom'] = snmom
            features['momentum_z'] = momentum_z

            ltt_flag = math.isclose(
                last_price, bid, abs_tol=self.tick_size / 2
            ) or math.isclose(last_price, ask, abs_tol=self.tick_size / 2)
            features['ltt'] = 1.0 if ltt_flag else 0.0

            depth_score = depth / max(self.min_depth, 1)
            microvol_z = delta_abs / max(self._microvol, self.tick_size)
            score = (
                self.signal_weight_momentum * momentum_z
                + self.signal_weight_vol * microvol_z
                + self.signal_weight_depth * depth_score
                - self.signal_weight_spread * spread_pct
            )
            probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(score, 60.0))))
            features['microvol_z'] = microvol_z
            features['depth_score'] = depth_score
            features['probability'] = probability

            decision = SignalDecision(False, False, None, None, probability, features)
            spread_limit_pct = self.spread_threshold_pct
            if spread_limit_pct is None:
                spread_limit_pct = (self.spread_limit / max(mid, self.tick_size)) * 100.0
            if spread_pct > spread_limit_pct or depth < self.min_depth:
                LOGGER.info(
                    'Condition met: option_micro_signal_liquidity_block',
                    extra={
                        'event': 'option_micro_signal_liquidity_block',
                        'spread': spread,
                        'spread_pct': spread_pct,
                        'depth': depth,
                    },
                )
                self._update_state(mid, ts_ns)
                return decision

            effective_side: PositionSide = position_side
            if effective_side == 'FLAT' and portfolio is not None and symbol:
                has_open = getattr(portfolio, 'has_open_position', None)
                if callable(has_open) and bool(has_open(symbol)):
                    LOGGER.info(
                        'Condition met: option_micro_signal_open_position_block',
                        extra={'event': 'option_micro_signal_open_position_block', 'symbol': symbol},
                    )
                    self._update_state(mid, ts_ns)
                    return decision
            momentum_gate = self.momentum_z_threshold
            if momentum_gate is None:
                momentum_gate = self.snmom_threshold
            if effective_side == 'FLAT':
                if (
                    momentum_z > momentum_gate
                    and self._passes_microvol_filter()
                    and ltt_flag
                    and math.isclose(last_price, ask, abs_tol=self.tick_size / 2)
                ):
                    LOGGER.info(
                        'Condition met: option_micro_signal_long_entry',
                        extra={'event': 'option_micro_signal_long_entry'},
                    )
                    decision = SignalDecision(True, False, 'LONG', 'LONG_ENTRY', probability, features)
                elif (
                    momentum_z < -momentum_gate
                    and self._passes_microvol_filter()
                    and ltt_flag
                    and math.isclose(last_price, bid, abs_tol=self.tick_size / 2)
                ):
                    LOGGER.info(
                        'Condition met: option_micro_signal_short_entry',
                        extra={'event': 'option_micro_signal_short_entry'},
                    )
                    decision = SignalDecision(True, False, 'SHORT', 'SHORT_ENTRY', probability, features)
            else:
                exit_reason = self._should_exit(mid, momentum_z, effective_side, entry_mid)
                if exit_reason:
                    LOGGER.info(
                        'Condition met: option_micro_signal_exit',
                        extra={
                            'event': 'option_micro_signal_exit',
                            'reason': exit_reason,
                            'side': effective_side,
                        },
                    )
                    decision = SignalDecision(False, True, effective_side, exit_reason, probability, features)

            self._update_state(mid, ts_ns)
            return decision
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                'Failure in OptionMicroSignal.on_tick: %s',
                exc,
                extra={'event': 'option_micro_signal_error'},
                exc_info=exc,
            )
            return SignalDecision(False, False, None, None, 0.0, features)

    def _passes_microvol_filter(self) -> bool:
        """Args: None. Returns: bool. Raises: None."""

        if len(self._microvol_buffer) < 200:
            return True
        threshold = self._percentile(list(self._microvol_buffer), self.microvol_percentile)
        return self._microvol >= threshold

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        """Args: values, percentile. Returns: float. Raises: None."""

        if not values:
            return 0.0
        ranked = sorted(values)
        pct = max(0.0, min(float(percentile), 100.0)) / 100.0
        index = int(round((len(ranked) - 1) * pct))
        return ranked[index]

    def _should_exit(
        self,
        mid: float,
        momentum_z: float,
        position_side: PositionSide,
        entry_mid: float | None,
    ) -> str | None:
        """Args: mid, momentum_z, position_side, entry_mid. Returns: optional reason. Raises: None."""

        momentum_gate = self.momentum_z_threshold
        if momentum_gate is None:
            momentum_gate = self.snmom_threshold
        if position_side == "LONG":
            if (
                entry_mid is not None
                and (entry_mid - mid) >= self.tick_size * self.adverse_ticks
            ):
                return "HARD_STOP"
            if momentum_z < -momentum_gate:
                return "MOMENTUM_REVERSAL"
        elif position_side == "SHORT":
            if (
                entry_mid is not None
                and (mid - entry_mid) >= self.tick_size * self.adverse_ticks
            ):
                return "HARD_STOP"
            if momentum_z > momentum_gate:
                return "MOMENTUM_REVERSAL"
        return None

    def _update_state(self, mid: float, ts_ns: int) -> None:
        self._last_mid = mid
        self._last_ts_ns = ts_ns


__all__ = ["OptionMicroSignal", "SignalDecision"]
