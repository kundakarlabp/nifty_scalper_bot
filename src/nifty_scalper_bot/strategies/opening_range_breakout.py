"""Opening range breakout strategy with NR7 confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal, Protocol, Sequence

from nifty_scalper_bot.strategies.signal_generator import (
    IndicatorMap,
    Position,
    Signal,
    Strategy,
)
from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class OpeningRangeSnapshot:
    """Opening range high/low for a completed trading session."""

    session: date
    high: float
    low: float


class OpeningRangeHistoryProvider(Protocol):
    """Protocol describing opening range history providers."""

    def get_opening_range_history(
        self,
        symbol: str,
        *,
        interval_minutes: int,
        limit: int,
    ) -> Sequence[OpeningRangeSnapshot]:
        """Return historical opening ranges for *symbol*."""


class OpeningRangeBreakoutStrategy(Strategy):
    """Opening range breakout with NR7 filter to reduce chop."""

    def __init__(
        self,
        *,
        history_provider: OpeningRangeHistoryProvider | None = None,
        description: str | None = None,
    ) -> None:
        """Initialise the strategy with optional history provider."""

        parameters: dict[str, object] = {
            "description": description
            or "Opening range breakout with NR7 confirmation",
        }
        super().__init__("Opening Range Breakout", parameters)
        self._history_provider = history_provider

    def get_required_indicators(self) -> list[str]:
        """Return indicator keys required for ORB evaluation."""

        return [
            "opening_range_high",
            "opening_range_low",
            "orb_break_direction",
            "orb_entry_price",
        ]

    def generate_signal(
        self,
        symbol: str,
        indicators: IndicatorMap,
        current_price: float,
        position: Position | None,
    ) -> Signal | None:
        """Generate ORB signal when NR7 filter passes.

        Args:
            symbol: Trading symbol under evaluation.
            indicators: Indicator snapshot including OR metrics.
            current_price: Latest traded price (unused for now).
            position: Existing position metadata, if any.

        Returns:
            Optional :class:`Signal` when breakout conditions are met.

        Raises:
            None.
        """

        logger.debug(
            'Entered OpeningRangeBreakoutStrategy.generate_signal',
            extra={'event': 'opening_range_generate_enter', 'symbol': symbol},
        )
        try:
            orb_high_raw = indicators.get('opening_range_high')
            orb_low_raw = indicators.get('opening_range_low')
            if not isinstance(orb_high_raw, (int, float)) or not isinstance(
                orb_low_raw, (int, float)
            ):
                logger.info(
                    'Condition met: opening_range_missing',
                    extra={'event': 'opening_range_missing', 'symbol': symbol},
                )
                return None
            orb_high = float(orb_high_raw)
            orb_low = float(orb_low_raw)
            if orb_high <= orb_low:
                logger.info(
                    'Condition met: opening_range_invalid',
                    extra={
                        'event': 'opening_range_invalid',
                        'symbol': symbol,
                        'orb_high': orb_high,
                        'orb_low': orb_low,
                    },
                )
                return None
            current_range = orb_high - orb_low
            range_pct = current_range / max(orb_low, 1.0)

            if not self._passes_nr7(symbol, current_range):
                logger.info(
                    'Condition met: nr7_filter_failed',
                    extra={
                        'event': 'nr7_filter_failed',
                        'symbol': symbol,
                        'current_range': current_range,
                        'range_pct': range_pct,
                    },
                )
                return None

            logger.info(
                'Condition met: nr7_filter_passed',
                extra={
                    'event': 'nr7_filter_passed',
                    'symbol': symbol,
                    'current_range': current_range,
                    'range_pct': range_pct,
                },
            )

            if position:
                # Skip entries if position already open to avoid duplicates.
                return None

            direction = indicators.get('orb_break_direction')
            entry_raw = indicators.get('orb_entry_price')
            if not isinstance(entry_raw, (int, float)):
                entry_price = current_price
            else:
                entry_price = float(entry_raw)
            breakout_buffer = max(current_range * 0.05, 0.5)
            side: Literal["BUY", "SELL"]
            stop_loss_price: float | None
            take_profit_price: float | None
            if direction == "UP":
                if entry_price <= orb_high + breakout_buffer:
                    logger.info(
                        'Condition met: breakout_buffer_unmet',
                        extra={
                            'event': 'opening_range_buffer_failed',
                            'symbol': symbol,
                            'direction': 'UP',
                            'entry_price': entry_price,
                            'orb_high': orb_high,
                            'buffer': breakout_buffer,
                        },
                    )
                    return None
                side = "BUY"
                stop_loss_price, take_profit_price = self._ensure_rr(
                    "BUY", entry_price, orb_low, desired_rr=2.0
                )
            elif direction == "DOWN":
                if entry_price >= orb_low - breakout_buffer:
                    logger.info(
                        'Condition met: breakout_buffer_unmet',
                        extra={
                            'event': 'opening_range_buffer_failed',
                            'symbol': symbol,
                            'direction': 'DOWN',
                            'entry_price': entry_price,
                            'orb_low': orb_low,
                            'buffer': breakout_buffer,
                        },
                    )
                    return None
                side = "SELL"
                stop_loss_price, take_profit_price = self._ensure_rr(
                    "SELL", entry_price, orb_high, desired_rr=2.0
                )
            else:
                return None

            if stop_loss_price is None or take_profit_price is None:
                return None

            reason = "Opening range breakout with NR7 confirmation"
            metadata = {
                'strategy': self.name,
                'orb_high': orb_high,
                'orb_low': orb_low,
                'nr7': True,
                'breakout_buffer': breakout_buffer,
                'range_pct': range_pct,
            }
            return Signal(
                action=side,
                symbol=symbol,
                quantity=1,
                confidence=self._bounded_confidence(0.6),
                reason=reason,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                'Failure in OpeningRangeBreakoutStrategy.generate_signal: %s',
                exc,
                extra={'event': 'opening_range_generate_error', 'symbol': symbol},
                exc_info=exc,
            )
            return None

    def _passes_nr7(self, symbol: str, current_range: float) -> bool:
        """Return ``True`` when current range is narrowest of last seven.

        Args:
            symbol: Trading symbol under evaluation.
            current_range: Range for the present session.

        Returns:
            ``True`` when the NR7 filter passes.

        Raises:
            None.
        """

        logger.debug(
            "Entered OpeningRangeBreakoutStrategy._passes_nr7",
            extra={"event": "nr7_check_enter", "symbol": symbol},
        )
        if current_range <= 0:
            return False

        if self._history_provider is None:
            return True

        history_ranges: list[float] = []
        try:
            snapshots = self._history_provider.get_opening_range_history(
                symbol,
                interval_minutes=30,
                limit=7,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failure in OpeningRangeBreakoutStrategy._passes_nr7: %s",
                exc,
                extra={"event": "nr7_history_error", "symbol": symbol},
            )
            return False

        for snap in self._iter_valid_snapshots(snapshots):
            history_ranges.append(max(snap.high - snap.low, 0.0))

        if len(history_ranges) < 6:
            return True

        narrowest = min(history_ranges)
        return current_range <= narrowest

    @staticmethod
    def _iter_valid_snapshots(
        snapshots: Iterable[OpeningRangeSnapshot],
    ) -> Iterable[OpeningRangeSnapshot]:
        """Yield snapshots with valid range data."""

        for snap in snapshots:
            if not isinstance(snap, OpeningRangeSnapshot):
                continue
            if snap.high is None or snap.low is None:
                continue
            if not isinstance(snap.high, (int, float)):
                continue
            if not isinstance(snap.low, (int, float)):
                continue
            yield snap


__all__ = [
    "OpeningRangeSnapshot",
    "OpeningRangeHistoryProvider",
    "OpeningRangeBreakoutStrategy",
]
