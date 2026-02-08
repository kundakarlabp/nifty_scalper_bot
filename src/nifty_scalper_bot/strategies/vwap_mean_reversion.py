"""VWAP-driven mean reversion strategy utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


def _coerce_float(value: Any) -> float | None:
    """Return float representation when *value* is numeric."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


@dataclass(slots=True, frozen=True)
class VWAPSignal:
    """Structured signal emitted by the VWAP mean reversion model."""

    action: Literal["BUY", "SELL", "HOLD"]
    reason: str
    metadata: dict[str, Any]


class VWAPMeanReversionStrategy:
    """Generate directional signals based on spot deviations from futures VWAP."""

    def __init__(
        self,
        *,
        lookback: int = 20,
        threshold_pct: float = 0.0015,
    ) -> None:
        """Initialise strategy thresholds."""

        self._lookback = max(int(lookback), 1)
        self._threshold_pct = max(float(threshold_pct), 0.0)

    def generate_signal(self, market_data: Mapping[str, Any]) -> VWAPSignal:
        """Return VWAP mean reversion signal using futures volume data.

        Args:
            market_data: Snapshot containing spot quote and futures OHLC bars.

        Returns:
            VWAPSignal representing BUY/SELL/HOLD decision with context.

        Raises:
            KeyError: If market_data is missing mandatory keys.
        """

        logger.debug(
            'Entered VWAPMeanReversionStrategy.generate_signal',
            extra={'event': 'vwap_mean_reversion_enter'},
        )
        try:
            spot_snapshot = market_data.get('NIFTY')
            if not isinstance(spot_snapshot, Mapping):
                logger.info(
                    'Condition met: spot_price_missing',
                    extra={'event': 'vwap_mean_reversion_no_spot'},
                )
                return VWAPSignal(
                    action="HOLD",
                    reason="spot_price_missing",
                    metadata={"threshold_pct": self._threshold_pct},
                )
            spot_price_value = _coerce_float(spot_snapshot.get('ltp'))
            if spot_price_value is None:
                logger.info(
                    'Condition met: spot_price_invalid',
                    extra={'event': 'vwap_mean_reversion_invalid_spot'},
                )
                return VWAPSignal(
                    action="HOLD",
                    reason="spot_price_missing",
                    metadata={"threshold_pct": self._threshold_pct},
                )
            spot_price = spot_price_value
            bars_payload = market_data.get('NIFTYFUT_bars') or market_data.get(
                'NIFTY_bars'
            )
            if not isinstance(bars_payload, Sequence):
                logger.info(
                    'Condition met: volume_data_missing',
                    extra={'event': 'vwap_mean_reversion_no_bars'},
                )
                return VWAPSignal(
                    action="HOLD",
                    reason="volume_data_missing",
                    metadata={"threshold_pct": self._threshold_pct},
                )
            bars = list(bars_payload)[-self._lookback :]
            closes: list[float] = []
            volumes: list[float] = []
            for bar in bars:
                if not isinstance(bar, Mapping):
                    continue
                close_value = _coerce_float(bar.get("close"))
                volume_value = _coerce_float(bar.get("volume"))
                if close_value is None or volume_value is None:
                    continue
                closes.append(close_value)
                volumes.append(max(volume_value, 0.0))
            total_volume = sum(volumes)
            if total_volume <= 0.0 or not closes:
                logger.info(
                    'Condition met: volume_data_missing',
                    extra={'event': 'vwap_mean_reversion_zero_volume'},
                )
                return VWAPSignal(
                    action="HOLD",
                    reason="volume_data_missing",
                    metadata={"threshold_pct": self._threshold_pct},
                )
            vwap_numerator = sum(
                close * volume for close, volume in zip(closes, volumes)
            )
            vwap = vwap_numerator / total_volume
            deviation = 0.0
            if vwap > 0:
                deviation = (spot_price - vwap) / vwap
            threshold = self._threshold_pct
            volatility_pct = 0.0
            if len(closes) > 1 and vwap > 0:
                mean_close = sum(closes) / len(closes)
                variance = sum(
                    (close - mean_close) ** 2 for close in closes
                ) / len(closes)
                volatility_pct = (variance**0.5) / vwap
                threshold = max(threshold, volatility_pct * 0.5)
            logger.debug(
                'Computed VWAP volatility-adjusted threshold',
                extra={
                    'event': 'vwap_mean_reversion_threshold',
                    'base_threshold_pct': self._threshold_pct,
                    'dynamic_threshold_pct': threshold,
                    'volatility_pct': volatility_pct,
                },
            )
            if deviation <= -threshold:
                action: Literal["BUY", "SELL", "HOLD"] = "BUY"
                reason = "spot_below_vwap"
            elif deviation >= threshold:
                action = "SELL"
                reason = "spot_above_vwap"
            else:
                action = "HOLD"
                reason = "within_band"
            deviation_sigma = (
                deviation / volatility_pct if volatility_pct > 0 else 0.0
            )
            metadata = {
                'spot_price': spot_price,
                'vwap': vwap,
                'deviation_pct': deviation,
                'total_volume': total_volume,
                'bars_used': len(closes),
                'threshold_pct': threshold,
                'volatility_pct': volatility_pct,
                'deviation_sigma': deviation_sigma,
            }
            logger.info(
                'Condition met: vwap_signal_computed',
                extra={
                    'event': 'vwap_mean_reversion_decision',
                    'action': action,
                    'reason': reason,
                    'deviation_pct': deviation,
                },
            )
            return VWAPSignal(action=action, reason=reason, metadata=metadata)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                'Failure in VWAPMeanReversionStrategy.generate_signal: %s',
                exc,
                extra={'event': 'vwap_mean_reversion_error'},
                exc_info=exc,
            )
            raise


__all__ = ["VWAPMeanReversionStrategy", "VWAPSignal"]
