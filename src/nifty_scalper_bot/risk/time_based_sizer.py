"""Time-of-day position sizing helpers."""

from __future__ import annotations

from datetime import datetime

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class TimeBasedSizer:
    """Adjust position size based on intraday session windows."""

    def get_time_multiplier(self) -> float:
        """Return size multiplier derived from the current exchange clock.

        Args:
            None.

        Returns:
            float: Position size multiplier between 0.0 and 1.2.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered TimeBasedSizer.get_time_multiplier",
            extra={"event": "time_sizer_enter"},
        )
        try:
            now = datetime.now()
            minutes = now.hour * 60 + now.minute

            market_open = 9 * 60 + 15
            morning_end = 9 * 60 + 45
            afternoon_start = 14 * 60 + 30
            late_start = 15 * 60
            market_close = 15 * 60 + 30

            if market_open <= minutes < morning_end:
                multiplier = 0.5
            elif morning_end <= minutes < afternoon_start:
                multiplier = 1.0
            elif afternoon_start <= minutes < late_start:
                multiplier = 0.7
            elif late_start <= minutes < market_close:
                multiplier = 0.0 if self._is_expiry_day() else 0.5
            else:
                multiplier = 0.0

            LOGGER.info(
                "Condition met: time_multiplier_resolved",
                extra={
                    "event": "time_multiplier_resolved",
                    "minutes": minutes,
                    "multiplier": multiplier,
                },
            )
            return multiplier
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in TimeBasedSizer.get_time_multiplier: %s",
                exc,
                extra={"event": "time_sizer_error"},
                exc_info=exc,
            )
            return 0.0

    def _is_expiry_day(self) -> bool:
        """Return expiry flag for the current trading session.

        Args:
            None.

        Returns:
            bool: ``True`` when the current day is a Thursday expiry.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered TimeBasedSizer._is_expiry_day",
            extra={"event": "time_sizer_expiry_check"},
        )
        try:
            result = datetime.now().weekday() == 3
            LOGGER.debug(
                "Condition met: expiry_day_flag",
                extra={"event": "expiry_day_flag", "value": result},
            )
            return result
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in TimeBasedSizer._is_expiry_day: %s",
                exc,
                extra={"event": "time_sizer_expiry_error"},
                exc_info=exc,
            )
            return False
