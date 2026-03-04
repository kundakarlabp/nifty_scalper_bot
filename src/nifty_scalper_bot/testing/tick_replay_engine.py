"""Deterministic historical tick replay engine."""

from __future__ import annotations

import csv
import time
from typing import Any, Callable

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class TickReplayEngine:
    """Replay historical ticks through callback handlers."""

    def __init__(self, tick_file: str, speed: float = 1.0) -> None:
        """Args: tick_file, speed; Returns: None; Raises: None."""
        self.tick_file = tick_file
        self.speed = speed
        self.ticks: list[dict[str, Any]] = []
        self.index = 0

    def load(self) -> None:
        """Args: none; Returns: None; Raises: OSError, ValueError."""
        LOGGER.debug('Entered TickReplayEngine.load')
        try:
            with open(self.tick_file, encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    self.ticks.append(
                        {
                            'symbol': str(row['symbol']),
                            'ltp': float(row['ltp']),
                            'timestamp': float(row['timestamp']),
                        }
                    )
            LOGGER.info('Condition met: tick_replay_loaded count=%s', len(self.ticks))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error('Failure in TickReplayEngine.load: %s', exc)
            raise

    def replay(self, tick_callback: Callable[[dict[str, Any]], None]) -> None:
        """Args: tick_callback; Returns: None; Raises: Exception."""
        LOGGER.debug('Entered TickReplayEngine.replay')
        prev_ts: float | None = None
        try:
            for tick in self.ticks:
                if prev_ts is not None:
                    delay = (tick['timestamp'] - prev_ts) / max(self.speed, 0.0001)
                    if delay > 0:
                        time.sleep(delay)
                tick_callback(tick)
                prev_ts = float(tick['timestamp'])
                self.index += 1
        except Exception as exc:  # noqa: BLE001
            LOGGER.error('Failure in TickReplayEngine.replay: %s', exc)
            raise
