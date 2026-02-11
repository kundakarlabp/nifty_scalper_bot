"""Runtime controller for dynamic symbol universe membership."""

from __future__ import annotations

from typing import Iterable

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class UniverseController:
    """Track active universe snapshots and emit diffs only on membership changes."""

    def __init__(self) -> None:
        """Initialize empty universe snapshots. Args: none. Returns: None. Raises: None."""
        self.current_universe: set[str] = set()
        self.previous_universe: set[str] = set()

    def update(self, new_universe: Iterable[str]) -> tuple[set[str], set[str]]:
        """Apply new universe and return added/removed symbols. Args: new_universe. Returns: tuple[set[str], set[str]]. Raises: None."""
        next_universe = {
            str(symbol) for symbol in new_universe if str(symbol or "").strip()
        }
        added = next_universe - self.current_universe
        removed = self.current_universe - next_universe

        if added or removed:
            # Log only on transitions to avoid legacy per-tick warning spam.
            LOGGER.info(
                "Universe membership changed",
                extra={
                    "event": "universe_membership_changed",
                    "added": sorted(added),
                    "removed": sorted(removed),
                },
            )

        self.previous_universe = set(self.current_universe)
        self.current_universe = next_universe
        return added, removed
