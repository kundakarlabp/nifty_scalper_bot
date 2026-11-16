"""Purpose: Non-blocking, bounded history prewarming for selected NIFTY options.

Responsibilities: Prioritize selected and nearby same-expiry CE/PE contracts,
deduplicate outstanding tasks, and enforce a per-symbol cooldown.
Ownership: The caller's canonical market-data history orchestrator performs
all broker/history work and synchronizes the runner and indicator caches.
Safety: Best-effort diagnostic optimization; never alter trigger requirements,
risk thresholds, order routing, or global execution readiness.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable

LOGGER = logging.getLogger(__name__)


def _series_strike(symbol: str) -> tuple[str, int, str] | None:
    """Parse 5-digit NIFTY strikes without mistaking expiry digits for strikes."""
    s = str(symbol or "")
    if not s.startswith("NFO:NIFTY") or len(s) < 8:
        return None
    side = s[-2:]
    if side not in {"CE", "PE"} or not s[-7:-2].isdigit():
        return None
    return s[:-7], int(s[-7:-2]), side


def prioritized_vwap_symbols(
    selected_ce: str,
    selected_pe: str,
    universe: Iterable[str],
    *,
    limit: int = 6,
) -> tuple[str, ...]:
    """Selected contracts first, then nearby same-expiry CE/PE pairs."""
    ce = _series_strike(selected_ce)
    pe = _series_strike(selected_pe)
    if ce is None or pe is None or ce[0] != pe[0]:
        return ()
    selected = (selected_ce, selected_pe)
    candidates = {s for s in universe if _series_strike(s) is not None}
    ranked = sorted(
        (s for s in candidates if s not in selected),
        key=lambda s: (
            0 if _series_strike(s)[0] == ce[0] else 1,
            abs(_series_strike(s)[1] - (ce[1] if s.endswith("CE") else pe[1])),
            _series_strike(s)[2],
            s,
        ),
    )
    adjacent = [s for s in ranked if _series_strike(s)[0] == ce[0]]
    return tuple(dict.fromkeys((*selected, *adjacent)))[: max(2, limit)]


class VwapHistoryPrewarmer:
    """Rate-limited per-symbol tasks; keeps SMC and ORB evaluations unblocked."""

    def __init__(
        self,
        *,
        target_bars: int = 50,
        cooldown_seconds: float = 90.0,
        max_inflight: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.target_bars = max(50, int(target_bars))
        self.cooldown_seconds = max(15.0, float(cooldown_seconds))
        self.max_inflight = max(1, int(max_inflight))
        self._clock = clock
        self._last_attempt: dict[str, float] = {}
        self._inflight: dict[str, asyncio.Task[None]] = {}

    def note_attempt(self, symbol: str) -> None:
        """A normal universe hydration also counts as a request attempt."""
        self._last_attempt[symbol] = self._clock()

    async def _run(
        self,
        symbol: str,
        hydrate: Callable[[str, int], Awaitable[object]],
    ) -> None:
        try:
            await hydrate(symbol, self.target_bars)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.warning(
                "VWAP_PREWARM_FAILED symbol=%s error_type=%s",
                symbol, type(exc).__name__,
            )

    def schedule(
        self,
        selected_ce: str,
        selected_pe: str,
        universe: Iterable[str],
        *,
        completed_bars: Callable[[str], int],
        hydrate: Callable[[str, int], Awaitable[object]],
    ) -> tuple[str, ...]:
        """Schedule at most two fetch/sync operations; never await broker I/O."""
        for symbol, task in tuple(self._inflight.items()):
            if task.done():
                self._inflight.pop(symbol, None)
        now = self._clock()
        scheduled: list[str] = []
        for symbol in prioritized_vwap_symbols(selected_ce, selected_pe, universe):
            if len(self._inflight) >= self.max_inflight:
                break
            if symbol in self._inflight:
                continue
            if now - self._last_attempt.get(symbol, float("-inf")) < self.cooldown_seconds:
                continue
            try:
                if int(completed_bars(symbol)) >= self.target_bars:
                    continue
            except Exception:
                # Unknown history is not proof of readiness.
                pass
            self.note_attempt(symbol)
            task = asyncio.create_task(self._run(symbol, hydrate), name="vwap-prewarm")
            self._inflight[symbol] = task
            scheduled.append(symbol)
        return tuple(scheduled)

    def cancel_pending(self) -> None:
        for task in tuple(self._inflight.values()):
            if not task.done():
                task.cancel()
