"""Bounded temporal order-flow imbalance derived from canonical tick updates."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class _OfiState:
    version: object
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    observed_at: float
    events: deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=256)
    )
    snapshot: dict[str, object] = field(default_factory=dict)


class TemporalOfiAccumulator:
    """Maintain 1s/3s best-book OFI without owning market-data state."""

    def __init__(
        self,
        *,
        reset_gap_seconds: float = 5.0,
        max_events: int = 256,
    ) -> None:
        self._reset_gap_seconds = max(0.1, float(reset_gap_seconds))
        self._max_events = max(8, int(max_events))
        self._state: dict[str, _OfiState] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number else None

    def _best_book(
        self,
        quote: Mapping[str, Any],
    ) -> tuple[float, float, float, float] | None:
        bid = self._float(quote.get("bid") or quote.get("best_bid"))
        ask = self._float(quote.get("ask") or quote.get("best_ask"))
        depth = quote.get("depth")
        if not isinstance(depth, Mapping):
            return None
        bids = depth.get("buy")
        asks = depth.get("sell")
        if not isinstance(bids, list) or not bids:
            return None
        if not isinstance(asks, list) or not asks:
            return None
        best_bid = bids[0]
        best_ask = asks[0]
        if not isinstance(best_bid, Mapping) or not isinstance(best_ask, Mapping):
            return None
        bid_qty = self._float(best_bid.get("quantity"))
        ask_qty = self._float(best_ask.get("quantity"))
        if (
            bid is None
            or ask is None
            or bid_qty is None
            or ask_qty is None
            or bid <= 0.0
            or ask <= bid
            or bid_qty <= 0.0
            or ask_qty <= 0.0
        ):
            return None
        return bid, ask, bid_qty, ask_qty

    @staticmethod
    def _baseline_snapshot(
        bid_qty: float,
        ask_qty: float,
    ) -> dict[str, object]:
        return {
            "ofi_ready": False,
            "ofi_event": 0.0,
            "ofi_1s": 0.0,
            "ofi_3s": 0.0,
            "ofi_1s_normalized": 0.0,
            "ofi_3s_normalized": 0.0,
            "ofi_update_count_1s": 0,
            "ofi_update_count_3s": 0,
            "ofi_source": "runner_datahub_tick_updates",
            "queue_imbalance_top": (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1.0),
        }

    @staticmethod
    def _window(
        events: deque[tuple[float, float, float]],
        *,
        now: float,
        seconds: float,
    ) -> tuple[float, float, int]:
        selected = [event for event in events if now - event[0] <= seconds]
        if not selected:
            return 0.0, 0.0, 0
        total = sum(event[1] for event in selected)
        mean_depth = sum(event[2] for event in selected) / len(selected)
        return total, total / max(mean_depth, 1.0), len(selected)

    def update(
        self,
        symbol: str,
        quote: Mapping[str, Any],
        *,
        update_version: object | None,
        observed_at: float,
    ) -> dict[str, object]:
        """Return latest OFI snapshot after one accepted canonical tick."""
        book = self._best_book(quote)
        if book is None:
            with self._lock:
                self._state.pop(symbol, None)
            return self._baseline_snapshot(0.0, 0.0)
        bid, ask, bid_qty, ask_qty = book
        version: object = update_version
        if version in (None, "", 0, 0.0):
            version = (
                round(bid, 4),
                round(ask, 4),
                round(bid_qty, 2),
                round(ask_qty, 2),
            )
        now = float(observed_at)

        with self._lock:
            previous = self._state.get(symbol)
            if (
                previous is not None
                and previous.version == version
                and 0.0 <= now - previous.observed_at <= self._reset_gap_seconds
            ):
                return dict(previous.snapshot)

            if (
                previous is None
                or now <= previous.observed_at
                or now - previous.observed_at > self._reset_gap_seconds
            ):
                snapshot = self._baseline_snapshot(bid_qty, ask_qty)
                self._state[symbol] = _OfiState(
                    version=version,
                    bid=bid,
                    ask=ask,
                    bid_qty=bid_qty,
                    ask_qty=ask_qty,
                    observed_at=now,
                    events=deque(maxlen=self._max_events),
                    snapshot=snapshot,
                )
                return dict(snapshot)

            ofi_event = (
                (bid_qty if bid >= previous.bid else 0.0)
                - (previous.bid_qty if bid <= previous.bid else 0.0)
                - (ask_qty if ask <= previous.ask else 0.0)
                + (previous.ask_qty if ask >= previous.ask else 0.0)
            )
            depth_scale = max((bid_qty + ask_qty) / 2.0, 1.0)
            events = deque(previous.events, maxlen=self._max_events)
            events.append((now, float(ofi_event), float(depth_scale)))
            while events and now - events[0][0] > 3.0:
                events.popleft()

            ofi_1s, normalized_1s, count_1s = self._window(
                events,
                now=now,
                seconds=1.0,
            )
            ofi_3s, normalized_3s, count_3s = self._window(
                events,
                now=now,
                seconds=3.0,
            )
            snapshot = {
                "ofi_ready": count_1s >= 2,
                "ofi_event": float(ofi_event),
                "ofi_1s": ofi_1s,
                "ofi_3s": ofi_3s,
                "ofi_1s_normalized": normalized_1s,
                "ofi_3s_normalized": normalized_3s,
                "ofi_update_count_1s": count_1s,
                "ofi_update_count_3s": count_3s,
                "ofi_source": "runner_datahub_tick_updates",
                "queue_imbalance_top": (bid_qty - ask_qty)
                / max(bid_qty + ask_qty, 1.0),
            }
            self._state[symbol] = _OfiState(
                version=version,
                bid=bid,
                ask=ask,
                bid_qty=bid_qty,
                ask_qty=ask_qty,
                observed_at=now,
                events=events,
                snapshot=snapshot,
            )
            return dict(snapshot)


__all__ = ["TemporalOfiAccumulator"]
