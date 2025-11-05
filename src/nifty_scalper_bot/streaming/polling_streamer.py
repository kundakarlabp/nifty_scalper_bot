"""Polling-based market data streamer for Zerodha REST APIs."""

from __future__ import annotations

from contextlib import suppress
import math
import threading
import time
from typing import Any, Callable, Iterable, Sequence

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.metrics import Counter, Gauge

LOGGER = get_logger(__name__)


class PollingStreamer:
    """Production-safe poller that replaces WebSocket streaming."""

    def __init__(
        self,
        *,
        broker_client: Any,
        on_tick: Callable[[dict[str, Any]], None],
        instrument_resolver: Any,
        poll_interval_ms: int = 700,
        batch_size: int = 200,
        require_depth: bool = False,
        warn_on_rate_limit: bool = True,
    ) -> None:
        self._broker = broker_client
        self._on_tick = on_tick
        self._resolver = instrument_resolver
        self._interval_s = max(0.2, float(poll_interval_ms) / 1000.0)
        self._batch_size = max(1, int(batch_size))
        self._tokens: set[int] = set()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._require_depth = bool(require_depth)
        self._warn_on_rate_limit = bool(warn_on_rate_limit)
        self._rate_limit_warned = False

        self._m_poll_ok = Counter("poll_ok_total", "Successful polling rounds")
        self._m_poll_err = Counter("poll_err_total", "Polling rounds with error")
        self._m_tokens = Gauge("poll_tokens", "Tracked tokens for polling")
        self._m_interval = Gauge("poll_interval_seconds", "Polling interval (seconds)")

        with suppress(Exception):
            self._m_interval.set(self._interval_s)

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        worker = threading.Thread(
            target=self._run, name="polling-streamer", daemon=True
        )
        self._thread = worker
        worker.start()
        LOGGER.info(
            "Polling streamer started: interval=%.3fs batch_size=%d",
            self._interval_s,
            self._batch_size,
        )

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
        LOGGER.info("Polling streamer stopped")

    def is_running(self) -> bool:
        """Return ``True`` when the polling worker thread is active."""

        thread = self._thread
        if thread is None:
            return False
        if self._stop.is_set():
            return False
        return thread.is_alive()

    def ensure_running(self) -> None:
        """Ensure the polling worker thread is running."""

        if not self.is_running():
            self.start()

    def subscribe(self, tokens: Sequence[int]) -> None:
        with self._lock:
            self.ensure_running()
            self._tokens.update(int(token) for token in tokens)
            count = len(self._tokens)
            self._set_token_metric(count)
        self._maybe_warn_rate_limits(count)

    def subscribe_tokens(self, tokens: Sequence[int]) -> None:
        """Alias for :meth:`subscribe` used by env-driven bootstrapping."""

        self.subscribe(tokens)

    def unsubscribe(self, tokens: Sequence[int]) -> None:
        with self._lock:
            for token in tokens:
                self._tokens.discard(int(token))
            count = len(self._tokens)
            self._set_token_metric(count)
        self._maybe_warn_rate_limits(count)

    def tracked_tokens(self) -> list[int]:
        with self._lock:
            return sorted(self._tokens)

    def _set_token_metric(self, count: int) -> None:
        with suppress(Exception):
            self._m_tokens.set(count)

    def _maybe_warn_rate_limits(self, token_count: int) -> None:
        if not self._warn_on_rate_limit:
            return
        if token_count <= 0:
            self._rate_limit_warned = False
            return
        if self._interval_s <= 0:
            return
        polls_per_second = 1.0 / self._interval_s
        batches_per_poll = math.ceil(token_count / self._batch_size)
        estimated_reqs_per_sec = batches_per_poll * polls_per_second
        safe_capacity = self._batch_size * polls_per_second
        if token_count > safe_capacity:
            if not self._rate_limit_warned:
                LOGGER.warning(
                    "Token count %d may exceed REST rate limits", token_count
                )
                LOGGER.debug(
                    "Polling pressure details",
                    extra={
                        "tokens": token_count,
                        "batch_size": self._batch_size,
                        "interval_s": self._interval_s,
                        "estimated_reqs_per_sec": estimated_reqs_per_sec,
                    },
                )
                self._rate_limit_warned = True
        elif self._rate_limit_warned:
            LOGGER.info(
                "Token count back within estimated rate limits",
                extra={
                    "tokens": token_count,
                    "batch_size": self._batch_size,
                    "interval_s": self._interval_s,
                },
            )
            self._rate_limit_warned = False

    def _chunks(self, payload: list[int], size: int) -> Iterable[list[int]]:
        for idx in range(0, len(payload), size):
            yield payload[idx : idx + size]

    def _run(self) -> None:
        backoff = self._interval_s
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                tokens = self.tracked_tokens()
                if tokens:
                    for batch in self._chunks(tokens, self._batch_size):
                        ticks = self._fetch_ticks(batch)
                        for tick in ticks:
                            with suppress(Exception):
                                self._on_tick(tick)
                with suppress(Exception):
                    self._m_poll_ok.inc()
                backoff = self._interval_s
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Polling round failed: %s", exc)
                with suppress(Exception):
                    self._m_poll_err.inc()
                backoff = min(max(backoff * 2.0, self._interval_s), 8.0)
            elapsed = max(0.0, time.monotonic() - started)
            sleep_for = max(0.0, backoff - elapsed)
            self._stop.wait(sleep_for)

    def _fetch_ticks(self, batch: list[int]) -> list[dict[str, Any]]:
        timestamp_ms = int(time.time() * 1000)
        if not self._require_depth:
            ticks = self._try_ltp_bulk(batch, timestamp_ms)
            if ticks:
                return ticks

        ticks = self._try_quote_bulk(batch, timestamp_ms)
        if ticks:
            return ticks

        if self._require_depth:
            # Depth is required but the bulk endpoint failed. Fall back to LTP
            # to at least provide price updates while surfacing logs.
            ticks = self._try_ltp_bulk(batch, timestamp_ms)
            if ticks:
                return ticks

        get_quote_single = getattr(self._broker, "get_quote_by_token", None)
        ticks = []
        if callable(get_quote_single):
            for token in batch:
                try:
                    quote = get_quote_single(int(token))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug(
                        "Single quote lookup failed",
                        extra={"token": int(token), "error": str(exc)},
                    )
                    continue
                lp = float(quote.get("last_price", 0.0) or 0.0)
                if lp <= 0:
                    continue
                tick: dict[str, Any] = {
                    "instrument_token": int(token),
                    "last_price": lp,
                    "timestamp": timestamp_ms,
                }
                depth = quote.get("depth")
                if isinstance(depth, dict):
                    tick["depth"] = depth
                ticks.append(tick)
            return ticks

        LOGGER.debug("Polling fallback hit without quote helpers; skipping batch")
        return []

    def _try_ltp_bulk(
        self, batch: list[int], timestamp_ms: int
    ) -> list[dict[str, Any]] | None:
        fetch_ltp = getattr(self._broker, "get_ltp_bulk", None)
        if not callable(fetch_ltp):
            return None
        try:
            data = fetch_ltp(batch)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("LTP bulk fetch failed", extra={"error": str(exc)})
            return None
        ticks: list[dict[str, Any]] = []
        for token in batch:
            ltp = float(data.get(int(token), 0.0) or 0.0)
            if ltp <= 0:
                continue
            ticks.append(
                {
                    "instrument_token": int(token),
                    "last_price": ltp,
                    "timestamp": timestamp_ms,
                }
            )
        return ticks or None

    def _try_quote_bulk(
        self, batch: list[int], timestamp_ms: int
    ) -> list[dict[str, Any]] | None:
        fetch_quote_bulk = getattr(self._broker, "get_quote_bulk", None)
        if not callable(fetch_quote_bulk):
            return None
        try:
            quote_map = fetch_quote_bulk(batch)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Quote bulk fetch failed", extra={"error": str(exc)})
            return None
        ticks: list[dict[str, Any]] = []
        for token, quote in quote_map.items():
            lp = float(quote.get("last_price", 0.0) or 0.0)
            if lp <= 0:
                continue
            tick: dict[str, Any] = {
                "instrument_token": int(token),
                "last_price": lp,
                "timestamp": timestamp_ms,
            }
            depth = quote.get("depth")
            if isinstance(depth, dict):
                tick["depth"] = depth
            ticks.append(tick)
        return ticks or None
