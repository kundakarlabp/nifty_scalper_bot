"""Canonical data hub for cached ticks, orders, and positions."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re
from threading import RLock
import time
from typing import Any, Callable, Iterable, Mapping, Optional, TypedDict, cast

from nifty_scalper_bot.data.symbols import (
    NIFTY_SPOT_CANONICAL_SYMBOL,
    normalize_hub_symbol,
)
from nifty_scalper_bot.storage.hub_store import HubStore, WalEntry
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.options_math import (
    black_scholes_greeks,
    implied_volatility,
)

LOGGER = get_logger(__name__)


Tick = dict[str, Any]
OrderListener = Callable[[dict[str, Any]], None]
TickListener = Callable[[dict[str, Any]], None]


class Freshness(TypedDict, total=False):
    """Container describing cached quote freshness metrics."""

    ok: bool
    mono_age_ms: float | None
    server_age_ms: float | None
    effective_ms: float | None
    threshold_ms: float
    reason: str | None


_ORDER_STATE_MACHINE: dict[str, set[str]] = {
    "": {
        "pending",
        "submitted",
        "partial_filled",
        "filled",
        "cancelled",
        "rejected",
        "expired",
    },
    "pending": {
        "submitted",
        "partial_filled",
        "cancelled",
        "rejected",
        "expired",
    },
    "submitted": {
        "partial_filled",
        "filled",
        "cancelled",
        "rejected",
        "expired",
    },
    "partial_filled": {
        "filled",
        "cancelled",
        "rejected",
        "expired",
    },
    "filled": set(),
    "cancelled": set(),
    "rejected": set(),
    "expired": set(),
}


class DataHub:
    """Central façade exposing cached market, order, and broker state."""

    def __init__(
        self,
        mdm: Any,
        resolver: Any,
        *,
        options_only: bool = True,
        store: HubStore | None = None,
        checkpoint_interval: float = 5.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialise hub with managers and cache containers."""

        self._mdm = mdm
        self._resolver = resolver
        self._options_only = options_only
        self._lock = RLock()
        self._quotes: dict[str, Tick] = {}
        self._positions: dict[str, dict[str, Any]] = {}
        self._orders: dict[str, dict[str, Any]] = {}
        self._subs: dict[str, list[TickListener]] = {}
        self._order_subs: list[OrderListener] = []
        self._order_sequences: dict[str, int] = {}
        self._order_status: dict[str, str] = {}
        self._iv_cache: dict[str, float] = {}
        self._oi_cache: dict[str, int] = {}
        self._greeks_cache: dict[str, dict[str, float]] = {}
        self._store = store
        self._checkpoint_interval = max(0.1, float(checkpoint_interval))
        self._clock = clock or time.time
        self._monotonic: Callable[[], float] = time.monotonic
        self._now: Callable[[], float]
        self._start_mono = self._monotonic()
        base_wall = float(self._clock())
        base_mono = float(self._monotonic())
        self._monotonic_offset = base_wall - base_mono
        self._now = self._monotonic_now
        self._last_snapshot_ts = 0.0
        self._option_chain_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._option_chain_ttl = 5.0
        self._spot_symbol = NIFTY_SPOT_CANONICAL_SYMBOL
        self._risk_free_rate = 0.06

        error_cooldown = self._parse_env_float("TICK_ERROR_LOG_COOLDOWN_SEC")
        if error_cooldown is None:
            error_cooldown = 10.0
        self._error_log_cooldown = max(0.0, error_cooldown)

        self._poll_interval_s = self._resolve_poll_interval_s()
        self._poll_interval_ms = self._poll_interval_s * 1000.0

        stale_override = self._parse_env_float("STALE_TICK_MAX_AGE_MS")
        if stale_override is not None:
            self._stale_tick_max_age_ms = max(0.0, stale_override)
        else:
            adaptive_ms = int(self._poll_interval_s * 1000.0 * 2.5)
            self._stale_tick_max_age_ms = float(max(2000, min(adaptive_ms, 5000)))

        drift_budget = self._parse_env_float("DRIFT_BUDGET_MS")
        if drift_budget is None:
            drift_budget = 400.0
        self._drift_budget_ms = max(0.0, drift_budget)

        warmup_grace = self._parse_env_float("WARMUP_GRACE_S")
        if warmup_grace is None:
            warmup_grace = 5.0
        self._warmup_grace_s = max(0.0, warmup_grace)
        if self._warmup_grace_s > 0.0:
            self._warmup_reset_gap = max(
                self._warmup_grace_s,
                self._poll_interval_s * 4.0,
                1.0,
            )
        else:
            self._warmup_reset_gap = 0.0
        self._warmup_deadline: float | None = None
        self._warmup_drop_ms = 10_000.0
        self._stale_hard_cap_ms = 10_000.0
        self._stale_quorum_required = 2
        self._stale_quorum: dict[str, int] = {}
        self._last_tick_seen_mono: float | None = None
        self._listener_error_streak = 0
        self._last_listener_error_log = 0.0
        self._last_stale_log = 0.0
        self._log_lock = RLock()

        if self._store is not None:
            self._restore_from_store()

        self.reset_warmup()

    # ------------------------------------------------------------------
    # Symbol helpers
    @staticmethod
    def normalize(sym: Any) -> str:
        """Normalise trading symbol with prefix trimming."""

        return normalize_hub_symbol(sym)

    def _allow(self, symbol: str) -> bool:
        if not symbol:
            return False
        if not self._options_only:
            return True
        return (
            symbol.endswith("CE")
            or symbol.endswith("PE")
            or symbol == NIFTY_SPOT_CANONICAL_SYMBOL
        )

    def reset_warmup(self) -> None:
        """Restart the warm-up grace period for stale tick evaluation."""

        if self._warmup_grace_s <= 0.0:
            self._warmup_deadline = None
            self._last_tick_seen_mono = None
            self._stale_quorum.clear()
            return
        now_mono = float(self._monotonic())
        self._warmup_deadline = now_mono + self._warmup_grace_s
        self._last_tick_seen_mono = None
        self._stale_quorum.clear()

    # ------------------------------------------------------------------
    # Broker account state
    def get_account_snapshot(self, *, force: bool = False) -> dict[str, float]:
        """Return broker margin snapshot sourced from the market data manager.

        Args:
            force: Force refresh against the market data manager when ``True``.

        Returns:
            dict[str, float]: Normalized snapshot of broker balances.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered DataHub.get_account_snapshot",
            extra={"event": "data_hub_account_snapshot_enter", "force": force},
        )
        fetcher = getattr(self._mdm, "get_account_snapshot", None)
        if not callable(fetcher):
            LOGGER.error(
                "data_hub_account_snapshot_missing_mdm",
                extra={"event": "data_hub_account_snapshot_missing_mdm"},
            )
            return {}
        try:
            snapshot = fetcher(force=force)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error(
                "Failure in DataHub.get_account_snapshot: %s",
                exc,
                extra={"event": "data_hub_account_snapshot_error"},
                exc_info=exc,
            )
            return {}
        normalized = dict(snapshot or {})
        if normalized:
            LOGGER.info(
                "data_hub_account_snapshot_available",
                extra={
                    "event": "data_hub_account_snapshot_available",
                    "available": normalized.get("available"),
                    "net": normalized.get("net"),
                },
            )
        else:
            LOGGER.warning(
                "data_hub_account_snapshot_empty",
                extra={"event": "data_hub_account_snapshot_empty"},
            )
        return normalized

    def get_available_balance(self, *, force: bool = False) -> float | None:
        """Return available broker balance with market data manager as source.

        Args:
            force: Force refresh of the upstream market data cache when ``True``.

        Returns:
            float | None: Positive available balance when present, otherwise ``None``.

        Raises:
            None.
        """

        LOGGER.debug(
            "Entered DataHub.get_available_balance",
            extra={"event": "data_hub_available_balance_enter", "force": force},
        )
        balance_value: float | None = None
        fetcher = getattr(self._mdm, "get_available_balance", None)
        if callable(fetcher):
            try:
                balance_value = fetcher(force=force)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "Failure in DataHub.get_available_balance: %s",
                    exc,
                    extra={"event": "data_hub_available_balance_error"},
                    exc_info=exc,
                )
                balance_value = None
        if balance_value is None:
            snapshot = self.get_account_snapshot(force=force)
            for key in (
                "available",
                "live_balance",
                "cash",
                "opening_balance",
                "net",
            ):
                value_option = snapshot.get(key)
                if value_option is None:
                    continue
                try:
                    numeric = float(value_option)
                except (TypeError, ValueError):
                    continue
                if numeric > 0:
                    balance_value = numeric
                    break
        if balance_value is not None and balance_value > 0:
            resolved_balance = float(balance_value)
            LOGGER.info(
                "data_hub_available_balance_resolved",
                extra={
                    "event": "data_hub_available_balance_resolved",
                    "balance": round(resolved_balance, 2),
                },
            )
            return resolved_balance
        LOGGER.warning(
            "data_hub_available_balance_unavailable",
            extra={"event": "data_hub_available_balance_unavailable"},
        )
        return None

    def _update_warmup_window(self, now_mono: float) -> bool:
        if self._warmup_grace_s <= 0.0:
            self._warmup_deadline = None
            self._last_tick_seen_mono = now_mono
            return False

        deadline = self._warmup_deadline
        if deadline is not None and now_mono >= deadline:
            deadline = None

        last_seen = self._last_tick_seen_mono
        if (
            last_seen is not None
            and self._warmup_reset_gap > 0.0
            and now_mono - last_seen >= self._warmup_reset_gap
        ):
            deadline = now_mono + self._warmup_grace_s

        if deadline is None:
            base_deadline = self._start_mono + self._warmup_grace_s
            if now_mono < base_deadline:
                deadline = base_deadline

        self._warmup_deadline = deadline
        self._last_tick_seen_mono = now_mono
        return deadline is not None and now_mono < deadline

    # ------------------------------------------------------------------
    # Tick ingestion & fan-out
    def ingest_tick(self, tick: Mapping[str, Any]) -> None:
        """Cache *tick* and fan-out to registered listeners."""

        payload = self._sanitize_tick(dict(tick))
        symbol = self.normalize(
            payload.get("symbol") or payload.get("tradingsymbol") or ""
        )
        if not self._allow(symbol):
            return
        now_mono = float(self._monotonic())
        warmup_active = self._update_warmup_window(now_mono)

        payload_copy = dict(payload)
        ingest_wall = float(self._now())
        drift_ms = self._annotate_tick_timings(payload_copy, ingest_wall, now_mono)
        (
            effective_age_ms,
            curr_mono_age_ms,
            server_age_ms,
            _,
        ) = self._compute_tick_age_metrics(
            payload_copy, now_mono=now_mono, now_wall=ingest_wall
        )
        previous_tick: dict[str, Any] | None = None
        with self._lock:
            cached = self._quotes.get(symbol)
            if cached is not None:
                previous_tick = dict(cached)
        already_cached = previous_tick is not None

        previous_mono_age_ms: float | None = None
        if previous_tick is not None:
            _, prev_mono_age_ms, _, _ = self._compute_tick_age_metrics(
                previous_tick, now_mono=now_mono, now_wall=ingest_wall
            )
            previous_mono_age_ms = prev_mono_age_ms

        decision_mono_age_ms = curr_mono_age_ms
        if previous_mono_age_ms is not None and (
            decision_mono_age_ms is None or previous_mono_age_ms > decision_mono_age_ms
        ):
            decision_mono_age_ms = previous_mono_age_ms

        drop_stale, drop_reason = self._should_drop_stale(
            symbol,
            already_cached=already_cached,
            warmup_active=warmup_active,
            mono_age_ms=decision_mono_age_ms,
            server_age_ms=server_age_ms,
        )
        if drop_stale:
            age_for_log = (
                server_age_ms if server_age_ms is not None else effective_age_ms
            )
            self._log_stale_tick(
                symbol,
                age_for_log or 0.0,
                mono_ms=decision_mono_age_ms,
                server_ms=server_age_ms,
                drift_ms=drift_ms,
                threshold_ms=self._stale_tick_max_age_ms,
                reason=drop_reason,
            )
            return
        with self._lock:
            self._quotes[symbol] = payload_copy
            callbacks = list(self._subs.get(symbol, ()))
        self._capture_option_metrics(symbol, payload_copy)
        self._persist_wal("quote", symbol, payload_copy)
        self._maybe_checkpoint()
        had_error = False
        for callback in callbacks:
            try:
                callback(dict(payload_copy))
            except Exception:  # pragma: no cover - defensive
                had_error = True
                self._listener_error_streak += 1
                if self._should_log_listener_error():
                    failure_age = self._compute_tick_age_ms(payload_copy)
                    LOGGER.error(
                        "tick_listener_failed consecutive=%d symbol=%s age_ms=%s",
                        self._listener_error_streak,
                        symbol,
                        (
                            f"{failure_age:.0f}"
                            if failure_age is not None
                            else "unknown"
                        ),
                        exc_info=True,
                    )
        if not had_error:
            self._listener_error_streak = 0

    def get_option_chain(
        self, expiry: str, *, force_refresh: bool = False
    ) -> list[dict[str, Any]] | None:
        """Return option chain for *expiry* using the underlying market data source."""

        key = str(expiry or "").strip().lower()
        if not key:
            return None
        now = self._clock()
        cached: tuple[float, list[dict[str, Any]]] | None = None
        with self._lock:
            cached = self._option_chain_cache.get(key)
            if (
                cached is not None
                and not force_refresh
                and now - cached[0] <= self._option_chain_ttl
            ):
                return [dict(item) for item in cached[1]]

        provider = self._resolve_chain_provider()
        if provider is None:
            return cached[1] if cached is not None else None

        try:
            chain = provider(expiry=expiry)
        except TypeError:
            chain = provider(expiry)
        if chain is None:
            return cached[1] if cached is not None else None

        normalized: list[dict[str, Any]] = []
        for entry in chain:
            if isinstance(entry, Mapping):
                normalized.append(dict(entry))
        with self._lock:
            self._option_chain_cache[key] = (now, normalized)
        return [dict(item) for item in normalized]

    def is_fresh(self, symbol: str) -> tuple[bool, Freshness]:
        """Return whether ``symbol`` has a fresh quote cached."""

        normalized = self.normalize(symbol)
        meta: Freshness = {
            "ok": True,
            "threshold_ms": self._stale_tick_max_age_ms,
            "reason": None,
            "mono_age_ms": None,
            "server_age_ms": None,
            "effective_ms": None,
        }
        if not normalized:
            return True, meta

        with self._lock:
            cached = self._quotes.get(normalized)
        if cached is None:
            meta["ok"] = False
            meta["reason"] = "no_tick"
            return False, meta

        now_mono = float(self._monotonic())
        now_wall = float(self._now())
        (
            effective_ms,
            mono_age_ms,
            server_age_ms,
            _,
        ) = self._compute_tick_age_metrics(cached, now_mono=now_mono, now_wall=now_wall)
        meta["effective_ms"] = effective_ms
        meta["mono_age_ms"] = mono_age_ms
        meta["server_age_ms"] = server_age_ms

        drop_stale, reason = self._should_drop_stale(
            normalized,
            already_cached=True,
            warmup_active=self._in_warmup(now_mono),
            mono_age_ms=mono_age_ms,
            server_age_ms=server_age_ms,
        )
        meta["reason"] = reason
        fresh = not drop_stale
        meta["ok"] = fresh
        return fresh, meta

    def _compute_tick_age_ms(self, payload: Mapping[str, Any]) -> float | None:
        age_ms, _, _, _ = self._compute_tick_age_metrics(payload)
        return age_ms

    def _compute_tick_age_metrics(
        self,
        payload: Mapping[str, Any],
        *,
        now_mono: float | None = None,
        now_wall: float | None = None,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        if now_mono is None:
            now_mono = float(self._monotonic())
        if now_wall is None:
            now_wall = float(self._now())

        ingest_mono = self._float(payload.get("ingest_mono_s"))
        if ingest_mono is None:
            ingest_mono = self._float(payload.get("_ingest_mono"))
        mono_age_ms: float | None = None
        if ingest_mono is not None:
            mono_age_ms = max(0.0, (now_mono - ingest_mono) * 1000.0)

        server_ts = self._float(payload.get("server_ts_s"))
        if server_ts is None:
            server_ts_ms = self._float(payload.get("_server_ts_ms"))
            if server_ts_ms is not None:
                server_ts = server_ts_ms / 1000.0

        server_age_ms: float | None = None
        if server_ts is not None:
            server_age_ms = max(0.0, (now_wall - server_ts) * 1000.0)

        drift_ms = self._float(payload.get("_server_drift_ms"))

        effective: float | None
        if mono_age_ms is not None and server_age_ms is not None:
            effective = max(mono_age_ms, server_age_ms)
        elif server_age_ms is not None:
            effective = server_age_ms
        else:
            effective = mono_age_ms

        return effective, mono_age_ms, server_age_ms, drift_ms

    def _should_drop_stale(
        self,
        symbol: str,
        *,
        already_cached: bool,
        warmup_active: bool,
        mono_age_ms: float | None,
        server_age_ms: float | None,
    ) -> tuple[bool, str | None]:
        if not already_cached:
            self._stale_quorum.pop(symbol, None)

        if warmup_active:
            self._stale_quorum.pop(symbol, None)
            if server_age_ms is not None and server_age_ms >= self._warmup_drop_ms:
                return True, "warmup_hard_cap"
            return False, None

        threshold = self._stale_tick_max_age_ms
        if threshold <= 0.0:
            self._stale_quorum.pop(symbol, None)
            return False, None

        hard_cap = max(self._stale_hard_cap_ms, threshold)
        if server_age_ms is not None and server_age_ms >= hard_cap:
            self._stale_quorum.pop(symbol, None)
            return True, "hard_cap"

        if (
            mono_age_ms is None
            or server_age_ms is None
            or mono_age_ms <= threshold
            or server_age_ms <= threshold + self._drift_budget_ms
        ):
            self._stale_quorum.pop(symbol, None)
            return False, None

        count = self._stale_quorum.get(symbol, 0) + 1
        self._stale_quorum[symbol] = count
        if count >= self._stale_quorum_required:
            self._stale_quorum.pop(symbol, None)
            return True, "quorum"

        return False, None

    def _annotate_tick_timings(
        self, payload: dict[str, Any], ingest_wall: float, ingest_mono: float
    ) -> float | None:
        payload["ingest_mono_s"] = ingest_mono
        payload["_ingest_mono"] = ingest_mono
        timestamp = self._extract_tick_timestamp(payload)
        if timestamp is None or ingest_wall <= 0:
            payload.pop("_server_drift_ms", None)
            payload.pop("_server_ts_ms", None)
            payload.pop("server_ts_s", None)
            return None
        drift_ms = max(0.0, (ingest_wall - timestamp) * 1000.0)
        payload["_server_drift_ms"] = drift_ms
        payload["server_ts_s"] = timestamp
        payload["_server_ts_ms"] = timestamp * 1000.0
        return drift_ms

    def _monotonic_now(self) -> float:
        return self._monotonic_offset + self._monotonic()

    def _extract_tick_timestamp(self, payload: Mapping[str, Any]) -> float | None:
        candidates = (
            "exchange_timestamp",
            "last_trade_time",
            "ts_ms",
            "ts",
            "timestamp",
            "last_update",
            "last_update",
            "last_trade_time",
            "timestamp",
            "ts",
            "ts_ms",
        )
        for key in candidates:
            raw = payload.get(key)
            if raw is None:
                continue
            if isinstance(raw, datetime):
                value = raw
            elif isinstance(raw, str):
                text = raw.strip()
                if not text:
                    continue
                try:
                    numeric = float(text)
                except ValueError:
                    try:
                        parsed = datetime.fromisoformat(text)
                    except ValueError:
                        continue
                    value = parsed
                else:
                    if numeric > 1_000_000_000_000:
                        numeric /= 1000.0
                    return numeric
            else:
                try:
                    numeric = float(raw)
                except (TypeError, ValueError):
                    continue
                if numeric > 1_000_000_000_000:
                    numeric /= 1000.0
                return numeric

            if isinstance(value, datetime):
                aware = value
                if value.tzinfo is None:
                    aware = value.replace(tzinfo=timezone.utc)
                return aware.timestamp()
        return None

    def _in_warmup(self, now_mono: float | None = None) -> bool:
        if self._warmup_grace_s <= 0:
            return False
        current = now_mono if now_mono is not None else float(self._monotonic())
        deadline = self._warmup_deadline
        if deadline is not None:
            return current < deadline
        return (current - self._start_mono) <= self._warmup_grace_s

    @staticmethod
    def _parse_env_float(key: str) -> float | None:
        raw = os.getenv(key)
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _coerce_positive_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric <= 0.0:
            return None
        return numeric

    def _resolve_poll_interval_s(self) -> float:
        attr_sources = (
            getattr(self._mdm, "_interval_s", None),
            getattr(self._mdm, "poll_interval_seconds", None),
            getattr(getattr(self._mdm, "streamer", None), "_interval_s", None),
        )
        for candidate in attr_sources:
            value = self._coerce_positive_float(candidate)
            if value is not None:
                return max(0.2, value)

        env_seconds = (
            "POLL_INTERVAL_SECONDS",
            "POLL_INTERVAL_SEC",
            "POLLING__INTERVAL_SEC",
        )
        for key in env_seconds:
            value = self._parse_env_float(key)
            if value is not None and value > 0.0:
                return max(0.2, value)

        env_ms = (
            "POLL_INTERVAL_MS",
            "POLLING__INTERVAL_MS",
            "POLL_INTERVAL_MILLISECONDS",
        )
        for key in env_ms:
            value = self._parse_env_float(key)
            if value is not None and value > 0.0:
                return max(0.2, value / 1000.0)

        return 0.7

    def _should_log_listener_error(self) -> bool:
        if self._error_log_cooldown <= 0:
            return True
        now = self._now()
        with self._log_lock:
            if now - self._last_listener_error_log >= self._error_log_cooldown:
                self._last_listener_error_log = now
                return True
        return False

    def _log_stale_tick(
        self,
        symbol: str,
        age_ms: float,
        *,
        mono_ms: float | None,
        server_ms: float | None,
        drift_ms: float | None,
        threshold_ms: float,
        reason: str | None,
    ) -> None:
        if self._error_log_cooldown > 0:
            now = self._now()
            with self._log_lock:
                if now - self._last_stale_log < self._error_log_cooldown:
                    return
                self._last_stale_log = now
        LOGGER.warning(
            "stale_tick_dropped symbol=%s age_ms=%0.0f threshold_ms=%0.0f reason=%s",
            symbol,
            age_ms,
            threshold_ms,
            reason or "threshold",
        )
        LOGGER.info(
            (
                "stale_diag symbol=%s age_ms=%0.0f mono_age_ms=%s "
                "server_age_ms=%s drift_ms=%s poll_ms=%0.0f "
                "threshold_ms=%0.0f drift_budget_ms=%0.0f reason=%s"
            ),
            symbol,
            age_ms,
            f"{mono_ms:.0f}" if mono_ms is not None else "na",
            f"{server_ms:.0f}" if server_ms is not None else "na",
            f"{drift_ms:.0f}" if drift_ms is not None else "na",
            self._poll_interval_ms,
            threshold_ms,
            self._drift_budget_ms,
            reason or "threshold",
        )

    def _resolve_chain_provider(self) -> Callable[..., Any] | None:
        for source in (self._mdm, self._resolver):
            if source is None:
                continue
            for attr in ("get_option_chain", "fetch_option_chain", "option_chain"):
                candidate = getattr(source, attr, None)
                if callable(candidate):
                    return candidate
        return None

    def subscribe_ticks(self, symbol: str, cb: TickListener) -> None:
        """Subscribe *cb* to tick updates for *symbol*."""

        normalized = self.normalize(symbol)
        if not self._allow(normalized):
            return
        warm: Optional[Tick] = None
        should_wire = False
        with self._lock:
            listeners = self._subs.setdefault(normalized, [])
            if cb not in listeners:
                listeners.append(cb)
            should_wire = len(listeners) == 1
            warm = self._quotes.get(normalized)
        if warm:
            try:
                cb(dict(warm))
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("tick_listener_warm_failed")
        if should_wire and hasattr(self._mdm, "subscribe"):
            self._mdm.subscribe(normalized, self.ingest_tick)

    def unsubscribe_ticks(self, symbol: str, cb: TickListener) -> None:
        """Remove *cb* subscription for *symbol*."""

        normalized = self.normalize(symbol)
        if not normalized:
            return
        should_release = False
        with self._lock:
            listeners = self._subs.get(normalized)
            if not listeners:
                return
            try:
                listeners.remove(cb)
            except ValueError:
                pass
            if not listeners:
                self._subs.pop(normalized, None)
                should_release = True
        if should_release and hasattr(self._mdm, "unsubscribe"):
            self._mdm.unsubscribe(normalized, self.ingest_tick)

    def get_quote(self, symbol: str, *, allow_pull: bool = True) -> Optional[Tick]:
        """Return cached quote, optionally pulling via market data manager."""

        normalized = self.normalize(symbol)
        with self._lock:
            cached = self._quotes.get(normalized)
        if cached:
            return dict(cached)
        if allow_pull:
            pulled = self.pull_quote(normalized)
            if pulled is not None:
                return dict(pulled)
        return None

    def get_iv(self, symbol: str, *, allow_fallback: bool = True) -> Optional[float]:
        """Return implied volatility for *symbol* if known."""

        normalized = self.normalize(symbol)
        with self._lock:
            cached = self._iv_cache.get(normalized)
            quote = dict(self._quotes.get(normalized, {}))
        if cached is not None:
            return cached
        if not allow_fallback:
            return None
        computed = self._compute_implied_volatility(normalized, quote)
        if computed is not None:
            with self._lock:
                self._iv_cache[normalized] = computed
        return computed

    def get_oi(self, symbol: str) -> Optional[int]:
        """Return last known open interest for *symbol*."""

        normalized = self.normalize(symbol)
        with self._lock:
            cached = self._oi_cache.get(normalized)
        return cached

    def get_greeks(self, symbol: str) -> Optional[dict[str, float]]:
        """Return Black-Scholes greeks for *symbol* if available."""

        normalized = self.normalize(symbol)
        with self._lock:
            cached = self._greeks_cache.get(normalized)
            quote = dict(self._quotes.get(normalized, {}))
        if cached is not None:
            return dict(cached)
        greeks = self._compute_greeks(normalized, quote)
        if greeks is None:
            return None
        with self._lock:
            self._greeks_cache[normalized] = greeks
        return dict(greeks)

    def expected_move(self, days: float = 1.0) -> Optional[float]:
        """Return daily expected move using ATM IV."""

        spot_quote = self.get_quote(self._spot_symbol, allow_pull=False)
        if not spot_quote:
            return None
        spot = self._float(
            spot_quote.get("ltp")
            or spot_quote.get("last_price")
            or spot_quote.get("close")
            or spot_quote.get("open")
        )
        if spot is None or spot <= 0.0:
            return None
        atm_iv = self._atm_iv(spot)
        if atm_iv is None:
            return None
        days = max(float(days), 0.0)
        if days == 0.0:
            return 0.0
        return spot * atm_iv * (days / 365.0) ** 0.5

    def pull_quote(self, symbol: str) -> Optional[Tick]:
        """Fetch the latest quote via the market data manager and cache it."""

        normalized = self.normalize(symbol)
        if not self._allow(normalized):
            return None
        if not hasattr(self._mdm, "pull_quote"):
            return None
        quote = self._mdm.pull_quote(normalized)
        if isinstance(quote, Mapping):
            self.ingest_tick(quote)
            return dict(quote)
        return None

    # ------------------------------------------------------------------
    # Positions
    def replace_positions(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Replace cached positions with *rows*."""

        snapshot: dict[str, dict[str, Any]] = {}
        for row in rows:
            symbol = self.normalize(row.get("symbol"))
            if not self._allow(symbol):
                continue
            qty_raw = row.get("quantity", 0)
            avg_raw = row.get("average_price", 0.0)
            try:
                qty_val = int(float(qty_raw or 0))
            except Exception:
                qty_val = 0
            try:
                avg_val = float(avg_raw or 0.0)
            except Exception:
                avg_val = 0.0
            side = "LONG" if qty_val > 0 else "SHORT" if qty_val < 0 else "FLAT"
            snapshot[symbol] = {
                "symbol": symbol,
                "quantity": abs(qty_val),
                "average_price": avg_val,
                "side": side,
            }
        with self._lock:
            self._positions = snapshot
        snapshot_payload = {key: dict(value) for key, value in snapshot.items()}
        self._persist_wal("positions", "__all__", snapshot_payload)
        self._maybe_checkpoint()

    def refresh_positions(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Compatibility wrapper for position refresh events."""

        self.replace_positions(rows)

    def positions(self) -> list[dict[str, Any]]:
        """Return cached positions."""

        with self._lock:
            return [dict(value) for value in self._positions.values()]

    # ------------------------------------------------------------------
    # Orders
    def upsert_order(self, payload: Mapping[str, Any]) -> None:
        """Insert or update cached order state from *payload*."""

        symbol = self.normalize(payload.get("symbol"))
        order_id = str(payload.get("order_id") or payload.get("id") or "").strip()
        if not order_id or not self._allow(symbol):
            return
        side_raw = payload.get("side") or payload.get("transaction_type") or ""
        status_raw = payload.get("status") or payload.get("order_status") or ""
        filled_raw = payload.get("filled_quantity") or payload.get("filled")
        status_normalized = str(status_raw).lower()
        sequence = self._sequence(payload)
        row = {
            "order_id": order_id,
            "symbol": symbol,
            "side": str(side_raw).upper(),
            "quantity": self._int(payload.get("quantity")),
            "filled_quantity": self._int(filled_raw),
            "status": status_normalized,
            "price": self._float(payload.get("price")),
            "trigger_price": self._float(payload.get("trigger_price")),
            "parent_order_id": payload.get("parent_order_id"),
            "child_order_ids": list(payload.get("child_order_ids", [])),
            "ts": self._ts(payload.get("timestamp") or payload.get("updated_at")),
            "status_sequence": sequence,
        }
        with self._lock:
            previous_sequence = self._order_sequences.get(order_id)
            if (
                sequence is not None
                and previous_sequence is not None
                and sequence <= previous_sequence
            ):
                LOGGER.debug(
                    "order_event_dropped_duplicate",
                    extra={
                        "event": "order_event_dropped_duplicate",
                        "order_id": order_id,
                        "sequence": sequence,
                        "previous": previous_sequence,
                    },
                )
                return
            previous_status = self._order_status.get(order_id, "")
            if not self._is_valid_transition(previous_status, status_normalized):
                LOGGER.warning(
                    "order_event_invalid_transition",
                    extra={
                        "event": "order_event_invalid_transition",
                        "order_id": order_id,
                        "from": previous_status,
                        "to": status_normalized,
                    },
                )
                return
            self._orders[order_id] = row
            if sequence is not None:
                self._order_sequences[order_id] = sequence
            elif order_id not in self._order_sequences:
                self._order_sequences[order_id] = 0
            self._order_status[order_id] = status_normalized
            listeners = list(self._order_subs)
        self._persist_wal("order", order_id, dict(row))
        self._maybe_checkpoint()
        for listener in listeners:
            try:
                listener(dict(row))
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("order_listener_failed")

    def orders(self) -> list[dict[str, Any]]:
        """Return cached orders."""

        with self._lock:
            return [dict(value) for value in self._orders.values()]

    def subscribe_orders(self, cb: OrderListener) -> None:
        """Register *cb* for order updates."""

        with self._lock:
            if cb not in self._order_subs:
                self._order_subs.append(cb)

    # ------------------------------------------------------------------
    def checkpoint(self) -> None:
        """Persist a durable snapshot of the hub state."""

        if self._store is None:
            return
        with self._lock:
            quotes_payload = {
                symbol: dict(tick) for symbol, tick in self._quotes.items()
            }
            positions_payload = {
                symbol: dict(value) for symbol, value in self._positions.items()
            }
            orders_payload = {
                order_id: dict(value) for order_id, value in self._orders.items()
            }
            sequences_payload = dict(self._order_sequences)
            status_payload = dict(self._order_status)
        try:
            self._store.save_snapshot("quotes", quotes_payload)
            self._store.save_snapshot("positions", positions_payload)
            self._store.save_snapshot("orders", orders_payload)
            self._store.save_snapshot("order_sequences", sequences_payload)
            self._store.save_snapshot("order_status", status_payload)
            self._store.purge_wal()
        except Exception:  # pragma: no cover - defensive persistence guard
            LOGGER.exception("data_hub_checkpoint_failed")
            raise

    def _persist_wal(self, kind: str, key: str, payload: Mapping[str, Any]) -> None:
        if self._store is None:
            return
        try:
            self._store.append_wal(kind, key, dict(payload))
        except Exception:  # pragma: no cover - defensive persistence guard
            LOGGER.exception(
                "data_hub_wal_append_failed",
                extra={"event": "data_hub_wal_append_failed", "kind": kind, "key": key},
            )

    def _maybe_checkpoint(self) -> None:
        if self._store is None:
            return
        now = self._clock()
        if now - self._last_snapshot_ts < self._checkpoint_interval:
            return
        try:
            self.checkpoint()
        except Exception:
            return
        self._last_snapshot_ts = now

    def _restore_from_store(self) -> None:
        if self._store is None:
            return
        try:
            quotes_snapshot = self._store.load_snapshot("quotes") or {}
            positions_snapshot = self._store.load_snapshot("positions") or {}
            orders_snapshot = self._store.load_snapshot("orders") or {}
            sequences_snapshot = self._store.load_snapshot("order_sequences") or {}
            status_snapshot = self._store.load_snapshot("order_status") or {}
            wal_entries = self._store.load_wal()
        except Exception:  # pragma: no cover - defensive persistence guard
            LOGGER.exception("data_hub_restore_failed")
            return

        with self._lock:
            self._quotes = {
                symbol: dict(payload)
                for symbol, payload in quotes_snapshot.items()
                if isinstance(payload, Mapping)
            }
            self._rebuild_option_caches_unlocked()
            self._positions = {
                symbol: dict(payload)
                for symbol, payload in positions_snapshot.items()
                if isinstance(payload, Mapping)
            }
            self._orders = {
                order_id: dict(payload)
                for order_id, payload in orders_snapshot.items()
                if isinstance(payload, Mapping)
            }
            self._order_sequences = {
                str(order_id): int(float(value))
                for order_id, value in sequences_snapshot.items()
                if isinstance(value, (int, float, str)) and str(value).strip() != ""
            }
            self._order_status = {
                str(order_id): str(status)
                for order_id, status in status_snapshot.items()
            }
            for entry in wal_entries:
                self._apply_wal_entry(entry)
        self._last_snapshot_ts = self._clock()

    def _apply_wal_entry(self, entry: WalEntry) -> None:
        if not isinstance(entry, Mapping):  # pragma: no cover - defensive
            return
        kind = str(entry.get("kind", ""))
        key = str(entry.get("key", ""))
        payload = entry.get("payload")
        if kind == "quote" and isinstance(payload, Mapping):
            symbol = self.normalize(key or payload.get("symbol"))
            if symbol:
                self._quotes[symbol] = dict(payload)
                self._capture_option_metrics(symbol, dict(payload))
        elif kind == "positions" and isinstance(payload, Mapping):
            self._positions = {
                symbol: dict(row)
                for symbol, row in payload.items()
                if isinstance(row, Mapping)
            }
        elif kind == "order" and isinstance(payload, Mapping):
            order_id_raw = payload.get("order_id") or key
            if not order_id_raw:
                return
            order_id = str(order_id_raw)
            row = dict(payload)
            status = str(row.get("status") or "").lower()
            seq_value = row.get("status_sequence")
            sequence = None
            if isinstance(seq_value, (int, float, str)):
                try:
                    sequence = int(float(seq_value))
                except Exception:  # pragma: no cover - defensive
                    sequence = None
            self._orders[order_id] = row
            if sequence is not None:
                self._order_sequences[order_id] = sequence
            elif order_id not in self._order_sequences:
                self._order_sequences[order_id] = 0
            self._order_status[order_id] = status

    def as_dict(self) -> dict[str, Any]:
        """Return serialisable snapshot of hub state."""

        with self._lock:
            return {
                "quotes": {symbol: dict(tick) for symbol, tick in self._quotes.items()},
                "positions": [dict(value) for value in self._positions.values()],
                "orders": [dict(value) for value in self._orders.values()],
            }

    # ------------------------------------------------------------------
    def _is_valid_transition(self, previous: str, new: str) -> bool:
        prev = (previous or "").lower()
        nxt = (new or "").lower()
        if prev == nxt:
            return True
        if nxt not in _ORDER_STATE_MACHINE:
            return True
        if prev not in _ORDER_STATE_MACHINE:
            return True
        allowed = _ORDER_STATE_MACHINE.get(prev, set())
        return nxt in allowed

    def _sequence(self, payload: Mapping[str, Any]) -> Optional[int]:
        candidate_keys = (
            "status_sequence",
            "status_seq",
            "status_sequence_no",
            "sequence",
            "seq",
            "version",
        )
        for key in candidate_keys:
            raw = payload.get(key)
            if raw is None:
                continue
            if isinstance(raw, str) and not raw.strip():
                continue
            try:
                numeric = cast(float | int | str, raw)
                value = int(float(numeric))
            except Exception:
                continue
            if value < 0:
                continue
            return value
        return None

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _sanitize_tick(self, payload: dict[str, Any]) -> dict[str, Any]:
        numeric_fields = (
            "ltp",
            "last_price",
            "close",
            "open",
            "high",
            "low",
            "bid",
            "ask",
            "volume",
        )
        for key in numeric_fields:
            if key in payload:
                coerced = self._float(payload.get(key))
                if coerced is not None:
                    payload[key] = coerced
        return payload

    @staticmethod
    def _int(value: Any) -> int:
        if value in (None, ""):
            return 0
        try:
            return int(float(value))
        except Exception:
            return 0

    @staticmethod
    def _ts(value: Any) -> float:
        if value in (None, ""):
            return time.time()
        try:
            ts = float(value)
        except Exception:
            return time.time()
        if ts > 1_000_000_000_000:
            ts /= 1000.0
        return ts

    def _capture_option_metrics(self, symbol: str, tick: Mapping[str, Any]) -> None:
        if not symbol.endswith(("CE", "PE")):
            return
        iv_candidates = (
            "implied_volatility",
            "iv",
            "option_iv",
        )
        iv_value: float | None = None
        for key in iv_candidates:
            iv_value = self._float(tick.get(key))
            if iv_value is not None:
                if iv_value > 1.5:
                    iv_value /= 100.0
                break
        oi_value = tick.get("open_interest") or tick.get("oi")
        greeks_payload = tick.get("greeks")
        greeks_row: dict[str, float] | None = None
        if isinstance(greeks_payload, Mapping):
            greeks_row = {
                key: float(value)
                for key, value in greeks_payload.items()
                if isinstance(value, (int, float))
            }
        with self._lock:
            if iv_value is not None:
                self._iv_cache[symbol] = iv_value
            if isinstance(oi_value, (int, float)):
                self._oi_cache[symbol] = int(oi_value)
            if greeks_row:
                self._greeks_cache[symbol] = greeks_row

    def _compute_implied_volatility(
        self, symbol: str, quote: Mapping[str, Any]
    ) -> Optional[float]:
        if not symbol.endswith(("CE", "PE")):
            return None
        option_price = self._float(
            quote.get("ltp")
            or quote.get("last_price")
            or quote.get("close")
            or quote.get("bid")
            or quote.get("ask")
        )
        if option_price is None or option_price <= 0.0:
            return None
        underlying = self.get_quote(self._spot_symbol, allow_pull=False) or {}
        spot = self._float(
            underlying.get("ltp")
            or underlying.get("last_price")
            or underlying.get("close")
        )
        if spot is None or spot <= 0.0:
            return None
        parsed = self._parse_option_symbol(symbol)
        if parsed is None:
            return None
        _, expiry_ts, strike, is_call = parsed
        ttm = max(expiry_ts - self._clock(), 0.0) / 31557600.0
        if ttm <= 0.0:
            ttm = 1.0 / 252.0
        try:
            iv = implied_volatility(
                spot,
                strike,
                ttm,
                self._risk_free_rate,
                option_price,
                is_call=is_call,
            )
        except ValueError:
            return None
        return iv

    def _compute_greeks(
        self, symbol: str, quote: Mapping[str, Any]
    ) -> Optional[dict[str, float]]:
        iv = self.get_iv(symbol)
        if iv is None:
            return None
        underlying = self.get_quote(self._spot_symbol, allow_pull=False) or {}
        spot = self._float(
            underlying.get("ltp")
            or underlying.get("last_price")
            or underlying.get("close")
        )
        if spot is None or spot <= 0.0:
            return None
        parsed = self._parse_option_symbol(symbol)
        if parsed is None:
            return None
        _, expiry_ts, strike, is_call = parsed
        ttm = max(expiry_ts - self._clock(), 0.0) / 31557600.0
        if ttm <= 0.0:
            ttm = 1.0 / 252.0
        greeks = black_scholes_greeks(
            spot,
            strike,
            ttm,
            self._risk_free_rate,
            iv,
            is_call=is_call,
        )
        return greeks

    def _parse_option_symbol(
        self, symbol: str
    ) -> Optional[tuple[str, float, float, bool]]:
        pattern = re.compile(
            r"^(?P<base>[A-Z]+)(?P<year>\d{2})(?P<month>[A-Z]{3})(?P<strike>\d+)(?P<opt>CE|PE)$"
        )
        match = pattern.match(symbol)
        if not match:
            return None
        base = match.group("base")
        year = int(match.group("year"))
        month_str = match.group("month")
        strike = float(match.group("strike"))
        opt = match.group("opt")
        months = {
            "JAN": 1,
            "FEB": 2,
            "MAR": 3,
            "APR": 4,
            "MAY": 5,
            "JUN": 6,
            "JUL": 7,
            "AUG": 8,
            "SEP": 9,
            "OCT": 10,
            "NOV": 11,
            "DEC": 12,
        }
        month = months.get(month_str)
        if month is None:
            return None
        year += 2000 if year < 100 else 0
        expiry = datetime(year, month, 28, 15, 30)
        expiry_ts = expiry.timestamp()
        return base, expiry_ts, strike, opt == "CE"

    def _rebuild_option_caches_unlocked(self) -> None:
        self._iv_cache.clear()
        self._oi_cache.clear()
        self._greeks_cache.clear()
        for symbol, tick in self._quotes.items():
            self._capture_option_metrics(symbol, tick)

    def _atm_iv(self, spot: float) -> Optional[float]:
        best_diff = float("inf")
        best_iv: Optional[float] = None
        with self._lock:
            symbols = list(self._quotes.keys())
        now = self._clock()
        for symbol in symbols:
            if not symbol.endswith(("CE", "PE")):
                continue
            parsed = self._parse_option_symbol(symbol)
            if parsed is None:
                continue
            _base, expiry, strike, _ = parsed
            if expiry <= now:
                continue
            iv = self.get_iv(symbol, allow_fallback=True)
            if iv is None:
                continue
            diff = abs(strike - spot)
            if diff < best_diff:
                best_diff = diff
                best_iv = iv
        return best_iv


__all__ = ["DataHub", "OrderListener", "TickListener", "Tick"]
