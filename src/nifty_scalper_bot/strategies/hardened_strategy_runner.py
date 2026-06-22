"""Runtime resilience extensions for StrategyRunner.

The class keeps the established execution path and adds two narrow protections:
exact entry-window reason propagation and a bounded circuit for repeated,
identical candidate-selection programming failures.
"""

from __future__ import annotations

from contextlib import suppress
import logging
import os
import threading
import time
from typing import Any, Mapping

from nifty_scalper_bot.strategies import runner as _legacy


_LegacyStrategyRunner = _legacy.StrategyRunner


class HardenedStrategyRunner(_LegacyStrategyRunner):
    """Strategy runner with deterministic candidate-exception containment."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._initialize_candidate_exception_circuit()

    def _initialize_candidate_exception_circuit(self) -> None:
        if not hasattr(self, "_candidate_exception_circuit_lock"):
            self._candidate_exception_circuit_lock = threading.RLock()
        if not hasattr(self, "_candidate_exception_circuit"):
            self._candidate_exception_circuit: dict[str, Any] = {
                "fingerprint": None,
                "count": 0,
                "first_seen": 0.0,
                "last_seen": 0.0,
                "open_until": 0.0,
                "opened_at": 0.0,
                "last_trace_id": None,
                "last_symbol": None,
            }

    @staticmethod
    def _candidate_exception_policy() -> tuple[int, float, float]:
        def _number(name: str, default: float, minimum: float) -> float:
            try:
                parsed = float(os.getenv(name, str(default)) or default)
            except (TypeError, ValueError):
                parsed = default
            return max(parsed, minimum)

        threshold = int(
            _number("CANDIDATE_EXCEPTION_CIRCUIT_THRESHOLD", 3.0, 2.0)
        )
        window_seconds = _number(
            "CANDIDATE_EXCEPTION_CIRCUIT_WINDOW_SECONDS", 30.0, 1.0
        )
        cooldown_seconds = _number(
            "CANDIDATE_EXCEPTION_CIRCUIT_COOLDOWN_SECONDS", 300.0, 5.0
        )
        return threshold, window_seconds, cooldown_seconds

    @staticmethod
    def _candidate_error_fingerprint(details: Mapping[str, Any] | None) -> str:
        payload = dict(details or {})
        error_type = str(payload.get("error_type") or "Exception").strip()
        error = " ".join(str(payload.get("error") or "unknown").split())
        return f"{error_type}:{error}"[:500]

    def candidate_selection_circuit_snapshot(self) -> dict[str, Any]:
        """Return a read-only diagnostic snapshot for health/status surfaces."""
        self._initialize_candidate_exception_circuit()
        now = time.monotonic()
        with self._candidate_exception_circuit_lock:
            state = dict(self._candidate_exception_circuit)
        state["open"] = bool(float(state.get("open_until") or 0.0) > now)
        state["remaining_seconds"] = max(
            0.0, float(state.get("open_until") or 0.0) - now
        )
        return state

    def _candidate_circuit_open_result(
        self,
        *,
        symbol: str,
        trace_id: str | None,
    ) -> Any | None:
        self._initialize_candidate_exception_circuit()
        now = time.monotonic()
        with self._candidate_exception_circuit_lock:
            state = self._candidate_exception_circuit
            open_until = float(state.get("open_until") or 0.0)
            if open_until <= 0.0:
                return None
            if now >= open_until:
                previous = dict(state)
                state.update(
                    {
                        "fingerprint": None,
                        "count": 0,
                        "first_seen": 0.0,
                        "last_seen": 0.0,
                        "open_until": 0.0,
                        "opened_at": 0.0,
                        "last_trace_id": None,
                        "last_symbol": None,
                    }
                )
                logger = getattr(self, "_logger", _legacy.LOGGER)
                logger.warning(
                    "CANDIDATE_EXCEPTION_CIRCUIT_RECOVERED previous_fingerprint=%s",
                    previous.get("fingerprint"),
                    extra={
                        "event": "CANDIDATE_EXCEPTION_CIRCUIT_RECOVERED",
                        "previous_fingerprint": previous.get("fingerprint"),
                    },
                )
                return None
            details = {
                "fingerprint": state.get("fingerprint"),
                "exception_count": int(state.get("count") or 0),
                "opened_at_monotonic": state.get("opened_at"),
                "remaining_seconds": max(0.0, open_until - now),
                "last_trace_id": state.get("last_trace_id"),
                "last_symbol": state.get("last_symbol"),
            }

        with suppress(Exception):
            self._reset_execution_state(symbol)
        return self._reject_signal_execution(
            symbol=symbol,
            trace_id=trace_id,
            reason="candidate_selection_circuit_open",
            details=details,
        )

    def _record_candidate_selection_exception(
        self,
        *,
        symbol: str,
        trace_id: str | None,
        details: Mapping[str, Any] | None,
    ) -> bool:
        """Record one programming failure; return True when the circuit opens."""
        self._initialize_candidate_exception_circuit()
        threshold, window_seconds, cooldown_seconds = (
            self._candidate_exception_policy()
        )
        now = time.monotonic()
        fingerprint = self._candidate_error_fingerprint(details)
        opened = False
        with self._candidate_exception_circuit_lock:
            state = self._candidate_exception_circuit
            same_failure = state.get("fingerprint") == fingerprint
            within_window = (
                now - float(state.get("first_seen") or 0.0)
            ) <= window_seconds
            if not same_failure or not within_window:
                state.update(
                    {
                        "fingerprint": fingerprint,
                        "count": 1,
                        "first_seen": now,
                        "last_seen": now,
                        "open_until": 0.0,
                        "opened_at": 0.0,
                    }
                )
            else:
                state["count"] = int(state.get("count") or 0) + 1
                state["last_seen"] = now
            state["last_trace_id"] = trace_id
            state["last_symbol"] = symbol
            if (
                int(state.get("count") or 0) >= threshold
                and float(state.get("open_until") or 0.0) <= now
            ):
                state["opened_at"] = now
                state["open_until"] = now + cooldown_seconds
                opened = True
                snapshot = dict(state)
            else:
                snapshot = dict(state)

        if opened:
            logger = getattr(self, "_logger", _legacy.LOGGER)
            logger.critical(
                "CANDIDATE_EXCEPTION_CIRCUIT_OPEN symbol=%s count=%s cooldown_s=%s fingerprint=%s trace_id=%s",
                symbol,
                snapshot.get("count"),
                cooldown_seconds,
                fingerprint,
                trace_id,
                extra={
                    "event": "CANDIDATE_EXCEPTION_CIRCUIT_OPEN",
                    "symbol": symbol,
                    "trace_id": trace_id,
                    "exception_count": snapshot.get("count"),
                    "cooldown_seconds": cooldown_seconds,
                    "fingerprint": fingerprint,
                },
            )
        return opened

    def _reject_signal_execution(self, *args: Any, **kwargs: Any) -> Any:
        """Preserve the exact gate reason when candidate selection is time-blocked."""
        reason = kwargs.get("reason")
        positional = list(args)
        if reason is None and len(positional) >= 3:
            reason = positional[2]
        if reason == "no_execution_ready_candidate":
            selector = getattr(self, "_trade_candidate_selector", None)
            gate_reason = getattr(selector, "last_entry_window_reason", None)
            if gate_reason is None:
                gate_reason = getattr(selector, "_last_entry_window_reason", None)
            if gate_reason:
                details = dict(kwargs.get("details") or {})
                details.update(
                    {
                        "original_reason": "no_execution_ready_candidate",
                        "stage": "entry_window_gate",
                        "entry_window_reason": str(gate_reason),
                    }
                )
                kwargs["details"] = details
                kwargs["reason"] = str(gate_reason)
                if len(positional) >= 3:
                    positional[2] = str(gate_reason)
        return super()._reject_signal_execution(*positional, **kwargs)

    def _handle_entry_signal_inner(
        self,
        signal: Any,
        base_symbol: str,
        trade_symbol: str,
        trade_price: float,
        timestamp: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        trace_id = kwargs.get("trace_id")
        if trace_id is None and args:
            trace_id = args[0]
        blocked = self._candidate_circuit_open_result(
            symbol=base_symbol,
            trace_id=trace_id,
        )
        if blocked is not None:
            return blocked

        result = super()._handle_entry_signal_inner(
            signal,
            base_symbol,
            trade_symbol,
            trade_price,
            timestamp,
            *args,
            **kwargs,
        )
        if getattr(result, "reason", None) != "candidate_selection_exception":
            return result

        details = dict(getattr(result, "details", {}) or {})
        opened = self._record_candidate_selection_exception(
            symbol=base_symbol,
            trace_id=trace_id,
            details=details,
        )
        if not opened:
            return result

        snapshot = self.candidate_selection_circuit_snapshot()
        return _legacy.SignalExecutionResult(
            False,
            "candidate_selection_circuit_open",
            details={
                **details,
                "fingerprint": snapshot.get("fingerprint"),
                "exception_count": snapshot.get("count"),
                "remaining_seconds": snapshot.get("remaining_seconds"),
                "trace_id": trace_id,
            },
        )


__all__ = ["HardenedStrategyRunner"]
