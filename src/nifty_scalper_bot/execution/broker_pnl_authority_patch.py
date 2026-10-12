"""Broker-account P&L diagnostic helpers for native PositionManager.

Zerodha ``data.net`` position rows are authoritative for quantity/exposure, but
their ``realised`` field is legacy and must not be used as the live account P&L
authority.  Native diagnostics retain position reconciliation while sourcing P&L
evidence from Zerodha trades/positions, with margins M2M retained as diagnostic
fallback rather than strategy realised-P&L authority.

The bot's local fill ledger remains the strategy P&L authority. Broker/account
differences are observable diagnostics only; they never become entry/readiness
blockers.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from nifty_scalper_bot.data.rest.broker_pnl_snapshot import (
    _extract_account_m2m,
    _finite_float,
    _strategy_day_marked_pnl,
    _strategy_tradebook_realized_pnl,
    _zerodha_get_pnl_snapshot,
)
from nifty_scalper_bot.utils.market_hours import (
    get_runtime_market_mode,
    post_market_broker_refresh_seconds,
    post_market_quiet_mode_enabled,
)

_DEFAULT_REFRESH_SECONDS = 15.0
_DEFAULT_MAX_AGE_SECONDS = 120.0
_MATCH_TOLERANCE_RUPEES = 1.0


def _refresh_seconds() -> float:
    raw = os.getenv("BROKER_PNL_REFRESH_SECONDS", str(_DEFAULT_REFRESH_SECONDS))
    try:
        return max(2.0, min(float(raw), 300.0))
    except (TypeError, ValueError):
        return _DEFAULT_REFRESH_SECONDS


def _effective_refresh_seconds() -> float:
    base = _refresh_seconds()
    if not post_market_quiet_mode_enabled():
        return base
    try:
        mode = get_runtime_market_mode()
    except Exception:
        return base
    if mode not in {"POST_MARKET", "HOLIDAY"}:
        return base
    return max(base, post_market_broker_refresh_seconds())


def _max_age_seconds() -> float:
    raw = os.getenv("BROKER_PNL_MAX_AGE_SECONDS", str(_DEFAULT_MAX_AGE_SECONDS))
    try:
        return max(5.0, min(float(raw), 900.0))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_AGE_SECONDS


def _materialize_positions(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return payload
    try:
        return list(payload)
    except TypeError:
        return payload


def _strip_row_legacy_realized(row: object) -> object:
    if not isinstance(row, Mapping):
        return row
    clean = dict(row)
    clean.pop("realised", None)
    clean.pop("realized", None)
    return clean


def _strip_legacy_position_pnl(payload: Any) -> Any:
    """Remove only legacy realised fields while preserving exposure payload shape."""

    materialized = _materialize_positions(payload)
    if isinstance(materialized, Mapping):
        clean = dict(materialized)
        if (
            "net" in clean
            and isinstance(clean.get("net"), Sequence)
            and not isinstance(clean.get("net"), (str, bytes))
        ):
            clean["net"] = [
                _strip_row_legacy_realized(row) for row in clean.get("net", [])
            ]
        elif (
            "positions" in clean
            and isinstance(clean.get("positions"), Sequence)
            and not isinstance(clean.get("positions"), (str, bytes))
        ):
            clean["positions"] = [
                _strip_row_legacy_realized(row)
                for row in clean.get("positions", [])
            ]
        else:
            clean = dict(_strip_row_legacy_realized(clean))
        return clean
    if isinstance(materialized, Sequence) and not isinstance(
        materialized, (str, bytes)
    ):
        return [_strip_row_legacy_realized(row) for row in materialized]
    return materialized


def _broker_supports_dedicated_pnl(owner: Any) -> bool:
    broker = getattr(owner, "_broker_client", None)
    return callable(getattr(broker, "get_pnl_snapshot", None))


def _should_emit_pnl_log(self: Any, fingerprint: object, now_mono: float) -> bool:
    """Throttle repetitive diagnostic logs while always surfacing state changes."""

    with getattr(self, "_lock"):
        previous = getattr(self, "_broker_pnl_last_log_fingerprint", None)
        last_log = float(getattr(self, "_broker_pnl_last_log_mono", 0.0) or 0.0)
        should_log = previous != fingerprint or now_mono - last_log >= 60.0
        if should_log:
            self._broker_pnl_last_log_fingerprint = fingerprint
            self._broker_pnl_last_log_mono = now_mono
        return should_log


def refresh_broker_pnl_diagnostic(
    self: Any,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Refresh account P&L authority; mismatches remain diagnostics only."""

    now_mono = time.monotonic()
    with getattr(self, "_lock"):
        cached = dict(getattr(self, "_broker_account_pnl_snapshot", {}) or {})
        last_fetch = float(getattr(self, "_broker_pnl_last_fetch_mono", 0.0) or 0.0)
    if (
        not force
        and cached
        and last_fetch > 0.0
        and now_mono - last_fetch < _effective_refresh_seconds()
    ):
        return cached

    broker = getattr(self, "_broker_client", None)
    fetcher = getattr(broker, "get_pnl_snapshot", None)
    if not callable(fetcher):
        return cached

    try:
        raw = fetcher()
        if not isinstance(raw, Mapping):
            raise RuntimeError("broker P&L snapshot is not a mapping")
        margin_realized = _finite_float(raw.get("account_realized"))
        unrealized = _finite_float(raw.get("account_unrealized"))
    except Exception as exc:  # P&L evidence is diagnostic, never an entry blocker
        error = f"{type(exc).__name__}: {exc}"
        with getattr(self, "_lock"):
            self._broker_pnl_fetch_error = error
            self._broker_pnl_last_fetch_mono = now_mono
        logger = getattr(self, "_logger", None)
        warning = getattr(logger, "warning", None)
        if callable(warning) and _should_emit_pnl_log(
            self, ("unavailable", error), now_mono
        ):
            warning(
                "PNL_BROKER_DIAGNOSTIC_UNAVAILABLE error=%s",
                error,
                extra={
                    "event": "PNL_BROKER_DIAGNOSTIC_UNAVAILABLE",
                    "error": error,
                    "diagnostic_only": True,
                },
            )
        return cached

    persist_baseline = False
    with getattr(self, "_lock"):
        strategy_realized = float(getattr(self, "_local_realized_pnl", 0.0) or 0.0)
        tradebook_realized = _finite_float(raw.get("strategy_tradebook_realized_gross"))
        tradebook_fills = int(
            _finite_float(raw.get("strategy_tradebook_fill_count")) or 0
        )
        positions_closed = _finite_float(raw.get("strategy_day_closed_gross"))
        positions_marked = _finite_float(raw.get("strategy_day_marked_gross"))
        positions_flat = (
            (_finite_float(raw.get("strategy_day_rows")) or 0) > 0
            and positions_closed is not None
            and positions_marked is not None
            and abs(positions_marked - positions_closed) <= _MATCH_TOLERANCE_RUPEES
        )
        broker_evidence: float | None
        evidence_source: str
        if tradebook_realized is not None and tradebook_fills > 0:
            broker_evidence = tradebook_realized
            evidence_source = "zerodha_trades"
        elif positions_flat:
            broker_evidence = positions_closed
            evidence_source = "zerodha_positions_day"
        else:
            broker_evidence = margin_realized
            evidence_source = "zerodha_margins_m2m"

        difference = (
            None
            if broker_evidence is None
            else float(broker_evidence) - strategy_realized
        )
        if difference is None:
            status = "unavailable"
        elif (
            evidence_source == "zerodha_margins_m2m"
            and abs(difference) <= _MATCH_TOLERANCE_RUPEES
        ):
            # Account M2M is not independent evidence of strategy fills.
            status = "unverified"
        elif abs(difference) <= _MATCH_TOLERANCE_RUPEES:
            status = "matched"
        else:
            status = "mismatch"
        positions_difference = (
            None if positions_closed is None else positions_closed - strategy_realized
        )
        margin_difference = (
            None
            if margin_realized is None
            else float(margin_realized) - strategy_realized
        )
        snapshot = dict(raw)
        snapshot.update(
            {
                "account_realized": margin_realized,
                "account_unrealized": (
                    None if unrealized is None else float(unrealized)
                ),
                "margin_m2m_realized": margin_realized,
                "margin_m2m_unrealized": (
                    None if unrealized is None else float(unrealized)
                ),
                "broker_realized_evidence": broker_evidence,
                "broker_realized_evidence_source": evidence_source,
                "strategy_realized": strategy_realized,
                "difference": difference,
                "margin_vs_strategy_difference": margin_difference,
                "positions_vs_strategy_difference": positions_difference,
                "status": status,
                "diagnostic_only": True,
                "fetched_monotonic": now_mono,
            }
        )
        self._broker_account_pnl_snapshot = snapshot
        self._broker_pnl_last_fetch_mono = now_mono
        self._broker_pnl_fetch_error = None

        today = self._trading_date_ist()
        if (
            getattr(self, "_pnl_trading_date", None) != today
            or getattr(self, "_session_opening_realized_baseline", None) != 0.0
            or getattr(self, "_baseline_source", None) != "zerodha_margins_m2m"
        ):
            self._pnl_trading_date = today
            self._session_opening_realized_baseline = 0.0
            self._baseline_established_at = datetime.now(timezone.utc)
            self._baseline_source = "zerodha_margins_m2m"
            self._pnl_product_scope = "ACCOUNT_EQUITY_FNO"
            persist_baseline = True

        # Keep the strategy ledger as PositionManager's accounting authority.
        # Broker/account P&L is exposed separately to RiskManager.
        self._broker_realized_pnl = None
        self._refresh_realized_pnl_locked()
        self._pnl_authority = "local_confirmed_ledger"
        self._pnl_reconciliation_status = f"broker_account_diagnostic_{status}"

    logger = getattr(self, "_logger", None)
    log = (
        getattr(logger, "warning", None)
        if status == "mismatch"
        else getattr(logger, "info", None)
    )
    log_fingerprint = (
        "available",
        status,
        None if broker_evidence is None else round(float(broker_evidence), 2),
        round(float(strategy_realized), 2),
    )
    if callable(log) and _should_emit_pnl_log(self, log_fingerprint, now_mono):
        log(
            "PNL_BROKER_DIAGNOSTIC broker_evidence=%s strategy_realized=%.2f "
            "difference=%s status=%s source=%s margin_m2m_realized=%s "
            "positions_closed=%s",
            broker_evidence,
            strategy_realized,
            difference,
            status,
            evidence_source,
            margin_realized,
            positions_closed,
            extra={
                "event": "PNL_BROKER_DIAGNOSTIC",
                "broker_realized_evidence": broker_evidence,
                "strategy_realized": strategy_realized,
                "difference": difference,
                "margin_m2m_realized": margin_realized,
                "margin_vs_strategy_difference": margin_difference,
                "positions_vs_strategy_difference": positions_difference,
                "status": status,
                "source": evidence_source,
                "diagnostic_only": True,
            },
        )

    if persist_baseline:
        try:
            self.save_state()
        except Exception:
            warning = getattr(logger, "warning", None)
            if callable(warning):
                warning("PNL_BASELINE_PERSIST_FAILED", exc_info=True)
    return snapshot


def enrich_pnl_reconciliation_snapshot(
    self: Any, original: Mapping[str, object]
) -> dict[str, object]:
    """Add broker-account diagnostics to the native local-ledger snapshot."""
    base = dict(original)
    diagnostic = self.refresh_broker_pnl_diagnostic()
    with getattr(self, "_lock"):
        error = getattr(self, "_broker_pnl_fetch_error", None)
    base.update(
        {
            "strategy_realized": float(
                diagnostic.get(
                    "strategy_realized",
                    getattr(self, "_local_realized_pnl", 0.0) or 0.0,
                )
            ),
            "broker_account_realized": diagnostic.get("account_realized"),
            "broker_account_unrealized": diagnostic.get("account_unrealized"),
            "broker_account_total": diagnostic.get("account_total"),
            "broker_margin_m2m_realized": diagnostic.get("margin_m2m_realized"),
            "broker_margin_m2m_unrealized": diagnostic.get("margin_m2m_unrealized"),
            "broker_realized_evidence": diagnostic.get("broker_realized_evidence"),
            "broker_realized_evidence_source": diagnostic.get(
                "broker_realized_evidence_source"
            ),
            "broker_tradebook_realized_gross": diagnostic.get(
                "strategy_tradebook_realized_gross"
            ),
            "broker_tradebook_fill_count": diagnostic.get(
                "strategy_tradebook_fill_count", 0
            ),
            "broker_strategy_day_marked_gross": diagnostic.get(
                "strategy_day_marked_gross"
            ),
            "broker_strategy_day_closed_gross": diagnostic.get(
                "strategy_day_closed_gross"
            ),
            "broker_strategy_day_rows": diagnostic.get("strategy_day_rows", 0),
            "broker_vs_strategy_realized_difference": diagnostic.get("difference"),
            "broker_positions_vs_strategy_difference": diagnostic.get(
                "positions_vs_strategy_difference"
            ),
            "broker_account_pnl_status": diagnostic.get("status", "unavailable"),
            "broker_account_pnl_source": diagnostic.get("source"),
            "broker_account_pnl_observed_at": diagnostic.get("observed_at"),
            "broker_account_pnl_error": error,
            "pnl_diagnostic_only": True,
        }
    )
    return base


__all__ = [
    "_extract_account_m2m",
    "_strategy_day_marked_pnl",
    "_strategy_tradebook_realized_pnl",
    "_strip_legacy_position_pnl",
    "_zerodha_get_pnl_snapshot",
    "enrich_pnl_reconciliation_snapshot",
    "refresh_broker_pnl_diagnostic",
]
