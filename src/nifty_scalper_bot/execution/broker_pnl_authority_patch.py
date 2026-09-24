"""Separate broker exposure reconciliation from broker P&L authority.

Zerodha ``data.net`` position rows are authoritative for quantity/exposure, but
their ``realised`` field is legacy and must not be used as the live account P&L
authority.  This patch keeps position reconciliation unchanged while sourcing
account-risk P&L from Zerodha margins (``m2m_realised``/``m2m_unrealised``).

The bot's local fill ledger remains the strategy P&L authority. Broker/account
differences are observable diagnostics only; they never become entry/readiness
blockers.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from nifty_scalper_bot.utils.symbols import is_strategy_instrument

_PATCH_APPLIED = False
_ORIGINAL_POSITION_INIT: Any = None
_ORIGINAL_SYNCHRONIZE_WITH_BROKER: Any = None
_ORIGINAL_PNL_RECONCILIATION_SNAPSHOT: Any = None

_DEFAULT_REFRESH_SECONDS = 15.0
_DEFAULT_MAX_AGE_SECONDS = 120.0
_MATCH_TOLERANCE_RUPEES = 1.0


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _refresh_seconds() -> float:
    raw = os.getenv("BROKER_PNL_REFRESH_SECONDS", str(_DEFAULT_REFRESH_SECONDS))
    try:
        return max(2.0, min(float(raw), 300.0))
    except (TypeError, ValueError):
        return _DEFAULT_REFRESH_SECONDS


def _max_age_seconds() -> float:
    raw = os.getenv("BROKER_PNL_MAX_AGE_SECONDS", str(_DEFAULT_MAX_AGE_SECONDS))
    try:
        return max(5.0, min(float(raw), 900.0))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_AGE_SECONDS


def _extract_account_m2m(payload: object) -> tuple[float | None, float | None]:
    """Extract Zerodha account M2M from a raw margin payload."""

    if not isinstance(payload, Mapping):
        return None, None
    utilised = payload.get("utilised")
    if not isinstance(utilised, Mapping):
        utilised = payload.get("utilized")
    if not isinstance(utilised, Mapping):
        utilised = payload

    realized = None
    unrealized = None
    for key in ("m2m_realised", "m2m_realized"):
        if key in utilised:
            realized = _finite_float(utilised.get(key))
            break
    for key in ("m2m_unrealised", "m2m_unrealized"):
        if key in utilised:
            unrealized = _finite_float(utilised.get(key))
            break
    return realized, unrealized


def _strategy_symbol(record: Mapping[str, Any]) -> str:
    raw_symbol = (
        record.get("tradingsymbol")
        or record.get("symbol")
        or record.get("instrument")
        or ""
    )
    text = str(raw_symbol).strip().upper()
    exchange = str(record.get("exchange") or "").strip().upper()
    if text and ":" not in text and exchange:
        return f"{exchange}:{text}"
    return text


def _strategy_day_marked_pnl(
    rows: object,
) -> tuple[float | None, float | None, int]:
    """Calculate marked and closed gross P&L from Zerodha ``data.day`` rows.

    Formula follows Zerodha's position economics:
    sell_value - buy_value + net_quantity * last_price * multiplier.
    This is diagnostic account/broker evidence only and never overwrites the
    bot-owned strategy fill ledger.
    """

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None, None, 0
    marked_total = 0.0
    closed_total = 0.0
    seen = 0
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        symbol = _strategy_symbol(item)
        if not symbol or not is_strategy_instrument(symbol):
            continue
        if str(item.get("product") or "").strip().upper() != "MIS":
            continue
        buy_value = _finite_float(item.get("buy_value"))
        sell_value = _finite_float(item.get("sell_value"))
        quantity = _finite_float(
            item.get("quantity", item.get("net_quantity", item.get("net_qty")))
        )
        last_price = _finite_float(item.get("last_price", item.get("ltp")))
        multiplier = _finite_float(item.get("multiplier"))
        if buy_value is None or sell_value is None or quantity is None:
            continue
        if multiplier is None:
            multiplier = 1.0
        if last_price is None:
            if abs(quantity) > 1e-9:
                continue
            last_price = 0.0
        marked_total += (
            sell_value - buy_value + quantity * last_price * multiplier
        )
        if abs(quantity) <= 1e-9:
            closed_total += sell_value - buy_value
        seen += 1
    if not seen:
        return None, None, 0
    return marked_total, closed_total, seen


def _strategy_tradebook_realized_pnl(
    rows: object,
) -> tuple[float | None, int]:
    """Calculate realized gross P&L from current-day Zerodha trade fills."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None, 0

    inventory: dict[str, list[list[float]]] = {}
    realized = 0.0
    seen = 0
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        symbol = _strategy_symbol(item)
        if not symbol or not is_strategy_instrument(symbol):
            continue
        if str(item.get("product") or "").strip().upper() != "MIS":
            continue
        side = str(
            item.get("transaction_type") or item.get("side") or ""
        ).strip().upper()
        quantity = _finite_float(
            item.get("quantity", item.get("filled_quantity"))
        )
        price = _finite_float(item.get("average_price", item.get("price")))
        if side not in {"BUY", "SELL"} or quantity is None or price is None:
            continue
        if quantity <= 0 or price <= 0:
            continue

        signed = float(quantity) if side == "BUY" else -float(quantity)
        remaining = abs(signed)
        lots = inventory.setdefault(symbol, [])
        while remaining > 1e-9 and lots and lots[0][0] * signed < 0:
            lot_qty, lot_price = lots[0]
            matched = min(abs(lot_qty), remaining)
            if signed < 0:
                realized += (float(price) - lot_price) * matched
            else:
                realized += (lot_price - float(price)) * matched
            remaining -= matched
            lot_remaining = abs(lot_qty) - matched
            if lot_remaining <= 1e-9:
                lots.pop(0)
            else:
                lots[0][0] = (
                    lot_remaining if lot_qty > 0 else -lot_remaining
                )

        if remaining > 1e-9:
            lots.append(
                [remaining if signed > 0 else -remaining, float(price)]
            )
        seen += 1

    if not seen:
        return None, 0
    return realized, seen


def _zerodha_get_pnl_snapshot(self: Any) -> dict[str, Any]:
    """Return dedicated broker P&L evidence without changing exposure semantics."""

    margins_fetcher = getattr(self, "get_account_margins", None)
    if not callable(margins_fetcher):
        raise RuntimeError("broker account margins endpoint unavailable")

    margins = margins_fetcher(segment="equity")
    realized, unrealized = _extract_account_m2m(margins)

    day_marked: float | None = None
    day_closed: float | None = None
    day_rows = 0
    tradebook_realized: float | None = None
    tradebook_fill_count = 0
    positions_error: str | None = None
    tradebook_error: str | None = None
    acquire = getattr(self, "_acquire_bucket", None)
    bucket = getattr(self, "_GENERAL_BUCKET", None)
    request = getattr(self, "_make_request", None)
    ensure_json = getattr(self, "_ensure_json", None)

    try:
        if callable(acquire) and bucket is not None:
            acquire(bucket)
        if callable(request) and callable(ensure_json):
            response = ensure_json(
                request(
                    "GET",
                    "/portfolio/positions",
                    operation_label="pnl.positions",
                )
            )
            data = response.get("data") if isinstance(response, Mapping) else None
            day = data.get("day") if isinstance(data, Mapping) else None
            day_marked, day_closed, day_rows = _strategy_day_marked_pnl(day)
    except Exception as exc:  # diagnostic enrichment must never impair P&L authority
        positions_error = f"{type(exc).__name__}: {exc}"

    try:
        if callable(acquire) and bucket is not None:
            acquire(bucket)
        if callable(request) and callable(ensure_json):
            response = ensure_json(
                request("GET", "/trades", operation_label="pnl.trades")
            )
            trades = response.get("data") if isinstance(response, Mapping) else None
            tradebook_realized, tradebook_fill_count = (
                _strategy_tradebook_realized_pnl(trades)
            )
    except Exception as exc:  # diagnostic enrichment must never impair P&L authority
        tradebook_error = f"{type(exc).__name__}: {exc}"

    margin_realized = None if realized is None else float(realized)
    margin_unrealized = None if unrealized is None else float(unrealized)
    return {
        "account_realized": margin_realized,
        "account_unrealized": margin_unrealized,
        "account_total": (
            None
            if margin_realized is None
            else margin_realized + float(margin_unrealized or 0.0)
        ),
        "margin_m2m_realized": margin_realized,
        "margin_m2m_unrealized": margin_unrealized,
        "strategy_tradebook_realized_gross": tradebook_realized,
        "strategy_tradebook_fill_count": int(tradebook_fill_count),
        "strategy_day_marked_gross": day_marked,
        "strategy_day_closed_gross": day_closed,
        "strategy_day_rows": int(day_rows),
        "source": "zerodha_margins_m2m",
        "tradebook_source": "zerodha_trades",
        "tradebook_error": tradebook_error,
        "positions_source": "zerodha_positions_day",
        "positions_error": positions_error,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }


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


def _patched_position_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _ORIGINAL_POSITION_INIT(self, *args, **kwargs)
    self._broker_account_pnl_snapshot = {}
    self._broker_pnl_last_fetch_mono = 0.0
    self._broker_pnl_fetch_error = None
    self._broker_pnl_last_log_mono = 0.0
    self._broker_pnl_last_log_fingerprint = None
    # Persisted ``broker_realized_pnl`` came from legacy position rows. Retire
    # that authority immediately; the local fill ledger stays intact.
    with getattr(self, "_lock"):
        self._broker_realized_pnl = None
        self._refresh_realized_pnl_locked()


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
        and now_mono - last_fetch < _refresh_seconds()
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
        tradebook_realized = _finite_float(
            raw.get("strategy_tradebook_realized_gross")
        )
        tradebook_fills = int(
            _finite_float(raw.get("strategy_tradebook_fill_count")) or 0
        )
        positions_closed = _finite_float(raw.get("strategy_day_closed_gross"))
        positions_marked = _finite_float(raw.get("strategy_day_marked_gross"))
        positions_flat = (
            (_finite_float(raw.get("strategy_day_rows")) or 0) > 0
            and positions_closed is not None
            and positions_marked is not None
            and abs(positions_marked - positions_closed)
            <= _MATCH_TOLERANCE_RUPEES
        )
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
        status = (
            "unavailable"
            if difference is None
            else (
                "matched"
                if abs(difference) <= _MATCH_TOLERANCE_RUPEES
                else "mismatch"
            )
        )
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


def get_broker_account_pnl_snapshot(
    self: Any,
    *,
    force: bool = False,
) -> dict[str, Any]:
    return dict(refresh_broker_pnl_diagnostic(self, force=force))


def get_broker_account_realized_pnl(
    self: Any,
    *,
    force: bool = False,
    max_age_s: float | None = None,
) -> float | None:
    snapshot = refresh_broker_pnl_diagnostic(self, force=force)
    realized = _finite_float(snapshot.get("broker_realized_evidence"))
    fetched = _finite_float(snapshot.get("fetched_monotonic"))
    if realized is None or fetched is None:
        return None
    max_age = _max_age_seconds() if max_age_s is None else max(0.0, float(max_age_s))
    if time.monotonic() - fetched > max_age:
        return None
    return float(realized)


def get_strategy_realized_pnl(self: Any) -> float:
    with getattr(self, "_lock"):
        return float(getattr(self, "_local_realized_pnl", 0.0) or 0.0)


def _patched_synchronize_with_broker(self: Any, broker_positions: Any) -> Any:
    """Use position rows for exposure only when dedicated P&L authority exists."""

    payload = _materialize_positions(broker_positions)
    if _broker_supports_dedicated_pnl(self):
        payload = _strip_legacy_position_pnl(payload)
    result = _ORIGINAL_SYNCHRONIZE_WITH_BROKER(self, payload)
    if _broker_supports_dedicated_pnl(self):
        refresh_broker_pnl_diagnostic(self)
    return result


def _patched_pnl_reconciliation_snapshot(self: Any) -> dict[str, object]:
    base = dict(_ORIGINAL_PNL_RECONCILIATION_SNAPSHOT(self))
    diagnostic = refresh_broker_pnl_diagnostic(self)
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
            "broker_account_total": diagnostic.get("account_total"),
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


def apply_patches() -> None:
    """Install dedicated broker P&L authority after existing position patches."""

    global _PATCH_APPLIED
    global _ORIGINAL_POSITION_INIT
    global _ORIGINAL_SYNCHRONIZE_WITH_BROKER
    global _ORIGINAL_PNL_RECONCILIATION_SNAPSHOT
    if _PATCH_APPLIED:
        return

    from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient
    from nifty_scalper_bot.execution.position_manager import PositionManager

    if not callable(getattr(ZerodhaKiteClient, "get_pnl_snapshot", None)):
        ZerodhaKiteClient.get_pnl_snapshot = _zerodha_get_pnl_snapshot

    if getattr(PositionManager, "_broker_pnl_authority_patch", False):
        _PATCH_APPLIED = True
        return

    _ORIGINAL_POSITION_INIT = PositionManager.__init__
    _ORIGINAL_SYNCHRONIZE_WITH_BROKER = PositionManager.synchronize_with_broker
    _ORIGINAL_PNL_RECONCILIATION_SNAPSHOT = PositionManager.pnl_reconciliation_snapshot

    PositionManager.__init__ = _patched_position_init
    PositionManager.synchronize_with_broker = _patched_synchronize_with_broker
    PositionManager.refresh_broker_pnl_diagnostic = refresh_broker_pnl_diagnostic
    PositionManager.get_broker_account_pnl_snapshot = get_broker_account_pnl_snapshot
    PositionManager.get_broker_account_realized_pnl = get_broker_account_realized_pnl
    PositionManager.get_strategy_realized_pnl = get_strategy_realized_pnl
    PositionManager.pnl_reconciliation_snapshot = _patched_pnl_reconciliation_snapshot
    PositionManager._broker_pnl_authority_patch = True
    _PATCH_APPLIED = True


__all__ = [
    "apply_patches",
    "_extract_account_m2m",
    "_strategy_day_marked_pnl",
    "_strategy_tradebook_realized_pnl",
    "_strip_legacy_position_pnl",
]
