"""Runtime role:
- Parsing/canonicalization utilities for active contract diagnostics.
- Runtime active contract selection belongs to InstrumentManager.
- Must not derive live futures from selected options."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo
import calendar
import re
from typing import Any, Iterable, Mapping, Sequence

NIFTY_FUT_RE = re.compile(r"^NIFTY(?P<yy>\d{2})(?P<mon>[A-Z]{3})FUT$")
NIFTY_OPT_RE = re.compile(r"^NIFTY(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<strike>\d{4,6})(?P<side>CE|PE)$")
MONTH_ABBR_TO_NUM = {m.upper(): i for i, m in enumerate(calendar.month_abbr) if m}
IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class ActiveFutureResolution:
    symbol: str | None
    source: str
    expired_input: bool = False
    reason: str | None = None


def normalize_symbol(symbol: Any) -> str:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    if ":" not in raw and raw.startswith("NIFTY"):
        if raw in {"NIFTY", "NIFTY50", "NSE:NIFTY", "NSE:NIFTY50"}:
            return "NSE:NIFTY"
        return f"NFO:{raw}"
    return raw


def _trading_date(now: datetime | date | None = None) -> date:
    if now is None:
        return datetime.now(IST).date()
    if isinstance(now, datetime):
        if now.tzinfo is None:
            return now.date()
        return now.astimezone(IST).date()
    return now


def canonical_nifty_future_symbol(symbol: Any) -> str | None:
    canonical = normalize_symbol(symbol)
    if not canonical or ":" not in canonical:
        return None
    exchange, tradingsymbol = canonical.split(":", 1)
    if exchange != "NFO" or not NIFTY_FUT_RE.fullmatch(tradingsymbol):
        return None
    return f"NFO:{tradingsymbol}"


def canonical_nifty_option_symbol(symbol: Any) -> str | None:
    canonical = normalize_symbol(symbol)
    if not canonical or ":" not in canonical:
        return None
    exchange, tradingsymbol = canonical.split(":", 1)
    if exchange != "NFO" or not NIFTY_OPT_RE.fullmatch(tradingsymbol):
        return None
    return f"NFO:{tradingsymbol}"


def _last_thursday(year: int, month: int) -> date:
    d = date(year, month, calendar.monthrange(year, month)[1])
    while d.weekday() != 3:
        d = date.fromordinal(d.toordinal() - 1)
    return d


def parse_nifty_future_expiry(symbol: Any) -> date | None:
    canonical = canonical_nifty_future_symbol(symbol)
    if not canonical:
        return None
    match = NIFTY_FUT_RE.fullmatch(canonical.split(":", 1)[1])
    if not match:
        return None
    year = 2000 + int(match.group("yy"))
    month = MONTH_ABBR_TO_NUM.get(match.group("mon"))
    return _last_thursday(year, month) if month else None


def parse_instrument_expiry(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value).strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw[:10], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def is_nifty_future_expired(symbol: Any, *, now: datetime | date | None = None) -> bool:
    expiry = parse_nifty_future_expiry(symbol)
    if expiry is None:
        return False
    return _trading_date(now) > expiry


def _is_nifty_future_row(row: Mapping[str, Any]) -> bool:
    tradingsymbol = str(row.get("tradingsymbol") or row.get("symbol") or "").upper()
    name = str(row.get("name") or "").upper()
    instrument_type = str(row.get("instrument_type") or row.get("segment") or "").upper()
    exchange = str(row.get("exchange") or "NFO").upper()
    return exchange == "NFO" and (name == "NIFTY" or tradingsymbol.startswith("NIFTY")) and tradingsymbol.endswith("FUT") and "FUT" in instrument_type and bool(NIFTY_FUT_RE.fullmatch(tradingsymbol))


def resolve_active_nifty_future_from_instruments(instruments: Iterable[Mapping[str, Any]], *, now: datetime | date | None = None) -> ActiveFutureResolution:
    today = _trading_date(now)
    best_symbol: str | None = None
    best_expiry: date | None = None
    for row in instruments:
        if not isinstance(row, Mapping) or not _is_nifty_future_row(row):
            continue
        tradingsymbol = str(row.get("tradingsymbol") or row.get("symbol") or "").upper()
        expiry = parse_instrument_expiry(row.get("expiry")) or parse_nifty_future_expiry(f"NFO:{tradingsymbol}")
        if expiry is None or expiry < today:
            continue
        if best_expiry is None or expiry < best_expiry:
            best_expiry = expiry
            best_symbol = f"NFO:{tradingsymbol}"
    return ActiveFutureResolution(best_symbol, "instrument_master", reason=None if best_symbol else "no_unexpired_nifty_future_found")


def derive_nifty_future_from_selected_options(selected_option_symbols: Sequence[Any] | None, *, now: datetime | date | None = None) -> ActiveFutureResolution:
    import os
    if os.getenv("EXECUTION_MODE", os.getenv("MODE", "")).strip().upper() == "LIVE":
        raise RuntimeError("derive_nifty_future_from_selected_options is diagnostic only; live runtime must use InstrumentManager")
    symbols = [canonical_nifty_option_symbol(s) for s in (selected_option_symbols or [])]
    months = set()
    for symbol in [s for s in symbols if s]:
        match = NIFTY_OPT_RE.fullmatch(symbol.split(":", 1)[1])
        if match:
            months.add(f"{match.group('yy')}{match.group('mon')}")
    if len(months) != 1:
        return ActiveFutureResolution(None, "selected_options", reason="ambiguous_or_missing_option_month")
    candidate = f"NFO:NIFTY{next(iter(months))}FUT"
    if is_nifty_future_expired(candidate, now=now):
        return ActiveFutureResolution(None, "selected_options", expired_input=True, reason="derived_future_expired")
    return ActiveFutureResolution(candidate, "selected_options")


def resolve_active_nifty_future(
    *,
    instruments: Iterable[Mapping[str, Any]] | None = None,
    selected_option_symbols: Sequence[Any] | None = None,
    configured_symbol: Any | None = None,
    now: datetime | date | None = None,
    allow_selected_option_fallback: bool = False,
) -> ActiveFutureResolution:
    if instruments is not None:
        resolved = resolve_active_nifty_future_from_instruments(instruments, now=now)
        if resolved.symbol:
            return resolved
    if allow_selected_option_fallback:
        selected = derive_nifty_future_from_selected_options(selected_option_symbols, now=now)
        if selected.symbol:
            return selected
    configured = canonical_nifty_future_symbol(configured_symbol)
    if configured and not is_nifty_future_expired(configured, now=now):
        return ActiveFutureResolution(configured, "configured_symbol")
    return ActiveFutureResolution(None, "unresolved", reason="active_future_unresolved")
