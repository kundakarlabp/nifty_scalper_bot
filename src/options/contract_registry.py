"""Contract metadata registry backed by NSE/Kite contract files."""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from src.utils.expiry import last_tuesday_of_month

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class Contract:
    """Normalized contract metadata entry."""

    symbol: str
    expiry: datetime
    strike: int
    option_type: str
    token: int
    tradingsymbol: str
    lot_size: int
    weekly: bool


class InstrumentRegistry:
    """Registry that keeps option contract metadata fresh."""

    def __init__(
        self,
        *,
        source: str | os.PathLike[str] | None = None,
        tz: ZoneInfo | None = None,
    ) -> None:
        self._source = Path(source) if source and _is_file_path(source) else source
        self._tz = tz or IST
        self._contracts: dict[tuple[str, str, int, str], Contract] = {}
        self._expiries: dict[str, list[datetime]] = {}
        self._lot_by_symbol: dict[str, int] = {}
        self._lot_by_symbol_expiry: dict[tuple[str, str], int] = {}
        self._checksum: str | None = None
        self._last_refresh: datetime | None = None

    # ------------------------------------------------------------------
    def refresh(self, *, force: bool = False) -> None:
        """Refresh registry from the configured contract master."""

        source = self._resolve_source()
        if source is None:
            logger.debug("InstrumentRegistry.refresh skipped: no source configured")
            return

        raw = self._load_bytes(source)
        checksum = hashlib.md5(raw).hexdigest()
        if not force and checksum == self._checksum:
            return

        rows = list(self._iter_rows(raw))
        if not rows:
            logger.warning("InstrumentRegistry.refresh: contract file empty")
            return

        self._contracts.clear()
        self._expiries.clear()
        self._lot_by_symbol.clear()
        self._lot_by_symbol_expiry.clear()

        for row in rows:
            contract = self._normalize_row(row)
            if contract is None:
                continue
            expiry_key = contract.expiry.date().isoformat()
            key = (
                contract.symbol,
                expiry_key,
                int(contract.strike),
                contract.option_type,
            )
            self._contracts[key] = contract
            self._lot_by_symbol.setdefault(contract.symbol, contract.lot_size)
            self._lot_by_symbol_expiry[(contract.symbol, expiry_key)] = contract.lot_size
            self._expiries.setdefault(contract.symbol, [])
            self._expiries[contract.symbol].append(contract.expiry)

        for symbol, values in self._expiries.items():
            deduped = sorted({dt for dt in values})
            self._expiries[symbol] = deduped

        self._checksum = checksum
        self._last_refresh = datetime.now(UTC)

    # ------------------------------------------------------------------
    def lookup(
        self, symbol: str, expiry: str, strike: int, option_type: str
    ) -> dict[str, object] | None:
        """Return contract metadata as a dictionary if known."""

        key = (symbol.upper(), expiry, int(strike), option_type.upper())
        contract = self._contracts.get(key)
        if contract is None:
            return None
        return {
            "token": contract.token,
            "tradingsymbol": contract.tradingsymbol,
            "lot_size": contract.lot_size,
            "expiry": contract.expiry,
            "weekly": contract.weekly,
        }

    # ------------------------------------------------------------------
    def lot_size(self, symbol: str, expiry: str | None = None) -> int | None:
        """Return lot size for ``symbol`` optionally scoped to ``expiry``."""

        symbol_key = symbol.upper()
        if expiry:
            lot = self._lot_by_symbol_expiry.get((symbol_key, expiry))
            if lot:
                return lot
        return self._lot_by_symbol.get(symbol_key)

    # ------------------------------------------------------------------
    def next_expiry(self, symbol: str, now: datetime | None = None) -> datetime | None:
        """Return the next expiry datetime for ``symbol`` if available."""

        symbol_key = symbol.upper()
        expiries = self._expiries.get(symbol_key, [])
        if not expiries:
            return None
        ts_now = self._coerce_tz(now)
        for expiry in expiries:
            if expiry >= ts_now:
                return expiry
        return expiries[-1]

    # ------------------------------------------------------------------
    def next_weekly_expiry(
        self, symbol: str, now: datetime | None = None
    ) -> datetime | None:
        """Return next weekly expiry using exchange data when available."""

        expiry = self.next_expiry(symbol, now)
        if expiry is None:
            return None
        last_monthly = last_tuesday_of_month(expiry)
        if expiry.date() == last_monthly.date():
            # Skip monthly contract when searching for weekly expiry
            expiries = self._expiries.get(symbol.upper(), [])
            for item in expiries:
                if item > expiry and item.date() != last_tuesday_of_month(item).date():
                    return item
            return expiry
        return expiry

    # ------------------------------------------------------------------
    @property
    def last_refresh(self) -> datetime | None:
        return self._last_refresh

    # ------------------------------------------------------------------
    def _resolve_source(self) -> str | os.PathLike[str] | None:
        if self._source:
            return self._source
        env_path = os.getenv("INSTRUMENTS__CSV") or os.getenv("INSTRUMENTS_CSV")
        return env_path

    # ------------------------------------------------------------------
    def _load_bytes(self, source: str | os.PathLike[str]) -> bytes:
        if _is_url(source):
            resp = requests.get(str(source), timeout=10)
            resp.raise_for_status()
            return resp.content
        path = Path(source)
        return path.read_bytes()

    # ------------------------------------------------------------------
    def _iter_rows(self, payload: bytes) -> Iterator[Mapping[str, object]]:
        buffer = io.BytesIO(payload)
        if zipfile.is_zipfile(buffer):
            with zipfile.ZipFile(buffer) as zf:
                name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
                if name is None:
                    return iter([])
                data = zf.read(name)
        else:
            data = payload
        text = data.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        return iter(reader)

    # ------------------------------------------------------------------
    def _normalize_row(self, row: Mapping[str, object]) -> Contract | None:
        try:
            symbol = str(row.get("name") or row.get("tradingsymbol") or "").upper()
            if not symbol:
                return None
            option_type_raw = str(row.get("instrument_type") or "").upper()
            option_type = _normalize_option_type(option_type_raw)
            if option_type not in {"CE", "PE"}:
                return None
            token = int(float(str(row.get("instrument_token") or row.get("token") or 0)))
            strike = int(float(str(row.get("strike") or 0)))
            lot_size = int(float(str(row.get("lot_size") or 0)))
            tradingsymbol = str(row.get("tradingsymbol") or "")
            expiry_raw = row.get("expiry") or row.get("last_trading_date")
            if isinstance(expiry_raw, datetime):
                expiry_dt = expiry_raw
            else:
                expiry_dt = _parse_expiry(str(expiry_raw))
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(hour=15, minute=30, tzinfo=self._tz)
            else:
                expiry_dt = expiry_dt.astimezone(self._tz)
            weekly = expiry_dt.date() != last_tuesday_of_month(expiry_dt).date()
        except Exception as exc:  # pragma: no cover - defensive parsing
            logger.debug("InstrumentRegistry._normalize_row skipped row: %s", exc)
            return None

        return Contract(
            symbol=symbol,
            expiry=expiry_dt,
            strike=strike,
            option_type=option_type,
            token=token,
            tradingsymbol=tradingsymbol,
            lot_size=lot_size if lot_size > 0 else 0,
            weekly=weekly,
        )

    # ------------------------------------------------------------------
    def _coerce_tz(self, value: datetime | None) -> datetime:
        if value is None:
            return datetime.now(self._tz)
        if value.tzinfo is None:
            return value.replace(tzinfo=self._tz)
        return value.astimezone(self._tz)


def _normalize_option_type(value: str) -> str:
    if value in {"CE", "PE"}:
        return value
    if value.endswith("CE"):
        return "CE"
    if value.endswith("PE"):
        return "PE"
    if value.endswith("CALL"):
        return "CE"
    if value.endswith("PUT"):
        return "PE"
    if value in {"CALL", "BUY"}:
        return "CE"
    if value in {"PUT", "SELL"}:
        return "PE"
    return value


def _is_url(value: str | os.PathLike[str]) -> bool:
    return str(value).startswith("http://") or str(value).startswith("https://")


def _is_file_path(value: str | os.PathLike[str]) -> bool:
    try:
        return Path(value).exists()
    except OSError:
        return False


def _parse_expiry(raw: str) -> datetime:
    candidate = raw.strip()
    fmts = [
        "%Y-%m-%d",
        "%d-%b-%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt
        except ValueError:
            continue
    return datetime.fromisoformat(candidate)
