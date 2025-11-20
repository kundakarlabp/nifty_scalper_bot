# src/nifty_scalper_bot/data/instruments.py
"""
Instrument resolver and CSV/SQLite instrument loader.

Provides:
- InstrumentResolver: broker/CSV/DB-backed instrument token resolver.
- ensure_sqlite: create/open a sqlite db for cached instruments.
- refresh_from_csv: read instruments CSV and persist into sqlite.
- load_rows_for_resolver: return rows (mapping) usable by resolver.warm_from_broker_dump.

Enhancements in this optimized version:
- Configurable negative cache TTL.
- Attach a "master symbol map" (from instrument master CSV) so resolver can validate
  generated symbols against the exchange master list (source of truth).
- Helpers for expiry calculation suitable for the 2025 Tuesday-expiry regime.
- Weekly month single-character encoding (including O/N/D for Oct/Nov/Dec).
- Holiday-aware expiry adjustment (pull-back logic).
- Conservative and thread-safe cache handling and logging.
"""
from __future__ import annotations

import csv
import os
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import logging

LOGGER = logging.getLogger("nifty_scalper_bot.data.instruments")

# --- Configuration ----------------------------------------------------------
_DEFAULT_NEG_TTL = 300.0  # seconds - negative cache TTL
_NEG_TTL = float(os.getenv("NSB__INSTRUMENT_NEG_TTL", _DEFAULT_NEG_TTL))

# Well-known fallback tokens for indices (kept minimal; extend if required)
WELL_KNOWN: Dict[str, int] = {
    "NIFTY": 256265,  # canonical NIFTY 50 token (example)
    "BANKNIFTY": 260105,
    "NSE:NIFTY": 256265,
    "NSE:BANKNIFTY": 260105,
}

# Canonical human-readable names for some tokens (for format_token_as_symbol)
CANONICAL_TOKENS: Dict[int, str] = {
    256265: "NIFTY 50",
    260105: "NIFTY BANK",
}

# Per-index expiry policy. target_weekday: 0=Mon,1=Tue,... weekly indicates if weekly expiry is allowed
INDEX_EXPIRY_POLICY: Dict[str, Dict[str, Any]] = {
    "NIFTY": {"weekly": True, "expiry_weekday": 1},  # Tuesday
    "BANKNIFTY": {"weekly": False, "expiry_weekday": 1},  # monthly only
    # Add other roots if needed
}

# Single-char mapping for weekly month codes (10->O, 11->N, 12->D)
WEEKLY_MONTH_CODE: Dict[int, str] = {10: "O", 11: "N", 12: "D"}

# SQLite schema
_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS instruments (
    instrument_token INTEGER PRIMARY KEY,
    exchange TEXT,
    tradingsymbol TEXT NOT NULL,
    lot_size INTEGER,
    tick_size REAL,
    expiry TEXT,
    strike REAL,
    instrument_type TEXT,
    raw_json TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tradingsymbol ON instruments(tradingsymbol);
CREATE INDEX IF NOT EXISTS idx_exchange_tradingsymbol ON instruments(exchange, tradingsymbol);
"""


# --- Data classes -----------------------------------------------------------
@dataclass(slots=True)
class Instrument:
    tradingsymbol: str
    exchange: Optional[str]
    instrument_token: int
    lot_size: Optional[int] = None
    tick_size: Optional[float] = None
    expiry: Optional[str] = None
    strike: Optional[float] = None
    instrument_type: Optional[str] = None
    raw: Optional[Mapping[str, Any]] = None


class BrokerError(Exception):
    """Generic broker/resolver error sentinel."""


# --- InstrumentResolver ----------------------------------------------------
class InstrumentResolver:
    """
    Caches broker instruments and provides reliable symbol resolution for the rest of the bot.

    ctor: InstrumentResolver(broker_client: Optional[Any]) -> resolver
    """

    def __init__(self, broker_client: Any | None = None) -> None:
        self._broker = broker_client

        # symbol -> token (upper-cased keys). store both "NFO:..." and bare symbols
        self._by_symbol: Dict[str, int] = {}

        # token -> base tradingsymbol (not prefixed)
        self._symbol_by_token: Dict[int, str] = {}

        # token -> exchange hint
        self._exchange_by_token: Dict[int, str] = {}

        # lightweight option contract catalogue: base -> [contract dicts]
        self._option_contracts: Dict[str, List[Dict[str, Any]]] = {}

        # negative cache: key -> expiry_ts
        self._neg_cache: Dict[str, float] = {}
        self._warned_no_token: set[str] = set()
        self._neg_ttl = _NEG_TTL

        # external authoritative master symbol map (attached by MarketDataManager)
        # mapping: "NFO:SYMBOL" or "SYMBOL" -> token
        self._master_map: Optional[Mapping[str, int]] = None

        # locks
        self._lock = threading.RLock()

        # seed well-known tokens
        self._seed_well_known()

    # -------------------- public API ---------------------------------------
    def attach_master_map(self, master_map: Mapping[str, int]) -> None:
        """
        Attach a read-only master symbol->token mapping (usually parsed from Kite / instruments CSV).
        This map is used for authoritative validation of generated symbols and for lot-size lookups.
        """
        with self._lock:
            self._master_map = master_map
        LOGGER.info("InstrumentResolver attached master instrument map (size=%d)", len(master_map) if master_map is not None else 0)

    def warm(self) -> None:
        """
        Warm caches from broker if possible. Non-fatal on failure.
        """
        LOGGER.debug("Entered InstrumentResolver.warm", extra={"event": "instrument_resolver_warm_enter"})
        items: Optional[Iterable[Mapping[str, Any]]] = None

        for name in ("list_instruments", "get_instruments", "instruments", "load_instruments", "fetch_instruments"):
            fn = getattr(self._broker, name, None)
            if callable(fn):
                try:
                    items = fn()
                except Exception as exc:
                    LOGGER.warning("InstrumentResolver: broker.%s() failed: %s", name, exc)
                break

        if items:
            count = 0
            for row in items:
                if not isinstance(row, Mapping):
                    continue
                try:
                    self._ingest_instrument_row(row)
                    count += 1
                except Exception:
                    LOGGER.exception("ingest_instrument_row failed for row: %s", row)
            LOGGER.info("InstrumentResolver: warmed from broker with %d rows", count, extra={"event": "instrument_resolver_warm_broker"})
        else:
            LOGGER.info("InstrumentResolver: no broker instrument dump available; using well-known fallbacks", extra={"event": "instrument_resolver_warm_no_broker"})
        self._seed_well_known()
        LOGGER.info("InstrumentResolver ready with %d symbols", len(self._by_symbol), extra={"event": "instrument_resolver_ready"})

    def warm_from_broker_dump(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Warm caches directly from caller-supplied rows (CSV/DB dumps)."""
        LOGGER.debug("Entered InstrumentResolver.warm_from_broker_dump", extra={"event": "instrument_resolver_warm_dump_enter"})
        if rows is None:
            return
        with self._lock:
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                try:
                    self._ingest_instrument_row(row)
                except Exception:
                    LOGGER.exception("instrument_resolver_ingest_error for row: %s", row)
            self._seed_well_known()
        LOGGER.info("InstrumentResolver warmed from dump: symbols=%d", len(self._by_symbol), extra={"event": "instrument_resolver_warm_dump_complete"})

    def upsert(self, symbol: str, token: int, *, exchange: Optional[str] = None) -> None:
        """
        Add or update a symbol->token mapping. Accepts 'NFO:SYMBOL' or 'SYMBOL'.
        """
        if not symbol:
            LOGGER.debug("instrument_resolver_upsert_blank called", extra={"event": "instrument_resolver_upsert_blank"})
            return
        try:
            normalized = str(symbol).strip().upper()
            token_int = int(token)
            base_symbol = normalized.split(":", 1)[-1] or normalized
            exchange_hint = (exchange or "").strip().upper() or None
            with self._lock:
                self._by_symbol[normalized] = token_int
                self._by_symbol.setdefault(base_symbol, token_int)
                if exchange_hint:
                    self._by_symbol.setdefault(f"{exchange_hint}:{base_symbol}", token_int)
                    self._exchange_by_token[token_int] = exchange_hint
                self._symbol_by_token[token_int] = base_symbol
                self._clear_negative_cache_for_key(base_symbol)
            LOGGER.info("instrument_resolver_upsert: %s -> %s", base_symbol, token_int, extra={"event": "instrument_resolver_upsert"})
        except Exception:
            LOGGER.exception("Failure in InstrumentResolver.upsert", extra={"event": "instrument_resolver_upsert_error", "symbol": symbol, "token": token})

    def resolve(self, symbol: str | int | None) -> Optional[int]:
        """
        Resolve a symbol (or numeric token) to int token. Returns None if not found.
        """
        if symbol is None:
            return None

        # numeric direct path
        try:
            if isinstance(symbol, (int, float)):
                return int(symbol)
            if isinstance(symbol, str) and symbol.strip().isdigit():
                return int(symbol.strip())
        except Exception:
            pass

        key = str(symbol).strip().upper()
        if not key:
            return None

        now_ts = time.time()
        with self._lock:
            exp = self._neg_cache.get(key)
            if exp and exp > now_ts:
                LOGGER.debug("instrument_resolver_negative_cache_hit", extra={"event": "instrument_resolver_negative_cache", "symbol": key})
                return None

            # positive cache exact / base-only
            if key in self._by_symbol:
                return int(self._by_symbol[key])
            base = key.split(":", 1)[-1]
            if base in self._by_symbol:
                return int(self._by_symbol[base])

        # attempt broker/store heuristics
        try:
            token = self._search_token_via_broker_or_store(key)
            if token:
                with self._lock:
                    self._by_symbol.setdefault(key, int(token))
                    self._symbol_by_token.setdefault(int(token), key.split(":", 1)[-1])
                return int(token)
        except Exception:
            LOGGER.exception("broker_token_search_error", extra={"event": "instrument_resolver_broker_search_error", "symbol": key})

        # negative cache miss
        with self._lock:
            self._neg_cache[key] = now_ts + self._neg_ttl
            self._warned_no_token.add(key.split(":", 1)[-1])
        LOGGER.warning("instrument_resolver_no_token", extra={"event": "instrument_resolver_no_token", "symbol": key})
        return None

    def resolve_symbol_to_token(self, symbol: str) -> Optional[int]:
        """Compatibility wrapper used by some modules."""
        try:
            return self.resolve(symbol)
        except Exception:
            LOGGER.exception("instrument_resolver_token_error", extra={"event": "instrument_resolver_token_error", "symbol": symbol})
            return None

    def lookup(self, symbol: str | int | None) -> Optional[Dict[str, Any]]:
        """
        Return instrument metadata (instrument_token, symbol, exchange if available).
        """
        if symbol is None:
            return None
        try:
            normalized = str(symbol).strip().upper()
            strip_prefix = os.getenv("RESOLVER_STRIP_EXCHANGE_PREFIX", "true").lower() == "true"
            if strip_prefix and ":" in normalized:
                normalized = normalized.split(":", 1)[-1].strip()

            # numeric token lookup
            if normalized.isdigit():
                token = int(normalized)
                base = self._symbol_by_token.get(token)
                exchange = self._exchange_by_token.get(token)
                if base:
                    out = {"instrument_token": token, "symbol": base}
                    if exchange:
                        out["exchange"] = exchange
                    return out
                return None

            token = self.resolve(normalized)
            if token is None:
                candidates = [normalized, normalized.split(":", 1)[-1]]
                with self._lock:
                    for cand in candidates:
                        t = self._by_symbol.get(cand)
                        if t:
                            return {"instrument_token": int(t), "symbol": cand}
                return None

            meta: Dict[str, Any] = {"instrument_token": int(token)}
            with self._lock:
                sym = self._symbol_by_token.get(int(token))
                exc = self._exchange_by_token.get(int(token))
            if sym:
                meta["symbol"] = sym
            if exc:
                meta["exchange"] = exc
            return meta
        except Exception:
            LOGGER.exception("instrument_resolver_lookup_error", extra={"event": "instrument_resolver_lookup_error", "symbol": symbol})
            return None

    def resolve_exchange(self, symbol: str) -> Optional[str]:
        """Return exchange hint for symbol when available (best-effort)."""
        try:
            token = self.resolve_symbol_to_token(symbol)
            if token is not None:
                exchange = self._exchange_by_token.get(int(token))
                if exchange:
                    return exchange
            try:
                return self.exchange_for_symbol(symbol)
            except BrokerError:
                return None
        except Exception:
            LOGGER.exception("instrument_resolver_exchange_error", extra={"event": "instrument_resolver_exchange_error", "symbol": symbol})
            return None

    def exchange_for_symbol(self, symbol: str) -> str:
        """Enforce exchange for option symbols (raise BrokerError for invalid formats)."""
        exchange, _ = self._normalize_option_symbol(symbol)
        if exchange is not None and exchange != "NFO":
            raise BrokerError("Only NFO exchange is supported for NIFTY options")
        return "NFO"

    def tradingsymbol_for_order(self, symbol: str) -> str:
        """Return tradingsymbol suitable for order placement (no exchange prefix)."""
        _exchange, tradingsymbol = self._normalize_option_symbol(symbol)
        return tradingsymbol

    def canonicalize(self, symbol: str) -> Tuple[str, Optional[str], str]:
        """
        Normalize a symbol and return (tradingsymbol, exchange, segment_type).
        Segment types: "OPTIONS", "INDEX", "FUTURES", "UNKNOWN"
        """
        if not symbol:
            raise BrokerError("Symbol required")
        raw = str(symbol).strip().upper()
        if ":" in raw:
            prefix, remainder = raw.split(":", 1)
            exchange = prefix or None
            tradingsymbol = remainder.strip()
        else:
            exchange = None
            tradingsymbol = raw

        tradingsymbol = tradingsymbol.replace(" ", "")
        if tradingsymbol.endswith("CE") or tradingsymbol.endswith("PE"):
            segment = "OPTIONS"
        elif tradingsymbol.endswith("FUT"):
            segment = "FUTURES"
        elif tradingsymbol in ("NIFTY", "BANKNIFTY"):
            segment = "INDEX"
        else:
            segment = "UNKNOWN"
        return tradingsymbol, exchange, segment

    def build_quote_keys(self, symbol: str) -> Tuple[str, List[str]]:
        """
        Build canonical symbol and candidate quote keys for broker lookups.
        Returns (canonical_symbol, [candidate_keys...])
        """
        canonical = str(symbol).strip().upper()
        if ":" in canonical:
            _pref, canonical = canonical.split(":", 1)
        canonical = canonical.replace(" ", "")
        candidates: List[str] = []
        if canonical.endswith("CE") or canonical.endswith("PE"):
            candidates.append(f"NFO:{canonical}")
        candidates.append(canonical)
        return canonical, candidates

    def format_token_as_symbol(self, token: int) -> str:
        """
        Format token into a human readable exchange:symbol string using canonical tokens.
        """
        try:
            token_int = int(token)
            canonical = CANONICAL_TOKENS.get(token_int)
            exchange = self._exchange_by_token.get(token_int, "NSE")
            if canonical:
                return f"{exchange}:{canonical}"
            base = self._symbol_by_token.get(token_int)
            if base:
                return f"{exchange}:{base}"
            return str(token)
        except Exception:
            return str(token)

    def candidates_for_quote(self, raw: str) -> Tuple[str, ...]:
        """
        Build candidate keys suitable for many broker.quote_any() functions.
        """
        canonical, keys = self.build_quote_keys(raw)
        out: List[str] = []
        out.extend(keys)
        with self._lock:
            token = None
            t = self._by_symbol.get(canonical)
            if t:
                token = t
            else:
                base = canonical.split(":", 1)[-1]
                token = self._by_symbol.get(base)
            if token:
                out.append(str(int(token)))
        # dedupe preserving order
        seen = set()
        ordered = []
        for k in out:
            if k and k not in seen:
                ordered.append(k)
                seen.add(k)
        return tuple(ordered)

    # -------------------- instrument generation & expiry helpers --------------
    def find_expiry_dates(self, year: int, month: int, root: str, holidays: Optional[Iterable[date]] = None) -> List[date]:
        """
        Return candidate expiry dates for the given month/year for the provided root (e.g. "NIFTY").
        Uses INDEX_EXPIRY_POLICY to determine the anchor weekday (e.g. Tuesday).
        The returned list includes all weekly anchor days in the month (adjusted for holidays by pulling back).
        """
        holidays_set = set(holidays or [])
        policy = INDEX_EXPIRY_POLICY.get(root.upper(), {"weekly": True, "expiry_weekday": 1})
        weekday = int(policy.get("expiry_weekday", 1))
        results: List[date] = []

        # Start from the 1st of the month and find first matching weekday
        try:
            d = date(year, month, 1)
        except Exception:
            return results

        # advance to first target weekday in month
        while d.weekday() != weekday:
            d += timedelta(days=1)

        while d.month == month:
            exp = d
            # if expiry falls on holiday, push back to previous trading day(s)
            while exp in holidays_set or exp.weekday() >= 5:  # avoid weekends
                exp -= timedelta(days=1)
            results.append(exp)
            d += timedelta(days=7)
        return results

    def _weekly_month_code(self, dt: date) -> str:
        """
        Return single-character monthly code used in weekly instrument naming.
        """
        return WEEKLY_MONTH_CODE.get(dt.month, str(dt.month))  # fallback to numeric if map not present

    def generate_weekly_symbol(self, root: str, expiry_dt: date, strike: int, right: str, year_two_digit: Optional[int] = None) -> str:
        """
        Build weekly symbol like: NIFTY25N0424000PE
        - root: e.g. 'NIFTY'
        - expiry_dt: actual expiry date (date object)
        - strike: integer strike (e.g. 24000)
        - right: 'CE' or 'PE'
        - year_two_digit: optional override for two-digit year; default from expiry_dt
        """
        root_clean = root.strip().upper()
        yy = f"{(year_two_digit if year_two_digit is not None else expiry_dt.year % 100):02d}"
        mcode = self._weekly_month_code(expiry_dt)
        dd = f"{expiry_dt.day:02d}"
        strike_s = str(int(strike))
        return f"{root_clean}{yy}{mcode}{dd}{strike_s}{right}"

    def generate_monthly_symbol(self, root: str, expiry_dt: date, strike: int, right: str, year_two_digit: Optional[int] = None) -> str:
        """
        Build monthly symbol like: NIFTY25NOV24000PE
        - Uses 3-letter month for monthly contracts.
        """
        root_clean = root.strip().upper()
        yy = f"{(year_two_digit if year_two_digit is not None else expiry_dt.year % 100):02d}"
        month_code = expiry_dt.strftime("%b").upper()
        strike_s = str(int(strike))
        return f"{root_clean}{yy}{month_code}{strike_s}{right}"

    def validate_symbol_exists(self, symbol: str) -> bool:
        """
        Validate that a generated or requested symbol exists in authoritative master_map (if attached)
        or in the local in-memory by_symbol caches.
        """
        if not symbol:
            return False
        key = str(symbol).strip().upper()
        with self._lock:
            # check local caches first
            if key in self._by_symbol or key.split(":", 1)[-1] in self._by_symbol:
                return True
            # check attached master map
            mm = self._master_map
            if mm is not None:
                if key in mm or key.split(":", 1)[-1] in mm:
                    return True
        return False

    def get_lot_size_for_root(self, root: str) -> Optional[int]:
        """
        Return a representative lot size for the given root (e.g. 'NIFTY').
        This uses the attached master_map if available (expects master_map values to be tokens;
        the caller should use MarketDataManager to query lot sizes from parsed CSV).
        NOTE: InstrumentResolver does not persist lot-size per symbol; this helper is provided
        for convenience when master_map is enhanced by MarketDataManager.
        """
        # This function is intentionally conservative: resolver does not know lot sizes by default.
        return None

    # -------------------- internal helpers ---------------------------------
    def _clear_negative_cache_for_key(self, symbol: str) -> None:
        base = symbol.split(":", 1)[-1].strip().upper()
        with self._lock:
            for k in (symbol.strip().upper(), base):
                self._neg_cache.pop(k, None)
            self._warned_no_token.discard(base)

    def _ingest_instrument_row(self, row: Mapping[str, Any]) -> None:
        """
        Load a single instrument row into resolver caches.

        Accepts flexible keys: 'tradingsymbol'|'symbol', 'instrument_token'|'token', 'exchange',
        'lot_size'|'lot', 'strike', 'expiry', 'instrument_type' etc.
        """
        try:
            tradingsymbol = str(row.get("tradingsymbol") or row.get("symbol") or "").strip()
            if not tradingsymbol:
                return
            exchange = (row.get("exchange") or "").strip().upper() or None
            token_value = row.get("instrument_token") or row.get("token")
            if token_value is None:
                return
            try:
                token_int = int(float(token_value))
            except (TypeError, ValueError):
                LOGGER.debug("instrument_resolver_ingest_token_cast_failed", extra={"event": "instrument_resolver_ingest_token_cast_failed", "tradingsymbol": tradingsymbol, "exchange": exchange, "token_value": token_value})
                return

            with self._lock:
                key = tradingsymbol.upper()
                # primary keys
                self._by_symbol.setdefault(key, token_int)
                if exchange:
                    self._by_symbol.setdefault(f"{exchange}:{key}", token_int)
                # base symbol fallback (no prefix)
                base_key = key.split(":", 1)[-1]
                self._by_symbol.setdefault(base_key, token_int)

                # map token back to symbol and exchange if absent
                self._symbol_by_token.setdefault(token_int, tradingsymbol)
                if exchange:
                    self._exchange_by_token.setdefault(token_int, exchange)

                # if option contract, record light metadata
                itype = (row.get("instrument_type") or row.get("instrumentType") or "").strip().upper()
                if itype in ("CE", "PE", "OPT", "OPTION"):
                    base = self._base_index_from_tradingsymbol(tradingsymbol)
                    if base:
                        self._option_contracts.setdefault(base, [])
                        try:
                            strike_val = None
                            if row.get("strike") not in (None, "", "NULL"):
                                strike_val = float(row.get("strike"))
                        except Exception:
                            strike_val = None
                        contract = {
                            "instrument_token": token_int,
                            "tradingsymbol": tradingsymbol,
                            "option_type": "CE" if tradingsymbol.endswith("CE") else "PE",
                            "expiry": row.get("expiry"),
                            "strike": strike_val,
                            "lot_size": row.get("lot_size") or row.get("lot"),
                            "tick_size": row.get("tick_size") or row.get("ticksize"),
                            "raw": dict(row),
                        }
                        self._option_contracts[base].append(contract)

                self._clear_negative_cache_for_key(tradingsymbol)
        except Exception:
            LOGGER.exception("Failure in InstrumentResolver._ingest_instrument_row", extra={"event": "instrument_resolver_ingest_error", "row_repr": str(dict(row))})
            raise

    def _seed_well_known(self) -> None:
        """Populate resolver caches with baked-in fallbacks (indices etc)."""
        with self._lock:
            for key, value in WELL_KNOWN.items():
                try:
                    token_int = int(value)
                except (TypeError, ValueError) as exc:
                    LOGGER.error("Failure in InstrumentResolver._seed_well_known: %s", exc, extra={"event": "instrument_resolver_seed_error", "key": key, "value": value})
                    continue
                normalized_key = key.upper()
                alias_symbol = normalized_key.split(":", 1)[-1]
                exchange = (normalized_key.split(":", 1)[0] if ":" in normalized_key else "NSE")
                self._by_symbol.setdefault(normalized_key, token_int)
                if ":" not in normalized_key:
                    self._by_symbol.setdefault(f"{exchange}:{alias_symbol}", token_int)
                self._by_symbol.setdefault(alias_symbol, token_int)
                self._symbol_by_token.setdefault(token_int, alias_symbol)
                self._exchange_by_token.setdefault(token_int, exchange)
                self._clear_negative_cache_for_key(alias_symbol)
            for token, canonical_symbol in CANONICAL_TOKENS.items():
                try:
                    t_int = int(token)
                except Exception:
                    continue
                self._symbol_by_token.setdefault(t_int, canonical_symbol)
                self._exchange_by_token.setdefault(t_int, "NSE")
                self._clear_negative_cache_for_key(canonical_symbol)

    # -------------------- token search helpers --------------------------------
    def _search_token_via_broker_or_store(self, symbol: str) -> Optional[int]:
        """
        Best-effort search via broker helpers or list_instruments fallback.
        """
        if self._broker is None:
            return None

        # Direct helper
        direct = getattr(self._broker, "instrument_token_for", None)
        if callable(direct):
            try:
                token = direct(symbol)
                if token:
                    return int(token)
            except Exception:
                LOGGER.debug("broker.instrument_token_for failed", exc_info=True)

        # finder-style helper
        finder = getattr(self._broker, "find_instrument", None)
        if callable(finder):
            try:
                ins = finder(symbol)
                token = self._extract_token_from_instrument(ins)
                if token:
                    return int(token)
            except Exception:
                LOGGER.debug("broker.find_instrument failed", exc_info=True)

        # search_instruments
        search = getattr(self._broker, "search_instruments", None)
        if callable(search):
            try:
                hits = list(search(symbol))
                for ins in hits:
                    token = self._extract_token_from_instrument(ins)
                    if token:
                        return int(token)
            except Exception:
                LOGGER.debug("broker.search_instruments failed", exc_info=True)

        # list_instruments fallback: iterate & match heuristics
        list_fn = getattr(self._broker, "list_instruments", None)
        if callable(list_fn):
            try:
                for ins in list_fn():
                    token = self._extract_token_from_instrument(ins)
                    s = (ins.get("tradingsymbol") or ins.get("symbol") or "").strip().upper()
                    if token and (s == symbol or s.endswith(symbol) or symbol.endswith(s)):
                        return int(token)
            except Exception:
                LOGGER.debug("broker.list_instruments fallback failed", exc_info=True)

        return None

    @staticmethod
    def _extract_token_from_instrument(ins: Any) -> Optional[int]:
        if not isinstance(ins, Mapping):
            return None
        token = ins.get("instrument_token") or ins.get("token") or ins.get("instrumentToken")
        if token is None:
            return None
        try:
            return int(float(token))
        except Exception:
            return None

    @staticmethod
    def _base_index_from_tradingsymbol(tradingsymbol: str) -> Optional[str]:
        """
        Heuristic to detect base index from tradingsymbol string,
        e.g. "NIFTY25OCT25900CE" -> "NIFTY"
        """
        up = tradingsymbol.upper().replace(" ", "")
        if up.startswith("NIFTY"):
            return "NIFTY"
        if up.startswith("BANKNIFTY"):
            return "BANKNIFTY"
        return None

    @staticmethod
    def _normalize_option_symbol(symbol: str) -> Tuple[Optional[str], str]:
        """
        Return (exchange, tradingsymbol) for option contracts.
        Raises BrokerError if invalid.
        """
        if not symbol:
            raise BrokerError("Symbol required for option resolution")
        raw = symbol.strip().upper()
        if ":" in raw:
            prefix, remainder = raw.split(":", 1)
            exchange = prefix or None
            tradingsymbol = remainder.strip()
        else:
            exchange = None
            tradingsymbol = raw
        tradingsymbol = tradingsymbol.replace(" ", "")
        if not tradingsymbol:
            raise BrokerError("Tradingsymbol missing for option resolution")
        if tradingsymbol.endswith("FUT"):
            raise BrokerError("Futures disabled for this bot")
        if not (tradingsymbol.endswith("CE") or tradingsymbol.endswith("PE")):
            raise BrokerError("Only NIFTY options (CE/PE) are allowed")
        return exchange, tradingsymbol


# ------------------------- CSV / SQLite helpers -------------------------------
def ensure_sqlite(path: str | Path) -> sqlite3.Connection:
    """
    Ensure a SQLite DB exists at path and has the instruments table.
    Returns a sqlite3.Connection (caller should .close()).
    """
    db_path = str(path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        with closing(conn.cursor()) as cur:
            cur.executescript(_SQLITE_SCHEMA)
            conn.commit()
    except Exception:
        LOGGER.exception("ensure_sqlite failed to create schema", extra={"event": "ensure_sqlite_error", "path": db_path})
        raise
    return conn


def refresh_from_csv(conn: sqlite3.Connection, csv_path: str | Path) -> Dict[str, Any]:
    """
    Read an instruments CSV and persist option rows to the sqlite DB.

    CSV expected columns (common): instrument_token, exchange, tradingsymbol, lot_size, expiry, strike, instrument_type, tick_size
    Returns a summary dict.
    """
    csv_path = str(csv_path)
    summary: Dict[str, Any] = {"stored": 0, "skipped": 0, "errors": 0}
    p = Path(csv_path)
    if not p.exists():
        LOGGER.warning("refresh_from_csv: csv path does not exist: %s", csv_path, extra={"event": "refresh_from_csv_missing", "path": csv_path})
        return summary

    conn_rowcount = 0
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except Exception:
        LOGGER.exception("Failed to read instruments CSV", extra={"event": "refresh_from_csv_read_error", "path": csv_path})
        summary["errors"] = 1
        return summary

    try:
        with closing(conn.cursor()) as cur:
            for row in rows:
                try:
                    token = row.get("instrument_token") or row.get("token")
                    if token is None:
                        summary["skipped"] += 1
                        continue
                    token_int = int(float(token))
                    tradingsymbol = (row.get("tradingsymbol") or row.get("symbol") or "").strip()
                    if not tradingsymbol:
                        summary["skipped"] += 1
                        continue
                    exchange = (row.get("exchange") or "").strip().upper() or None
                    lot_size = row.get("lot_size") or row.get("lot") or None
                    try:
                        lot_int = int(lot_size) if lot_size not in (None, "", "NULL") else None
                    except Exception:
                        lot_int = None
                    expiry = row.get("expiry") or None
                    strike = row.get("strike") or None
                    try:
                        strike_f = float(strike) if strike not in (None, "", "NULL") else None
                    except Exception:
                        strike_f = None
                    instrument_type = (row.get("instrument_type") or row.get("instrumentType") or "").strip().upper()
                    tick_size = row.get("tick_size") or row.get("ticksize") or None
                    try:
                        tick_f = float(tick_size) if tick_size not in (None, "", "NULL") else None
                    except Exception:
                        tick_f = None
                    raw_json = None
                    updated_at = datetime.now(timezone.utc).isoformat()
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO instruments (
                            instrument_token, exchange, tradingsymbol, lot_size, tick_size, expiry, strike, instrument_type, raw_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (token_int, exchange, tradingsymbol, lot_int, tick_f, expiry, strike_f, instrument_type, raw_json, updated_at),
                    )
                    conn_rowcount += 1
                except Exception:
                    summary["errors"] += 1
                    LOGGER.exception("refresh_from_csv: failed to insert row: %s", row)
            conn.commit()
    except Exception:
        LOGGER.exception("refresh_from_csv: DB write failed", extra={"event": "refresh_from_csv_db_error", "path": csv_path})
        summary["errors"] += 1
    summary["stored"] = conn_rowcount
    LOGGER.info("refresh_from_csv complete: stored=%d errors=%d skipped=%d", summary["stored"], summary["errors"], summary["skipped"], extra={"event": "refresh_from_csv_complete", "path": csv_path})
    return summary


def load_rows_for_resolver(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Load instrument rows from sqlite database and return iterable of dict-like rows suitable
    for passing to resolver.warm_from_broker_dump(rows).

    By default returns only option contracts (CE/PE) and known indices (keep DB lightweight).
    """
    rows: List[Dict[str, Any]] = []
    try:
        with closing(conn.cursor()) as cur:
            cur.execute("SELECT instrument_token, exchange, tradingsymbol, lot_size, tick_size, expiry, strike, instrument_type, updated_at FROM instruments")
            fetched = cur.fetchall()
            for r in fetched:
                d = {
                    "instrument_token": int(r["instrument_token"]) if r["instrument_token"] is not None else None,
                    "exchange": r["exchange"],
                    "tradingsymbol": r["tradingsymbol"],
                    "lot_size": r["lot_size"],
                    "tick_size": r["tick_size"],
                    "expiry": r["expiry"],
                    "strike": r["strike"],
                    "instrument_type": r["instrument_type"],
                    "updated_at": r["updated_at"],
                }
                t = (d.get("instrument_type") or "").upper() if d.get("instrument_type") else ""
                tsym = (d.get("tradingsymbol") or "").upper()
                # Only include option contracts and those index names we care about
                if tsym.endswith("CE") or tsym.endswith("PE") or tsym in ("NIFTY", "BANKNIFTY"):
                    rows.append(d)
    except Exception:
        LOGGER.exception("load_rows_for_resolver failed", extra={"event": "load_rows_for_resolver_error"})
    LOGGER.info("load_rows_for_resolver loaded %d rows", len(rows), extra={"event": "load_rows_for_resolver_complete"})
    return rows
