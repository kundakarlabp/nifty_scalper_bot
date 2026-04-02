# src/nifty_scalper_bot/data/instruments.py
"""
Instrument resolver and CSV/SQLite instrument loader.

Provides:
- InstrumentResolver: broker/CSV/DB-backed instrument token resolver.
- ensure_sqlite: create/open a sqlite db for cached instruments.
- refresh_from_csv: read instruments CSV and persist into sqlite.
- load_rows_for_resolver: return rows (mapping) usable by resolver.warm_from_broker_dump.

Optimized / hardened version:
- Thread-safe caches, defensive logging.
- Option contract catalog accessible via option_contracts(base, force_refresh=False).
- Dynamic lot_size resolution via get_lot_size(symbol_or_base).
- Expiry parsing / weekly-monthly helpers (includes O/N/D month codes).
- Negative cache TTL configurable via env.
"""
from __future__ import annotations

import csv
import calendar
import math
import os
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from nifty_scalper_bot.config.paths import get_data_dir
from nifty_scalper_bot.config.settings import get_settings
from nifty_scalper_bot.utils.symbols import canonical

import logging

LOGGER = logging.getLogger("nifty_scalper_bot.data.instruments")

# --- Configurable constants -------------------------------------------------
_DEFAULT_NEG_TTL = 300.0  # seconds - negative cache TTL
_NEG_TTL = float(os.getenv("NSB__INSTRUMENT_NEG_TTL", _DEFAULT_NEG_TTL))

# Well-known fallback tokens for indexes (kept minimal; can be extended)
WELL_KNOWN: Dict[str, int] = {
    "NIFTY": 256265,  # example canonical token
    "BANKNIFTY": 260105,
    "NSE:NIFTY": 256265,
    "NSE:BANKNIFTY": 260105,
}

# Canonical human-readable names for format_token_as_symbol
CANONICAL_TOKENS: Dict[int, str] = {
    256265: "NIFTY",
    260105: "BANKNIFTY",
}

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

# Month code mapping for weekly option single-character month code
_WEEKLY_MONTH_MAP: Dict[int, str] = {10: "O", 11: "N", 12: "D"}


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


@dataclass(frozen=True, slots=True)
class OptionContract:
    """Resolved option contract model. Args: tradingsymbol, expiry, strike. Returns: OptionContract. Raises: None."""

    tradingsymbol: str
    expiry: date
    strike: float


@dataclass(frozen=True, slots=True)
class ResolvedSymbol:
    """Authoritative standardized instrument object."""
    tradingsymbol: str
    instrument_token: int
    expiry: Optional[date]
    lot_size: int
    instrument_type: str  # FUT, CE, PE, SPOT, INDEX


class BrokerError(Exception):
    """Generic broker/resolver error sentinel."""


def atm_strike_for_spot(spot: float, step: int) -> int:
    """Return the nearest ATM strike for a spot price (Args/Returns/Raises)."""
    try:
        if spot <= 0:
            raise ValueError("spot must be positive")
        if step <= 0:
            raise ValueError("step must be positive")
        return int(round(spot / step) * step)
    except Exception as exc:
        LOGGER.error("Failure in atm_strike_for_spot: %s", exc)
        raise


def strike_for_delta(
    spot: float,
    vol: float,
    ttm: float,
    target_delta: float,
    *,
    call: bool = True,
    rate: float = 0.06,
    max_iterations: int = 60,
) -> float:
    """Return an approximate strike that matches a target delta (Args/Returns/Raises)."""
    try:
        if spot <= 0:
            raise ValueError("spot must be positive")
        if vol <= 0:
            raise ValueError("vol must be positive")
        if ttm <= 0:
            raise ValueError("ttm must be positive")
        if not (0 < target_delta < 1):
            raise ValueError("target_delta must be between 0 and 1")

        from nifty_scalper_bot.utils.options_math import black_scholes_greeks

        low = spot * 0.5
        high = spot * 1.5

        for _ in range(max_iterations):
            mid = (low + high) / 2.0
            greeks = black_scholes_greeks(spot, mid, ttm, rate, vol, is_call=call)
            delta = abs(greeks["delta"]) if not call else greeks["delta"]

            if call:
                if delta > target_delta:
                    low = mid
                else:
                    high = mid
            else:
                if delta > target_delta:
                    high = mid
                else:
                    low = mid

        return (low + high) / 2.0
    except Exception as exc:
        LOGGER.error("Failure in strike_for_delta: %s", exc)
        raise


# --- InstrumentResolver ----------------------------------------------------
class InstrumentResolver:
    """
    Caches broker instruments and provides reliable symbol resolution for the rest of the bot.

    Construction:
        resolver = InstrumentResolver(broker_client)

    Public methods:
      - warm()
      - warm_from_broker_dump(rows)
      - upsert(symbol, token, exchange=...)
      - resolve(symbol_or_token)
      - lookup(symbol_or_token)
      - resolve_exchange(symbol)
      - option_contracts(base, force_refresh=False)
      - get_lot_size(symbol_or_base)
    """

    def __init__(self, broker_client: Any | None = None) -> None:
        self._broker = broker_client

        # Main caches (upper-cased keys)
        self._by_symbol: Dict[str, int] = {}  # e.g. "NFO:NIFTY25OCT25900CE"
        self._symbol_by_token: Dict[int, str] = {}  # token -> base tradingsymbol
        self._exchange_by_token: Dict[int, str] = {}  # token -> exchange

        # Lightweight option contract catalogue: base -> [contracts]
        # Each contract is a dict with minimal fields (instrument_token, tradingsymbol, expiry, strike, option_type, lot_size, tick_size, raw)
        self._option_contracts: Dict[str, List[Dict[str, Any]]] = {}
        self._future_contracts: Dict[str, List[Dict[str, Any]]] = {}

        # Negative cache (key -> expiry_ts) and a warned set to avoid noisy logs
        self._neg_cache: Dict[str, float] = {}
        self._warned_no_token: set[str] = set()
        self._neg_ttl = _NEG_TTL

        # threading guard for caches
        self._lock = threading.RLock()
        self._warmed = False

        # seed well-known tokens
        # self._seed_well_known()

    # ------------------------- public API ---------------------------------
    # ------------------------- public API ---------------------------------
    def warm(self, cache_path: str = "instruments.csv", force: bool = False) -> None:
        """
        Smart Warmup:
        1. Checks if local 'instruments.csv' is fresh (< 24h). If yes, loads it.
        2. If stale/missing, fetches from broker and saves to 'instruments.csv'.
        3. If broker fails (Rate Limit), falls back to stale cache to prevent crash.
        """
        LOGGER.debug("InstrumentResolver.warm() entered", extra={"event": "instrument_resolver_warm_enter"})
        
        path_obj = get_data_dir() / Path(cache_path).name
        is_fresh = False

        # 1. Check Cache Freshness
        if path_obj.exists() and not force:
            try:
                mtime = path_obj.stat().st_mtime
                age_hours = (time.time() - mtime) / 3600.0
                if age_hours < 24:
                    is_fresh = True
                    LOGGER.info("✅ Cache is fresh (Age: %.1fh). Loading local data...", age_hours)
                    if self._try_load_csv(path_obj):
                        self._seed_well_known()
                        return
            except Exception as e:
                LOGGER.warning("Cache check failed: %s", e)

        # 2. Fetch from Broker (if cache is stale or missing)
        LOGGER.info("⬇️ Fetching fresh instruments from broker...")
        items: Optional[Iterable[Mapping[str, Any]]] = None
        fetch_success = False

        try:
            # Try all known broker methods
            for name in ("list_instruments", "get_instruments", "instruments", "load_instruments", "fetch_instruments"):
                fn = getattr(self._broker, name, None)
                if callable(fn):
                    items = fn()
                    if items:
                        break
            
            if items:
                fetch_success = True
                count = 0
                # Process and Save to Cache
                to_save = []
                with self._lock:
                    for row in items:
                        if not isinstance(row, Mapping): continue
                        self._ingest_instrument_row(row)
                        to_save.append(row)
                        count += 1
                
                LOGGER.info("InstrumentResolver: warmed from broker with %d rows", count, extra={"event": "instrument_resolver_warm_broker"})
                self._save_csv(path_obj, to_save)
            else:
                LOGGER.warning("Broker returned no instruments.")

        except Exception as exc:
            # 3. CRITICAL FALLBACK: If broker fails (Rate Limit), use stale cache
            LOGGER.error("❌ Broker fetch failed (Rate Limit?): %s", exc)
            if path_obj.exists():
                LOGGER.warning("⚠️ ACTIVATING FALLBACK: Using stale cache to keep bot alive.")
                if self._try_load_csv(path_obj):
                    LOGGER.info("✅ Fallback successful.")
                    self._seed_well_known()
                    return

        # Finalize
        self._seed_well_known()
        self._warmed = True
        LOGGER.info("resolver_warm_loaded", extra={"event": "resolver_warm_loaded", "symbols": len(self._by_symbol)})
        LOGGER.info("InstrumentResolver ready with %d symbols", len(self._by_symbol), extra={"event": "instrument_resolver_ready"})

    # --- Helper Methods (Add these to the class) ---
    
    def _try_load_csv(self, path_obj: Path) -> bool:
        """Helper to load rows from CSV file safely."""
        try:
            with open(path_obj, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.warm_from_broker_dump(rows)
            return True
        except Exception as e:
            LOGGER.error("Failed to load CSV cache: %s", e)
            return False

    def _save_csv(self, path_obj: Path, rows: List[Any]) -> None:
        """Helper to save broker rows to CSV."""
        if not rows: return
        try:
            # Get fieldnames from first valid row
            fieldnames = list(rows[0].keys())
            tmp_file = path_obj.with_suffix(".tmp")
            with open(tmp_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            tmp_file.replace(path_obj)
            LOGGER.info("Saved instruments to %s", path_obj)
        except Exception as e:
            LOGGER.error("Failed to save CSV cache: %s", e)

    def warm_from_broker_dump(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """
        Warm caches directly from caller-supplied rows (CSV/DB dumps).
        """
        LOGGER.debug("InstrumentResolver.warm_from_broker_dump entered", extra={"event": "instrument_resolver_warm_dump_enter"})
        if rows is None:
            return
        with self._lock:
            self._future_contracts = {}
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                try:
                    self._ingest_instrument_row(row)
                except Exception:
                    LOGGER.exception("instrument_resolver_ingest_error for row")
            self._finalize_futures()
            self._seed_well_known()
            self._warmed = True
        LOGGER.info("resolver_warm_loaded", extra={"event": "resolver_warm_loaded", "symbols": len(self._by_symbol)})
        LOGGER.info("InstrumentResolver warmed from dump: symbols=%d", len(self._by_symbol), extra={"event": "instrument_resolver_warm_dump_complete"})

    def upsert(self, symbol: str, token: int, *, exchange: Optional[str] = None) -> None:
        """
        Insert or refresh resolver caches for symbol -> token mapping.
        Accepts either 'NFO:SYMBOL' or 'SYMBOL'.
        """
        if not symbol:
            LOGGER.debug("InstrumentResolver.upsert called with blank symbol")
            return
        try:
            normalized = str(symbol).strip().upper()
            token_int = int(token)
            base_symbol = normalized.split(":", 1)[-1] or normalized
            exchange_hint = (exchange or "").strip().upper() or None
            with self._lock:
                # Store exchange-prefixed, bare and base-only forms for flexible lookups
                self._by_symbol[normalized] = token_int
                self._by_symbol[base_symbol] = token_int
                if exchange_hint:
                    self._by_symbol[f"{exchange_hint}:{base_symbol}"] = token_int
                    self._exchange_by_token[token_int] = exchange_hint
                self._symbol_by_token[token_int] = base_symbol
                # clear any negative/no-token state
                self._clear_negative_cache_for_key(base_symbol)
            LOGGER.debug("Condition met: instrument_resolver_upsert", extra={"event": "instrument_resolver_upsert", "symbol": base_symbol, "token": token_int, "exchange": exchange_hint or ""})
        except Exception:
            LOGGER.exception("Failure in InstrumentResolver.upsert")

    def resolve(self, symbol: str | int | None) -> Optional[int]:
        """
        Resolve a symbol (or token-like input) to an integer instrument token.
        """
        if symbol is None:
            return None

        # numeric direct
        try:
            if isinstance(symbol, (int, float)):
                return int(symbol)
            if isinstance(symbol, str) and symbol.strip().isdigit():
                return int(symbol.strip())
        except Exception as e:
            __import__("logging").getLogger(__name__).exception("[CRITICAL] unhandled exception", exc_info=True)
            raise

        key = str(symbol).strip().upper()
        if not key:
            return None

        now_ts = time.time()
        with self._lock:
            exp = self._neg_cache.get(key)
            if exp and exp > now_ts:
                LOGGER.debug("instrument_resolver_negative_cache_hit", extra={"event": "instrument_resolver_negative_cache", "symbol": key})
                return None

        with self._lock:
            if key in self._by_symbol:
                return int(self._by_symbol[key])
            base = key.split(":", 1)[-1]
            if base in self._by_symbol:
                return int(self._by_symbol[base])

        # try broker or search heuristics
        try:
            token = self._search_token_via_broker_or_store(key)
            if token:
                with self._lock:
                    self._by_symbol[key] = int(token)
                    self._symbol_by_token.setdefault(int(token), key.split(":", 1)[-1])
                return int(token)
        except Exception:
            LOGGER.exception("broker_token_search_error", extra={"event": "instrument_resolver_broker_search_error", "symbol": key})

        # negative cache miss
        with self._lock:
            self._neg_cache[key] = now_ts + self._neg_ttl
            self._warned_no_token.add(key.split(":", 1)[-1])
        if token is None:
            # Log at WARNING in both live and paper mode.
            # In live mode we previously raised RuntimeError which propagated as an
            # unhandled exception during startup symbol resolution, crashing the
            # instrument-load loop and leaving some symbols without tokens.
            # Callers already handle None return (skip or fallback) — raising here
            # is overly destructive.  The negative cache prevents log spam.
            LOGGER.warning(
                "instrument_resolver_no_token",
                extra={
                    "event": "instrument_resolver_no_token",
                    "symbol": key,
                    "live_mode": get_settings().enable_live,
                },
            )
        return None

    # compatibility wrapper
    def resolve_symbol_to_token(self, symbol: str) -> Optional[int]:
        try:
            return self.resolve(symbol)
        except Exception:
            LOGGER.exception("instrument_resolver_token_error", extra={"event": "instrument_resolver_token_error", "symbol": symbol})
            return None

    def lookup(self, symbol: str | int | None) -> Optional[Dict[str, Any]]:
        """
        Return instrument metadata (token, exchange, symbol) for a symbol when available.
        """
        if symbol is None:
            return None
        try:
            normalized = str(symbol).strip().upper()
            strip_prefix = os.getenv("RESOLVER_STRIP_EXCHANGE_PREFIX", "true").lower() == "true"
            if strip_prefix and ":" in normalized:
                normalized = normalized.split(":", 1)[-1].strip()

            # numeric token path
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
        """Return the exchange required for an option symbol; raise BrokerError if invalid."""
        exchange, _tradingsymbol = self._normalize_option_symbol(symbol)
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
        """Build canonical symbol and candidate quote keys for broker lookups."""
        canonical_symbol = canonical(str(symbol))
        if ":" in canonical_symbol:
            _pref, canonical_symbol = canonical_symbol.split(":", 1)
        canonical_symbol = canonical_symbol.replace(" ", "")
        candidates: List[str] = []
        if canonical_symbol.endswith(("CE", "PE", "FUT")):
            candidates.append(f"NFO:{canonical_symbol}")
        else:
            candidates.append(f"NSE:{canonical_symbol}")
        candidates.append(canonical_symbol)
        return canonical_symbol, candidates

    def format_token_as_symbol(self, token: int) -> str:
        """Format token into a canonical exchange-qualified symbol."""
        try:
            token_int = int(token)
            canonical_symbol = CANONICAL_TOKENS.get(token_int)
            exchange = self._exchange_by_token.get(token_int, "NSE")
            if canonical_symbol:
                return canonical(f"{exchange}:{canonical_symbol}")
            base = self._symbol_by_token.get(token_int)
            if base:
                return canonical(f"{exchange}:{base}")
            return str(token)
        except Exception:
            return str(token)

    def candidates_for_quote(self, raw: str) -> Tuple[str, ...]:
        """
        Helper: build candidate keys suitable for many broker.quote_any() functions.
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
        # unique preserve order
        seen = set()
        ordered = []
        for k in out:
            if k and k not in seen:
                ordered.append(k)
                seen.add(k)
        return tuple(ordered)

    # ------------------------- small helpers --------------------------------
    def _clear_negative_cache_for_key(self, symbol: str) -> None:
        base = symbol.split(":", 1)[-1].strip().upper()
        with self._lock:
            for k in (symbol.strip().upper(), base):
                if k in self._neg_cache:
                    self._neg_cache.pop(k, None)
            self._warned_no_token.discard(base)

    def _ingest_instrument_row(self, row: Mapping[str, Any]) -> None:
        """
        Load a single instrument row into resolver caches.

        Accepts flexible keys: 'tradingsymbol'|'symbol', 'instrument_token'|'token', 'exchange',
        'lot_size'|'lot', 'strike', 'expiry', 'instrument_type' etc.
        """
        try:
            tradingsymbol = row.get("tradingsymbol")
            instrument_token = row.get("instrument_token")
            exchange = row.get("exchange")
            itype = (row.get("instrument_type") or "").upper()

            # HARD GUARD — skip invalid rows
            if not tradingsymbol or not instrument_token:
                return

            try:
                token_int = int(instrument_token)
            except (TypeError, ValueError):
                return

            key = str(tradingsymbol).upper()
            ts_upper = key
            if not (ts_upper.startswith("NIFTY") or ts_upper.startswith("BANKNIFTY")):
                return
            tradingsymbol = str(tradingsymbol).strip()
            exchange = str(exchange).strip().upper() if exchange else None

            with self._lock:
                # primary keys
                self._by_symbol[key] = token_int
                if exchange:
                    self._by_symbol[f"{exchange}:{key}"] = token_int
                base_key = key.split(":", 1)[-1] if ":" in key else key
                if base_key:
                    self._by_symbol[base_key] = token_int

                # map token back to symbol and exchange if absent
                self._symbol_by_token.setdefault(token_int, tradingsymbol)
                if exchange:
                    self._exchange_by_token.setdefault(token_int, exchange)

                # record option and future contract metadata for lightweight lookups by base
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
                elif itype in ("FUT", "FUTURE"):
                    base = self._base_index_from_tradingsymbol(tradingsymbol)
                    if not base:
                        return

                    # HARD RESET — ensures only latest contracts exist per ingestion cycle
                    if base not in self._future_contracts:
                        self._future_contracts[base] = []

                    contract = {
                        "instrument_token": token_int,
                        "tradingsymbol": tradingsymbol,
                        "instrument_type": "FUT",
                        "expiry": row.get("expiry"),
                        "lot_size": row.get("lot_size") or row.get("lot"),
                        "tick_size": row.get("tick_size") or row.get("ticksize"),
                        "raw": dict(row),
                    }

                    self._future_contracts[base].append(contract)

                # clear negative cache for this symbol
                self._clear_negative_cache_for_key(tradingsymbol)
                self._clear_negative_cache_for_key(key)
                self._clear_negative_cache_for_key(f"NFO:{key}")
        except Exception as e:
            LOGGER.error(f"INGEST FAIL: {row} | ERROR: {e}")
            LOGGER.exception("Failure in InstrumentResolver._ingest_instrument_row")
            raise

    def select_tokens_for_universe(
        self,
        base: str = "NIFTY",
        spot_price: float | None = None,
        strikes_around_atm: int = 2,
        strike_step: int = 50,
    ) -> list[int]:
        """
        Select a comprehensive list of tokens for the trading universe.
        Includes:
        - Index Spot (from WELL_KNOWN)
        - Nearest Future
        - ATM Option Strikes (+/- strikes_around_atm) for nearest expiry
        """
        tokens: set[int] = set()
        symbols_selected: list[str] = []

        # 1. Spot Token
        spot_token = WELL_KNOWN.get(base)
        if spot_token:
            tokens.add(spot_token)
            symbols_selected.append(f"{base} Spot")

        # 2. Nearest Future
        with self._lock:
            futures = self._future_contracts.get(base, [])
            if futures:
                # _finalize_futures already ensures only the nearest is kept
                fut_tok = futures[0]["instrument_token"]
                tokens.add(fut_tok)
                symbols_selected.append(f"{base} Future")

        # 3. ATM Options
        if spot_price and spot_price > 0:
            atm_strike = atm_strike_for_spot(spot_price, strike_step)
            # Ensure at least 5 strikes are selected (2 around ATM = 5 total strikes)
            actual_around = max(2, strikes_around_atm)
            strikes = [
                atm_strike + (i * strike_step)
                for i in range(-actual_around, actual_around + 1)
            ]

            all_options = self.option_contracts(base)
            if all_options:
                # Find nearest expiry
                expiries = sorted({
                    self.parse_expiry_string(o["expiry"])
                    for o in all_options
                    if self.parse_expiry_string(o["expiry"]) and self.parse_expiry_string(o["expiry"]) >= date.today()
                })
                
                if expiries:
                    nearest_expiry = expiries[0]
                    option_count = 0
                    for o in all_options:
                        o_expiry = self.parse_expiry_string(o["expiry"])
                        if o_expiry == nearest_expiry and o["strike"] in strikes:
                            tokens.add(o["instrument_token"])
                            symbols_selected.append(o["tradingsymbol"])
                            option_count += 1
                    
                    LOGGER.info(
                        "Universe selection: spot=%.2f expiry=%s strikes=%d options=%d",
                        spot_price,
                        nearest_expiry,
                        len(strikes),
                        option_count,
                        extra={"event": "universe_selection", "tokens": list(tokens), "symbols": symbols_selected}
                    )

        return sorted(list(tokens))

    def _finalize_futures(self) -> None:
        """
        Ensure only the nearest valid expiry FUT is retained per index. Args: None. Returns: None. Raises: None.
        """
        for base, contracts in list(self._future_contracts.items()):
            valid: List[Dict[str, Any]] = []

            for contract in contracts:
                expiry = contract.get("expiry")
                if not expiry:
                    continue
                expiry_dt = self.parse_expiry_string(expiry)
                if expiry_dt and expiry_dt >= date.today():
                    valid.append(contract)

            if not valid:
                continue

            nearest = min(valid, key=lambda value: self.parse_expiry_string(value.get("expiry")) or date.max)
            self._future_contracts[base] = [nearest]

            ts = str(nearest["tradingsymbol"]).upper()
            tok = int(nearest["instrument_token"])
            self._by_symbol[ts] = tok
            self._by_symbol[f"NFO:{ts}"] = tok
            self._clear_negative_cache_for_key(ts)
            self._clear_negative_cache_for_key(f"NFO:{ts}")
            LOGGER.info("Futures finalized: %s -> %s", base, ts)

    def _seed_well_known(self) -> None:
        """Populate resolver caches with baked-in fallbacks (indices etc)."""
        with self._lock:
            for key, value in WELL_KNOWN.items():
                try:
                    token_int = int(value)
                except (TypeError, ValueError) as exc:
                    LOGGER.error("Failure in InstrumentResolver._seed_well_known: %s", exc, extra={"key": key, "value": value})
                    continue
                normalized_key = key.upper()
                alias_symbol = normalized_key.split(":", 1)[-1]
                exchange = (normalized_key.split(":", 1)[0] if ":" in normalized_key else "NSE")
                self._by_symbol[normalized_key] = token_int
                if ":" not in normalized_key:
                    self._by_symbol[f"{exchange}:{alias_symbol}"] = token_int
                self._by_symbol[alias_symbol] = token_int
                self._symbol_by_token.setdefault(token_int, alias_symbol)
                self._exchange_by_token.setdefault(token_int, exchange)
                self._clear_negative_cache_for_key(alias_symbol)
            # canonical token names (human readable)
            for token, canonical_symbol in CANONICAL_TOKENS.items():
                try:
                    t_int = int(token)
                except Exception:
                    continue
                self._symbol_by_token.setdefault(t_int, canonical_symbol)
                self._exchange_by_token.setdefault(t_int, "NSE")
                self._clear_negative_cache_for_key(canonical_symbol)

    # ------------------------- token search helpers -------------------------
    def _search_token_via_broker_or_store(self, symbol: str) -> Optional[int]:
        """
        Try best-effort search on broker or static store to obtain an instrument token.
        """
        if self._broker is None:
            return None

        # 1) Direct helper
        direct = getattr(self._broker, "instrument_token_for", None)
        if callable(direct):
            try:
                token = direct(symbol)
                if token:
                    return int(token)
            except Exception:
                LOGGER.debug("broker.instrument_token_for failed", exc_info=True)

        # 2) Broker find_instrument style
        finder = getattr(self._broker, "find_instrument", None)
        if callable(finder):
            try:
                ins = finder(symbol)
                token = self._extract_token_from_instrument(ins)
                if token:
                    return int(token)
            except Exception:
                LOGGER.debug("broker.find_instrument failed", exc_info=True)

        # 3) search_instruments
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

        # 4) list_instruments fallback (iterate and match)
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

    # ------------------------- New helper APIs -----------------------------

    def ensure_core_index_tokens(self) -> None:
        """
        Guarantees that canonical index tokens (NIFTY, BANKNIFTY) are present
        in the resolver cache, regardless of broker dump completeness.
        """
        for symbol, token in WELL_KNOWN.items():
            try:
                # Use the dedicated upsert method to ensure symbol, token, and exchange (NSE) are mapped
                self.upsert(
                    symbol=symbol,
                    token=token,
                    exchange=symbol.split(":", 1)[0] if ":" in symbol else "NSE",
                )
                LOGGER.debug(
                    "Fixed missing core index token",
                    extra={"event": "instrument_resolver_core_fix", "symbol": symbol}
                )
            except Exception as exc:
                LOGGER.error(
                    "Failure to upsert well-known token %s: %s",
                    symbol,
                    exc,
                    extra={"event": "instrument_resolver_core_fix_error"},
                )
        # Ensure the underlying symbol caches are primed for all well-known entries
        self._seed_well_known()

    
    def option_contracts(self, base: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Return lightweight option contract metadata for base (e.g., 'NIFTY' or 'BANKNIFTY').

        When force_refresh=True this currently just rebuilds from in-memory cache (CSV/DB ingestion is external).
        """
        key = (base or "").strip().upper()
        if not key:
            return []
        with self._lock:
            lst = list(self._option_contracts.get(key) or [])
        # return shallow copy to avoid accidental external mutation
        return [dict(item) for item in lst]

    def get_option_contracts(self, underlying: str) -> List[OptionContract]:
        """Return normalized option contracts. Args: underlying. Returns: OptionContract list. Raises: None."""
        try:
            contracts = self.option_contracts(underlying)
            normalized: list[OptionContract] = []
            for contract in contracts:
                tradingsymbol = str(contract.get("tradingsymbol") or "").strip().upper()
                if not tradingsymbol:
                    continue
                expiry_raw = contract.get("expiry")
                expiry_value: date | None = None
                if isinstance(expiry_raw, datetime):
                    expiry_value = expiry_raw.date()
                elif isinstance(expiry_raw, date):
                    expiry_value = expiry_raw
                elif expiry_raw:
                    text = str(expiry_raw).strip()
                    if "T" in text:
                        text = text.split("T", 1)[0]
                    try:
                        expiry_value = date.fromisoformat(text)
                    except ValueError:
                        expiry_value = None
                strike_raw = contract.get("strike")
                if expiry_value is None or strike_raw in (None, ""):
                    continue
                try:
                    strike = float(strike_raw)
                except (TypeError, ValueError):
                    continue
                normalized.append(
                    OptionContract(
                        tradingsymbol=tradingsymbol,
                        expiry=expiry_value,
                        strike=strike,
                    )
                )
            return normalized
        except Exception as exc:
            LOGGER.error("Failure in get_option_contracts: %s", exc)
            return []

    def get_token(self, symbol: str | int | None) -> Optional[int]:
        """Return token for symbol. Args: symbol. Returns: token or None. Raises: RuntimeError."""
        if not self._warmed:
            return None
        token = self.resolve(symbol)
        if not token:
            if get_settings().enable_live:
                raise RuntimeError(
                    f"Resolver failed: {symbol} not found in instrument dump"
                )
            return None
        return token

    def sync_nfo_from_broker(self, instruments: list) -> int:
        """
        Sync NFO instruments from broker API response into _option_contracts.

        Populates the internal cache from broker.list_instruments("NFO").
        Uses tradingsymbol-prefix heuristic (same as _ingest_instrument_row)
        for reliable base detection regardless of CSV 'name' field quality.

        Args:
            instruments: List of instrument dicts from broker.list_instruments("NFO")

        Returns:
            Count of option contracts synced
        """
        synced = 0
        skipped = 0
        seen_tokens: set = set()  # deduplicate (cache has multi-key→same row)

        with self._lock:
            for inst in instruments:
                try:
                    ts = str(inst.get("tradingsymbol") or inst.get("symbol") or "").strip()
                    if not ts:
                        skipped += 1
                        continue

                    token = inst.get("instrument_token")
                    if not token:
                        skipped += 1
                        continue

                    # Deduplicate — _instrument_cache stores same row under multiple keys
                    token_key = str(token)
                    if token_key in seen_tokens:
                        continue
                    seen_tokens.add(token_key)

                    ts_upper = ts.upper()

                    # ✅ FIX: Use tradingsymbol suffix to detect CE/PE (reliable)
                    is_ce_pe = ts_upper.endswith("CE") or ts_upper.endswith("PE")
                    if not is_ce_pe:
                        skipped += 1
                        continue

                    # ✅ FIX: Use tradingsymbol prefix to detect base (same as _ingest_instrument_row)
                    base = self._base_index_from_tradingsymbol(ts_upper)
                    if not base:
                        # Fallback: try the name field
                        name_raw = str(inst.get("name") or "").strip().upper()
                        if name_raw in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
                            base = name_raw
                        else:
                            skipped += 1
                            continue

                    # Determine option type from suffix
                    opt_type = "CE" if ts_upper.endswith("CE") else "PE"

                    # Parse strike safely
                    strike_val = None
                    try:
                        raw_strike = inst.get("strike")
                        if raw_strike not in (None, "", "NULL"):
                            strike_val = float(raw_strike)
                    except (ValueError, TypeError):
                        strike_val = None

                    token_int = int(token)

                    contract = {
                        "instrument_token": token_int,
                        "tradingsymbol": ts,
                        "name": base,
                        "option_type": opt_type,
                        "expiry": inst.get("expiry"),
                        "strike": strike_val,
                        "instrument_type": opt_type,
                        "lot_size": inst.get("lot_size") or inst.get("lot") or 25,
                        "tick_size": inst.get("tick_size"),
                        "exchange": "NFO",
                    }

                    # Add to option_contracts cache
                    self._option_contracts.setdefault(base, []).append(contract)

                    # Also add to symbol resolution caches
                    nfo_symbol = f"NFO:{ts}"
                    self._by_symbol[nfo_symbol] = token_int
                    self._by_symbol[ts_upper] = token_int
                    self._symbol_by_token[token_int] = ts
                    self._exchange_by_token[token_int] = "NFO"

                    synced += 1

                except Exception as e:
                    skipped += 1
                    if synced == 0 and skipped <= 3:
                        LOGGER.warning(
                            "sync_nfo_from_broker skip: %s (row=%s)",
                            e,
                            str(inst)[:200],
                        )
                    continue

        # Log summary
        with self._lock:
            for base_key, contracts in self._option_contracts.items():
                if contracts:
                    LOGGER.info(
                        "sync_nfo_from_broker: %s = %d contracts", base_key, len(contracts)
                    )

        if synced == 0 and len(instruments) > 0:
            # Diagnostic: log first 3 instruments for debugging
            for i, sample in enumerate(instruments[:3]):
                LOGGER.warning(
                    "sync_nfo_from_broker 0-synced diagnostic row[%d]: "
                    "tradingsymbol=%s name=%s instrument_type=%s",
                    i,
                    sample.get("tradingsymbol"),
                    sample.get("name"),
                    sample.get("instrument_type"),
                )

        return synced


    def get_lot_size(self, symbol_or_base: str) -> Optional[int]:
        """
        Return a reasonable lot size for a given symbol or base index.

        - If symbol_or_base is an instrument token or tradingsymbol found in caches, return its lot_size.
        - Otherwise, for an index base (NIFTY/BANKNIFTY) return the most common lot_size among cached contracts.
        """
        if not symbol_or_base:
            return None
        key = str(symbol_or_base).strip().upper()
        # If numeric token
        try:
            if key.isdigit():
                token = int(key)
                with self._lock:
                    # find row in option_contracts by token
                    for base, contracts in self._option_contracts.items():
                        for c in contracts:
                            if int(c.get("instrument_token") or 0) == token:
                                lot = c.get("lot_size")
                                return int(lot) if lot not in (None, "", "NULL") else None
        except Exception as e:
            __import__("logging").getLogger(__name__).exception("[CRITICAL] unhandled exception", exc_info=True)
            raise

        # direct symbol lookup
        with self._lock:
            # exact symbol
            token = self._by_symbol.get(key)
            if token:
                # find in symbol_by_token if present in option_contracts
                for base, contracts in self._option_contracts.items():
                    for c in contracts:
                        if int(c.get("instrument_token") or 0) == int(token):
                            lot = c.get("lot_size")
                            try:
                                return int(lot) if lot not in (None, "", "NULL") else None
                            except Exception:
                                return None
            # base detection fallback
            base = key.split(":", 1)[-1]
            base = base.replace(" ", "")
            if base in ("NIFTY", "BANKNIFTY"):
                # compute modal lot size across cached contracts
                counts: Dict[int, int] = {}
                for contract in self._option_contracts.get(base, []):
                    lot = contract.get("lot_size")
                    try:
                        l = int(lot) if lot not in (None, "", "NULL") else None
                    except Exception:
                        l = None
                    if l is not None:
                        counts[l] = counts.get(l, 0) + 1
                if counts:
                    # return most common
                    return max(counts.items(), key=lambda kv: kv[1])[0]
        return None

    # ------------------------- expiry & month-code helpers -----------------
    @staticmethod
    def _week_month_code(dt: date) -> str:
        """
        Return single-character month code used by exchange weekly naming:
        months 1-9 => "1"-"9", 10->"O", 11->"N", 12->"D"
        """
        m = dt.month
        if m in _WEEKLY_MONTH_MAP:
            return _WEEKLY_MONTH_MAP[m]
        return str(m)

    @staticmethod
    def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
        """
        Return the date of the last given weekday (0=Mon..6=Sun) for month/year.
        """
        # get last day of month
        last_day = calendar.monthrange(year, month)[1]
        d = date(year, month, last_day)
        offset = (d.weekday() - weekday) % 7
        return d - timedelta(days=offset)

    @staticmethod
    def is_monthly_expiry(the_date: date, target_weekday: int = 1) -> bool:
        """
        Determine if the_date is the last target_weekday of its month.
        target_weekday default=1 (Tuesday) to reflect new 2025 convention for NIFTY.
        """
        last = InstrumentResolver._last_weekday_of_month(the_date.year, the_date.month, target_weekday)
        return the_date == last

    @staticmethod
    def parse_expiry_string(val: Any) -> Optional[date]:
        """
        Parse expiry in many common formats to a date object (UTC).
        Accepts YYYY-MM-DD, dd-mmm-YYYY, compact formats etc.
        """
        if val is None:
            return None
        if isinstance(val, date) and not isinstance(val, datetime):
            return val
        s = str(val).strip()
        if not s:
            return None
        # try ISO-ish
        fmts = (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%d-%b-%Y",
            "%d %b %Y",
            "%d-%m-%Y",
            "%d%b%Y",
            "%d%b%y",
        )
        for f in fmts:
            try:
                dt = datetime.strptime(s, f)
                return dt.date()
            except Exception:
                continue
        # fallback: try numeric timestamp
        try:
            ts = float(s)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.date()
        except Exception as e:
            __import__("logging").getLogger(__name__).exception("[CRITICAL] unhandled exception", exc_info=True)
            raise
        return None

    # ------------------------- static utilities ---------------------------
    @staticmethod
    def _coerce(val: Any) -> Any:
        if val in (None, "", "NULL"):
            return None
        return val

    # ------------------------- end InstrumentResolver ---------------------


class SymbolResolver:
    """
    Authoritative single-source-of-truth for symbol resolution.
    
    Translates generic aliases ("NIFTY_FUT", "NIFTY_OPTION") and 
    explicit symbols into standardized ResolvedSymbol objects.
    """

    def __init__(self, instrument_resolver: InstrumentResolver) -> None:
        self._resolver = instrument_resolver

    def resolve(self, query: str) -> ResolvedSymbol:
        """
        Resolve query into a standardized ResolvedSymbol.
        
        Args:
            query: Alias ("NIFTY_FUT") or explicit symbol ("NIFTY24MARFUT")
            
        Returns:
            ResolvedSymbol object
            
        Raises:
            BrokerError: If resolution fails
        """
        q = query.strip().upper()
        
        # 1. Handle Generic Aliases
        if q == "NIFTY" or q == "NSE:NIFTY":
            return self._resolve_index("NIFTY")
        if q == "BANKNIFTY" or q == "NSE:BANKNIFTY":
            return self._resolve_index("BANKNIFTY")
            
        if q == "NIFTY_FUT":
            return self._resolve_future("NIFTY")
        if q == "BANKNIFTY_FUT":
            return self._resolve_future("BANKNIFTY")
            
        if q == "NIFTY_OPTION":
            # Default to Near-Month ATM CE
            return self._resolve_default_option("NIFTY")
        if q == "BANKNIFTY_OPTION":
            return self._resolve_default_option("BANKNIFTY")

        # 2. Handle Explicit Symbols (Lookup)
        meta = self._resolver.lookup(q)
        if not meta:
            raise BrokerError(f"Could not resolve symbol: {query}")
            
        ts = str(meta["symbol"])
        token = int(meta["instrument_token"])
        
        # Determine type and details from metadata if possible
        # We search through the caches to find the full row
        lot_size = self._resolver.get_lot_size(ts) or 50
        
        # Infer instrument type
        itype = "INDEX"
        if ts.endswith("FUT"): itype = "FUT"
        elif ts.endswith("CE"): itype = "CE"
        elif ts.endswith("PE"): itype = "PE"
        elif ts in ("NIFTY", "BANKNIFTY"): itype = "INDEX"
        
        # Try to find expiry in option/future caches
        expiry_dt = None
        base = self._resolver._base_index_from_tradingsymbol(ts)
        if base:
            contracts = (self._resolver._option_contracts.get(base, []) + 
                         self._resolver._future_contracts.get(base, []))
            for c in contracts:
                if c["tradingsymbol"] == ts:
                    expiry_raw = c.get("expiry")
                    expiry_dt = self._resolver.parse_expiry_string(expiry_raw)
                    break

        return ResolvedSymbol(
            tradingsymbol=ts,
            instrument_token=token,
            expiry=expiry_dt,
            lot_size=lot_size,
            instrument_type=itype
        )

    def _resolve_index(self, base: str) -> ResolvedSymbol:
        token = WELL_KNOWN.get(base)
        if not token:
            raise BrokerError(f"Unknown index base: {base}")
        return ResolvedSymbol(
            tradingsymbol=base,
            instrument_token=token,
            expiry=None,
            lot_size=self._resolver.get_lot_size(base) or 1,
            instrument_type="INDEX"
        )

    def _resolve_future(self, base: str) -> ResolvedSymbol:
        futures = self._resolver._future_contracts.get(base, [])
        if not futures:
            raise BrokerError(f"No futures found for {base}")
            
        # Sort by expiry to find the near-month
        def get_expiry(f):
            return self._resolver.parse_expiry_string(f.get("expiry")) or date.max
            
        near_fut = min(futures, key=get_expiry)
        return ResolvedSymbol(
            tradingsymbol=near_fut["tradingsymbol"],
            instrument_token=near_fut["instrument_token"],
            expiry=get_expiry(near_fut),
            lot_size=near_fut.get("lot_size") or 50,
            instrument_type="FUT"
        )

    def _resolve_default_option(self, base: str) -> ResolvedSymbol:
        # Heuristic: Near-month ATM CE
        # This is a fallback; usually StrategyRunner uses StrikeSelector
        options = self._resolver.option_contracts(base)
        if not options:
            raise BrokerError(f"No options found for {base}")
            
        # Filter for CE and find near expiry
        ce_options = [o for o in options if o["tradingsymbol"].endswith("CE")]
        if not ce_options:
            ce_options = options # Fallback to whatever is available
            
        def get_expiry(o):
            return self._resolver.parse_expiry_string(o.get("expiry")) or date.max
            
        near_expiry = min(ce_options, key=get_expiry)
        # Narrow to that expiry and find middle strike (crude ATM approx)
        at_expiry = [o for o in ce_options if get_expiry(o) == near_expiry]
        middle_opt = sorted(at_expiry, key=lambda o: float(o.get("strike") or 0))[len(at_expiry)//2]
        
        return ResolvedSymbol(
            tradingsymbol=middle_opt["tradingsymbol"],
            instrument_token=middle_opt["instrument_token"],
            expiry=get_expiry(middle_opt),
            lot_size=middle_opt.get("lot_size") or 50,
            instrument_type=middle_opt.get("option_type", "CE")
        )

# -------------------------
# CSV / SQLite helpers
# -------------------------
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
    Returns a summary dict.
    """
    csv_path = str(csv_path)
    summary: Dict[str, Any] = {"stored": 0, "skipped": 0, "errors": 0}
    if not Path(csv_path).exists():
        LOGGER.warning("refresh_from_csv: csv path does not exist: %s", csv_path, extra={"event": "refresh_from_csv_missing", "path": csv_path})
        return summary

    conn_rowcount = 0
    settings = get_settings()
    only_index_options = bool(
        getattr(settings.instruments, "sync_only_index_options", True)
    )
    raw_filter = getattr(
        settings.instruments,
        "sync_instruments_filter",
        "NIFTY,BANKNIFTY,FINNIFTY,MIDCPNIFTY",
    )
    allowed = [
        item.strip().upper() for item in str(raw_filter).split(",") if item.strip()
    ]
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
                    if only_index_options:
                        segment = str(row.get("segment") or "").strip().upper()
                        name = str(row.get("name") or "").strip().upper()
                        if segment != "NFO-OPT":
                            summary["skipped"] += 1
                            continue
                        if not any(name.startswith(prefix) for prefix in allowed):
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
                    LOGGER.exception("refresh_from_csv: failed to insert row")
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
