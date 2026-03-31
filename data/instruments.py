# src/nifty_scalper_bot/data/instruments.py
from __future__ import annotations

import dataclasses as _dc
import functools
import logging
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

LOGGER = logging.getLogger(__name__)


@_dc.dataclass(frozen=True)
class Resolution:
    """Final resolution returned by InstrumentResolver.resolve()."""
    symbol: str                     # normalized input (e.g., NIFTY25OCT25900CE or NIFTY25N2524000CE)
    exchange: Optional[str]         # NSE / NFO / BSE (None when unknown)
    segment: Optional[str]          # 'equity' or 'derivatives' (None when unknown)
    token: Optional[int]            # broker token if found
    kind: str                       # 'index', 'equity', 'option'
    root: Optional[str] = None      # e.g., NIFTY
    right: Optional[str] = None     # CE / PE for options
    strike: Optional[float] = None  # numeric strike for options (if any)
    expiry_raw: Optional[str] = None  # raw YYMON or YYMDD (e.g., '25OCT' or '25N25') from parsed symbol
    candidates: Tuple[str, ...] = ()  # quote keys we propose for REST


class InstrumentResolver:
    """
    World-class resolver for Indian cash/derivatives instruments.

    Responsibilities
    ----------------
    - Parse raw user/bot symbols and infer: exchange, segment, type (index/equity/option)
    - Build high-quality candidate quote keys for broker REST (incl. 'NFO:' prefix)
    - Resolve broker tokens via local caches, MDM maps, or Broker search/index
    - Provide a consistent, minimal API for other subsystems (UM, MDM, Telegram)

    Dependencies (all optional)
    ---------------------------
    - mdm: MarketDataManager-like; may expose:
        .token_for_symbol(symbol) -> Optional[int]
        .symbol_for_token(token) -> Optional[str]
        .scan_instrument_token(symbol) -> Optional[int]   (if implemented)
        .get_latest_tick(symbol_or_token) -> Optional[Mapping]
        .instrument_catalog() -> Optional[Mapping[str, Any]] or iterable
    - broker: Zerodha-like client; may expose:
        .find_instrument(symbol) -> Optional[Mapping]
        .search_instruments(query) -> Iterable[Mapping]
        .instrument_token_for(symbol) -> Optional[int]
        .quote_any(keys: Iterable[str]) -> Mapping
    - store: any read-only catalog/index; if present, used before broker search

    Thread-safety: internal caches guarded by RLock.
    """

    # --- Regexes -------------------------------------------------------------
    _WS = re.compile(r"\s+")
    _EXPLICIT_EXCH = re.compile(r"^(?P<exch>[A-Za-z]{2,4}):(?P<rest>.+)$")

    # Monthly-style index option tradingsymbol (Zerodha/NSE F&O monthly style):
    #   ROOT + YY + MMM + STRIKE + (CE|PE)
    #   e.g., NIFTY25OCT25900CE
    _IDX_OPT_MONTHLY = re.compile(
        r"^(?P<root>NIFTY|BANKNIFTY|FINNIFTY|MIDCPNIFTY)"
        r"(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<strike>\d{1,6})(?P<right>CE|PE)$",
        re.IGNORECASE,
    )

    # Weekly-style index option tradingsymbol (YY + single-char month code + DD + STRIKE + CE/PE)
    # where month code is 1-9, O, N, D (example: NIFTY25N2524000CE -> YY=25, mcode=N, dd=25)
    _IDX_OPT_WEEKLY = re.compile(
        r"^(?P<root>NIFTY|BANKNIFTY|FINNIFTY|MIDCPNIFTY)"
        r"(?P<yy>\d{2})(?P<mcode>[1-9OND])(?P<dd>\d{2})(?P<strike>\d{1,6})(?P<right>CE|PE)$",
        re.IGNORECASE,
    )

    # Basic NSE equity, accepts RELIANCE, HDFCBANK, etc.; also 'NIFTY 50' index cash.
    _PLAIN_EQUITY = re.compile(r"^[A-Z0-9][A-Z0-9\-\s\.]*$", re.IGNORECASE)

    _MONTHS = {
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    }

    # Map from single-char weekly month code to month number
    _SINGLE_MONTH_CODE_TO_NUM: Dict[str, int] = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
        "O": 10, "N": 11, "D": 12,
    }

    # Known index cash aliases that brokers use for spot quotes
    _INDEX_SPOT_ALIASES: Dict[str, Tuple[str, ...]] = {
        "NIFTY": ("NIFTY 50", "NSE:NIFTY 50", "NSE:NIFTY"),
        "BANKNIFTY": ("NIFTY BANK", "NSE:NIFTY BANK"),
        "FINNIFTY": ("NIFTY FIN SERVICE", "NSE:NIFTY FIN SERVICE"),
        "MIDCPNIFTY": ("NIFTY MIDCAP SELECT", "NSE:NIFTY MIDCAP SELECT"),
    }

    def __init__(
        self,
        *,
        mdm: Any = None,
        broker: Any = None,
        store: Any = None,
        enable_token_cache: bool = True,
        log_prefix: str = "resolver",
    ) -> None:
        self._mdm = mdm
        self._broker = broker
        self._store = store
        self._log_prefix = log_prefix

        self._lock = threading.RLock()
        self._token_cache: Dict[str, int] = {}
        self._reverse_cache: Dict[int, str] = {}

        # Negative (failed) cache to avoid repeated lookups in hot paths
        self._neg_cache: Dict[str, float] = {}
        self._neg_ttl = 300.0  # seconds

        self._token_cache_enabled = enable_token_cache

    # --------------------------------------------------------------------- API
    def resolve(self, raw: str) -> Resolution:
        """
        Resolve a raw symbol to normalized metadata and candidates.

        This never throws; on failure, you'll get kind='equity' with no token.
        """
        now = time.time()
        s0 = (raw or "").strip()
        s0 = self._WS.sub("", s0)  # remove all whitespace; brokers expect tight TS

        exch_hint: Optional[str] = None
        m = self._EXPLICIT_EXCH.match(s0)
        if m:
            exch_hint = m.group("exch").upper()
            s0 = m.group("rest").upper()
        else:
            s0 = s0.upper()

        # Detect option vs equity/index
        opt = self._parse_index_option(s0)
        if opt:
            root, yy, mon, strike, right = opt
            segment = "derivatives"
            exch = exch_hint or "NFO"
            kind = "option"
            expiry_raw = f"{yy}{mon}"

            token = self._token_for_symbol(s0)
            candidates = self._option_candidates(
                s0, root=root, exch=exch, token=token, yy=yy, mon=mon
            )
            res = Resolution(
                symbol=s0,
                exchange=exch,
                segment=segment,
                token=token,
                kind=kind,
                root=root,
                right=right,
                strike=strike,
                expiry_raw=expiry_raw,
                candidates=candidates,
            )
            self._log_resolution(res)
            return res

        # Equity / index cash path
        # Normalize common index aliases (NIFTY → NIFTY 50, etc.) for spot quotes
        normalized_equity, exch, segment = self._normalize_equity_like(
            s0, exch_hint=exch_hint
        )
        token = self._token_for_symbol(normalized_equity)
        candidates = self._equity_candidates(normalized_equity, exch, token)

        res = Resolution(
            symbol=normalized_equity,
            exchange=exch,
            segment=segment,
            token=token,
            kind="index" if normalized_equity in self._index_spot_set() else "equity",
            root=None,
            right=None,
            strike=None,
            expiry_raw=None,
            candidates=candidates,
        )
        self._log_resolution(res)
        return res

    # ------------------------------------------------------------- Parse/Build
    def _parse_index_option(
        self, ts: str
    ) -> Optional[Tuple[str, str, str, float, str]]:
        """
        Parse index option symbol in either weekly or monthly formats.

        Returns:
            (root, yy, mon_or_mcode_or_mcodeday, strike, right)

        For weekly:
            mon is returned as single-char month code concatenated with day, e.g. 'N25'
            so expiry_raw will become '25N25' from caller's perspective (yy + mon field)

        For monthly:
            mon is 'NOV', 'OCT', etc.
        """
        # 1) Try weekly format first: ROOT + YY + M + DD + STRIKE + (CE|PE)
        m_week = self._IDX_OPT_WEEKLY.match(ts)
        if m_week:
            root = m_week.group("root").upper()
            yy = m_week.group("yy")
            mcode = m_week.group("mcode").upper()
            dd = m_week.group("dd")
            right = m_week.group("right").upper()
            try:
                strike = float(m_week.group("strike"))
            except Exception:
                return None
            # Return mon as single-char+day (e.g., 'N25') so candidates builder can interpret
            return root, yy, f"{mcode}{dd}", strike, right

        # 2) Fallback to monthly format: ROOT + YY + MMM + STRIKE + (CE|PE)
        m_mon = self._IDX_OPT_MONTHLY.match(ts)
        if m_mon:
            root = m_mon.group("root").upper()
            yy = m_mon.group("yy")
            mon = m_mon.group("mon").upper()
            right = m_mon.group("right").upper()
            if mon not in self._MONTHS:
                return None
            try:
                strike = float(m_mon.group("strike"))
            except Exception:
                return None
            return root, yy, mon, strike, right

        return None

    def _index_spot_aliases(self, key: str) -> Tuple[str, ...]:
        return self._INDEX_SPOT_ALIASES.get(key.upper(), ())

    @functools.lru_cache(maxsize=64)
    def _index_spot_set(self) -> frozenset:
        s: set = set()
        for _, vals in self._INDEX_SPOT_ALIASES.items():
            s.update(v.upper() for v in vals)
        s.update(self._INDEX_SPOT_ALIASES.keys())
        return frozenset(s)

    def _normalize_equity_like(
        self, sym: str, *, exch_hint: Optional[str] = None
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Return (normalized_symbol, exchange, segment)
        """
        # Treat bare 'NIFTY' as spot index (brokers often prefer 'NIFTY 50')
        if sym == "NIFTY":
            return "NIFTY 50", exch_hint or "NSE", "equity"
        if sym == "BANKNIFTY":
            return "NIFTY BANK", exch_hint or "NSE", "equity"
        if sym == "FINNIFTY":
            return "NIFTY FIN SERVICE", exch_hint or "NSE", "equity"
        if sym == "MIDCPNIFTY":
            return "NIFTY MIDCAP SELECT", exch_hint or "NSE", "equity"

        # Already looks like an index alias (e.g., NIFTY 50). Keep as-is.
        if sym in self._index_spot_set():
            return sym, exch_hint or "NSE", "equity"

        # Generic equity fallback
        return sym, exch_hint or "NSE", "equity"

    def _option_candidates(
        self,
        symbol: str,
        *,
        root: str,
        exch: str,
        token: Optional[int],
        yy: str,
        mon: str,
    ) -> Tuple[str, ...]:
        """
        Build a robust set of keys for broker.quote_any().
        Order matters: better keys first.

        Accept both monthly and weekly-derived 'mon' values:
        - Monthly: mon like 'NOV' -> build NFO:ROOTYYMON (monthly parent)
        - Weekly: mon like 'N25' (mcode + dd) -> build NFO:ROOTYYMDD (weekly)
        """
        keys: List[str] = []
        # 1) Fully qualified trading symbol
        keys.append(f"{exch}:{symbol}")
        # 2) Bare trading symbol (some SDKs accept without prefix)
        keys.append(symbol)

        # 3) Weekly vs monthly variants
        try:
            if mon and len(mon) >= 2 and mon[0] in self._SINGLE_MONTH_CODE_TO_NUM and mon[1:].isdigit():
                # mon looks like single-char + day, e.g., "N25"
                mcode = mon[0]
                dd = mon[1:]
                # Build weekly-style root-based keys for broader matching
                keys.append(f"{exch}:{root}{yy}{mcode}{dd}")
                keys.append(f"{root}{yy}{mcode}{dd}")
            else:
                # Monthly-style parent keys (useful for some broker APIs)
                keys.append(f"{exch}:{root}{yy}{mon}")
                keys.append(root)
        except Exception:
            # Fallback to monthly style if anything goes wrong
            keys.append(f"{exch}:{root}{yy}{mon}")
            keys.append(root)

        # 4) Token, if known
        if token is not None:
            keys.append(str(token))

        # Ensure uniqueness while preserving order
        seen = set()
        ordered = []
        for k in keys:
            if k and k not in seen:
                ordered.append(k)
                seen.add(k)
        return tuple(ordered)

    def _equity_candidates(
        self,
        symbol: str,
        exch: Optional[str],
        token: Optional[int],
    ) -> Tuple[str, ...]:
        keys: List[str] = []

        # Prefer explicit exchange prefix first if we know it
        if exch:
            keys.append(f"{exch}:{symbol}")
        keys.append(symbol)

        # For index aliases, include the family (e.g., 'NSE:NIFTY')
        if symbol in self._index_spot_set():
            for alias in self._index_spot_aliases("NIFTY"):  # safe superset for spot
                keys.append(alias)

        if token is not None:
            keys.append(str(token))

        seen = set()
        ordered = []
        for k in keys:
            if k and k not in seen:
                ordered.append(k)
                seen.add(k)
        return tuple(ordered)

    # --------------------------------------------------------------- Token map
    def _token_for_symbol(self, symbol: str) -> Optional[int]:
        """Resolve token from caches → MDM → Broker/store."""
        key = symbol.upper()

        # Negative cache
        with self._lock:
            exp = self._neg_cache.get(key)
            if exp and exp > time.time():
                return None

        # Positive cache
        if self._token_cache_enabled:
            with self._lock:
                if key in self._token_cache:
                    return self._token_cache[key]

        # 1) MDM direct
        token: Optional[int] = None
        mdm = self._mdm
        try:
            if mdm is not None:
                # common helper name variants across codebases
                for fn in ("token_for_symbol", "scan_instrument_token"):
                    f = getattr(mdm, fn, None)
                    if callable(f):
                        token = f(symbol)  # type: ignore[call-arg]
                        if token:
                            break
        except Exception as exc:
            LOGGER.debug(
                "mdm_token_lookup_error",
                extra={"event": f"{self._log_prefix}_mdm_token_lookup_error", "sym": key},
                exc_info=exc,
            )

        # 2) Broker/store fallback (search)
        if token is None and (self._broker or self._store):
            try:
                token = self._search_token_via_broker_or_store(symbol)
            except Exception as exc:
                LOGGER.debug(
                    "broker_token_search_error",
                    extra={
                        "event": f"{self._log_prefix}_broker_token_search_error",
                        "sym": key,
                    },
                    exc_info=exc,
                )

        # Cache the result (or negative)
        with self._lock:
            if token is not None:
                self._token_cache[key] = int(token)
                self._reverse_cache[int(token)] = key
            else:
                self._neg_cache[key] = time.time() + self._neg_ttl
        return token

    def _search_token_via_broker_or_store(self, symbol: str) -> Optional[int]:
        """Try best-effort search on broker/store to obtain an instrument token."""
        # store first (if it’s a static catalog, it’s cheaper)
        if self._store is not None:
            token = self._search_store(symbol)
            if token:
                return token

        # broker search APIs differ; try gently
        br = self._broker
        if br is None:
            return None

        # direct helper
        direct = getattr(br, "instrument_token_for", None)
        if callable(direct):
            token = direct(symbol)
            if token:
                return int(token)

        finder = getattr(br, "find_instrument", None)
        if callable(finder):
            ins = finder(symbol)
            token = self._extract_token_from_instrument(ins)
            if token:
                return int(token)

        search = getattr(br, "search_instruments", None)
        if callable(search):
            hits = list(search(symbol))  # type: ignore[call-arg]
            for ins in hits:
                token = self._extract_token_from_instrument(ins)
                if token:
                    return int(token)
        return None

    def _search_store(self, symbol: str) -> Optional[int]:
        """Search in a static store/catalog if available."""
        store = self._store
        try:
            if store is None:
                return None
            # tolerant: store can be Mapping or iterable of dicts
            if isinstance(store, Mapping):
                ent = store.get(symbol) or store.get(symbol.upper())
                if ent:
                    return self._extract_token_from_instrument(ent)
            else:
                for ent in store:
                    token = self._extract_token_from_instrument(ent)
                    if token:
                        name = (ent.get("tradingsymbol") or ent.get("symbol") or "").upper()
                        if name == symbol.upper():
                            return int(token)
        except Exception:
            return None
        return None

    @staticmethod
    def _extract_token_from_instrument(ent: Optional[Mapping[str, Any]]) -> Optional[int]:
        if not ent:
            return None
        for k in ("instrument_token", "token", "instrumentToken", "tradingsymbol_token"):
            v = ent.get(k) if isinstance(ent, Mapping) else None
            if v is None:
                continue
            try:
                iv = int(v)
                if iv > 0:
                    return iv
            except Exception:
                continue
        return None

    # --------------------------------------------------------------- Utilities
    def _log_resolution(self, res: Resolution) -> None:
        try:
            LOGGER.info(
                "instrument_resolved",
                extra={
                    "event": f"{self._log_prefix}_resolved",
                    "symbol": res.symbol,
                    "exchange": res.exchange,
                    "segment": res.segment,
                    "token": res.token,
                    "kind": res.kind,
                    "root": res.root,
                    "right": res.right,
                    "strike": res.strike,
                    "expiry": res.expiry_raw,
                    "candidates": list(res.candidates),
                },
            )
        except Exception:
            # Logging must never break resolution
            pass

    # ---------------------------------------------------------- Public helpers
    def candidates_for_quote(self, raw: str) -> Tuple[str, ...]:
        """
        Shortcut to build the best keys for broker.quote_any().
        """
        res = self.resolve(raw)
        return res.candidates

    def token_for(self, raw: str) -> Optional[int]:
        """
        High-level helper: resolve -> token.
        """
        return self.resolve(raw).token
