# src/utils/strike_selector.py
"""
Strike resolution helpers and market-time gates.

Clean signatures (no exchange-calendars):
- is_market_open() -> bool, IST gate using configured trading window
- get_instrument_tokens(kite_instance=None, spot_price: float|None=None) -> dict|None
    Reads symbols from `settings.instruments`
    Returns a dict with ATM math and CE/PE tokens for the chosen target strike.

Safe in shadow mode (when kite_instance is None).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from statistics import median
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, cast

from src.config import OptionSelectorSettings, settings
from src.risk.greeks import OptionType, bs_price_delta_gamma, implied_vol_newton
from src.utils.market_time import IST, is_market_open as _market_is_open

try:
    # Optional; only imported if installed
    from kiteconnect import KiteConnect  # type: ignore
except Exception:  # pragma: no cover
    KiteConnect = object  # type: ignore


logger = logging.getLogger(__name__)


_nifty_lot_warned = False
_nifty_lot_missing_warned = False


# -----------------------------------------------------------------------------
# Parsers
# -----------------------------------------------------------------------------
def parse_nfo_symbol(tsym: str):
    """Parse a standard NFO option trading symbol.

    Expected format: ``NIFTY25AUG19850CE`` or ``NIFTY25AUG19850PE``.
    Returns a dict with ``strike`` and ``option_type`` or ``None`` if no match.
    """
    m = re.search(r"([A-Z]+)\d{2}[A-Z]{3}(\d+)(CE|PE)$", tsym)
    if not m:
        return None
    k = float(m.group(2))
    opt = m.group(3)
    return {"strike": k, "option_type": opt}


# -----------------------------------------------------------------------------
# Module-level rate limiting and small caches
# -----------------------------------------------------------------------------
_last_api_call: Dict[str, float] = {}
_last_api_lock = threading.Lock()

_instruments_cache: Optional[List[Dict[str, Any]]] = None
_instruments_cache_ts: float = 0.0

_ltp_cache: Dict[str, tuple[float, float]] = {}  # symbol -> (price, ts)

_DEFAULT_SELECTOR: OptionSelectorSettings = OptionSelectorSettings()  # type: ignore[call-arg]


_SelectorT = TypeVar("_SelectorT")


def _selector_value(name: str, default: _SelectorT) -> _SelectorT:
    """Return an ``option_selector`` setting, falling back to ``default``."""

    cfg = getattr(settings, "option_selector", None)
    source = cfg if cfg is not None else _DEFAULT_SELECTOR
    try:
        value = getattr(source, name)
    except AttributeError:
        return default
    resolved = value if value is not None else default
    return cast(_SelectorT, resolved)


def _rate_limited(call_key: str, min_interval_sec: float | None = None) -> bool:
    """
    Returns True if we should WAIT (i.e., too soon), False if OK to call now.
    """
    interval = (
        float(min_interval_sec)
        if min_interval_sec is not None
        else float(
            _selector_value(
                "rate_limit_interval_seconds",
                _DEFAULT_SELECTOR.rate_limit_interval_seconds,
            )
        )
    )
    with _last_api_lock:
        now = time.time()
        last = _last_api_call.get(call_key, 0.0)
        # When tests freeze time (e.g., via freezegun) ``now`` may stop
        # advancing. In that case waiting for the interval would loop
        # forever, so treat it as if the rate limit has elapsed.
        if now <= last:
            _last_api_call[call_key] = now
            return False
        if now - last < interval:
            return True
        _last_api_call[call_key] = now
        return False


def _rate_call(fn, call_key: str, *args, **kwargs) -> Any:
    """Helper to rate-limit a function call."""
    sleep_s = float(
        _selector_value(
            "rate_limit_sleep_seconds", _DEFAULT_SELECTOR.rate_limit_sleep_seconds
        )
    )
    while _rate_limited(call_key):
        time.sleep(sleep_s)  # cooperative wait
    return fn(*args, **kwargs)


# -----------------------------------------------------------------------------
# Pure time gate (IST)
# -----------------------------------------------------------------------------
def is_market_open() -> bool:
    """Return ``True`` if current time falls within market hours."""
    return _market_is_open(datetime.now(tz=IST))


# -----------------------------------------------------------------------------
# Data fetchers (safe when kite is None)
# -----------------------------------------------------------------------------
def _fetch_instruments_nfo(
    kite: Optional[KiteConnect],
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch the full list of NFO instruments (options) with a short cache & rate limit.
    """
    global _instruments_cache, _instruments_cache_ts
    now = time.time()

    # Return cached copy if fresh enough
    ttl = float(
        _selector_value(
            "instruments_cache_ttl_seconds",
            _DEFAULT_SELECTOR.instruments_cache_ttl_seconds,
        )
    )
    if _instruments_cache is not None and (now - _instruments_cache_ts) < ttl:
        trade_symbol = str(
            getattr(getattr(settings, "instruments", object()), "trade_symbol", "")
        ).upper()
        if not any(
            str(row.get("name", "")).upper() == trade_symbol
            for row in _instruments_cache
        ):
            msg = f"Trade symbol {trade_symbol} not found in NFO instruments dump"
            logger.warning(msg)
            raise ValueError(msg)
        return _instruments_cache

    if not kite or kite is object:
        logger.debug("Kite instance missing; cannot fetch instruments (shadow mode).")
        return None

    try:
        call_key = "kite-instruments-nfo"
        # Correct signature for Kite: instruments(exchange)
        instruments = _rate_call(kite.instruments, call_key, "NFO")
        if isinstance(instruments, list) and instruments:
            trade_symbol = str(
                getattr(getattr(settings, "instruments", object()), "trade_symbol", "")
            ).upper()
            if not any(
                str(row.get("name", "")).upper() == trade_symbol for row in instruments
            ):
                msg = f"Trade symbol {trade_symbol} not found in NFO instruments dump"
                logger.warning(msg)
                raise ValueError(msg)
            _instruments_cache = instruments
            _instruments_cache_ts = now
            return instruments
        return None
    except Exception as e:
        logger.warning("Failed to fetch instruments from Kite: %s", e)
        return None


def _get_spot_ltp(kite: Optional[KiteConnect], symbol: str) -> Optional[float]:
    """
    Fetch the current spot LTP via Kite with a tiny cache. Safe if kite is None.
    """
    now = time.time()

    # cache hit?
    cached = _ltp_cache.get(symbol)
    ttl = float(
        _selector_value(
            "ltp_cache_ttl_seconds", _DEFAULT_SELECTOR.ltp_cache_ttl_seconds
        )
    )
    if cached and (now - cached[1]) < ttl:
        return float(cached[0])

    if not kite or kite is object:
        return None

    try:
        data = _rate_call(kite.ltp, f"kite-ltp-{symbol}", [symbol])
        px: Optional[float] = None
        if symbol in data:
            px = float(data[symbol]["last_price"])
        else:
            values = [float(v["last_price"]) for v in data.values()]
            if values:
                px = values[0]
        if px is not None:
            _ltp_cache[symbol] = (px, now)
            return px
        return None
    except Exception as e:
        logger.warning("Failed to fetch spot LTP for %s: %s", symbol, e)
        return None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _infer_step(trade_symbol: str) -> int:
    """
    Infer option strike step with overrides from configuration.

    ``settings.instruments.strike_step`` wins when set; otherwise the
    option-selector profile provides defaults (banknifty vs. fallback).
    """
    inst = getattr(settings, "instruments", object())
    step = getattr(inst, "strike_step", None)
    if step:
        try:
            return int(step)
        except Exception:
            pass

    s = (trade_symbol or "").upper()
    if "BANKNIFTY" in s:
        return int(
            _selector_value(
                "banknifty_strike_step", _DEFAULT_SELECTOR.banknifty_strike_step
            )
        )
    return int(
        _selector_value(
            "fallback_strike_step", _DEFAULT_SELECTOR.fallback_strike_step
        )
    )


def _nearest_strike(p: float, step: int | None = None) -> int:
    """Return the nearest strike rounded to ``step``."""
    target_step = int(step) if step is not None else int(
        _selector_value(
            "fallback_strike_step", _DEFAULT_SELECTOR.fallback_strike_step
        )
    )
    return int(target_step * round(float(p) / target_step))


def resolve_weekly_atm(
    spot: float,
    instruments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Tuple[str, int]]:
    """Resolve current-week ATM option trading symbols and lot size."""

    trade_symbol = str(getattr(settings.instruments, "trade_symbol", ""))
    step = _infer_step(trade_symbol)
    strike = _nearest_strike(spot, step)
    nfo = instruments or []
    if not nfo:
        try:
            nfo = _fetch_instruments_nfo(None) or []
        except Exception:
            nfo = []
    expiry, bucket = _select_expiry_bucket(
        datetime.now(IST),
        nfo,
        trade_symbol,
    )
    ce_sym = pe_sym = None
    lot: Optional[int] = None
    for row in bucket:
        try:
            if int(row.get("strike", 0)) != strike:
                continue
            lot = lot or int(row.get("lot_size", 0) or 0)
            itype = row.get("instrument_type")
            ts = row.get("tradingsymbol")
            if itype == "CE":
                ce_sym = ts
            elif itype == "PE":
                pe_sym = ts
        except Exception:
            continue
    out: Dict[str, Tuple[str, int]] = {}
    if ce_sym and lot:
        out["ce"] = (str(ce_sym), int(lot))
    if pe_sym and lot:
        out["pe"] = (str(pe_sym), int(lot))
    return out


def _stringify_expiry(expiry: Any) -> Optional[str]:
    """Return the broker-provided expiry coerced to ``str`` if present."""

    if expiry in (None, ""):
        return None
    if isinstance(expiry, (datetime, date)):
        return expiry.isoformat()
    try:
        text = str(expiry).strip()
    except Exception:
        return None
    return text or None


def _expiry_as_date(expiry: Any) -> Optional[date]:
    """Return ``expiry`` as a ``date`` when parsable."""

    if isinstance(expiry, datetime):
        return expiry.date()
    if isinstance(expiry, date):
        return expiry
    if expiry in (None, ""):
        return None
    try:
        text = str(expiry).strip()
    except Exception:
        return None
    if not text:
        return None
    text = text.split()[0]
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _group_by_expiry(
    dump: List[Dict[str, Any]], trade_symbol: str
) -> Dict[str, Dict[str, Any]]:
    """Return instrument rows grouped by expiry for ``trade_symbol``."""

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in dump:
        try:
            if row.get("segment") != "NFO-OPT":
                continue
            if str(row.get("name", "")).upper() != trade_symbol.upper():
                continue
            expiry_val = row.get("expiry")
            expiry_str = _stringify_expiry(expiry_val)
            if not expiry_str:
                continue
            bucket = grouped.setdefault(
                expiry_str,
                {"rows": [], "date": _expiry_as_date(expiry_val)},
            )
            if bucket.get("date") is None:
                bucket["date"] = _expiry_as_date(expiry_val)
            bucket["rows"].append(row)
        except Exception:
            continue
    return grouped


def _select_expiry_bucket(
    now: datetime, dump: List[Dict[str, Any]], trade_symbol: str
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """Pick an expiry bucket using broker metadata and configuration."""

    grouped = _group_by_expiry(dump, trade_symbol)
    if not grouped:
        return None, []

    today = now.date()
    candidates: List[tuple[str, Optional[date], List[Dict[str, Any]]]] = []
    for exp, meta in grouped.items():
        candidates.append((exp, meta.get("date"), meta.get("rows", [])))

    def _sort_key(item: tuple[str, Optional[date], List[Dict[str, Any]]]) -> tuple[int, int, str]:
        exp_str, exp_dt, _rows = item
        if exp_dt is not None:
            return (0, exp_dt.toordinal(), exp_str)
        return (1, 0, exp_str)

    candidates.sort(key=_sort_key)

    mode = str(
        getattr(getattr(settings, "strategy", object()), "option_expiry_mode", "nearest")
    ).lower()

    future = [c for c in candidates if c[1] is None or c[1] >= today]

    if mode == "today":
        for exp_str, exp_dt, rows in candidates:
            if exp_dt and exp_dt == today:
                return exp_str, rows
        if future:
            return future[0][0], future[0][2]
        return candidates[0][0], candidates[0][2]

    if mode == "next":
        if len(future) >= 2:
            return future[1][0], future[1][2]
        if future:
            return future[-1][0], future[-1][2]
        return candidates[-1][0], candidates[-1][2]

    # default: nearest expiry
    if future:
        return future[0][0], future[0][2]
    return candidates[-1][0], candidates[-1][2]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def get_instrument_tokens(
    kite_instance: Optional[KiteConnect] = None,
    spot_price: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Resolve spot token, ATM and target option strikes and tokens for the configured trade symbol.

    Args:
        kite_instance: optional KiteConnect instance; if None, runs in shadow mode.
        spot_price: optionally pass in a spot LTP (otherwise we'll fetch from Kite).

    Returns:
        {
            "spot_token": int,
            "spot_price": float | None,
            "atm_strike": int | None,
            "target_strike": int | None,
            "expiry": str | None,  # 'YYYY-MM-DD'
            "tokens": {"ce": Optional[int], "pe": Optional[int]},
        }
        or None on unrecoverable failure.
    """
    try:
        # --- read config ---
        inst = settings.instruments
        trade_symbol = str(inst.trade_symbol)
        spot_symbol = str(inst.spot_symbol)
        spot_token = int(inst.instrument_token)

        # strike step and target offset
        step = _infer_step(trade_symbol)
        strike_range = int(inst.strike_range)

        # --- resolve spot price ---
        px = None
        if spot_price is not None:
            try:
                px = float(spot_price)
            except Exception:
                px = None
        if px is None or px <= 0.0:
            px = _get_spot_ltp(kite_instance, spot_symbol)

        # Without a spot price, we can still return static info
        if px is None or px <= 0.0:
            logger.debug(
                "Spot LTP unavailable; returning minimal structure without strikes/tokens."
            )
            return {
                "spot_token": spot_token,
                "spot_price": None,
                "atm_strike": None,
                "target_strike": None,
                "expiry": None,
                "tokens": {"ce": None, "pe": None},
                "atm_tokens": {"ce": None, "pe": None},
                "prewarm_contracts": {"ce": {}, "pe": {}},
                "prewarm_tokens": {"ce": [], "pe": []},
            }

        # --- ATM rounding and target selection ---
        atm = int(round(px / step) * step)
        target = int(atm + strike_range * step)

        # --- fetch instruments and resolve CE/PE tokens (shadow-safe) ---
        try:
            nfo = _fetch_instruments_nfo(kite_instance)
        except ValueError as e:
            logger.warning("Instrument configuration issue: %s", e)
            return None
        nfo = nfo or []
        if not nfo:
            logger.debug(
                "NFO instruments unavailable; returning strike math without tokens."
            )
            return {
                "spot_token": spot_token,
                "spot_price": float(px),
                "atm_strike": atm,
                "target_strike": target,
                "expiry": None,
                "tokens": {"ce": None, "pe": None},
                "atm_tokens": {"ce": None, "pe": None},
                "prewarm_contracts": {"ce": {}, "pe": {}},
                "prewarm_tokens": {"ce": [], "pe": []},
            }

        def _extract_tokens(payload: Mapping[int, Dict[str, Any]]) -> List[int]:
            toks: List[int] = []
            for contract in payload.values():
                tok = _coerce_int(contract.get("token"))
                if tok is not None:
                    toks.append(tok)
            return toks

        expiry: Optional[str] = None
        contracts: Dict[str, Dict[str, Any]] = {}
        prewarm_contracts: Dict[str, Dict[int, Dict[str, Any]]] = {"ce": {}, "pe": {}}
        prewarm_tokens: Dict[str, List[int]] = {"ce": [], "pe": []}
        ce_token = pe_token = atm_ce = atm_pe = None
        strike_selected = target
        lot_size = 0

        global _nifty_lot_warned, _nifty_lot_missing_warned

        for attempt in range(2):
            expiry_candidate, bucket = _select_expiry_bucket(
                datetime.now(IST),
                nfo,
                trade_symbol,
            )
            expiry = expiry_candidate
            if not bucket:
                logger.warning(
                    "Unable to locate expiry bucket for trade symbol %s", trade_symbol
                )
                break

            contracts_by_strike: Dict[int, Dict[str, Mapping[str, Any]]] = {}
            instrument_rows: Dict[int, Dict[str, Any]] = {}
            lot_candidates: list[int] = []

            for row in bucket:
                try:
                    token_int = _coerce_int(row.get("instrument_token"))
                    if token_int is None:
                        continue
                    instrument_rows[token_int] = dict(row)
                    strike_val = _coerce_int(row.get("strike"))
                    if strike_val is None:
                        continue
                    opt_type = str(row.get("instrument_type", "")).upper()
                    if opt_type not in {"CE", "PE"}:
                        continue
                    contracts_by_strike.setdefault(strike_val, {})[opt_type] = row
                    lot_val = _coerce_int(row.get("lot_size"))
                    if lot_val:
                        lot_candidates.append(lot_val)
                except Exception:
                    continue

            def _find_token(target_strike: int, option_type: str) -> Optional[int]:
                row = contracts_by_strike.get(target_strike, {}).get(option_type)
                if not row:
                    return None
                return _coerce_int(row.get("instrument_token"))

            def _contract_from_row(
                row: Mapping[str, Any], fallback_strike: int
            ) -> Dict[str, Any]:
                expiry_val = _stringify_expiry(row.get("expiry"))
                strike_val = _coerce_int(row.get("strike")) or fallback_strike
                lot_val = _coerce_int(row.get("lot_size"))
                return {
                    "symbol": trade_symbol,
                    "token": _coerce_int(row.get("instrument_token")),
                    "tradingsymbol": row.get("tradingsymbol"),
                    "expiry": expiry_val,
                    "strike": strike_val,
                    "segment": row.get("segment"),
                    "instrument_type": row.get("instrument_type"),
                    "lot_size": lot_val,
                }

            ce_token = _find_token(target, "CE")
            pe_token = _find_token(target, "PE")
            atm_ce = _find_token(atm, "CE")
            atm_pe = _find_token(atm, "PE")

            using_atm = False
            if not ce_token and atm_ce:
                ce_token = atm_ce
                using_atm = True
            if not pe_token and atm_pe:
                pe_token = atm_pe
                using_atm = True
            strike_selected = atm if using_atm else target

            lot_size = 0
            if lot_candidates:
                lot_size = max(lot_candidates, key=lot_candidates.count)
            symbol_upper = trade_symbol.upper()
            if symbol_upper == "NIFTY":
                if lot_size and lot_size != 75:
                    if not _nifty_lot_warned:
                        logger.warning(
                            "NIFTY contract lot size mismatch; forcing 75",
                            extra={"lot_size": lot_size or 0, "symbol": trade_symbol},
                        )
                        _nifty_lot_warned = True
                    lot_size = 75
                elif lot_size <= 0:
                    if not _nifty_lot_missing_warned:
                        logger.warning(
                            "NIFTY contract missing lot size; defaulting to 75",
                            extra={"symbol": trade_symbol},
                        )
                        _nifty_lot_missing_warned = True
                    lot_size = 75
            if lot_size <= 0:
                logger.warning(
                    "strike_selector: unable to resolve lot size",
                    extra={"symbol": trade_symbol},
                )

            def _build_contract(token_val: Optional[int]) -> Dict[str, Any] | None:
                token_int = _coerce_int(token_val)
                if token_int is None:
                    return None
                row = instrument_rows.get(token_int)
                if not row:
                    return None
                contract = _contract_from_row(row, strike_selected)
                contract["lot_size"] = (
                    lot_size if lot_size > 0 else contract.get("lot_size")
                )
                return contract

            contracts = {}
            ce_contract = _build_contract(ce_token)
            pe_contract = _build_contract(pe_token)
            if ce_contract:
                contracts["ce"] = ce_contract
            if pe_contract:
                contracts["pe"] = pe_contract

            prewarm_contracts = {"ce": {}, "pe": {}}
            for offset in (-1, 0, 1):
                strike_val = atm + offset * step
                for opt_type in ("CE", "PE"):
                    contract_row = contracts_by_strike.get(strike_val, {}).get(opt_type)
                    if contract_row is None:
                        continue
                    contract = _contract_from_row(contract_row, strike_val)
                    if lot_size > 0 and (contract.get("lot_size") in (None, 0)):
                        contract["lot_size"] = lot_size
                    prewarm_contracts[opt_type.lower()][strike_val] = contract

            prewarm_tokens = {
                "ce": _extract_tokens(prewarm_contracts["ce"]),
                "pe": _extract_tokens(prewarm_contracts["pe"]),
            }

            if ce_token and pe_token:
                break
            if attempt == 0:
                global _instruments_cache, _instruments_cache_ts
                _instruments_cache, _instruments_cache_ts = None, 0.0
                nfo = _fetch_instruments_nfo(kite_instance) or []
                continue
            break

        if not expiry:
            return {
                "spot_token": spot_token,
                "spot_price": float(px),
                "atm_strike": atm,
                "target_strike": target,
                "expiry": None,
                "tokens": {"ce": None, "pe": None},
                "atm_tokens": {"ce": None, "pe": None},
                "prewarm_contracts": {"ce": {}, "pe": {}},
                "prewarm_tokens": {"ce": [], "pe": []},
            }

        result = {
            "spot_token": int(spot_token),
            "spot_price": float(px),
            "atm_strike": int(atm),
            "target_strike": int(strike_selected),
            "expiry": expiry,
            "tokens": {"ce": ce_token, "pe": pe_token},
            "atm_tokens": {"ce": atm_ce, "pe": atm_pe},
            "lot_size": lot_size,
            "contracts": contracts,
            "prewarm_contracts": prewarm_contracts,
            "prewarm_tokens": prewarm_tokens,
        }

        if not ce_token or not pe_token:
            logger.warning(
                "Missing option token for strike %s, expiry %s, trade symbol %s",
                strike_selected,
                expiry,
                trade_symbol,
            )
            result["error"] = "no_option_token"
        return result

    except Exception as e:
        logger.exception("get_instrument_tokens failed: %s", e)
        return None


def health_check() -> Dict[str, Any]:
    """
    Lightweight readiness info for probes.
    """
    return {
        "ok": True,
        "ist_open": is_market_open(),
        "trade_symbol": getattr(
            getattr(settings, "instruments", object()), "trade_symbol", "NIFTY"
        ),
    }


# -----------------------------------------------------------------------------
# Simple strike selection helpers (used by strategy)
# -----------------------------------------------------------------------------


@dataclass
class StrikeInfo:
    strike: int
    meta: Dict[str, Any] | None = None


def _default_option_info_fetcher(strike: int) -> Optional[Dict[str, Any]]:
    return None


_option_info_fetcher: Callable[[int], Optional[Dict[str, Any]]] = (
    _default_option_info_fetcher
)


def set_option_info_fetcher(fn: Callable[[int], Optional[Dict[str, Any]]]) -> None:
    """Allow tests or callers to provide option market info."""
    global _option_info_fetcher
    _option_info_fetcher = fn


def select_strike(
    spot: float, score: int, allow_pm1_if_score_ge: Optional[int] = None
) -> Optional[StrikeInfo]:
    """Pick an ATM (or +/-1) strike subject to basic liquidity guards."""
    step = _infer_step(
        getattr(getattr(settings, "instruments", object()), "trade_symbol", "")
    )
    atm = int(round(float(spot) / step) * step)
    candidates = [atm]
    threshold = int(
        allow_pm1_if_score_ge
        if allow_pm1_if_score_ge is not None
        else int(
            _selector_value(
                "allow_pm1_score_threshold",
                _DEFAULT_SELECTOR.allow_pm1_score_threshold,
            )
        )
    )
    if score >= threshold:
        candidates.extend([atm - step, atm + step])

    min_oi = float(
        _selector_value("min_open_interest", _DEFAULT_SELECTOR.min_open_interest)
    )
    max_spread = float(
        _selector_value("max_spread_pct", _DEFAULT_SELECTOR.max_spread_pct)
    )

    for st in candidates:
        info = _option_info_fetcher(st) or {}
        oi = float(info.get("oi", 0.0) or 0.0)
        spreads = info.get("spreads")
        if isinstance(spreads, (list, tuple)) and spreads:
            spread_pct = float(median([float(x) for x in spreads]))
        else:
            spread_pct = float(info.get("spread_pct", 999.0))
        if oi >= min_oi and spread_pct <= max_spread:
            return StrikeInfo(strike=int(st), meta={"oi": oi, "spread_pct": spread_pct})
    return None


def select_strike_by_delta(
    spot: float,
    opt: OptionType,
    expiry,
    target: float,
    band: float,
    chain: list[dict],
) -> dict | None:
    """Pick a strike near ``target`` absolute delta from ``chain``."""

    cand: list[tuple[float, dict]] = []
    iv_guess = float(
        _selector_value("delta_iv_guess", _DEFAULT_SELECTOR.delta_iv_guess)
    )
    min_oi = float(
        _selector_value("min_open_interest", _DEFAULT_SELECTOR.min_open_interest)
    )
    max_spread = float(
        _selector_value("max_spread_pct", _DEFAULT_SELECTOR.max_spread_pct)
    )
    px_floor = float(
        _selector_value(
            "delta_min_option_price", _DEFAULT_SELECTOR.delta_min_option_price
        )
    )
    px_pct = float(
        _selector_value(
            "delta_option_price_pct_of_spot",
            _DEFAULT_SELECTOR.delta_option_price_pct_of_spot,
        )
    )
    min_time = float(
        _selector_value(
            "delta_min_time_to_expiry_years",
            _DEFAULT_SELECTOR.delta_min_time_to_expiry_years,
        )
    )

    for row in chain:
        K = row["strike"]
        T = max(min_time, (expiry - datetime.now(IST)).days / 365)
        px_est = max(px_floor, spot * px_pct)
        iv = implied_vol_newton(px_est, spot, K, T, 0.06, 0.0, opt, guess=iv_guess) or iv_guess
        _, d, _ = bs_price_delta_gamma(spot, K, T, 0.06, 0.0, iv, opt)
        if (
            d is not None
            and abs(abs(d) - target) <= band
            and row.get("oi", 0) >= min_oi
            and row.get("median_spread_pct", 1.0) <= max_spread
        ):
            cand.append((abs(abs(d) - target), row))
    if not cand:
        return None
    return sorted(cand, key=lambda x: x[0])[0][1]


def needs_reatm(
    entry_spot: float, current_spot: float, drift_pct: float | None = None
) -> bool:
    """Return True if spot drift from entry exceeds ``drift_pct`` percent."""
    try:
        entry = float(entry_spot)
        curr = float(current_spot)
        if entry <= 0 or curr <= 0:
            return False
        threshold = (
            float(drift_pct)
            if drift_pct is not None
            else float(
                _selector_value(
                    "needs_reatm_pct", _DEFAULT_SELECTOR.needs_reatm_pct
                )
            )
        )
        return abs(curr - entry) / entry * 100.0 >= threshold
    except Exception:
        return False
def _coerce_int(value: Any) -> Optional[int]:
    """Return ``value`` coerced to ``int`` when possible."""

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


