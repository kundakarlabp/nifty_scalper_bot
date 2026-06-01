"""Strike selection helpers for option trading strategies.

Runtime role:
- Deprecated strike selector wrapper; runtime must delegate to InstrumentManager context.
- Option-chain ranking is diagnostic/fallback metadata only.
- Must not be live contract authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import isfinite
import os
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    cast,
)

from nifty_scalper_bot.options.resolver import OptionResolver
from nifty_scalper_bot.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from nifty_scalper_bot.config.settings import LiquiditySettings, SelectorSettings
    from nifty_scalper_bot.data.data_hub import DataHub
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager

LOGGER = get_logger(__name__)
_SYMBOL_STRIKE_PATTERN = re.compile(
    r"(?P<strike>\d{4,6})(?P<option>CE|PE)$", re.IGNORECASE
)
_SYMBOL_EXPIRY_PATTERN = re.compile(r"(\d{2}[A-Z]{3}\d{2}|\d{6})")


def _parse_strike_from_symbol(ts: str) -> float | None:
    """Extract strike from Zerodha-style option symbols. Args: ts. Returns: strike or None. Raises: None."""

    symbol = (ts or "").strip().upper().split(":")[-1]
    match = _SYMBOL_STRIKE_PATTERN.search(symbol)
    if match is not None:
        try:
            return float(match.group("strike"))
        except (TypeError, ValueError):
            return None
    tail = (
        symbol.rsplit("CE", 1)[0].rsplit("PE", 1)[0]
        if symbol.endswith(("CE", "PE"))
        else symbol
    )
    digits = ""
    for ch in reversed(tail):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    try:
        return float(digits) if digits else None
    except (TypeError, ValueError):
        return None


def _extract_expiry_token(symbol: str) -> str | None:
    """Extract expiry token from trading symbol. Args: symbol. Returns: expiry token or None. Raises: None."""

    normalized = (symbol or "").strip().upper().split(":")[-1]
    match = _SYMBOL_EXPIRY_PATTERN.search(normalized)
    if match is None:
        return None
    return match.group(1)


def _nearest_strike(price: float, step: int = 50) -> float:
    """Return nearest strike for *price*. Args: price, step. Returns: strike. Raises: None."""

    if step <= 0:
        return float(price)
    return float(round(price / step) * step)


@dataclass(frozen=True, slots=True)
class SelectedContract:
    """Selected option contract returned to the strategy runner."""

    symbol: str
    option_type: str
    strike: float
    expiry: datetime
    ltp: float
    delta: float | None
    metadata: Mapping[str, Any]


@dataclass(slots=True)
class _OptionContract:
    symbol: str
    option_type: str
    strike: float
    expiry: datetime
    ltp: float
    delta: float | None
    bid: float | None
    ask: float | None
    open_interest: float | None
    trades: float | None
    depth_buy: Sequence[Mapping[str, Any]]
    depth_sell: Sequence[Mapping[str, Any]]
    raw: Mapping[str, Any]

    def spread_pct(self) -> float | None:
        bid = self.bid
        ask = self.ask
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None
        if ask <= bid:
            return 0.0
        mid = (ask + bid) / 2.0
        if mid <= 0:
            return None
        return float((ask - bid) / mid * 100.0)

    def top_depth(self, levels: int = 3) -> float:
        def _collect(levels_payload: Sequence[Mapping[str, Any]]) -> float:
            total = 0.0
            count = 0
            for entry in levels_payload:
                if count >= levels:
                    break
                qty = entry.get("quantity") or entry.get("qty") or entry.get("size")
                if qty is None:
                    continue
                try:
                    total += float(qty)
                    count += 1
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    continue
            return total

        return _collect(self.depth_buy) + _collect(self.depth_sell)

    @staticmethod
    def from_mapping(payload: Mapping[str, Any]) -> _OptionContract | None:
        symbol = str(
            payload.get("tradingsymbol") or payload.get("symbol") or ""
        ).strip()
        if not symbol:
            return None
        option_type = symbol[-2:].upper()
        if option_type not in {"CE", "PE"}:
            normalized_symbol = symbol.upper().replace("_", "-")
            for chunk in reversed(normalized_symbol.split("-")):
                chunk = chunk.strip()
                if chunk.endswith("CE") or chunk.endswith("PE"):
                    option_type = chunk[-2:]
                    break
            else:
                explicit_type = str(
                    payload.get("option_type") or payload.get("type") or ""
                ).upper()
                if explicit_type in {"CE", "PE"}:
                    option_type = explicit_type
                else:
                    return None
        strike_value = _coerce_float(
            payload.get("strike")
            or payload.get("strike_price")
            or payload.get("strikePrice")
        )
        if strike_value is None:
            strike_value = _parse_strike_from_symbol(symbol)
            if strike_value is None:
                return None
        strike = float(strike_value)
        if int(strike) % 50 != 0:
            LOGGER.warning("Unexpected strike: %s", strike)
        expiry_raw = payload.get("expiry") or payload.get("expiry_date")
        expiry_dt = _parse_expiry(expiry_raw)
        if expiry_dt is None:
            return None
        ltp_value = _coerce_float(
            payload.get("ltp")
            or payload.get("last_price")
            or payload.get("close")
            or payload.get("price")
        )
        bid = _coerce_float(
            payload.get("bid")
            or payload.get("bid_price")
            or payload.get("best_bid_price")
        )
        ask = _coerce_float(
            payload.get("ask")
            or payload.get("ask_price")
            or payload.get("best_ask_price")
        )
        delta = _coerce_float(payload.get("delta"))
        oi = _coerce_float(
            payload.get("open_interest")
            or payload.get("oi")
            or payload.get("openInterest")
        )
        trades = _coerce_float(
            payload.get("trades")
            or payload.get("trade_count")
            or payload.get("volume_traded")
            or payload.get("number_of_trades")
        )
        depth = payload.get("depth") or {}
        buy_depth = depth.get("buy") if isinstance(depth, Mapping) else None
        sell_depth = depth.get("sell") if isinstance(depth, Mapping) else None
        depth_buy = (
            tuple(buy_depth)
            if isinstance(buy_depth, Sequence)
            and not isinstance(buy_depth, (str, bytes))
            else tuple()
        )
        depth_sell = (
            tuple(sell_depth)
            if isinstance(sell_depth, Sequence)
            and not isinstance(sell_depth, (str, bytes))
            else tuple()
        )
        if ltp_value is None:
            if bid is not None and ask is not None:
                ltp_value = (bid + ask) / 2.0
            else:
                ltp_value = 0.0
        return _OptionContract(
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiry=expiry_dt,
            ltp=float(ltp_value),
            delta=delta,
            bid=bid,
            ask=ask,
            open_interest=oi,
            trades=trades,
            depth_buy=depth_buy,
            depth_sell=depth_sell,
            raw=payload,
        )


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _parse_expiry(value: Any) -> datetime | None:
    """Parse expiry payload into a timezone-aware datetime.

    Args:
        value: Raw expiry field from the option chain entry.

    Returns:
        datetime | None: Parsed expiry timestamp in UTC, or ``None`` when parsing
        fails.

    Raises:
        None.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        return (
            value.astimezone(timezone.utc)
            if value.tzinfo
            else value.replace(tzinfo=timezone.utc)
        )
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            raw = int(value)
            if raw > 10**11:
                return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except (OverflowError, ValueError, OSError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in (
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%d-%b-%Y",
            "%d-%b-%y",
            "%d %b %Y",
            "%d %B %Y",
            "%d%b%Y",
            "%d%b%y",
            "%Y%m%d",
            "%d-%m-%Y",
            "%d%m%Y",
        ):
            try:
                parsed = datetime.strptime(text, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        if text.isdigit():
            digits = text
            if len(digits) == 8:
                for fmt in ("%Y%m%d", "%d%m%Y"):
                    try:
                        parsed = datetime.strptime(digits, fmt)
                        return parsed.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
            try:
                raw = int(digits)
                if raw > 10**11:
                    return datetime.fromtimestamp(raw / 1000.0, tz=timezone.utc)
                return datetime.fromtimestamp(raw, tz=timezone.utc)
            except (OverflowError, ValueError, OSError):
                return None
        try:
            parsed = datetime.fromisoformat(text)
            return (
                parsed.astimezone(timezone.utc)
                if parsed.tzinfo
                else parsed.replace(tzinfo=timezone.utc)
            )
        except ValueError:
            return None
    return None


class StrikeSelector:
    """Select option contracts using ATM or delta targeting with liquidity filters."""

    def __init__(
        self,
        *,
        data_hub: "DataHub",
        selector_settings: "SelectorSettings",
        liquidity_settings: "LiquiditySettings",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._data_hub = data_hub
        self._selector_settings = selector_settings
        self._liquidity_settings = liquidity_settings
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._active: MutableMapping[str, SelectedContract] = {}
        self._contract_cache = {}  # {symbol: (timestamp, contract_details)}
        self._cache_ttl_seconds = 1
        self._deterministic_resolver: OptionResolver | None = None
        self._locked_ce: SelectedContract | None = None
        self._locked_pe: SelectedContract | None = None
        self._locked_spot: float | None = None
        self._locked_at: datetime | None = None


    @staticmethod
    def _basket_get(basket: Any, key: str, default: Any = None) -> Any:
        if isinstance(basket, Mapping):
            return basket.get(key, default)
        return getattr(basket, key, default)

    @staticmethod
    def _symbol_strike(symbol: str) -> float:
        match = _SYMBOL_STRIKE_PATTERN.search(str(symbol or ""))
        return float(match.group("strike")) if match else 0.0

    def _contract_from_active_basket(
        self,
        basket: Any,
        *,
        option_type: str,
        underlying_price: float,
    ) -> SelectedContract | None:
        symbol_key = "selected_ce" if option_type == "CE" else "selected_pe"
        token_key = "selected_ce_token" if option_type == "CE" else "selected_pe_token"
        symbol = self._basket_get(basket, symbol_key)
        if not symbol:
            return None
        symbol = str(symbol).strip().upper()
        token = self._basket_get(basket, token_key)
        token_by_symbol = dict(self._basket_get(basket, "token_by_symbol", {}) or {})
        if token is None:
            token = token_by_symbol.get(symbol)
        quote = None
        get_quote = getattr(self._data_hub, "get_quote", None)
        if callable(get_quote):
            try:
                quote = get_quote(symbol, allow_pull=False)
            except TypeError:
                quote = get_quote(symbol)
            except Exception:
                quote = None
        ltp = 0.0
        if isinstance(quote, Mapping):
            for key in ("ltp", "last_price", "price", "close"):
                value = _coerce_float(quote.get(key))
                if value is not None and value > 0:
                    ltp = float(value)
                    break
            if ltp <= 0:
                ltp = float(_mid_price(_coerce_float(quote.get("bid")), _coerce_float(quote.get("ask"))) or 0.0)
        expiry_raw = self._basket_get(basket, "option_expiry", None)
        expiry = _parse_expiry(expiry_raw) or datetime.now(timezone.utc)
        strike = float(self._basket_get(basket, "atm_strike", 0) or 0) or self._symbol_strike(symbol)
        return SelectedContract(
            symbol=symbol,
            option_type=option_type,
            strike=float(strike),
            expiry=expiry,
            ltp=float(ltp),
            delta=None,
            metadata={
                "instrument_token": token,
                "source": "active_contract_basket",
                "underlying_price": float(underlying_price),
            },
        )

    @property
    def settings(self) -> "SelectorSettings":
        """Return selector configuration."""
        return self._selector_settings

    def _get_mdm_and_resolver(
        self,
    ) -> tuple[Any | None, Any | None]:  # Changed from MarketDataManager to Any
        mdm = getattr(self._data_hub, "market_data_manager", None) or getattr(
            self._data_hub, "_mdm", None
        )
        resolver = getattr(self._data_hub, "resolver", None) or getattr(
            self._data_hub, "_resolver", None
        )
        # Removed the cast() calls that triggered the NameError
        return mdm, resolver

    def _get_broker_client(self) -> Any | None:
        """Resolve broker client from data hub context. Args: none. Returns: broker|None. Raises: none."""

        broker = getattr(self._data_hub, "broker", None)
        if broker is not None:
            return broker
        mdm = getattr(self._data_hub, "market_data_manager", None) or getattr(
            self._data_hub, "_mdm", None
        )
        return getattr(mdm, "broker", None) if mdm is not None else None

    def _resolve_deterministic_contract(
        self,
        *,
        option_type: str,
        underlying_price: float,
    ) -> SelectedContract | None:
        """Resolve ATM contract from broker instrument dump. Args: option_type/underlying_price. Returns: contract|None. Raises: none."""

        broker = self._get_broker_client()
        if broker is None:
            return None
        if self._deterministic_resolver is None:
            self._deterministic_resolver = OptionResolver(broker)
        try:
            pair = self._deterministic_resolver.resolve_atm_pair(underlying_price)
        except Exception as e:
            LOGGER.exception("Failure in deterministic option resolver: %s", e)
            return None
        instrument = pair.ce if option_type == "CE" else pair.pe
        return SelectedContract(
            symbol=f"NFO:{instrument.tradingsymbol}",
            option_type=instrument.option_type,
            strike=float(instrument.strike),
            expiry=datetime.combine(instrument.expiry, datetime.min.time(), tzinfo=timezone.utc),
            ltp=0.0,
            delta=None,
            metadata={"instrument_token": int(instrument.instrument_token)},
        )

    def select_contract(
        self,
        *,
        underlying: str,
        side: "Literal['BUY', 'SELL']",
        option_type: "Literal['CE', 'PE']",
        underlying_price: float,
        **_: Any,
    ) -> SelectedContract | None:
        """Select an option contract, utilizing in-memory caching for speed."""

        # Note: Assumes _contract_cache and _cache_ttl_seconds are defined in __init__

        # --- START CRITICAL FIX: CACHE CHECK (Latency Optimization) ---
        step = int(getattr(self._selector_settings, "strike_step", 50) or 50)
        cache_price = round(underlying_price / step) * step 
        
        cache_key = f"{underlying}_{option_type}_{cache_price}_{self._selector_settings.expiry}"
        current_time = self._clock().timestamp()

        if cache_key in self._contract_cache:
            timestamp, details = self._contract_cache[cache_key]
            if (current_time - timestamp) < self._cache_ttl_seconds:
                LOGGER.debug(
                    "✅ CACHE HIT: Returning cached contract for %s",
                    cache_key,
                    extra={
                        "event": "strike_cache_hit",
                        "age_sec": round(current_time - timestamp, 2),
                    },
                )
                return details  # Return cached contract instantly

            # Cache expired: remove and proceed to recalculate
            del self._contract_cache[cache_key]
            LOGGER.debug("Cache expired for %s. Recalculating.", cache_key)
        # --- END CRITICAL FIX: CACHE CHECK ---

        LOGGER.info(
            f"🕵️ SELECTOR ENTRY: Looking for {side} {option_type} on {underlying} @ {underlying_price}"
        )

        normalized = underlying.strip().upper()
        if not normalized:
            raise ValueError("underlying must be provided")
        if underlying_price <= 0:
            LOGGER.warning(
                "Invalid underlying price for %s: %s", normalized, underlying_price
            )
            return None

        try:
            requested_option_type = _normalize_option_type(option_type)
        except ValueError as exc:
            LOGGER.error("Failure in select_contract_option_type: %s", exc)
            return None

        if requested_option_type is None:
            LOGGER.info("Condition met: selector_missing_option_type")
            return None

        live_mode = os.getenv("EXECUTION_MODE", os.getenv("MODE", "")).strip().upper() == "LIVE"
        basket = None
        get_basket = getattr(self._data_hub, "get_active_contract_basket", None)
        if callable(get_basket):
            try:
                basket = get_basket()
            except Exception:
                basket = None
        if live_mode:
            if basket is None:
                LOGGER.warning(
                    "STRIKE_SELECTOR_ACTIVE_BASKET_MISSING option_type=%s",
                    requested_option_type,
                    extra={"event": "STRIKE_SELECTOR_ACTIVE_BASKET_MISSING", "option_type": requested_option_type},
                )
                return None
            selected = self._contract_from_active_basket(
                basket, option_type=requested_option_type, underlying_price=underlying_price
            )
            LOGGER.info(
                "DEPRECATED_STRIKE_SELECTOR_WRAPPER_USED source=active_contract_basket live_mode=%s symbol=%s",
                live_mode,
                getattr(selected, "symbol", None),
                extra={
                    "event": "DEPRECATED_STRIKE_SELECTOR_WRAPPER_USED",
                    "source": "active_contract_basket",
                    "live_mode": live_mode,
                    "symbol": getattr(selected, "symbol", None),
                },
            )
            return selected

        fallback_enabled = os.getenv("OPTION_CHAIN_SELECTOR_FALLBACK_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
        if not fallback_enabled:
            LOGGER.warning(
                "DEPRECATED_STRIKE_SELECTOR_WRAPPER_USED source=fallback_disabled live_mode=%s",
                live_mode,
                extra={"event": "DEPRECATED_STRIKE_SELECTOR_WRAPPER_USED", "source": "fallback_disabled", "live_mode": live_mode},
            )
            return None
        LOGGER.warning(
            "DEPRECATED_STRIKE_SELECTOR_WRAPPER_USED source=legacy_option_chain live_mode=%s",
            live_mode,
            extra={"event": "DEPRECATED_STRIKE_SELECTOR_WRAPPER_USED", "source": "legacy_option_chain", "live_mode": live_mode},
        )

        sticky_seconds = int(getattr(self._selector_settings, "option_sticky_seconds", 120) or 120)
        min_move = float(getattr(self._selector_settings, "option_reselection_min_move", 80.0) or 80.0)
        force_reselect_after = int(getattr(self._selector_settings, "option_force_reselect_after_seconds", 300) or 300)
        now = self._clock()
        locked = self._locked_ce if requested_option_type == "CE" else self._locked_pe
        if locked is not None and self._locked_spot is not None and self._locked_at is not None:
            move = abs(float(underlying_price) - float(self._locked_spot))
            age = (now - self._locked_at).total_seconds()
            if move < min_move and age < sticky_seconds:
                LOGGER.info("OPTION_STICKY_LOCK_ACTIVE symbol=%s move=%.2f age_s=%.1f", locked.symbol, move, age)
                return locked
            if move < min_move and age < force_reselect_after:
                LOGGER.info("OPTION_RESELECTION_SKIPPED_SMALL_MOVE move=%.2f age_s=%.1f threshold=%.2f", move, age, min_move)
                return locked
            LOGGER.info("OPTION_RESELECTION_TRIGGERED move=%.2f age_s=%.1f", move, age)

        deterministic = self._resolve_deterministic_contract(
            option_type=requested_option_type,
            underlying_price=underlying_price,
        )
        if deterministic is not None:
            if requested_option_type == "CE":
                self._locked_ce = deterministic
            else:
                self._locked_pe = deterministic
            self._locked_spot = float(underlying_price)
            self._locked_at = now
            return deterministic

        # --- TRACE 1: FETCH CHAIN ---
        LOGGER.info(f"🟡 Fetching option chain for {normalized}...")
        chain = self._data_hub.get_option_chain(self._selector_settings.expiry)

        if not chain:
            LOGGER.warning(f"❌ CHAIN ERROR: No option chain returned for {normalized}")
            return None

        LOGGER.info(f"✅ CHAIN SUCCESS: Received {len(chain)} raw items.")

        # --- TRACE 2: PARSE & FILTER TYPE ---
        option_contracts = [
            _OptionContract.from_mapping(entry)
            for entry in chain
            if isinstance(entry, Mapping)
        ]

        candidates = [
            contract
            for contract in option_contracts
            if contract is not None and contract.option_type == requested_option_type
        ]

        if not candidates:
            LOGGER.warning(
                f"❌ FILTER ERROR: No {requested_option_type} contracts found out of {len(option_contracts)} items."
            )
            return None

        LOGGER.info(
            f"✅ TYPE MATCH: Found {len(candidates)} {requested_option_type} candidates."
        )

        # --- TRACE 3: SELECT EXPIRY ---
        now = self._clock()
        target_expiry = self._select_expiry(candidates, now)
        if target_expiry is None:
            LOGGER.info("Unable to determine expiry for %s", normalized)
            return None

        filtered_by_expiry = [
            contract
            for contract in candidates
            if contract.expiry.date() == target_expiry.date()
        ]

        if not filtered_by_expiry:
            LOGGER.info(
                f"❌ EXPIRY ERROR: No contracts for target expiry {target_expiry}"
            )
            return None

        LOGGER.info(
            f"✅ EXPIRY MATCH: {len(filtered_by_expiry)} contracts for {target_expiry}"
        )

        # --- TRACE 4: RANKING & DELTA ---
        ranked = self._rank_contracts(filtered_by_expiry, underlying_price)
        LOGGER.info(f"ℹ️ Ranked {len(ranked)} contracts by proximity.")

        selection: SelectedContract | None = None

        for contract in ranked:
            # Trace Rejection Reasons
            if not self._within_delta(contract):
                # LOGGER.debug(f"Skipping {contract.symbol}: Delta mismatch")
                continue
            if not self._passes_liquidity(contract):
                LOGGER.debug(f"Skipping {contract.symbol}: Failed liquidity check")
                continue

            # --- quote-availability gate (one refresh) ---
            mdm, resolver = self._get_mdm_and_resolver()

            lookup_symbol = _normalize_exchange_symbol(contract.symbol)
            final_symbol = lookup_symbol or contract.symbol
            token = None

            if resolver is not None:
                try:
                    meta = resolver.lookup(lookup_symbol) or resolver.lookup(
                        contract.symbol
                    )
                    if isinstance(meta, Mapping):
                        ts2 = str(meta.get("tradingsymbol") or "").strip().upper()
                        ex = str(meta.get("exchange") or "NFO").strip().upper()
                        token = meta.get("instrument_token")
                        if ts2:
                            final_symbol = _normalize_exchange_symbol(f"{ex}:{ts2}")
                            lookup_symbol = final_symbol
                    if token is not None:
                        token_meta = resolver.lookup(token)
                        if token_meta is None:
                            try:
                                resolver.option_contracts(
                                    normalized, force_refresh=True
                                )
                                token_meta = resolver.lookup(token)
                            except Exception as exc:  # noqa: BLE001
                                LOGGER.error(
                                    "Failure in select_contract token refresh: %s", exc
                                )
                            if token_meta is None:
                                LOGGER.warning(
                                    "Condition met: selector_invalid_token",
                                    extra={
                                        "event": "selector_invalid_token",
                                        "symbol": final_symbol,
                                        "token": token,
                                    },
                                )
                                continue
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug(f"Resolver lookup failed for {contract.symbol}: {exc}")

            query_symbols: list[str] = []
            seen_symbols: set[str] = set()
            for candidate in (
                lookup_symbol,
                final_symbol,
                contract.symbol.strip().upper(),
            ):
                normalized_candidate = (candidate or "").strip().upper()
                if not normalized_candidate or normalized_candidate in seen_symbols:
                    continue
                query_symbols.append(normalized_candidate)
                seen_symbols.add(normalized_candidate)

            # --- TRACE 5: QUOTE CHECK ---
            if mdm is not None:
                try:
                    bars = mdm.get_ohlc_bars(final_symbol) or []
                    if len(bars) == 0:
                        LOGGER.warning(
                            "Condition met: selector_excluded_zero_bars",
                            extra={
                                "event": "selector_excluded_zero_bars",
                                "symbol": final_symbol,
                            },
                        )
                        continue
                    quote_available = False
                    attempted_refresh = False
                    for query_symbol in query_symbols or [final_symbol]:
                        has_quote = bool(mdm.get_quote(query_symbol))
                        if not has_quote and not attempted_refresh:
                            # LOGGER.info(f"🔄 Refreshing quote for {query_symbol}...")
                            mdm.refresh_quote_now(query_symbol)
                            attempted_refresh = True
                            has_quote = bool(mdm.get_quote(query_symbol))
                        if has_quote:
                            quote_available = True
                            break

                    if not quote_available:
                        LOGGER.warning(
                            f"⚠️ Rejecting {final_symbol}: No Quote Data Available"
                        )
                        continue
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(f"Quote gate exception: {exc}")
                    continue

            # Calculate LTP
            ltp = (
                contract.ltp
                if contract.ltp > 0
                else (_mid_price(contract.bid, contract.ask) or 0.0)
            )

            selection = SelectedContract(
                symbol=final_symbol,
                option_type=contract.option_type,
                strike=contract.strike,
                expiry=contract.expiry,
                ltp=float(ltp),
                delta=contract.delta,
                metadata={**dict(contract.raw), "instrument_token": token},
            )

            # --- START CRITICAL FIX: CACHE WRITE ---
            if selection:
                self._contract_cache[cache_key] = (current_time, selection)
                LOGGER.info(
                    f"✅ STRIKE LOCKED: {selection.symbol} (Strike: {selection.strike}, LTP: {selection.ltp})"
                )
            # --- END CRITICAL FIX: CACHE WRITE ---

            LOGGER.debug(
                "Selected contract %s for %s using mode %s",
                selection.symbol,
                normalized,
                self._selector_settings.mode,
            )
            return selection

        LOGGER.warning(
            f"❌ EXHAUSTION: Checked all {len(ranked)} ranked contracts but none passed filters."
        )
        return None

    def register_open(self, underlying: str, contract: SelectedContract) -> None:
        self._active[underlying.strip().upper()] = contract

    def resolve_active(self, underlying: str) -> SelectedContract | None:
        return self._active.get(underlying.strip().upper())

    def clear_active(self, underlying: str) -> None:
        self._active.pop(underlying.strip().upper(), None)

    def _resolve_expiry_target(
        self,
        candidates: Sequence[datetime],
        regime: str,
        now: datetime,
        roll_minutes: int,
        special_dates: set[date],
    ) -> tuple[datetime | None, str]:
        """Resolve preferred expiry using configured regime."""

        if not candidates:
            return None, "empty_pool"
        normalized_regime = (regime or "weekly").strip().lower()
        selection = candidates[0]
        reason = "weekly_default"
        if special_dates:
            for candidate in candidates:
                if candidate >= now and candidate.date() in special_dates:
                    selection = candidate
                    reason = "special_event_match"
                    break
        if normalized_regime == "monthly":
            selection = candidates[-1]
            reason = "monthly_regime"
        elif normalized_regime == "auto":
            selection = candidates[0]
            reason = "auto_weekly"
        elif normalized_regime == "special":
            if selection.date() not in special_dates:
                for candidate in candidates:
                    if candidate.date() in special_dates:
                        selection = candidate
                        reason = "special_regime_match"
                        break
        elif normalized_regime == "weekly":
            selection = candidates[0]
            reason = "weekly_regime"
        roll_seconds = max(0, int(roll_minutes)) * 60.0
        if roll_seconds > 0 and len(candidates) > 1 and selection is not None:
            time_to_expiry = (selection - now).total_seconds()
            if time_to_expiry <= roll_seconds:
                for candidate in candidates:
                    if candidate > selection:
                        selection = candidate
                        reason = "roll_window_advance"
                        break
        return selection, reason

    def _select_expiry(
        self, contracts: Iterable[_OptionContract], now: datetime
    ) -> datetime | None:
        def _to_utc_aware(dt: Any) -> datetime:
            if not isinstance(dt, datetime) and isinstance(dt, date):
                dt = datetime.combine(dt, datetime.min.time())
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        now_aware = _to_utc_aware(now)
        expiry_dates = sorted(
            {_to_utc_aware(contract.expiry) for contract in contracts if contract.expiry is not None}
        )
        if not expiry_dates:
            return None
        upcoming = [candidate for candidate in expiry_dates if candidate >= now_aware]
        pool = upcoming or expiry_dates
        regime = (
            (
                getattr(self._selector_settings, "expiry_regime", None)
                or self._selector_settings.expiry
                or "weekly"
            )
            .strip()
            .lower()
        )
        roll_minutes = int(getattr(self._selector_settings, "expiry_roll_minutes", 0))
        special_tokens = getattr(self._selector_settings, "special_expiries", ())
        special_dates = {
            parsed.date()
            for token in special_tokens
            if isinstance(token, str)
            for parsed in [_parse_expiry(token)]
            if parsed is not None
        }
        selection, reason = self._resolve_expiry_target(
            pool, regime, now_aware, roll_minutes, special_dates
        )
        if selection is None:
            return None
        LOGGER.info(
            "Condition met: selector_expiry_regime",
            extra={
                "event": "selector_expiry_regime",
                "mode": regime,
                "reason": reason,
                "selected": selection.isoformat(),
                "candidates": [candidate.isoformat() for candidate in pool],
                "roll_minutes": roll_minutes,
            },
        )
        return selection

    def _rank_contracts(
        self, contracts: Sequence[_OptionContract], underlying_price: float
    ) -> list[_OptionContract]:
        mode = (self._selector_settings.mode or "ATM").strip().upper()
        if mode == "DELTA_TARGET":
            target = abs(float(self._selector_settings.delta_target))
            return sorted(
                contracts,
                key=lambda c: _delta_score(c.delta, target),
            )
        step = int(getattr(self._selector_settings, "strike_step", 50) or 50)
        atm = _nearest_strike(underlying_price, step)
        window = float(max(100, step))
        active_contracts = [
            contract
            for contract in contracts
            if abs(float(contract.strike) - atm) <= window
        ]
        if not active_contracts:
            active_contracts = list(contracts)
        return sorted(active_contracts, key=lambda c: abs(c.strike - underlying_price))

    def _within_delta(self, contract: _OptionContract) -> bool:
        mode = (self._selector_settings.mode or "ATM").strip().upper()
        if mode != "DELTA_TARGET":
            return True
        if contract.delta is None:
            return False
        target = abs(float(self._selector_settings.delta_target))
        tol = abs(float(self._selector_settings.delta_tolerance))
        return abs(abs(contract.delta) - target) <= tol

    def _passes_liquidity(self, contract: _OptionContract) -> bool:
        liq = self._liquidity_settings
        if liq.min_oi > 0 and (contract.open_interest or 0.0) < liq.min_oi:
            return False
        spread_pct = contract.spread_pct()
        if (
            liq.max_spread_pct > 0
            and spread_pct is not None
            and spread_pct > liq.max_spread_pct
        ):
            return False
        if liq.min_trades > 0 and (contract.trades or 0.0) < liq.min_trades:
            return False
        if liq.min_top3_depth > 0 and contract.top_depth() < liq.min_top3_depth:
            return False
        return True


def _delta_score(delta: float | None, target: float) -> float:
    if delta is None or not isfinite(delta):
        return float("inf")
    return abs(abs(delta) - target)


def _normalize_option_type(option_type: str | None) -> str | None:
    """Normalise and validate an option type token."""

    if option_type is None:
        return None
    token = option_type.strip().upper()
    if not token:
        return None
    if token in {"CE", "PE"}:
        return token
    raise ValueError(f"Unsupported option type: {option_type!r}")


def _mid_price(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


def _normalize_exchange_symbol(symbol: str) -> str:
    """Return an exchange-qualified trading symbol string."""

    normalized = (symbol or "").strip().upper()
    if not normalized:
        return normalized
    if normalized.startswith(("NFO:", "NSE:", "BSE:")):
        return normalized
    if normalized.endswith(("CE", "PE")):
        return f"NFO:{normalized}"
    return normalized


__all__ = [
    "StrikeSelector",
    "SelectedContract",
    "_parse_strike_from_symbol",
    "_extract_expiry_token",
]
