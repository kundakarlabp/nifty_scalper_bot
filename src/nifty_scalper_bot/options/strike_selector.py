"""Strike selection helpers for option trading strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import isfinite
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

from nifty_scalper_bot.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from nifty_scalper_bot.config.settings import LiquiditySettings, SelectorSettings
    from nifty_scalper_bot.data.data_hub import DataHub

LOGGER = get_logger(__name__)

try:
    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    # FIX: Correct import path from 'resolver' to 'instruments'
    from nifty_scalper_bot.data.instruments import InstrumentResolver
except Exception:  # pragma: no cover - defensive import
    MarketDataManager = object  # type: ignore
    InstrumentResolver = object  # type: ignore


def _parse_strike_from_symbol(ts: str) -> float | None:
    s = (ts or "").upper()
    tail = s.rsplit("CE", 1)[0].rsplit("PE", 1)[0] if s.endswith(("CE", "PE")) else s
    num = ""
    for ch in reversed(tail):
        if ch.isdigit():
            num = ch + num
        else:
            break
    try:
        return float(num) if num else None
    except Exception:  # pragma: no cover - defensive
        return None


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

    @property
    def settings(self) -> "SelectorSettings":
        """Return selector configuration."""
        return self._selector_settings

    def _get_mdm_and_resolver(
        self,
    ) -> tuple[MarketDataManager | None, InstrumentResolver | None]:
        mdm = getattr(self._data_hub, "market_data_manager", None) or getattr(
            self._data_hub, "_mdm", None
        )
        resolver = getattr(self._data_hub, "resolver", None) or getattr(
            self._data_hub, "_resolver", None
        )
        return cast(MarketDataManager | None, mdm), cast(
            InstrumentResolver | None, resolver
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
        cache_key = f"{underlying}_{option_type}_{self._selector_settings.expiry}"
        current_time = self._clock().timestamp()
        
        if cache_key in self._contract_cache:
            timestamp, details = self._contract_cache[cache_key]
            if (current_time - timestamp) < self._cache_ttl_seconds:
                LOGGER.debug(
                    "✅ CACHE HIT: Returning cached contract for %s", cache_key, 
                    extra={"event": "strike_cache_hit", "age_sec": round(current_time - timestamp, 2)}
                )
                return details # Return cached contract instantly
            
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

        LOGGER.info(f"✅ TYPE MATCH: Found {len(candidates)} {requested_option_type} candidates.")

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
            LOGGER.info(f"❌ EXPIRY ERROR: No contracts for target expiry {target_expiry}")
            return None

        LOGGER.info(f"✅ EXPIRY MATCH: {len(filtered_by_expiry)} contracts for {target_expiry}")

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
                    meta = resolver.lookup(lookup_symbol) or resolver.lookup(contract.symbol)
                    if isinstance(meta, Mapping):
                        ts2 = str(meta.get("tradingsymbol") or "").strip().upper()
                        ex = str(meta.get("exchange") or "NFO").strip().upper()
                        token = meta.get("instrument_token")
                        if ts2:
                            final_symbol = _normalize_exchange_symbol(f"{ex}:{ts2}")
                            lookup_symbol = final_symbol
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
                        LOGGER.warning(f"⚠️ Rejecting {final_symbol}: No Quote Data Available")
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
                LOGGER.info(f"✅ STRIKE LOCKED: {selection.symbol} (Strike: {selection.strike}, LTP: {selection.ltp})")
            # --- END CRITICAL FIX: CACHE WRITE ---
        
            LOGGER.debug(
                "Selected contract %s for %s using mode %s",
                selection.symbol,
                normalized,
                self._selector_settings.mode,
            )
            return selection
        
        LOGGER.warning(f"❌ EXHAUSTION: Checked all {len(ranked)} ranked contracts but none passed filters.")
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
        expiry_dates = sorted(
            {contract.expiry for contract in contracts if contract.expiry is not None}
        )
        if not expiry_dates:
            return None
        upcoming = [candidate for candidate in expiry_dates if candidate >= now]
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
            pool, regime, now, roll_minutes, special_dates
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
        return sorted(contracts, key=lambda c: abs(c.strike - underlying_price))

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


__all__ = ["StrikeSelector", "SelectedContract"]
