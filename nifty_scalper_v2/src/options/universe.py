"""Option universe: chain loading, strike selection, expiry calculation."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..broker.base import BrokerGateway


def _next_thursday(from_date: date | None = None) -> date:
    d = from_date or date.today()
    days_ahead = (3 - d.weekday()) % 7  # Thursday=3
    if days_ahead == 0:
        days_ahead = 7
    return d + timedelta(days=days_ahead)


def _next_month_last_thursday(from_date: date | None = None) -> date:
    d = from_date or date.today()
    # Last Thursday of next month
    if d.month == 12:
        target_month, target_year = 1, d.year + 1
    else:
        target_month, target_year = d.month + 1, d.year
    # Find last day of target month
    if target_month == 12:
        last_day = date(target_year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(target_year, target_month + 1, 1) - timedelta(days=1)
    # Walk back to Thursday
    days_back = (last_day.weekday() - 3) % 7
    return last_day - timedelta(days=days_back)


class OptionUniverse:
    """Option chain loader and strike selection."""

    def __init__(self, broker: "BrokerGateway", underlying: str = "NIFTY") -> None:
        self._broker = broker
        self._underlying = underlying
        self._chain_cache: dict[str, dict] = {}
        self._instrument_cache: dict[str, int] = {}  # symbol → token

    def get_atm_strike(self, spot_price: float, interval: float = 50.0) -> float:
        """Round spot to nearest strike interval."""
        return round(spot_price / interval) * interval

    def get_strike_range(self, spot: float, n_strikes: int = 5, interval: float = 50.0) -> list[float]:
        atm = self.get_atm_strike(spot, interval)
        return [atm + interval * i for i in range(-n_strikes, n_strikes + 1)]

    def get_weekly_expiry(self, from_date: date | None = None) -> date:
        return _next_thursday(from_date)

    def get_monthly_expiry(self, from_date: date | None = None) -> date:
        return _next_month_last_thursday(from_date)

    def get_next_expiry(self, preferred: str = "weekly") -> date:
        if preferred == "monthly":
            return self.get_monthly_expiry()
        return self.get_weekly_expiry()

    async def load_chain(self, expiry: date) -> dict:
        """Fetch option chain for given expiry. Returns {strike: {ce_oi, pe_oi, ce_ltp, pe_ltp}}."""
        key = str(expiry)
        if key in self._chain_cache:
            return self._chain_cache[key]
        # Real implementation would call broker API
        chain: dict = {}
        self._chain_cache[key] = chain
        return chain

    def resolve_option_symbol(
        self, underlying: str, strike: float, expiry: date, option_type: str
    ) -> str:
        """Build NSE option symbol string. e.g. NIFTY24JUN23500CE."""
        expiry_str = expiry.strftime("%y%b").upper()
        strike_int = int(strike)
        return f"{underlying}{expiry_str}{strike_int}{option_type.upper()}"

    def filter_liquid(
        self,
        chain: dict,
        min_oi: int = 100,
        min_price: float = 5.0,
        max_spread_pct: float = 3.0,
    ) -> list[dict]:
        """Filter chain entries by liquidity criteria."""
        results = []
        for strike, data in chain.items():
            for side in ("ce", "pe"):
                oi = data.get(f"{side}_oi", 0)
                ltp = data.get(f"{side}_ltp", 0.0)
                if oi >= min_oi and ltp >= min_price:
                    results.append({"strike": strike, "type": side.upper(), **data})
        return results
