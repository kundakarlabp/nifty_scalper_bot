"""Deterministic execution policy for market order routing.

The policy converts MARKET intents into pegged LIMIT orders to achieve
predictable execution while respecting liquidity guardrails.  Quotes are
retrieved from :class:`~nifty_scalper_bot.data.data_hub.DataHub`; when the
current spread breaches the configured maximum the order is rejected early
instead of attempting a low-probability fill.

Prices are derived from the mid-point adjusted by ``peg_k`` multiples of the
spread.  Each retry widens the peg by one additional multiple, creating a
ladder that progressively becomes more aggressive.  The consumer (typically
``OrderManager``) can use the generated plan to modify orders between retries
and to enforce the overall timeout budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from nifty_scalper_bot.config.env_utils import parse_float_env
from typing import Any, Callable, Literal, Mapping, Sequence

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.utils.errors import OrderPlacementError

Side = Literal["BUY", "SELL"]


def _first_float(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        number: float | None = None
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, str):
            token = value.strip()
            if not token:
                continue
            try:
                number = float(token)
            except ValueError:
                continue
        else:
            continue
        if number is None:
            continue
        if number > 0:
            return number
    return None


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Immutable execution ladder returned by :class:`ExecutionPolicy`."""

    limit_prices: tuple[float, ...]
    spread_pct: float
    step_timeout: float

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        if not self.limit_prices:
            raise ValueError("limit_prices cannot be empty")
        if self.step_timeout <= 0:
            raise ValueError("step_timeout must be positive")


@dataclass(slots=True)
class ExecutionPolicy:
    """Compute pegged limit prices for market order routing."""

    data_hub: DataHub
    peg_k: float = 0.25
    retry_steps: int = 3
    timeout_sec: float = 4.0
    max_spread_pct: float | None = 0.015
    clock: Callable[[], float] | None = None

    def __post_init__(self) -> None:
        self.retry_steps = max(1, int(self.retry_steps))
        self.peg_k = max(0.0, float(self.peg_k))
        self.timeout_sec = max(0.5, float(self.timeout_sec))
        if self.max_spread_pct is not None:
            self.max_spread_pct = max(0.0, float(self.max_spread_pct))
        if self.clock is None:
            from time import time

            self.clock = time

    # Public API ---------------------------------------------------------
    def build_plan(self, symbol: str, side: Side) -> ExecutionPlan:
        """Return a pegged execution plan for ``symbol`` and ``side``."""

        normalized = DataHub.normalize(symbol)
        quote = self.data_hub.get_quote(normalized, allow_pull=True) or {}
        bid = _first_float(quote, "best_bid", "best_bid_price", "bid", "buy_price")
        ask = _first_float(quote, "best_ask", "best_ask_price", "ask", "sell_price")
        last_price = _first_float(quote, "ltp", "last_price", "close") or 0.0

        if bid is None and ask is None:
            if last_price <= 0:
                raise OrderPlacementError("Quote unavailable for pegged execution")
            bid = ask = last_price
        elif bid is None:
            bid = min(ask or last_price, last_price or ask or 0.0)
        elif ask is None:
            ask = max(bid or last_price, last_price or bid or 0.0)

        bid = float(bid or 0.0)
        ask = float(ask or 0.0)
        if ask <= 0 and bid <= 0:
            raise OrderPlacementError("Invalid quote for pegged execution")
        if ask <= 0:
            ask = max(bid, last_price)
        if bid <= 0:
            bid = min(ask, last_price)

        mid = (ask + bid) / 2.0 if ask > 0 and bid > 0 else max(ask, bid)
        spread = max(ask - bid, 0.0)
        spread_pct = 0.0
        if mid > 0:
            spread_pct = spread / mid
        effective_max_spread_pct = self._effective_max_spread_pct(normalized, last_price, mid)
        if (
            effective_max_spread_pct is not None
            and spread_pct > effective_max_spread_pct
        ):
            raise OrderPlacementError(
                f"Spread too wide ({spread_pct:.3f} > {effective_max_spread_pct:.3f})"
            )

        if spread <= 0:
            spread = max(mid * 0.0005, 0.05)

        prices = self._ladder_prices(mid, spread, side)
        step_timeout = max(self.timeout_sec / len(prices), 0.25)
        return ExecutionPlan(
            limit_prices=tuple(prices), spread_pct=spread_pct, step_timeout=step_timeout
        )

    def _effective_max_spread_pct(
        self, symbol: str, last_price: float, mid: float
    ) -> float | None:
        if self.max_spread_pct is None:
            return None
        base = float(self.max_spread_pct)
        if base == 0.0:
            return 0.0
        upper = symbol.upper()
        is_option = upper.endswith("CE") or upper.endswith("PE")
        if not is_option:
            return base
        try:
            atm_limit = parse_float_env(os.getenv("OPTION_EXEC_MAX_SPREAD_ATM_PCT"), 0.03)
            low_premium_limit = parse_float_env(os.getenv("OPTION_EXEC_MAX_SPREAD_LOW_PREMIUM_PCT"), 0.06)
            hard_cap = parse_float_env(os.getenv("OPTION_EXEC_MAX_SPREAD_HARD_CAP_PCT"), 0.10)
            low_premium_cutoff = parse_float_env(os.getenv("OPTION_LOW_PREMIUM_CUTOFF"), 50.0)
        except ValueError:
            atm_limit = 0.03
            low_premium_limit = 0.06
            hard_cap = 0.10
            low_premium_cutoff = 50.0
        reference = float(last_price or mid or 0.0)
        if reference > 0 and reference <= low_premium_cutoff:
            return min(max(base, low_premium_limit), hard_cap)
        return min(max(base, atm_limit), hard_cap)

    # Internal helpers ---------------------------------------------------
    def _ladder_prices(self, mid: float, spread: float, side: Side) -> Sequence[float]:
        direction = 1.0 if side == "BUY" else -1.0
        prices: list[float] = []
        base = max(mid, 0.01)
        peg = max(spread, 0.0)
        for step in range(self.retry_steps):
            multiplier = self.peg_k + step * max(self.peg_k, 0.05)
            adjustment = direction * multiplier * peg
            price = base + adjustment
            price = max(price, 0.05)
            prices.append(price)
        return prices


__all__ = ["ExecutionPlan", "ExecutionPolicy"]
