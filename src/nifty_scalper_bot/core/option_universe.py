"""Dynamic option universe construction for NIFTY contracts.

Runtime role:
- Deprecated runtime universe builder; live universe must come from InstrumentManager ActiveContractBasket.
- Legacy calendar fallback is non-live/env-gated only.
- Must not manually build live NIFTY symbols."""

from __future__ import annotations

from dataclasses import dataclass
import os
from datetime import date, datetime, time, timedelta
from typing import Any, Callable

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class OptionUniverseConfig:
    """Configuration consumed by :class:`OptionUniverseManager`."""

    underlying: str = "NIFTY"
    exchange: str = "NFO"
    strike_step: int = 50
    strikes_around_atm: int = 2
    expiry_roll_hours: float = 12.0
    market_close_hour: int = 15
    market_close_minute: int = 30
    spot_recalc_threshold_points: float = 25.0


class OptionUniverseManager:
    """Build and maintain a rolling liquid option universe around ATM."""

    def __init__(
        self,
        config: OptionUniverseConfig | dict[str, Any] | Any,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize manager state.

        Args:
            config: Option universe configuration object.
            now_fn: Optional clock override used by tests.

        Returns:
            None.

        Raises:
            ValueError: When strike step or strike window configuration is invalid.
        """
        self._config = self._coerce_config(config)
        if self._config.strike_step <= 0:
            raise ValueError("strike_step must be positive")
        if self._config.strikes_around_atm < 0:
            raise ValueError("strikes_around_atm must be non-negative")

        self._now_fn = now_fn or datetime.now
        self._latest_spot: float | None = None
        self._atm_strike: int | None = None
        self._current_expiry: date | None = None
        self._current_universe: list[str] = []
        self._frozen_universe: set[str] = set()
        self._last_recalc_spot: float | None = None

    def update_underlying(self, spot_price: float) -> None:
        """Refresh ATM, expiry and universe based on latest underlying spot.

        Args:
            spot_price: Latest NIFTY spot or index LTP.

        Returns:
            None.

        Raises:
            ValueError: When *spot_price* is not positive.
        """
        if spot_price <= 0:
            raise ValueError("spot_price must be positive")

        now = self._now_fn()
        self._latest_spot = float(spot_price)
        new_atm = int(
            round(self._latest_spot / self._config.strike_step)
            * self._config.strike_step
        )
        new_expiry = self._resolve_active_expiry(now)

        atm_changed = new_atm != self._atm_strike
        expiry_changed = new_expiry != self._current_expiry

        if expiry_changed and self._current_expiry is not None:
            LOGGER.debug(
                "OptionUniv: expiry rolled %s -> %s",
                self._current_expiry,
                new_expiry,
                extra={"event": "option_universe_expiry_roll"},
            )

        if atm_changed:
            LOGGER.debug(
                "OptionUniv: ATM updated -> ATM=%s expiry=%s",
                new_atm,
                new_expiry,
                extra={"event": "option_universe_atm_update"},
            )

        should_refresh = expiry_changed or not self._frozen_universe
        if should_refresh:
            self._atm_strike = new_atm
            self._current_expiry = new_expiry
            self._current_universe = self._build_universe(new_atm, new_expiry)
            self._frozen_universe = set(self._current_universe)
            LOGGER.debug(
                "OptionUniv: Universe refreshed -> %s",
                self._current_universe,
                extra={"event": "option_universe_refreshed"},
            )
        elif atm_changed:
            LOGGER.debug(
                "OptionUniv: Symbol ignored (not in frozen universe rebuild window)",
                extra={"event": "option_universe_frozen_skip", "atm": new_atm},
            )

    def get_current_universe(self) -> list[str]:
        """Return the latest full option universe.

        Args:
            None.

        Returns:
            list[str]: Current CE/PE symbol universe.

        Raises:
            None.
        """
        return list(self._current_universe)[:8]

    def get_primary_symbols(self) -> list[str]:
        """Return ATM-centered symbols used as primary scan targets.

        Args:
            None.

        Returns:
            list[str]: ATM ± configured strike range contracts.

        Raises:
            None.
        """
        return self.get_current_universe()

    def get_filtered_universe(self, spot_ltp: float) -> list[str]:
        """Return the frozen intraday universe while tracking spot diagnostics."""
        if spot_ltp <= 0:
            return list(self._current_universe)[:8]
        threshold = max(1.0, float(self._config.spot_recalc_threshold_points))
        if self._last_recalc_spot is None:
            self.update_underlying(spot_ltp)
        elif abs(spot_ltp - self._last_recalc_spot) >= threshold:
            # Keep universe frozen intraday; only track diagnostic spot drift.
            self._latest_spot = float(spot_ltp)
        self._last_recalc_spot = float(spot_ltp)
        return self.get_current_universe()

    def _coerce_config(
        self, config: OptionUniverseConfig | dict[str, Any] | Any
    ) -> OptionUniverseConfig:
        """Normalize external config objects into :class:`OptionUniverseConfig`."""
        if isinstance(config, OptionUniverseConfig):
            return config
        if isinstance(config, dict):
            payload = dict(config)
        else:
            payload = {
                "underlying": getattr(config, "underlying", "NIFTY"),
                "exchange": getattr(config, "exchange", "NFO"),
                "strike_step": getattr(config, "strike_step", 50),
                "strikes_around_atm": getattr(config, "strikes_around_atm", 3),
                "expiry_roll_hours": getattr(config, "expiry_roll_hours", 12.0),
                "market_close_hour": getattr(config, "market_close_hour", 15),
                "market_close_minute": getattr(config, "market_close_minute", 30),
                "spot_recalc_threshold_points": getattr(
                    config, "spot_recalc_threshold_points", 25.0
                ),
            }
        return OptionUniverseConfig(
            underlying=str(payload.get("underlying", "NIFTY")).upper(),
            exchange=str(payload.get("exchange", "NFO")).upper(),
            strike_step=int(payload.get("strike_step", 50)),
            strikes_around_atm=int(payload.get("strikes_around_atm", 3)),
            expiry_roll_hours=float(payload.get("expiry_roll_hours", 12.0)),
            market_close_hour=int(payload.get("market_close_hour", 15)),
            market_close_minute=int(payload.get("market_close_minute", 30)),
            spot_recalc_threshold_points=float(
                payload.get("spot_recalc_threshold_points", 25.0)
            ),
        )

    def _resolve_weekly_monthly_expiries(
        self, now: datetime
    ) -> tuple[date | None, date | None]:
        """Resolve nearest weekly then next monthly expiry candidates."""
        calendar = self._build_expiry_calendar(now.date())
        roll_delta = timedelta(hours=self._config.expiry_roll_hours)
        weekly = next(
            (
                item
                for item in calendar
                if (
                    datetime.combine(
                        item,
                        time(
                            self._config.market_close_hour,
                            self._config.market_close_minute,
                        ),
                    )
                    - now
                )
                > roll_delta
            ),
            None,
        )
        monthly_candidates = [
            item
            for item in self._monthly_expiries(now.date(), months=8)
            if item >= now.date()
        ]
        monthly = next(
            (item for item in monthly_candidates if weekly is None or item > weekly),
            None,
        )
        return weekly, monthly

    def _build_universe(self, atm_strike: int, expiry: date) -> list[str]:
        """Legacy CE/PE calendar symbol list, disabled for live runtime."""
        live_mode = os.getenv("EXECUTION_MODE", os.getenv("MODE", "")).strip().upper() == "LIVE"
        legacy_enabled = os.getenv("OPTION_UNIVERSE_LEGACY_FALLBACK_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
        LOGGER.warning("DEPRECATED_OPTION_UNIVERSE_WRAPPER_USED", extra={"event": "DEPRECATED_OPTION_UNIVERSE_WRAPPER_USED", "legacy_enabled": legacy_enabled, "live_mode": live_mode})
        if live_mode or not legacy_enabled:
            raise RuntimeError("OptionUniverseManager manual generation disabled; use InstrumentManager ActiveContractBasket")
        expiry_code = self._format_expiry_code(expiry)
        strikes = [
            atm_strike + (offset * self._config.strike_step)
            for offset in range(
                -self._config.strikes_around_atm, self._config.strikes_around_atm + 1
            )
        ]
        symbols: list[str] = []
        for strike in strikes:
            symbols.append(
                f"{self._config.exchange}:{self._config.underlying}{expiry_code}{strike}CE"
            )
            symbols.append(
                f"{self._config.exchange}:{self._config.underlying}{expiry_code}{strike}PE"
            )
        return symbols

    def _resolve_active_expiry(self, now: datetime) -> date:
        """Select nearest valid expiry, rolling early near expiry cutoff."""
        candidates = self._build_expiry_calendar(now.date())
        if not candidates:
            raise ValueError("no expiry candidates generated")

        roll_delta = timedelta(hours=self._config.expiry_roll_hours)
        for candidate in candidates:
            expiry_dt = datetime.combine(
                candidate,
                time(self._config.market_close_hour, self._config.market_close_minute),
            )
            if expiry_dt - now > roll_delta:
                return candidate
        return candidates[-1]

    def _build_expiry_calendar(self, anchor: date) -> list[date]:
        """Generate weekly, monthly and quarterly expiry candidates."""
        weekly = self._weekly_expiries(anchor, weeks=16)
        monthly = self._monthly_expiries(anchor, months=8)
        quarterly = self._quarterly_expiries(anchor, quarters=6)
        ordered = sorted({*weekly, *monthly, *quarterly})
        return [item for item in ordered if item >= anchor]

    def _weekly_expiries(self, anchor: date, weeks: int) -> list[date]:
        """Build Tuesday weekly expiry schedule from *anchor*."""
        days_ahead = (1 - anchor.weekday()) % 7
        first = anchor + timedelta(days=days_ahead)
        return [first + timedelta(days=7 * idx) for idx in range(weeks)]

    def _monthly_expiries(self, anchor: date, months: int) -> list[date]:
        """Build monthly expiries (last Tuesday) from *anchor*."""
        expiries: list[date] = []
        year = anchor.year
        month = anchor.month
        for _ in range(months):
            first_next = date(year + (month // 12), ((month % 12) + 1), 1)
            cursor = first_next - timedelta(days=1)
            while cursor.weekday() != 1:
                cursor -= timedelta(days=1)
            expiries.append(cursor)
            month += 1
            if month > 12:
                month = 1
                year += 1
        return expiries

    def _quarterly_expiries(self, anchor: date, quarters: int) -> list[date]:
        """Build quarterly expiries (last Tuesday of Mar/Jun/Sep/Dec)."""
        expiries: list[date] = []
        quarter_months = (3, 6, 9, 12)
        year = anchor.year
        while len(expiries) < quarters:
            for month in quarter_months:
                if len(expiries) >= quarters:
                    break
                first_next = date(year + (month // 12), ((month % 12) + 1), 1)
                cursor = first_next - timedelta(days=1)
                while cursor.weekday() != 1:
                    cursor -= timedelta(days=1)
                if cursor >= anchor:
                    expiries.append(cursor)
            year += 1
        return expiries

    def _format_expiry_code(self, expiry: date) -> str:
        """Format expiry into weekly or monthly Zerodha option code."""
        monthly_expiry = expiry in set(
            self._monthly_expiries(expiry.replace(day=1), months=1)
        )
        if monthly_expiry:
            return expiry.strftime("%y%b").upper()
        month_map = {10: "O", 11: "N", 12: "D"}
        month_token = month_map.get(expiry.month, str(expiry.month))
        return f"{expiry.strftime('%y')}{month_token}{expiry.strftime('%d')}"
