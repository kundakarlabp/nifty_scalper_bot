"""ATM option contract selection driven entirely by broker instrument tokens.

Instead of constructing Zerodha tradingsymbol strings manually (which is
error-prone and expiry-format-dependent), this module fetches the live NFO
instrument dump and selects contracts by their intrinsic attributes:

    • name == "NIFTY"
    • nearest available expiry
    • strikes within ±200 of the current ATM
    • instrument_type in {"CE", "PE"}

Every returned dict includes the `instrument_token` field so callers can
subscribe, fetch quotes, and request historical data entirely by token,
with zero manual string construction.

Usage::

    contracts = get_atm_contracts(kite_client, underlying_price=24850.0)
    tokens = [c["instrument_token"] for c in contracts]
"""

from __future__ import annotations

from datetime import date, datetime
import logging
from typing import Any

LOGGER = logging.getLogger("nifty_scalper_bot.core.contract_selector")

_STRIKE_BAND = 200      # include strikes ATM ± this many points
_STRIKE_STEP_DEFAULT = 50  # fallback step when no instruments loaded


def _coerce_expiry(value: Any) -> date | None:
    """Safely coerce expiry field from broker row to a date object.

    Args: value – raw expiry value (datetime, date, str, or other).
    Returns: date or None on failure.
    Raises: None.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
    return None


def get_atm_contracts(
    kite: Any,
    underlying_price: float,
    *,
    strike_band: int = _STRIKE_BAND,
    strike_step: int | None = None,
) -> list[dict[str, Any]]:
    """Return NIFTY option contracts for the nearest expiry around the ATM strike.

    Selects contracts purely from broker instrument metadata — no symbol
    string construction.  All returned dicts carry `instrument_token` so
    callers can subscribe without any further resolution.

    Args:
        kite: broker client (must have .instruments(exchange) method).
        underlying_price: current NIFTY spot price used to compute ATM.
        strike_band: include all strikes within ±strike_band of ATM (default 200).
        strike_step: optional override for the strike rounding step.
                     Auto-detected from the instrument dump when None.

    Returns:
        List of instrument dicts, each containing at minimum:
            instrument_token (int), tradingsymbol (str), strike (float),
            expiry (date), instrument_type (str in {"CE","PE"}).

    Raises:
        RuntimeError when no NIFTY instruments are found or no valid
        contracts match the ATM window.
        ValueError when underlying_price is not positive.
    """
    if underlying_price <= 0:
        raise ValueError(
            f"underlying_price must be positive, got {underlying_price}"
        )

    LOGGER.info(
        "ContractSelector: fetching NFO instruments for ATM=%.2f …",
        underlying_price,
    )
    all_instruments: list[dict] = list(kite.instruments("NFO"))

    # ── filter to NIFTY options only ─────────────────────────────────────────
    nifty_opts = [
        inst for inst in all_instruments
        if str(inst.get("name", "")).upper() == "NIFTY"
        and str(inst.get("instrument_type", "")).upper() in ("CE", "PE")
    ]

    if not nifty_opts:
        raise RuntimeError(
            "ContractSelector: no NIFTY option instruments found in NFO dump. "
            "Check broker authentication."
        )

    # ── resolve nearest valid expiry ─────────────────────────────────────────
    today = date.today()
    expiries: list[date] = []
    for inst in nifty_opts:
        exp = _coerce_expiry(inst.get("expiry"))
        if exp and exp >= today:
            expiries.append(exp)

    if not expiries:
        raise RuntimeError(
            "ContractSelector: no future-dated NIFTY expiries found in NFO dump. "
            "Instrument dump may be stale."
        )

    nearest_expiry = min(expiries)

    # ── auto-detect strike step ───────────────────────────────────────────────
    near_strikes = sorted(
        {
            float(inst.get("strike", 0) or 0)
            for inst in nifty_opts
            if _coerce_expiry(inst.get("expiry")) == nearest_expiry
        }
    )
    if strike_step is None:
        if len(near_strikes) >= 2:
            diffs = [
                near_strikes[i + 1] - near_strikes[i]
                for i in range(min(5, len(near_strikes) - 1))
                if near_strikes[i + 1] - near_strikes[i] > 0
            ]
            detected_step = int(min(diffs)) if diffs else _STRIKE_STEP_DEFAULT
        else:
            detected_step = _STRIKE_STEP_DEFAULT
        strike_step = detected_step

    # ── compute ATM and select contracts ─────────────────────────────────────
    atm = round(underlying_price / strike_step) * strike_step

    selected: list[dict[str, Any]] = []
    for inst in nifty_opts:
        exp = _coerce_expiry(inst.get("expiry"))
        if exp != nearest_expiry:
            continue
        try:
            strike = float(inst.get("strike") or 0)
        except (TypeError, ValueError):
            continue
        if abs(strike - atm) > strike_band:
            continue

        token_raw = inst.get("instrument_token")
        if token_raw is None:
            continue
        try:
            token = int(token_raw)
        except (TypeError, ValueError):
            continue

        selected.append(
            {
                "instrument_token": token,
                "tradingsymbol": str(inst.get("tradingsymbol") or "").strip(),
                "strike": strike,
                "expiry": exp,
                "instrument_type": str(inst.get("instrument_type") or "").upper(),
                "lot_size": inst.get("lot_size"),
                "tick_size": inst.get("tick_size"),
                "exchange": str(inst.get("exchange") or "NFO").upper(),
            }
        )

    if not selected:
        raise RuntimeError(
            f"ContractSelector: no NIFTY contracts found for expiry={nearest_expiry} "
            f"ATM={atm} band=±{strike_band}. Spot={underlying_price}"
        )

    LOGGER.info(
        "ContractSelector: selected %d contracts | expiry=%s | ATM=%s | band=±%s",
        len(selected),
        nearest_expiry,
        atm,
        strike_band,
        extra={
            "event": "contract_selector_done",
            "count": len(selected),
            "nearest_expiry": str(nearest_expiry),
            "atm": atm,
        },
    )
    return selected
