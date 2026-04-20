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

from datetime import date, datetime, timedelta
import logging
import os
from typing import Any

LOGGER = logging.getLogger("nifty_scalper_bot.core.contract_selector")

_STRIKE_BAND = 200      # include strikes ATM ± this many points
_STRIKE_STEP_DEFAULT = 50  # fallback step when no instruments loaded

# Liquidity check knobs — the old 3-day / 10-bar gate was too strict and
# repeatedly rejected ATM contracts for freshly listed weekly series.  The
# defaults are now configurable and lean forgiving, and transient historical
# fetch failures no longer disqualify a contract.
_LIQUIDITY_LOOKBACK_DAYS = max(
    1, int(os.getenv("OPTION_LIQUIDITY_LOOKBACK_DAYS", "7") or 7)
)
_LIQUIDITY_MIN_BARS = max(
    1, int(os.getenv("OPTION_LIQUIDITY_MIN_BARS", "3") or 3)
)
_LIQUIDITY_ENABLED = os.getenv(
    "OPTION_LIQUIDITY_HISTORY_CHECK", "1"
).strip().lower() not in {"0", "false", "no", "off"}


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

    # ── best-effort liquidity screen ─────────────────────────────────────────
    # Historical data gating was previously opt-out and rejected any contract
    # with fewer than 10 minute-bars in the last 3 trading days.  This failed
    # on freshly listed strikes (common on expiry day) and on any transient
    # broker error.  We now run the check as a "demotion" filter: tokens with
    # proven activity move to the head of the list, everything else stays
    # available as an ATM fallback so downstream selection never starves.
    if _LIQUIDITY_ENABLED and selected:
        interval = 'minute'
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=_LIQUIDITY_LOOKBACK_DAYS)
        liquid: list[dict[str, Any]] = []
        fallback: list[dict[str, Any]] = []
        skipped_no_history: list[int] = []
        skipped_errors: list[int] = []
        for contract in selected:
            token = contract['instrument_token']
            try:
                data = kite.historical_data(token, from_dt, to_dt, interval)
            except Exception as exc:  # noqa: BLE001 - best effort
                # Transient broker error — keep the contract but log once.
                skipped_errors.append(token)
                LOGGER.debug(
                    'liquidity_history_fetch_failed token=%s err=%s',
                    token,
                    exc,
                )
                fallback.append(contract)
                continue
            if data and len(data) >= _LIQUIDITY_MIN_BARS:
                liquid.append(contract)
            else:
                skipped_no_history.append(token)
                fallback.append(contract)
        if skipped_no_history:
            LOGGER.info(
                'liquidity_demotion: %d tokens lack %d+ bars over %d days '
                '(kept as ATM fallback)',
                len(skipped_no_history),
                _LIQUIDITY_MIN_BARS,
                _LIQUIDITY_LOOKBACK_DAYS,
                extra={
                    'event': 'contract_selector_liquidity_demotion',
                    'tokens': skipped_no_history,
                    'lookback_days': _LIQUIDITY_LOOKBACK_DAYS,
                    'min_bars': _LIQUIDITY_MIN_BARS,
                },
            )
        if skipped_errors:
            LOGGER.info(
                'liquidity_probe_errors: %d tokens had transient history errors '
                '(kept as ATM fallback)',
                len(skipped_errors),
                extra={
                    'event': 'contract_selector_liquidity_errors',
                    'tokens': skipped_errors,
                },
            )
        # Liquid contracts first, then the demotion bucket — callers that only
        # need the best strikes use the prefix; full universe subscribers keep
        # everything.
        selected = liquid + fallback

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
