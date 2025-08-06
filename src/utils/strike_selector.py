# src/utils/strike_selector.py

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def get_next_expiry_date(kite_instance: KiteConnect) -> str:
    """
    Finds the next expiry date (Thursday) that has available instruments.
    Returns the date in YYYY-MM-DD format.
    """
    today = datetime.today()
    # Find the next Thursday
    days_ahead = (3 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7 # If today is Thursday, get next Thursday
    next_thursday = today + timedelta(days=days_ahead)

    # Format for Kite API (YYYY-MM-DD)
    return next_thursday.strftime("%Y-%m-%d")

def _format_expiry_for_symbol(expiry_str: str) -> str:
    """
    Formats a YYYY-MM-DD expiry string into the NSE option symbol format (YYMONDD).
    E.g., '2025-08-07' -> '25AUG07'
    """
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        # Format: YYMONDD
        return expiry_date.strftime("%y%b%d").upper()
    except ValueError as e:
        logger.error(f"Error formatting expiry date '{expiry_str}': {e}")
        return ""

def get_instrument_tokens(symbol: str = "NIFTY", offset: int = 0, kite_instance: KiteConnect = None):
    """
    Gets instrument tokens for spot, ATM CE, and PE for the next expiry.
    Offset allows getting ITM/OTM strikes.
    Note: This function assumes the 'symbol' base is standard (e.g., 'NIFTY').
    """
    if not kite_instance:
        logger.error("KiteConnect instance is required.")
        return None

    try:
        # Fetch instruments
        instruments = kite_instance.instruments("NFO")
        spot_instruments = kite_instance.instruments("NSE") # For spot price

        # 1. Get Spot Price
        # The spot symbol on NSE is often just the name, e.g., "NIFTY 50"
        # The LTP request usually uses "NSE:" prefix
        spot_symbol_ltp = f"NSE:{symbol}" # Try with prefix first
        ltp_data = kite_instance.ltp([spot_symbol_ltp])
        spot_price = ltp_data.get(spot_symbol_ltp, {}).get('last_price')

        if spot_price is None:
            logger.debug(f"Could not fetch spot price for {spot_symbol_ltp}, trying without prefix.")
            # Fallback: Try without 'NSE:' prefix
            ltp_data_fallback = kite_instance.ltp([symbol])
            spot_price = ltp_data_fallback.get(symbol, {}).get('last_price')
            if spot_price is None:
                logger.error(f"Failed to fetch spot price for '{symbol}'.")
                return None

        # 2. Calculate Strike
        atm_strike = round(spot_price / 50) * 50 + (offset * 50)
        # Get expiry in YYYY-MM-DD format
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance)
        # Format expiry for symbol construction (YYMONDD)
        expiry_for_symbol = _format_expiry_for_symbol(expiry_yyyy_mm_dd)

        if not expiry_for_symbol:
            logger.error("Could not format expiry date for symbol construction.")
            return None

        logger.debug(f"Searching for {symbol} options. Spot: {spot_price}, ATM Strike: {atm_strike}, Expiry (YYYY-MM-DD): {expiry_yyyy_mm_dd}, Expiry (Symbol Format): {expiry_for_symbol}")

        # 3. Find Symbols and Tokens
        ce_symbol = None
        pe_symbol = None
        ce_token = None
        pe_token = None
        spot_token = None

        # Find Spot Token
        # Spot symbol on NSE might be just "NIFTY 50" or similar.
        # The instrument list might list it differently than the LTP request.
        for inst in spot_instruments:
             # Common ways spot might be listed in instruments
             if inst['tradingsymbol'] == symbol or inst['tradingsymbol'] == spot_symbol_ltp.replace("NSE:", ""):
                  spot_token = inst['instrument_token']
                  logger.debug(f"Found spot token: {spot_token} for symbol: {inst['tradingsymbol']}")
                  break

        # Find Option Tokens using the correctly formatted symbol
        expected_ce_symbol = f"{symbol.replace(' ', '')}{expiry_for_symbol}{atm_strike}CE"
        expected_pe_symbol = f"{symbol.replace(' ', '')}{expiry_for_symbol}{atm_strike}PE"
        logger.debug(f"Looking for CE symbol: {expected_ce_symbol}")
        logger.debug(f"Looking for PE symbol: {expected_pe_symbol}")

        for inst in instruments:
            # Check expiry match (Kite returns datetime.date)
            inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
            if inst_expiry_str == expiry_yyyy_mm_dd:
                # Match the full trading symbol
                if inst['tradingsymbol'] == expected_ce_symbol:
                    ce_symbol = inst['tradingsymbol']
                    ce_token = inst['instrument_token']
                    logger.debug(f"Found CE Token: {ce_token} for {ce_symbol}")
                elif inst['tradingsymbol'] == expected_pe_symbol:
                    pe_symbol = inst['tradingsymbol']
                    pe_token = inst['instrument_token']
                    logger.debug(f"Found PE Token: {pe_token} for {pe_symbol}")

        if not ce_token or not pe_token:
            logger.warning(f"Could not find tokens for {symbol} {expiry_yyyy_mm_dd} Strike:{atm_strike} CE/PE. Expected Symbols: {expected_ce_symbol}, {expected_pe_symbol}")

        return {
            "spot_price": spot_price,
            "spot_token": spot_token,
            "atm_strike": atm_strike,
            "ce_symbol": ce_symbol,
            "ce_token": ce_token,
            "pe_symbol": pe_symbol,
            "pe_token": pe_token,
            "expiry": expiry_yyyy_mm_dd # Return the standard YYYY-MM-DD expiry
        }
    except Exception as e:
        logger.error(f"Error in get_instrument_tokens: {e}", exc_info=True)
        return None