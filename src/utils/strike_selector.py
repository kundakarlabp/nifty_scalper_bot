# src/utils/strike_selector.py

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def get_next_expiry_date(kite_instance: KiteConnect) -> str:
    """
    Finds the next expiry date (Thursday) that has available instruments.
    """
    today = datetime.today()
    # Find the next Thursday
    days_ahead = (3 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7 # If today is Thursday, get next Thursday
    next_thursday = today + timedelta(days=days_ahead)
    
    # Format for Kite API (YYYY-MM-DD)
    return next_thursday.strftime("%Y-%m-%d")

def get_instrument_tokens(symbol: str = "NIFTY", offset: int = 0, kite_instance: KiteConnect = None):
    """
    Gets instrument tokens for spot, ATM CE, and PE for the next expiry.
    Offset allows getting ITM/OTM strikes.
    """
    if not kite_instance:
        logger.error("KiteConnect instance is required.")
        return None

    try:
        instruments = kite_instance.instruments("NFO")
        spot_instruments = kite_instance.instruments("NSE") # For spot price
        
        # 1. Get Spot Price
        spot_symbol_full = f"NSE:{symbol}" # Adjust if needed (e.g., NSE:NIFTY 50)
        ltp_data = kite_instance.ltp([spot_symbol_full])
        spot_price = ltp_data.get(spot_symbol_full, {}).get('last_price')
        
        if spot_price is None:
            logger.warning(f"Could not fetch spot price for {spot_symbol_full}")
            # Fallback: Try without 'NSE:' prefix if that's how it's listed
            ltp_data_fallback = kite_instance.ltp([symbol])
            spot_price = ltp_data_fallback.get(symbol, {}).get('last_price')
            if spot_price is None:
                logger.error("Failed to fetch spot price.")
                return None

        # 2. Calculate Strike
        atm_strike = round(spot_price / 50) * 50 + (offset * 50)
        expiry = get_next_expiry_date(kite_instance)

        # 3. Find Symbols and Tokens
        ce_symbol = None
        pe_symbol = None
        ce_token = None
        pe_token = None
        spot_token = None

        # Find Spot Token
        for inst in spot_instruments:
             # Adjust condition based on how your spot symbol is listed
             if inst['tradingsymbol'] == symbol or inst['tradingsymbol'] == spot_symbol_full.replace("NSE:", ""):
                  spot_token = inst['instrument_token']
                  break

        # Find Option Tokens
        for inst in instruments:
            # Check expiry match (Kite returns datetime.date)
            inst_expiry_str = inst['expiry'].strftime("%Y-%m-%d") if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
            if inst_expiry_str == expiry:
                if f"{symbol}{expiry.replace('-', '')}{atm_strike}CE" in inst['tradingsymbol']:
                    ce_symbol = inst['tradingsymbol']
                    ce_token = inst['instrument_token']
                elif f"{symbol}{expiry.replace('-', '')}{atm_strike}PE" in inst['tradingsymbol']:
                    pe_symbol = inst['tradingsymbol']
                    pe_token = inst['instrument_token']
        
        if not ce_token or not pe_token:
            logger.warning(f"Could not find tokens for {symbol} {expiry} {atm_strike}CE/PE")

        return {
            "spot_price": spot_price,
            "spot_token": spot_token,
            "atm_strike": atm_strike,
            "ce_symbol": ce_symbol,
            "ce_token": ce_token,
            "pe_symbol": pe_symbol,
            "pe_token": pe_token,
            "expiry": expiry
        }
    except Exception as e:
        logger.error(f"Error in get_instrument_tokens: {e}", exc_info=True)
        return None
