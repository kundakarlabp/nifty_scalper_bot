from kiteconnect import KiteConnect
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# --- Helper function to get the correct LTP symbol ---
def _get_spot_ltp_symbol():
    from src.config import Config
    return Config.SPOT_SYMBOL  # e.g., "NSE:NIFTY 50"
# --- End Helper ---

def _extract_base_symbol(config_symbol: str) -> str:
    """
    Converts Config.SPOT_SYMBOL (e.g., 'NSE:NIFTY 50') to base symbol used in instruments (e.g., 'NIFTY')
    """
    try:
        if ":" in config_symbol:
            return config_symbol.split(":")[1].split()[0].strip().upper()
        return config_symbol.split()[0].strip().upper()
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract base symbol from '{config_symbol}': {e}")
        return "NIFTY"  # Fallback to NIFTY

def get_next_expiry_date(kite_instance: KiteConnect) -> str:
    if not kite_instance:
        logger.error("❌ KiteConnect instance is required to find expiry.")
        return ""

    try:
        all_nfo_instruments = kite_instance.instruments("NFO")
        from src.config import Config
        base_symbol = _extract_base_symbol(Config.SPOT_SYMBOL)

        index_instruments = [i for i in all_nfo_instruments if i['name'] == base_symbol]
        if not index_instruments:
            logger.error(f"❌ No instruments found for index base '{base_symbol}' from SPOT_SYMBOL='{Config.SPOT_SYMBOL}'.")
            return ""

        unique_expiries = sorted(set(i['expiry'] for i in index_instruments))
        today = datetime.today().date()

        for expiry_date in unique_expiries:
            if expiry_date >= today:
                return expiry_date.strftime('%Y-%m-%d')

        logger.warning("⚠️ No future expiry found. Using latest available.")
        return unique_expiries[-1].strftime('%Y-%m-%d')

    except Exception as e:
        logger.error(f"❌ Error in get_next_expiry_date: {e}", exc_info=True)
        return ""

def _format_expiry_for_symbol(expiry_str: str) -> str:
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        return expiry_date.strftime("%y%b%d").upper()
    except ValueError as e:
        logger.error(f"❌ Error formatting expiry date '{expiry_str}': {e}")
        return ""

def get_instrument_tokens(symbol: str = "NIFTY", offset: int = 0, kite_instance: KiteConnect = None):
    if not kite_instance:
        logger.error("❌ KiteConnect instance is required.")
        return None

    try:
        from src.config import Config
        base_symbol = _extract_base_symbol(Config.SPOT_SYMBOL)

        instruments = kite_instance.instruments("NFO")
        spot_instruments = kite_instance.instruments("NSE")
        spot_symbol_ltp = _get_spot_ltp_symbol()

        ltp_data = kite_instance.ltp([spot_symbol_ltp])
        spot_price = ltp_data.get(spot_symbol_ltp, {}).get('last_price')

        if spot_price is None:
            logger.error(f"❌ Failed to fetch spot price for LTP symbol '{spot_symbol_ltp}'.")
            return None
        logger.info(f"✅ Spot price fetched: {spot_price} using {spot_symbol_ltp}")

        atm_strike = round(spot_price / 50) * 50 + (offset * 50)
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance)
        if not expiry_yyyy_mm_dd:
            logger.error("❌ No valid expiry date found.")
            return None

        expiry_symbol = _format_expiry_for_symbol(expiry_yyyy_mm_dd)
        if not expiry_symbol:
            logger.error("❌ Could not format expiry date.")
            return None

        logger.debug(f"🔍 Searching {base_symbol} | Spot: {spot_price} | Strike: {atm_strike} | Expiry: {expiry_yyyy_mm_dd} ({expiry_symbol})")

        filtered_instruments = [i for i in instruments if i['name'] == base_symbol]
        logger.debug(f"Found {len(filtered_instruments)} instruments for {base_symbol}")

        ce_token = pe_token = ce_symbol = pe_symbol = None

        expected_ce = f"{base_symbol}{expiry_symbol}{atm_strike}CE"
        expected_pe = f"{base_symbol}{expiry_symbol}{atm_strike}PE"
        logger.debug(f"Expected symbols: CE={expected_ce}, PE={expected_pe}")

        for inst in filtered_instruments:
            if inst['expiry'].strftime('%Y-%m-%d') != expiry_yyyy_mm_dd:
                continue
            if inst['tradingsymbol'] == expected_ce:
                ce_symbol, ce_token = inst['tradingsymbol'], inst['instrument_token']
                logger.debug(f"✅ CE token: {ce_token}")
            elif inst['tradingsymbol'] == expected_pe:
                pe_symbol, pe_token = inst['tradingsymbol'], inst['instrument_token']
                logger.debug(f"✅ PE token: {pe_token}")

        if not ce_token and not pe_token:
            logger.warning(f"⚠️ Could not find CE/PE tokens for {base_symbol} {atm_strike} on {expiry_yyyy_mm_dd}")

        return {
            "spot_price": spot_price,
            "spot_token": None,  # Add if needed
            "atm_strike": atm_strike,
            "ce_symbol": ce_symbol,
            "ce_token": ce_token,
            "pe_symbol": pe_symbol,
            "pe_token": pe_token,
            "expiry": expiry_yyyy_mm_dd
        }

    except Exception as e:
        logger.error(f"❌ Error in get_instrument_tokens: {e}", exc_info=True)
        return None