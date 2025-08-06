from kiteconnect import KiteConnect
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def _get_spot_ltp_symbol():
    from src.config import Config
    return Config.SPOT_SYMBOL  # e.g., "NSE:NIFTY 50"

def get_next_expiry_date(kite_instance: KiteConnect) -> str:
    if not kite_instance:
        logger.error("KiteConnect instance is required to find expiry.")
        return ""

    try:
        all_nfo_instruments = kite_instance.instruments("NFO")
        from src.config import Config
        spot_symbol_config = Config.SPOT_SYMBOL

        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1]
        else:
            potential_name_with_spaces = spot_symbol_config

        base_symbol_for_search = potential_name_with_spaces.split()[0]

        index_instruments = [inst for inst in all_nfo_instruments if inst['name'] == base_symbol_for_search]

        if not index_instruments:
            logger.error(f"No instruments found for base name '{base_symbol_for_search}'.")
            return ""

        unique_expiries = set(inst['expiry'] for inst in index_instruments)
        sorted_expiries = sorted(unique_expiries)
        today = datetime.today().date()

        for expiry_date in sorted_expiries:
            expiry_obj = expiry_date if not isinstance(expiry_date, str) else datetime.strptime(expiry_date, "%Y-%m-%d").date()
            if expiry_obj >= today:
                return expiry_obj.strftime('%Y-%m-%d')

        return sorted_expiries[-1].strftime('%Y-%m-%d') if hasattr(sorted_expiries[-1], 'strftime') else str(sorted_expiries[-1])

    except Exception as e:
        logger.error(f"Error in get_next_expiry_date: {e}", exc_info=True)
        return ""

def _format_expiry_for_symbol(expiry_str: str) -> str:
    """Formats 'YYYY-MM-DD' to 'YYMON' e.g., '2025-08-07' -> '25AUG'."""
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        return expiry_date.strftime("%y%b").upper()
    except ValueError as e:
        logger.error(f"Error formatting expiry date '{expiry_str}': {e}")
        return ""

def get_instrument_tokens(symbol: str = "NIFTY", offset: int = 0, kite_instance: KiteConnect = None):
    if not kite_instance:
        logger.error("KiteConnect instance is required.")
        return None

    try:
        from src.config import Config
        spot_symbol_config = Config.SPOT_SYMBOL

        if ':' in spot_symbol_config:
            potential_name_with_spaces = spot_symbol_config.split(':', 1)[1]
        else:
            potential_name_with_spaces = spot_symbol_config

        base_symbol_for_search = potential_name_with_spaces.split()[0]

        instruments = kite_instance.instruments("NFO")
        spot_symbol_ltp = _get_spot_ltp_symbol()
        ltp_data = kite_instance.ltp([spot_symbol_ltp])
        spot_price = ltp_data.get(spot_symbol_ltp, {}).get('last_price')

        if spot_price is None:
            logger.error(f"Failed to fetch spot price for LTP symbol '{spot_symbol_ltp}'.")
            return None

        logger.info(f"Spot price fetched: {spot_price} for {spot_symbol_ltp}")
        atm_strike = round(spot_price / 50) * 50 + (offset * 50)
        expiry_yyyy_mm_dd = get_next_expiry_date(kite_instance)

        if not expiry_yyyy_mm_dd:
            logger.error("Could not determine valid expiry date.")
            return None

        expiry_for_symbol = _format_expiry_for_symbol(expiry_yyyy_mm_dd)

        if not expiry_for_symbol:
            logger.error("Could not format expiry for symbol.")
            return None

        expected_ce_symbol = f"{base_symbol_for_search}{expiry_for_symbol}{atm_strike}CE"
        expected_pe_symbol = f"{base_symbol_for_search}{expiry_for_symbol}{atm_strike}PE"

        index_instruments = [inst for inst in instruments if inst['name'] == base_symbol_for_search]
        logger.debug(f"Found {len(index_instruments)} instruments for {base_symbol_for_search}.")

        ce_token, pe_token, ce_symbol, pe_symbol = None, None, None, None

        for inst in index_instruments:
            expiry_str = inst['expiry'].strftime('%Y-%m-%d') if hasattr(inst['expiry'], 'strftime') else str(inst['expiry'])
            if expiry_str == expiry_yyyy_mm_dd:
                if inst['tradingsymbol'] == expected_ce_symbol:
                    ce_symbol = inst['tradingsymbol']
                    ce_token = inst['instrument_token']
                    logger.debug(f"✅ CE Token: {ce_token} ({ce_symbol})")
                elif inst['tradingsymbol'] == expected_pe_symbol:
                    pe_symbol = inst['tradingsymbol']
                    pe_token = inst['instrument_token']
                    logger.debug(f"✅ PE Token: {pe_token} ({pe_symbol})")

        if not ce_token and not pe_token:
            logger.warning(f"❌ Tokens not found for {expected_ce_symbol} / {expected_pe_symbol}")

        return {
            "spot_price": spot_price,
            "spot_token": None,
            "atm_strike": atm_strike,
            "ce_symbol": ce_symbol,
            "ce_token": ce_token,
            "pe_symbol": pe_symbol,
            "pe_token": pe_token,
            "expiry": expiry_yyyy_mm_dd
        }

    except Exception as e:
        logger.error(f"Error in get_instrument_tokens: {e}", exc_info=True)
        return None