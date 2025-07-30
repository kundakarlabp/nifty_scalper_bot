# src/utils/strike_selector.py
"""
Utility functions for selecting Nifty 50 Index Option strikes.
This module helps the bot automatically choose which options contracts to trade
based on the spot price, expiry date, and selection criteria (ATM, ITM, OTM).
"""
import logging
import math
from typing import List, Optional, Dict, Any
from kiteconnect import KiteConnect

# Standard lot size for Nifty Index Options
NIFTY_OPTION_LOT_SIZE = 50
# Standard strike price step for Nifty Index Options
NIFTY_STRIKE_STEP = 50

logger = logging.getLogger(__name__)

def get_nearest_strike(target_price: float, step: int = NIFTY_STRIKE_STEP) -> int:
    """
    Calculates the nearest strike price based on a target price and a standard step.

    This function rounds a given price (like the Nifty 50 spot price) to the nearest
    valid strike price for options. For Nifty, strikes are typically in increments of 50.

    Args:
        target_price (float): The price to round (e.g., the current Nifty 50 spot price).
        step (int): The standard interval between strikes (default 50 for Nifty).

    Returns:
        int: The nearest valid strike price (e.g., 18000, 18050).

    Example:
        >>> get_nearest_strike(18023.45)
        18000
        >>> get_nearest_strike(18025.0)
        18050
    """
    return int(round(target_price / step) * step)

def get_itm_otm_strikes(atm_strike: int,
                       strike_step: int = NIFTY_STRIKE_STEP,
                       offset: int = 1) -> tuple[int, int]:
    """
    Calculates the ITM and OTM strikes adjacent to a given ATM strike.

    This is a helper function if you want to explicitly calculate nearby strikes.
    Note: The definition of ITM/OTM depends on Call (CE) or Put (PE).
    For a Call: ITM is lower than ATM, OTM is higher.
    For a Put: ITM is higher than ATM, OTM is lower.

    Args:
        atm_strike (int): The calculated At-The-Money strike price.
        strike_step (int): The standard interval between strikes (default 50).
        offset (int): Number of steps away from ATM (default 1 for nearest).

    Returns:
        tuple[int, int]: A tuple containing (ITM strike, OTM strike) based on the offset.
                         This is a simplified absolute calculation.
    """
    itm_strike = atm_strike - (offset * strike_step)
    otm_strike = atm_strike + (offset * strike_step)
    return itm_strike, otm_strike

def select_nifty_option_strikes(
    kite: KiteConnect,
    expiry: str, # Format: 'YYYY-MM-DD'
    option_type: str, # 'CE' for Call, 'PE' for Put
    strike_criteria: str = "ATM", # e.g., 'ATM', 'ITM', 'OTM', 'ATM+1', 'ATM-1'
    instrument_mapping: Optional[Dict[int, Dict[str, Any]]] = None
) -> Optional[List[int]]:
    """
    Selects Nifty 50 Index Option instrument tokens based on specified criteria.

    This is the main function. It fetches the current Nifty 50 spot price,
    calculates the target strike based on `strike_criteria`, fetches the list
    of Nifty options instruments, and finds the matching instrument token(s).

    Args:
        kite (KiteConnect): An authenticated KiteConnect instance. This is needed
                            to fetch the spot price and the instrument list.
        expiry (str): The expiry date for the options in 'YYYY-MM-DD' format.
                      Example: '2023-12-28'.
        option_type (str): The type of option, either 'CE' (Call) or 'PE' (Put).
        strike_criteria (str): The rule for selecting the strike price.
            - 'ATM': At The Money (nearest to spot price).
            - 'ITM': In The Money (nearest).
                - For CE: Strike < Spot (requires knowing spot and option type).
                - For PE: Strike > Spot.
            - 'OTM': Out The Money (nearest).
                - For CE: Strike > Spot.
                - For PE: Strike < Spot.
            - 'ATM+1': The next Out-Of-The-Money strike.
            - 'ATM-1': The next In-The-Money strike.
            (You can extend this logic for ATM+2, etc.)
        instrument_mapping (Optional[Dict]): A pre-fetched dictionary mapping
            instrument tokens to their full details (from `kite.instruments()`).
            If provided, it avoids an extra API call. If None, the function
            will call `kite.instruments("NFO")`.

    Returns:
        Optional[List[int]]: A list containing the instrument token(s) for the
                           selected strike(s). Returns None if an error occurs
                           or no matching instrument is found.

    Example Usage (Conceptual):
        # Assuming `kite` is an authenticated KiteConnect object
        # and `expiry_date` is a string like "2023-12-28"

        # Select ATM Call
        ce_tokens = select_nifty_option_strikes(kite, expiry_date, "CE", "ATM")
        if ce_tokens:
            print(f"Selected ATM CE Token: {ce_tokens[0]}")

        # Select ATM Put
        pe_tokens = select_nifty_option_strikes(kite, expiry_date, "PE", "ATM")
        if pe_tokens:
            print(f"Selected ATM PE Token: {pe_tokens[0]}")
    """
    try:
        # --- 1. Get Nifty 50 Spot Price ---
        # Using kite.ltp() is a standard and efficient way to get the last traded price.
        # The instrument identifier for Nifty 50 Index on NSE is "NSE:NIFTY 50".
        ltp_data = kite.ltp(["NSE:NIFTY 50"])
        nifty_ltp = ltp_data.get("NSE:NIFTY 50", {}).get('last_price')

        if not nifty_ltp:
            logger.error("❌ Could not fetch Nifty 50 spot price using kite.ltp().")
            return None # Indicate failure

        logger.info(f"📈 Fetched Nifty 50 Spot Price: {nifty_ltp}")

        # --- 2. Calculate Target Strike Price ---
        atm_strike = get_nearest_strike(nifty_ltp, NIFTY_STRIKE_STEP)
        logger.debug(f"🎯 Calculated ATM Strike: {atm_strike}")

        target_strike = atm_strike
        # Determine the target strike based on the criteria
        if strike_criteria.upper() == "ITM":
            # Simplified logic: For CE, ITM is lower; for PE, ITM is higher.
            # This picks the adjacent strike in the ITM direction.
            offset = -1 if option_type.upper() == "CE" else 1
            target_strike = atm_strike + (offset * NIFTY_STRIKE_STEP)
        elif strike_criteria.upper() == "OTM":
            # Simplified logic: For CE, OTM is higher; for PE, OTM is lower.
            offset = 1 if option_type.upper() == "CE" else -1
            target_strike = atm_strike + (offset * NIFTY_STRIKE_STEP)
        elif strike_criteria.upper() == "ATM+1":
            # Next OTM strike
            offset = 1 if option_type.upper() == "CE" else -1
            target_strike = atm_strike + (offset * NIFTY_STRIKE_STEP)
        elif strike_criteria.upper() == "ATM-1":
            # Next ITM strike
            offset = -1 if option_type.upper() == "CE" else 1
            target_strike = atm_strike + (offset * NIFTY_STRIKE_STEP)
        # You can add more elif blocks for 'ATM+2', 'ATM-3', etc., if needed.

        logger.info(
            f"🎯 Target Strike Determined: {target_strike} "
            f"(Criteria: {strike_criteria}, Option Type: {option_type})"
        )

        # --- 3. Get Instrument List ---
        # Fetch all instruments on the NFO (National Stock Exchange - Futures & Options) segment.
        # This is a potentially large list, so it's good to do it once.
        instruments = instrument_mapping
        if instruments is None:
            logger.debug("📥 Fetching full NFO instrument list from Kite...")
            instruments_list = kite.instruments("NFO")
            # Convert list to a dictionary for O(1) average time complexity lookups.
            # Key: instrument_token, Value: full instrument details dictionary.
            instruments = {inst['instrument_token']: inst for inst in instruments_list}
            logger.debug(f"📥 Fetched and indexed {len(instruments)} NFO instruments.")

        # --- 4. Find Matching Instrument Token(s) ---
        selected_tokens = []
        # Iterate through the instrument dictionary to find matches.
        for token, details in instruments.items():
            # Match exchange (must be NFO)
            if details.get('exchange') != 'NFO':
                continue
            # Match name (should be 'NIFTY' for Nifty Index Options)
            if details.get('name') != 'NIFTY':
                continue
            # Match expiry date (format from Kite is usually date object, compare as string)
            # Ensure the expiry date format matches what Kite provides.
            instrument_expiry = details.get('expiry')
            if instrument_expiry and instrument_expiry.strftime('%Y-%m-%d') != expiry:
                continue
            # Match strike price
            if details.get('strike') != target_strike:
                continue
            # Match instrument type (CE or PE)
            if details.get('instrument_type') != option_type.upper():
                continue

            # If all criteria match, we found our instrument
            selected_tokens.append(token)
            trading_symbol = details.get('tradingsymbol')
            logger.info(
                f"✅ Match Found: Token={token}, Symbol={trading_symbol}, "
                f"Strike={target_strike}, Type={option_type.upper()}, Expiry={expiry}"
            )
            # For a given expiry, strike, and type, there should typically be only one match.
            # Breaking here is efficient. If you anticipate multiple (e.g., different
            # varieties), you might want to collect them all.
            break

        # --- 5. Return Result ---
        if not selected_tokens:
            logger.error(
                f"❌ No matching Nifty option found for "
                f"Expiry={expiry}, Strike={target_strike}, Type={option_type.upper()}"
            )
            return None # Indicate no match found

        logger.info(f"✅ Final Selected Token(s): {selected_tokens}")
        return selected_tokens

    except Exception as e:
        # Catch any unexpected errors during the process
        logger.error(f"❌ An error occurred in select_nifty_option_strikes: {e}", exc_info=True)
        return None # Indicate failure

# Example of how this might be used within main.py or another module:
# def example_usage():
#     from kiteconnect import KiteConnect
#     # You need a valid, authenticated KiteConnect instance
#     # kite = KiteConnect(api_key="YOUR_API_KEY")
#     # kite.set_access_token("YOUR_VALID_ACCESS_TOKEN")
#
#     expiry_date = "2023-12-28" # This should be determined dynamically, e.g., by expiry_selector
#
#     # Select ATM Call
#     ce_tokens = select_nifty_option_strikes(kite, expiry_date, "CE", "ATM")
#     # Select ATM Put
#     pe_tokens = select_nifty_option_strikes(kite, expiry_date, "PE", "ATM")
#
#     if ce_tokens and pe_tokens:
#         print("Selected ATM Straddle Tokens:", ce_tokens[0], pe_tokens[0])
#     else:
#         print("Failed to select one or both strikes.")
