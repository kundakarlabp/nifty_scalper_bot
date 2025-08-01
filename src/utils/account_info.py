from kiteconnect import KiteConnect
from config import Config

def get_dynamic_account_balance() -> float:
    kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
    kite.set_access_token(Config.KITE_ACCESS_TOKEN)
    margins = kite.margins(segment="equity")
    return float(margins['available']['cash'])  # or 'net' if you prefer net exposure
