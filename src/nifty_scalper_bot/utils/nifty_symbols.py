# src/nifty_scalper_bot/utils/nifty_symbols.py
from datetime import datetime, timedelta
from kiteconnect import KiteConnect

kite: KiteConnect  # will be injected at runtime

def next_tuesday() -> datetime:
    today = datetime.now().date()
    days_ahead = (1 - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return datetime.combine(today + timedelta(days=days_ahead), datetime.min.time())

def is_monthly_expiry(expiry: datetime) -> bool:
    return (expiry + timedelta(days=7)).month != expiry.month

def nifty_symbol(expiry: datetime, strike: int, kind: str) -> str:
    yy = expiry.strftime("%y")
    if is_monthly_expiry(expiry):
        mmm = expiry.strftime("%b").upper()
        return f"NIFTY{yy}{mmm}{strike}{kind}"
    else:
        month_code = "123456789OND"[expiry.month - 1]
        dd = expiry.strftime("%d")
        return f"NIFTY{yy}{month_code}{dd}{strike}{kind}"

def get_live_symbols(kite_instance) -> list[str]:
    global kite
    kite = kite_instance
    expiry = next_tuesday()
    spot = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
    strike = int(round(spot / 50) * 50)
    
    syms = [
        f"NFO:{nifty_symbol(expiry, strike, 'CE')}",
        f"NFO:{nifty_symbol(expiry, strike, 'PE')}",
    ]
    return syms
