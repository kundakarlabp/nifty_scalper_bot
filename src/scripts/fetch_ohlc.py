import os
import pandas as pd
from kiteconnect import KiteConnect

def fetch_ohlc(instrument_token, from_date, to_date, interval):
    kite = KiteConnect(api_key=os.environ["ZERODHA_API_KEY"])
    kite.set_access_token(os.environ["ZERODHA_ACCESS_TOKEN"])

    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
        continuous=False
    )
    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)
    return df

if __name__ == "__main__":
    from_date = "2024-07-01"
    to_date = "2024-07-12"
    interval = "5minute"
    instrument_token = 256265  # NIFTY 50 index

    df = fetch_ohlc(instrument_token, from_date, to_date, interval)
    df.to_csv("data/nifty_ohlc.csv")
    print("✅ Saved: data/nifty_ohlc.csv")
