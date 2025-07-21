import time
import requests
from kiteconnect import KiteConnect
from datetime import datetime

# --- Configuration ---
API_KEY = "6jgqkwnlht6r29ug"
API_SECRET = "fgfuhf0fgw9eeme0uudu2n6uexsmhj5t"
ACCESS_TOKEN = "bDP82Wibv31aEFcq8EmbQ9VI1amqWuvW"
TELEGRAM_TOKEN = "7650520741:AAG5A8NNKH67KNetloHd51bCQSrMB3untEM"
TELEGRAM_CHAT_ID = "6931456598"

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# --- Utilities ---
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

def get_ltp(symbol):
    try:
        quote = kite.ltp([symbol])
        return quote[symbol]['last_price']
    except:
        return None

# --- Signal Logic (Dummy Example - Replace with real indicators) ---
def generate_signal():
    now = datetime.now().strftime("%H:%M:%S")
    signal = {
        "type": "BUY CE",  # or "BUY PE"
        "strike": 22500,
        "price": 91,
        "target": 109,
        "sl": 87,
        "confidence": 9,
        "reason": "EMA✔ ADX✔ RSI✔ VWAP✔ BB✔ MACD✔ ST✔ SMC✔ Momentum✔",
        "oi": "+12%",
        "delta": 0.61,
        "theta": -2.4,
        "iv": "16.7%",
        "expiry": "18-Jul-25",
        "timestamp": now
    }
    return signal

# --- Order Execution & Tracking ---
def place_order(trade_type, strike):
    try:
        symbol = f"NIFTY{strike}CE" if "CE" in trade_type else f"NIFTY{strike}PE"
        order = kite.place_order(
            tradingsymbol=symbol,
            exchange=kite.EXCHANGE_NFO,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=50,
            order_type=kite.ORDER_TYPE_MARKET,
            product=kite.PRODUCT_MIS,
            variety=kite.VARIETY_REGULAR
        )
        return symbol, order
    except Exception as e:
        return None, f"❌ Order Failed: {str(e)}"

# --- Main Loop with Live Exit Tracking ---
if __name__ == "__main__":
    last_signal = None
    active_trade = None
    entry_price = 0
    trailing_sl = 0

    while True:
        signal = generate_signal()

        if signal["confidence"] >= 9 and signal != last_signal:
            msg = (
                f"\n📈 {signal['type']} {signal['strike']} — ₹{signal['price']} 🎯{signal['target']} | 🛑{signal['sl']}\n"
                f"⭐ {signal['confidence']}/10 | {signal['reason']}\n"
                f"📊 OI{signal['oi']} | Δ{signal['delta']} | Θ{signal['theta']} | IV {signal['iv']} | Exp: {signal['expiry']}"
            )
            send_telegram(msg)

            symbol, result = place_order(signal["type"], signal["strike"])
            send_telegram(f"🔁 Trade Placed: {signal['type']} {signal['strike']}\nResult: {result}")

            if symbol:
                active_trade = {
                    "symbol": symbol,
                    "sl": signal["sl"],
                    "tp": signal["target"]
                }
                entry_price = signal["price"]
                trailing_sl = entry_price - 1.5  # initial TSL

            last_signal = signal

        # --- Exit Management ---
        if active_trade:
            ltp = get_ltp(active_trade["symbol"])
            if ltp:
                if ltp > entry_price:
                    trailing_sl = max(trailing_sl, ltp - 1.5)

                if ltp >= active_trade["tp"]:
                    send_telegram(f"🎯 Target Hit: {active_trade['symbol']} at ₹{ltp}")
                    active_trade = None
                elif ltp <= trailing_sl:
                    send_telegram(f"🛑 SL/TSL Hit: {active_trade['symbol']} at ₹{ltp}")
                    active_trade = None

        time.sleep(1)
