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
QUANTITY = 50

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# --- Telegram Alert ---
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

# --- Get Live Price ---
def get_ltp(symbol):
    try:
        quote = kite.ltp([symbol])
        return quote[symbol]['last_price']
    except:
        return None

# --- Signal Generator (dummy logic) ---
def generate_signal():
    now = datetime.now().strftime("%H:%M:%S")
    return {
        "type": "BUY CE",
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

# --- Place Zerodha GTT Order ---
def place_gtt_order(trade_type, strike, entry, sl, tp):
    try:
        symbol = f"NIFTY{strike}CE" if "CE" in trade_type else f"NIFTY{strike}PE"
        gtt_params = {
            "tradingsymbol": symbol,
            "exchange": kite.EXCHANGE_NFO,
            "trigger_type": kite.GTT_TYPE_OCO,
            "last_price": entry,
            "orders": [
                {"transaction_type": kite.TRANSACTION_TYPE_SELL, "quantity": QUANTITY, "price": tp, "order_type": kite.ORDER_TYPE_LIMIT, "product": kite.PRODUCT_MIS},
                {"transaction_type": kite.TRANSACTION_TYPE_SELL, "quantity": QUANTITY, "price": sl, "order_type": kite.ORDER_TYPE_LIMIT, "product": kite.PRODUCT_MIS}
            ]
        }
        response = kite.place_gtt(**gtt_params)
        return symbol, response['trigger_id']
    except Exception as e:
        return None, f"❌ GTT Failed: {str(e)}"

# --- Main Loop ---
if __name__ == "__main__":
    last_signal = None
    active_trade = None

    while True:
        signal = generate_signal()

        if signal['confidence'] >= 9 and signal != last_signal and not active_trade:
            msg = (
                f"📈 {signal['type']} {signal['strike']} — ₹{signal['price']} 🎯{signal['target']} | 🛑{signal['sl']}\n"
                f"⭐ {signal['confidence']}/10 | {signal['reason']}\n"
                f"📊 OI{signal['oi']} | Δ{signal['delta']} | Θ{signal['theta']} | IV {signal['iv']} | Exp: {signal['expiry']}"
            )
            send_telegram(msg)

            symbol, gtt_id = place_gtt_order(signal["type"], signal["strike"], signal["price"], signal["sl"], signal["target"])
            send_telegram(f"📤 GTT Placed: {symbol} ID: {gtt_id}")

            if symbol:
                active_trade = {
                    "symbol": symbol,
                    "gtt_id": gtt_id,
                    "entry": signal["price"],
                    "sl": signal["sl"],
                    "tp": signal["target"]
                }
                last_signal = signal

        # SL/TP Auto-Review & Update
        if active_trade:
            ltp = get_ltp(active_trade["symbol"])
            if ltp:
                new_tp = round(ltp + 15, 1)
                new_sl = round(ltp - 10, 1)
                if new_tp > active_trade["tp"] or new_sl > active_trade["sl"]:
                    try:
                        kite.delete_gtt(active_trade["gtt_id"])
                        symbol, gtt_id = place_gtt_order("BUY CE", int(active_trade["symbol"][5:-2]), ltp, new_sl, new_tp)
                        active_trade.update({"gtt_id": gtt_id, "sl": new_sl, "tp": new_tp})
                        send_telegram(f"🔁 GTT Modified: SL={new_sl} TP={new_tp} for {symbol}")
                    except:
                        pass

        time.sleep(3)
