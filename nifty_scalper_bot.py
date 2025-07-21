# nifty_scalper_bot.py (Live Market Order Version with SL/TP Handling)
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

# --- Dummy Signal Logic (Replace with real logic) ---
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

# --- Order Placement Logic ---
def place_market_order(trade_type, strike):
    try:
        tradingsymbol = f"NIFTY{strike}CE" if "CE" in trade_type else f"NIFTY{strike}PE"
        order = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=QUANTITY,
            order_type=kite.ORDER_TYPE_MARKET,
            product=kite.PRODUCT_MIS
        )
        return tradingsymbol, order['order_id']
    except Exception as e:
        return None, f"❌ Order Failed: {str(e)}"

def place_exit_orders(tradingsymbol, target, sl):
    try:
        ltp = get_ltp(tradingsymbol)
        if not ltp:
            return None, None

        tp_order = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=QUANTITY,
            order_type=kite.ORDER_TYPE_LIMIT,
            price=target,
            product=kite.PRODUCT_MIS
        )

        sl_order = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=QUANTITY,
            order_type=kite.ORDER_TYPE_SL,
            trigger_price=sl,
            price=sl,
            product=kite.PRODUCT_MIS
        )
        return tp_order['order_id'], sl_order['order_id']
    except Exception as e:
        send_telegram(f"❌ Exit Order Failed: {str(e)}")
        return None, None

# --- Real-Time Bot Loop ---
if __name__ == "__main__":
    last_signal = None
    active_trade = None

    while True:
        signal = generate_signal()

        if signal['confidence'] >= 9 and signal != last_signal and not active_trade:
            msg = (
                f"\n📈 {signal['type']} {signal['strike']} — ₹{signal['price']} 🎯{signal['target']} | 🛑{signal['sl']}\n"
                f"⭐ {signal['confidence']}/10 | {signal['reason']}\n"
                f"📊 OI{signal['oi']} | Δ{signal['delta']} | Θ{signal['theta']} | IV {signal['iv']} | Exp: {signal['expiry']}"
            )
            send_telegram(msg)

            symbol, order_id = place_market_order(signal["type"], signal["strike"])
            if symbol:
                tp_id, sl_id = place_exit_orders(symbol, signal["target"], signal["sl"])
                send_telegram(f"📤 Entry: {symbol} Order ID: {order_id}\n🎯 TP Order ID: {tp_id} | 🛑 SL Order ID: {sl_id}")
                active_trade = {
                    "symbol": symbol,
                    "tp": signal['target'],
                    "sl": signal['sl'],
                    "tp_id": tp_id,
                    "sl_id": sl_id
                }
                last_signal = signal

        # Modify SL/TP dynamically (optional logic can be added)

        time.sleep(3)
