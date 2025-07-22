# NiftyNinja: Advanced Autonomous Nifty Options Scalper 🥷

import os
import time
import threading
import logging
import hashlib
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import requests

# --- ENVIRONMENT CONFIGURATION ---
API_KEY = os.getenv("Z_API_KEY")
API_SECRET = os.getenv("Z_API_SECRET")
ACCESS_TOKEN = os.getenv("Z_ACCESS_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
LOT_SIZE = 75
RISK_PER_TRADE = 0.01  # 1% risk per trade

# --- KITE SETUP ---
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# --- LOGGING SETUP ---
logging.basicConfig(filename="niftyninja.log", level=logging.INFO, format="%(asctime)s - %(message)s")
def log(msg):
    print(msg)
    logging.info(msg)

# --- TELEGRAM ---
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        log(f"Telegram Error: {e}")

# --- LTP FETCH ---
def get_ltp(symbol):
    try:
        quote = kite.ltp([symbol])
        return quote[symbol]['last_price']
    except Exception as e:
        log(f"LTP Fetch Error: {e}")
        return None

# --- STRIKE BUILDER ---
def get_strike_symbol(strike, opt_type="CE"):
    today = datetime.now().date()
    while today.weekday() != 3:  # Next Wednesday
        today += timedelta(days=1)
    expiry = today.strftime('%d%b%y').upper()
    return f"NIFTY{expiry}{strike}{opt_type}"

# --- SIGNAL GENERATION (Dummy, Replace with ML Logic) ---
def generate_signal():
    now = datetime.now().strftime("%H:%M:%S")
    return {
        "type": "BUY CE",
        "strike": 22500,
        "price": 91,
        "target": 109,
        "sl": 87,
        "confidence": 9.2,
        "reason": "EMA✔ ADX✔ RSI✔ VWAP✔ BB✔ MACD✔ ST✔ SMC✔ Momentum✔",
        "oi": "+12%", "delta": 0.61, "theta": -2.4, "iv": "16.7%",
        "expiry": "18-Jul-25", "timestamp": now
    }

# --- SIGNAL HASHING ---
def signal_hash(signal):
    base = f"{signal['type']}-{signal['strike']}-{signal['price']}-{signal['target']}-{signal['sl']}"
    return hashlib.sha256(base.encode()).hexdigest()

# --- POSITION SIZING ---
def get_position_size(price):
    try:
        profile = kite.margins("equity")
        capital = profile['available']['cash']
        risk_amount = capital * RISK_PER_TRADE
        loss_per_lot = abs(price - signal['sl']) * LOT_SIZE
        lots = max(1, int(risk_amount / loss_per_lot))
        return lots * LOT_SIZE
    except Exception as e:
        log(f"Position size error: {e}")
        return LOT_SIZE

# --- ENTRY ORDER ---
def place_market_order(signal):
    try:
        opt_type = "CE" if "CE" in signal['type'] else "PE"
        tradingsymbol = get_strike_symbol(signal['strike'], opt_type)
        qty = get_position_size(signal['price'])
        order = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=qty,
            order_type=kite.ORDER_TYPE_MARKET,
            product=kite.PRODUCT_MIS
        )
        return tradingsymbol, order['order_id'], qty
    except Exception as e:
        log(f"Order Error: {e}")
        send_telegram(f"❌ Order Failed: {str(e)}")
        return None, None, None

# --- EXIT ORDERS ---
def place_exit_orders(symbol, target, sl, qty):
    try:
        kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=qty,
            order_type=kite.ORDER_TYPE_LIMIT,
            price=target,
            product=kite.PRODUCT_MIS
        )
        kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=symbol,
            transaction_type=kite.TRANSACTION_TYPE_SELL,
            quantity=qty,
            order_type=kite.ORDER_TYPE_SL,
            trigger_price=sl,
            price=sl,
            product=kite.PRODUCT_MIS
        )
        return True
    except Exception as e:
        log(f"Exit Order Error: {e}")
        send_telegram(f"❌ Exit Order Failed: {str(e)}")
        return False

# --- MONITOR PRICE POST ENTRY ---
def monitor_trade(symbol, tp, sl):
    while True:
        ltp = get_ltp(symbol)
        if not ltp:
            continue
        if ltp >= tp:
            send_telegram(f"🎯 Target Hit @ ₹{ltp}")
            break
        elif ltp <= sl:
            send_telegram(f"🛑 Stop Loss Hit @ ₹{ltp}")
            break
        time.sleep(10)

# --- MAIN BOT LOOP ---
if __name__ == "__main__":
    last_signal_hash = None
    active = False
    running = True

    def telegram_commands():
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        while True:
            try:
                response = requests.get(url).json()
                msgs = response['result'][-5:]
                for m in msgs:
                    if 'message' not in m: continue
                    cmd = m['message']['text'].lower()
                    if cmd == "stop":
                        nonlocal running
                        running = False
                        send_telegram("⛔ Bot stopped.")
                    elif cmd == "start":
                        running = True
                        send_telegram("✅ Bot resumed.")
                    elif cmd == "check":
                        send_telegram("🔎 Bot is running.")
            except: pass
            time.sleep(10)

    threading.Thread(target=telegram_commands, daemon=True).start()

    while True:
        try:
            if not running:
                time.sleep(5)
                continue

            signal = generate_signal()
            s_hash = signal_hash(signal)

            if signal['confidence'] >= 9 and s_hash != last_signal_hash and not active:
                last_signal_hash = s_hash
                msg = (
                    f"\n📈 {signal['type']} {signal['strike']} — ₹{signal['price']} 🎯{signal['target']} | 🛑{signal['sl']}\n"
                    f"⭐ {signal['confidence']}/10 | {signal['reason']}\n"
                    f"📊 OI{signal['oi']} | Δ{signal['delta']} | Θ{signal['theta']} | IV {signal['iv']} | Exp: {signal['expiry']}"
                )
                send_telegram(msg)
                log(msg)

                symbol, order_id, qty = place_market_order(signal)
                if symbol:
                    success = place_exit_orders(symbol, signal['target'], signal['sl'], qty)
                    if success:
                        send_telegram(f"📤 Entry Done: {symbol}\nQty: {qty} 🎯TP: {signal['target']} | 🛑SL: {signal['sl']}")
                        threading.Thread(target=monitor_trade, args=(symbol, signal['target'], signal['sl'])).start()
                        active = True

            time.sleep(3)

        except Exception as e:
            log(f"Critical Error: {e}")
            send_telegram(f"⚠️ Bot Error: {str(e)}")
            time.sleep(10)
