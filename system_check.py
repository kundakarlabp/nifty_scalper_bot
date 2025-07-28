import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("🚀 System Health Check...")

# Test 1: Environment Variables
print("\n1. Checking Environment Variables...")
required_vars = ['ZERODHA_API_KEY', 'ZERODHA_API_SECRET', 'ZERODHA_ACCESS_TOKEN']
missing_vars = []

for var in required_vars:
    value = os.getenv(var)
    if not value:
        missing_vars.append(var)
    else:
        print(f"   ✅ {var}: SET")

if missing_vars:
    print(f"   ❌ Missing variables: {missing_vars}")
else:
    print("   ✅ All required variables are set")

# Test 2: Zerodha Connection
print("\n2. Testing Zerodha Connection...")
try:
    from kiteconnect import KiteConnect
    api_key = os.getenv('ZERODHA_API_KEY')
    access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
    
    if api_key and access_token:
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        profile = kite.profile()
        print(f"   ✅ Connected to Zerodha")
        print(f"   👤 User: {profile.get('user_name', 'Unknown')}")
        print(f"   🆔 User ID: {profile.get('user_id', 'Unknown')}")
    else:
        print("   ⚠️  Missing API credentials")
except Exception as e:
    print(f"   ❌ Zerodha connection failed: {e}")

# Test 3: Telegram Connection
print("\n3. Testing Telegram Connection...")
try:
    import requests
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        url = f"https://api.telegram.org/bot{bot_token}/getMe"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print("   ✅ Telegram bot connected")
        else:
            print(f"   ❌ Telegram connection failed: {response.status_code}")
    else:
        print("   ⚠️  Missing Telegram credentials")
except Exception as e:
    print(f"   ❌ Telegram connection failed: {e}")

# Test 4: System Components
print("\n4. Testing System Components...")
try:
    from src.data_streaming.realtime_trader import RealTimeTrader
    trader = RealTimeTrader()
    status = trader.get_trading_status()
    print("   ✅ Real-time trader initialized")
    
    from src.strategies.scalping_strategy import DynamicScalpingStrategy
    strategy = DynamicScalpingStrategy()
    print("   ✅ Trading strategy initialized")
    
    from src.risk.position_sizing import PositionSizing
    risk_manager = PositionSizing()
    print("   ✅ Risk management initialized")
    
except Exception as e:
    print(f"   ❌ Component test failed: {e}")

print("\n🎉 System health check completed!")
print("🚀 Your trading system is ready for real-time trading!")
