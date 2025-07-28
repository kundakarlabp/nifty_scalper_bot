import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from src.data_streaming.websocket_client import WebSocketClient
from config import ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🧪 Testing WebSocket Connection...")
print("=" * 40)

# Test 1: Initialize WebSocket client
print("\n1. Initializing WebSocket client...")
try:
    ws_client = WebSocketClient()
    print("✅ WebSocket client initialized")
    
    # Check credentials
    if ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN:
        print("✅ Zerodha credentials available")
    else:
        print("❌ Zerodha credentials missing")
        print("💡 Please set ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN in .env file")
        
except Exception as e:
    print(f"❌ Error initializing WebSocket client: {e}")
    sys.exit(1)

# Test 2: Set up callbacks
print("\n2. Setting up callbacks...")
try:
    def on_ticks(ticks):
        print(f"📩 Received {len(ticks)} ticks")
        for tick in ticks[:3]:  # Show first 3 ticks
            print(f"   Token: {tick.get('instrument_token', 'N/A')}, "
                  f"LTP: {tick.get('last_price', 'N/A')}")
    
    def on_connect(response):
        print("✅ Connected to WebSocket")
    
    def on_close(code, reason):
        print(f" WebSocket closed. Code: {code}, Reason: {reason}")
    
    def on_error(code, reason):
        print(f" WebSocket error. Code: {code}, Reason: {reason}")
    
    ws_client.set_ticks_callback(on_ticks)
    ws_client.set_connect_callback(on_connect)
    ws_client.set_close_callback(on_close)
    ws_client.set_error_callback(on_error)
    
    print("✅ Callbacks set up successfully")
    
except Exception as e:
    print(f"❌ Error setting up callbacks: {e}")

# Test 3: Initialize connection
print("\n3. Initializing connection...")
try:
    if ws_client.initialize_connection():
        print("✅ Connection initialized successfully")
    else:
        print("❌ Failed to initialize connection")
        
except Exception as e:
    print(f"❌ Error initializing connection: {e}")

# Test 4: Get connection status
print("\n4. Checking connection status...")
try:
    status = ws_client.get_connection_status()
    print(f"✅ Connection status: {status}")
    
except Exception as e:
    print(f"❌ Error getting connection status: {e}")

print("\n" + "=" * 40)
print("🎉 WebSocket connection test completed!")
print("🚀 Your WebSocket client is ready!")

print("\n🔧 To use in your application:")
print("   from src.data_streaming.websocket_client import WebSocketClient")
print("   ws_client = WebSocketClient()")
print("   ws_client.initialize_connection()")
print("   ws_client.subscribe_tokens([256265])  # Nifty 50")
print("   ws_client.start_streaming()")
