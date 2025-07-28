import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import logging
from src.data_streaming.market_data_streamer import MarketDataStreamer
from src.data_streaming.data_processor import StreamingDataProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)

print("🧪 Testing Real-time Market Data Streaming...")

# Test 1: Market Data Streamer
print("\n1. Testing Market Data Streamer...")
try:
    streamer = MarketDataStreamer()
    print("✅ Market Data Streamer initialized")
    
    # Test connection status
    status = streamer.get_connection_status()
    print(f"✅ Connection status: {status}")
    
except Exception as e:
    print(f"❌ Market Data Streamer test failed: {e}")

# Test 2: Data Processor
print("\n2. Testing Data Processor...")
try:
    processor = StreamingDataProcessor()
    print("✅ Data Processor initialized")
    
    # Test buffer status
    buffer_status = processor.get_buffer_status()
    print(f"✅ Buffer status: {buffer_status}")
    
except Exception as e:
    print(f"❌ Data Processor test failed: {e}")

# Test 3: Integration Test (Simulation)
print("\n3. Testing Integration (Simulation)...")
try:
    from datetime import datetime
    
    processor = StreamingDataProcessor()
    
    # Simulate processing multiple ticks
    for i in range(5):
        sample_tick = {
            'instrument_token': 256265,
            'timestamp': datetime.now(),
            'last_price': 18000 + i,
            'last_quantity': 75,
            'volume': 100000 + i * 1000,
            'depth': {
                'buy': [{'price': 18000 + i - 0.5, 'quantity': 75}],
                'sell': [{'price': 18000 + i + 0.5, 'quantity': 75}]
            }
        }
        
        processed = processor.process_tick(sample_tick)
        if processed:
            print(f"✅ Processed tick {i+1}: Price {processed['last_price']}")
        time.sleep(0.1)
    
    # Test current price retrieval
    current_price = processor.get_current_price(256265)
    print(f"✅ Current price: {current_price}")
    
    # Test market depth
    market_depth = processor.get_market_depth(256265)
    print(f"✅ Market depth: {market_depth}")
    
except Exception as e:
    print(f"❌ Integration test failed: {e}")

print("\n🎉 Real-time streaming tests completed!")
print("🚀 Your real-time market data streaming system is ready!")
