import time
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_streaming.data_processor import StreamingDataProcessor
from datetime import datetime

def test_data_processing():
    """Test data processing flow"""
    print("🧪 Testing Data Processing Flow...")
    
    processor = StreamingDataProcessor()
    
    # Simulate a few ticks
    print("\n1. Simulating market data ticks...")
    for i in range(3):
        sample_tick = {
            'instrument_token': 256265,
            'timestamp': datetime.now(),
            'last_price': 18000 + i * 5,
            'last_quantity': 75,
            'volume': 1000000 + i * 10000,
            'depth': {
                'buy': [{'price': 18000 + i * 5 - 0.5, 'quantity': 75}],
                'sell': [{'price': 18000 + i * 5 + 0.5, 'quantity': 75}]
            }
        }
        
        processed = processor.process_tick(sample_tick)
        if processed:
            print(f"   ✅ Processed tick {i+1}: Price {processed['last_price']}")
        time.sleep(0.5)
    
    # Check current price
    current_price = processor.get_current_price(256265)
    print(f"\n2. Current Price: {current_price}")
    
    # Check market depth
    market_depth = processor.get_market_depth(256265)
    print(f"3. Market Depth: {market_depth}")
    
    # Check buffer status
    buffer_status = processor.get_buffer_status()
    print(f"4. Buffer Status: {buffer_status}")
    
    print("\n🎉 Data processing test completed!")

if __name__ == "__main__":
    test_data_processing()
