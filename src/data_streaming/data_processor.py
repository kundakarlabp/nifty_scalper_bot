import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StreamingDataProcessor:
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.data_buffers = {}  # token -> deque of ticks
        self.ohlc_data = {}     # token -> DataFrame of OHLC data
        self.last_tick_time = {}  # token -> last tick timestamp
        
    def process_tick(self, tick: Dict) -> Optional[Dict]:
        """Process a single tick and update buffers"""
        try:
            token = tick.get('instrument_token')
            if not token:
                logger.warning("Tick missing instrument token")
                return None
            
            # Initialize buffer for this token if not exists
            if token not in self.data_buffers:
                self.data_buffers[token] = deque(maxlen=self.buffer_size)
                self.ohlc_data[token] = pd.DataFrame()
                self.last_tick_time[token] = None
            
            # Add tick to buffer
            processed_tick = self._process_single_tick(tick)
            self.data_buffers[token].append(processed_tick)
            
            # Update last tick time
            self.last_tick_time[token] = processed_tick.get('timestamp')
            
            logger.debug(f"Processed tick for token {token}")
            return processed_tick
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
            return None
    
    def _process_single_tick(self, tick: Dict) -> Dict:
        """Process individual tick data"""
        try:
            processed = {
                'instrument_token': tick.get('instrument_token'),
                'timestamp': tick.get('timestamp', datetime.now()),
                'last_price': tick.get('last_price', 0),
                'last_quantity': tick.get('last_quantity', 0),
                'buy_quantity': tick.get('buy_quantity', 0),
                'sell_quantity': tick.get('sell_quantity', 0),
                'volume': tick.get('volume', 0),
                'oi': tick.get('oi', 0),
                'oi_day_high': tick.get('oi_day_high', 0),
                'oi_day_low': tick.get('oi_day_low', 0),
                'net_change': tick.get('net_change', 0)
            }
            
            # Add depth data if available
            if 'depth' in tick:
                depth = tick['depth']
                processed['bid_price'] = depth.get('buy', [{}])[0].get('price', 0) if depth.get('buy') else 0
                processed['bid_quantity'] = depth.get('buy', [{}])[0].get('quantity', 0) if depth.get('buy') else 0
                processed['ask_price'] = depth.get('sell', [{}])[0].get('price', 0) if depth.get('sell') else 0
                processed['ask_quantity'] = depth.get('sell', [{}])[0].get('quantity', 0) if depth.get('sell') else 0
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing single tick: {e}")
            return {
                'instrument_token': tick.get('instrument_token'),
                'timestamp': datetime.now(),
                'last_price': 0,
                'last_quantity': 0
            }
    
    def update_ohlc(self, token: int, timeframe: str = '1min') -> Optional[pd.DataFrame]:
        """Update OHLC data for a token"""
        try:
            if token not in self.data_buffers or not self.data_buffers[token]:
                logger.warning(f"No data buffer for token {token}")
                return None
            
            # Convert buffer to DataFrame
            ticks_df = pd.DataFrame(list(self.data_buffers[token]))
            if ticks_df.empty:
                return None
            
            # Set timestamp as index
            ticks_df['timestamp'] = pd.to_datetime(ticks_df['timestamp'])
            ticks_df.set_index('timestamp', inplace=True)
            
            # Resample to OHLC
            ohlc = ticks_df['last_price'].resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            # Add volume data
            if 'volume' in ticks_df.columns:
                volume = ticks_df['volume'].resample(timeframe).last().fillna(0)
                ohlc['volume'] = volume
            
            # Update stored OHLC data
            if token not in self.ohlc_data:
                self.ohlc_data[token] = ohlc
            else:
                # Merge with existing data
                self.ohlc_data[token] = pd.concat([self.ohlc_data[token], ohlc]).drop_duplicates()
            
            logger.debug(f"Updated OHLC data for token {token} with {len(ohlc)} bars")
            return ohlc
            
        except Exception as e:
            logger.error(f"Error updating OHLC data for token {token}: {e}")
            return None
    
    def get_latest_data(self, token: int, bars: int = 50) -> Optional[pd.DataFrame]:
        """Get latest OHLC data for a token"""
        try:
            if token not in self.ohlc_data:
                logger.warning(f"No OHLC data for token {token}")
                return None
            
            ohlc_data = self.ohlc_data[token]
            if ohlc_data.empty:
                return None
            
            # Return last N bars
            return ohlc_data.tail(bars).copy()
            
        except Exception as e:
            logger.error(f"Error getting latest data for token {token}: {e}")
            return None
    
    def get_current_price(self, token: int) -> Optional[float]:
        """Get current price for a token"""
        try:
            if token not in self.data_buffers:
                return None
            
            buffer = self.data_buffers[token]
            if not buffer:
                return None
            
            # Get latest tick
            latest_tick = buffer[-1]
            return latest_tick.get('last_price')
            
        except Exception as e:
            logger.error(f"Error getting current price for token {token}: {e}")
            return None
    
    def get_market_depth(self, token: int) -> Optional[Dict]:
        """Get market depth for a token"""
        try:
            if token not in self.data_buffers:
                return None
            
            buffer = self.data_buffers[token]
            if not buffer:
                return None
            
            # Get latest tick with depth data
            latest_tick = buffer[-1]
            depth_data = {}
            
            for key in ['bid_price', 'bid_quantity', 'ask_price', 'ask_quantity']:
                if key in latest_tick:
                    depth_data[key] = latest_tick[key]
            
            return depth_data if depth_data else None
            
        except Exception as e:
            logger.error(f"Error getting market depth for token {token}: {e}")
            return None
    
    def clear_buffer(self, token: int):
        """Clear data buffer for a token"""
        try:
            if token in self.data_buffers:
                self.data_buffers[token].clear()
            if token in self.ohlc_data:
                self.ohlc_data[token] = pd.DataFrame()
            if token in self.last_tick_time:
                self.last_tick_time[token] = None
            
            logger.info(f"Cleared buffer for token {token}")
            
        except Exception as e:
            logger.error(f"Error clearing buffer for token {token}: {e}")
    
    def get_buffer_status(self) -> Dict:
        """Get status of all buffers"""
        try:
            status = {}
            for token in self.data_buffers:
                status[token] = {
                    'buffer_size': len(self.data_buffers[token]),
                    'ohlc_size': len(self.ohlc_data.get(token, [])),
                    'last_tick': self.last_tick_time.get(token, 'N/A')
                }
            return status
            
        except Exception as e:
            logger.error(f"Error getting buffer status: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    processor = StreamingDataProcessor()
    
    # Simulate processing some ticks
    sample_tick = {
        'instrument_token': 256265,
        'timestamp': datetime.now(),
        'last_price': 18000.5,
        'last_quantity': 75,
        'buy_quantity': 1000,
        'sell_quantity': 800,
        'volume': 100000,
        'depth': {
            'buy': [{'price': 18000.0, 'quantity': 75}],
            'sell': [{'price': 18001.0, 'quantity': 75}]
        }
    }
    
    processed = processor.process_tick(sample_tick)
    print(f"Processed tick: {processed}")
    
    current_price = processor.get_current_price(256265)
    print(f"Current price: {current_price}")
